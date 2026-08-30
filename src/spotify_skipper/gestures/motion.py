"""Temporal gesture detection: L3.

Three detectors share this module and the same L4 interface — `update(sample, now)`
returning a `GestureEvent` or None, plus `.state` and `.last_reason` for the HUD:

  * `TransitionDetector` (4.4.2) — fires on a POSE CHANGE in place. The default.
  * `HoldDetector`       (4.4.1) — fires when one pose is held still long enough.
  * `MotionDetector`     (4.4)   — fires on a directed travel.

Everything spatial is measured in hand-widths, never pixels (see Phase 4.2).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


class State(Enum):
    IDLE = "IDLE"          # nothing interesting
    ARMED = "ARMED"        # trigger pose held and steady; watching for the trigger
    PENDING = "PENDING"    # candidate seen; confirming before firing
    FIRED = "FIRED"        # emitted this frame
    COOLDOWN = "COOLDOWN"  # ignoring input
    RESET = "RESET"        # cooldown over; waiting for the pose to be released


@dataclass
class Sample:
    t: float                # seconds, monotonic
    center: np.ndarray      # (2,) iso coords
    scale: float            # hand width, iso units
    pose: str
    conf: float


@dataclass
class GestureEvent:
    name: str
    t: float
    travel_hand_widths: float   # always 0.0 for hold/transition gestures
    duration_s: float
    peak_speed: float           # likewise 0.0 — kept so L4 sees one event shape


def _drift(sample: Sample, anchor: Sample) -> float:
    """Distance from the anchor, in anchor hand-widths."""
    return float(np.linalg.norm(sample.center - anchor.center)
                 / max(1e-6, anchor.scale))


# ============================================================== 4.4 motion
@dataclass
class MotionConfig:
    trigger_pose: str = "OPEN_PALM"
    direction: str = "right"        # right | left | up | down
    arm_frames: int = 6             # consecutive pose frames needed to arm (~0.25 s)
    arm_stillness: float = 0.55     # max hand-widths/sec of drift while arming
    travel_hand_widths: float = 1.6 # required displacement, in hand widths
    max_duration_s: float = 0.70    # must complete within this
    min_duration_s: float = 0.10    # ...but not faster than this (rejects jitter)
    lateral_ratio: float = 0.60     # |cross-axis| must stay under this * |along-axis|
    monotonic_ratio: float = 0.80   # >=80% of frames must move the right way
    pose_grace_frames: int = 3      # pose may flicker off for this many frames
    confirm_s: float = 0.40         # hold-still window that separates swipe from wave
    reversal_ratio: float = 0.50    # cancel if the hand returns this far back
    cooldown_s: float = 3.0
    release_frames: int = 5         # frames without the pose before re-arming


_AXIS = {"right": (0, +1.0), "left": (0, -1.0), "down": (1, +1.0), "up": (1, -1.0)}


class MotionDetector:
    """Feed one Sample per frame; get a GestureEvent (or None) back."""

    def __init__(self, cfg: MotionConfig, name: str = "skip"):
        self.cfg, self.name = cfg, name
        self.state = State.IDLE
        self.buf: deque[Sample] = deque(maxlen=90)     # ~3 s of history
        self._pose_run = 0
        self._miss_run = 0
        self._release_run = 0
        self._armed_at: Sample | None = None
        self._pending: tuple | None = None
        self._cooldown_until = 0.0
        self.last_reason = ""       # why we did NOT fire — invaluable when tuning

    # -- helpers -----------------------------------------------------------
    def _speed(self, n=4) -> float:
        """Recent PATH speed in hand-widths per second, over up to n+1 samples.

        Path length, not net displacement: across a direction reversal the net
        displacement cancels to ~0 and a waving hand would look 'at rest' exactly at
        its turning points -- which is precisely where a wave would otherwise anchor
        a fresh gesture window. Measuring distance travelled removes that hole.
        """
        seg = list(self.buf)[-(n + 1):]
        if len(seg) < 2:
            return 0.0
        dist = sum(float(np.linalg.norm(b.center - a.center))
                   for a, b in zip(seg, seg[1:]))
        dt = max(1e-3, seg[-1].t - seg[0].t)
        return dist / max(1e-6, seg[-1].scale) / dt

    def _reset(self, state=State.IDLE):
        self.state = state
        self._armed_at = None
        self._pending = None
        self._pose_run = 0
        self._miss_run = 0

    def _reanchor_or_expire(self, sample: "Sample", dt: float, speed: float):
        """Slide the anchor forward ONLY to a rest point; otherwise abandon.

        Re-anchoring to a moving frame would let the second half of a wave start a
        fresh window and fire. Dropping to IDLE forces a full re-arm (pose held +
        stillness), which an oscillating hand can never satisfy.
        """
        if speed <= self.cfg.arm_stillness:
            self._armed_at = sample                 # new rest point
        elif dt > self.cfg.max_duration_s:
            self.last_reason = "window expired without a rest point"
            self._reset(State.IDLE)

    # -- main --------------------------------------------------------------
    def update(self, sample: Sample | None, now: float) -> GestureEvent | None:
        cfg = self.cfg

        # ---- cooldown / release gating ----------------------------------
        if self.state in (State.COOLDOWN, State.FIRED):
            if now < self._cooldown_until:
                self.state = State.COOLDOWN
                return None
            self.state = State.RESET
            self._release_run = 0

        # ---- PENDING: confirm the travel was not half of an oscillation ---
        if self.state is State.PENDING:
            event, deadline, anchor_center, along0, scale0 = self._pending
            axis, sign = _AXIS[cfg.direction]
            if sample is not None:
                self.buf.append(sample)
                back = float(sample.center[axis] - anchor_center[axis]) * sign / scale0
                if back < along0 * (1.0 - cfg.reversal_ratio):
                    # the hand came back -> this was a wave, not a swipe
                    self.last_reason = f"reversed to {back:.2f} of {along0:.2f} (wave)"
                    self._reset(State.IDLE)
                    return None
            # a hand that vanished mid-window left the frame in the swipe direction:
            # that is consistent with a swipe, so we let the timer run.
            if now >= deadline:
                self.state = State.FIRED
                self._cooldown_until = now + cfg.cooldown_s
                self._pending = None
                self.last_reason = "fired"
                return event
            return None

        if self.state is State.RESET:
            # One continuous motion must never fire twice. Re-arm only after the user
            # either drops the trigger pose or brings the hand back to rest.
            if sample is not None:
                self.buf.append(sample)
            settled = (sample is None or sample.pose != cfg.trigger_pose
                       or self._speed() <= cfg.arm_stillness)
            if settled:
                self._release_run += 1
                if self._release_run >= cfg.release_frames:
                    self._reset(State.IDLE)
            else:
                self._release_run = 0
            return None

        # ---- no hand this frame -----------------------------------------
        if sample is None:
            self._miss_run += 1
            if self._miss_run > cfg.pose_grace_frames:
                self._reset(State.IDLE)
            return None

        self.buf.append(sample)

        # ---- pose bookkeeping -------------------------------------------
        if sample.pose == cfg.trigger_pose:
            self._pose_run += 1
            self._miss_run = 0
        else:
            self._miss_run += 1
            if self._miss_run > cfg.pose_grace_frames:
                self._reset(State.IDLE)
                self.last_reason = "pose lost"
                return None

        # ---- IDLE -> ARMED ----------------------------------------------
        if self.state is State.IDLE:
            if self._pose_run >= cfg.arm_frames and self._speed() <= cfg.arm_stillness:
                self.state = State.ARMED
                self._armed_at = sample
            return None

        # ---- ARMED: look for the travel ---------------------------------
        # INVARIANT: the travel window always starts from a REST point. This is what
        # makes waving impossible to confuse with a swipe -- a wave never rests.
        anchor = self._armed_at
        dt = sample.t - anchor.t
        speed = self._speed()

        if dt < cfg.min_duration_s:
            return None

        axis, sign = _AXIS[cfg.direction]
        cross = 1 - axis
        delta = sample.center - anchor.center
        scale = max(1e-6, anchor.scale)
        along = float(delta[axis] * sign) / scale
        lateral = abs(float(delta[cross])) / scale

        if along < cfg.travel_hand_widths or lateral > cfg.lateral_ratio * max(along, 1e-6):
            if along >= cfg.travel_hand_widths:
                self.last_reason = f"too diagonal ({lateral:.2f} vs {along:.2f})"
            else:
                self.last_reason = f"travel {along:.2f} < {cfg.travel_hand_widths}"
            self._reanchor_or_expire(sample, dt, speed)
            return None

        # monotonicity: of the frames that actually MOVED, nearly all must move the
        # right way. Near-zero steps are excluded, otherwise the still frames at the
        # start of the window drag the ratio below threshold and nothing ever fires.
        seg = [s for s in self.buf if s.t >= anchor.t]
        peak = 0.0
        if len(seg) >= 3:
            steps = np.diff([s.center[axis] for s in seg]) * sign
            moving = np.abs(steps) > 0.02 * scale        # noise floor
            if moving.sum() >= 2:
                good = float(np.mean(steps[moving] > 0))
                if good < cfg.monotonic_ratio:
                    self.last_reason = f"not monotonic ({good:.2f})"
                    self._reanchor_or_expire(sample, dt, speed)
                    return None
            dts = np.diff([s.t for s in seg])
            peak = float(np.max(np.abs(steps)) / scale / max(1e-3, float(np.mean(dts))))

        # ---- candidate accepted; confirm before firing --------------------
        event = GestureEvent(self.name, now, along, dt, peak)
        if cfg.confirm_s <= 0:
            self.state = State.FIRED
            self._cooldown_until = now + cfg.cooldown_s
            self.last_reason = "fired"
            return event
        self.state = State.PENDING
        self._pending = (event, now + cfg.confirm_s, anchor.center.copy(), along, scale)
        self.last_reason = "confirming"
        return None


# ============================================================ 4.4.1 hold
@dataclass
class HoldConfig:
    trigger_pose: str = "PEACE"
    hold_s: float = 1.2         # must be this long — a real deliberate commitment
    max_drift: float = 0.9      # hand-widths of allowed wander during the hold
    min_conf: float = 0.6
    cooldown_s: float = 3.0
    release_frames: int = 5


class HoldDetector:
    """Fires when a pose is held steadily for hold_s. Simpler and more reliable
    than a swipe in cluttered scenes, but slower to trigger."""

    def __init__(self, cfg: HoldConfig, name="skip"):
        self.cfg, self.name = cfg, name
        self._start: Sample | None = None
        self._cooldown_until = 0.0
        self._release_run = 0
        self.state = State.IDLE
        self.last_reason = ""

    def update(self, sample, now):
        cfg = self.cfg
        if now < self._cooldown_until:
            self.state = State.COOLDOWN
            return None
        if self.state is State.COOLDOWN:      # cooldown just ended
            self.state = State.RESET
            self._release_run = 0
        if self.state is State.RESET:
            if sample is None or sample.pose != cfg.trigger_pose:
                self._release_run += 1
                if self._release_run >= cfg.release_frames:
                    self.state = State.IDLE
            return None

        if sample is None or sample.pose != cfg.trigger_pose or sample.conf < cfg.min_conf:
            self._start = None
            self.state = State.IDLE
            return None

        if self._start is None:
            self._start = sample
            self.state = State.ARMED
            self.last_reason = "armed"
            return None

        if _drift(sample, self._start) > cfg.max_drift:
            self._start = sample                      # moved: restart the clock
            self.last_reason = "drifted; hold restarted"
            return None
        if sample.t - self._start.t >= cfg.hold_s:
            self.state = State.FIRED
            self._cooldown_until = now + cfg.cooldown_s
            self._start = None
            self.last_reason = "fired"
            return GestureEvent(self.name, now, 0.0, cfg.hold_s, 0.0)
        return None


# ====================================================== 4.4.2 transition
@dataclass
class TransitionConfig:
    stage_a_pose: str = "ZERO"        # the "O" — what arms the gesture
    stage_b_pose: str = "OK_THREE"    # three fingers up — what fires it
    arm_frames: int = 5               # consecutive stage-A frames needed to arm
    max_transition_s: float = 0.60    # B must arrive within this of leaving A
    min_transition_s: float = 0.05    # ...but not instantly (rejects tracking pops)
    confirm_frames: int = 2           # consecutive stage-B frames needed to fire
    max_drift: float = 0.80           # hand-widths the hand may wander, A -> B
    min_conf: float = 0.50
    pose_grace_frames: int = 2        # frames of NONE tolerated mid-transition
    cooldown_s: float = 3.0
    release_frames: int = 5           # frames off stage B before re-arming


class TransitionDetector:
    """Fires on a POSE CHANGE rather than a movement.

    IDLE (collecting stage A) -> ARMED (stage A held) -> PENDING (left stage A,
    waiting for stage B) -> FIRED -> COOLDOWN -> RESET.

    Note what is absent versus MotionDetector: no direction, no travel, no
    monotonicity, no confirmation window. Those guards exist to separate a swipe from
    the first stroke of a wave; a closed thumb-index loop has no such natural twin.
    Stillness is guarded by cumulative drift rather than path speed, because a
    stationary gesture has no turning points to be fooled by.
    """

    def __init__(self, cfg: TransitionConfig, name: str = "skip"):
        self.cfg, self.name = cfg, name
        self.state = State.IDLE
        self._pose_run = 0          # consecutive stage-A frames while arming
        self._b_run = 0             # consecutive stage-B frames while confirming
        self._miss_run = 0
        self._release_run = 0
        self._anchor: Sample | None = None   # drift reference while arming
        self._last_a: Sample | None = None   # last stage-A frame = window start
        self._cooldown_until = 0.0
        self.last_reason = ""

    def _reset(self, state=State.IDLE):
        self.state = state
        self._pose_run = self._b_run = self._miss_run = 0
        self._anchor = self._last_a = None

    def update(self, sample: Sample | None, now: float) -> GestureEvent | None:
        cfg = self.cfg

        # ---- cooldown / release gating ----------------------------------
        if self.state in (State.COOLDOWN, State.FIRED):
            if now < self._cooldown_until:
                self.state = State.COOLDOWN
                return None
            self.state = State.RESET
            self._release_run = 0

        if self.state is State.RESET:
            # After firing you are still holding stage B. Require it to be dropped
            # before anything can arm again — this is what makes "reset, then wait
            # for the O again" literal, and stops one long hold re-firing.
            if sample is None or sample.pose != cfg.stage_b_pose:
                self._release_run += 1
                if self._release_run >= cfg.release_frames:
                    self._reset(State.IDLE)
            else:
                self._release_run = 0
            return None

        # ---- no hand this frame -----------------------------------------
        if sample is None:
            self._miss_run += 1
            if self._miss_run > cfg.pose_grace_frames:
                self.last_reason = "hand lost"
                self._reset(State.IDLE)
            return None

        in_a = sample.pose == cfg.stage_a_pose and sample.conf >= cfg.min_conf

        # ---- IDLE: accumulate a steady stage A --------------------------
        if self.state is State.IDLE:
            if in_a:
                if self._anchor is None:
                    self._anchor = sample
                if _drift(sample, self._anchor) > cfg.max_drift:
                    self._anchor, self._pose_run = sample, 0   # moved: restart
                self._pose_run += 1
                self._last_a = sample
                self._miss_run = 0
                if self._pose_run >= cfg.arm_frames:
                    self.state = State.ARMED
                    self.last_reason = "armed"
            else:
                self._pose_run = 0
                self._anchor = None
            return None

        # ---- still in stage A: slide the window start forward ------------
        if in_a:
            self._last_a = sample
            self._miss_run = 0
            if self.state is State.PENDING:      # fell back into the O; re-arm
                self.state = State.ARMED
                self._b_run = 0
            return None

        if self.state is State.ARMED:            # just left stage A
            self.state = State.PENDING
            self._b_run = 0

        # ---- PENDING: the transition window ------------------------------
        anchor = self._last_a
        dt = sample.t - anchor.t
        if dt > cfg.max_transition_s:
            self.last_reason = f"transition window expired ({dt:.2f}s)"
            self._reset(State.IDLE)
            return None

        drift = _drift(sample, anchor)
        if drift > cfg.max_drift:
            # the hand moved away instead of changing shape in place
            self.last_reason = f"drifted {drift:.2f}hw during the transition"
            self._reset(State.IDLE)
            return None

        if sample.pose == cfg.stage_b_pose and sample.conf >= cfg.min_conf:
            self._b_run += 1
            self._miss_run = 0
            if dt < cfg.min_transition_s:
                self.last_reason = "transition too fast (tracking pop?)"
                return None
            if self._b_run >= cfg.confirm_frames:
                self.state = State.FIRED
                self._cooldown_until = now + cfg.cooldown_s
                self.last_reason = "fired"
                event = GestureEvent(self.name, now, 0.0, dt, 0.0)
                self._anchor = self._last_a = None
                return event
            return None

        # anything else mid-window: tolerate a short flicker, then abandon
        self._b_run = 0
        self._miss_run += 1
        if self._miss_run > cfg.pose_grace_frames:
            self.last_reason = f"pose {sample.pose!r} interrupted the transition"
            self._reset(State.IDLE)
        return None
