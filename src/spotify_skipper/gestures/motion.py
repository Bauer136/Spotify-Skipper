"""Temporal gesture detection: L3.

Three detectors share this module and the same L4 interface — `update(sample, now)`
returning a `GestureEvent` or None, plus `.state` and `.last_reason` for the HUD:

  * `TransitionDetector` (4.4.2) — fires on a POSE CHANGE in place. The default.
  * `HoldDetector`       (4.4.1) — fires when one pose is held still long enough.
  * `MotionDetector`     (4.4)   — fires on a directed travel. NOT YET WRITTEN.

Everything spatial is measured in hand-widths, never pixels (see Phase 4.2).
"""
from __future__ import annotations

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
