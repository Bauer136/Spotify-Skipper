"""Behavioural tests for the transition FSM, driven by synthetic pose sequences.

The swipe suite below feeds trajectories; this one feeds POSE STREAMS at a constant
position, because the gesture is a shape change in place. Where a position matters
(the drift guard) it is passed explicitly.
"""
import numpy as np
import pytest

from src.spotify_skipper.gestures.motion import (Sample, TransitionConfig,
                                                 TransitionDetector)

FPS = 30.0
DT = 1.0 / FPS
SCALE = 0.12
X0, Y0 = 0.5, 0.4

A, B = "ZERO", "OK_THREE"


def feed(poses, cfg=None, positions=None, conf=1.0):
    """poses: list of pose names, or None for 'no hand this frame'."""
    det = TransitionDetector(cfg or TransitionConfig())
    fires = []
    for i, pose in enumerate(poses):
        t = i * DT
        if pose is None:
            sample = None
        else:
            pos = positions[i] if positions else (X0, Y0)
            sample = Sample(t=t, center=np.array(pos, float), scale=SCALE,
                            pose=pose, conf=conf)
        if det.update(sample, t):
            fires.append(round(t, 2))
    return fires


def hold(pose, n):
    return [pose] * n


# ---------------------------------------------------------------- must fire
@pytest.mark.parametrize("name,seq", [
    ("clean transition",      hold(A, 8) + hold(B, 10)),
    ("long hold of the O",    hold(A, 60) + hold(B, 10)),
    ("one dropped frame",     hold(A, 8) + [None] + hold(B, 10)),
    ("two dropped frames",    hold(A, 8) + [None, None] + hold(B, 10)),
])
def test_deliberate_transitions_fire_exactly_once(name, seq):
    assert len(feed(seq)) == 1, name


# ------------------------------------------------------------ must not fire
@pytest.mark.parametrize("name,seq", [
    ("the O held forever",      hold(A, 120)),
    ("OK sign, never an O",     hold(B, 120)),
    ("O too brief to arm",      hold(A, 2) + hold(B, 20)),
    ("gap longer than window",  hold(A, 8) + [None] * 30 + hold(B, 10)),
    ("wrong stage B",           hold(A, 8) + hold("OPEN_PALM", 20)),
    ("fist opening to a palm",  hold("FIST", 8) + hold("OPEN_PALM", 20)),
    ("interrupted by a palm",   hold(A, 8) + hold("OPEN_PALM", 5) + hold(B, 10)),
    ("fidgeting in and out",    (hold(A, 3) + [None] * 3) * 12),
])
def test_decoys_never_fire(name, seq):
    assert feed(seq) == [], name


def test_transition_while_the_hand_travels_is_rejected():
    """Fingers uncurling while you reach for something is not the gesture."""
    seq = hold(A, 8) + hold(B, 10)
    pos = [(X0, Y0)] * 8 + [(X0 + 0.5 * SCALE * (k + 1), Y0) for k in range(10)]
    assert feed(seq, positions=pos) == []


def test_low_confidence_pose_never_arms():
    assert feed(hold(A, 8) + hold(B, 10), conf=0.2) == []


def test_holding_stage_b_never_refires():
    """One sustained OK sign is one skip, not a stream of them."""
    assert len(feed(hold(A, 8) + hold(B, 400))) == 1


def test_must_return_to_the_o_to_fire_again():
    """After the cooldown, stage B alone is not enough: the O must be re-formed."""
    seq = hold(A, 8) + hold(B, 10) + [None] * 20 + hold(B, 200)
    assert len(feed(seq)) == 1


def test_repeats_after_cooldown_and_a_fresh_o():
    seq = (hold(A, 8) + hold(B, 10) + [None] * 120
           + hold(A, 8) + hold(B, 10) + [None] * 120)
    assert len(feed(seq)) == 2


def test_cooldown_collapses_a_rapid_repeat():
    seq = hold(A, 8) + hold(B, 10) + [None] * 8 + hold(A, 8) + hold(B, 10)
    assert len(feed(seq)) == 1


def test_reason_explains_a_rejection():
    det = TransitionDetector(TransitionConfig())
    for i, pose in enumerate(hold(A, 8) + [None] * 30):
        sample = None if pose is None else Sample(
            t=i * DT, center=np.array([X0, Y0]), scale=SCALE, pose=pose, conf=1.0)
        det.update(sample, i * DT)
    assert "lost" in det.last_reason or "expired" in det.last_reason
