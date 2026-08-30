"""Behavioural tests for the hold FSM (4.4.1).

The guide ships no suite for HoldDetector; this is it. Same shape as the transition
suite: synthetic pose streams in, events out, no camera and no MediaPipe.
"""
import numpy as np
import pytest

from src.spotify_skipper.gestures.motion import HoldConfig, HoldDetector, Sample

FPS = 30.0
DT = 1.0 / FPS
SCALE = 0.12
X0, Y0 = 0.5, 0.4
P = "PEACE"                       # HoldConfig.trigger_pose default
HOLD_FRAMES = int(HoldConfig().hold_s * FPS)     # 36 at 30 FPS


def feed(poses, cfg=None, positions=None, conf=1.0):
    det = HoldDetector(cfg or HoldConfig())
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


def test_a_steady_hold_fires_exactly_once():
    assert len(feed(hold(P, HOLD_FRAMES + 10))) == 1


def test_a_hold_shorter_than_hold_s_never_fires():
    assert feed(hold(P, HOLD_FRAMES - 5)) == []


@pytest.mark.parametrize("name,seq", [
    ("wrong pose",   hold("OPEN_PALM", 200)),
    ("no hand",      [None] * 200),
    ("pose flicker", (hold(P, 10) + [None]) * 15),
])
def test_decoys_never_fire(name, seq):
    assert feed(seq) == [], name


def test_low_confidence_never_fires():
    assert feed(hold(P, 200), conf=0.2) == []


def test_a_single_dropped_frame_restarts_the_clock():
    """HoldDetector has no pose_grace_frames — unlike TransitionDetector, one lost
    frame costs you the whole hold. Documented here so it is a choice, not a surprise."""
    seq = hold(P, 20) + [None] + hold(P, 20)
    assert feed(seq) == []


def test_drifting_past_max_drift_restarts_the_clock():
    seq = hold(P, 200)
    pos = [(X0 + 0.20 * SCALE * k, Y0) for k in range(200)]
    assert feed(seq, positions=pos) == []


def test_small_wander_inside_max_drift_still_fires():
    n = HOLD_FRAMES + 10
    seq = hold(P, n)
    pos = [(X0 + 0.30 * SCALE * np.sin(0.3 * k), Y0) for k in range(n)]
    assert len(feed(seq, positions=pos)) == 1


def test_holding_continuously_never_refires():
    """One long hold is one skip, not a stream — the release gate must block re-arming."""
    assert len(feed(hold(P, 400))) == 1


def test_repeats_after_cooldown_and_a_fresh_hold():
    seq = hold(P, HOLD_FRAMES + 10) + [None] * 150 + hold(P, HOLD_FRAMES + 10)
    assert len(feed(seq)) == 2
