"""Behavioural tests for the gesture FSM, driven by synthetic hand trajectories.

No camera, no MediaPipe, no recorded data: pure geometry in, events out. This is the
fastest possible feedback loop on the guards in motion.py, and it is what stops a
"harmless" threshold tweak from quietly re-enabling wave triggering.

Positions are in iso units; SCALE is the hand width, so 1.0 "hand-width" == SCALE.
"""
import numpy as np
import pytest

from src.spotify_skipper.gestures.motion import MotionConfig, MotionDetector, Sample

FPS = 30.0
DT = 1.0 / FPS
SCALE = 0.12
X0, Y0 = 0.5, 0.4


def feed(trajectory, cfg=None, pose="OPEN_PALM"):
    """Run a trajectory (list of (x, y) or None) and return the fire times."""
    det = MotionDetector(cfg or MotionConfig())
    fires = []
    for i, pos in enumerate(trajectory):
        t = i * DT
        sample = None if pos is None else Sample(
            t=t, center=np.array(pos, float), scale=SCALE, pose=pose, conf=1.0)
        if det.update(sample, t):
            fires.append(round(t, 2))
    return fires


def still(n, x=0.0, y=0.0):
    return [(X0 + x * SCALE, Y0 + y * SCALE)] * n


def swipe(hand_widths, frames, x0=0.0):
    return [(X0 + (x0 + hand_widths * (k + 1) / frames) * SCALE, Y0)
            for k in range(frames)]


def diagonal(hw, vw, frames):
    return [(X0 + hw * SCALE * (k + 1) / frames, Y0 + vw * SCALE * (k + 1) / frames)
            for k in range(frames)]


def waving(freq_hz, amplitude_hw, seconds=6.0):
    n = int(seconds * FPS)
    return still(12) + [(X0 + amplitude_hw * SCALE * np.sin(2 * np.pi * freq_hz * i * DT), Y0)
                        for i in range(n)]


# ---------------------------------------------------------------- must fire
@pytest.mark.parametrize("name,traj", [
    ("normal 0.40s swipe",  still(12) + swipe(2.2, 12) + still(20, 2.2)),
    ("fast 0.20s swipe",    still(12) + swipe(2.2, 6) + still(20, 2.2)),
    ("slow 0.60s swipe",    still(12) + swipe(2.0, 18) + still(20, 2.0)),
    ("long 3.5hw swipe",    still(12) + swipe(3.5, 15) + still(20, 3.5)),
    ("hand exits frame",    still(12) + swipe(2.2, 12) + [None] * 30),
    ("hand drops to lap",   still(12) + swipe(2.2, 12) + [(X0 + 2.2 * SCALE, 0.95)] * 30),
])
def test_deliberate_swipes_fire_exactly_once(name, traj):
    assert len(feed(traj)) == 1, name


# ------------------------------------------------------------ must not fire
@pytest.mark.parametrize("name,traj", [
    ("wave 0.8Hz wide",  waving(0.8, 2.5)),
    ("wave 1.0Hz",       waving(1.0, 2.0)),
    ("wave 1.5Hz",       waving(1.5, 1.5)),
    ("wave 2.0Hz",       waving(2.0, 1.5)),
    ("wave 2.5Hz fast",  waving(2.5, 1.2)),
    ("diagonal reach",   still(12) + diagonal(2.2, 2.0, 12) + still(20, 2.2, 2.0)),
    ("slow 1.5s drift",  still(12) + [(X0 + 2.2 * SCALE * (k + 1) / 45, Y0)
                                      for k in range(45)] + still(20, 2.2)),
    ("hand enters fast", [(0.1 + 3.0 * SCALE * (k + 1) / 10, Y0)
                          for k in range(10)] + still(20, 3.0)),
    ("jitter in place",  still(6) + [(X0 + 0.15 * SCALE * np.sin(9 * k), Y0)
                                     for k in range(120)]),
    ("swipe wrong way",  still(12) + [(X0 - 2.2 * SCALE * (k + 1) / 12, Y0)
                                      for k in range(12)] + still(20, -2.2)),
])
def test_decoys_never_fire(name, traj):
    assert feed(traj) == [], name


def test_wrong_pose_never_fires():
    assert feed(still(12) + swipe(2.2, 12) + still(20, 2.2), pose="FIST") == []


def test_cooldown_collapses_a_rapid_double_swipe():
    traj = still(12) + swipe(2.2, 12) + still(6, 2.2) + swipe(2.2, 12, 2.2) + still(20, 4.4)
    assert len(feed(traj)) == 1


def test_repeat_swipes_fire_again_after_the_cooldown():
    traj, x = [], 0.0
    for _ in range(3):
        traj += still(12, x) + swipe(2.2, 12, x) + still(120, x + 2.2)   # ~4.4s apart
        x += 2.2
    assert len(feed(traj)) == 3


def test_confirm_window_rejects_a_snap_back():
    """Returning the hand immediately reads as a wave -- by design."""
    traj = (still(12) + swipe(2.2, 12)
            + [(X0 + 2.2 * SCALE * (1 - (k + 1) / 6), Y0) for k in range(6)]
            + still(30))
    assert feed(traj, MotionConfig(confirm_s=0.40)) == []


def test_disabling_confirmation_reintroduces_wave_triggering():
    """Documents WHY confirm_s exists. If this ever passes with confirm_s=0,
    something else changed and the wave analysis in the guide needs revisiting."""
    assert feed(waving(1.0, 2.0), MotionConfig(confirm_s=0.0)) != []
