# scripts/debug_engine.py — Checkpoint M4: the whole pipeline on screen.
#
# L1 -> L2 -> L3 -> L4 running live, with every quantity the tuning work needs
# printed over the frame: raw and smoothed pose, confidence, hand scale, detector
# state, and — the one that saves the 1 am debugging session — `reason`, the guard
# that rejected the last near-miss.
import argparse
import sys
import time
from collections import deque
from dataclasses import fields
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root,
# so `src.` would not resolve. Prepend the repo root before importing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from src.spotify_skipper.camera import Camera
from src.spotify_skipper.config import camera_kwargs, load_config
from src.spotify_skipper.hands import HandTracker
from src.spotify_skipper.gestures.engine import GestureEngine
from src.spotify_skipper.gestures.motion import (HoldConfig, HoldDetector,
                                                 MotionConfig, MotionDetector,
                                                 TransitionConfig, TransitionDetector)
from src.spotify_skipper.gestures.poses import classify_pose

DETECTORS = {"transition": (TransitionConfig, TransitionDetector),
             "motion": (MotionConfig, MotionDetector),
             "hold": (HoldConfig, HoldDetector)}

FLASH_S = 1.0        # how long the red SKIP overlay stays up after an event


def _subset(section: dict, cls) -> dict:
    """Keep only the keys `cls` actually declares.

    [gesture] in config.toml holds the union of all three detectors' tunables, so
    every build would otherwise die on the other two detectors' keys.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in section.items() if k in known}


# These two builders move into config.py in Phase 8.2, where the real app needs
# them too. They live here while the HUD is the only caller.
def build_pose_fn(cfg, backend: str | None = None):
    p = cfg.section("pose")
    backend = backend or p.get("backend", "rules")
    if backend == "rules":
        return classify_pose, "rules"
    # Imported lazily: the rules path must not require scikit-learn/joblib.
    from src.spotify_skipper.gestures.classifier import MLPoseClassifier
    path = Path(p.get("ml_model_path", "models/pose_clf.joblib"))
    if not path.is_absolute():
        path = ROOT / path
    return MLPoseClassifier(path, float(p.get("ml_min_conf", 0.75))), "ml"


def build_detector(cfg, kind: str | None = None):
    g = cfg.section("gesture")
    kind = kind or g.get("type", "transition")
    if kind not in DETECTORS:
        raise SystemExit(f"unknown gesture type {kind!r}; pick one of {sorted(DETECTORS)}")
    config_cls, detector_cls = DETECTORS[kind]
    return detector_cls(config_cls(**_subset(g, config_cls))), kind


class FpsMeter:
    """Rolling FPS — the frame-count thresholds in [gesture] assume ~20-30."""
    def __init__(self, window=30):
        self._t = deque(maxlen=window)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", metavar="SPEC", default=None,
                    help="camera index, /dev path, or name fragment; overrides "
                         "config.toml. `python scripts/cameras.py` lists them.")
    ap.add_argument("--pose", choices=["rules", "ml"], default=None,
                    help="L2 backend; overrides [pose].backend")
    ap.add_argument("--gesture", choices=sorted(DETECTORS), default=None,
                    help="L3 detector; overrides [gesture].type")
    args = ap.parse_args()

    cfg = load_config()
    cam_kw = camera_kwargs(cfg, args.camera)
    pose_fn, pose_backend = build_pose_fn(cfg, args.pose)
    detector, gesture_kind = build_detector(cfg, args.gesture)

    p = cfg.section("pose")
    engine = GestureEngine(
        pose_fn, detector,
        smoothing_frames=int(p.get("smoothing_frames", 3)),
        min_hand_scale=float(p.get("min_hand_scale", 0.05)),
        max_hand_scale=float(p.get("max_hand_scale", 0.60)),
    )

    h_cfg = cfg.section("hands")
    # absolute, so the HUD works from any working directory
    tracker = HandTracker(
        model_path=ROOT / h_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=int(h_cfg.get("num_hands", 1)),
        min_detection_confidence=float(h_cfg.get("min_detection_confidence", 0.6)),
        min_presence_confidence=float(h_cfg.get("min_presence_confidence", 0.6)),
        min_tracking_confidence=float(h_cfg.get("min_tracking_confidence", 0.6)),
    )

    fps = FpsMeter()
    fired_at = -FLASH_S          # so nothing flashes on the first frame
    count = 0
    print(f"pose={pose_backend} gesture={gesture_kind} — q or ESC to quit")
    try:
        with Camera(**cam_kw) as cam:
            while True:
                fr = cam.read()
                if fr is None:
                    break
                h, w = fr.image.shape[:2]
                obs = tracker.process(fr.image, fr.timestamp_ms)
                now = time.monotonic()
                events = engine.process(obs, w, h, now=now)

                y = 25
                for k, v in engine.debug.items():
                    cv2.putText(fr.image, f"{k}: {v}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 22
                rate = fps.tick()
                # orange under 20 FPS: the frame-count tunables stop meaning what
                # they say long before the gesture visibly stops working
                colour = (0, 255, 0) if rate >= 20 else (0, 165, 255)
                cv2.putText(fr.image, f"{rate:5.1f} FPS  {pose_backend}/{gesture_kind}"
                                      f"  fired={count}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

                if events:
                    fired_at = now
                    count += 1
                    for e in events:
                        print(f"EVENT #{count} {e}")
                if now - fired_at < FLASH_S:
                    cv2.rectangle(fr.image, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
                    cv2.putText(fr.image, "SKIP", (w // 2 - 90, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                cv2.imshow("engine", fr.image)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
