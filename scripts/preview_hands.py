# scripts/preview_hands.py
import sys
import time
from collections import deque
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root,
# so `src.` would not resolve. Prepend the repo root before importing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2, numpy as np
from src.spotify_skipper.camera import Camera
from src.spotify_skipper.hands import HandTracker

EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
         (13,17),(17,18),(18,19),(19,20), (0,17)]

def draw(img, lm):
    h, w = img.shape[:2]
    pts = (lm[:, :2] * [w, h]).astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 3, (0, 0, 255), -1)
        cv2.putText(img, str(i), tuple(p + 4), cv2.FONT_HERSHEY_PLAIN, 0.7,
                    (255, 255, 255), 1)


class FpsMeter:
    """Rolling FPS over the last `window` frames — Phase 3.4 wants 20-30."""
    def __init__(self, window=30):
        self._t = deque(maxlen=window)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def main():
    # absolute, so the viewer works from any working directory
    tracker = HandTracker(model_path=ROOT / "models" / "hand_landmarker.task")
    fps = FpsMeter()
    with Camera() as cam:
        while True:
            f = cam.read()
            if f is None: break
            for obs in tracker.process(f.image, f.timestamp_ms):
                draw(f.image, obs.landmarks)
                cv2.putText(f.image, f"{obs.handedness} {obs.score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            rate = fps.tick()
            # green once we clear the 20 FPS floor Phase 4's thresholds assume
            colour = (0, 255, 0) if rate >= 20 else (0, 165, 255)
            cv2.putText(f.image, f"{rate:5.1f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.imshow("hands", f.image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
