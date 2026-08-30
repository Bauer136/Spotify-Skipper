# scripts/calibrate_pose.py  — hold the pose, read the numbers, set the constants
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root,
# so `src.` would not resolve. Prepend the repo root before importing it.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2, numpy as np
from src.spotify_skipper.camera import Camera
from src.spotify_skipper.hands import HandTracker
from src.spotify_skipper import features as F
from src.spotify_skipper.gestures.poses import classify_pose


def main():
    # absolute, so calibration works from any working directory
    tracker = HandTracker(model_path=ROOT / "models" / "hand_landmarker.task")
    with Camera() as cam:
        while True:
            fr = cam.read()
            if fr is None: break
            h, w = fr.image.shape[:2]
            for obs in tracker.process(fr.image, fr.timestamp_ms):
                iso = F.to_iso(obs.landmarks, w, h)
                canon = F.canonicalize(iso, obs.handedness)
                if canon is None: continue
                ext = F.extension_ratios(canon)
                txt = " ".join(f"{n[:3]}={v:.2f}" for n, v in zip(F.FINGER_ORDER, ext))
                cv2.putText(fr.image, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)
                cv2.putText(fr.image, f"open={F.palm_openness(canon):.2f} "
                                      f"spread={np.mean(F.spreads(canon)):.2f} "
                                      f"scale={F.hand_scale(iso):.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                # four_ext is the quantity the OPEN_PALM/FIST rules actually gate on
                e = dict(zip(F.FINGER_ORDER, ext))
                four = [e["index"], e["middle"], e["ring"], e["pinky"]]
                cv2.putText(fr.image, f"four_ext min={min(four):.2f} max={max(four):.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                # ZERO / OK_THREE gate on these two specifically, not on mean spread
                sp = F.spreads(canon)
                three = [e["middle"], e["ring"], e["pinky"]]
                cv2.putText(fr.image, f"circle_gap={sp[0]:.2f} three_spread={np.mean(sp[2:4]):.2f} "
                                      f"three_ext min={min(three):.2f} max={max(three):.2f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                # live label: what L2 would actually hand the motion FSM this frame
                r = classify_pose(canon)
                colour = (0, 255, 0) if r.name != "NONE" else (0, 0, 255)
                cv2.putText(fr.image, f"{r.name} {r.confidence:.2f}", (10, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.imshow("calibrate", fr.image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
