"""Record labelled landmark clips to data/clips/*.npz (no images stored)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import camera_kwargs, load_config
from .hands import HandTracker


def main():
    ap = argparse.ArgumentParser(description="Record hand-landmark clips.")
    ap.add_argument("--label", required=True,
                    help="clip label, e.g. skip_transition / idle / decoy_ok_talking")
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="auto-stop after N seconds (0 = manual stop with SPACE)")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--camera", metavar="SPEC", default=None,
                    help="camera index, /dev path, or name fragment; overrides "
                         "config.toml. `python scripts/cameras.py` lists them.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = HandTracker(model_path=args.model)

    recording, frames, start = False, [], 0.0
    print("SPACE = start/stop clip   |   q = quit")

    with Camera(**camera_kwargs(load_config(), args.camera)) as cam:
        while True:
            fr = cam.read()
            if fr is None:
                break
            h, w = fr.image.shape[:2]
            obs = tracker.process(fr.image, fr.timestamp_ms)
            now = time.monotonic()

            if recording:
                if obs:
                    o = obs[0]
                    frames.append((now, o.landmarks.copy(),
                                   1.0 if o.handedness == "Right" else 0.0))
                else:
                    frames.append((now, np.full((21, 3), np.nan, np.float32), -1.0))
                if args.seconds and (now - start) >= args.seconds:
                    _save(out_dir, args.label, frames, w, h)
                    recording, frames = False, []

            colour = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(fr.image,
                        f"{'REC' if recording else 'idle'}  label={args.label}  "
                        f"n={len(frames)}  hands={len(obs)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.imshow("recorder", fr.image)

            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                if recording:
                    _save(out_dir, args.label, frames, w, h)
                    recording, frames = False, []
                else:
                    recording, frames, start = True, [], now
            elif k == ord('q'):
                break
    cv2.destroyAllWindows()


def _save(out_dir: Path, label: str, frames, w: int, h: int):
    if len(frames) < 5:
        print("clip too short, discarded")
        return
    ts = np.array([f[0] for f in frames], dtype=np.float64)
    lm = np.stack([f[1] for f in frames]).astype(np.float32)   # (T, 21, 3)
    hd = np.array([f[2] for f in frames], dtype=np.float32)    # 1=Right 0=Left -1=none
    name = out_dir / f"{label}_{int(time.time()*1000)}.npz"
    np.savez_compressed(name, timestamps=ts - ts[0], landmarks=lm, handedness=hd,
                        label=label, frame_w=w, frame_h=h)
    print(f"saved {name}  ({len(frames)} frames, {ts[-1]-ts[0]:.1f}s)")


if __name__ == "__main__":
    main()
