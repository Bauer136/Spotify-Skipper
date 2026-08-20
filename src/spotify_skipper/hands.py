"""MediaPipe HandLandmarker wrapper -> plain NumPy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class HandObservation:
    """One hand in one frame, in image-normalised coordinates (0..1)."""
    landmarks: np.ndarray      # (21, 3) float32 — x, y in 0..1; z relative depth
    world: np.ndarray | None   # (21, 3) float32 metres, origin at hand centre
    handedness: str            # "Left" | "Right" (selfie-view convention)
    score: float               # handedness confidence
    timestamp_ms: int


class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task", num_hands=1,
                 min_detection_confidence=0.6, min_presence_confidence=0.6,
                 min_tracking_confidence=0.6):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing — download it (see Phase 3.1 of BUILD_GUIDE.md)")
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, bgr_image, timestamp_ms: int) -> list[HandObservation]:
        # MediaPipe rejects non-increasing timestamps in VIDEO mode.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        rgb = bgr_image[:, :, ::-1].copy()          # BGR -> RGB, contiguous
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(mp_img, timestamp_ms)

        out: list[HandObservation] = []
        for i, lms in enumerate(res.hand_landmarks):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            world = None
            if res.hand_world_landmarks:
                world = np.array([[p.x, p.y, p.z]
                                  for p in res.hand_world_landmarks[i]], dtype=np.float32)
            hd = res.handedness[i][0]
            out.append(HandObservation(arr, world, hd.category_name, hd.score,
                                       timestamp_ms))
        return out

    def close(self):
        self._lm.close()
