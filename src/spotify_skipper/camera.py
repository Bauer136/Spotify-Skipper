"""Cross-platform webcam capture."""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import cv2


@dataclass
class Frame:
    image: object          # np.ndarray, BGR, mirrored
    timestamp_ms: int      # monotonic, milliseconds — MediaPipe needs this
    index: int


class Camera:
    def __init__(self, device=0, width=640, height=480, fps=30, mirror=True,
                 backend="auto"):
        self.device, self.width, self.height = device, width, height
        self.fps, self.mirror, self.backend = fps, mirror, backend
        self.cap = None
        self._i = 0
        self._t0 = None

    # --- backend selection -------------------------------------------------
    def _api(self) -> int:
        if self.backend != "auto":
            return getattr(cv2, f"CAP_{self.backend.upper()}")
        system = platform.system()
        if system == "Windows":
            return cv2.CAP_DSHOW      # opens far faster than MSMF, fewer surprises
        if system == "Linux":
            return cv2.CAP_V4L2
        return cv2.CAP_ANY            # macOS -> AVFoundation via CAP_ANY

    # --- lifecycle ---------------------------------------------------------
    def open(self):
        self.cap = cv2.VideoCapture(self.device, self._api())
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.device!r}. "
                "Linux: check `v4l2-ctl --list-devices` and that nothing else holds it. "
                "Windows: check Settings > Privacy > Camera > 'Let desktop apps access'."
            )
        # MJPG lets most webcams hit 30 fps at 640x480; raw YUYV often caps at 10.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep latency low: never hand us a frame that is 5 frames old.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._t0 = time.monotonic()
        return self

    def read(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        if self.mirror:
            img = cv2.flip(img, 1)
        ts = int((time.monotonic() - self._t0) * 1000)
        self._i += 1
        return Frame(image=img, timestamp_ms=ts, index=self._i)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
