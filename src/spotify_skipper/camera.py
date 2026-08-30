"""Cross-platform webcam capture."""
from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2

from .devices import format_table, list_devices, probe, resolve


def _api_for_backend(backend: str = "auto") -> int:
    """OpenCV capture backend for this platform (or an explicit override)."""
    if backend != "auto":
        return getattr(cv2, f"CAP_{backend.upper()}")
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW      # opens far faster than MSMF, fewer surprises
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY            # macOS -> AVFoundation via CAP_ANY


class _FrameGrabber(threading.Thread):
    """Reads the camera as fast as it will go, keeping only the NEWEST frame.

    Without this, capture and MediaPipe inference run serially and you pay both costs
    per frame: ~20 FPS capture plus ~20 FPS inference gives ~10 FPS end to end
    (1/(1/20 + 1/20)). Overlapping them costs the max of the two, not the sum.

    Only the newest frame is kept. If the consumer is slower than the camera the
    intermediate frames are dropped rather than queued — a gesture recogniser wants
    the freshest view of the world, not a backlog of stale ones.
    """

    def __init__(self, cap, t0):
        super().__init__(daemon=True, name="camera-grabber")
        self._cap, self._t0 = cap, t0
        self._cond = threading.Condition()
        self._frame = None          # (image, timestamp_ms), cleared once consumed
        self._stopping = threading.Event()
        self.captured = 0
        self._eof = False
        self.dropped = 0            # frames the consumer never saw

    def run(self):
        while not self._stopping.is_set():
            ok, img = self._cap.read()
            ts = int((time.monotonic() - self._t0) * 1000)
            with self._cond:
                if not ok:
                    self._eof = True
                    self._cond.notify_all()
                    return
                if self._frame is not None:
                    self.dropped += 1        # consumer was busy; newest wins
                self._frame = (img, ts)
                self.captured += 1
                self._cond.notify()

    def take(self, timeout):
        """Block until a frame the caller has not already seen is available."""
        with self._cond:
            if self._frame is None and not self._eof:
                self._cond.wait(timeout)
            if self._frame is None:
                return None
            f, self._frame = self._frame, None
            return f

    def stop(self):
        self._stopping.set()


@dataclass
class Frame:
    image: object          # np.ndarray, BGR, mirrored
    timestamp_ms: int      # monotonic, milliseconds — MediaPipe needs this
    index: int


class Camera:
    def __init__(self, device=0, width=640, height=480, fps=30, mirror=True,
                 backend="auto", threaded=True, read_timeout_s=2.0):
        self.device, self.width, self.height = device, width, height
        self.fps, self.mirror, self.backend = fps, mirror, backend
        self.threaded, self.read_timeout_s = threaded, read_timeout_s
        self._grabber = None
        self.cap = None
        self.resolved = None
        self._i = 0
        self._t0 = None

    # --- backend selection -------------------------------------------------
    def _api(self) -> int:
        return _api_for_backend(self.backend)

    # --- lifecycle ---------------------------------------------------------
    def open(self):
        # `device` may be an index, a /dev path, or a name fragment like "logitech".
        # Resolving here (not at construction) keeps hot-plugged cameras working.
        target = resolve(self.device)
        self.resolved = target
        self.cap = cv2.VideoCapture(target, self._api())
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.device!r} (resolved to {target!r}).\n"
                f"{format_table(probe(list_devices()))}\n"
                "Pick one of the rows marked PROBE=OK above — a 'meta' node never "
                "delivers frames. Linux: check nothing else holds the camera. "
                "Windows: Settings > Privacy > Camera > 'Let desktop apps access'."
            )
        # MJPG lets most webcams hit 30 fps at 640x480; raw YUYV often caps at 10.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Keep latency low: never hand us a frame that is 5 frames old.
        #
        # ONLY in serial mode. A single V4L2 buffer stops the driver filling frame N+1
        # while userspace holds frame N, which halves capture throughput (measured:
        # 19.9 -> 10.0 FPS on a C270). Threaded mode gets freshness a better way — the
        # grabber drains the ring and keeps only the newest frame — so it leaves the
        # driver's default buffering alone and gets the full frame rate.
        if not self.threaded:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self._t0 = time.monotonic()
        if self.threaded:
            self._grabber = _FrameGrabber(self.cap, self._t0)
            self._grabber.start()
        return self

    def read(self) -> Frame | None:
        if self._grabber is not None:
            got = self._grabber.take(self.read_timeout_s)
            if got is None:
                return None
            img, ts = got
        else:
            ok, img = self.cap.read()
            if not ok:
                return None
            ts = int((time.monotonic() - self._t0) * 1000)
        # mirrored exactly once, here — never in the grabber, so frames that get
        # dropped are never flipped for nothing (Phase 2.1 invariant)
        if self.mirror:
            img = cv2.flip(img, 1)
        self._i += 1
        return Frame(image=img, timestamp_ms=ts, index=self._i)

    @property
    def dropped(self) -> int:
        """Frames captured but never consumed — nonzero means inference is the limit."""
        return self._grabber.dropped if self._grabber else 0

    def close(self):
        # stop and join BEFORE release: releasing while the thread sits in cap.read()
        # crashes inside OpenCV
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=2.0)
            self._grabber = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
