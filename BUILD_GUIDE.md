# Spotify Skipper — Full Build Guide

Build a local program that watches your webcam, tracks your hands with **OpenCV +
MediaPipe**, recognises a **custom hand gesture**, and fires **one** Spotify Web API
call to skip the currently playing track.

Everything — capture, landmark extraction, feature engineering, gesture
classification, debouncing — happens **on your machine**. The only bytes that leave
are the HTTPS request that skips the song (plus a token refresh roughly once an
hour, and the one-time browser login).

Target: build and tune on **Linux**, then deploy the same source tree to a
**Windows** PC.

> **Read this first:** this guide is written to be executed top-to-bottom by a human
> *or* by an agent (e.g. another Claude Code session). Every phase ends with a
> **Checkpoint** — a command to run and the exact output you should see. Do not
> continue past a failing checkpoint.

---

## Table of contents

| Phase | What you build | Est. time |
|---|---|---|
| [0](#phase-0--decisions-and-constraints) | Decisions, constraints, egress budget | 10 min |
| [1](#phase-1--project-scaffold) | Repo scaffold, venv, dependencies | 20 min |
| [2](#phase-2--camera-capture-layer) | Cross-platform camera capture | 20 min |
| [3](#phase-3--hand-tracking-layer) | MediaPipe HandLandmarker wrapper | 30 min |
| [4](#phase-4--the-custom-gesture-engine) | **The gesture engine** (features, poses, motion FSM, ML path) | 2–3 h |
| [5](#phase-5--gesture-data-recorder) | Dataset recorder (landmarks only, no video) | 40 min |
| [6](#phase-6--training-and-offline-evaluation) | Train / replay / tune harness | 1 h |
| [7](#phase-7--spotify-integration) | PKCE auth + the single skip call | 45 min |
| [8](#phase-8--wiring-the-application-together) | Main loop, config, logging, dry-run | 45 min |
| [9](#phase-9--tuning-for-zero-false-positives) | False-positive hunting | 1 h |
| [10](#phase-10--windows-deployment) | Deploy Linux → Windows | 45 min |
| [11](#phase-11--run-at-login-background-operation) | systemd user unit / Task Scheduler | 30 min |
| [12](#phase-12--verification-and-privacy-audit) | Prove only one request leaves | 20 min |
| [A–E](#appendix-a--landmark-index-reference) | Appendices: landmark map, troubleshooting, raw-PKCE, PyInstaller, tray icon | — |

---

## Phase 0 — Decisions and constraints

### 0.1 The stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | MediaPipe ships cp39–cp312 wheels for Linux **and** Windows. Python 3.13 has no MediaPipe wheel yet — **do not use 3.13**. |
| Capture | `opencv-python` | Works on V4L2 (Linux) and DirectShow/MSMF (Windows) from the same call. |
| Hand tracking | `mediapipe` Tasks API — `HandLandmarker` | 21 3-D landmarks/hand, CPU-only, ~5–15 ms/frame. The newer Tasks API (not the legacy `mp.solutions.hands`) is the supported one. |
| Gesture engine | Custom — hand-written feature extractor + rule FSM, with an optional scikit-learn classifier | You need full control over false positives; a black-box gesture recogniser will skip your songs while you eat a sandwich. |
| Spotify auth | **Authorization Code with PKCE** via `spotipy` | Desktop app ⇒ no client secret on disk. Refresh tokens work headlessly after first login. |
| Config | `config.toml` (stdlib `tomllib`) + `.env` for the client ID | No secrets in code; identical on both OSes. |
| Packaging | Ship the **source tree + bootstrap script** | PyInstaller **cannot cross-compile** Linux→Windows. See [Phase 10](#phase-10--windows-deployment). |

### 0.2 The gesture (default spec)

The default trigger is a **deliberate open-palm swipe to the right**, chosen because
it almost never occurs by accident:

1. Hand is visible with an **open palm** (all four fingers extended, thumb out).
2. Palm is held **roughly still** for ≥ 6 frames (~0.25 s) — this *arms* the gesture.
3. Palm then travels **right** by ≥ 1.6 hand-widths within **0.7 s**, with vertical
   drift < 60 % of horizontal travel.
4. Fire → **3 s cooldown** → hand must leave the open-palm pose before re-arming.

Every threshold above is a config value. Phase 4 explains how to define an entirely
different gesture (fist-flick, peace-sign hold, thumb-flick, pinch-drag) with the
same machinery.

### 0.3 Network egress budget (the "runs locally" requirement)

| Event | Destination | Frequency |
|---|---|---|
| First-time OAuth login | `accounts.spotify.com` (in your browser) | once, manually |
| Access-token refresh | `POST accounts.spotify.com/api/token` | ~1×/hour while running |
| **The skip** | `POST api.spotify.com/v1/me/player/next` | 1× per accepted gesture |

Nothing else. No frames, no landmarks, no telemetry. Phase 12 shows how to *prove*
this with `tcpdump`/Wireshark.

> Optional strictness: with the `user-read-playback-state` scope you could pre-check
> for an active device, but that is a **second** request per skip. This guide leaves
> that scope **off** by default and instead handles the `404 NO_ACTIVE_DEVICE`
> response — one request, always.

### 0.4 Hard prerequisites

- A **Spotify Premium** account. Playback-control endpoints return `403` on Free.
- A webcam on both machines.
- On Windows, an account with permission to install Python.

### 0.5 Milestone checklist

Copy this into your notes and tick as you go:

```
[ ] M1  venv + deps import cleanly (Phase 1)
[ ] M2  live webcam window with FPS counter (Phase 2)
[ ] M3  21 landmarks drawn on your hand at >20 FPS (Phase 3)
[ ] M4  on-screen debug shows pose name + engine state (Phase 4)
[ ] M5  recorded >= 20 positive and >= 10 negative clips (Phase 5)
[ ] M6  replay harness reports 0 false positives on negatives (Phase 6)
[ ] M7  `python -m spotify_skipper.auth` prints "Authorised as <you>" (Phase 7)
[ ] M8  `--dry-run` prints SKIP on gesture, nothing otherwise (Phase 8)
[ ] M9  10-minute idle soak: 0 false positives (Phase 9)
[ ] M10 real skip works on Linux (Phase 8)
[ ] M11 same tree runs on Windows (Phase 10)
[ ] M12 starts at login, no console window (Phase 11)
```

---

## Phase 1 — Project scaffold

### 1.1 Create the tree

```bash
cd ~/projects/Spotify-Skipper
mkdir -p src/spotify_skipper/gestures src/spotify_skipper/actions \
         models data/clips tests scripts logs
touch src/spotify_skipper/__init__.py \
      src/spotify_skipper/gestures/__init__.py \
      src/spotify_skipper/actions/__init__.py
```

Final layout (build it in this order):

```
Spotify-Skipper/
├── BUILD_GUIDE.md
├── README.md
├── requirements.txt
├── config.toml                 # tunables, checked in
├── .env.example                # SPOTIFY_CLIENT_ID=...
├── .gitignore
├── models/
│   └── hand_landmarker.task    # downloaded in 3.1 (~7 MB, not checked in)
├── data/clips/                 # recorded landmark clips (.npz)
├── logs/
├── scripts/
│   ├── bootstrap.sh            # Linux setup
│   └── bootstrap.ps1           # Windows setup
├── src/spotify_skipper/
│   ├── __init__.py
│   ├── __main__.py             # entry point
│   ├── config.py               # load config.toml + .env
│   ├── camera.py               # Phase 2
│   ├── hands.py                # Phase 3
│   ├── features.py             # Phase 4.2
│   ├── gestures/
│   │   ├── __init__.py
│   │   ├── poses.py            # Phase 4.3  static pose rules
│   │   ├── motion.py           # Phase 4.4  motion FSM
│   │   ├── classifier.py       # Phase 4.6  optional ML pose head
│   │   └── engine.py           # Phase 4.5  orchestrator + debouncer
│   ├── actions/
│   │   ├── __init__.py
│   │   └── spotify.py          # Phase 7
│   ├── auth.py                 # Phase 7 one-shot login
│   ├── recorder.py             # Phase 5
│   ├── train.py                # Phase 6
│   ├── replay.py               # Phase 6
│   └── app.py                  # Phase 8
└── tests/
    └── test_features.py
```

### 1.2 `.gitignore`

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.env
models/*.task
logs/*.log*
.spotify_token_cache
dist/
build/
*.spec
EOF
```

### 1.3 `requirements.txt`

Pin versions — MediaPipe's API has moved before and will again.

```bash
cat > requirements.txt <<'EOF'
opencv-contrib-python==4.11.0.86
mediapipe==0.10.14
numpy>=1.26,<2.0
spotipy==2.24.0
python-dotenv==1.0.1
platformdirs==4.2.2
# Phase 6 (optional ML path) — omit if you stay rule-based
scikit-learn==1.5.1
joblib==1.4.2
EOF
```

> `numpy<2.0` is deliberate: MediaPipe 0.10.14 wheels are built against the NumPy 1.x
> ABI. If you later see `_ARRAY_API not found`, this pin is why it exists.

> Pin `opencv-contrib-python`, **not** `opencv-python`. MediaPipe 0.10.14 depends on
> the contrib build, so listing plain `opencv-python` gets you *both* distributions
> installed. They ship the same `cv2` module path, so whichever pip unpacks second
> silently wins and your pin means nothing — `cv2.__version__` will not match what
> you asked for. One OpenCV distribution only.

### 1.4 Virtual environment (Linux)

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv libgl1 libglib2.0-0 v4l-utils
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`libgl1` and `libglib2.0-0` are OpenCV's runtime shared-library deps on a headless
or minimal Ubuntu; without them `import cv2` fails with `libGL.so.1: cannot open
shared object file`.

### 1.5 Checkpoint M1

```bash
python - <<'EOF'
from importlib.metadata import version
import cv2, mediapipe as mp, numpy as np, spotipy  # noqa: F401
print("opencv   ", cv2.__version__)
print("mediapipe", mp.__version__)
print("numpy    ", np.__version__)
print("spotipy  ", version("spotipy"))
EOF
```

Expect four version lines and **no warnings about NumPy ABI**. If `import cv2`
raises `libGL.so.1`, install the packages in 1.4.

> `spotipy` has no `__version__` attribute — its `__init__.py` is nothing but
> star-imports, and the version string exists only in the distribution metadata.
> `spotipy.__version__` raises `AttributeError` on a perfectly healthy install, so
> read it with `importlib.metadata.version()` instead. The bare `import spotipy`
> stays: proving the import works is the actual point of the checkpoint.

---

## Phase 2 — Camera capture layer

Goal: one class that opens the right camera backend on either OS, delivers
**mirrored** BGR frames, and never blocks the pipeline with stale buffered frames.

### 2.1 Why mirroring matters

An unmirrored webcam shows your right hand on the **left** of the image, so "swipe
right" would mean *decreasing* x. Flipping the frame horizontally (`cv2.flip(f, 1)`)
gives you a mirror view where user-right = image-right, which is what your intuition
expects **and** what MediaPipe assumes: its handedness labels are defined for a
selfie-view (already-mirrored) image. Flip once, at capture, and everything
downstream is consistent.

### 2.2 `src/spotify_skipper/camera.py`

```python
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
```

**Design notes an agent should preserve:**

- `timestamp_ms` is **monotonic and strictly increasing**. MediaPipe's VIDEO running
  mode rejects a timestamp that is not greater than the previous one — never use
  wall-clock time here.
- Backend selection is a function, not a constant, so the same file runs on both OSes.
- Failures raise with the *fix* in the message; you will read that message on the
  Windows box at 11 pm.

### 2.3 Smoke test

```bash
python - <<'EOF'
import time, cv2
from src.spotify_skipper.camera import Camera
with Camera() as cam:
    n, t0 = 0, time.time()
    while True:
        f = cam.read()
        if f is None: break
        n += 1
        fps = n / max(1e-6, time.time() - t0)
        cv2.putText(f.image, f"{fps:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("camera", f.image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
EOF
```

### 2.4 Checkpoint M2

You see yourself, mirrored (raise your right hand — it appears on the right of the
window), at **≥ 25 FPS**. If FPS is < 15, see [Appendix B](#appendix-b--troubleshooting).

---

## Phase 3 — Hand tracking layer

### 3.1 Download the model

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
ls -lh models/hand_landmarker.task    # ~7.5 MB
```

This file is the **only** asset the tracker needs, and it is loaded from disk at
startup — no network access at runtime. Bundle it when you copy the project to
Windows.

### 3.2 `src/spotify_skipper/hands.py`

```python
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
```

**Key facts to internalise:**

- `landmarks` x/y are normalised to the **image**, so they are *anisotropic*: 0.1 in x
  is a different physical distance from 0.1 in y unless the frame is square. Phase 4
  corrects for this with the aspect ratio. Getting this wrong makes your swipe
  threshold depend on camera resolution.
- `z` is a **relative** depth (smaller = closer to camera), not metres. Use it only
  for coarse orientation cues.
- `world` landmarks *are* metric (metres, origin at the hand's geometric centre) and
  are rotation-friendly, but they carry no information about where the hand is in the
  frame — so they are great for **pose**, useless for **motion across the frame**.
  This guide uses image landmarks for motion and canonicalised image landmarks for
  pose; `world` is available if you want to extend the classifier.
- `num_hands=1` is a real performance win. Raise it only if you plan two-hand gestures.

### 3.3 Debug viewer

```python
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
```

Two things in that header are load-bearing, and both bite only when you run the
file the obvious way:

- `python scripts/preview_hands.py` puts **`scripts/`** on `sys.path[0]`, not your
  working directory — that is how Python treats a script's own folder. Without the
  `sys.path.insert`, `from src.…` dies with `ModuleNotFoundError: No module named
  'src'` even when you are standing in the repo root. (Testing the same imports
  through `python -` or a REPL *does* work, because there `sys.path[0]` is the
  current directory — an easy way to convince yourself a broken script is fine.)
- `HandTracker`'s default `model_path` is relative to the **working directory**, so
  passing `ROOT / "models" / "hand_landmarker.task"` is what lets the viewer run
  from anywhere instead of only from the repo root.

The `FpsMeter` overlay exists to make 3.4's frame-rate check measurable rather than
eyeballed: it turns orange below the 20 FPS floor that Phase 4's thresholds assume.

### 3.4 Checkpoint M3

Run `python scripts/preview_hands.py`. You should see a numbered skeleton locked to
your hand, and the label should read **Right** when you hold up your right hand. If
it says Left, your frame is not mirrored — fix Phase 2, do not "fix" it here.

Watch the FPS: with tracking on you should still hold **20–30 FPS** at 640×480 on a
modern CPU. Everything in Phase 4 assumes ~30 FPS; if you run much slower, scale the
frame-count thresholds in `config.toml` proportionally.

**If the counter reads well under 20**, profile the loop before you touch any
thresholds — time `cam.read()` and `tracker.process()` separately. Two causes are
common and neither is your code:

1. **The webcam is throttling itself.** Many UVC cameras lengthen exposure in dim
   light and halve their frame rate to afford it, while still *reporting* 30 FPS.
   On Linux: `v4l2-ctl -d /dev/video0 --list-ctrls | grep framerate`, then
   `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`. The setting is lost on
   replug or reboot, so put it in `Camera.open()` or a udev rule. On Windows the
   equivalent lives in the driver's property page ("Low Light Compensation").
   More light on your hands helps for the same reason.
2. **The loop is serial.** `cam.read()` blocks for the next frame and *then*
   inference runs, so the two costs add instead of overlapping — ~33 ms + ~27 ms
   caps you near 16 FPS on a laptop CPU. Moving capture to a background thread that
   keeps only the newest frame makes the loop inference-bound and clears 30 FPS.
   This is what Phase 2's "never blocks the pipeline with stale buffered frames"
   is pointing at.

---

## Phase 4 — The custom gesture engine

This is the heart of the project. Read 4.1 before writing any code — the layering is
what keeps the thing tunable.

### 4.1 Architecture of the engine

A gesture is **not** a single-frame classification. It is a *pattern over time*, and
treating it as one frame is the #1 cause of songs skipping while you scratch your
nose. Four layers, each with one job:

```
 frame ─► [L1 CANONICALISE] ─► [L2 POSE] ─► [L3 MOTION FSM] ─► [L4 DEBOUNCE] ─► event
          scale/rotation/       "is this    "did the armed     "is it too soon
          translation/hand      an OPEN_     palm travel        since the last
          invariance            PALM?"       right in time?"    fire?"
```

| Layer | Input | Output | Failure it prevents |
|---|---|---|---|
| **L1 Canonicalise** | raw (21,3) landmarks | scale/rotation/position-invariant (21,2) + hand scale + palm centre | "works at 40 cm, fails at 120 cm"; "works only if I hold my hand upright" |
| **L2 Pose** | canonical landmarks | pose label + confidence (`OPEN_PALM`, `FIST`, `PEACE`, `NONE`) | random hand shapes triggering |
| **L3 Motion FSM** | pose stream + palm centre stream | candidate gesture events | static poses triggering while you just hold your hand up |
| **L4 Debounce** | candidate events | at most one action per cooldown | one swipe skipping three songs |

Both the rule-based path (4.3) and the ML path (4.6) plug into **L2** only. L1, L3
and L4 are shared. That is the design decision that lets you swap recognisers without
re-testing the whole pipeline.

### 4.2 L1 — canonicalisation and features

`src/spotify_skipper/features.py`

```python
"""Landmark canonicalisation and feature extraction.

Coordinate systems used here:
  * raw        : MediaPipe image-normalised, x,y in 0..1, ANISOTROPIC.
  * iso        : aspect-corrected, isotropic, still frame-relative. Used for MOTION.
  * canonical  : translated to wrist, scaled by hand size, rotated upright,
                 mirrored to right-hand convention. Used for POSE.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 3          # bump whenever the vector changes; training data
                             # recorded under an older version must be re-extracted

# ---- landmark indices (see Appendix A) ---------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {                      # name -> (mcp, pip, dip, tip)
    "thumb":  (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index":  (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky":  (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


# ---- step 1: make coordinates isotropic --------------------------------------
def to_iso(landmarks: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Raw (21,3) -> (21,2) isotropic frame coords.

    x is multiplied by the aspect ratio so that one unit in x equals one unit in y
    in real-world terms. y stays in 0..1, so the whole space is "screen heights".
    All distances downstream are therefore in screen-heights and are resolution
    independent.
    """
    aspect = frame_w / float(frame_h)
    p = landmarks[:, :2].astype(np.float32).copy()
    p[:, 0] *= aspect
    return p


def hand_scale(iso: np.ndarray) -> float:
    """A robust size estimate: wrist -> middle MCP distance ("hand width")."""
    return float(np.linalg.norm(iso[MIDDLE_MCP] - iso[WRIST]))


def palm_center(iso: np.ndarray) -> np.ndarray:
    """Stabler than the wrist alone; used as the motion tracking point."""
    return iso[list(PALM)].mean(axis=0)


# ---- step 2: canonical pose space --------------------------------------------
def canonicalize(iso: np.ndarray, handedness: str = "Right") -> np.ndarray | None:
    """(21,2) iso -> (21,2) canonical: wrist at origin, |wrist->middleMCP| == 1,
    that vector pointing 'up' (-y), left hands mirrored onto right-hand space.

    Returns None for degenerate hands (edge-on to the camera).
    """
    p = iso - iso[WRIST]
    v = p[MIDDLE_MCP]
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        return None
    p = p / s

    # rotate so v points to (0,-1)
    theta = np.arctan2(v[1], v[0])
    d = -np.pi / 2 - theta
    c, sn = np.cos(d), np.sin(d)
    R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
    p = p @ R.T

    if handedness == "Left":          # mirror so one model covers both hands
        p[:, 0] *= -1.0
    return p.astype(np.float32)


# ---- step 3: interpretable scalar features -----------------------------------
def extension_ratios(canon: np.ndarray) -> np.ndarray:
    """Per finger: straight-line MCP->TIP distance divided by the summed segment
    lengths. ~1.0 = fully extended, ~0.45 = tightly curled. Scale/rotation free."""
    out = []
    for name in FINGER_ORDER:
        a, b, c, d = FINGERS[name]
        chain = (np.linalg.norm(canon[b] - canon[a]) +
                 np.linalg.norm(canon[c] - canon[b]) +
                 np.linalg.norm(canon[d] - canon[c]))
        direct = np.linalg.norm(canon[d] - canon[a])
        out.append(direct / chain if chain > 1e-6 else 0.0)
    return np.asarray(out, dtype=np.float32)


def spreads(canon: np.ndarray) -> np.ndarray:
    """Distances between adjacent fingertips — separates OPEN_PALM from a flat
    'karate chop' hand, and PEACE from two-fingers-together."""
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    return np.asarray([np.linalg.norm(canon[tips[i + 1]] - canon[tips[i]])
                       for i in range(4)], dtype=np.float32)


def palm_openness(canon: np.ndarray) -> float:
    """Mean fingertip distance from the wrist. Big for open palm, small for fist."""
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    return float(np.mean([np.linalg.norm(canon[t]) for t in tips]))


def static_feature_vector(canon: np.ndarray) -> np.ndarray:
    """The vector fed to the ML pose head (Phase 4.6). Length: 42 + 5 + 4 + 1 = 52."""
    return np.concatenate([canon.reshape(-1), extension_ratios(canon),
                           spreads(canon), [palm_openness(canon)]]).astype(np.float32)


STATIC_FEATURE_LEN = 52
```

**Why each choice matters** (an agent must not "simplify" these away):

- **Aspect correction** (`to_iso`) — without it, a horizontal swipe threshold tuned at
  640×480 breaks at 1280×720 and on a different webcam.
- **Scale by wrist→middle-MCP** — this length is nearly constant regardless of finger
  pose, unlike a bounding box, which shrinks when you make a fist. Using a bounding
  box makes "fist" and "open palm" indistinguishable after normalisation.
- **Rotation alignment** — lets you tilt your hand ±40° and still be recognised. If you
  *want* rotation to be meaningful (e.g. "thumbs-up vs thumbs-down"), keep the rotation
  angle as an extra feature instead of discarding it: append `theta` to the vector.
- **Left-hand mirroring** — one trained model, both hands, half the data.
- **Ratios, not raw distances** — makes L2 depth-invariant, so the gesture works at arm's
  length and at the desk.

### 4.3 L2 (Path A) — rule-based pose recognition

`src/spotify_skipper/gestures/poses.py`

```python
"""Rule-based static pose recognition on canonical landmarks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import (FINGER_ORDER, INDEX_TIP, MIDDLE_TIP,
                        extension_ratios, palm_openness, spreads)


@dataclass(frozen=True)
class PoseResult:
    name: str          # "OPEN_PALM" | "FIST" | "PEACE" | "NONE"
    confidence: float  # 0..1, a soft margin — used for hysteresis, not thresholds


# Tunables (mirror these into config.toml [poses]).
EXT_OPEN = 0.90      # a finger this straight counts as extended
EXT_CURL = 0.72      # a finger this bent counts as curled
OPENNESS_OPEN = 1.55 # mean fingertip distance from wrist, canonical units
SPREAD_MIN = 0.25    # min mean gap between adjacent fingertips for OPEN_PALM
PEACE_V_GAP = 0.35   # min index-tip to middle-tip distance for a convincing V

# Typical measured values (canonical units, where |wrist -> middle MCP| == 1).
# Use these as sanity bounds, then replace them with YOUR numbers from 4.3's
# calibration procedure:
#   open palm : four_ext 0.93-1.00 | openness 1.70-2.10 | spread 0.28-0.45
#   fist      : four_ext 0.55-0.70 | openness 0.80-1.05 | spread 0.10-0.25
#   relaxed   : four_ext 0.75-0.90 | openness 1.30-1.60  <- the dangerous middle ground


def _states(canon: np.ndarray) -> dict[str, float]:
    return dict(zip(FINGER_ORDER, extension_ratios(canon)))


def _margin(value: float, lo: float, hi: float) -> float:
    """Soft 0..1 score: 0 at/below lo, 1 at/above hi."""
    if hi <= lo:
        return float(value >= hi)
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def classify_pose(canon: np.ndarray) -> PoseResult:
    e = _states(canon)
    openness = palm_openness(canon)
    sp = float(np.mean(spreads(canon)))

    four_ext = min(e["index"], e["middle"], e["ring"], e["pinky"])
    four_curl = max(e["index"], e["middle"], e["ring"], e["pinky"])

    # OPEN_PALM: all four fingers straight, thumb out, fingers fanned, hand "big"
    if (four_ext >= EXT_OPEN and e["thumb"] >= 0.85
            and openness >= OPENNESS_OPEN and sp >= SPREAD_MIN):
        conf = min(_margin(four_ext, EXT_OPEN - 0.08, EXT_OPEN + 0.06),
                   _margin(openness, OPENNESS_OPEN - 0.20, OPENNESS_OPEN + 0.15),
                   _margin(sp, SPREAD_MIN - 0.08, SPREAD_MIN + 0.08))
        return PoseResult("OPEN_PALM", conf)

    # FIST: everything curled
    if four_curl <= EXT_CURL and openness < 1.05:
        return PoseResult("FIST", _margin(EXT_CURL - four_curl, 0.0, 0.10))

    # PEACE: index+middle straight, ring+pinky curled, clear V between the two tips
    if (e["index"] >= EXT_OPEN and e["middle"] >= EXT_OPEN
            and e["ring"] <= EXT_CURL and e["pinky"] <= EXT_CURL
            and np.linalg.norm(canon[INDEX_TIP] - canon[MIDDLE_TIP]) >= PEACE_V_GAP):
        return PoseResult("PEACE", _margin(min(e["index"], e["middle"]),
                                           EXT_OPEN - 0.08, EXT_OPEN + 0.06))

    return PoseResult("NONE", 0.0)
```

#### How to calibrate these thresholds for *your* hand (do this, do not guess)

```python
# scripts/calibrate_pose.py  — hold the pose, read the numbers, set the constants
import cv2, numpy as np
from src.spotify_skipper.camera import Camera
from src.spotify_skipper.hands import HandTracker
from src.spotify_skipper import features as F

tracker = HandTracker()
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
        cv2.imshow("calibrate", fr.image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
```

Procedure:

1. Hold your **open palm** at three distances (30 cm, 60 cm, 100 cm) and three
   rotations (upright, tilted left 30°, tilted right 30°). Write down the **minimum**
   `four_ext`, `openness` and `spread` you observe.
2. Hold **five non-gestures** — relaxed hand on the desk, holding a mug, typing pose,
   pointing, waving hello — and write down the **maximum** values.
3. Set each threshold **midway** between the two, biased toward the non-gesture side.
   If the ranges overlap, that pose is not separable by that feature — add another
   feature or pick a different pose.

Record the numbers in a comment block in `poses.py`. Future you (or a future agent)
will need them when the recogniser drifts.

### 4.4 L3 — the motion state machine

`src/spotify_skipper/gestures/motion.py`

```python

"""Temporal gesture detection: arm on a held pose, fire on a directed travel."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class State(Enum):
    IDLE = "IDLE"          # nothing interesting
    ARMED = "ARMED"        # trigger pose held and steady; watching for travel
    PENDING = "PENDING"    # travel seen; confirming it was not the stroke of a wave
    FIRED = "FIRED"        # emitted this frame
    COOLDOWN = "COOLDOWN"  # ignoring input
    RESET = "RESET"        # cooldown over; waiting for the pose to be released


@dataclass
class MotionConfig:
    trigger_pose: str = "OPEN_PALM"
    direction: str = "right"        # right | left | up | down
    arm_frames: int = 6             # consecutive pose frames needed to arm (~0.25 s)
    arm_stillness: float = 0.55     # max hand-widths/sec of drift while arming
    travel_hand_widths: float = 1.6 # required displacement, in hand widths
    max_duration_s: float = 0.70    # must complete within this
    min_duration_s: float = 0.10    # ...but not faster than this (rejects jitter)
    lateral_ratio: float = 0.60     # |cross-axis| must stay under this * |along-axis|
    monotonic_ratio: float = 0.80   # >=80% of frames must move the right way
    pose_grace_frames: int = 3      # pose may flicker off for this many frames
    confirm_s: float = 0.40         # hold-still window that separates swipe from wave
    reversal_ratio: float = 0.50    # cancel if the hand returns this far back
    cooldown_s: float = 3.0
    release_frames: int = 5         # frames without the pose before re-arming


@dataclass
class Sample:
    t: float                # seconds, monotonic
    center: np.ndarray      # (2,) iso coords
    scale: float            # hand width, iso units
    pose: str
    conf: float


@dataclass
class GestureEvent:
    name: str
    t: float
    travel_hand_widths: float
    duration_s: float
    peak_speed: float


_AXIS = {"right": (0, +1.0), "left": (0, -1.0), "down": (1, +1.0), "up": (1, -1.0)}


class MotionDetector:
    """Feed one Sample per frame; get a GestureEvent (or None) back."""

    def __init__(self, cfg: MotionConfig, name: str = "skip"):
        self.cfg, self.name = cfg, name
        self.state = State.IDLE
        self.buf: deque[Sample] = deque(maxlen=90)     # ~3 s of history
        self._pose_run = 0
        self._miss_run = 0
        self._release_run = 0
        self._armed_at: Sample | None = None
        self._pending: tuple | None = None
        self._cooldown_until = 0.0
        self.last_reason = ""       # why we did NOT fire — invaluable when tuning

    # -- helpers -----------------------------------------------------------
    def _speed(self, n=4) -> float:
        """Recent PATH speed in hand-widths per second, over up to n+1 samples.

        Path length, not net displacement: across a direction reversal the net
        displacement cancels to ~0 and a waving hand would look 'at rest' exactly at
        its turning points -- which is precisely where a wave would otherwise anchor
        a fresh gesture window. Measuring distance travelled removes that hole.
        """
        seg = list(self.buf)[-(n + 1):]
        if len(seg) < 2:
            return 0.0
        dist = sum(float(np.linalg.norm(b.center - a.center))
                   for a, b in zip(seg, seg[1:]))
        dt = max(1e-3, seg[-1].t - seg[0].t)
        return dist / max(1e-6, seg[-1].scale) / dt

    def _reset(self, state=State.IDLE):
        self.state = state
        self._armed_at = None
        self._pending = None
        self._pose_run = 0
        self._miss_run = 0

    def _reanchor_or_expire(self, sample: "Sample", dt: float, speed: float):
        """Slide the anchor forward ONLY to a rest point; otherwise abandon.

        Re-anchoring to a moving frame would let the second half of a wave start a
        fresh window and fire. Dropping to IDLE forces a full re-arm (pose held +
        stillness), which an oscillating hand can never satisfy.
        """
        if speed <= self.cfg.arm_stillness:
            self._armed_at = sample                 # new rest point
        elif dt > self.cfg.max_duration_s:
            self.last_reason = "window expired without a rest point"
            self._reset(State.IDLE)

    # -- main --------------------------------------------------------------
    def update(self, sample: Sample | None, now: float) -> GestureEvent | None:
        cfg = self.cfg

        # ---- cooldown / release gating ----------------------------------
        if self.state in (State.COOLDOWN, State.FIRED):
            if now < self._cooldown_until:
                self.state = State.COOLDOWN
                return None
            self.state = State.RESET
            self._release_run = 0

        # ---- PENDING: confirm the travel was not half of an oscillation ---
        if self.state is State.PENDING:
            event, deadline, anchor_center, along0, scale0 = self._pending
            axis, sign = _AXIS[cfg.direction]
            if sample is not None:
                self.buf.append(sample)
                back = float(sample.center[axis] - anchor_center[axis]) * sign / scale0
                if back < along0 * (1.0 - cfg.reversal_ratio):
                    # the hand came back -> this was a wave, not a swipe
                    self.last_reason = f"reversed to {back:.2f} of {along0:.2f} (wave)"
                    self._reset(State.IDLE)
                    return None
            # a hand that vanished mid-window left the frame in the swipe direction:
            # that is consistent with a swipe, so we let the timer run.
            if now >= deadline:
                self.state = State.FIRED
                self._cooldown_until = now + cfg.cooldown_s
                self._pending = None
                self.last_reason = "fired"
                return event
            return None

        if self.state is State.RESET:
            # One continuous motion must never fire twice. Re-arm only after the user
            # either drops the trigger pose or brings the hand back to rest.
            if sample is not None:
                self.buf.append(sample)
            settled = (sample is None or sample.pose != cfg.trigger_pose
                       or self._speed() <= cfg.arm_stillness)
            if settled:
                self._release_run += 1
                if self._release_run >= cfg.release_frames:
                    self._reset(State.IDLE)
            else:
                self._release_run = 0
            return None

        # ---- no hand this frame -----------------------------------------
        if sample is None:
            self._miss_run += 1
            if self._miss_run > cfg.pose_grace_frames:
                self._reset(State.IDLE)
            return None

        self.buf.append(sample)

        # ---- pose bookkeeping -------------------------------------------
        if sample.pose == cfg.trigger_pose:
            self._pose_run += 1
            self._miss_run = 0
        else:
            self._miss_run += 1
            if self._miss_run > cfg.pose_grace_frames:
                self._reset(State.IDLE)
                self.last_reason = "pose lost"
                return None

        # ---- IDLE -> ARMED ----------------------------------------------
        if self.state is State.IDLE:
            if self._pose_run >= cfg.arm_frames and self._speed() <= cfg.arm_stillness:
                self.state = State.ARMED
                self._armed_at = sample
            return None

        # ---- ARMED: look for the travel ---------------------------------
        # INVARIANT: the travel window always starts from a REST point. This is what
        # makes waving impossible to confuse with a swipe -- a wave never rests.
        anchor = self._armed_at
        dt = sample.t - anchor.t
        speed = self._speed()

        if dt < cfg.min_duration_s:
            return None

        axis, sign = _AXIS[cfg.direction]
        cross = 1 - axis
        delta = sample.center - anchor.center
        scale = max(1e-6, anchor.scale)
        along = float(delta[axis] * sign) / scale
        lateral = abs(float(delta[cross])) / scale

        if along < cfg.travel_hand_widths or lateral > cfg.lateral_ratio * max(along, 1e-6):
            if along >= cfg.travel_hand_widths:
                self.last_reason = f"too diagonal ({lateral:.2f} vs {along:.2f})"
            else:
                self.last_reason = f"travel {along:.2f} < {cfg.travel_hand_widths}"
            self._reanchor_or_expire(sample, dt, speed)
            return None

        # monotonicity: of the frames that actually MOVED, nearly all must move the
        # right way. Near-zero steps are excluded, otherwise the still frames at the
        # start of the window drag the ratio below threshold and nothing ever fires.
        seg = [s for s in self.buf if s.t >= anchor.t]
        peak = 0.0
        if len(seg) >= 3:
            steps = np.diff([s.center[axis] for s in seg]) * sign
            moving = np.abs(steps) > 0.02 * scale        # noise floor
            if moving.sum() >= 2:
                good = float(np.mean(steps[moving] > 0))
                if good < cfg.monotonic_ratio:
                    self.last_reason = f"not monotonic ({good:.2f})"
                    self._reanchor_or_expire(sample, dt, speed)
                    return None
            dts = np.diff([s.t for s in seg])
            peak = float(np.max(np.abs(steps)) / scale / max(1e-3, float(np.mean(dts))))

        # ---- candidate accepted; confirm before firing --------------------
        event = GestureEvent(self.name, now, along, dt, peak)
        if cfg.confirm_s <= 0:
            self.state = State.FIRED
            self._cooldown_until = now + cfg.cooldown_s
            self.last_reason = "fired"
            return event
        self.state = State.PENDING
        self._pending = (event, now + cfg.confirm_s, anchor.center.copy(), along, scale)
        self.last_reason = "confirming"
        return None
```

**Design rationale — do not remove these guards:**

| Guard | Prevents |
|---|---|
| `arm_frames` + `arm_stillness` | Hand *entering* the frame at speed reading as a swipe. |
| `min_duration_s` | Single-frame landmark jumps (tracking pops) firing. |
| `lateral_ratio` | Reaching for your coffee (a diagonal move) firing. |
| `monotonic_ratio` | Waving hello (oscillating) firing — waves are non-monotonic. |
| `pose_grace_frames` | One dropped detection frame killing a valid swipe. |
| `release_frames` after cooldown | Holding an open palm and drifting right re-firing forever. |
| **`confirm_s` + `reversal_ratio`** | **Waving.** See below — this is the guard that matters most. |
| Rest-point anchoring (`_reanchor_or_expire`) | A wave's second half starting a fresh window. |
| Path-length `_speed()` | A wave's turning point reading as "at rest". |
| `last_reason` | You, at 1 am, wondering why it never fires. Put it on screen. |

#### Why waving is the hard case (and how the three guards defeat it)

The **first stroke of a wave is physically identical to a swipe**: same pose, same
distance, same direction, started from rest. No causal detector can separate them from
that stroke alone. Three mechanisms, in combination, are what actually work — each was
verified against simulated wave trajectories from 0.8 Hz to 2.5 Hz:

1. **Path-length speed.** `_speed()` sums per-frame step *distances* rather than net
   displacement. Net displacement across a direction reversal cancels to ~0, so a
   waving hand looks perfectly "at rest" at exactly its turning points — which is
   where it would otherwise anchor a fresh gesture window. Measuring distance
   travelled closes that hole.
2. **Rest-point anchoring.** The travel window may only *begin* at a frame where the
   hand is at rest. If the window expires without firing, the attempt is abandoned to
   `IDLE` and a full re-arm (pose held + stillness) is required. A hand in continuous
   oscillation never satisfies that.
3. **The confirmation window.** On meeting the travel criteria the detector enters
   `PENDING` and waits `confirm_s`. If the hand returns more than `reversal_ratio` of
   the way back, the candidate is discarded as a wave. A swipe ends with the hand out,
   dropped, or off-frame; a wave comes back. This is the only guard that catches slow,
   wide waves — and it costs `confirm_s` of latency before the track skips.

Measured trade-off (6 swipe variants, 10 decoys including 5 wave frequencies):

| `confirm_s` | swipes detected | false positives | note |
|---|---|---|---|
| 0.00 (off) | 6/6 | 2/10 | slow wide waves (≤ 1 Hz) fire |
| 0.20 | 6/6 | 2/10 | still too short for a 1 Hz wave |
| 0.30 | 6/6 | 1/10 | 0.8 Hz wave still fires |
| **0.40 (default)** | **6/6** | **0/10** | ~0.4 s added latency |
| 0.60 | 5/6 | 0/10 | rejects a swipe whose hand returns within 0.6 s |

If the skip feels sluggish, lower `confirm_s` to 0.30 and accept that a slow deliberate
wave at someone can skip a track. If you wave a lot on video calls, raise it to 0.50.

**Consequence to document for users:** *swipe and hold briefly.* Snapping the hand
straight back to where it started within `confirm_s` reads as a wave and is discarded
by design.

**Choosing a different gesture** — all in config, no code changes:

| Desired gesture | `trigger_pose` | `direction` | Notes |
|---|---|---|---|
| Open-palm swipe right (default) | `OPEN_PALM` | `right` | Best signal-to-noise. |
| Swipe left = previous track | `OPEN_PALM` | `left` | Instantiate a *second* `MotionDetector` with its own action. |
| Fist "punch" down | `FIST` | `down` | Raise `travel_hand_widths` to ~2.0; fists are small so hand-width units shrink. |
| Peace-sign hold (no motion) | `PEACE` | — | Use `HoldDetector` (4.4.1) instead. |

#### 4.4.1 Alternative L3: hold detector (motionless gestures)

```python
@dataclass
class HoldConfig:
    trigger_pose: str = "PEACE"
    hold_s: float = 1.2         # must be this long — a real deliberate commitment
    max_drift: float = 0.9      # hand-widths of allowed wander during the hold
    min_conf: float = 0.6
    cooldown_s: float = 3.0
    release_frames: int = 5


class HoldDetector:
    """Fires when a pose is held steadily for hold_s. Simpler and more reliable
    than a swipe in cluttered scenes, but slower to trigger."""

    def __init__(self, cfg: HoldConfig, name="skip"):
        self.cfg, self.name = cfg, name
        self._start: Sample | None = None
        self._cooldown_until = 0.0
        self._release_run = 0
        self.state = State.IDLE

    def update(self, sample, now):
        cfg = self.cfg
        if now < self._cooldown_until:
            self.state = State.COOLDOWN
            return None
        if self.state is State.COOLDOWN:      # cooldown just ended
            self.state = State.RESET
            self._release_run = 0
        if self.state is State.RESET:
            if sample is None or sample.pose != cfg.trigger_pose:
                self._release_run += 1
                if self._release_run >= cfg.release_frames:
                    self.state = State.IDLE
            return None

        if sample is None or sample.pose != cfg.trigger_pose or sample.conf < cfg.min_conf:
            self._start = None
            self.state = State.IDLE
            return None

        if self._start is None:
            self._start = sample
            self.state = State.ARMED
            return None

        drift = float(np.linalg.norm(sample.center - self._start.center)
                      / max(1e-6, self._start.scale))
        if drift > cfg.max_drift:
            self._start = sample                      # moved: restart the clock
            return None
        if sample.t - self._start.t >= cfg.hold_s:
            self.state = State.FIRED
            self._cooldown_until = now + cfg.cooldown_s
            self._start = None
            return GestureEvent(self.name, now, 0.0, cfg.hold_s, 0.0)
        return None
```

### 4.5 L4 — the engine (orchestrator + debouncer)

`src/spotify_skipper/gestures/engine.py`

```python
"""Glues L1–L4 together and exposes one method: process(frame) -> events."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .. import features as F
from .motion import GestureEvent, MotionConfig, MotionDetector, Sample, State


class GestureEngine:
    def __init__(self, pose_fn, detector, smoothing_frames: int = 3,
                 min_hand_scale: float = 0.05, max_hand_scale: float = 0.60):
        """
        pose_fn : callable(canonical_landmarks) -> PoseResult
                  (rule-based `classify_pose` or the ML head from 4.6)
        detector: MotionDetector | HoldDetector
        """
        self.pose_fn = pose_fn
        self.detector = detector
        self.smoothing_frames = smoothing_frames
        self.min_hand_scale = min_hand_scale
        self.max_hand_scale = max_hand_scale
        self._pose_hist: deque[str] = deque(maxlen=max(1, smoothing_frames))
        self.debug: dict = {}

    def _smoothed_pose(self, pose_name: str) -> str:
        """Majority vote over the last N frames — kills single-frame flicker in
        BOTH directions (spurious detections and spurious drops)."""
        self._pose_hist.append(pose_name)
        if len(self._pose_hist) < self._pose_hist.maxlen:
            return "NONE"
        vals, counts = np.unique(np.array(self._pose_hist), return_counts=True)
        return str(vals[int(np.argmax(counts))])

    def process(self, observations, frame_w, frame_h, now=None) -> list[GestureEvent]:
        now = time.monotonic() if now is None else now
        sample = None

        if observations:
            # Deterministic choice when several hands are visible: the biggest one.
            best, best_scale, best_iso = None, -1.0, None
            for obs in observations:
                iso = F.to_iso(obs.landmarks, frame_w, frame_h)
                s = F.hand_scale(iso)
                if s > best_scale:
                    best, best_scale, best_iso = obs, s, iso

            # Reject implausible hand sizes (a face misdetected as a hand, or a hand
            # so far away the landmarks are noise).
            if self.min_hand_scale <= best_scale <= self.max_hand_scale:
                canon = F.canonicalize(best_iso, best.handedness)
                if canon is not None:
                    pose = self.pose_fn(canon)
                    smoothed = self._smoothed_pose(pose.name)
                    sample = Sample(t=now, center=F.palm_center(best_iso),
                                    scale=best_scale, pose=smoothed,
                                    conf=pose.confidence)
                    self.debug = {"pose_raw": pose.name, "pose": smoothed,
                                  "conf": round(pose.confidence, 2),
                                  "scale": round(best_scale, 3),
                                  "hand": best.handedness}
        if sample is None:
            self._pose_hist.append("NONE")
            self.debug = {"pose": "NONE", "conf": 0.0}

        event = self.detector.update(sample, now)
        self.debug["state"] = self.detector.state.value
        self.debug["reason"] = getattr(self.detector, "last_reason", "")
        return [event] if event else []
```

### 4.6 L2 (Path B) — the trainable pose head *(optional but recommended)*

Rules are fast to write and easy to debug, but if your pose is unusual (or your hand
is), a small classifier trained on **your** hand beats hand-tuned thresholds. It plugs
into the same `pose_fn` slot.

`src/spotify_skipper/gestures/classifier.py`

```python
"""scikit-learn pose head. Same interface as poses.classify_pose."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .. import features as F
from .poses import PoseResult


class MLPoseClassifier:
    def __init__(self, model_path="models/pose_clf.joblib", min_conf=0.75):
        bundle = joblib.load(Path(model_path))
        if bundle["feature_version"] != F.FEATURE_VERSION:
            raise RuntimeError(
                f"Model trained with feature v{bundle['feature_version']} but code is "
                f"v{F.FEATURE_VERSION}. Re-run `python -m spotify_skipper.train`.")
        self.pipe = bundle["pipeline"]
        self.labels = bundle["labels"]
        self.min_conf = min_conf

    def __call__(self, canon: np.ndarray) -> PoseResult:
        x = F.static_feature_vector(canon).reshape(1, -1)
        proba = self.pipe.predict_proba(x)[0]
        i = int(np.argmax(proba))
        label, conf = self.labels[i], float(proba[i])
        if label == "NONE" or conf < self.min_conf:
            return PoseResult("NONE", conf)
        return PoseResult(label, conf)
```

Two rules that decide whether this works:

1. **You must train a `NONE` class**, and it must be the *largest* class — recorded
   from real life: typing, drinking, gesturing while talking, hand at rest, phone in
   hand. A classifier trained only on gesture poses will confidently call everything a
   gesture.
2. **Never let the classifier trigger directly.** It only produces the pose label; L3
   and L4 still decide whether an action fires.

### 4.7 Checkpoint M4 — the debug HUD

Build this now; you will use it for the rest of the project.

```python
# scripts/debug_engine.py
import cv2
from src.spotify_skipper.camera import Camera
from src.spotify_skipper.hands import HandTracker
from src.spotify_skipper.gestures.poses import classify_pose
from src.spotify_skipper.gestures.motion import MotionConfig, MotionDetector
from src.spotify_skipper.gestures.engine import GestureEngine

engine = GestureEngine(classify_pose, MotionDetector(MotionConfig()))
tracker = HandTracker()
fired_at = 0.0
with Camera() as cam:
    while True:
        fr = cam.read()
        if fr is None: break
        h, w = fr.image.shape[:2]
        obs = tracker.process(fr.image, fr.timestamp_ms)
        events = engine.process(obs, w, h)
        y = 25
        for k, v in engine.debug.items():
            cv2.putText(fr.image, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 22
        if events:
            fired_at = cv2.getTickCount() / cv2.getTickFrequency()
            print("EVENT", events[0])
        nowt = cv2.getTickCount() / cv2.getTickFrequency()
        if nowt - fired_at < 1.0:
            cv2.rectangle(fr.image, (0, 0), (w-1, h-1), (0, 0, 255), 12)
            cv2.putText(fr.image, "SKIP", (w//2 - 60, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("engine", fr.image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
cv2.destroyAllWindows()
```

Pass criteria for M4:

- Holding an open palm still shows `state: ARMED`.
- A deliberate swipe right flashes **SKIP** and prints an event.
- A slow drift, a wave, and a diagonal reach do **not** fire — and `reason` tells you
  which guard rejected each one.
- Typing/idle shows `state: IDLE`, `pose: NONE`.

---

## Phase 5 — Gesture data recorder

You need recorded data for two reasons: to **train** the optional ML head, and — more
importantly — to **replay** real footage through the engine so you can tune thresholds
without waving at your monitor 400 times.

**Privacy by construction: the recorder stores landmarks, not images.** A `.npz` clip
is a few kilobytes of coordinates; nobody can reconstruct your living room from it.

### 5.1 `src/spotify_skipper/recorder.py`

```python
"""Record labelled landmark clips to data/clips/*.npz (no images stored)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .hands import HandTracker


def main():
    ap = argparse.ArgumentParser(description="Record hand-landmark clips.")
    ap.add_argument("--label", required=True,
                    help="clip label, e.g. skip_swipe / idle / decoy_wave")
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="auto-stop after N seconds (0 = manual stop with SPACE)")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = HandTracker(model_path=args.model)

    recording, frames, start = False, [], 0.0
    print("SPACE = start/stop clip   |   q = quit")

    with Camera() as cam:
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
```

### 5.2 What to record (this list *is* the quality of your recogniser)

Run each command and record the specified clips.

**Positives** — the gesture itself, ≥ 20 clips total:

```bash
python -m spotify_skipper.recorder --label skip_swipe
```

Vary deliberately: right hand and left hand; 40 cm / 80 cm / 120 cm from the camera;
slow swipe and fast swipe; hand tilted ±30°; sitting upright and slouched; sleeves up
and down; overhead light and lamp-only.

**Hard negatives** — things that must *never* fire, ≥ 15 clips:

```bash
python -m spotify_skipper.recorder --label decoy_wave        # waving hello
python -m spotify_skipper.recorder --label decoy_reach        # reaching for a mug
python -m spotify_skipper.recorder --label decoy_type         # typing
python -m spotify_skipper.recorder --label decoy_point        # pointing at the screen
python -m spotify_skipper.recorder --label decoy_scratch      # scratching your face
python -m spotify_skipper.recorder --label decoy_talk         # talking with your hands
python -m spotify_skipper.recorder --label decoy_stretch      # stretching arms
```

**Idle soak** — long, boring, no gestures, ≥ 10 minutes total:

```bash
python -m spotify_skipper.recorder --label idle --seconds 300
```

Sit and work normally. This clip is the single most valuable asset in the project: it
is what you measure false positives against.

### 5.3 Checkpoint M5

```bash
ls data/clips | wc -l
python - <<'EOF'
from pathlib import Path
import numpy as np, collections
c = collections.Counter()
secs = collections.Counter()
for p in Path("data/clips").glob("*.npz"):
    d = np.load(p, allow_pickle=True)
    lab = str(d["label"])
    c[lab] += 1
    secs[lab] += float(d["timestamps"][-1])
for k in sorted(c):
    print(f"{k:16s} {c[k]:3d} clips  {secs[k]/60:5.1f} min")
EOF
```

You want ≥ 20 `skip_swipe`, ≥ 15 decoys, ≥ 10 minutes of `idle`.

---

## Phase 6 — Training and offline evaluation

### 6.1 Replay harness — the tuning workbench

`src/spotify_skipper/replay.py`

```python
"""Replay recorded clips through the gesture engine and score it.

This is how you tune thresholds: change config, re-run, compare numbers.
No webcam, no waving, deterministic, seconds per experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config, build_engine
from .hands import HandObservation


def clip_events(path: Path, cfg) -> tuple[str, int, float]:
    d = np.load(path, allow_pickle=True)
    lm, ts, hd = d["landmarks"], d["timestamps"], d["handedness"]
    w, h = int(d["frame_w"]), int(d["frame_h"])
    engine = build_engine(cfg)                     # fresh engine per clip
    n = 0
    for i in range(len(ts)):
        obs = []
        if not np.isnan(lm[i]).any() and hd[i] >= 0:
            obs = [HandObservation(lm[i], None,
                                   "Right" if hd[i] > 0.5 else "Left", 1.0,
                                   int(ts[i] * 1000))]
        n += len(engine.process(obs, w, h, now=float(ts[i])))
    return str(d["label"]), n, float(ts[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tp = fn = fp = 0
    idle_secs = idle_fp = 0.0
    for p in sorted(Path(args.clips).glob("*.npz")):
        label, n, dur = clip_events(p, cfg)
        positive = label.startswith("skip")
        if positive:
            tp += min(n, 1)
            fn += 1 - min(n, 1)
            fp += max(0, n - 1)          # extra fires within one clip = false positives
        else:
            fp += n
            if label == "idle":
                idle_secs += dur
                idle_fp += n
        if args.verbose and (n != (1 if positive else 0)):
            print(f"  ! {p.name:40s} label={label:12s} fired={n}")

    recall = tp / max(1, tp + fn)
    print(f"\ndetections : TP={tp}  FN={fn}  FP={fp}")
    print(f"recall     : {recall:.1%}   (target >= 90%)")
    if idle_secs:
        print(f"idle FP rate: {idle_fp / (idle_secs/3600):.2f} per hour "
              f"over {idle_secs/60:.1f} min   (target 0.00)")


if __name__ == "__main__":
    main()
```

### 6.2 Training the optional ML pose head

`src/spotify_skipper/train.py`

```python
"""Train the static-pose classifier from recorded clips."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F

# Which frames of which clips carry which pose label.
# The middle of a swipe clip is the trigger pose; decoys/idle are NONE.
LABEL_MAP = {"skip_swipe": "OPEN_PALM"}
DEFAULT_LABEL = "NONE"


def load_dataset(clips_dir: Path):
    X, y, groups = [], [], []
    for gi, p in enumerate(sorted(clips_dir.glob("*.npz"))):
        d = np.load(p, allow_pickle=True)
        lm, hd = d["landmarks"], d["handedness"]
        w, h = int(d["frame_w"]), int(d["frame_h"])
        label = LABEL_MAP.get(str(d["label"]), DEFAULT_LABEL)
        for i in range(len(lm)):
            if np.isnan(lm[i]).any() or hd[i] < 0:
                continue
            iso = F.to_iso(lm[i], w, h)
            canon = F.canonicalize(iso, "Right" if hd[i] > 0.5 else "Left")
            if canon is None:
                continue
            X.append(F.static_feature_vector(canon))
            y.append(label)
            groups.append(gi)          # group = clip, so CV never splits a clip
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="models/pose_clf.joblib")
    args = ap.parse_args()

    X, y, g = load_dataset(Path(args.clips))
    print(f"{len(X)} samples, {X.shape[1]} features")
    for lab in sorted(set(y)):
        print(f"  {lab:12s} {int((y == lab).sum())}")
    assert X.shape[1] == F.STATIC_FEATURE_LEN, "feature length mismatch"

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                              alpha=1e-3, random_state=0)),
    ])
    # Group CV by clip: otherwise adjacent near-identical frames leak between
    # train and test and you get a meaningless 99.9%.
    scores = cross_val_score(pipe, X, y, groups=g,
                             cv=GroupKFold(n_splits=min(5, len(set(g)))))
    print(f"grouped CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    pipe.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe,
                 "labels": list(pipe.named_steps["mlp"].classes_),
                 "feature_version": F.FEATURE_VERSION}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
```

```bash
python -m spotify_skipper.train
```

Interpreting the CV number: **> 0.97 grouped accuracy** is fine. If it is ~1.000 you
probably forgot `groups=` and are leaking. If it is < 0.90, your `NONE` class contains
frames that really are open palms (you waved with an open hand in a decoy clip) — that
is a labelling problem, not a model problem.

Then switch the engine over in `config.toml`:

```toml
[pose]
backend = "ml"     # "rules" | "ml"
```

### 6.3 Unit tests

`tests/test_features.py`

```python
import numpy as np

from src.spotify_skipper import features as F

ASPECT = 640 / 480.0


def _fake_hand(scale=1.0, angle=0.0, offset=(0.5, 0.5), aspect=ASPECT):
    """A synthetic flat open hand in RAW (image-normalised) coordinates.

    x is pre-divided by the aspect ratio so that to_iso() restores the true
    proportions. Without this the rotation and the anisotropic x-scaling do not
    commute, the shape is sheared, and the invariance test fails for the wrong
    reason.
    """
    p = np.zeros((21, 3), np.float32)
    p[F.WRIST] = (0.0, 0.0, 0)
    for _, (mcp, pip, dip, tip) in F.FINGERS.items():
        x = (_col(mcp) - 2) * 0.25
        # the middle MCP is the scale reference, so it sits further out
        y0 = -1.0 if mcp == F.MIDDLE_MCP else -0.75
        for j, idx in enumerate((mcp, pip, dip, tip)):
            p[idx] = (x, y0 - 0.28 * j, 0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], np.float32)
    p[:, :2] = p[:, :2] * scale @ R.T
    p[:, 0] /= aspect
    p[:, :2] += np.asarray(offset, np.float32)
    return p


def _col(mcp):
    return [F.THUMB_CMC, F.INDEX_MCP, F.MIDDLE_MCP, F.RING_MCP, F.PINKY_MCP].index(mcp)


def test_canonicalization_is_scale_rotation_and_position_invariant():
    a = F.canonicalize(F.to_iso(_fake_hand(0.10, 0.0), 640, 480))
    b = F.canonicalize(F.to_iso(_fake_hand(0.25, 0.5, (0.2, 0.8)), 640, 480))
    assert np.allclose(a, b, atol=1e-4)


def test_canonical_frame_is_normalised_and_upright():
    a = F.canonicalize(F.to_iso(_fake_hand(0.2), 640, 480))
    assert np.allclose(a[F.WRIST], (0.0, 0.0), atol=1e-5)
    assert np.allclose(a[F.MIDDLE_MCP], (0.0, -1.0), atol=1e-5)


def test_left_hand_is_mirrored_onto_right_hand_space():
    raw = _fake_hand(0.2, 0.3)
    r = F.canonicalize(F.to_iso(raw, 640, 480), "Right")
    l = F.canonicalize(F.to_iso(raw, 640, 480), "Left")
    assert np.allclose(l[:, 0], -r[:, 0], atol=1e-5)
    assert np.allclose(l[:, 1], r[:, 1], atol=1e-5)


def test_extension_ratios_saturate_for_a_straight_hand():
    a = F.canonicalize(F.to_iso(_fake_hand(0.2), 640, 480))
    ext = F.extension_ratios(a)
    assert ext.shape == (5,)
    assert np.all(ext > 0.95)


def test_feature_vector_length_matches_declared_constant():
    a = F.canonicalize(F.to_iso(_fake_hand(0.2), 640, 480))
    assert F.static_feature_vector(a).shape == (F.STATIC_FEATURE_LEN,)


def test_degenerate_hand_returns_none():
    flat = np.zeros((21, 3), np.float32)
    assert F.canonicalize(F.to_iso(flat, 640, 480)) is None
```

> These tests were run against the code in Phase 4.2 and pass: invariance residual
> ~4e-7, extension ratios 1.000, feature length 52. If you change `features.py` and a
> test fails, the feature extractor is wrong — not the test.

`tests/test_motion.py` — the behavioural regression suite

```python
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
```

> This suite is the reason the FSM in 4.4 looks the way it does. Every guard in it was
> added in response to a decoy in this file that fired when it should not have. Run it
> after **any** change to `motion.py` or to the `[gesture]` section of `config.toml`:
> it takes 0.2 s and no webcam.

```bash
pip install pytest && python -m pytest tests -q
```

Expected: **27 passed** (6 feature tests + 21 motion tests).

### 6.4 Checkpoint M6

```bash
python -m spotify_skipper.replay --verbose
```

Targets before you go anywhere near the Spotify API:

- `recall ≥ 90 %` on `skip_swipe` clips,
- `FP = 0` on decoys,
- `idle FP rate = 0.00 per hour`.

If recall is low, loosen `travel_hand_widths` or `arm_frames`. If FP > 0, look at which
clip fired (`--verbose` prints it) and tighten the specific guard that let it through.
**Always change one value at a time and re-run.** This loop takes seconds; the
webcam loop takes minutes.

---

## Phase 7 — Spotify integration

### 7.1 Create the Spotify app

1. Go to <https://developer.spotify.com/dashboard> and log in with the account whose
   playback you want to control.
2. **Create app**:
   - *App name*: `Spotify Skipper` (anything).
   - *Redirect URI*: **`http://127.0.0.1:8888/callback`** — type it exactly. Spotify
     rejects the literal hostname `localhost`; it requires the loopback **IP**. A
     mismatch of even a trailing slash produces `INVALID_CLIENT: Invalid redirect URI`.
   - *Which API/SDKs*: tick **Web API**.
3. Save, open **Settings**, copy the **Client ID**. You do **not** need the client
   secret — PKCE is designed for apps like this that cannot keep a secret.
4. The app starts in *development mode*, which is fine: the owner account always has
   access (up to 25 additional users can be added under **User Management**).

### 7.2 Store the client ID

```bash
cat > .env.example <<'EOF'
# Copy to .env and fill in. .env is gitignored.
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
EOF
cp .env.example .env
$EDITOR .env
```

The client ID is not a secret in the PKCE model (it ships inside every mobile app),
but the **token cache is** — it grants playback control of your account. Phase 7.4
chmods it.

### 7.3 `src/spotify_skipper/actions/spotify.py`

```python
"""Spotify playback actions. Exactly one HTTPS request per skip."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyPKCE

log = logging.getLogger(__name__)

# The narrowest scope that can skip a track. Deliberately no read scopes:
# we never need to inspect your library or listening history.
SCOPE = "user-modify-playback-state"


class SpotifySkipper:
    def __init__(self, client_id: str | None = None, redirect_uri: str | None = None,
                 cache_path: str = ".spotify_token_cache", open_browser: bool = True):
        client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        if not client_id:
            raise RuntimeError("SPOTIFY_CLIENT_ID not set (see .env.example)")
        redirect_uri = (redirect_uri or os.environ.get("SPOTIFY_REDIRECT_URI")
                        or "http://127.0.0.1:8888/callback")

        cache = Path(cache_path)
        self.auth = SpotifyPKCE(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=SCOPE,
            open_browser=open_browser,
            cache_handler=CacheFileHandler(cache_path=str(cache)),
        )
        self.client = spotipy.Spotify(auth_manager=self.auth, requests_timeout=6,
                                      retries=0)      # we do our own error handling
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._rate_limited_until = 0.0
        self._cache_path = cache

    # -- one-time interactive login ---------------------------------------
    def authorize(self) -> str:
        """Opens a browser once, then persists a refresh token. Returns the token."""
        token = self.auth.get_access_token()      # cached/refreshed automatically
        self._harden_cache()
        return token

    def _harden_cache(self):
        try:
            if self._cache_path.exists() and os.name == "posix":
                os.chmod(self._cache_path, 0o600)
        except OSError:
            pass

    # -- the action ---------------------------------------------------------
    def skip(self, device_id: str | None = None) -> tuple[bool, str]:
        """Skip to the next track. Returns (ok, human-readable message).

        This is THE network call: POST https://api.spotify.com/v1/me/player/next
        Everything else in the program is local.
        """
        now = time.monotonic()
        if now < self._rate_limited_until:
            return False, "rate limited, backing off"
        with self._lock:
            if now - self._last_call < 0.5:      # belt-and-braces re-entrancy guard
                return False, "duplicate call suppressed"
            self._last_call = now
            try:
                self.client.next_track(device_id=device_id)   # 204 No Content on success
                log.info("skip: ok")
                return True, "skipped"
            except spotipy.SpotifyException as e:
                return False, self._explain(e)
            except Exception as e:                            # network down, DNS, TLS
                log.warning("skip failed: %s", e)
                return False, f"network error: {e}"

    def _explain(self, e: spotipy.SpotifyException) -> str:
        code = e.http_status
        if code == 404:
            return ("no active device — start playback on a device once, then the "
                    "gesture will work")
        if code == 403:
            return ("forbidden — playback control requires Spotify Premium, or the "
                    "current context does not allow skipping")
        if code == 401:
            return "token rejected — delete .spotify_token_cache and re-authorise"
        if code == 429:
            wait = int(e.headers.get("Retry-After", "5")) if e.headers else 5
            self._rate_limited_until = time.monotonic() + wait + 1
            return f"rate limited, retry after {wait}s"
        return f"spotify error {code}: {e.msg}"
```

### 7.4 One-shot authorisation script

`src/spotify_skipper/auth.py`

```python
"""Run once per machine: python -m spotify_skipper.auth"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from .actions.spotify import SpotifySkipper


def main() -> int:
    load_dotenv()
    sk = SpotifySkipper()
    print("A browser window will open. Log in and click Agree.")
    print("If no browser opens (headless box), copy the printed URL, open it "
          "elsewhere, then paste the FULL redirected URL back here.")
    sk.authorize()
    me = sk.client.me() if False else None      # deliberately skipped: needs an extra
                                                # scope and an extra request
    print("Authorised. Token cached in .spotify_token_cache")
    print("Now start playing something on any Spotify device, then run:")
    print("  python -m spotify_skipper --test-skip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```bash
python -m spotify_skipper.auth
```

What happens: spotipy opens your browser at `accounts.spotify.com`, spins up a
temporary local HTTP listener on `127.0.0.1:8888`, catches the redirect with the
authorisation code, and exchanges it (with the PKCE verifier) for an access token +
refresh token, which it writes to `.spotify_token_cache`. After this, refreshes are
silent and non-interactive — **you never need the browser again** unless you revoke
access or delete the cache.

### 7.5 Checkpoint M7

```bash
ls -l .spotify_token_cache          # exists, mode 600 on Linux
# start playback on your phone or desktop app, then:
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.spotify_skipper.actions.spotify import SpotifySkipper
ok, msg = SpotifySkipper().skip()
print(ok, msg)
EOF
```

The track should change. If you get `no active device`, Spotify has no idea where to
send the command — open the Spotify app anywhere and press play once; the device stays
"active" for a while after pausing.

---

## Phase 8 — Wiring the application together

> **Ordering note:** `replay.py` (Phase 6) imports `config.py`, written below. If you
> are following strictly in order, write `config.py` and `config.toml` now and then
> re-run the Phase 6 checkpoint.

### 8.1 `config.toml`

Every tunable lives here. Nothing in this file is a secret, so it is safe to commit —
which means your Linux tuning travels to Windows unchanged.

```toml
# ---------------------------------------------------------------- camera
[camera]
device  = 0          # Linux: 0 -> /dev/video0. Windows: 0 -> first DirectShow cam.
width   = 640
height  = 480
fps     = 30
mirror  = true       # keep true: user-right == image-right (see Phase 2.1)
backend = "auto"     # auto | v4l2 | dshow | msmf | any

# ---------------------------------------------------------------- tracking
[hands]
model_path               = "models/hand_landmarker.task"
num_hands                = 1
min_detection_confidence = 0.6
min_presence_confidence  = 0.6
min_tracking_confidence  = 0.6

# ---------------------------------------------------------------- pose (L2)
[pose]
backend        = "rules"                     # rules | ml
ml_model_path  = "models/pose_clf.joblib"
ml_min_conf    = 0.75
smoothing_frames = 3                         # majority vote window
min_hand_scale = 0.05                        # reject implausibly small hands
max_hand_scale = 0.60                        # ...and implausibly large ones

# ---------------------------------------------------------------- gesture (L3/L4)
[gesture]
type                = "motion"     # motion | hold
trigger_pose        = "OPEN_PALM"
direction           = "right"
arm_frames          = 6
arm_stillness       = 0.55
travel_hand_widths  = 1.6
min_duration_s      = 0.10
max_duration_s      = 0.70
lateral_ratio       = 0.60
monotonic_ratio     = 0.80
pose_grace_frames   = 3
confirm_s           = 0.40    # wave rejection window; see 4.4. 0 disables it
reversal_ratio      = 0.50    # cancel if the hand returns this fraction of the way back
cooldown_s          = 3.0
release_frames      = 5
# used only when type = "hold"
hold_s              = 1.2
max_drift           = 0.9

# ---------------------------------------------------------------- action
[spotify]
enabled      = true
cache_path   = ".spotify_token_cache"
device_id    = ""        # "" = whatever device is currently active

# ---------------------------------------------------------------- runtime
[app]
preview    = true        # show the debug window; set false for background operation
log_level  = "INFO"
log_file   = "logs/skipper.log"
beep       = true        # audible confirmation when a gesture fires
```

### 8.2 `src/spotify_skipper/config.py`

```python
"""Load config.toml + .env and build wired-up objects."""
from __future__ import annotations

import tomllib                       # stdlib on Python 3.11+
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name, {}))


def load_config(path: str | Path = "config.toml") -> Config:
    load_dotenv()                                  # pulls SPOTIFY_CLIENT_ID
    p = Path(path)
    with p.open("rb") as fh:
        return Config(tomllib.load(fh), p)


def build_pose_fn(cfg: Config):
    pose = cfg.section("pose")
    if pose.get("backend", "rules") == "ml":
        from .gestures.classifier import MLPoseClassifier
        return MLPoseClassifier(pose.get("ml_model_path", "models/pose_clf.joblib"),
                                float(pose.get("ml_min_conf", 0.75)))
    from .gestures.poses import classify_pose
    return classify_pose


def build_detector(cfg: Config):
    g = cfg.section("gesture")
    if g.get("type", "motion") == "hold":
        from .gestures.motion import HoldConfig, HoldDetector
        return HoldDetector(HoldConfig(
            trigger_pose=g.get("trigger_pose", "PEACE"),
            hold_s=float(g.get("hold_s", 1.2)),
            max_drift=float(g.get("max_drift", 0.9)),
            cooldown_s=float(g.get("cooldown_s", 3.0)),
            release_frames=int(g.get("release_frames", 5)),
        ))
    from .gestures.motion import MotionConfig, MotionDetector
    return MotionDetector(MotionConfig(
        trigger_pose=g.get("trigger_pose", "OPEN_PALM"),
        direction=g.get("direction", "right"),
        arm_frames=int(g.get("arm_frames", 6)),
        arm_stillness=float(g.get("arm_stillness", 0.55)),
        travel_hand_widths=float(g.get("travel_hand_widths", 1.6)),
        min_duration_s=float(g.get("min_duration_s", 0.10)),
        max_duration_s=float(g.get("max_duration_s", 0.70)),
        lateral_ratio=float(g.get("lateral_ratio", 0.60)),
        monotonic_ratio=float(g.get("monotonic_ratio", 0.80)),
        pose_grace_frames=int(g.get("pose_grace_frames", 3)),
        confirm_s=float(g.get("confirm_s", 0.40)),
        reversal_ratio=float(g.get("reversal_ratio", 0.50)),
        cooldown_s=float(g.get("cooldown_s", 3.0)),
        release_frames=int(g.get("release_frames", 5)),
    ))


def build_engine(cfg: Config):
    from .gestures.engine import GestureEngine
    pose = cfg.section("pose")
    return GestureEngine(
        build_pose_fn(cfg), build_detector(cfg),
        smoothing_frames=int(pose.get("smoothing_frames", 3)),
        min_hand_scale=float(pose.get("min_hand_scale", 0.05)),
        max_hand_scale=float(pose.get("max_hand_scale", 0.60)),
    )
```

### 8.3 `src/spotify_skipper/app.py` — the main loop

```python
"""Main application loop."""
from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path

import cv2

from .camera import Camera
from .config import build_engine, load_config
from .hands import HandTracker

log = logging.getLogger("spotify_skipper")


def setup_logging(cfg):
    a = cfg.section("app")
    Path(a.get("log_file", "logs/skipper.log")).parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.handlers.RotatingFileHandler(
        a.get("log_file", "logs/skipper.log"), maxBytes=1_000_000, backupCount=3)]
    if sys.stderr and sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=getattr(logging, a.get("log_level", "INFO")),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers, force=True)


def build_action(cfg, dry_run: bool):
    if dry_run or not cfg.section("spotify").get("enabled", True):
        def _noop():
            log.info("DRY-RUN: would skip track")
            return True, "dry-run"
        return _noop
    from .actions.spotify import SpotifySkipper
    sp_cfg = cfg.section("spotify")
    skipper = SpotifySkipper(cache_path=sp_cfg.get("cache_path", ".spotify_token_cache"),
                             open_browser=False)
    device_id = sp_cfg.get("device_id") or None

    def _skip():
        ok, msg = skipper.skip(device_id=device_id)
        (log.info if ok else log.warning)("skip -> %s (%s)", ok, msg)
        return ok, msg
    return _skip


def run(config_path="config.toml", dry_run=False, preview=None, test_skip=False):
    cfg = load_config(config_path)
    setup_logging(cfg)
    action = build_action(cfg, dry_run)

    if test_skip:
        print(action())
        return 0

    cam_cfg, hand_cfg, app_cfg = (cfg.section("camera"), cfg.section("hands"),
                                  cfg.section("app"))
    show = app_cfg.get("preview", True) if preview is None else preview
    tracker = HandTracker(
        model_path=hand_cfg.get("model_path", "models/hand_landmarker.task"),
        num_hands=int(hand_cfg.get("num_hands", 1)),
        min_detection_confidence=float(hand_cfg.get("min_detection_confidence", 0.6)),
        min_presence_confidence=float(hand_cfg.get("min_presence_confidence", 0.6)),
        min_tracking_confidence=float(hand_cfg.get("min_tracking_confidence", 0.6)),
    )
    engine = build_engine(cfg)

    log.info("starting: gesture=%s dry_run=%s preview=%s",
             cfg.section("gesture").get("type"), dry_run, show)
    frames, t0, flash_until = 0, time.monotonic(), 0.0

    try:
        with Camera(device=cam_cfg.get("device", 0),
                    width=int(cam_cfg.get("width", 640)),
                    height=int(cam_cfg.get("height", 480)),
                    fps=int(cam_cfg.get("fps", 30)),
                    mirror=bool(cam_cfg.get("mirror", True)),
                    backend=cam_cfg.get("backend", "auto")) as cam:
            while True:
                fr = cam.read()
                if fr is None:
                    log.warning("dropped frame; retrying")
                    time.sleep(0.05)
                    continue
                frames += 1
                h, w = fr.image.shape[:2]
                obs = tracker.process(fr.image, fr.timestamp_ms)

                for ev in engine.process(obs, w, h):
                    log.info("gesture %s travel=%.2fhw dur=%.2fs",
                             ev.name, ev.travel_hand_widths, ev.duration_s)
                    if app_cfg.get("beep", True):
                        print("\a", end="", flush=True)
                    action()
                    flash_until = time.monotonic() + 1.0

                if show:
                    y = 24
                    for k, v in engine.debug.items():
                        cv2.putText(fr.image, f"{k}: {v}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        y += 20
                    fps = frames / max(1e-6, time.monotonic() - t0)
                    cv2.putText(fr.image, f"{fps:.1f} FPS", (w - 120, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if time.monotonic() < flash_until:
                        cv2.rectangle(fr.image, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
                    cv2.imshow("Spotify Skipper", fr.image)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                elif frames % 900 == 0:
                    log.debug("alive: %d frames, %.1f FPS", frames,
                              frames / (time.monotonic() - t0))
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        tracker.close()
        cv2.destroyAllWindows()
        log.info("stopped after %d frames", frames)
    return 0
```

### 8.4 `src/spotify_skipper/__main__.py`

```python
import argparse
import sys

from .app import run


def main() -> int:
    ap = argparse.ArgumentParser(prog="spotify-skipper",
                                 description="Skip Spotify tracks with a hand gesture.")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect gestures but never call the Spotify API")
    ap.add_argument("--no-preview", dest="preview", action="store_false", default=None,
                    help="run headless (no window)")
    ap.add_argument("--test-skip", action="store_true",
                    help="fire one skip immediately and exit")
    a = ap.parse_args()
    return run(a.config, dry_run=a.dry_run, preview=a.preview, test_skip=a.test_skip)


if __name__ == "__main__":
    sys.exit(main())
```

### 8.5 Make the package importable

Simplest and most portable: run from the repo root with `src` on the path.

```bash
cat > pyproject.toml <<'EOF'
[project]
name = "spotify-skipper"
version = "0.1.0"
requires-python = ">=3.10,<3.13"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
EOF
pip install -e .
```

`pip install -e .` makes `python -m spotify_skipper` work from anywhere in the venv,
and it is one command on Windows too. (The `scripts/*.py` snippets in this guide import
via `src.spotify_skipper.…`; after the editable install you can simplify those to
`spotify_skipper.…`.)

### 8.6 Checkpoint M8 and M10

```bash
python -m spotify_skipper --dry-run              # M8: HUD + "DRY-RUN: would skip"
python -m spotify_skipper                        # M10: real skip on gesture
tail -f logs/skipper.log                         # in another terminal
```

M8 passes when: swiping prints `DRY-RUN: would skip` exactly once per swipe, and five
minutes of normal work prints nothing.
M10 passes when playback actually advances and the log shows `skip -> True (skipped)`.

---

## Phase 9 — Tuning for zero false positives

A skipper that fires when you did not mean it is worse than no skipper. Work through
these in order.

### 9.1 The tuning loop

```
record a clip of the failure  →  python -m spotify_skipper.replay --verbose
   →  change ONE value in config.toml  →  replay again  →  keep or revert
```

Never tune against the live camera; you cannot reproduce the exact motion twice.

### 9.2 Symptom → knob table

| Symptom | Knob | Direction |
|---|---|---|
| Won't fire; `reason: travel X < 1.6` | `travel_hand_widths` | ↓ to 1.2–1.4 |
| Won't fire; `reason: window expired` | `max_duration_s` | ↑ to 0.9 |
| Won't fire; `state` never reaches `ARMED` | `arm_frames` ↓ 4, `arm_stillness` ↑ 0.8, or pose thresholds in `poses.py` are too strict — re-run `calibrate_pose.py` |
| Fires when you reach for something | `lateral_ratio` ↓ 0.4, `arm_frames` ↑ 8 | |
| Fires while waving | `confirm_s` ↑ 0.50, `reversal_ratio` ↓ 0.35 | |
| Detects the swipe but never fires (`reason: reversed…`) | you are snapping your hand back — hold the end position, or `confirm_s` ↓ 0.30 | |
| Second skip in a row is ignored | expected: `cooldown_s` (3 s) **plus** the re-arm. Allow ~4 s between skips, or drop your hand between them | |
| Fires twice per swipe | `cooldown_s` ↑, `release_frames` ↑ | |
| Fires when someone walks past | `min_hand_scale` ↑, `num_hands = 1` | |
| Random single-frame fires | `smoothing_frames` ↑ 5, `min_duration_s` ↑ 0.15 | |
| Works close, not far | you broke scale invariance — check `to_iso`/`hand_scale`; distances must be divided by `scale` |
| Works with right hand only | handedness mirroring in `canonicalize` is not applied — check the `Left` branch |

### 9.3 Raise the bar deliberately

If you want the gesture to be *essentially impossible* to trigger by accident, layer a
second condition rather than tightening one to an extreme:

- **Two-stage gesture**: `HoldDetector` for 0.4 s *then* a swipe. Chain them by feeding
  the swipe detector only while the hold detector is satisfied.
- **Region gating**: require the palm centre to be in the upper half of the frame
  (`center[1] < 0.5`) — nobody swipes at face height by accident.
- **Rare pose**: use a pose you never make while working (e.g. index+pinky "rock horns")
  as `trigger_pose`. Rule it in `poses.py` with the calibration procedure from 4.3.

### 9.4 Checkpoint M9

```bash
python -m spotify_skipper --dry-run --no-preview &
# work normally for 10 minutes, then:
grep -c "DRY-RUN" logs/skipper.log        # must be exactly the number of deliberate swipes
```

Then re-run the full offline suite and record the numbers in `README.md` so a future
change can be compared against them:

```bash
python -m spotify_skipper.replay --verbose | tee docs/baseline.txt
```

---

## Phase 10 — Windows deployment

### 10.1 The one thing you cannot do

**PyInstaller cannot cross-compile.** A Linux build produces a Linux ELF binary; there
is no flag that emits a Windows `.exe`. Your realistic options:

| Option | Effort | Recommendation |
|---|---|---|
| **A. Ship the source tree + bootstrap script** | low | ✅ **Do this.** Identical code both sides, easy to update, easy to debug. |
| B. Run PyInstaller *on* the Windows PC | medium | Only if you want a double-clickable `.exe`. See [Appendix D](#appendix-d--optional-pyinstaller-exe-on-windows). |
| C. Wine / cross-build tricks | high | MediaPipe's native DLLs make this fragile. Not worth it. |

Everything below is Option A.

### 10.2 Make the code Windows-clean before you copy

Audit for these — the guide's code already complies:

- [ ] No hard-coded `/` paths. Use `pathlib.Path`, never string concatenation.
- [ ] No `/dev/video0` literals. The camera index comes from config.
- [ ] No `os.fork`, no `signal.SIGKILL`, no `select()` on non-sockets.
- [ ] Camera backend chosen by `platform.system()` (Phase 2.2).
- [ ] `os.chmod(0o600)` guarded by `if os.name == "posix"` (Phase 7.3).
- [ ] Text files opened with an explicit `encoding="utf-8"` (Windows defaults to cp1252).
- [ ] Line endings: add `.gitattributes` with `* text=auto` if using git.

### 10.3 Transfer the project

```bash
# on Linux — build a transfer archive INCLUDING the model, EXCLUDING secrets
cd ~/projects
tar --exclude='.venv' --exclude='__pycache__' --exclude='.env' \
    --exclude='.spotify_token_cache' --exclude='logs/*' \
    -czf spotify-skipper.tar.gz Spotify-Skipper
```

Copy `spotify-skipper.tar.gz` to Windows (USB, network share, or `git push` + `git
clone` — but keep `.env` and the token cache out of the repo).

**Do include** `models/hand_landmarker.task`, `config.toml`, and your tuned
`models/pose_clf.joblib` if you trained one. **Do not include** `.env` or
`.spotify_token_cache`; you will re-create them on Windows (7.2 and 7.4). Re-authorising
on the second machine is normal and takes 30 seconds.

### 10.4 Windows prerequisites

1. Install **Python 3.12 (64-bit)** from python.org. On the first installer screen tick
   **“Add python.exe to PATH”**. Do *not* use the Microsoft Store build — its sandboxed
   file locations complicate camera access and PyInstaller.
2. Grant camera permission: **Settings → Privacy & security → Camera** → *Camera access*
   **On** and *Let desktop apps access your camera* **On**. Without this,
   `cv2.VideoCapture` opens and returns black frames or fails silently — the single most
   common Windows failure.
3. If `import cv2` complains about missing DLLs, install the **Microsoft Visual C++
   2015–2022 Redistributable (x64)**.

### 10.5 `scripts/bootstrap.ps1`

```powershell
# Run from the project root:  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== Python ==" -ForegroundColor Cyan
py -3.12 --version

Write-Host "== venv ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

Write-Host "== model ==" -ForegroundColor Cyan
if (-not (Test-Path "models\hand_landmarker.task")) {
  New-Item -ItemType Directory -Force -Path models | Out-Null
  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
                    -OutFile "models\hand_landmarker.task"
}

Write-Host "== .env ==" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Edit .env and put your SPOTIFY_CLIENT_ID in it, then re-run." -ForegroundColor Yellow
  exit 1
}

Write-Host "== camera check ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -c "import cv2; c=cv2.VideoCapture(0, cv2.CAP_DSHOW); ok,_=c.read(); print('camera ok' if ok else 'CAMERA FAILED'); c.release()"

Write-Host "== authorise Spotify ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m spotify_skipper.auth

Write-Host "Bootstrap complete. Run: .\.venv\Scripts\python.exe -m spotify_skipper --dry-run" -ForegroundColor Green
```

And the Linux twin, `scripts/bootstrap.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt -e .
[ -f models/hand_landmarker.task ] || curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
[ -f .env ] || { cp .env.example .env; echo "Fill in .env, then re-run"; exit 1; }
./.venv/bin/python -m spotify_skipper.auth
echo "Run: ./.venv/bin/python -m spotify_skipper --dry-run"
```

### 10.6 Run on Windows

```powershell
cd C:\Users\<you>\Spotify-Skipper
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
.\.venv\Scripts\python.exe -m spotify_skipper --dry-run
.\.venv\Scripts\python.exe -m spotify_skipper
```

### 10.7 Expected Linux → Windows differences

| Difference | Effect | Action |
|---|---|---|
| Different webcam, different FOV | your hand occupies a different fraction of frame | re-check `travel_hand_widths`; it is in hand-widths so it *should* transfer, but verify |
| Lower/higher FPS | frame-count thresholds shift in time | if Windows runs at 15 FPS, halve `arm_frames`, `pose_grace_frames`, `release_frames` |
| DirectShow ignores `CAP_PROP_BUFFERSIZE` | ~1 extra frame of latency | harmless |
| Camera index 0 is a virtual cam (OBS, Windows Hello IR) | wrong or black video | try `device = 1`, `2` in `config.toml` |
| Windows Defender scans the venv on first run | first launch is slow | expected; subsequent runs are fast |

### 10.8 Checkpoint M11

On Windows: `--dry-run` shows the HUD at ≥ 15 FPS, your swipe flashes the frame, and
`logs\skipper.log` records the event. Then a real run actually skips a track.

---

## Phase 11 — Run at login / background operation

Set `preview = false` in `config.toml` first, otherwise you get a permanent video
window.

### 11.1 Windows — Task Scheduler (recommended)

`pythonw.exe` runs without a console window.

```powershell
$root = "C:\Users\<you>\Spotify-Skipper"
$action  = New-ScheduledTaskAction -Execute "$root\.venv\Scripts\pythonw.exe" `
           -Argument "-m spotify_skipper --no-preview" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
            -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "SpotifySkipper" -Action $action -Trigger $trigger `
                       -Settings $settings -Description "Hand-gesture Spotify skipper"
```

Manage it:

```powershell
Start-ScheduledTask  -TaskName SpotifySkipper
Stop-ScheduledTask   -TaskName SpotifySkipper
Unregister-ScheduledTask -TaskName SpotifySkipper -Confirm:$false
Get-Content .\logs\skipper.log -Tail 20 -Wait
```

**Simpler alternative:** put a shortcut to `pythonw.exe -m spotify_skipper
--no-preview` (with *Start in* set to the project folder) into
`shell:startup` (Win+R → `shell:startup`).

Because there is no window, keep the webcam LED and the log as your signs of life, and
remember the process is killed via Task Manager (`pythonw.exe`) or `Stop-ScheduledTask`.

### 11.2 Linux — systemd user service

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/spotify-skipper.service <<'EOF'
[Unit]
Description=Spotify gesture skipper
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=%h/projects/Spotify-Skipper
ExecStart=%h/projects/Spotify-Skipper/.venv/bin/python -m spotify_skipper --no-preview
Restart=on-failure
RestartSec=5
# tighten the blast radius: the process needs the camera and the network, nothing else
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now spotify-skipper
systemctl --user status spotify-skipper
journalctl --user -u spotify-skipper -f
```

If you want it to survive logout: `loginctl enable-linger $USER`.

### 11.3 Optional: single-instance guard

Two copies fighting over the camera is a confusing failure. Add to `app.run()`:

```python
import atexit, os
from pathlib import Path

def acquire_single_instance(lock_path="logs/.lock"):
    p = Path(lock_path)
    if p.exists():
        try:
            pid = int(p.read_text())
            os.kill(pid, 0)                       # raises if the pid is gone
            raise SystemExit(f"already running as pid {pid}")
        except (ValueError, OSError):
            pass                                  # stale lock
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()))
    atexit.register(lambda: p.unlink(missing_ok=True))
```

On Windows `os.kill(pid, 0)` also works for existence checks, so this is portable.

### 11.4 Checkpoint M12

Reboot. Within ~30 s of logging in, the webcam LED lights, `logs/skipper.log` shows
`starting: ...`, and a swipe skips a track. No console window appears.

---

## Phase 12 — Verification and privacy audit

### 12.1 Prove exactly one request leaves per skip

**Linux:**

```bash
sudo tcpdump -n -i any 'tcp port 443 and (host api.spotify.com or host accounts.spotify.com)' -c 50
```

Run the app, perform one gesture, and watch. You will see one short TLS conversation
with `api.spotify.com` (a handful of packets — TLS handshake + one POST + the 204),
plus an occasional `accounts.spotify.com` exchange when the hourly token refresh fires.
Nothing during idle detection, no matter how much you wave.

**Windows:** install Wireshark, capture on your active adapter with the display filter
`tls.handshake.extensions_server_name contains "spotify"`.

**Either OS, no packet capture needed:** point `[spotify] enabled = false` (or
`--dry-run`) and confirm zero network activity in `nethogs` / Task Manager's Network
column while gesturing repeatedly.

### 12.2 Prove no video leaves and none is stored

```bash
grep -rn "imwrite\|VideoWriter\|requests.post\|urlopen\|http" src/ | grep -v spotify
```

The only hits should be inside `actions/spotify.py`. The recorder writes `.npz`
landmark arrays only — verify:

```bash
python - <<'EOF'
import numpy as np
d = np.load(sorted(__import__("pathlib").Path("data/clips").glob("*.npz"))[0],
            allow_pickle=True)
print(list(d.keys()))         # timestamps, landmarks, handedness, label, frame_w, frame_h
print(d["landmarks"].shape)   # (T, 21, 3) — coordinates, not pixels
EOF
```

### 12.3 Secrets hygiene

```bash
git status --porcelain          # .env and .spotify_token_cache must NOT appear
ls -l .spotify_token_cache      # -rw------- on Linux
```

If the token cache ever leaks, revoke access at
<https://www.spotify.com/account/apps/> and delete the file.

### 12.4 Final acceptance test

| # | Test | Expected |
|---|---|---|
| 1 | Perform the gesture with music playing | track advances within ~1 s |
| 2 | Perform it twice in 1 s | exactly one skip (cooldown) |
| 3 | Wave, reach, type for 10 min | zero skips |
| 4 | Pause Spotify entirely, then gesture | log warns `no active device`, app keeps running |
| 5 | Unplug the network, then gesture | log warns `network error`, app keeps running |
| 6 | Cover the camera, then uncover | recovers, no crash |
| 7 | Leave running 8 hours | still responsive; log rotated, not unbounded |

---

## Appendix A — Landmark index reference

MediaPipe returns 21 landmarks per hand, in this fixed order:

```
                    8   12  16  20      <- TIP
                    |   |   |   |
                    7   11  15  19      <- DIP
                    |   |   |   |
                4   6   10  14  18      <- PIP
                |   |   |   |   |
                3   5---9---13--17      <- MCP  (5,9,13,17 form the knuckle line)
                |  /   /   /   /
                2 /   /   /   /
                |/___/___/___/
                1              (thumb CMC)
                 \
                  0            WRIST
```

| # | Name | # | Name |
|---|---|---|---|
| 0 | WRIST | 11 | MIDDLE_DIP |
| 1 | THUMB_CMC | 12 | MIDDLE_TIP |
| 2 | THUMB_MCP | 13 | RING_MCP |
| 3 | THUMB_IP | 14 | RING_PIP |
| 4 | THUMB_TIP | 15 | RING_DIP |
| 5 | INDEX_MCP | 16 | RING_TIP |
| 6 | INDEX_PIP | 17 | PINKY_MCP |
| 7 | INDEX_DIP | 18 | PINKY_PIP |
| 8 | INDEX_TIP | 19 | PINKY_DIP |
| 9 | MIDDLE_MCP | 20 | PINKY_TIP |
| 10 | MIDDLE_PIP | | |

Useful invariants: `0→9` is the hand's "up" axis and its scale reference; `5,9,13,17`
define the palm plane; fingertip curl is best measured as chain-length vs. direct
distance (Phase 4.2), never as raw y-coordinates.

---

## Appendix B — Troubleshooting

### Camera

| Symptom | Cause | Fix |
|---|---|---|
| `Could not open camera 0` (Linux) | device busy or wrong index | `v4l2-ctl --list-devices`; try `device = 2`; close Zoom/Chrome |
| Black frames on Windows | app camera permission off | Settings → Privacy & security → Camera → allow desktop apps |
| Opens but 5–10 FPS | webcam is delivering raw YUYV | keep the MJPG `CAP_PROP_FOURCC` line from Phase 2.2 |
| 3–5 s freeze on start (Windows) | MSMF backend probing | force `backend = "dshow"` |
| Image is not mirrored | `mirror = false` | set it true; do not compensate downstream |
| `/dev/video1` is the real camera | multi-endpoint UVC device | many webcams expose a metadata node at `video1`; use whichever index returns frames |

### MediaPipe

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: hand_landmarker.task` | model not downloaded | Phase 3.1 |
| `Packet timestamp mismatch` / `must be monotonically increasing` | non-monotonic timestamps | use the `Camera.timestamp_ms` monotonic counter, never `time.time()` |
| `A module that was compiled using NumPy 1.x...` | NumPy 2 installed | `pip install "numpy<2"` |
| No wheel found for mediapipe | Python 3.13 | use Python 3.12 |
| High CPU | too many hands / big frames | `num_hands = 1`, 640×480 |
| Landmarks jitter badly | low light | improve lighting before touching thresholds — the model is light-hungry |

### Gesture engine

| Symptom | Where to look |
|---|---|
| Never arms | `engine.debug["pose"]` — if always `NONE`, the pose rules are wrong for your hand; re-run `calibrate_pose.py` |
| Arms then immediately drops | `arm_stillness` too low, or `smoothing_frames` too high for your FPS |
| Fires on hand entry | increase `arm_frames`; entry motion is being read as travel |
| Behaviour differs between machines | FPS differs — everything measured in *frames* scales with FPS; prefer seconds-based thresholds if you see this a lot |

### Spotify

| HTTP | Meaning | Fix |
|---|---|---|
| 401 | token invalid/expired past refresh | delete `.spotify_token_cache`, re-run `python -m spotify_skipper.auth` |
| 403 | not Premium, or restricted context | Premium is required for playback control |
| 404 | `NO_ACTIVE_DEVICE` | press play once on any device, or set `device_id` in config |
| 429 | rate limited | already handled with backoff; you should never hit this with a 3 s cooldown |
| `INVALID_CLIENT: Invalid redirect URI` | dashboard URI ≠ code URI | must be exactly `http://127.0.0.1:8888/callback` in both places |
| Browser opens on a headless box | `open_browser=True` | set `open_browser=False` and paste the redirect URL manually |

---

## Appendix C — Dependency-free Spotify calls (no spotipy)

If you want to *see* every byte, replace `actions/spotify.py` with raw PKCE. This is
the complete flow; `requests` is the only dependency.

```python
import base64, hashlib, http.server, json, os, secrets, threading, urllib.parse, webbrowser
import requests

AUTH = "https://accounts.spotify.com/authorize"
TOKEN = "https://accounts.spotify.com/api/token"
REDIRECT = "http://127.0.0.1:8888/callback"
SCOPE = "user-modify-playback-state"


def _pkce_pair():
    verifier = base64.urlsafe_b64encode(os.urandom(64)).decode().rstrip("=")[:128]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def login(client_id, cache="token.json"):
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    q = urllib.parse.urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": REDIRECT,
        "scope": SCOPE, "code_challenge_method": "S256", "code_challenge": challenge,
        "state": state})
    box = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            box.update({k: v[0] for k, v in p.items()})
            self.send_response(200); self.end_headers()
            self.wfile.write(b"<h2>You can close this tab.</h2>")
        def log_message(self, *a): pass

    srv = http.server.HTTPServer(("127.0.0.1", 8888), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    webbrowser.open(f"{AUTH}?{q}")
    while "code" not in box and "error" not in box:
        pass
    assert box.get("state") == state, "CSRF: state mismatch"
    r = requests.post(TOKEN, timeout=10, data={
        "grant_type": "authorization_code", "code": box["code"],
        "redirect_uri": REDIRECT, "client_id": client_id, "code_verifier": verifier})
    r.raise_for_status()
    tok = r.json()
    open(cache, "w", encoding="utf-8").write(json.dumps(tok))
    return tok


def refresh(client_id, tok, cache="token.json"):
    r = requests.post(TOKEN, timeout=10, data={
        "grant_type": "refresh_token", "refresh_token": tok["refresh_token"],
        "client_id": client_id})
    r.raise_for_status()
    new = r.json()
    tok.update(new)                      # Spotify may or may not rotate the refresh token
    open(cache, "w", encoding="utf-8").write(json.dumps(tok))
    return tok


def skip(access_token):
    """THE call. One POST, 204 on success."""
    r = requests.post("https://api.spotify.com/v1/me/player/next", timeout=6,
                      headers={"Authorization": f"Bearer {access_token}"})
    return r.status_code, (r.text or "")[:200]
```

You must track `expires_in` yourself and call `refresh()` before it lapses (or on a
401). That bookkeeping is exactly what spotipy's `SpotifyPKCE` cache handler does for
you — which is why the main guide uses it.

---

## Appendix D — Optional: PyInstaller .exe on Windows

Only on the Windows machine, in the activated venv:

```powershell
pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed `
  --name SpotifySkipper `
  --collect-all mediapipe `
  --collect-all cv2 `
  --add-data "models\hand_landmarker.task;models" `
  --add-data "config.toml;." `
  src\spotify_skipper\__main__.py
```

Notes that will save you an evening:

- `--collect-all mediapipe` is mandatory: MediaPipe ships `.binarypb`/`.tflite` data
  files that PyInstaller's dependency analysis cannot see.
- `--onedir`, not `--onefile`: one-file unpacks ~200 MB to a temp dir on every launch
  (slow) and trips more antivirus heuristics.
- Bundled data lives under `sys._MEIPASS` at runtime. Resolve paths with:

  ```python
  import sys
  from pathlib import Path
  BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
  MODEL = BASE / "models" / "hand_landmarker.task"
  ```

- `.env` and `.spotify_token_cache` should stay **outside** the bundle, next to the
  `.exe`, so credentials are not baked into a distributable.
- Windows Defender may quarantine a fresh unsigned PyInstaller build. Add a folder
  exclusion, or stick with Option A from Phase 10.1.

---

## Appendix E — Optional: system-tray control

Background operation with no UI is unnerving. `pystray` gives you an icon with
Pause / Resume / Quit:

```python
# pip install pystray pillow
import threading

import pystray
from PIL import Image, ImageDraw


def make_icon(on_toggle, on_quit):
    img = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(img).ellipse((12, 12, 52, 52), fill=(30, 215, 96))   # Spotify green
    return pystray.Icon("skipper", img, "Spotify Skipper", menu=pystray.Menu(
        pystray.MenuItem("Pause / Resume", lambda i, it: on_toggle()),
        pystray.MenuItem("Quit", lambda i, it: (on_quit(), i.stop())),
    ))


# in app.run(): a threading.Event gates the loop
paused = threading.Event()
stop = threading.Event()
threading.Thread(target=lambda: make_icon(
    lambda: (paused.clear() if paused.is_set() else paused.set()),
    stop.set).run, daemon=True).start()
# ...then at the top of the loop body:
#   if paused.is_set(): time.sleep(0.1); continue
#   if stop.is_set():  break
```

Works on Windows and on most Linux desktops (needs an AppIndicator-capable tray on
GNOME).

---

## Appendix F — Build order for an autonomous agent

If handing this to another Claude Code session, this is the executable summary:

```
1.  mkdir tree (1.1) ; .gitignore (1.2) ; requirements.txt (1.3) ; venv (1.4)
2.  write src/spotify_skipper/camera.py   (2.2)   -> verify 2.3
3.  download models/hand_landmarker.task  (3.1)
4.  write src/spotify_skipper/hands.py    (3.2)   -> verify 3.3
5.  write src/spotify_skipper/features.py (4.2)
6.  write gestures/poses.py               (4.3)
7.  write gestures/motion.py              (4.4)
8.  write gestures/engine.py              (4.5)
9.  write config.toml (8.1) + config.py (8.2)     [needed by replay.py]
10. write recorder.py (5.1) ; STOP -> the human must record clips (5.2)
11. write replay.py (6.1) + tests (6.3)  -> iterate thresholds until 6.4 passes
12. (optional) classifier.py (4.6) + train.py (6.2)
13. write actions/spotify.py (7.3) + auth.py (7.4) ; STOP -> the human must
    create the Spotify app (7.1), fill .env (7.2), and run the browser login
14. write app.py (8.3) + __main__.py (8.4) + pyproject.toml (8.5)
15. verify --dry-run, then live (8.6)
16. write scripts/bootstrap.sh + bootstrap.ps1 (10.5)
17. package and deploy (10.3–10.6) ; autostart (11)
```

Three steps **require a human** and must not be faked: recording gesture clips,
creating the Spotify dashboard app, and the one-time browser OAuth consent. Everything
else is scriptable.

**Invariants an agent must not "optimise away":**

1. Timestamps into MediaPipe are monotonic (Phase 2.2/3.2).
2. The frame is mirrored exactly once, at capture (Phase 2.1).
3. All spatial thresholds are expressed in **hand-widths**, never pixels (Phase 4.2).
4. Pose classification never triggers an action directly — L3 and L4 always sit between
   (Phase 4.1).
5. Exactly one Spotify request per accepted gesture; failures are logged, never retried
   in a loop (Phase 7.3).
6. No frames, images, or landmarks are ever transmitted (Phase 12).

---

## References

- MediaPipe Hand Landmarker (Python): <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python>
- Model card / download: <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models>
- Spotify Web API — Skip to Next: <https://developer.spotify.com/documentation/web-api/reference/skip-users-playback-to-next-track>
- Spotify Authorization Code with PKCE: <https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow>
- Spotify scopes: <https://developer.spotify.com/documentation/web-api/concepts/scopes>
- spotipy docs: <https://spotipy.readthedocs.io/>
- OpenCV VideoCapture backends: <https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html>
