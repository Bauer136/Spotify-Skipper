# Spotify Skipper

Skip Spotify tracks with hand gestures, using a webcam. No cloud vision calls —
MediaPipe runs locally on CPU, and only playback commands leave the machine.

Runs on Linux and Windows from the same source tree.

## Status

Build is in progress, following [`BUILD_GUIDE.md`](BUILD_GUIDE.md). Phases 1–3 are
done: environment, camera capture, and hand tracking with a debug viewer. The
gesture engine (Phase 4 onward) is not written yet.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Linux also needs OpenCV's runtime libs:

```bash
sudo apt-get install -y python3.12-venv libgl1 libglib2.0-0 v4l-utils
```

Then download the hand-landmark model — it is gitignored (7.5 MB) and the tracker
will not start without it:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run the debug viewer

```bash
python scripts/preview_hands.py
```

Opens a window showing the mirrored camera feed with the 21-point hand skeleton,
the handedness label, and a frame-rate counter. Press `q` to quit. Needs a desktop
session — `cv2.imshow` will not work over plain SSH.

Working output looks like: skeleton locked to your hand, `Right 0.9x` top-left when
you raise your right hand, and an FPS readout that is green at 20+.

If the label reads `Left` for your right hand, the frame is not being mirrored —
fix `Camera` in `src/spotify_skipper/camera.py`, not the viewer. Everything
downstream assumes a mirrored (selfie-view) frame.

## Known issues

- **Frame rate is ~16 FPS, not the 20–30 the gesture engine assumes.** Capture and
  inference run serially, so their costs add (~33 ms + ~27 ms). Moving capture to a
  background thread that keeps only the newest frame measured 40 FPS on an i5-8350U.
- **Some webcams throttle themselves in dim light**, halving the frame rate while
  still reporting 30 FPS. On Linux:
  `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`. This resets on replug
  or reboot, so it belongs in `Camera.open()` or a udev rule.

Both are diagnosed in more detail in `BUILD_GUIDE.md` § 3.4.

## Layout

```
src/spotify_skipper/   camera capture, hand tracking, gesture engine, Spotify actions
scripts/               developer tools (debug viewers, recording, training)
models/                downloaded model assets — gitignored
data/                  recorded landmark clips — gitignored, no images stored
logs/                  runtime logs — gitignored
BUILD_GUIDE.md         full step-by-step build, phase by phase
```

## Privacy

Video frames are never written to disk or transmitted. The recorder (Phase 5)
stores landmark coordinates only. Your Spotify token is cached locally in
`.spotify_token_cache`, which is gitignored — do not commit it.
