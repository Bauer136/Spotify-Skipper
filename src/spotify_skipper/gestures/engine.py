"""Glues L1–L4 together and exposes one method: process(frame) -> events."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .. import features as F
from .motion import (GestureEvent, MotionConfig, MotionDetector, Sample, State,
                     TransitionConfig, TransitionDetector)


class GestureEngine:
    def __init__(self, pose_fn, detector, smoothing_frames: int = 3,
                 min_hand_scale: float = 0.05, max_hand_scale: float = 0.60):
        """
        pose_fn : callable(canonical_landmarks) -> PoseResult
                  (rule-based `classify_pose` or the ML head from 4.6)
        detector: MotionDetector | HoldDetector | TransitionDetector
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
