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
