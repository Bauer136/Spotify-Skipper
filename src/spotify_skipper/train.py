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
#
# CAUTION — this map is single-label-per-clip, which fitted the swipe (one pose held
# throughout) and does NOT fit a transition (two poses, plus the change between them).
# A `skip_transition` clip contains ZERO frames, OK_THREE frames, and ambiguous frames
# in between; labelling all of them one way poisons the classifier.
#
# Two workable options:
#   1. Record the two stages as SEPARATE clips for training purposes
#      (`--label pose_zero`, `--label pose_ok_three`) and map those, leaving the
#      full-gesture `skip_transition` clips for replay scoring only. Simplest — do this.
#   2. Extend the recorder to write a per-frame label array and read it here.
#
# Option 1 is what the map below assumes. Decoys/idle are NONE.
LABEL_MAP = {"pose_zero": "ZERO", "pose_ok_three": "OK_THREE"}
DEFAULT_LABEL = "NONE"

# Some clips must be EXCLUDED, not defaulted, because they are decoys at the GESTURE
# level but not at the POSE level:
#   skip_transition — contains real ZERO frames, real OK_THREE frames, and ambiguous
#                     ones between. No single frame label is correct.
#   decoy_hold_o    — "form the O and just hold it" is a perfect ZERO to L2. Labelling
#                     it NONE tells the classifier the same hand shape is both ZERO and
#                     not-ZERO. Measured: including it costs 3.4 points of CV accuracy
#                     (0.922 -> 0.888), and ZERO-vs-hold_o separability is only 0.846.
# Rejecting a held O is L3's job, not L2's — TransitionDetector already does it
# (arming is not firing). Both labels are for replay scoring (6.1) only.
EXCLUDE_FROM_TRAINING = {"skip_transition", "decoy_hold_o"}


def load_dataset(clips_dir: Path):
    X, y, groups = [], [], []
    for gi, p in enumerate(sorted(clips_dir.glob("*.npz"))):
        d = np.load(p, allow_pickle=True)
        lm, hd = d["landmarks"], d["handedness"]
        w, h = int(d["frame_w"]), int(d["frame_h"])
        raw_label = str(d["label"])
        if raw_label in EXCLUDE_FROM_TRAINING:
            continue
        label = LABEL_MAP.get(raw_label, DEFAULT_LABEL)
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
