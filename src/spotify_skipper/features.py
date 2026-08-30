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
