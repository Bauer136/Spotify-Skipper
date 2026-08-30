"""Rule-based static pose recognition on canonical landmarks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import (FINGER_ORDER, INDEX_TIP, MIDDLE_TIP,
                        extension_ratios, palm_openness, spreads)


@dataclass(frozen=True)
class PoseResult:
    name: str          # "OPEN_PALM" | "FIST" | "PEACE" | "ZERO" | "OK_THREE" | "NONE"
    confidence: float  # 0..1, a soft margin — used for hysteresis, not thresholds


# Tunables (mirror these into config.toml [poses]).
EXT_OPEN = 0.90      # a finger this straight counts as extended
EXT_CURL = 0.72      # a finger this bent counts as curled
OPENNESS_OPEN = 1.55 # mean fingertip distance from wrist, canonical units
SPREAD_MIN = 0.25    # min mean gap between adjacent fingertips for OPEN_PALM
PEACE_V_GAP = 0.35   # min index-tip to middle-tip distance for a convincing V

# Skip-gesture poses: ZERO (the "O") -> OK_THREE. Deliberately separate constants
# from EXT_OPEN/EXT_CURL above: these two rules get retuned often, and sharing the
# thresholds would move OPEN_PALM and FIST every time the skip gesture is adjusted.
CIRCLE_MAX = 0.30    # max thumb-tip to index-tip distance for the loop to count as closed
ZERO_INDEX_MIN = 0.72  # index is bent into the loop, NOT fully curled — separates O from FIST
EXT_CURL_3 = 0.72    # middle/ring/pinky this bent count as curled (stage A)
EXT_OPEN_3 = 0.90    # ...and this straight count as extended (stage B)
SPREAD_MIN_3 = 0.22  # min mean middle-ring / ring-pinky tip gap once they come up

# Typical measured values (canonical units, where |wrist -> middle MCP| == 1).
# Use these as sanity bounds, then replace them with YOUR numbers from 4.3's
# calibration procedure:
#   open palm : four_ext 0.93-1.00 | openness 1.70-2.10 | spread 0.28-0.45
#   fist      : four_ext 0.55-0.70 | openness 0.80-1.05 | spread 0.10-0.25
#   relaxed   : four_ext 0.75-0.90 | openness 1.30-1.60  <- the dangerous middle ground
# ZERO / OK_THREE thresholds above are starting points only — no measured ranges
# exist for them yet. Calibrate before trusting them (scripts/calibrate_pose.py).


def _states(canon: np.ndarray) -> dict[str, float]:
    return dict(zip(FINGER_ORDER, extension_ratios(canon)))


def _margin(value: float, lo: float, hi: float) -> float:
    """Soft 0..1 score: 0 at/below lo, 1 at/above hi."""
    if hi <= lo:
        return float(value >= hi)
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _circle_gap(canon: np.ndarray) -> float:
    """Thumb-tip to index-tip distance — spreads()[0]. Small when the O is closed."""
    return float(spreads(canon)[0])


def _three_spread(canon: np.ndarray) -> float:
    """Mean middle-ring and ring-pinky tip gap — spreads()[2:4]. How fanned the
    three fingers are once they come up."""
    return float(np.mean(spreads(canon)[2:4]))


def classify_pose(canon: np.ndarray) -> PoseResult:
    e = _states(canon)
    openness = palm_openness(canon)
    sp = float(np.mean(spreads(canon)))

    four_ext = min(e["index"], e["middle"], e["ring"], e["pinky"])
    four_curl = max(e["index"], e["middle"], e["ring"], e["pinky"])

    gap = _circle_gap(canon)
    three = (e["middle"], e["ring"], e["pinky"])

    # ZERO (skip gesture, stage A): thumb-index loop closed, other three curled.
    # The index guard is what separates this from a FIST — in a fist the index is
    # fully curled into the palm; here it is only bent around the loop.
    if (gap <= CIRCLE_MAX and e["index"] >= ZERO_INDEX_MIN
            and max(three) <= EXT_CURL_3):
        return PoseResult("ZERO", min(_margin(CIRCLE_MAX - gap, 0.0, 0.12),
                                      _margin(EXT_CURL_3 - max(three), 0.0, 0.10)))

    # OK_THREE (skip gesture, stage B): loop still closed, three fingers up and fanned
    if (gap <= CIRCLE_MAX and min(three) >= EXT_OPEN_3
            and _three_spread(canon) >= SPREAD_MIN_3):
        return PoseResult("OK_THREE", min(_margin(CIRCLE_MAX - gap, 0.0, 0.12),
                                          _margin(min(three), EXT_OPEN_3 - 0.08,
                                                  EXT_OPEN_3 + 0.06)))

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
