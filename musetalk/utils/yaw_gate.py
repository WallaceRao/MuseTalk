"""Gate lipsync during large head yaw / fast turns (fade back to original)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def yaw_proxy_from_face_landmarks(face_land_mark) -> Optional[float]:
    """Estimate yaw in roughly ``[-1, 1]`` from DWPose/68-pt face landmarks.

    Uses nose–eye distance asymmetry: ``(d_left - d_right) / (d_left + d_right)``.
    Near 0 ≈ frontal; magnitude grows toward profile. Returns ``None`` when
    landmarks are unusable.
    """
    lm = np.asarray(face_land_mark, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[0] < 48 or lm.shape[1] < 2:
        return None
    if not np.isfinite(lm).all() or np.allclose(lm, 0):
        return None
    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    nose = lm[30]
    d_left = float(np.linalg.norm(nose - left_eye))
    d_right = float(np.linalg.norm(nose - right_eye))
    denom = d_left + d_right
    if denom < 1e-3:
        return None
    # Also reject near-degenerate eye spans (bad pose / partial face).
    eye_w = float(np.linalg.norm(right_eye - left_eye))
    if eye_w < 1.0:
        return None
    return float((d_left - d_right) / denom)


def interpolate_sparse_yaw(
    sparse_yaw: Sequence[Optional[float]],
    n_frames: int,
) -> List[Optional[float]]:
    """Linearly interpolate yaw between keyframes; keep None outside spans."""
    if n_frames <= 0:
        return []
    out: List[Optional[float]] = [None] * n_frames
    keys = [i for i, v in enumerate(sparse_yaw) if v is not None and math.isfinite(float(v))]
    if not keys:
        return out
    for i, k in enumerate(keys):
        out[k] = float(sparse_yaw[k])  # type: ignore[arg-type]
        if i + 1 >= len(keys):
            continue
        k1 = keys[i + 1]
        y0 = float(sparse_yaw[k])  # type: ignore[arg-type]
        y1 = float(sparse_yaw[k1])  # type: ignore[arg-type]
        span = k1 - k
        if span <= 1:
            continue
        for j in range(k + 1, k1):
            t = (j - k) / float(span)
            out[j] = y0 + (y1 - y0) * t
    return out


def yaw_blend_weight(
    yaw: Optional[float],
    *,
    soft_start: float = 0.28,
    hard_max: float = 0.40,
) -> float:
    """Opacity multiplier that fades lipsync as |yaw| approaches profile.

    ``1`` while frontal (``|yaw| <= soft_start``), linearly down to ``0`` at
    ``hard_max``. Used at composite time so near-threshold turns crossfade
    even when the hard gate still keeps the frame.
    """
    if yaw is None:
        return 1.0
    try:
        y = abs(float(yaw))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(y):
        return 1.0
    soft = max(0.0, float(soft_start))
    hard = max(soft, float(hard_max))
    if y <= soft:
        return 1.0
    if y >= hard or hard <= soft:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (y - soft) / (hard - soft)))


def apply_yaw_turn_gate(
    speaking_mask: Sequence[bool],
    yaw_values: Sequence[Optional[float]],
    *,
    fps: float,
    abs_yaw_max: float = 0.40,
    turn_rate_max: float = 2.5,
    pad_frames: int = 0,
) -> Tuple[List[bool], dict]:
    """Clear speaking frames with high |yaw| or fast yaw change.

    Only core high-yaw / fast-turn frames are cleared (no neighbor pad).
    Pad used to hard-clear lead-in frames and left original talking mouths
    visible; edge soft-fade is left to ``blend_ramp`` + ``yaw_blend_weight``.
    ``pad_frames`` is kept for API compatibility but ignored for hard clear.
    """
    n = len(speaking_mask)
    out = list(speaking_mask)
    meta = {
        "cleared_frames": 0,
        "high_yaw_frames": 0,
        "fast_turn_frames": 0,
        "pad_frames": 0,
        "abs_yaw_max": float(abs_yaw_max),
        "turn_rate_max": float(turn_rate_max),
    }
    if n == 0:
        return out, meta
    if len(yaw_values) != n:
        raise ValueError("yaw_values length must match speaking_mask")

    fps = float(fps) if fps and fps > 0 else 25.0
    abs_max = max(0.0, float(abs_yaw_max))
    rate_max = max(0.0, float(turn_rate_max))
    drop = [False] * n
    _ = pad_frames  # unused: hard pad disabled

    for i in range(n):
        y = yaw_values[i]
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(yf):
            continue
        if abs_max > 0 and abs(yf) >= abs_max:
            drop[i] = True
            meta["high_yaw_frames"] += 1
        if i > 0 and rate_max > 0:
            y0 = yaw_values[i - 1]
            if y0 is None:
                continue
            try:
                y0f = float(y0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(y0f):
                continue
            rate = abs(yf - y0f) * fps
            if rate >= rate_max:
                drop[i] = True
                drop[i - 1] = True
                meta["fast_turn_frames"] += 1

    cleared = 0
    for i in range(n):
        if drop[i] and out[i]:
            out[i] = False
            cleared += 1
    meta["cleared_frames"] = cleared
    return out, meta
