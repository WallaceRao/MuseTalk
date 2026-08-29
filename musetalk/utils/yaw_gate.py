"""Gate lipsync during large head yaw (InsightFace degrees + shot hysteresis)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


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


def apply_yaw_hysteresis_gate(
    speaking_mask: Sequence[bool],
    yaw_deg: Sequence[Optional[float]],
    shot_ids: Optional[Sequence[int]] = None,
    *,
    off_deg: float = 60.0,
    on_deg: float = 45.0,
) -> Tuple[List[bool], dict]:
    """Clear speaking frames with per-shot yaw hysteresis.

    Within each camera shot (or the whole clip if ``shot_ids`` is None):

    - start in **sync** mode
    - ``|yaw| >= off_deg`` → enter **non-sync** (cancel lipsync)
    - stay non-sync until ``|yaw| <= on_deg`` → return to sync

    Missing yaw values do not flip state. Shot boundaries reset to sync.
    """
    n = len(speaking_mask)
    out = list(speaking_mask)
    off_thr = max(0.0, float(off_deg))
    on_thr = max(0.0, float(on_deg))
    if on_thr > off_thr:
        on_thr = off_thr

    meta = {
        "cleared_frames": 0,
        "off_deg": off_thr,
        "on_deg": on_thr,
        "entered_off": 0,
        "reentered_on": 0,
        "shots": 0,
    }
    if n == 0:
        return out, meta
    if len(yaw_deg) != n:
        raise ValueError("yaw_deg length must match speaking_mask")
    if shot_ids is not None and len(shot_ids) != n:
        raise ValueError("shot_ids length must match speaking_mask")

    syncing = True
    prev_shot: Optional[int] = None
    for i in range(n):
        sid = int(shot_ids[i]) if shot_ids is not None else 0
        if prev_shot is None or sid != prev_shot:
            syncing = True
            prev_shot = sid
            meta["shots"] += 1

        y = yaw_deg[i]
        if y is not None:
            try:
                yf = abs(float(y))
            except (TypeError, ValueError):
                yf = None
            if yf is not None and math.isfinite(yf):
                if syncing and yf >= off_thr:
                    syncing = False
                    meta["entered_off"] += 1
                elif (not syncing) and yf <= on_thr:
                    syncing = True
                    meta["reentered_on"] += 1

        if out[i] and not syncing:
            out[i] = False
            meta["cleared_frames"] += 1

    return out, meta
