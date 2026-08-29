"""Mouth / lower-face occlusion gate for lipsync.

Industry default for talking-head dubbing: if the mouth region is partially
covered (hand, mic, hair, object), skip lipsync and keep the original plate,
with a soft crossfade near the visibility threshold.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# CelebAMask-HQ / BiSeNet-19 class ids.
_LIP_CLASSES = (11, 12, 13)  # mouth, u_lip, l_lip
_OCCLUDE = (0, 16, 17, 18)  # background, cloth, hair, hat


def _clamp_box(box, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if box is None:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def mouth_roi_from_bboxes(
    face_bbox,
    mouth_bbox=None,
    *,
    frame_w: int,
    frame_h: int,
    lower_face_frac: float = 0.45,
    mouth_expand: float = 0.35,
) -> Optional[Tuple[int, int, int, int]]:
    """Prefer expanded mouth box; else lower portion of the face box."""
    if mouth_bbox is not None:
        box = _clamp_box(mouth_bbox, frame_w, frame_h)
        if box is not None:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(round(bw * mouth_expand))
            pad_y = int(round(bh * mouth_expand))
            return _clamp_box(
                (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
                frame_w,
                frame_h,
            )
    face = _clamp_box(face_bbox, frame_w, frame_h)
    if face is None:
        return None
    x1, y1, x2, y2 = face
    mid = y1 + int(round((y2 - y1) * (1.0 - float(lower_face_frac))))
    return _clamp_box((x1, mid, x2, y2), frame_w, frame_h)


def mouth_visibility_from_parsing(
    parsing: np.ndarray,
    *,
    roi_in_parse: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    """Mouth visibility in ``[0, 1]`` from a BiSeNet label map.

    Combines lip presence in the ROI with an occluder penalty (hair / cloth /
    hat / background). Hands mislabeled as skin still depress the score when
    lip classes disappear.
    """
    if parsing is None or parsing.size == 0:
        return 1.0
    labels = parsing
    if roi_in_parse is not None:
        x1, y1, x2, y2 = roi_in_parse
        labels = parsing[y1:y2, x1:x2]
        if labels.size == 0:
            return 1.0
    n = float(labels.size)
    if n < 16.0:
        return 1.0
    lip = float(np.isin(labels, _LIP_CLASSES).sum())
    occ = float(np.isin(labels, _OCCLUDE).sum())
    # ~8% lip pixels in an expanded mouth / lower-face ROI ≈ healthy.
    lip_score = min(1.0, lip / max(n * 0.08, 1.0))
    occ_pen = occ / n
    return max(0.0, min(1.0, lip_score * (1.0 - occ_pen)))


def score_mouth_visibility(
    frame_bgr: np.ndarray,
    face_bbox,
    face_parser,
    mouth_bbox=None,
    *,
    parse_size: Tuple[int, int] = (512, 512),
) -> Optional[float]:
    """Run face parsing on the face crop and return mouth visibility, or None."""
    if face_parser is None or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    face = _clamp_box(face_bbox, w, h)
    if face is None:
        return None
    fx1, fy1, fx2, fy2 = face
    crop = frame_bgr[fy1:fy2, fx1:fx2]
    if crop.size == 0:
        return None

    try:
        parsing = face_parser.parse_labels(crop, size=parse_size)
    except Exception:
        return None

    ph, pw = parsing.shape[:2]
    sx = pw / float(max(1, fx2 - fx1))
    sy = ph / float(max(1, fy2 - fy1))

    # Intersect mouth ROI with the face crop so parse coords stay valid.
    roi = mouth_roi_from_bboxes(
        face_bbox, mouth_bbox, frame_w=w, frame_h=h
    )
    roi_parse = None
    if roi is not None:
        rx1 = max(fx1, roi[0])
        ry1 = max(fy1, roi[1])
        rx2 = min(fx2, roi[2])
        ry2 = min(fy2, roi[3])
        if rx2 > rx1 + 2 and ry2 > ry1 + 2:
            px1 = int(round((rx1 - fx1) * sx))
            py1 = int(round((ry1 - fy1) * sy))
            px2 = int(round((rx2 - fx1) * sx))
            py2 = int(round((ry2 - fy1) * sy))
            px1 = max(0, min(pw - 1, px1))
            py1 = max(0, min(ph - 1, py1))
            px2 = max(0, min(pw, px2))
            py2 = max(0, min(ph, py2))
            if px2 > px1 + 2 and py2 > py1 + 2:
                roi_parse = (px1, py1, px2, py2)
    if roi_parse is None:
        py1 = int(round(ph * 0.55))
        roi_parse = (0, py1, pw, ph)

    return mouth_visibility_from_parsing(parsing, roi_in_parse=roi_parse)


def interpolate_sparse_scores(
    sparse: Sequence[Optional[float]],
    n_frames: int,
) -> List[Optional[float]]:
    """Hold/lerp visibility scores between keyframes; None outside spans."""
    if n_frames <= 0:
        return []
    out: List[Optional[float]] = [None] * n_frames
    keys = [
        i
        for i, v in enumerate(sparse)
        if v is not None and math.isfinite(float(v))
    ]
    if not keys:
        return out
    for i, k in enumerate(keys):
        out[k] = float(sparse[k])  # type: ignore[arg-type]
        if i + 1 >= len(keys):
            continue
        k1 = keys[i + 1]
        y0 = float(sparse[k])  # type: ignore[arg-type]
        y1 = float(sparse[k1])  # type: ignore[arg-type]
        span = k1 - k
        if span <= 1:
            continue
        for j in range(k + 1, k1):
            t = (j - k) / float(span)
            out[j] = y0 + (y1 - y0) * t
    # Hold edges.
    first, last = keys[0], keys[-1]
    for j in range(0, first):
        out[j] = out[first]
    for j in range(last + 1, n_frames):
        out[j] = out[last]
    return out


def mouth_occlusion_blend_weight(
    visibility: Optional[float],
    *,
    clear_below: float = 0.25,
    full_above: float = 0.55,
) -> float:
    """Opacity multiplier: 0 when occluded, 1 when mouth clearly visible."""
    if visibility is None:
        return 1.0
    try:
        v = float(visibility)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    lo = max(0.0, float(clear_below))
    hi = max(lo, float(full_above))
    if v <= lo:
        return 0.0
    if v >= hi or hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def apply_mouth_occlusion_gate(
    speaking_mask: Sequence[bool],
    visibility: Sequence[Optional[float]],
    *,
    clear_below: float = 0.25,
) -> Tuple[List[bool], dict]:
    """Hard-clear speaking frames whose mouth visibility is below threshold."""
    n = len(speaking_mask)
    out = list(speaking_mask)
    meta = {
        "cleared_frames": 0,
        "clear_below": float(clear_below),
        "scored_frames": 0,
        "min_visibility": None,
    }
    if n == 0:
        return out, meta
    if len(visibility) != n:
        raise ValueError("visibility length must match speaking_mask")

    lo = max(0.0, float(clear_below))
    cleared = 0
    scored = 0
    min_v = None
    for i in range(n):
        v = visibility[i]
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(vf):
            continue
        scored += 1
        min_v = vf if min_v is None else min(min_v, vf)
        if out[i] and vf < lo:
            out[i] = False
            cleared += 1
    meta["cleared_frames"] = cleared
    meta["scored_frames"] = scored
    meta["min_visibility"] = min_v
    return out, meta


def estimate_mouth_visibility_sequence(
    frames: Sequence[np.ndarray],
    coord_list: Sequence,
    face_parser,
    *,
    mouth_coord_list: Optional[Sequence] = None,
    indices: Optional[Sequence[int]] = None,
    stride: int = 3,
    coord_placeholder=(0.0, 0.0, 0.0, 0.0),
) -> List[Optional[float]]:
    """Score mouth visibility on a stride; interpolate for the full timeline."""
    n = len(frames)
    sparse: List[Optional[float]] = [None] * n
    if face_parser is None or n == 0:
        return sparse

    stride = max(1, int(stride))
    if indices is None:
        sample = list(range(0, n, stride))
    else:
        sample = sorted({int(i) for i in indices if 0 <= int(i) < n})
        # Subsample for cost control while covering the speaking span.
        if stride > 1 and len(sample) > stride:
            sample = sample[::stride]

    for i in sample:
        bbox = coord_list[i] if i < len(coord_list) else None
        if bbox is None or bbox == coord_placeholder:
            continue
        mouth = None
        if mouth_coord_list is not None and i < len(mouth_coord_list):
            mouth = mouth_coord_list[i]
        sparse[i] = score_mouth_visibility(
            frames[i], bbox, face_parser, mouth_bbox=mouth
        )
    return interpolate_sparse_scores(sparse, n)
