"""Filter speaking runs whose visual gender conflicts with VAD gender."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from musetalk.utils.vad_client import normalize_gender, vad_gender, vad_span

logger = logging.getLogger("musetalk_service")


def _contiguous_true_runs(
    speaking_mask: Sequence[bool],
    shot_ids: Sequence[int] | None = None,
) -> List[Tuple[int, int]]:
    """Inclusive (start, end) True runs; split when shot id changes."""
    n = len(speaking_mask)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not speaking_mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and speaking_mask[j]:
            if shot_ids is not None and int(shot_ids[j]) != int(shot_ids[i]):
                break
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _sample_frame_indices(start: int, end: int, max_samples: int = 8) -> List[int]:
    """Evenly sample inclusive [start, end], up to ``max_samples`` indices."""
    n = end - start + 1
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(start, end + 1))
    idxs = [
        start + int(round(i * (n - 1) / float(max_samples - 1)))
        for i in range(max_samples)
    ]
    out: List[int] = []
    seen = set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def visual_gender_for_run(
    frame_list: Sequence[np.ndarray],
    run_start: int,
    run_end: int,
    face_detector,
    *,
    max_samples: int = 8,
) -> Optional[str]:
    """Majority-vote visual gender for a speaking run; None if unclear."""
    if face_detector is None:
        return None
    detect_gender = getattr(face_detector, "detect_gender", None)
    if detect_gender is None:
        return None

    votes = {"male": 0, "female": 0}
    for idx in _sample_frame_indices(run_start, run_end, max_samples=max_samples):
        if idx < 0 or idx >= len(frame_list):
            continue
        rgb = cv2.cvtColor(frame_list[idx], cv2.COLOR_BGR2RGB)
        g = normalize_gender(detect_gender(rgb))
        if g in votes:
            votes[g] += 1

    if votes["male"] == 0 and votes["female"] == 0:
        return None
    if votes["male"] == votes["female"]:
        return None
    return "male" if votes["male"] > votes["female"] else "female"


def vad_gender_for_run(
    run_start: int,
    run_end: int,
    vad_segments: Sequence,
    *,
    fps: float,
    min_overlap_ratio: float = 0.5,
) -> Optional[str]:
    """Gender of the VAD segment that best overlaps this run, or None."""
    if not vad_segments or fps <= 0:
        return None
    t0 = run_start / float(fps)
    t1 = (run_end + 1) / float(fps)
    best_gender: Optional[str] = None
    best_overlap = 0.0
    for seg in vad_segments:
        vs, ve = vad_span(seg)
        if ve <= vs:
            continue
        overlap = max(0.0, min(t1, ve) - max(t0, vs))
        if overlap > best_overlap:
            best_overlap = overlap
            best_gender = vad_gender(seg)
    if best_overlap <= 0:
        return None
    run_dur = max(t1 - t0, 1e-6)
    if best_overlap / run_dur < float(min_overlap_ratio):
        return None
    return normalize_gender(best_gender)


def filter_speaking_mask_by_vad_gender(
    speaking_mask: Sequence[bool],
    frame_list: Sequence[np.ndarray],
    vad_segments: Sequence,
    face_detector,
    *,
    fps: float,
    shot_ids: Sequence[int] | None = None,
    max_samples: int = 8,
) -> Tuple[List[bool], dict]:
    """Drop pre-dilate speaking runs whose visual gender conflicts with VAD.

    Unclear VAD gender, unclear visual gender, or missing detector → keep run.
    Only definite male/female mismatches are zeroed out.
    """
    out = list(speaking_mask)
    meta = {
        "enabled": False,
        "dropped_runs": 0,
        "dropped_frames": 0,
        "checked_runs": 0,
        "pass_unclear": 0,
        "pass_match": 0,
        "mismatches": [],
    }
    if not out or not vad_segments or face_detector is None:
        return out, meta

    has_gender = any(vad_gender(seg) is not None for seg in vad_segments)
    if not has_gender:
        return out, meta

    meta["enabled"] = True
    runs = _contiguous_true_runs(out, shot_ids=shot_ids)
    for rs, re in runs:
        meta["checked_runs"] += 1
        vad_g = vad_gender_for_run(rs, re, vad_segments, fps=fps)
        if vad_g is None:
            meta["pass_unclear"] += 1
            continue
        vis_g = visual_gender_for_run(
            frame_list, rs, re, face_detector, max_samples=max_samples
        )
        if vis_g is None:
            meta["pass_unclear"] += 1
            continue
        if vis_g == vad_g:
            meta["pass_match"] += 1
            continue

        for i in range(rs, re + 1):
            out[i] = False
        n_frames = re - rs + 1
        meta["dropped_runs"] += 1
        meta["dropped_frames"] += n_frames
        meta["mismatches"].append(
            {"start": rs, "end": re, "visual": vis_g, "vad": vad_g}
        )

    if meta["dropped_runs"]:
        preview = meta["mismatches"][:8]
        more = (
            ""
            if len(meta["mismatches"]) <= 8
            else f" …(+{len(meta['mismatches']) - 8})"
        )
        logger.info(
            "Gender gate: drop %d runs / %d frames "
            "(checked=%d, match=%d, unclear=%d) mismatches=%s%s",
            meta["dropped_runs"],
            meta["dropped_frames"],
            meta["checked_runs"],
            meta["pass_match"],
            meta["pass_unclear"],
            preview,
            more,
        )
    else:
        logger.info(
            "Gender gate: no mismatches "
            "(checked=%d, match=%d, unclear=%d)",
            meta["checked_runs"],
            meta["pass_match"],
            meta["pass_unclear"],
        )
    return out, meta
