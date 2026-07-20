"""TenVAD HTTP client for speech segment detection."""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Sequence, Tuple, Union

import requests

logger = logging.getLogger("musetalk_service")

VadSegment = Tuple[float, float]


def normalize_vad_segments(
    segments: Sequence,
    *,
    strict: bool = True,
) -> List[VadSegment]:
    """Normalize raw segment items to ``[(start_sec, end_sec), ...]``."""
    out: List[VadSegment] = []
    for seg in segments:
        try:
            if isinstance(seg, dict):
                if "start" not in seg or "end" not in seg:
                    raise ValueError(
                        "Each VAD segment object must have 'start' and 'end' fields"
                    )
                start = float(seg["start"])
                end = float(seg["end"])
            elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                start, end = float(seg[0]), float(seg[1])
            else:
                raise ValueError(
                    "Each VAD segment must be {start,end} or [start,end]"
                )
        except (TypeError, ValueError, KeyError):
            if strict:
                raise
            continue
        if end > start:
            out.append((start, end))
    out.sort(key=lambda x: x[0])
    return out


def parse_vad_result(
    vad_result: Union[str, Sequence, None],
) -> Optional[List[VadSegment]]:
    """Parse optional client-provided VAD segments.

    Accepts a JSON string or an already-decoded list. Returns ``None`` when
    ``vad_result`` is empty/None (caller should run remote VAD). Empty list
    ``[]`` means "no speech" and skips remote VAD.
    """
    if vad_result is None:
        return None
    if isinstance(vad_result, str):
        text = vad_result.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VAD_result is not valid JSON: {exc}") from exc
    else:
        payload = vad_result
    if not isinstance(payload, (list, tuple)):
        raise ValueError("VAD_result must be a JSON array of {start,end} pairs")
    return normalize_vad_segments(payload)


def clip_vad_segments_to_window(
    segments: Sequence[VadSegment],
    *,
    window_start_sec: float,
    window_duration_sec: float,
) -> List[VadSegment]:
    """Clip absolute VAD segments into a local chunk timeline starting at 0."""
    t0 = float(window_start_sec)
    t1 = t0 + float(window_duration_sec)
    if t1 <= t0:
        return []
    out: List[VadSegment] = []
    for start, end in segments:
        a = max(float(start), t0)
        b = min(float(end), t1)
        if b > a:
            out.append((a - t0, b - t0))
    return out


def detect_voice_segments(
    audio_path: str,
    *,
    vad_url: str | None = None,
    timeout_sec: float = 120.0,
    vad_segments: Sequence[VadSegment] | None = None,
) -> List[VadSegment]:
    """Return voice segments, optionally using client-provided ``vad_segments``.

    When ``vad_segments`` is not ``None``, remote TenVAD is skipped and the
    provided list is normalized/returned as-is.
    """
    if vad_segments is not None:
        out = normalize_vad_segments(list(vad_segments), strict=True)
        logger.info(
            "VAD (client-provided): %d voice segments (skip remote detect)",
            len(out),
        )
        return out

    url = (vad_url or os.environ.get("MUSETALK_VAD_URL") or "http://127.0.0.1:8061/vad_detect/").strip()
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio for VAD not found: {audio_path}")

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"audio_file": (os.path.basename(audio_path), f)},
            timeout=timeout_sec,
        )
    response.raise_for_status()
    payload = response.json()
    segments = payload.get("voice_segments") or []
    out = normalize_vad_segments(segments, strict=False)
    logger.info("VAD (%s): %d voice segments from %s", url, len(out), audio_path)
    return out


def segments_to_frame_mask(
    segments: Sequence[VadSegment],
    n_frames: int,
    fps: float,
) -> List[bool]:
    """Convert time segments to a per-frame boolean mask."""
    fps = float(fps) if fps and fps > 0 else 25.0
    mask = [False] * n_frames
    for start, end in segments:
        i0 = max(0, int(start * fps))
        i1 = min(n_frames, int(round(end * fps)))
        for i in range(i0, i1):
            mask[i] = True
    return mask
