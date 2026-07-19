"""TenVAD HTTP client for speech segment detection."""

from __future__ import annotations

import logging
import os
from typing import List, Sequence, Tuple

import requests

logger = logging.getLogger("musetalk_service")

VadSegment = Tuple[float, float]


def detect_voice_segments(
    audio_path: str,
    *,
    vad_url: str | None = None,
    timeout_sec: float = 120.0,
) -> List[VadSegment]:
    """Call the TenVAD service and return ``[(start_sec, end_sec), ...]``."""
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
    out: List[VadSegment] = []
    for seg in segments:
        if isinstance(seg, dict):
            start = float(seg["start"])
            end = float(seg["end"])
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            start, end = float(seg[0]), float(seg[1])
        else:
            continue
        if end > start:
            out.append((start, end))
    out.sort(key=lambda x: x[0])
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
