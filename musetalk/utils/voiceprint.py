"""Speaker voiceprint helpers for multi-speaker face re-assignment.

The dubbing audio (``translated_voice.wav``) is a machine re-voice and is NOT
reliable for speaker discrimination (measured: CAM++ run on the English dub
collapses to noise on short windows). The ORIGINAL per-sentence audio the dub
pipeline leaves next to the output (``origin_seg_N.mp3``) is clean, so we key
the voice comparison off it.

Everything here fails SOFT: if ``modelscope`` / the CAM++ model / the original
segments / ``intermediate_result.json`` are missing, we return ``None`` and the
caller keeps the identity mapping (no behaviour change).
"""
from __future__ import annotations

import os
import json
from typing import List, Optional, Sequence

import numpy as np

SV_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"

_sv_pipeline = None
_sv_tried = False


def _get_sv():
    """Lazily build the CAM++ speaker-verification pipeline once per process."""
    global _sv_pipeline, _sv_tried
    if _sv_tried:
        return _sv_pipeline
    _sv_tried = True
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        _sv_pipeline = pipeline(task=Tasks.speaker_verification, model=SV_MODEL)
    except Exception:  # pragma: no cover - environment dependent
        _sv_pipeline = None
    return _sv_pipeline


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _seg_span(seg):
    from musetalk.utils.vad_client import vad_span

    return vad_span(seg)


def _load_intermediate(audio_path: str) -> Optional[dict]:
    """intermediate_result.json lives in the dub output (task) directory, which
    is the folder holding ``audio_path`` (or its parent). No API change needed."""
    here = os.path.dirname(os.path.abspath(audio_path))
    for cand in (
        os.path.join(here, "intermediate_result.json"),
        os.path.join(os.path.dirname(here), "intermediate_result.json"),
    ):
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def _origin_sentences(inter: dict) -> List[tuple]:
    out = []
    for s in (inter.get("sentence_list") or []):
        a, b = s.get("start"), s.get("end")
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if a > 1000 or b > 1000:  # stored in ms
            a, b = a / 1000.0, b / 1000.0
        if b - a > 600:  # bogus whole-file entry
            continue
        wav = s.get("origin_audio_seg")
        if not wav:
            continue
        out.append((a, b, wav))
    return out


def build_voice_cos(
    audio_path: str, vad_segments: Sequence
) -> Optional[np.ndarray]:
    """Return an ``(N, N)`` cosine matrix between the ``N`` VAD segments, using
    each segment's ORIGINAL source-audio embedding (matched by max time overlap).

    Returns ``None`` when any dependency is missing so the caller can no-op.
    A segment with no matching / on-disk origin clip gets a row that only
    self-matches (identity), so the greedy treats it as uncomparable.
    """
    n = len(vad_segments)
    if n < 2:
        return None
    sv = _get_sv()
    if sv is None:
        return None
    inter = _load_intermediate(audio_path)
    if not inter:
        return None
    sents = _origin_sentences(inter)
    if not sents:
        return None

    # per-segment origin wav = overlapping sentence with the largest intersection
    seg_wav: List[Optional[str]] = []
    for seg in vad_segments:
        s0, s1 = _seg_span(seg)
        best, bwav = 0.0, None
        for a, b, wav in sents:
            ov = min(s1, b) - max(s0, a)
            if ov > best and os.path.isfile(wav):
                best, bwav = ov, wav
        seg_wav.append(bwav)

    uniq: List[str] = []
    uidx = {}
    for w in seg_wav:
        if w and w not in uidx:
            uidx[w] = len(uniq)
            uniq.append(w)
    if len(uniq) < 2:
        return None

    try:
        embs = _l2n(
            np.asarray(sv(uniq, output_emb=True)["embs"], dtype=np.float64)
        )
    except Exception:
        return None

    C = np.eye(n, dtype=np.float64)
    valid = [i for i, w in enumerate(seg_wav) if w]
    for i in valid:
        ei = embs[uidx[seg_wav[i]]]
        for j in valid:
            C[i, j] = float(ei @ embs[uidx[seg_wav[j]]])
    return C


def plan_voice_reassign(
    audio_path: str,
    frame_tracks: Sequence,
    open_probs: dict,
    coord_list: Sequence,
    shot_ids: Sequence,
    *,
    fps: float,
    same_speaker_thr: float = 0.40,
    presence_floor: float = 0.50,
    piece_min_sec: float = 0.50,
    on_min_op_std: float = 0.02,
    on_min_mean_open: float = 0.35,
    min_scored_coverage: float = 0.5,
) -> dict:
    """Full re-assignment plan on the task's ORIGINAL per-sentence spans.

    The pipeline's own VAD merges contiguous dub sentences, so it cannot tell two
    speakers inside one merged block apart. We re-segment on the sentence list
    (which owns a clean origin-audio embedding each) and reuse the validated
    shot-piece greedy. Returns {"off": [(i0,i1)], "on": [...], "diag": [...]} of
    FRAME ranges, or an empty plan when any dependency is missing (identity).
    """
    empty = {"off": [], "on": [], "diag": []}
    inter = _load_intermediate(audio_path)
    if not inter:
        return empty
    sents = _origin_sentences(inter)
    spans = [(a, b) for a, b, _ in sents]
    if len(spans) < 2:
        return empty
    C = build_voice_cos(audio_path, spans)
    if C is None:
        return empty
    from musetalk.utils.vad_vsdlm_fusion import compute_voice_reassign_intervals

    try:
        return compute_voice_reassign_intervals(
            spans, C, frame_tracks, open_probs, coord_list, shot_ids,
            fps=fps, same_speaker_thr=same_speaker_thr,
            presence_floor=presence_floor, piece_min_sec=piece_min_sec,
            on_min_op_std=on_min_op_std, on_min_mean_open=on_min_mean_open,
            min_scored_coverage=min_scored_coverage,
        )
    except Exception:  # pragma: no cover - never break the run
        return empty
