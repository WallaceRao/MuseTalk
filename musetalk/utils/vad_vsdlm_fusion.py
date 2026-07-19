"""Fuse VAD speech segments with multi-face VSDLM speaking intervals.

Pipeline
--------
1. Track faces across frames (SCRFD + IoU) and run VSDLM lip-motion per face.
2. Collect contiguous visual speaking intervals per track (``speaker_id``).
3. Run VAD to get audio speech segments.
4. Assign each VAD segment to the nearest VSDLM speaker in time.
5. If one VAD segment overlaps multiple VSDLM speakers, split it at speaker
   switch points inside the segment.
6. Build a lipsync speaking mask for the MuseTalk primary face track.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from musetalk.utils.preprocessing import (
    _bbox_iou,
    _collect_scrfd_faces,
    _face_meets_min_area,
    _resize_for_detection,
    coord_placeholder,
)
from musetalk.utils.vad_client import VadSegment, detect_voice_segments

logger = logging.getLogger("musetalk_service")

BBox = Tuple[float, float, float, float]
VisualInterval = Tuple[int, float, float]  # (speaker_id, start_sec, end_sec)
AssignedSegment = Tuple[float, float, int]  # (start_sec, end_sec, speaker_id)


@dataclass
class FaceTrackState:
    track_id: int
    bbox: BBox
    last_seen: int


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Non-overlap gap (>=0). 0 means overlap or touch."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def track_faces_across_frames(
    frame_list: Sequence[np.ndarray],
    *,
    detect_short_side: int = 720,
    min_face_area_ratio: float = 1.0 / 50.0,
    track_min_iou: float = 0.3,
    max_miss: int = 8,
    detect_stride: int = 1,
    shot_ids: Sequence[int] | None = None,
) -> Tuple[List[Dict[int, BBox]], int]:
    """Multi-face IoU tracker.

    When ``shot_ids`` is provided, active tracks are cleared at every shot cut
    so face IDs never continue across camera changes.

    ``detect_stride`` > 1 runs SCRFD every N frames and holds the last matched
    boxes on intermediate frames (always detect on shot cuts / last frame).

    Returns
    -------
    frame_tracks : list of {track_id: bbox_xyxy in original frame coords}
    next_track_id : int
    """
    n = len(frame_list)
    if shot_ids is not None and len(shot_ids) != n:
        raise ValueError("shot_ids length must match frame_list")
    stride = max(1, int(detect_stride))

    active: List[FaceTrackState] = []
    next_id = 0
    frame_tracks: List[Dict[int, BBox]] = []

    for fi, frame in enumerate(frame_list):
        # Hard reset on shot cut — cross-shot IoU matching is unreliable.
        on_cut = (
            shot_ids is not None
            and fi > 0
            and int(shot_ids[fi]) != int(shot_ids[fi - 1])
        )
        if on_cut:
            active = []

        do_detect = (
            stride <= 1
            or on_cut
            or fi == 0
            or fi == n - 1
            or (fi % stride == 0)
            or not active
        )

        current: Dict[int, BBox] = {}
        if do_detect:
            detect_frame, sx, sy = _resize_for_detection(frame, detect_short_side)
            hd, wd = detect_frame.shape[:2]
            dets = [
                f
                for f in _collect_scrfd_faces(detect_frame)
                if _face_meets_min_area(f, wd, hd, min_face_area_ratio)
            ]
            # Map to original coords.
            dets_orig: List[BBox] = [
                (f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy) for f in dets
            ]

            matched_det = set()
            matched_track = set()
            pairs: List[Tuple[float, int, int]] = []
            for ti, tr in enumerate(active):
                for di, db in enumerate(dets_orig):
                    iou = _bbox_iou(tr.bbox, db)
                    if iou >= track_min_iou:
                        pairs.append((iou, ti, di))
            pairs.sort(reverse=True)

            for iou, ti, di in pairs:
                if ti in matched_track or di in matched_det:
                    continue
                matched_track.add(ti)
                matched_det.add(di)
                active[ti].bbox = dets_orig[di]
                active[ti].last_seen = fi
                current[active[ti].track_id] = dets_orig[di]

            for di, db in enumerate(dets_orig):
                if di in matched_det:
                    continue
                active.append(
                    FaceTrackState(track_id=next_id, bbox=db, last_seen=fi)
                )
                current[next_id] = db
                next_id += 1
        else:
            # Hold last boxes between detect frames (do not refresh last_seen).
            for tr in active:
                if fi - tr.last_seen <= max_miss:
                    current[tr.track_id] = tr.bbox

        # Drop stale tracks after consecutive detect misses.
        active = [t for t in active if fi - t.last_seen <= max_miss]
        frame_tracks.append(current)

    return frame_tracks, next_id


def _mask_to_intervals(
    mask: Sequence[bool],
    *,
    fps: float,
    speaker_id: int,
    shot_ids: Sequence[int] | None = None,
) -> List[VisualInterval]:
    n = len(mask)
    fps = float(fps) if fps and fps > 0 else 25.0
    intervals: List[VisualInterval] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            if shot_ids is not None and int(shot_ids[j]) != int(shot_ids[i]):
                break
            j += 1
        intervals.append((speaker_id, i / fps, j / fps))
        i = j
    return intervals


def compute_visual_speaking_intervals(
    vsdlm_detector,
    frame_list: Sequence[np.ndarray],
    frame_tracks: Sequence[Dict[int, BBox]],
    *,
    fps: float,
    shot_ids: Sequence[int] | None = None,
    skip_track_ids: Sequence[int] | None = None,
    only_multiface_frames: bool = False,
) -> List[VisualInterval]:
    """Run batched VSDLM per tracked face and return speaking intervals.

    Lip-motion activity, min-speak runs, and emitted intervals are all confined
    to a single shot when ``shot_ids`` is provided. Open scores at shot cuts /
    hard bbox jumps are discarded.

    When ``only_multiface_frames`` is True, faces are scored only on frames that
    contain 2+ tracks (enough for other-speaker exclusion). ``skip_track_ids``
    omits already-scored primary faces.
    """
    n = len(frame_list)
    fps = float(fps) if fps and fps > 0 else 25.0
    if shot_ids is not None and len(shot_ids) != n:
        raise ValueError("shot_ids length must match frame_list")
    skip = {int(x) for x in (skip_track_ids or ())}
    track_ids = sorted(
        {tid for ft in frame_tracks for tid in ft if int(tid) not in skip}
    )
    if not track_ids:
        return []

    open_probs: Dict[int, List[float]] = {
        tid: [float("nan")] * n for tid in track_ids
    }
    coord_streams: Dict[int, List[BBox]] = {
        tid: [coord_placeholder] * n for tid in track_ids
    }
    crop_keys: List[Tuple[int, int]] = []
    crops: List[np.ndarray] = []

    for fi, frame in enumerate(frame_list):
        tracks = frame_tracks[fi]
        if only_multiface_frames and len(tracks) < 2:
            continue
        for tid, bbox in tracks.items():
            if int(tid) in skip:
                continue
            if tid not in coord_streams:
                continue
            coord_streams[tid][fi] = bbox
            crop = vsdlm_detector._mouth_crop_from_face(frame, bbox)
            if crop is None or crop.size == 0:
                continue
            crop_keys.append((tid, fi))
            crops.append(crop)

    if crops:
        probs = vsdlm_detector._predict_open_batch(crops)
        for (tid, fi), prob in zip(crop_keys, probs):
            open_probs[tid][fi] = prob

    win = vsdlm_detector.activity_window
    thr = vsdlm_detector.activity_threshold
    open_thr = vsdlm_detector.open_threshold
    min_frames = (
        max(1, int(round(vsdlm_detector.min_speak_duration_sec * fps)))
        if vsdlm_detector.min_speak_duration_sec > 0
        else 1
    )

    intervals: List[VisualInterval] = []
    for tid in track_ids:
        probs = vsdlm_detector._invalidate_unstable_open_probs(
            open_probs[tid],
            coord_streams[tid],
            shot_ids=shot_ids,
        )
        motion = [False] * n
        for i in range(n):
            if not np.isfinite(probs[i]):
                continue
            cur_shot = int(shot_ids[i]) if shot_ids is not None else None
            finite = []
            for j in range(max(0, i - win), min(n, i + win + 1)):
                if shot_ids is not None and int(shot_ids[j]) != cur_shot:
                    continue
                if np.isfinite(probs[j]):
                    finite.append(probs[j])
            if len(finite) < 3:
                continue
            activity = float(np.std(finite))
            mean_open = float(np.mean(finite))
            motion[i] = activity >= thr and mean_open >= open_thr
        kept = vsdlm_detector._keep_runs_at_least(
            motion, min_frames, shot_ids=shot_ids
        )
        intervals.extend(
            _mask_to_intervals(
                kept, fps=fps, speaker_id=int(tid), shot_ids=shot_ids
            )
        )

    intervals.sort(key=lambda x: (x[1], x[2], x[0]))
    return intervals


def _speaker_switch_points(
    visual_intervals: Sequence[VisualInterval],
    t0: float,
    t1: float,
) -> List[float]:
    """Return times inside (t0, t1) where the active VSDLM speaker changes."""
    # Event-based: collect boundaries of overlapping intervals, then verify change.
    events = sorted(
        {
            x
            for _, s0, s1 in visual_intervals
            for x in (s0, s1)
            if t0 < x < t1
        }
    )
    if not events:
        return []

    def speaker_at(t: float) -> Optional[int]:
        best_id = None
        best_ov = 0.0
        # Instantaneous: prefer interval containing t; tie-break by duration left.
        for sid, s0, s1 in visual_intervals:
            if s0 <= t < s1:
                dur = s1 - s0
                if dur > best_ov:
                    best_ov = dur
                    best_id = sid
        return best_id

    switches: List[float] = []
    prev = speaker_at(t0 + 1e-6)
    for ev in events:
        cur = speaker_at(ev + 1e-6)
        if cur != prev and (cur is not None or prev is not None):
            # Only count as switch when speaker identity among speaking faces changes.
            if cur is not None and prev is not None and cur != prev:
                switches.append(ev)
            elif cur is not None and prev is None:
                # silence → speaker: useful split if another speaker already in segment
                switches.append(ev)
            elif cur is None and prev is not None:
                switches.append(ev)
            prev = cur
        else:
            prev = cur
    return switches


def _nearest_speaker(
    visual_intervals: Sequence[VisualInterval],
    t0: float,
    t1: float,
) -> Optional[int]:
    if not visual_intervals:
        return None
    best_id = None
    best_key = None
    mid = 0.5 * (t0 + t1)
    for sid, s0, s1 in visual_intervals:
        gap = _interval_gap(t0, t1, s0, s1)
        vmid = 0.5 * (s0 + s1)
        # Prefer overlap (gap=0), then smaller gap, then closer centers.
        key = (gap, abs(vmid - mid))
        if best_key is None or key < best_key:
            best_key = key
            best_id = sid
    return best_id


def _speakers_overlapping(
    visual_intervals: Sequence[VisualInterval],
    t0: float,
    t1: float,
) -> List[int]:
    ids = []
    seen = set()
    for sid, s0, s1 in visual_intervals:
        if _interval_overlap(t0, t1, s0, s1) > 0 and sid not in seen:
            ids.append(sid)
            seen.add(sid)
    return ids


def fuse_vad_with_vsdlm(
    vad_segments: Sequence[VadSegment],
    visual_intervals: Sequence[VisualInterval],
    *,
    max_assign_gap_sec: float = 0.5,
) -> List[AssignedSegment]:
    """Assign / split VAD segments onto VSDLM speakers.

    - Overlap with one speaker → assign to that speaker.
    - Overlap with multiple → split at VSDLM speaker-switch times.
    - No overlap → nearest speaker only if gap <= ``max_assign_gap_sec``;
      otherwise ``speaker_id=-1`` (treat as narration / off-screen speech).
    """
    if not vad_segments:
        return []
    if not visual_intervals:
        return [(s, e, -1) for s, e in vad_segments]

    max_gap = max(0.0, float(max_assign_gap_sec))
    assigned: List[AssignedSegment] = []
    for vad_start, vad_end in vad_segments:
        overlapping = _speakers_overlapping(visual_intervals, vad_start, vad_end)
        if len(overlapping) <= 1:
            if overlapping:
                sid = overlapping[0]
            else:
                sid = _nearest_speaker(visual_intervals, vad_start, vad_end)
                if sid is not None:
                    # Reject far assignments (voiceover / unrelated face turns).
                    best_gap = min(
                        _interval_gap(vad_start, vad_end, s0, s1)
                        for s, s0, s1 in visual_intervals
                        if s == sid
                    )
                    if best_gap > max_gap:
                        sid = -1
                else:
                    sid = -1
            assigned.append((vad_start, vad_end, int(sid)))
            continue

        switches = _speaker_switch_points(visual_intervals, vad_start, vad_end)
        cuts = [vad_start] + switches + [vad_end]
        cuts = sorted(set(round(c, 6) for c in cuts))
        for i in range(len(cuts) - 1):
            a, b = cuts[i], cuts[i + 1]
            if b - a < 1e-4:
                continue
            ov = _speakers_overlapping(visual_intervals, a, b)
            if len(ov) == 1:
                sid = ov[0]
            else:
                sid = _nearest_speaker(visual_intervals, a, b)
                if sid is not None:
                    best_gap = min(
                        _interval_gap(a, b, s0, s1)
                        for s, s0, s1 in visual_intervals
                        if s == sid
                    )
                    if best_gap > max_gap:
                        sid = -1
                else:
                    sid = -1
            if sid is None:
                sid = -1
            assigned.append((a, b, int(sid)))
    return assigned


def match_primary_track_id(
    coord_list: Sequence[Tuple[float, float, float, float]],
    frame_tracks: Sequence[Dict[int, BBox]],
) -> Optional[int]:
    """Vote which multi-face track_id matches MuseTalk's single tracked face."""
    votes: Dict[int, float] = {}
    n = min(len(coord_list), len(frame_tracks))
    for i in range(n):
        bbox = coord_list[i]
        if bbox == coord_placeholder:
            continue
        best_tid = None
        best_iou = 0.0
        for tid, tb in frame_tracks[i].items():
            iou = _bbox_iou(bbox, tb)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        if best_tid is not None and best_iou >= 0.1:
            votes[best_tid] = votes.get(best_tid, 0.0) + best_iou
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def assigned_segments_to_mask(
    assigned: Sequence[AssignedSegment],
    n_frames: int,
    fps: float,
    *,
    speaker_id: Optional[int] = None,
) -> List[bool]:
    """Frames covered by assigned VAD pieces (optionally filtered by speaker)."""
    fps = float(fps) if fps and fps > 0 else 25.0
    mask = [False] * n_frames
    for t0, t1, sid in assigned:
        if speaker_id is not None and sid != speaker_id:
            continue
        if sid < 0 and speaker_id is not None:
            continue
        i0 = max(0, int(t0 * fps))
        i1 = min(n_frames, int(round(t1 * fps)))
        for i in range(i0, i1):
            mask[i] = True
    return mask


def build_fused_speaking_mask(
    vsdlm_detector,
    frame_list: Sequence[np.ndarray],
    coord_list: Sequence[Tuple[float, float, float, float]],
    audio_path: str,
    *,
    fps: float,
    vad_url: str | None = None,
    detect_short_side: int = 720,
    min_face_area_ratio: float = 1.0 / 50.0,
    detect_stride: int = 3,
    loose_min_speak_duration_sec: float = 0.2,
    max_assign_gap_sec: float = 0.5,
    shot_ids: Sequence[int] | None = None,
    shot_cut_hist_threshold: float = 0.45,
    mouth_coord_list: Sequence[Tuple[float, float, float, float] | None] | None = None,
    mouth_mar_list: Sequence[float | None] | None = None,
) -> Tuple[List[bool], dict]:
    """VAD ∩ loose primary lip-motion, with multi-face exclusion.

    Policy
    ------
    1. Score MuseTalk primary face once (batched VSDLM + dual-crop) → loose gate.
    2. Multi-face SCRFD tracking (detect stride) + VSDLM only on non-primary
       faces in multi-face frames → other-speaker exclusion / assign meta.
    3. VAD → speech segments; assign/split to visual speakers for meta.
    4. Primary lipsync frame iff:
         - MuseTalk face present
         - audio VAD active
         - primary face has *loose* VSDLM lip motion
         - no *other* face has a strict visual speaking turn at this frame

    All face tracking / lip-motion analysis is confined within shot boundaries
    (``shot_ids`` or auto-detected cuts).
    """
    from musetalk.utils.active_speaker import detect_shot_ids
    from musetalk.utils.vad_client import segments_to_frame_mask

    n = len(frame_list)
    fps = float(fps) if fps and fps > 0 else 25.0

    if shot_ids is None:
        shot_ids = detect_shot_ids(
            frame_list, hist_threshold=shot_cut_hist_threshold
        )
    elif len(shot_ids) != n:
        raise ValueError("shot_ids length must match frame_list")
    n_shots = (max(shot_ids) + 1) if shot_ids else 1

    frame_tracks, n_tracks = track_faces_across_frames(
        frame_list,
        detect_short_side=detect_short_side,
        min_face_area_ratio=min_face_area_ratio,
        detect_stride=detect_stride,
        shot_ids=shot_ids,
    )
    primary_id = match_primary_track_id(coord_list, frame_tracks)

    # One primary pass: open scores → loose gate (+ strict intervals for meta).
    primary_open = vsdlm_detector.score_open_probs(
        frame_list,
        coord_list,
        mouth_coord_list=mouth_coord_list,
        mouth_mar_list=mouth_mar_list,
        shot_ids=shot_ids,
    )
    primary_loose, _ = vsdlm_detector.speaking_mask_from_open_probs(
        primary_open,
        coord_list,
        fps=fps,
        min_speak_duration_sec=loose_min_speak_duration_sec,
        shot_ids=shot_ids,
    )
    primary_loose = primary_loose[:n]
    primary_strict, _ = vsdlm_detector.speaking_mask_from_open_probs(
        primary_open,
        coord_list,
        fps=fps,
        min_speak_duration_sec=vsdlm_detector.min_speak_duration_sec,
        shot_ids=shot_ids,
    )
    primary_intervals = (
        _mask_to_intervals(
            primary_strict,
            fps=fps,
            speaker_id=int(primary_id),
            shot_ids=shot_ids,
        )
        if primary_id is not None
        else []
    )

    # Other faces: only multi-face frames, skip primary (already scored).
    other_intervals = compute_visual_speaking_intervals(
        vsdlm_detector,
        frame_list,
        frame_tracks,
        fps=fps,
        shot_ids=shot_ids,
        skip_track_ids=[primary_id] if primary_id is not None else None,
        only_multiface_frames=True,
    )
    visual_intervals = sorted(
        primary_intervals + other_intervals,
        key=lambda x: (x[1], x[2], x[0]),
    )

    vad_segments = detect_voice_segments(audio_path, vad_url=vad_url)
    assigned = fuse_vad_with_vsdlm(
        vad_segments,
        visual_intervals,
        max_assign_gap_sec=max_assign_gap_sec,
    )
    vad_mask = segments_to_frame_mask(vad_segments, n, fps)

    # Exclude primary only while *another* face is visually speaking (strict turns).
    other_visual = [False] * n
    for sid, t0, t1 in other_intervals:
        i0 = max(0, int(t0 * fps))
        i1 = min(n, int(round(t1 * fps)))
        for i in range(i0, i1):
            other_visual[i] = True

    mask = [False] * n
    for i in range(n):
        if coord_list[i] == coord_placeholder:
            continue
        if not vad_mask[i]:
            continue
        if other_visual[i]:
            continue
        if not primary_loose[i]:
            continue
        mask[i] = True

    meta = {
        "n_face_tracks": n_tracks,
        "n_visual_intervals": len(visual_intervals),
        "n_vad_segments": len(vad_segments),
        "n_assigned_segments": len(assigned),
        "primary_track_id": primary_id,
        "visual_intervals": visual_intervals,
        "vad_segments": list(vad_segments),
        "assigned_segments": assigned,
        "loose_min_speak_duration_sec": loose_min_speak_duration_sec,
        "max_assign_gap_sec": max_assign_gap_sec,
        "primary_loose_frames": sum(primary_loose),
        "vad_frames": sum(vad_mask),
        "other_speaker_frames": sum(other_visual),
        "raw_speaking_frames": sum(mask),
        "n_shots": n_shots,
        "shot_ids": list(shot_ids) if shot_ids is not None else None,
        "detect_stride": int(detect_stride),
        "vsdlm_providers": list(getattr(vsdlm_detector, "providers", [])),
    }
    logger.info(
        "VAD∩VSDLM fusion: shots=%d tracks=%d visual_iv=%d vad=%d/%d assigned=%d "
        "primary=%s loose=%d other_visual=%d raw=%d/%d "
        "(detect_stride=%d, vsdlm=%s)",
        n_shots,
        n_tracks,
        len(visual_intervals),
        sum(vad_mask),
        n,
        len(assigned),
        primary_id,
        sum(primary_loose),
        sum(other_visual),
        sum(mask),
        n,
        int(detect_stride),
        ",".join(getattr(vsdlm_detector, "providers", []) or ["?"]),
    )
    return mask, meta
