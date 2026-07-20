"""Fuse VAD speech segments with multi-face VSDLM speaking intervals.

Pipeline
--------
1. Track faces across frames (SCRFD + IoU) and run VSDLM lip-motion per face.
2. Collect contiguous visual speaking intervals per track (``speaker_id``).
3. Run VAD to get audio speech segments.
4. Assign each VAD segment to the nearest VSDLM speaker in time.
5. If one VAD segment overlaps multiple VSDLM speakers, split it at speaker
   switch points inside the segment.
6. Per frame, pick the face with strongest recent lip-open as active speaker
   and gate lipsync as VAD ∩ that face's loose VSDLM lip motion.
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
    min_speak_duration_sec: float | None = None,
) -> List[VisualInterval]:
    """Run batched VSDLM per tracked face and return speaking intervals.

    Lip-motion activity, min-speak runs, and emitted intervals are all confined
    to a single shot when ``shot_ids`` is provided. Open scores at shot cuts /
    hard bbox jumps are discarded.

    When ``only_multiface_frames`` is True, faces are scored only on frames that
    contain 2+ tracks (enough for other-speaker exclusion). ``skip_track_ids``
    omits already-scored primary faces.

    ``min_speak_duration_sec`` overrides the detector default when set (e.g. loose
    gate uses 0.2s while strict turns keep 0.5s).
    """
    open_probs, coord_streams = score_track_open_probs(
        vsdlm_detector,
        frame_list,
        frame_tracks,
        shot_ids=shot_ids,
        skip_track_ids=skip_track_ids,
        only_multiface_frames=only_multiface_frames,
    )
    return intervals_from_track_open_probs(
        vsdlm_detector,
        open_probs,
        coord_streams,
        fps=fps,
        shot_ids=shot_ids,
        min_speak_duration_sec=min_speak_duration_sec,
    )


def score_track_open_probs(
    vsdlm_detector,
    frame_list: Sequence[np.ndarray],
    frame_tracks: Sequence[Dict[int, BBox]],
    *,
    shot_ids: Sequence[int] | None = None,
    skip_track_ids: Sequence[int] | None = None,
    only_multiface_frames: bool = False,
) -> Tuple[Dict[int, List[float]], Dict[int, List[BBox]]]:
    """Batch-score VSDLM open probs for every tracked face."""
    n = len(frame_list)
    if shot_ids is not None and len(shot_ids) != n:
        raise ValueError("shot_ids length must match frame_list")
    skip = {int(x) for x in (skip_track_ids or ())}
    track_ids = sorted(
        {tid for ft in frame_tracks for tid in ft if int(tid) not in skip}
    )
    open_probs: Dict[int, List[float]] = {
        tid: [float("nan")] * n for tid in track_ids
    }
    coord_streams: Dict[int, List[BBox]] = {
        tid: [coord_placeholder] * n for tid in track_ids
    }
    if not track_ids:
        return open_probs, coord_streams

    crop_keys: List[Tuple[int, int]] = []
    crops: List[np.ndarray] = []
    for fi, frame in enumerate(frame_list):
        tracks = frame_tracks[fi]
        if only_multiface_frames and len(tracks) < 2:
            continue
        for tid, bbox in tracks.items():
            if int(tid) in skip or tid not in coord_streams:
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

    for tid in track_ids:
        open_probs[tid] = vsdlm_detector._invalidate_unstable_open_probs(
            open_probs[tid],
            coord_streams[tid],
            shot_ids=shot_ids,
        )
    return open_probs, coord_streams


def intervals_from_track_open_probs(
    vsdlm_detector,
    open_probs: Dict[int, List[float]],
    coord_streams: Dict[int, List[BBox]],
    *,
    fps: float,
    shot_ids: Sequence[int] | None = None,
    min_speak_duration_sec: float | None = None,
) -> List[VisualInterval]:
    """Build per-track speaking intervals from precomputed open probs."""
    n = 0
    if open_probs:
        n = len(next(iter(open_probs.values())))
    fps = float(fps) if fps and fps > 0 else 25.0
    min_speak = (
        float(min_speak_duration_sec)
        if min_speak_duration_sec is not None
        else float(vsdlm_detector.min_speak_duration_sec)
    )
    intervals: List[VisualInterval] = []
    for tid, probs in open_probs.items():
        coords = coord_streams.get(tid) or [coord_placeholder] * n
        mask, _ = vsdlm_detector.speaking_mask_from_open_probs(
            probs,
            coords,
            fps=fps,
            min_speak_duration_sec=min_speak,
            shot_ids=shot_ids,
        )
        intervals.extend(
            _mask_to_intervals(
                mask, fps=fps, speaker_id=int(tid), shot_ids=shot_ids
            )
        )
    intervals.sort(key=lambda x: (x[1], x[2], x[0]))
    return intervals


def any_track_speaking_mask(
    vsdlm_detector,
    open_probs: Dict[int, List[float]],
    coord_streams: Dict[int, List[BBox]],
    *,
    fps: float,
    min_speak_duration_sec: float,
    shot_ids: Sequence[int] | None = None,
) -> List[bool]:
    """OR of per-track loose/strict speaking masks."""
    n = 0
    if open_probs:
        n = len(next(iter(open_probs.values())))
    out = [False] * n
    for tid, probs in open_probs.items():
        coords = coord_streams.get(tid) or [coord_placeholder] * n
        mask, _ = vsdlm_detector.speaking_mask_from_open_probs(
            probs,
            coords,
            fps=fps,
            min_speak_duration_sec=min_speak_duration_sec,
            shot_ids=shot_ids,
        )
        for i, v in enumerate(mask[:n]):
            if v:
                out[i] = True
    return out


def merge_primary_open_into_track(
    open_probs: Dict[int, List[float]],
    coord_streams: Dict[int, List[BBox]],
    primary_id: Optional[int],
    primary_open: Sequence[float],
    primary_coords: Sequence[BBox],
) -> None:
    """Prefer MuseTalk dual-crop open scores on the matched primary track."""
    if primary_id is None or primary_id not in open_probs:
        return
    n = len(open_probs[primary_id])
    for i in range(min(n, len(primary_open), len(primary_coords))):
        if primary_coords[i] == coord_placeholder:
            continue
        if not np.isfinite(primary_open[i]):
            continue
        # Keep track bbox if present; overlay higher-quality open.
        open_probs[primary_id][i] = float(primary_open[i])
        if coord_streams[primary_id][i] == coord_placeholder:
            coord_streams[primary_id][i] = primary_coords[i]


def pick_active_speaker_track(
    frame_tracks: Sequence[Dict[int, BBox]],
    open_probs: Dict[int, List[float]],
    *,
    shot_ids: Sequence[int] | None = None,
    activity_window: int = 4,
) -> List[Optional[int]]:
    """Per frame, pick the face with strongest recent lip-open activity.

    Used so multi-face shots follow the talking person (e.g. adult behind child)
    instead of MuseTalk's size/center primary track.
    """
    n = len(frame_tracks)
    win = max(1, int(activity_window))
    out: List[Optional[int]] = [None] * n
    for i in range(n):
        tracks = frame_tracks[i]
        if not tracks:
            continue
        cur_shot = int(shot_ids[i]) if shot_ids is not None else None
        best_tid = None
        best_key = None
        for tid in tracks:
            probs = open_probs.get(tid)
            if not probs:
                continue
            finite = []
            for j in range(max(0, i - win), min(n, i + win + 1)):
                if shot_ids is not None and int(shot_ids[j]) != cur_shot:
                    continue
                if tid not in frame_tracks[j]:
                    continue
                v = probs[j] if j < len(probs) else float("nan")
                if np.isfinite(v):
                    finite.append(float(v))
            if not finite:
                continue
            mean_open = float(np.mean(finite))
            activity = float(np.std(finite)) if len(finite) >= 2 else 0.0
            # Prefer lip motion (std), then mean open, then current open.
            cur = float(probs[i]) if i < len(probs) and np.isfinite(probs[i]) else 0.0
            key = (activity, mean_open, cur)
            if best_key is None or key > best_key:
                best_key = key
                best_tid = int(tid)
        out[i] = best_tid
    return out


def open_presence_mask(
    vsdlm_detector,
    open_probs: Sequence[float],
    coord_list: Sequence[BBox],
    *,
    fps: float,
    min_speak_duration_sec: float,
    shot_ids: Sequence[int] | None = None,
    activity_window: int | None = None,
) -> List[bool]:
    """Sustained mouth-open evidence (no activity/std requirement).

    Complements the activity gate: when lips stay fully open (open≈1), std
    collapses and activity-based VSDLM falsely drops. Under VAD this open
    presence is still a useful visual cue for the selected speaker.
    """
    n = len(coord_list)
    if len(open_probs) != n:
        raise ValueError("open_probs length must match coord_list")
    win = max(1, int(activity_window or vsdlm_detector.activity_window))
    open_thr = float(vsdlm_detector.open_threshold)
    raw: List[bool] = []
    for i in range(n):
        if coord_list[i] == coord_placeholder or not np.isfinite(open_probs[i]):
            raw.append(False)
            continue
        if float(open_probs[i]) < open_thr:
            raw.append(False)
            continue
        cur_shot = int(shot_ids[i]) if shot_ids is not None else None
        finite = []
        for j in range(max(0, i - win), min(n, i + win + 1)):
            if shot_ids is not None and int(shot_ids[j]) != cur_shot:
                continue
            if coord_list[j] == coord_placeholder:
                continue
            v = open_probs[j]
            if np.isfinite(v):
                finite.append(float(v))
        if len(finite) < 2:
            raw.append(False)
            continue
        raw.append(float(np.mean(finite)) >= open_thr)

    fps = float(fps) if fps and fps > 0 else 25.0
    duration = max(0.0, float(min_speak_duration_sec))
    min_frames = max(1, int(round(duration * fps))) if duration > 0 else 1
    return vsdlm_detector._keep_runs_at_least(raw, min_frames, shot_ids=shot_ids)


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
    vad_segments: Sequence[Tuple[float, float]] | None = None,
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
    """VAD ∩ lip-motion on the MuseTalk primary face.

    Policy
    ------
    1. Track all faces (SCRFD) within shot boundaries (for diagnostics / assign).
    2. Score VSDLM open on every track; overlay MuseTalk dual-crop on primary.
    3. Pick an "active speaker" by lip activity for logging only.
    4. Frame is speaking iff:
         - audio VAD active
         - MuseTalk **primary** face is present
         - primary has lip **motion** (open-prob std, or MAR std when the
           mouth stays open and VSDLM open saturates). Static open never passes.
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

    # Score all tracks once, then derive loose/strict intervals.
    open_probs, coord_streams = score_track_open_probs(
        vsdlm_detector,
        frame_list,
        frame_tracks,
        shot_ids=shot_ids,
        only_multiface_frames=False,
    )

    # MuseTalk primary: dual-crop + MAR soft-closed (higher quality on that face).
    primary_open = vsdlm_detector.score_open_probs(
        frame_list,
        coord_list,
        mouth_coord_list=mouth_coord_list,
        mouth_mar_list=mouth_mar_list,
        shot_ids=shot_ids,
    )
    merge_primary_open_into_track(
        open_probs,
        coord_streams,
        primary_id,
        primary_open,
        coord_list,
    )
    # If MuseTalk primary never matched a SCRFD track, keep its scores under a
    # synthetic id so single-face / unmatched cases still gate correctly.
    synthetic_primary_id: Optional[int] = None
    if primary_id is None and any(
        c != coord_placeholder for c in coord_list
    ):
        synthetic_primary_id = -1
        open_probs[synthetic_primary_id] = [
            float(x) if np.isfinite(x) else float("nan") for x in primary_open
        ]
        coord_streams[synthetic_primary_id] = list(coord_list)

    active_speaker = pick_active_speaker_track(
        frame_tracks,
        open_probs,
        shot_ids=shot_ids,
        activity_window=vsdlm_detector.activity_window,
    )

    loose_intervals = intervals_from_track_open_probs(
        vsdlm_detector,
        open_probs,
        coord_streams,
        fps=fps,
        shot_ids=shot_ids,
        min_speak_duration_sec=loose_min_speak_duration_sec,
    )
    strict_intervals = intervals_from_track_open_probs(
        vsdlm_detector,
        open_probs,
        coord_streams,
        fps=fps,
        shot_ids=shot_ids,
        min_speak_duration_sec=vsdlm_detector.min_speak_duration_sec,
    )
    visual_intervals = sorted(
        loose_intervals,
        key=lambda x: (x[1], x[2], x[0]),
    )

    fallback_primary = (
        primary_id if primary_id is not None else synthetic_primary_id
    )

    # Primary visual gate: require lip motion. Sustained-open mouths may use
    # softer open-std or landmark MAR std (open often saturates at ~1).
    primary_visual = [False] * n
    if fallback_primary is not None and fallback_primary in open_probs:
        p_coords = coord_streams.get(fallback_primary) or [coord_placeholder] * n
        p_probs = open_probs[fallback_primary]
        primary_visual, _ = vsdlm_detector.speaking_mask_from_open_probs(
            p_probs,
            p_coords,
            fps=fps,
            min_speak_duration_sec=loose_min_speak_duration_sec,
            shot_ids=shot_ids,
            soft_open_mean=0.75,
            soft_activity_threshold=0.05,
            mouth_mar_list=mouth_mar_list,
            mar_activity_threshold=vsdlm_detector.mar_activity_threshold,
        )
        primary_visual = list(primary_visual[:n])

    vad_segments = detect_voice_segments(
        audio_path,
        vad_url=vad_url,
        vad_segments=vad_segments,
    )
    assigned = fuse_vad_with_vsdlm(
        vad_segments,
        visual_intervals,
        max_assign_gap_sec=max_assign_gap_sec,
    )
    vad_mask = segments_to_frame_mask(vad_segments, n, fps)

    mask = [False] * n
    primary_loose_frames = 0
    for i in range(n):
        if not vad_mask[i]:
            continue
        if coord_list[i] == coord_placeholder:
            continue
        if not primary_visual[i]:
            continue
        mask[i] = True
        primary_loose_frames += 1

    # Meta: other-face strict turns vs MuseTalk primary (diagnostics only).
    other_visual = [False] * n
    if primary_id is not None:
        for sid, t0, t1 in strict_intervals:
            if int(sid) == int(primary_id):
                continue
            i0 = max(0, int(t0 * fps))
            i1 = min(n, int(round(t1 * fps)))
            for i in range(i0, i1):
                other_visual[i] = True

    # How often activity-picker disagrees with primary on speaking frames.
    switched = 0
    for i, v in enumerate(mask):
        if not v:
            continue
        if active_speaker[i] is not None and fallback_primary is not None:
            if int(active_speaker[i]) != int(fallback_primary):
                switched += 1

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
        "primary_loose_frames": primary_loose_frames,
        "active_speaker_loose_frames": primary_loose_frames,
        "active_speaker_switched_frames": switched,
        "vad_frames": sum(vad_mask),
        "other_speaker_frames": sum(other_visual),
        "raw_speaking_frames": sum(mask),
        "n_shots": n_shots,
        "shot_ids": list(shot_ids) if shot_ids is not None else None,
        "detect_stride": int(detect_stride),
        "vsdlm_providers": list(getattr(vsdlm_detector, "providers", [])),
        "speaker_select": "primary",
    }
    logger.info(
        "VAD∩VSDLM fusion: shots=%d tracks=%d visual_iv=%d vad=%d/%d assigned=%d "
        "primary=%s primary_loose=%d picker_disagree=%d other_visual=%d raw=%d/%d "
        "(detect_stride=%d, select=primary, vsdlm=%s)",
        n_shots,
        n_tracks,
        len(visual_intervals),
        sum(vad_mask),
        n,
        len(assigned),
        primary_id,
        primary_loose_frames,
        switched,
        sum(other_visual),
        sum(mask),
        n,
        int(detect_stride),
        ",".join(getattr(vsdlm_detector, "providers", []) or ["?"]),
    )
    return mask, meta
