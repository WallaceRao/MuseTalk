"""Offline: inspect VSDLM speaker attribution (assigned) + per-track open-prob
stats per VAD segment, to see if utterance-level attribution separates the
on-screen talker from the off-screen listener (seg7)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from musetalk.service.long_video import decode_video_frames
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.active_speaker import VSDLMDetector, detect_shot_ids
from musetalk.utils.vad_client import vad_span
import musetalk.utils.vad_vsdlm_fusion as F

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"

def load_vad(path):
    j = json.load(open(path)); sl = j.get("sentence_list") or j.get("sentences") or []
    out = []
    for s in sl:
        a, b = s.get("start"), s.get("end")
        if a is None: continue
        if a and a > 1000: a, b = a/1000.0, b/1000.0
        out.append((float(a), float(b)))
    return sorted(out)

frames, fps = decode_video_frames(VIDEO)
coord_list, frame_list, mouth_coord, mouth_mar = get_landmark_and_bbox(
    upperbondrange=0, detect_stride=3, frames=frames,
    detect_short_side=720, min_face_area_ratio=0.003, return_mouth_coords=True)
n = len(frame_list)
vad = load_vad(INTER)

vd = VSDLMDetector(model_path="./third_party/VSDLM/vsdlm_m.onnx",
    open_threshold=0.15, activity_threshold=0.12, activity_window=4,
    min_speak_duration_sec=0.5, mar_open_threshold=0.12, mar_activity_threshold=0.04,
    temporal_max_radius=2, soft_closed_mar=0.12, batch_size=64)

mask, meta = F.build_fused_speaking_mask(
    vd, frame_list, coord_list, AUDIO, fps=fps, vad_segments=vad,
    detect_short_side=720, min_face_area_ratio=0.003, detect_stride=3,
    loose_min_speak_duration_sec=0.2, max_assign_gap_sec=0.5,
    mouth_coord_list=mouth_coord, mouth_mar_list=mouth_mar,
    speak_open_threshold=0.5, speak_mar_threshold=0.12)

assigned = meta["assigned_segments"]  # (t0,t1,sid)
vi = meta["visual_intervals"]         # (sid,t0,t1)
primary = meta["primary_track_id"]
print("primary_track_id:", primary, " n_visual_intervals:", len(vi))
print("\nNEW fused gate (op>=0.5 & mar>=0.12) -> per-VAD-segment drive%:")
print(f"{'idx':>3} {'span':>14} {'drive%':>7}")
for i, (a, b) in enumerate(vad):
    f0, f1 = int(round(a*fps)), int(round(b*fps))
    idxs = range(max(0,f0), min(n,f1))
    if not idxs: continue
    lst = list(idxs)
    dp = float(np.mean([bool(mask[k]) for k in lst]) * 100)
    print(f"{i:>3} {a:6.2f}-{b:6.2f} {dp:7.0f}")
print("\nPer-VAD-segment attribution (sid=-1 => off-screen/narration):")
print(f"{'idx':>3} {'span':>14} {'sids(overlap)':>28} {'primary_in?':>11}")
for i, (a, b) in enumerate(vad):
    sids = sorted({int(s) for s, t0, t1 in vi if min(t1, b) - max(t0, a) > 0.05})
    # which assigned pieces cover this vad seg
    ap = [(round(t0,1), round(t1,1), int(s)) for t0, t1, s in assigned if min(t1, b) - max(t0, a) > 0.05]
    prim = "YES" if (primary is not None and int(primary) in sids) else "no"
    print(f"{i:>3} {a:6.2f}-{b:6.2f} {str(sids):>28} {prim:>11}  assigned={ap}")
