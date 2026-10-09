"""Sweep (mar_threshold, shot_fill_ratio) on the real fused pipeline to trim
listener over-fill (seg8) without dropping real talkers (seg11/12/13)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from musetalk.service.long_video import decode_video_frames
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.active_speaker import VSDLMDetector
import musetalk.utils.vad_vsdlm_fusion as F

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"

def load_vad(path):
    j = json.load(open(path)); out = []
    for s in (j.get("sentence_list") or []):
        a, b = s.get("start"), s.get("end")
        if a is None: continue
        if a > 1000: a, b = a/1000.0, b/1000.0
        out.append((float(a), float(b)))
    return sorted(out)

frames, fps = decode_video_frames(VIDEO)
coord_list, frame_list, mouth_coord, mouth_mar = get_landmark_and_bbox(
    upperbondrange=0, detect_stride=3, frames=frames,
    detect_short_side=720, min_face_area_ratio=0.003, return_mouth_coords=True)
n = len(frame_list); vad = load_vad(INTER)
vd = VSDLMDetector(model_path="./third_party/VSDLM/vsdlm_m.onnx",
    open_threshold=0.15, activity_threshold=0.12, activity_window=4,
    min_speak_duration_sec=0.5, mar_open_threshold=0.12, mar_activity_threshold=0.04,
    temporal_max_radius=2, soft_closed_mar=0.12, batch_size=64)

TALK = [1,2,3,5,6,10,11,12,13,16,18]
LISTEN = [8,15,19]
def seg_drive(mask):
    out = {}
    for i,(a,b) in enumerate(vad):
        idx=np.arange(max(0,int(round(a*fps))),min(n,int(round(b*fps))))
        out[i]=float(np.mean([bool(mask[k]) for k in idx])*100) if len(idx) else 0
    return out

for mar_thr, ratio in [(0.12,0.35),(0.12,0.45),(0.15,0.40),(0.15,0.45),(0.18,0.45)]:
    mask, meta = F.build_fused_speaking_mask(
        vd, frame_list, coord_list, AUDIO, fps=fps, vad_segments=vad,
        detect_short_side=720, min_face_area_ratio=0.003, detect_stride=3,
        loose_min_speak_duration_sec=0.2, max_assign_gap_sec=0.5,
        mouth_coord_list=mouth_coord, mouth_mar_list=mouth_mar,
        speak_open_threshold=0.5, speak_mar_threshold=mar_thr, shot_fill_ratio=ratio)
    d = seg_drive(mask)
    tmin = min(d[i] for i in TALK); tavg = np.mean([d[i] for i in TALK])
    lmax = max(d[i] for i in LISTEN)
    print(f"mar={mar_thr:.2f} ratio={ratio:.2f} | TALK min={tmin:3.0f} avg={tavg:3.0f} | "
          f"LISTEN max={lmax:3.0f} | seg7={d[7]:.0f} seg9={d[9]:.0f} seg8={d[8]:.0f}")
