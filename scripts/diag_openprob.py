"""Diagnostic: per-frame VSDLM open-prob on the PRIMARY tracked face + render
coord/mouth boxes for seg2 (real talker, wrongly suppressed) and seg7
(off-screen listener, wrongly driven) to see what the detector actually sees."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from musetalk.service.long_video import decode_video_frames
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.active_speaker import VSDLMDetector

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"
OUT = "asd_debug/av_ground"
os.makedirs(OUT, exist_ok=True)

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

vd = VSDLMDetector(model_path="./third_party/VSDLM/vsdlm_m.onnx",
    open_threshold=0.15, activity_threshold=0.12, activity_window=4,
    min_speak_duration_sec=0.5, mar_open_threshold=0.12, mar_activity_threshold=0.04,
    temporal_max_radius=2, soft_closed_mar=0.12, batch_size=64)
op = np.asarray(vd.score_open_probs(frame_list, coord_list,
    mouth_coord_list=mouth_coord, mouth_mar_list=mouth_mar), dtype=np.float64)

vad = load_vad(INTER)
mar = np.asarray([m if (m is not None and np.isfinite(m)) else np.nan for m in mouth_mar], dtype=np.float64)
print("Per-VAD-segment PRIMARY open-prob + MAR stats:")
print(f"{'idx':>3} {'span':>14} {'face%':>6} {'op_mean':>7} {'op_std':>6} {'op>0.5%':>7} {'mar_mean':>8} {'mar_std':>7} {'mar_rng':>7}")
for i, (a, b) in enumerate(vad):
    f0, f1 = int(round(a*fps)), int(round(b*fps))
    idxs = np.arange(max(0,f0), min(n,f1))
    if len(idxs)==0: continue
    face = np.array([coord_list[k] != coord_placeholder for k in idxs])
    o = op[idxs]
    m = mar[idxs]
    m = m[np.isfinite(m)]
    mm = float(np.mean(m)) if len(m) else float('nan')
    ms = float(np.std(m)) if len(m) else float('nan')
    mr = float(np.ptp(m)) if len(m) else float('nan')
    # candidate rule fractions
    faceidx = idxs[face]
    o2 = op[faceidx]; m2 = mar[faceidx]
    fr_op = float((o2 >= 0.5).mean() * 100) if len(o2) else 0
    fr_mar = float(((m2 >= 0.12) & np.isfinite(m2)).mean() * 100) if len(m2) else 0
    fr_both = float(((o2 >= 0.5) & (m2 >= 0.12) & np.isfinite(m2)).mean() * 100) if len(o2) else 0
    print(f"{i:>3} {a:6.2f}-{b:6.2f} {face.mean()*100:6.0f} {np.nanmean(o):7.2f} "
          f"{np.nanstd(o):6.2f} {(o>0.5).mean()*100:7.0f} {mm:8.3f} {ms:7.3f} {mr:7.3f}"
          f" | op%{fr_op:3.0f} mar%{fr_mar:3.0f} BOTH%{fr_both:3.0f}")

def render(seg_idx, name):
    a, b = vad[seg_idx]
    ts = np.linspace(a+0.1, b-0.1, 4)
    tiles = []
    for t in ts:
        k = min(n-1, max(0, int(round(t*fps))))
        im = frame_list[k].copy()
        cx = coord_list[k]
        if cx != coord_placeholder:
            x1,y1,x2,y2 = map(int, cx)
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            mc = mouth_coord[k] if k < len(mouth_coord) else None
            if mc is not None and mc != coord_placeholder:
                mx1,my1,mx2,my2 = map(int, mc)
                cv2.rectangle(im,(mx1,my1),(mx2,my2),(0,0,255),2)
        cv2.putText(im, f"seg{seg_idx} t={t:.1f} op={op[k]:.2f}",
                    (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        h,w = im.shape[:2]; im = cv2.resize(im,(int(w*260/h),260))
        tiles.append(im)
    grid = np.hstack(tiles)
    p = f"{OUT}/{name}.jpg"
    cv2.imwrite(p, grid); print("wrote", p)

render(2, "seg2_talker")
render(7, "seg7_offscreen")
render(6, "seg6_talker")
render(8, "seg8_listener")
render(9, "seg9_maybe")
render(16, "seg16_maybe")
