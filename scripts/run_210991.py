"""End-to-end run of task/210991 through the REAL engine (all downstream
gates: expand/yaw/occlusion/gender) with the new op+MAR+shot-fill gate.
Dumps the final per-frame speaking mask for analysis."""
import os
import sys
import json
import subprocess

os.environ.setdefault("MUSETALK_LIPSYNC_BACKEND", "latentsync")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musetalk.utils.vad_client import VadSegment
from musetalk.service.logging_setup import setup_logging
from musetalk.service.engine import ServiceConfig, MuseTalkEngine

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"
OUTPUT = os.path.abspath("verify_210991_fast.mp4")
_gres = int(os.environ.get("GEN_RES", "0") or 0)
if _gres:
    OUTPUT = os.path.abspath(f"verify_210991_res{_gres}.mp4")
_out = os.environ.get("RUN_OUTPUT")
if _out:
    OUTPUT = os.path.abspath(_out)


def build_vad(path):
    j = json.load(open(path))
    sl = j.get("sentence_list") or j.get("sentences") or []
    out = []
    for s in sl:
        a, b = s.get("start"), s.get("end")
        if a is None or b is None:
            continue
        if a > 1000:
            a, b = a / 1000.0, b / 1000.0
        if b > a:
            out.append(VadSegment(float(a), float(b), None))
    return sorted(out, key=lambda x: x.start)


def main():
    setup_logging()
    vad = build_vad(INTER)
    print(f"vad segments: {len(vad)}")
    vpath, apath = VIDEO, AUDIO
    _maxsec = float(os.environ.get("MAX_SECONDS", "0") or 0)
    if _maxsec > 0:
        os.makedirs("asd_debug", exist_ok=True)
        vtrim = os.path.abspath(f"asd_debug/in_{int(_maxsec)}s.mp4")
        atrim = os.path.abspath(f"asd_debug/in_{int(_maxsec)}s.wav")
        subprocess.run(
            f'ffmpeg -y -loglevel error -nostdin -i "{VIDEO}" -t {_maxsec} '
            f'-c:v libx264 -preset ultrafast -crf 16 -pix_fmt yuv420p "{vtrim}"',
            shell=True,
        )
        subprocess.run(
            f'ffmpeg -y -loglevel error -nostdin -i "{AUDIO}" -t {_maxsec} "{atrim}"',
            shell=True,
        )
        vad = [s for s in vad if s.start < _maxsec]
        vpath, apath = vtrim, atrim
        print(f"[trim] first {int(_maxsec)}s -> {len(vad)} vad segs, {vtrim}")
    cfg = ServiceConfig()
    cfg.lipsync_backend = os.environ["MUSETALK_LIPSYNC_BACKEND"]
    _stream = os.environ.get("MUSETALK_STREAM_COMPOSITE")
    if _stream is not None:
        cfg.stream_composite = _stream.strip().lower() in {"1", "true", "yes", "on"}
    _guid = os.environ.get("MUSETALK_LATENTSYNC_GUIDANCE")
    if _guid is not None:
        cfg.latentsync_guidance_scale = float(_guid)
    _steps = os.environ.get("MUSETALK_LATENTSYNC_STEPS")
    if _steps is not None:
        cfg.latentsync_inference_steps = max(1, int(_steps))
    _dc = os.environ.get("MUSETALK_LATENTSYNC_DEEPCACHE")
    if _dc is not None:
        cfg.latentsync_enable_deepcache = _dc.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if _gres:
        cfg.latentsync_gen_resolution = _gres
    _stride = os.environ.get("MUSETALK_BBOX_DETECT_STRIDE")
    if _stride is not None:
        cfg.bbox_detect_stride = max(0, int(_stride))
    _feather = os.environ.get("MUSETALK_PASTE_FEATHER_RATIO")
    if _feather is not None:
        cfg.latentsync_paste_feather_ratio = float(_feather)
    _expand = os.environ.get("MUSETALK_PASTE_EXPAND_RATIO")
    if _expand is not None:
        cfg.latentsync_paste_expand_ratio = float(_expand)
    _mouth = os.environ.get("MUSETALK_LATENTSYNC_PASTE_MOUTH_ONLY")
    if _mouth is not None:
        cfg.latentsync_paste_mouth_only = _mouth.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    _mexp = os.environ.get("MUSETALK_PASTE_MOUTH_EXPAND_RATIO")
    if _mexp is not None:
        cfg.latentsync_paste_mouth_expand_ratio = float(_mexp)
    print(
        f"[cfg] bbox_detect_stride={cfg.bbox_detect_stride} "
        f"feather={cfg.latentsync_paste_feather_ratio} "
        f"expand={cfg.latentsync_paste_expand_ratio} "
        f"mouth_only={cfg.latentsync_paste_mouth_only} "
        f"mouth_expand={cfg.latentsync_paste_mouth_expand_ratio} "
        f"steps={cfg.latentsync_inference_steps} -> {OUTPUT}"
    )
    engine = MuseTalkEngine(cfg)
    result = engine.run_lipsync(vpath, apath, OUTPUT, vad_segments=vad)
    print("RESULT:", json.dumps({k: result[k] for k in
          ("output_path", "frame_count", "speaking_frames", "lipsync_frames")
          if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
