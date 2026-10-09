"""End-to-end run of task/210991 through the REAL engine (all downstream
gates: expand/yaw/occlusion/gender) with the new op+MAR+shot-fill gate.
Dumps the final per-frame speaking mask for analysis."""
import os
import sys
import json

os.environ.setdefault("MUSETALK_LIPSYNC_BACKEND", "latentsync")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musetalk.utils.vad_client import VadSegment
from musetalk.service.logging_setup import setup_logging
from musetalk.service.engine import ServiceConfig, MuseTalkEngine

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"
OUTPUT = os.path.abspath("verify_210991_16.mp4")


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
    cfg = ServiceConfig()
    cfg.lipsync_backend = os.environ["MUSETALK_LIPSYNC_BACKEND"]
    engine = MuseTalkEngine(cfg)
    result = engine.run_lipsync(VIDEO, AUDIO, OUTPUT, vad_segments=vad)
    print("RESULT:", json.dumps({k: result[k] for k in
          ("output_path", "frame_count", "speaking_frames", "lipsync_frames")
          if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
