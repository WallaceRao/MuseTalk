"""End-to-end verification of lipsync_close_mouth_on_silence.

Builds a "dub shorter than on-screen talking" case from task/215912:
  - video: first 30s (person keeps talking on screen the whole time)
  - dubbing audio: real speech only in 0-10s, 10-30s padded with silence
  - vad_segments (client-provided) mark the 0-10s speech only

Runs the engine twice (gate OFF vs ON) and prints speaking/lipsync frame
counts plus per-mode Close-mouth-on-silence log so the tail extension is
visible. Dumps tail frames (20s / 28s) from both outputs for visual check.

Usage:
  MUSETALK_LIPSYNC_BACKEND=latentsync \
    /home/ubuntu/miniconda3/envs/heng/bin/python scripts/verify_close_mouth.py
"""
import os
import sys
import json
import subprocess

os.environ.setdefault("MUSETALK_LIPSYNC_BACKEND", "latentsync")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musetalk.utils.vad_client import VadSegment
from musetalk.service.logging_setup import setup_logging
from musetalk.service.engine import ServiceConfig, MuseTalkEngine

VIDEO = os.path.abspath("cm_debug/cm_video30.mp4")
AUDIO = os.path.abspath("cm_debug/cm_dub30.wav")
OUT_OFF = os.path.abspath("outputs/cm_off.mp4")
OUT_ON = os.path.abspath("outputs/cm_on.mp4")

# Speech present in the padded dub (0-10s); 10-30s is silence.
VAD = [VadSegment(3.88, 5.28, None), VadSegment(7.0, 10.0, None)]


def load_cfg():
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from server import _load_service_config
    return _load_service_config()


def run(engine, out, close_on):
    cfg = engine.config
    cfg.lipsync_close_mouth_on_silence = close_on
    res = engine.run_lipsync(VIDEO, AUDIO, out, vad_segments=list(VAD))
    print(f"[close_on={close_on}] -> {json.dumps({k: res[k] for k in ('frame_count', 'speaking_frames', 'lipsync_frames')}, ensure_ascii=False)}")
    return res


def dump_frames(video, tag):
    for t in (20.0, 28.0):
        dst = os.path.abspath(f"cm_debug/{tag}_t{int(t)}.jpg")
        subprocess.run(
            f'ffmpeg -y -loglevel error -nostdin -ss {t} -i "{video}" '
            f'-frames:v 1 "{dst}"',
            shell=True,
        )
        print("dumped", dst, os.path.exists(dst))


def main():
    setup_logging()
    cfg = load_cfg()
    print(f"[cfg] backend={cfg.lipsync_backend} vad_fusion={cfg.use_vad_fusion} "
          f"use_vsdlm={cfg.use_vsdlm} close_default={cfg.lipsync_close_mouth_on_silence}")
    engine = MuseTalkEngine(cfg)
    run(engine, OUT_OFF, close_on=False)
    dump_frames(OUT_OFF, "cm_off")
    run(engine, OUT_ON, close_on=True)
    dump_frames(OUT_ON, "cm_on")
    print("DONE")


if __name__ == "__main__":
    main()
