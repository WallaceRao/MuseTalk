"""Render a video+audio pair through the PRODUCTION service path (MuseTalkEngine).

This is the "our code" counterpart of the official TBDub script runs: same input,
same gates, same config the systemd service uses (only MUSETALK_LIPSYNC_BACKEND is
forced if the caller did not set it), so a difference here can only come from our
own pipeline - not from the model.

Usage:
  python scripts/run_lipsync_service.py [video] [audio] [output]

The production launcher's ``export MUSETALK_*`` lines are imported first (an env the
caller already set always wins), so a run here is gated exactly like the live service
instead of silently falling back to the ServiceConfig defaults.

Defaults to the 12s example that was validated against the official script:
  asd_debug/in_en_12s.mp4 + asd_debug/in_en_12s.wav -> outputs/service_12s.mp4
"""
import json
import os
import re
import sys
import time

PROD_LAUNCHER = "/home/ubuntu/system_startup/muse.sh"


def _parse_export_value(raw: str) -> str:
    """Resolve one shell ``export`` right-hand side to a plain string.

    Handles the three forms the launcher actually uses: quoted values, trailing
    inline comments, and ``${VAR:-default}`` (the environment here never pre-sets
    those vars, so the default is the value the running service sees).
    """
    raw = raw.strip()
    if raw[:1] in ('"', "'"):
        quote, body = raw[0], raw[1:]
        end = body.find(quote)
        if end >= 0:
            raw = body[:end]  # quoted: the inline-comment rule does not apply
        else:
            raw = body
    else:
        # unquote, then drop a trailing  # comment  (only outside quotes)
        raw = raw.split(" #", 1)[0].strip()
    if raw.startswith("${") and raw.endswith("}") and ":-" in raw:
        raw = raw.split(":-", 1)[1][:-1]
    return raw.strip('"').strip("'")


def load_production_env(path: str = PROD_LAUNCHER) -> int:
    """Apply 'export KEY=VALUE' lines from the production launcher, without running it."""
    try:
        text = open(path, "r").read()
    except OSError as exc:
        print("[env] %s not readable (%s); using defaults" % (path, exc), flush=True)
        return 0
    applied = 0
    for key, raw in re.findall(r"^\s*export\s+([A-Z0-9_]+)=(.*)$", text, re.M):
        if key in os.environ:  # the caller's own value wins
            continue
        os.environ[key] = _parse_export_value(raw)
        applied += 1
    print("[env] %d vars from %s" % (applied, path), flush=True)
    return applied


os.environ.setdefault("MUSETALK_LIPSYNC_BACKEND", "tbdub")
load_production_env()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from musetalk.service.logging_setup import setup_logging
from musetalk.service.engine import MuseTalkEngine

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "asd_debug", "in_en_12s.mp4"
)
AUDIO = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, "asd_debug", "in_en_12s.wav"
)
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    ROOT, "outputs", "service_12s.mp4"
)


def main():
    setup_logging()
    from server import _load_service_config

    cfg = _load_service_config()
    print(
        "[cfg] backend=%s close_mouth_on_silence=%s stream_composite=%s "
        "full_speaking_shots=%s"
        % (
            cfg.lipsync_backend,
            cfg.lipsync_close_mouth_on_silence,
            cfg.stream_composite,
            cfg.lipsync_full_speaking_shots,
        ),
        flush=True,
    )
    print("[cfg] video=%s audio=%s out=%s" % (VIDEO, AUDIO, OUT), flush=True)
    engine = MuseTalkEngine(cfg)
    t0 = time.time()
    res = engine.run_lipsync(VIDEO, AUDIO, OUT)
    print(
        "[run] %s in %.0fs %s"
        % (
            OUT,
            time.time() - t0,
            json.dumps(
                {
                    k: res[k]
                    for k in ("frame_count", "speaking_frames", "lipsync_frames")
                    if k in res
                },
                ensure_ascii=False,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
