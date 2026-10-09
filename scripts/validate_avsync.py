"""Offline validation: does LR-ASD AV-sync confidence separate on-screen
speech from off-screen speech on task/210991?

Mirrors production preprocessing (decode -> get_landmark_and_bbox) then scores
the tracked primary face with LR-ASD and aggregates per VAD segment.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musetalk.service.long_video import decode_video_frames
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.active_speaker import LRASDDetector
from musetalk.utils.vad_client import vad_span

D = "/opt/oss/wujiedub/video_translate/20261009/22/3"
VIDEO = f"{D}/input_videos/e02f89df39.mp4"
AUDIO = f"{D}/task/210991/translated_voice.wav"
INTER = f"{D}/task/210991/intermediate_result.json"
ASD_MODEL = "./third_party/LR-ASD/weight/finetuning_TalkSet.model"

# expected labels (sec) for readability
EXPECT = {
    0: "real", 1: "real", 2: "real", 3: "real", 4: "real", 5: "real", 6: "real",
    7: "OFFSCREEN",
    8: "?", 9: "pre-speech?", 10: "real", 11: "real", 12: "real", 13: "real",
    14: "noface", 15: "?", 16: "real", 17: "?", 18: "real", 19: "real",
}


def load_vad(path):
    j = json.load(open(path))
    sl = j.get("sentence_list") or j.get("sentences") or []
    out = []
    for s in sl:
        a, b = s.get("start"), s.get("end")
        if a is None or b is None:
            continue
        if a and a > 1000:
            a, b = a / 1000.0, b / 1000.0
        out.append((float(a), float(b)))
    return sorted(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    frames, fps = decode_video_frames(VIDEO)
    print(f"decoded {len(frames)} frames @ {fps:.3f}")
    coord_list, frame_list, mouth_coord, mouth_mar = get_landmark_and_bbox(
        upperbondrange=0,
        detect_stride=3,
        frames=frames,
        detect_short_side=720,
        min_face_area_ratio=0.003,
        return_mouth_coords=True,
    )
    n = len(frame_list)
    print(f"landmarks done: {n} frames")

    det = LRASDDetector(model_path=ASD_MODEL, device=device, threshold=0.0)
    av_mask, av_scores = det.compute_speaking_mask(
        audio_path=AUDIO,
        frame_list=frame_list,
        coord_list=coord_list,
        fps=fps,
    )
    scores = np.asarray([s if np.isfinite(s) else -99.0 for s in av_scores], dtype=np.float64)

    vad = load_vad(INTER)
    print(f"\n{'idx':>3} {'span':>14} {'dur':>5} {'face%':>6} "
          f"{'AVmed':>7} {'AVp75':>7} {'AV≥0%':>7} {'mask%':>6} {'exp':>10}")
    for i, (a, b) in enumerate(vad):
        f0 = max(0, int(round(a * fps)))
        f1 = min(n, int(round(b * fps)))
        if f0 >= f1:
            continue
        idxs = np.arange(f0, f1)
        face = np.array([coord_list[k] != coord_placeholder for k in idxs])
        # only score frames where a face is present
        sc = scores[idxs][face] if face.any() else np.array([-99.0])
        avmed = float(np.median(sc)) if len(sc) else -99
        avp75 = float(np.percentile(sc, 75)) if len(sc) else -99
        ge0 = float((sc >= 0.0).mean() * 100) if len(sc) else 0
        maskp = float(np.mean([av_mask[k] for k in idxs]) * 100)
        facep = float(face.mean() * 100)
        print(f"{i:>3} {a:6.2f}-{b:6.2f} {b-a:5.2f} {facep:6.0f} "
              f"{avmed:7.2f} {avp75:7.2f} {ge0:7.0f} {maskp:6.0f} {EXPECT.get(i,''):>10}")


if __name__ == "__main__":
    main()
