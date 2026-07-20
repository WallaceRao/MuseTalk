#!/usr/bin/env python3
"""Call the MuseTalk lip-sync HTTP API with local file paths."""

import os
import time
import json
import requests

base_url = "http://127.0.0.1:8765"

def check_health(base_url: str, timeout: float) -> dict:
    url = f"{base_url.rstrip('/')}/health"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_lipsync(
    base_url: str,
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    force_chunk: bool = False,
    chunk_duration_sec: float | None = None,
    VAD_result: str = "",
    timeout: float | None = None,
) -> bool:
    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)

    for label, path in (
        ("video", video_path),
        ("audio", audio_path),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "video_path": video_path,
        "audio_path": audio_path,
        "output_path": output_path,
        "force_chunk": force_chunk,
        "VAD_result": VAD_result,
    }
    if chunk_duration_sec is not None:
        payload["chunk_duration_sec"] = chunk_duration_sec

    url = f"{base_url.rstrip('/')}/lipsync"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if not response.ok:
            return False
        data = response.json()
        return bool(data.get("success"))
    except Exception:
        return False
     
     
def get_vad_result_from_intermediate_file(intermediate_file_path: str) -> str:
    ret = []
    with open(intermediate_file_path, "r") as f:
        intermediate_data = json.load(f)
        if 'sentence_list' in intermediate_data:
            sentence_list = intermediate_data["sentence_list"]
            for sentence in sentence_list:
                ret.append({'start':sentence["start"]/1000.0, 'end':sentence["end"]/1000.0})
    return json.dumps(ret)




if __name__ == "__main__":
    started = time.time()
    video = '/opt/oss/wujiedub/video_translate/20260610/10/1891/task/36454/36454_0_1891_watermarked.mp4'
    video = '/opt/oss/wujiedub/video_translate/20260716/18/3497/task/77442/77442_0_3497_watermarked.mp4'
    #video = '/opt/oss/wujiedub/video_translate/20260716/19/3668/task/77445/77445_0_3668_watermarked.mp4'
    #video = '/opt/oss/wujiedub/video_translate/20260701/18/1088/input_videos/147f96fa7a.mp4'
    
    video = '/opt/oss/wujiedub/video_translate/20260719/13/1088/category_task/1346/83366/58543_969_1088_watermarked.mp4'

    video = '/opt/oss/wujiedub/video_translate/20260719/21/1088/category_task/1368/84977/31656_568_1088_watermarked.mp4'
    audio = '2min.wav'
    #audio = '/opt/oss/wujiedub/video_translate/20260717/14/1088/category_task/1284/79010/translated_voice.wav'
    audio = '/opt/oss/wujiedub/video_translate/20260719/13/1088/category_task/1346/83366/translated_voice.wav'
    audio = '/opt/oss/wujiedub/video_translate/20260719/21/1088/category_task/1368/84977/translated_voice.wav'
    output = '/home/ubuntu/raoyonghui/MuseTalk/test_output.mp4'

    intermediate_file_path = '/opt/oss/wujiedub/video_translate/20260719/21/1088/category_task/1368/84977/intermediate_result.json'

    vad_result = get_vad_result_from_intermediate_file(intermediate_file_path)

    print(f"vad_result: {vad_result}")
    ok = run_lipsync(
        base_url,
        video,
        audio,
        output,
        timeout=9600,
        VAD_result = vad_result,
    )
    elapsed = time.time() - started

    print(f"\nCompleted in {elapsed:.1f}s success={ok}")
