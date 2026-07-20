#!/usr/bin/env python3
"""Call the MuseTalk lip-sync HTTP API with local file paths."""

import os
import time

import requests

base_url = "http://10.206.16.6:8765"

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
    ok = run_lipsync(
        base_url,
        video,
        audio,
        output,
        timeout=9600,
    )
    elapsed = time.time() - started

    print(f"\nCompleted in {elapsed:.1f}s success={ok}")
