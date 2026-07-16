"""Long-video helpers: frame-accurate split, validation, and concat."""

from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass
from typing import List

from musetalk.service.ffmpeg_env import ensure_ffmpeg_env

logger = logging.getLogger("musetalk_service")


@dataclass
class VideoStreamInfo:
    fps: float
    frame_count: int
    duration_sec: float


@dataclass
class MediaSegment:
    index: int
    start_frame: int
    frame_count: int
    start_sec: float
    duration_sec: float


def _parse_fps(rate: str) -> float:
    rate = rate.strip()
    if not rate or rate == "0/0":
        raise ValueError(f"Invalid frame rate: {rate!r}")
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den)
    return float(rate)


def probe_duration(path: str) -> float:
    ensure_ffmpeg_env()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def probe_video_frame_count(path: str) -> int:
    ensure_ffmpeg_env()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    count = result.stdout.strip()
    if not count:
        raise RuntimeError(f"Could not count video frames: {path}")
    return int(count)


def probe_video_stream(path: str) -> VideoStreamInfo:
    ensure_ffmpeg_env()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json

    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    duration_sec = float(payload["format"]["duration"])

    fps_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = _parse_fps(fps_raw)
    frame_count = probe_video_frame_count(path)
    return VideoStreamInfo(fps=fps, frame_count=frame_count, duration_sec=duration_sec)


def compute_effective_frame_count(
    video_path: str,
    audio_path: str,
    fps: float | None = None,
) -> tuple[int, float]:
    video_info = probe_video_stream(video_path)
    fps = fps or video_info.fps
    audio_duration = probe_duration(audio_path)
    audio_frames = math.floor(audio_duration * fps)
    effective_frames = min(video_info.frame_count, audio_frames)
    return max(1, effective_frames), fps


def compute_segments(
    total_frames: int,
    fps: float,
    chunk_duration_sec: float,
) -> List[MediaSegment]:
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    fps = float(fps)
    chunk_frames = max(1, int(round(chunk_duration_sec * fps)))

    segments: List[MediaSegment] = []
    start_frame = 0
    index = 0
    while start_frame < total_frames:
        frame_count = min(chunk_frames, total_frames - start_frame)
        segments.append(
            MediaSegment(
                index=index,
                start_frame=start_frame,
                frame_count=frame_count,
                start_sec=start_frame / fps,
                duration_sec=frame_count / fps,
            )
        )
        start_frame += frame_count
        index += 1
    return segments


def validate_frame_count(path: str, expected: int, label: str) -> int:
    actual = probe_video_frame_count(path)
    if actual != expected:
        raise RuntimeError(
            f"{label}: frame count mismatch, expected {expected}, got {actual} ({path})"
        )
    logger.info("%s: frame count validated (%d frames)", label, actual)
    return actual


def split_video_segment_frames(
    video_path: str,
    start_frame: int,
    frame_count: int,
    fps: float,
    output_path: str,
) -> None:
    """Frame-accurate video split using trim filter (decode-based, no keyframe seek)."""
    ensure_ffmpeg_env()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    end_frame = start_frame + frame_count
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        video_path,
        "-vf",
        vf,
        "-fps_mode",
        "cfr",
        "-r",
        f"{fps:.6f}",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def split_video_segment_frames_select(
    video_path: str,
    start_frame: int,
    frame_count: int,
    fps: float,
    output_path: str,
) -> None:
    """Fallback frame split using select filter."""
    ensure_ffmpeg_env()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    last_frame = start_frame + frame_count - 1
    vf = f"select='between(n\\,{start_frame}\\,{last_frame})',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        video_path,
        "-vf",
        vf,
        "-fps_mode",
        "cfr",
        "-r",
        f"{fps:.6f}",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def split_video_segment_frames_with_validation(
    video_path: str,
    segment: MediaSegment,
    fps: float,
    output_path: str,
) -> None:
    label = f"chunk {segment.index} video split"
    try:
        split_video_segment_frames(
            video_path,
            segment.start_frame,
            segment.frame_count,
            fps,
            output_path,
        )
        validate_frame_count(output_path, segment.frame_count, label)
    except RuntimeError:
        logger.warning(
            "%s failed trim-based split, retrying with select filter",
            label,
        )
        split_video_segment_frames_select(
            video_path,
            segment.start_frame,
            segment.frame_count,
            fps,
            output_path,
        )
        validate_frame_count(output_path, segment.frame_count, label)


def split_audio_segment_frames(
    audio_path: str,
    start_frame: int,
    frame_count: int,
    fps: float,
    output_path: str,
    sample_rate: int = 16000,
) -> None:
    """Sample-accurate audio split; -ss is placed after -i for decode accuracy."""
    ensure_ffmpeg_env()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    start_sec = start_frame / fps
    duration_sec = frame_count / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        audio_path,
        "-ss",
        f"{start_sec:.9f}",
        "-t",
        f"{duration_sec:.9f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def mux_video_with_source_audio(
    video_path: str,
    audio_source_path: str,
    output_path: str,
) -> None:
    """Mux processed silent video with audio copied from the source video."""
    ensure_ffmpeg_env()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-i",
        video_path,
        "-i",
        audio_source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def has_audio_stream(path: str) -> bool:
    ensure_ffmpeg_env()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def concat_videos(video_paths: List[str], output_path: str) -> None:
    ensure_ffmpeg_env()
    if not video_paths:
        raise ValueError("No video segments to concatenate")
    if len(video_paths) == 1:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", video_paths[0], "-c", "copy", output_path],
            check=True,
        )
        return

    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in video_paths:
            escaped = path.replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.isfile(list_path):
            os.remove(list_path)
