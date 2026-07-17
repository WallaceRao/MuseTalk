"""Long-video helpers: frame-accurate split, validation, and concat."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

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


def decode_video_frames(video_path: str) -> tuple[list, float]:
    """Decode all video frames into memory (BGR), return (frames, fps)."""
    ensure_ffmpeg_env()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


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


class FFmpegRawVideoWriter:
    """Write BGR frames to libx264 via ffmpeg rawvideo stdin pipe (single encode)."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 18,
        preset: str = "veryfast",
    ):
        ensure_ffmpeg_env()
        self.output_path = output_path
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid output size: {self.width}x{self.height}")
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.crf = int(crf)
        self.preset = preset
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_chunks: list[bytes] = []
        self._closed = False
        self._frame_bytes = self.width * self.height * 3

    def __enter__(self) -> "FFmpegRawVideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.abort()
        else:
            self.close()
        return False

    def _stderr_text(self) -> str:
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace").strip()

    def open(self) -> None:
        if self._proc is not None:
            return
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        def _drain_stderr() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            try:
                while True:
                    chunk = self._proc.stderr.read(4096)
                    if not chunk:
                        break
                    self._stderr_chunks.append(chunk)
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()
        logger.info(
            "ffmpeg rawvideo writer opened: %dx%d @ %.3ffps -> %s",
            self.width,
            self.height,
            self.fps,
            self.output_path,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FFmpegRawVideoWriter is not open")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape={frame.shape}")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = frame[: self.height, : self.width]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        payload = frame.tobytes()
        if len(payload) != self._frame_bytes:
            raise ValueError(
                f"Frame byte size mismatch: got {len(payload)}, expected {self._frame_bytes}"
            )
        try:
            self._proc.stdin.write(payload)
        except BrokenPipeError as exc:
            err = self._stderr_text()
            raise RuntimeError(f"ffmpeg stdin pipe broken while writing: {err or exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is None:
            return

        ret = -1
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                ret = self._proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=30)
                raise RuntimeError("ffmpeg encode timed out after closing stdin")
        finally:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=5)

        if ret != 0:
            err = self._stderr_text()
            if os.path.isfile(self.output_path):
                try:
                    os.remove(self.output_path)
                except OSError:
                    pass
            raise RuntimeError(f"ffmpeg exited with code {ret}: {err or 'no stderr'}")

    def abort(self) -> None:
        """Best-effort kill and remove incomplete output."""
        self._closed = True
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                if self._proc.poll() is None:
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=15)
                    except Exception:
                        pass
            finally:
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=2)
                self._proc = None
        if os.path.isfile(self.output_path):
            try:
                os.remove(self.output_path)
            except OSError:
                pass
