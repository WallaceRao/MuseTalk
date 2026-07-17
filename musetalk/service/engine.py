import copy
import glob
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.service.long_video import (
    compute_effective_frame_count,
    compute_segments,
    concat_videos,
    decode_video_frames,
    has_audio_stream,
    mux_video_with_source_audio,
    probe_duration,
    split_audio_segment_frames,
    split_video_segment_frames_with_validation,
    validate_frame_count,
)
from musetalk.utils.active_speaker import LRASDDetector
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
from musetalk.utils.utils import datagen, get_file_type, load_all_model

from musetalk.service.ffmpeg_env import ensure_ffmpeg_env, ensure_ffmpeg_ready

logger = logging.getLogger("musetalk_service")


@dataclass
class ServiceConfig:
    gpu_id: int = 0
    ffmpeg_path: str = "./ffmpeg-4.4-amd64-static/"
    vae_type: str = "sd-vae"
    unet_config: str = "./models/musetalkV15/musetalk.json"
    unet_model_path: str = "./models/musetalkV15/unet.pth"
    whisper_dir: str = "./models/whisper"
    version: str = "v15"
    use_float16: bool = True
    extra_margin: int = 10
    fps: int = 25
    audio_padding_length_left: int = 2
    audio_padding_length_right: int = 2
    batch_size: int = 8
    parsing_mode: str = "jaw"
    left_cheek_width: int = 90
    right_cheek_width: int = 90
    use_lr_asd: bool = True
    asd_model_path: str = "./third_party/LR-ASD/weight/finetuning_TalkSet.model"
    asd_threshold: float = 0.0
    bbox_shift: int = 0
    # Long video: auto-chunk when duration exceeds threshold (seconds).
    auto_chunk_threshold_sec: float = 120.0
    chunk_duration_sec: float = 60.0
    # Detect face bbox every N frames; intermediates are linearly interpolated.
    bbox_detect_stride: int = 3
    # Downscale for detection when short side exceeds this; <= threshold keeps original.
    detect_short_side: int = 720
    # Max simultaneous inference jobs (one engine instance per slot).
    max_concurrent_requests: int = 1
    # Optional per-slot GPU assignment, e.g. [0, 1]. Cycles when slots > len(gpu_ids).
    gpu_ids: Optional[List[int]] = None


class MuseTalkEngine:
    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self.device = torch.device(
            f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self._ensure_ffmpeg()
        self._load_models()

    def _ensure_ffmpeg(self) -> None:
        try:
            ensure_ffmpeg_ready(self.config.ffmpeg_path)
        except RuntimeError as exc:
            logger.warning("%s", exc)

    def _load_models(self) -> None:
        logger.info("Loading MuseTalk models on %s", self.device)
        cfg = self.config
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=cfg.unet_model_path,
            vae_type=cfg.vae_type,
            unet_config=cfg.unet_config,
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)

        if cfg.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path=cfg.whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(cfg.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        if cfg.version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=cfg.left_cheek_width,
                right_cheek_width=cfg.right_cheek_width,
            )
        else:
            self.fp = FaceParsing()

        self.asd_detector = None
        if cfg.use_lr_asd:
            self.asd_detector = LRASDDetector(
                model_path=cfg.asd_model_path,
                device=self.device,
                threshold=cfg.asd_threshold,
            )
        logger.info("MuseTalk models loaded")

    @torch.no_grad()
    def run_lipsync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        *,
        force_chunk: bool = False,
        chunk_duration_sec: Optional[float] = None,
    ) -> dict:
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)
        output_path = os.path.abspath(output_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cfg = self.config
        chunk_sec = chunk_duration_sec if chunk_duration_sec is not None else cfg.chunk_duration_sec
        file_type = get_file_type(video_path)

        if file_type == "video":
            video_duration = probe_duration(video_path)
            audio_duration = probe_duration(audio_path)
            effective_duration = min(video_duration, audio_duration)
            should_chunk = force_chunk or effective_duration > cfg.auto_chunk_threshold_sec
            if should_chunk:
                logger.info(
                    "Long video detected (video=%.1fs audio=%.1fs), chunking with %.0fs segments",
                    video_duration,
                    audio_duration,
                    chunk_sec,
                )
                return self._run_lipsync_chunked(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    chunk_duration_sec=chunk_sec,
                )

        return self._run_lipsync_single(
            video_path,
            audio_path,
            output_path,
            audio_mux_source=video_path if file_type == "video" else None,
        )

    @torch.no_grad()
    def _run_lipsync_chunked(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        chunk_duration_sec: float,
    ) -> dict:
        effective_frames, fps = compute_effective_frame_count(video_path, audio_path)
        segments = compute_segments(effective_frames, fps, chunk_duration_sec)
        expected_total_frames = sum(segment.frame_count for segment in segments)

        logger.info(
            "Processing %d chunks (fps=%.3f, effective_frames=%d, expected_total=%d)",
            len(segments),
            fps,
            effective_frames,
            expected_total_frames,
        )

        total_frame_count = 0
        total_speaking_frames = 0
        total_lipsync_frames = 0
        chunk_outputs: list[str] = []

        with tempfile.TemporaryDirectory(prefix="musetalk_chunks_") as chunk_root:
            for segment in segments:
                chunk_video = os.path.join(chunk_root, f"chunk_{segment.index:04d}_video.mp4")
                chunk_audio = os.path.join(chunk_root, f"chunk_{segment.index:04d}_audio.wav")
                chunk_output = os.path.join(chunk_root, f"chunk_{segment.index:04d}_out.mp4")

                logger.info(
                    "Chunk %d/%d: frames [%d, %d) count=%d",
                    segment.index + 1,
                    len(segments),
                    segment.start_frame,
                    segment.start_frame + segment.frame_count,
                    segment.frame_count,
                )
                split_video_segment_frames_with_validation(
                    video_path, segment, fps, chunk_video
                )
                split_audio_segment_frames(
                    audio_path,
                    segment.start_frame,
                    segment.frame_count,
                    fps,
                    chunk_audio,
                )

                result = self._run_lipsync_single(
                    chunk_video,
                    chunk_audio,
                    chunk_output,
                    audio_mux_source=None,
                )
                validate_frame_count(
                    chunk_output,
                    segment.frame_count,
                    f"chunk {segment.index} lipsync output",
                )
                chunk_outputs.append(chunk_output)
                total_frame_count += result["frame_count"]
                total_speaking_frames += result["speaking_frames"]
                total_lipsync_frames += result["lipsync_frames"]

            concat_temp = os.path.join(chunk_root, "concat_video_only.mp4")
            concat_videos(chunk_outputs, concat_temp)
            validate_frame_count(concat_temp, expected_total_frames, "concatenated video")
            if has_audio_stream(video_path):
                mux_video_with_source_audio(concat_temp, video_path, output_path)
            else:
                shutil.move(concat_temp, output_path)

        result = {
            "output_path": output_path,
            "frame_count": total_frame_count,
            "speaking_frames": total_speaking_frames,
            "lipsync_frames": total_lipsync_frames,
            "chunk_count": len(segments),
            "chunked": True,
        }
        logger.info("Chunked lip-sync completed: %s", result)
        return result

    @torch.no_grad()
    def _run_lipsync_single(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        *,
        audio_mux_source: Optional[str] = None,
    ) -> dict:
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)
        output_path = os.path.abspath(output_path)

        cfg = self.config
        bbox_shift = 0 if cfg.version == "v15" else cfg.bbox_shift

        with tempfile.TemporaryDirectory(prefix="musetalk_") as temp_root:
            temp_dir = os.path.join(temp_root, cfg.version)
            os.makedirs(temp_dir, exist_ok=True)

            input_basename = os.path.splitext(os.path.basename(video_path))[0]
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_basename = f"{input_basename}_{audio_basename}"

            file_type = get_file_type(video_path)
            if file_type == "video":
                logger.info("Decoding video frames into memory: %s", video_path)
                frame_list, fps = decode_video_frames(video_path)
            elif file_type == "image":
                frame = cv2.imread(video_path)
                if frame is None:
                    raise RuntimeError(f"Failed to read image: {video_path}")
                frame_list = [frame]
                fps = float(cfg.fps)
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, "*.png"))
                input_img_list = sorted(
                    input_img_list,
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
                )
                if not input_img_list:
                    raise ValueError(f"No PNG frames found in directory: {video_path}")
                frame_list = []
                for path in tqdm(input_img_list, desc="reading images"):
                    frame = cv2.imread(path)
                    if frame is None:
                        raise RuntimeError(f"Failed to read frame: {path}")
                    frame_list.append(frame)
                fps = float(cfg.fps)
            else:
                raise ValueError(f"Unsupported video input: {video_path}")

            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=cfg.audio_padding_length_left,
                audio_padding_length_right=cfg.audio_padding_length_right,
            )

            logger.info(
                "Extracting landmarks for %d frames (detect_stride=%d, detect_short_side=%d)",
                len(frame_list),
                cfg.bbox_detect_stride,
                cfg.detect_short_side,
            )
            coord_list, frame_list = get_landmark_and_bbox(
                upperbondrange=bbox_shift,
                detect_stride=cfg.bbox_detect_stride,
                frames=frame_list,
                detect_short_side=cfg.detect_short_side,
            )

            video_num = min(len(whisper_chunks), len(frame_list))
            whisper_chunks = whisper_chunks[:video_num]
            coord_list = coord_list[:video_num]
            frame_list = frame_list[:video_num]

            speaking_mask = None
            speaking_frames = 0
            if cfg.use_lr_asd and self.asd_detector is not None:
                logger.info("Running LR-ASD active speaker detection")
                speaking_mask, _ = self.asd_detector.compute_speaking_mask(
                    audio_path=audio_path,
                    frame_list=frame_list,
                    coord_list=coord_list,
                    fps=fps,
                )
                speaking_mask = speaking_mask[:video_num]
                speaking_frames = sum(speaking_mask)
                logger.info(
                    "LR-ASD: %d/%d frames marked as speaking (threshold=%s)",
                    speaking_frames,
                    len(speaking_mask),
                    cfg.asd_threshold,
                )
            else:
                speaking_mask = [True] * video_num
                speaking_frames = video_num

            infer_indices = []
            for i in range(video_num):
                if not speaking_mask[i]:
                    continue
                bbox = coord_list[i]
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if cfg.version == "v15":
                    y2 = min(y2 + cfg.extra_margin, frame_list[i].shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                infer_indices.append(i)

            if not infer_indices and not any(b != coord_placeholder for b in coord_list):
                raise ValueError("No valid face detected in the input video")

            res_by_index: dict[int, np.ndarray] = {}
            if infer_indices:
                logger.info(
                    "Encoding latents for %d speaking frames (skipped %d)",
                    len(infer_indices),
                    video_num - len(infer_indices),
                )
                infer_latents = []
                infer_whisper = []
                for i in infer_indices:
                    bbox = coord_list[i]
                    x1, y1, x2, y2 = bbox
                    if cfg.version == "v15":
                        y2 = min(y2 + cfg.extra_margin, frame_list[i].shape[0])
                    crop_frame = frame_list[i][y1:y2, x1:x2]
                    crop_frame = cv2.resize(
                        crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                    )
                    infer_latents.append(self.vae.get_latents_for_unet(crop_frame))
                    infer_whisper.append(whisper_chunks[i])

                gen = datagen(
                    whisper_chunks=infer_whisper,
                    vae_encode_latents=infer_latents,
                    batch_size=cfg.batch_size,
                    delay_frame=0,
                    device=self.device,
                )
                total = int(np.ceil(float(len(infer_indices)) / cfg.batch_size))
                logger.info(
                    "Starting MuseTalk inference for %d/%d frames",
                    len(infer_indices),
                    video_num,
                )
                produced = []
                for _, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                    audio_feature_batch = self.pe(whisper_batch)
                    latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch,
                        self.timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    recon = self.vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        produced.append(res_frame)
                for idx, res_frame in zip(infer_indices, produced):
                    res_by_index[idx] = res_frame
            else:
                logger.info("No speaking frames to infer; writing original frames")

            logger.info("Compositing frames with VideoWriter (no intermediate PNG)")
            lipsync_frames = 0
            height, width = frame_list[0].shape[:2]
            width_even = width - (width % 2)
            height_even = height - (height % 2)
            writer_path = os.path.join(temp_dir, f"temp_{output_basename}_writer.mp4")
            temp_vid_path = os.path.join(temp_dir, f"temp_{output_basename}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                writer_path, fourcc, float(fps), (width_even, height_even)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {writer_path}")

            try:
                for i in tqdm(range(video_num)):
                    bbox = coord_list[i]
                    ori_frame = frame_list[i]
                    x1, y1, x2, y2 = bbox
                    if cfg.version == "v15":
                        y2 = min(y2 + cfg.extra_margin, ori_frame.shape[0])

                    res_frame = res_by_index.get(i)
                    if res_frame is None:
                        combine_frame = ori_frame
                    else:
                        try:
                            ori_copy = copy.deepcopy(ori_frame)
                            res_frame = cv2.resize(
                                res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                            )
                            if cfg.version == "v15":
                                combine_frame = get_image(
                                    ori_copy,
                                    res_frame,
                                    [x1, y1, x2, y2],
                                    mode=cfg.parsing_mode,
                                    fp=self.fp,
                                )
                            else:
                                combine_frame = get_image(
                                    ori_copy, res_frame, [x1, y1, x2, y2], fp=self.fp
                                )
                            lipsync_frames += 1
                        except Exception:
                            combine_frame = ori_frame

                    if (
                        combine_frame.shape[0] != height_even
                        or combine_frame.shape[1] != width_even
                    ):
                        combine_frame = combine_frame[:height_even, :width_even]
                    writer.write(combine_frame)
            finally:
                writer.release()

            cmd_reencode = (
                f"ffmpeg -y -v warning -i {writer_path} "
                f"-vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            )
            ret = os.system(cmd_reencode)
            if ret != 0:
                raise RuntimeError(f"ffmpeg re-encode failed with code {ret}")

            if audio_mux_source and has_audio_stream(audio_mux_source):
                logger.info("Muxing original video audio from %s", audio_mux_source)
                mux_video_with_source_audio(temp_vid_path, audio_mux_source, output_path)
            else:
                shutil.move(temp_vid_path, output_path)

        result = {
            "output_path": output_path,
            "frame_count": video_num,
            "speaking_frames": speaking_frames,
            "lipsync_frames": lipsync_frames,
            "chunked": False,
            "chunk_count": 1,
        }
        logger.info("Lip-sync completed: %s", result)
        return result
