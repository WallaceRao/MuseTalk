import glob
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.service.long_video import (
    FFmpegRawVideoWriter,
    compute_effective_frame_count,
    compute_segments,
    concat_videos,
    decode_video_frame_range,
    decode_video_frames,
    has_audio_stream,
    mux_video_with_source_audio,
    probe_duration,
    split_audio_segment_frames,
    validate_frame_count,
)
from musetalk.utils.active_speaker import (
    LRASDDetector,
    VSDLMDetector,
    detect_shot_ids,
    dilate_speaking_mask,
    expand_speaking_mask_to_shots,
    expand_speaking_mask_within_vad,
    prune_short_speaking_runs,
)
from musetalk.utils.vad_vsdlm_fusion import build_fused_speaking_mask
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import (
    blend_frames,
    compute_segment_blend_alphas,
    get_image,
    match_face_color_lab,
)
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import (
    _bbox_iou,
    coord_placeholder,
    get_landmark_and_bbox,
)
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
    # Mouth-activity gate for lipsync (visual-only). Preferred over LR-ASD for dubbing.
    use_vsdlm: bool = True
    vsdlm_model_path: str = "./third_party/VSDLM/vsdlm_m.onnx"
    vsdlm_open_threshold: float = 0.15
    vsdlm_activity_threshold: float = 0.12
    vsdlm_activity_window: int = 4
    # Require contiguous lip-motion for at least this many seconds to count as speaking.
    vsdlm_min_speak_duration_sec: float = 0.5
    # Landmark MAR rescue when VSDLM under-scores slightly-open speaking.
    # When MAR >= this, also score a face-band crop and take max(open).
    vsdlm_mar_open_threshold: float = 0.12
    # Landmark MAR activity used when VSDLM open saturates (sustained open).
    vsdlm_mar_activity_threshold: float = 0.04
    # Soft attenuate VSDLM open when landmark MAR is below this (0 = off).
    # 0.12 suppresses closed/near-closed false opens (e.g. red-lipstick crops
    # that VSDLM scores ~0.7 while MAR≈0.05); 0.06 was too weak for that.
    vsdlm_soft_closed_mar: float = 0.12
    # Batched ONNX inference size for VSDLM open scoring.
    vsdlm_batch_size: int = 64
    # Fuse TenVAD speech segments with multi-face VSDLM (assign + split at switches).
    use_vad_fusion: bool = True
    vad_url: str = "http://127.0.0.1:8061/vad_detect/"
    # Loose lip-motion duration for VAD∩VSDLM intersection (shorter than strict turns).
    vsdlm_loose_min_speak_duration_sec: float = 0.2
    # Max time gap when assigning a VAD segment to a non-overlapping visual speaker.
    vad_assign_max_gap_sec: float = 0.5
    # SCRFD multi-face detect every N frames inside VAD×VSDLM fusion.
    asd_face_detect_stride: int = 3
    # Legacy AV-sync ASD (optional fallback when use_vsdlm=False).
    use_lr_asd: bool = False
    asd_model_path: str = "./third_party/LR-ASD/weight/finetuning_TalkSet.model"
    asd_threshold: float = 0.0
    # Drop raw speaking runs shorter than this before dilation.
    # Also applied again after dilation to remove face-drop islands.
    asd_pre_dilate_min_sec: float = 0.2
    # Legacy fallback: expand speaking by N frames each side when VAD expand
    # is unavailable (no VAD segments / vad expand disabled).
    asd_mask_dilate: int = 8
    # When VAD segments are available: expand each speaking run toward its
    # parent VAD by up to this many seconds per side. Clamped to the VAD
    # bounds and never across shot cuts. 0 = fall back to asd_mask_dilate.
    asd_vad_expand_max_sec: float = 3.0
    # Keep dilation / VAD expand inside the same camera shot (no cross-cut bleed).
    asd_dilate_respect_shots: bool = True
    # After speaking detection: lipsync whole shots that contain speaking
    # long enough to pass ``lipsync_shot_min_speak_sec``.
    lipsync_full_speaking_shots: bool = False
    # Contiguous speaking (post-dilate) required inside a shot before expanding
    # lipsync to the whole shot. Filters brief VSDLM false positives
    # (e.g. ~19/33/46/54/93s). 1.5s is stricter than V1's 1.1s (drops ~20s FP expands).
    lipsync_shot_min_speak_sec: float = 1.5
    # Shots shorter than this use ``lipsync_short_shot_min_speak_sec`` instead
    # (recovers brief real turns in ~1–2s cuts without relaxing long shots).
    lipsync_short_shot_sec: float = 2.0
    lipsync_short_shot_min_speak_sec: float = 0.7
    # When a shot fails whole-shot expand, still lipsync original runs >= this.
    lipsync_keep_partial_min_sec: float = 0.5
    # Open-prob temporal max (±N frames). 0 = off.
    vsdlm_temporal_max_radius: int = 2
    # Bhattacharyya hist distance threshold for declaring a shot cut (0~1).
    shot_cut_hist_threshold: float = 0.45
    bbox_shift: int = 0
    # Long video: auto-chunk when duration exceeds threshold (seconds).
    auto_chunk_threshold_sec: float = 120.0
    chunk_duration_sec: float = 60.0
    # Detect face bbox every N frames; intermediates are linearly interpolated.
    bbox_detect_stride: int = 3
    # Downscale for detection when short side exceeds this; <= threshold keeps original.
    detect_short_side: int = 720
    # Ignore faces smaller than this fraction of the full frame area (no track/ASD/lipsync).
    min_face_area_ratio: float = 1.0 / 100.0
    # CodeFormer: restore only MuseTalk-generated speaking-face crops.
    use_codeformer: bool = True
    codeformer_model_path: str = "./models/codeformer/codeformer.pth"
    # Higher = closer to input identity; lower = stronger restoration. Typical 0.5–0.8.
    codeformer_fidelity: float = 0.7
    codeformer_use_fp16: bool = True
    codeformer_batch_size: int = 2
    # Restore every N speaking frames; intermediates reuse the previous restored face.
    # Keep 1 for temporal stability; stride>1 causes flicker when head pose changes.
    codeformer_stride: int = 1
    # Soften lipsync↔original transitions over N frames at each speaking-segment edge.
    blend_ramp_frames: int = 5
    # Match generated face crop colors to the original crop before parsing blend.
    use_color_match: bool = True
    # Max simultaneous inference jobs (one engine instance per slot).
    max_concurrent_requests: int = 1
    # Optional per-slot GPU assignment, e.g. [0, 1]. Cycles when slots > len(gpu_ids).
    gpu_ids: Optional[List[int]] = None
    # Lip generation backend: "latentsync" (V2 default) or "musetalk" (V1).
    lipsync_backend: str = "latentsync"
    latentsync_repo: str = "./third_party/LatentSync"
    latentsync_unet_config: str = "./third_party/LatentSync/configs/unet/stage2.yaml"
    latentsync_ckpt: str = "./models/latentsync15/latentsync_unet.pt"
    latentsync_whisper: str = "./models/latentsync15/whisper/tiny.pt"
    latentsync_vae: str = "./models/sd-vae"
    latentsync_inference_steps: int = 18
    latentsync_guidance_scale: float = 1.0
    latentsync_enable_deepcache: bool = True
    latentsync_seed: int = 1247


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

    @property
    def _use_latentsync(self) -> bool:
        return str(self.config.lipsync_backend).strip().lower() in {
            "latentsync",
            "latentsync15",
            "ls",
            "v2",
        }

    def _load_models(self) -> None:
        cfg = self.config
        self.latentsync = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.fp = None
        self.weight_dtype = torch.float16 if cfg.use_float16 else torch.float32
        self.timesteps = torch.tensor([0], device=self.device)

        if self._use_latentsync:
            from musetalk.service.latentsync_backend import LatentSyncBackend

            logger.info("Loading LatentSync 1.5 lipsync backend on %s", self.device)
            self.latentsync = LatentSyncBackend(
                device=self.device,
                repo_root=cfg.latentsync_repo,
                unet_config=cfg.latentsync_unet_config,
                checkpoint=cfg.latentsync_ckpt,
                whisper_tiny=cfg.latentsync_whisper,
                vae_path=cfg.latentsync_vae,
                inference_steps=cfg.latentsync_inference_steps,
                guidance_scale=cfg.latentsync_guidance_scale,
                enable_deepcache=cfg.latentsync_enable_deepcache,
                seed=cfg.latentsync_seed,
                use_float16=cfg.use_float16,
            )
        else:
            logger.info("Loading MuseTalk models on %s", self.device)
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=cfg.unet_model_path,
                vae_type=cfg.vae_type,
                unet_config=cfg.unet_config,
                device=self.device,
            )

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

        self.vsdlm_detector = None
        self.asd_detector = None
        if cfg.use_vsdlm:
            self.vsdlm_detector = VSDLMDetector(
                model_path=cfg.vsdlm_model_path,
                open_threshold=cfg.vsdlm_open_threshold,
                activity_threshold=cfg.vsdlm_activity_threshold,
                activity_window=cfg.vsdlm_activity_window,
                min_speak_duration_sec=cfg.vsdlm_min_speak_duration_sec,
                mar_open_threshold=cfg.vsdlm_mar_open_threshold,
                mar_activity_threshold=cfg.vsdlm_mar_activity_threshold,
                temporal_max_radius=cfg.vsdlm_temporal_max_radius,
                soft_closed_mar=cfg.vsdlm_soft_closed_mar,
                batch_size=cfg.vsdlm_batch_size,
            )
            logger.info(
                "VSDLM mouth-activity gate ready "
                "(std>=%.2f, mean_open>=%.2f, win=%d, min_speak=%.2fs, "
                "dual_crop_mar>=%.2f, batch=%d, providers=%s)",
                cfg.vsdlm_activity_threshold,
                cfg.vsdlm_activity_threshold,
                cfg.vsdlm_activity_window,
                cfg.vsdlm_min_speak_duration_sec,
                cfg.vsdlm_mar_open_threshold,
                cfg.vsdlm_batch_size,
                ",".join(self.vsdlm_detector.providers),
            )
        elif cfg.use_lr_asd:
            self.asd_detector = LRASDDetector(
                model_path=cfg.asd_model_path,
                device=self.device,
                threshold=cfg.asd_threshold,
            )

        self.codeformer = None
        # CodeFormer restores MuseTalk face crops; skip for LatentSync full-frame output.
        if cfg.use_codeformer and not self._use_latentsync:
            try:
                from musetalk.utils.codeformer_restorer import CodeFormerRestorer

                self.codeformer = CodeFormerRestorer(
                    model_path=cfg.codeformer_model_path,
                    device=self.device,
                    fidelity_weight=cfg.codeformer_fidelity,
                    use_fp16=cfg.codeformer_use_fp16,
                    batch_size=cfg.codeformer_batch_size,
                )
            except Exception as exc:
                logger.warning(
                    "CodeFormer disabled (failed to load): %s",
                    exc,
                )
                self.codeformer = None
        logger.info(
            "Lip-sync engine ready (backend=%s)",
            "latentsync" if self._use_latentsync else "musetalk",
        )

    @torch.no_grad()
    def run_lipsync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        *,
        force_chunk: bool = False,
        chunk_duration_sec: Optional[float] = None,
        vad_segments: Optional[List[Tuple[float, float]]] = None,
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
                    vad_segments=vad_segments,
                )

        return self._run_lipsync_single(
            video_path,
            audio_path,
            output_path,
            audio_mux_source=audio_path,
            vad_segments=vad_segments,
        )

    @torch.no_grad()
    def _run_lipsync_chunked(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        chunk_duration_sec: float,
        *,
        vad_segments: Optional[List[Tuple[float, float]]] = None,
    ) -> dict:
        from musetalk.utils.vad_client import clip_vad_segments_to_window

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
                # Decode directly from the original (no lossy re-encode). Lossy
                # x264 chunk intermediates were dropping micro-open speaking cues.
                logger.info(
                    "Decoding original frames [%d, %d) without re-encode",
                    segment.start_frame,
                    segment.start_frame + segment.frame_count,
                )
                chunk_frames, chunk_fps = decode_video_frame_range(
                    video_path,
                    segment.start_frame,
                    segment.frame_count,
                    fps=fps,
                )
                split_audio_segment_frames(
                    audio_path,
                    segment.start_frame,
                    segment.frame_count,
                    fps,
                    chunk_audio,
                )

                chunk_vad = None
                if vad_segments is not None:
                    chunk_t0 = float(segment.start_frame) / float(fps)
                    chunk_dur = float(segment.frame_count) / float(fps)
                    chunk_vad = clip_vad_segments_to_window(
                        vad_segments,
                        window_start_sec=chunk_t0,
                        window_duration_sec=chunk_dur,
                    )

                result = self._run_lipsync_single(
                    video_path,
                    chunk_audio,
                    chunk_output,
                    audio_mux_source=None,
                    preloaded_frames=chunk_frames,
                    preloaded_fps=chunk_fps,
                    vad_segments=chunk_vad,
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
                del chunk_frames

            concat_temp = os.path.join(chunk_root, "concat_video_only.mp4")
            concat_videos(chunk_outputs, concat_temp)
            validate_frame_count(concat_temp, expected_total_frames, "concatenated video")
            if has_audio_stream(audio_path):
                logger.info("Muxing input audio into final output: %s", audio_path)
                mux_video_with_source_audio(concat_temp, audio_path, output_path)
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
        preloaded_frames: Optional[List[np.ndarray]] = None,
        preloaded_fps: Optional[float] = None,
        vad_segments: Optional[List[Tuple[float, float]]] = None,
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

            if preloaded_frames is not None:
                frame_list = list(preloaded_frames)
                fps = float(preloaded_fps) if preloaded_fps and preloaded_fps > 0 else float(cfg.fps)
                logger.info(
                    "Using preloaded frames: %d @ %.3ffps (no intermediate re-encode)",
                    len(frame_list),
                    fps,
                )
            else:
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

            whisper_chunks = None
            if not self._use_latentsync:
                whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                    audio_path
                )
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
                "Extracting landmarks for %d frames (detect_stride=%d, detect_short_side=%d, min_face_area_ratio=%.4f)",
                len(frame_list),
                cfg.bbox_detect_stride,
                cfg.detect_short_side,
                cfg.min_face_area_ratio,
            )
            coord_list, frame_list, mouth_coord_list, mouth_mar_list = get_landmark_and_bbox(
                upperbondrange=bbox_shift,
                detect_stride=cfg.bbox_detect_stride,
                frames=frame_list,
                detect_short_side=cfg.detect_short_side,
                min_face_area_ratio=cfg.min_face_area_ratio,
                return_mouth_coords=True,
            )

            if whisper_chunks is not None:
                video_num = min(len(whisper_chunks), len(frame_list))
                whisper_chunks = whisper_chunks[:video_num]
            else:
                video_num = len(frame_list)
            coord_list = coord_list[:video_num]
            frame_list = frame_list[:video_num]
            mouth_coord_list = mouth_coord_list[:video_num]
            mouth_mar_list = mouth_mar_list[:video_num]

            speaking_mask = None
            speaking_frames = 0
            fusion_meta = None
            if cfg.use_vsdlm and self.vsdlm_detector is not None:
                if cfg.use_vad_fusion:
                    logger.info(
                        "Running VAD×VSDLM fusion (vad=%s%s)",
                        "client-provided" if vad_segments is not None else cfg.vad_url,
                        f", {len(vad_segments)} segs" if vad_segments is not None else "",
                    )
                    speaking_mask, fusion_meta = build_fused_speaking_mask(
                        self.vsdlm_detector,
                        frame_list,
                        coord_list,
                        audio_path,
                        fps=fps,
                        vad_url=cfg.vad_url,
                        vad_segments=vad_segments,
                        detect_short_side=cfg.detect_short_side,
                        min_face_area_ratio=cfg.min_face_area_ratio,
                        detect_stride=cfg.asd_face_detect_stride,
                        loose_min_speak_duration_sec=cfg.vsdlm_loose_min_speak_duration_sec,
                        max_assign_gap_sec=cfg.vad_assign_max_gap_sec,
                        shot_cut_hist_threshold=cfg.shot_cut_hist_threshold,
                        mouth_coord_list=mouth_coord_list,
                        mouth_mar_list=mouth_mar_list,
                    )
                    speaking_mask = speaking_mask[:video_num]
                    raw_speaking = int(fusion_meta.get("raw_speaking_frames", sum(speaking_mask)))
                    logger.info(
                        "VAD×VSDLM: shots=%s tracks=%s visual_iv=%s vad=%s assigned=%s primary=%s raw=%d/%d",
                        fusion_meta.get("n_shots"),
                        fusion_meta.get("n_face_tracks"),
                        fusion_meta.get("n_visual_intervals"),
                        fusion_meta.get("n_vad_segments"),
                        fusion_meta.get("n_assigned_segments"),
                        fusion_meta.get("primary_track_id"),
                        raw_speaking,
                        video_num,
                    )
                else:
                    logger.info("Running VSDLM mouth-activity detection")
                    shot_ids_vsdlm = None
                    if cfg.asd_dilate_respect_shots:
                        shot_ids_vsdlm = detect_shot_ids(
                            frame_list[:video_num],
                            hist_threshold=cfg.shot_cut_hist_threshold,
                        )
                    speaking_mask, _ = self.vsdlm_detector.compute_speaking_mask(
                        frame_list=frame_list,
                        coord_list=coord_list,
                        fps=fps,
                        shot_ids=shot_ids_vsdlm,
                        mouth_coord_list=mouth_coord_list,
                        mouth_mar_list=mouth_mar_list,
                    )
                    speaking_mask = speaking_mask[:video_num]
                    raw_speaking = sum(speaking_mask)
                if cfg.asd_pre_dilate_min_sec > 0:
                    pre_shot_ids = None
                    if cfg.asd_dilate_respect_shots:
                        if (
                            fusion_meta is not None
                            and fusion_meta.get("shot_ids") is not None
                        ):
                            pre_shot_ids = list(fusion_meta["shot_ids"])[:video_num]
                        else:
                            pre_shot_ids = detect_shot_ids(
                                frame_list[:video_num],
                                hist_threshold=cfg.shot_cut_hist_threshold,
                            )
                    before_prune = sum(speaking_mask)
                    speaking_mask = prune_short_speaking_runs(
                        speaking_mask,
                        min_duration_sec=cfg.asd_pre_dilate_min_sec,
                        fps=fps,
                        shot_ids=pre_shot_ids,
                    )
                    logger.info(
                        "Pre-dilate prune: drop runs < %.2fs → %d/%d (was %d)",
                        cfg.asd_pre_dilate_min_sec,
                        sum(speaking_mask),
                        len(speaking_mask),
                        before_prune,
                    )
                    raw_speaking = sum(speaking_mask)
                if cfg.asd_vad_expand_max_sec > 0 or cfg.asd_mask_dilate > 0:
                    valid_face = [b != coord_placeholder for b in coord_list]
                    shot_ids = None
                    n_shots = 1
                    if cfg.asd_dilate_respect_shots:
                        if fusion_meta is not None and fusion_meta.get("shot_ids") is not None:
                            shot_ids = list(fusion_meta["shot_ids"])[:video_num]
                            n_shots = int(
                                fusion_meta.get("n_shots")
                                or ((max(shot_ids) + 1) if shot_ids else 1)
                            )
                        else:
                            shot_ids = detect_shot_ids(
                                frame_list[:video_num],
                                hist_threshold=cfg.shot_cut_hist_threshold,
                            )
                            n_shots = (max(shot_ids) + 1) if shot_ids else 1
                        logger.info(
                            "Shot cuts for speaking expand: %d shots (hist_threshold=%.2f)",
                            n_shots,
                            cfg.shot_cut_hist_threshold,
                        )

                    vad_for_expand = None
                    if fusion_meta is not None and fusion_meta.get("vad_segments"):
                        vad_for_expand = list(fusion_meta["vad_segments"])
                    elif vad_segments:
                        vad_for_expand = list(vad_segments)

                    before_expand = sum(speaking_mask)
                    if (
                        cfg.asd_vad_expand_max_sec > 0
                        and vad_for_expand
                    ):
                        speaking_mask = expand_speaking_mask_within_vad(
                            speaking_mask,
                            vad_for_expand,
                            fps=fps,
                            max_expand_sec=cfg.asd_vad_expand_max_sec,
                            shot_ids=shot_ids,
                            valid_face=valid_face,
                        )
                        logger.info(
                            "VAD speaking expand: ±%.2fs (shot_aware=%s) → %d/%d (was %d)",
                            cfg.asd_vad_expand_max_sec,
                            shot_ids is not None,
                            sum(speaking_mask),
                            len(speaking_mask),
                            before_expand,
                        )
                    elif cfg.asd_mask_dilate > 0:
                        speaking_mask = dilate_speaking_mask(
                            speaking_mask,
                            cfg.asd_mask_dilate,
                            valid_face=valid_face,
                            shot_ids=shot_ids,
                        )
                        logger.info(
                            "Frame dilate ±%d (no VAD expand) → %d/%d (was %d)",
                            cfg.asd_mask_dilate,
                            sum(speaking_mask),
                            len(speaking_mask),
                            before_expand,
                        )

                    # Drop expand islands left by face-drop holes (e.g. 1-frame fragments).
                    if cfg.asd_pre_dilate_min_sec > 0:
                        before_post = sum(speaking_mask)
                        speaking_mask = prune_short_speaking_runs(
                            speaking_mask,
                            min_duration_sec=cfg.asd_pre_dilate_min_sec,
                            fps=fps,
                            shot_ids=shot_ids,
                        )
                        logger.info(
                            "Post-expand prune: drop runs < %.2fs → %d/%d (was %d)",
                            cfg.asd_pre_dilate_min_sec,
                            sum(speaking_mask),
                            len(speaking_mask),
                            before_post,
                        )
                speaking_frames = sum(speaking_mask)
                logger.info(
                    "Speaking gate: %d/%d after expand (raw=%d, vad_fusion=%s, "
                    "vad_expand=%.2fs, dilate±%d, min_speak=%.2fs)",
                    speaking_frames,
                    len(speaking_mask),
                    raw_speaking,
                    cfg.use_vad_fusion,
                    cfg.asd_vad_expand_max_sec,
                    cfg.asd_mask_dilate,
                    cfg.vsdlm_min_speak_duration_sec,
                )
            elif cfg.use_lr_asd and self.asd_detector is not None:
                logger.info("Running LR-ASD active speaker detection")
                speaking_mask, _ = self.asd_detector.compute_speaking_mask(
                    audio_path=audio_path,
                    frame_list=frame_list,
                    coord_list=coord_list,
                    fps=fps,
                )
                speaking_mask = speaking_mask[:video_num]
                raw_speaking = sum(speaking_mask)
                if cfg.asd_pre_dilate_min_sec > 0:
                    pre_shot_ids = None
                    if cfg.asd_dilate_respect_shots:
                        pre_shot_ids = detect_shot_ids(
                            frame_list[:video_num],
                            hist_threshold=cfg.shot_cut_hist_threshold,
                        )
                    before_prune = sum(speaking_mask)
                    speaking_mask = prune_short_speaking_runs(
                        speaking_mask,
                        min_duration_sec=cfg.asd_pre_dilate_min_sec,
                        fps=fps,
                        shot_ids=pre_shot_ids,
                    )
                    logger.info(
                        "Pre-dilate prune: drop runs < %.2fs → %d/%d (was %d)",
                        cfg.asd_pre_dilate_min_sec,
                        sum(speaking_mask),
                        len(speaking_mask),
                        before_prune,
                    )
                    raw_speaking = sum(speaking_mask)
                if cfg.asd_mask_dilate > 0:
                    valid_face = [b != coord_placeholder for b in coord_list]
                    shot_ids = None
                    n_shots = 1
                    if cfg.asd_dilate_respect_shots:
                        shot_ids = detect_shot_ids(
                            frame_list[:video_num],
                            hist_threshold=cfg.shot_cut_hist_threshold,
                        )
                        n_shots = (max(shot_ids) + 1) if shot_ids else 1
                        logger.info(
                            "Shot cuts for ASD dilate: %d shots (hist_threshold=%.2f)",
                            n_shots,
                            cfg.shot_cut_hist_threshold,
                        )
                    speaking_mask = dilate_speaking_mask(
                        speaking_mask,
                        cfg.asd_mask_dilate,
                        valid_face=valid_face,
                        shot_ids=shot_ids,
                    )
                    if cfg.asd_pre_dilate_min_sec > 0:
                        before_post = sum(speaking_mask)
                        speaking_mask = prune_short_speaking_runs(
                            speaking_mask,
                            min_duration_sec=cfg.asd_pre_dilate_min_sec,
                            fps=fps,
                            shot_ids=shot_ids,
                        )
                        logger.info(
                            "Post-dilate prune: drop runs < %.2fs → %d/%d (was %d)",
                            cfg.asd_pre_dilate_min_sec,
                            sum(speaking_mask),
                            len(speaking_mask),
                            before_post,
                        )
                speaking_frames = sum(speaking_mask)
                logger.info(
                    "LR-ASD: %d/%d speaking after dilate±%d (raw=%d, threshold=%s, shot_aware=%s)",
                    speaking_frames,
                    len(speaking_mask),
                    cfg.asd_mask_dilate,
                    raw_speaking,
                    cfg.asd_threshold,
                    cfg.asd_dilate_respect_shots,
                )
            else:
                speaking_mask = [True] * video_num
                speaking_frames = video_num

            # Speaking detection stays as-is above. For lipsync, expand the
            # gate to whole camera shots that contain any speaking frame.
            if (
                cfg.lipsync_full_speaking_shots
                and speaking_mask is not None
                and speaking_frames < video_num
            ):
                shot_ids_for_expand = None
                if fusion_meta is not None and fusion_meta.get("shot_ids") is not None:
                    shot_ids_for_expand = list(fusion_meta["shot_ids"])[:video_num]
                else:
                    shot_ids_for_expand = detect_shot_ids(
                        frame_list[:video_num],
                        hist_threshold=cfg.shot_cut_hist_threshold,
                    )
                n_shots = (max(shot_ids_for_expand) + 1) if shot_ids_for_expand else 0
                before_expand = speaking_frames
                speaking_mask = expand_speaking_mask_to_shots(
                    speaking_mask,
                    shot_ids_for_expand,
                    min_speak_duration_sec=cfg.lipsync_shot_min_speak_sec,
                    short_shot_sec=cfg.lipsync_short_shot_sec,
                    short_shot_min_speak_sec=cfg.lipsync_short_shot_min_speak_sec,
                    keep_partial_min_sec=cfg.lipsync_keep_partial_min_sec,
                    fps=fps,
                )
                speaking_frames = sum(speaking_mask)
                speak_shot_ids = {
                    int(shot_ids_for_expand[i])
                    for i, v in enumerate(speaking_mask)
                    if v
                }
                # Log time spans of lipsync coverage (full shots and/or partial runs).
                shot_spans: list[str] = []
                for sid in sorted(speak_shot_ids):
                    idxs = [
                        i
                        for i, s in enumerate(shot_ids_for_expand)
                        if int(s) == sid and speaking_mask[i]
                    ]
                    if not idxs:
                        continue
                    t0 = idxs[0] / float(fps)
                    t1 = (idxs[-1] + 1) / float(fps)
                    full = all(
                        speaking_mask[i]
                        for i, s in enumerate(shot_ids_for_expand)
                        if int(s) == sid
                    )
                    tag = "full" if full else "partial"
                    shot_spans.append(f"{sid}:{t0:.2f}-{t1:.2f}s({tag})")
                logger.info(
                    "Lipsync by shot: %d/%d shots "
                    "(min_run>=%.2fs, short<%.2fs→%.2fs, partial>=%.2fs) → "
                    "expand gate %d → %d/%d frames | shots=[%s]",
                    len(speak_shot_ids),
                    n_shots,
                    cfg.lipsync_shot_min_speak_sec,
                    cfg.lipsync_short_shot_sec,
                    cfg.lipsync_short_shot_min_speak_sec,
                    cfg.lipsync_keep_partial_min_sec,
                    before_expand,
                    speaking_frames,
                    video_num,
                    ", ".join(shot_spans),
                )

            infer_indices = []
            frame_h0, frame_w0 = frame_list[0].shape[:2]
            min_area = cfg.min_face_area_ratio * float(frame_w0 * frame_h0)
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
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                infer_indices.append(i)

            if not infer_indices and not any(b != coord_placeholder for b in coord_list):
                raise ValueError("No valid face detected in the input video")

            res_by_index: dict[int, np.ndarray] = {}
            full_frame_by_index: dict[int, np.ndarray] = {}
            if infer_indices and self._use_latentsync:
                if self.latentsync is None:
                    raise RuntimeError("LatentSync backend is not loaded")
                full_frame_by_index = self.latentsync.lipsync_indices(
                    frame_list=frame_list,
                    audio_path=audio_path,
                    infer_indices=infer_indices,
                    fps=float(fps),
                    temp_dir=os.path.join(temp_dir, "latentsync"),
                )
            elif infer_indices:
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

            logger.info("Compositing frames with ffmpeg rawvideo pipe (libx264)")
            lipsync_frames = 0
            height, width = frame_list[0].shape[:2]
            temp_vid_path = os.path.join(temp_dir, f"temp_{output_basename}.mp4")

            # LatentSync already pastes lips onto full frames; composite is a
            # full-frame replace with optional edge blend. MuseTalk keeps the
            # face-crop + CodeFormer + parsing blend path.
            if self._use_latentsync:
                active_mask = [i in full_frame_by_index for i in range(video_num)]
                blend_alphas = compute_segment_blend_alphas(
                    active_mask, ramp_frames=cfg.blend_ramp_frames
                )
                if cfg.blend_ramp_frames > 0:
                    logger.info(
                        "LatentSync composite: blend_ramp_frames=%d, synced=%d/%d",
                        cfg.blend_ramp_frames,
                        len(full_frame_by_index),
                        video_num,
                    )
                with FFmpegRawVideoWriter(
                    temp_vid_path,
                    width=width,
                    height=height,
                    fps=float(fps),
                    crf=18,
                    preset="veryfast",
                ) as writer:
                    for i in tqdm(range(video_num)):
                        ori_frame = frame_list[i]
                        synced = full_frame_by_index.get(i)
                        if synced is None:
                            writer.write(ori_frame)
                            continue
                        alpha = blend_alphas[i] if i < len(blend_alphas) else 1.0
                        if alpha <= 0.001:
                            writer.write(ori_frame)
                            continue
                        combine_frame = synced
                        if combine_frame.shape[0] != height or combine_frame.shape[1] != width:
                            combine_frame = cv2.resize(
                                combine_frame, (width, height), interpolation=cv2.INTER_LANCZOS4
                            )
                        if alpha < 0.999:
                            combine_frame = blend_frames(ori_frame, combine_frame, alpha)
                        lipsync_frames += 1
                        writer.write(combine_frame)
            else:
                # Batch CodeFormer restore before the per-frame blend loop.
                # Stride reuse must stay on the same face track: if bbox IoU drops or
                # speaking frames are far apart, force a fresh restore (never paste the
                # previous person's restored face onto a new bbox).
                restored_by_index: dict[int, np.ndarray] = {}
                if res_by_index and self.codeformer is not None:
                    infer_keys = sorted(res_by_index.keys())
                    stride = max(1, int(cfg.codeformer_stride))
                    cf_reuse_min_iou = 0.35
                    cf_max_frame_gap = max(stride * 3, 6)

                    restore_keys: list[int] = []
                    last_key = None
                    for j, idx in enumerate(infer_keys):
                        need = (j % stride == 0) or (j == len(infer_keys) - 1)
                        if last_key is not None and not need:
                            gap = idx - last_key
                            iou = _bbox_iou(coord_list[last_key], coord_list[idx])
                            if gap > cf_max_frame_gap or iou < cf_reuse_min_iou:
                                need = True
                        if need:
                            restore_keys.append(idx)
                            last_key = idx

                    face_batch = [res_by_index[i].astype(np.uint8) for i in restore_keys]
                    logger.info(
                        "CodeFormer restoring %d/%d faces (fp16=%s, batch_size=%d, stride=%d)",
                        len(face_batch),
                        len(infer_keys),
                        self.codeformer.use_fp16,
                        self.codeformer.batch_size,
                        stride,
                    )
                    restored_faces = self.codeformer.restore_faces(face_batch)
                    restored_map = dict(zip(restore_keys, restored_faces))
                    last_face = None
                    last_key = None
                    for idx in infer_keys:
                        if idx in restored_map:
                            last_face = restored_map[idx]
                            last_key = idx
                            restored_by_index[idx] = last_face
                            continue
                        can_reuse = (
                            last_face is not None
                            and last_key is not None
                            and (idx - last_key) <= cf_max_frame_gap
                            and _bbox_iou(coord_list[last_key], coord_list[idx])
                            >= cf_reuse_min_iou
                        )
                        if can_reuse:
                            restored_by_index[idx] = last_face
                        else:
                            # Identity/track changed: keep this frame's MuseTalk face.
                            restored_by_index[idx] = res_by_index[idx].astype(np.uint8)
                            last_face = None
                            last_key = None
                else:
                    restored_by_index = {
                        i: frame.astype(np.uint8) for i, frame in res_by_index.items()
                    }

                with FFmpegRawVideoWriter(
                    temp_vid_path,
                    width=width,
                    height=height,
                    fps=float(fps),
                    crf=18,
                    preset="veryfast",
                ) as writer:
                    active_mask = [i in restored_by_index for i in range(video_num)]
                    blend_alphas = compute_segment_blend_alphas(
                        active_mask, ramp_frames=cfg.blend_ramp_frames
                    )
                    if cfg.blend_ramp_frames > 0 or cfg.use_color_match:
                        logger.info(
                            "Composite smoothing: color_match=%s, blend_ramp_frames=%d",
                            cfg.use_color_match,
                            cfg.blend_ramp_frames,
                        )

                    for i in tqdm(range(video_num)):
                        bbox = coord_list[i]
                        ori_frame = frame_list[i]
                        face = restored_by_index.get(i)
                        # No face / invalid box: keep original frame (avoid ghost paste).
                        if (
                            face is None
                            or bbox == coord_placeholder
                            or bbox[2] <= bbox[0]
                            or bbox[3] <= bbox[1]
                        ):
                            writer.write(ori_frame)
                            continue

                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        if cfg.version == "v15":
                            y2 = min(y2 + cfg.extra_margin, ori_frame.shape[0])
                        if x2 <= x1 or y2 <= y1:
                            writer.write(ori_frame)
                            continue

                        alpha = blend_alphas[i] if i < len(blend_alphas) else 1.0
                        if alpha <= 0.001:
                            writer.write(ori_frame)
                            continue

                        try:
                            ori_copy = ori_frame.copy()
                            res_frame = cv2.resize(face, (x2 - x1, y2 - y1))
                            if cfg.use_color_match:
                                ori_crop = ori_frame[y1:y2, x1:x2]
                                if ori_crop.size > 0:
                                    res_frame = match_face_color_lab(res_frame, ori_crop)
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
                            if alpha < 0.999:
                                combine_frame = blend_frames(ori_frame, combine_frame, alpha)
                            lipsync_frames += 1
                        except Exception:
                            combine_frame = ori_frame

                        writer.write(combine_frame)

            if audio_mux_source and has_audio_stream(audio_mux_source):
                logger.info("Muxing input audio into final output: %s", audio_mux_source)
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
