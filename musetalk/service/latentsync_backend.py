"""LatentSync 1.5 lipsync backend for speaking-gated segments.

Keeps the V1 speaking / face / shot gate untouched. Only replaces MuseTalk
frame generation: contiguous speaking runs are exported as short clips,
run through LatentSync, then mapped back as full frames.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from musetalk.service.long_video import (
    FFmpegRawVideoWriter,
    decode_video_frames,
    split_audio_segment_frames,
)

logger = logging.getLogger("musetalk_service")

_REPO_DEFAULT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "LatentSync")
)


def contiguous_runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Group sorted frame indices into inclusive (start, end) runs."""
    if not indices:
        return []
    ordered = sorted(int(i) for i in indices)
    runs: List[Tuple[int, int]] = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


@dataclass
class LatentSyncPaths:
    repo_root: str
    unet_config: str
    checkpoint: str
    whisper_tiny: str
    vae_path: str
    mask_image: str
    scheduler_config_dir: str


class LatentSyncBackend:
    """Load LatentSync 1.5 once and lipsync speaking clips."""

    def __init__(
        self,
        *,
        device: torch.device,
        repo_root: str = _REPO_DEFAULT,
        unet_config: Optional[str] = None,
        checkpoint: str = "./models/latentsync15/latentsync_unet.pt",
        whisper_tiny: str = "./models/latentsync15/whisper/tiny.pt",
        vae_path: str = "./models/sd-vae",
        inference_steps: int = 20,
        guidance_scale: float = 1.5,
        enable_deepcache: bool = True,
        seed: int = 1247,
        use_float16: bool = True,
    ):
        self.device = device
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.enable_deepcache = bool(enable_deepcache)
        self.seed = int(seed)
        self.use_float16 = bool(use_float16)

        repo_root = os.path.abspath(repo_root)
        if not os.path.isdir(repo_root):
            raise FileNotFoundError(f"LatentSync repo not found: {repo_root}")

        unet_config = os.path.abspath(
            unet_config
            or os.path.join(repo_root, "configs", "unet", "stage2.yaml")
        )
        checkpoint = os.path.abspath(checkpoint)
        whisper_tiny = os.path.abspath(whisper_tiny)
        vae_path = os.path.abspath(vae_path)
        mask_image = os.path.join(repo_root, "latentsync", "utils", "mask.png")
        scheduler_config_dir = os.path.join(repo_root, "configs")

        for path, label in (
            (unet_config, "unet config"),
            (checkpoint, "unet checkpoint"),
            (whisper_tiny, "whisper tiny"),
            (vae_path, "VAE"),
            (mask_image, "mask image"),
            (scheduler_config_dir, "scheduler config dir"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"LatentSync {label} not found: {path}")

        self.paths = LatentSyncPaths(
            repo_root=repo_root,
            unet_config=unet_config,
            checkpoint=checkpoint,
            whisper_tiny=whisper_tiny,
            vae_path=vae_path,
            mask_image=mask_image,
            scheduler_config_dir=scheduler_config_dir,
        )
        self._ensure_repo_on_path()
        self._load_pipeline()

    def _ensure_repo_on_path(self) -> None:
        root = self.paths.repo_root
        if root not in sys.path:
            sys.path.insert(0, root)

    def _load_pipeline(self) -> None:
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        from accelerate.utils import set_seed
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        cfg = OmegaConf.load(self.paths.unet_config)
        self.config = cfg
        self.resolution = int(cfg.data.resolution)
        self.num_frames = int(cfg.data.num_frames)

        is_fp16 = (
            self.use_float16
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device.index or 0)[0] > 7
        )
        dtype = torch.float16 if is_fp16 else torch.float32
        self.weight_dtype = dtype

        logger.info(
            "Loading LatentSync 1.5 on %s (res=%d, steps=%d, guidance=%.2f, fp16=%s)",
            self.device,
            self.resolution,
            self.inference_steps,
            self.guidance_scale,
            is_fp16,
        )

        # FaceDetector resolves checkpoints/auxiliary relative to CWD.
        prev_cwd = os.getcwd()
        os.chdir(self.paths.repo_root)
        try:
            scheduler = DDIMScheduler.from_pretrained(self.paths.scheduler_config_dir)

            cross_dim = int(cfg.model.cross_attention_dim)
            if cross_dim == 384:
                whisper_path = self.paths.whisper_tiny
            elif cross_dim == 768:
                whisper_path = self.paths.whisper_tiny.replace("tiny.pt", "small.pt")
            else:
                raise NotImplementedError(
                    f"Unsupported cross_attention_dim={cross_dim}"
                )

            audio_encoder = Audio2Feature(
                model_path=whisper_path,
                device=str(self.device),
                num_frames=self.num_frames,
                audio_feat_length=list(cfg.data.audio_feat_length),
            )

            vae = AutoencoderKL.from_pretrained(self.paths.vae_path, torch_dtype=dtype)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0

            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(cfg.model),
                self.paths.checkpoint,
                device="cpu",
            )
            unet = unet.to(dtype=dtype)

            pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to(self.device)

            # Warm FaceDetector once while CWD is LatentSync root so speaking
            # clips reuse InsightFace instead of reloading buffalo_l each run.
            try:
                from latentsync.utils.face_detector import FaceDetector

                face_device = str(self.device) if self.device.type == "cuda" else "cuda"
                pipeline._face_detector = FaceDetector(device=face_device)
                logger.info("LatentSync FaceDetector cached for reuse")
            except Exception as exc:
                logger.warning("LatentSync FaceDetector warm-up skipped: %s", exc)

            if self.enable_deepcache:
                try:
                    from DeepCache import DeepCacheSDHelper

                    helper = DeepCacheSDHelper(pipe=pipeline)
                    helper.set_params(cache_interval=3, cache_branch_id=0)
                    helper.enable()
                    logger.info("LatentSync DeepCache enabled")
                except Exception as exc:
                    logger.warning("DeepCache disabled: %s", exc)

            if self.seed >= 0:
                set_seed(self.seed)

            self.pipeline = pipeline
        finally:
            os.chdir(prev_cwd)

        logger.info("LatentSync 1.5 models loaded")

    def lipsync_indices(
        self,
        frame_list: List[np.ndarray],
        audio_path: str,
        infer_indices: Sequence[int],
        fps: float,
        temp_dir: str,
    ) -> Dict[int, np.ndarray]:
        """Run LatentSync on contiguous speaking runs; return full BGR frames."""
        runs = contiguous_runs(infer_indices)
        if not runs:
            return {}

        os.makedirs(temp_dir, exist_ok=True)
        out: Dict[int, np.ndarray] = {}
        logger.info(
            "LatentSync: %d speaking frames in %d contiguous run(s)",
            len(infer_indices),
            len(runs),
        )

        for run_i, (start, end) in enumerate(runs):
            n = end - start + 1
            clip_dir = os.path.join(temp_dir, f"ls_run_{run_i:04d}")
            os.makedirs(clip_dir, exist_ok=True)
            clip_video = os.path.join(clip_dir, "input.mp4")
            clip_audio = os.path.join(clip_dir, "input.wav")
            clip_out = os.path.join(clip_dir, "output.mp4")

            h, w = frame_list[start].shape[:2]
            with FFmpegRawVideoWriter(
                clip_video, width=w, height=h, fps=float(fps), crf=18, preset="veryfast"
            ) as writer:
                for i in range(start, end + 1):
                    writer.write(frame_list[i])

            split_audio_segment_frames(
                audio_path,
                start_frame=start,
                frame_count=n,
                fps=float(fps),
                output_path=clip_audio,
                sample_rate=16000,
            )

            logger.info(
                "LatentSync run %d/%d: frames [%d, %d] (%d @ %.3ffps)",
                run_i + 1,
                len(runs),
                start,
                end,
                n,
                fps,
            )
            synced = self._infer_clip(clip_video, clip_audio, clip_out, clip_dir)
            if not synced:
                logger.warning(
                    "LatentSync produced no frames for run [%d, %d]; keeping originals",
                    start,
                    end,
                )
                continue

            # Align lengths: take min(n, len(synced)) from the start of the run.
            use_n = min(n, len(synced))
            if use_n < n:
                logger.warning(
                    "LatentSync run [%d, %d] returned %d/%d frames",
                    start,
                    end,
                    use_n,
                    n,
                )
            for offset in range(use_n):
                frame = synced[offset]
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
                out[start + offset] = frame

        return out

    def _infer_clip(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        temp_dir: str,
    ) -> List[np.ndarray]:
        prev_cwd = os.getcwd()
        os.chdir(self.paths.repo_root)
        try:
            self.pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=video_out_path,
                num_frames=self.num_frames,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                weight_dtype=self.weight_dtype,
                width=self.resolution,
                height=self.resolution,
                mask_image_path=self.paths.mask_image,
                temp_dir=os.path.join(temp_dir, "pipeline_tmp"),
                video_fps=25,
            )
        except Exception:
            logger.exception("LatentSync inference failed for %s", video_path)
            return []
        finally:
            os.chdir(prev_cwd)

        if not os.path.isfile(video_out_path):
            return []
        frames, _ = decode_video_frames(video_out_path)
        return frames
