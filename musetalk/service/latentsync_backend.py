"""LatentSync 1.6 lipsync backend for speaking-gated segments.

Keeps the V1 speaking / face / shot gate untouched. Only replaces MuseTalk
frame generation: contiguous speaking runs are exported as short clips,
run through LatentSync, then mapped back as full frames.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def _bbox_iou(a, b) -> float:
    """IoU of two (x1,y1,x2,y2) boxes; 0.0 for None/placeholder/degenerate."""
    try:
        ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
        bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except (TypeError, ValueError, IndexError):
        return 0.0
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def cleanup_runs(
    runs: List[Tuple[int, int]],
    *,
    min_run_frames: int,
    merge_gap_frames: int,
    coord_list: Optional[Sequence] = None,
    shot_ids: Optional[Sequence[int]] = None,
    merge_iou_min: float = 0.5,
) -> List[Tuple[int, int]]:
    """Merge short gaps then drop micro-runs to cut per-run fixed overhead.

    Two morphological passes over inclusive (start, end) runs:
      1. close: join neighbours separated by <= ``merge_gap_frames`` non-speaking
         frames, but ONLY when the two runs stay inside the same camera shot.
         A LatentSync clip is denoised as ONE face across its whole span, so
         stitching a shot cut / person change into a single run drives the wrong
         mouth across the cut. ``shot_ids`` (from detect_shot_ids, HSV-histogram
         + grayscale MAE) is the authoritative gate and is robust to same-framing
         cuts where bbox IoU would stay high; when it is unavailable we fall back
         to a bbox-IoU continuity heuristic over ``coord_list``.
      2. open: drop any run still shorter than ``min_run_frames`` (isolated
         micro-segments like a 3-frame word tail -> kept as the original plate).
    Set both knobs to 0 to disable and reproduce the raw contiguous-run layout.
    """
    if not runs:
        return runs
    merge_gap = max(0, int(merge_gap_frames))
    min_len = max(0, int(min_run_frames))
    if merge_gap == 0 and min_len <= 1:
        return list(runs)

    def _same_face(prev_end: int, next_start: int) -> bool:
        # Primary gate: both boundary frames must be in the same shot. Unknown
        # shot id (out of range) -> treat as different to stay safe.
        if shot_ids is not None:
            a = shot_ids[prev_end] if 0 <= prev_end < len(shot_ids) else None
            b = shot_ids[next_start] if 0 <= next_start < len(shot_ids) else None
            return a is not None and a == b
        # Fallback (no shot info): keep the old (unguarded) temporal-only merge
        # when there is no face data or a non-positive threshold.
        if coord_list is None or merge_iou_min <= 0:
            return True
        a = coord_list[prev_end] if 0 <= prev_end < len(coord_list) else None
        b = coord_list[next_start] if 0 <= next_start < len(coord_list) else None
        return _bbox_iou(a, b) >= merge_iou_min

    merged: List[List[int]] = [[int(runs[0][0]), int(runs[0][1])]]
    for s, e in runs[1:]:
        s, e = int(s), int(e)
        if (
            s - merged[-1][1] - 1 <= merge_gap
            and _same_face(merged[-1][1], s)
        ):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kept = [(s, e) for (s, e) in merged if (e - s + 1) >= min_len]
    return kept


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
    """Load LatentSync 1.6 once and lipsync speaking clips."""

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
        fastpath: bool = True,
        compile_unet: bool = False,
        gen_resolution: Optional[int] = None,
        clip_preset: str = "ultrafast",
    ):
        self.device = device
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.enable_deepcache = bool(enable_deepcache)
        self.seed = int(seed)
        self.use_float16 = bool(use_float16)
        self.fastpath = bool(fastpath)
        self.compile_unet = bool(compile_unet)
        # Optional override of the face-crop generation resolution; None/0 keeps
        # the model's trained resolution (512). Lower = big speedup, lower quality.
        self.gen_resolution = int(gen_resolution) if gen_resolution else None
        # x264 preset for the transient per-run clip mp4s (input/output) that the
        # pipeline round-trips. These files are discarded, so use the fastest
        # preset; crf is kept low so face quality fed into the UNet is preserved.
        self.clip_preset = str(clip_preset)
        # Sliding-window stride for the pipeline (frames between window starts).
        # None/>=num_frames = original non-overlapping 16-frame blocks (each
        # frame denoised once). A smaller stride overlaps consecutive windows
        # and averages the overlap, bridging the block-boundary seams that make
        # the mouth jump during fast motion. Cost grows ~num_frames/stride.
        # Kill-switch env: MUSETALK_LATENTSYNC_STRIDE (empty/0 keeps default).
        _stride_env = os.environ.get("MUSETALK_LATENTSYNC_STRIDE", "").strip()
        self.window_stride = int(_stride_env) if _stride_env and int(_stride_env) > 0 else None

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

        if self.fastpath and self.device.type == "cuda":
            # Speaking clips are near-fixed 16-frame shapes, so the cuDNN
            # autotuner + TF32 pay off across the many denoise steps.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("LatentSync fastpath enabled: TF32 + cudnn.benchmark")

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
            "Loading LatentSync 1.6 on %s (res=%d, steps=%d, guidance=%.2f, fp16=%s)",
            self.device,
            self.resolution,
            self.inference_steps,
            self.guidance_scale,
            is_fp16,
        )
        _t_load = time.perf_counter()

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
            logger.info(
                "[timing] latentsync whisper/audio_encoder load: %.3fs",
                time.perf_counter() - _t_load,
            )
            _t_load = time.perf_counter()

            vae = AutoencoderKL.from_pretrained(self.paths.vae_path, torch_dtype=dtype)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0
            # VRAM: decode per-sample slices to cap the VAE activation peak.
            # Slicing is lossless (2D VAE is per-sample; measured <=3/255 and
            # frame-uniform). Do NOT enable tiling here: this sd-vae has
            # config.sample_size=256 -> tile_latent_min_size=32, so our 64x64
            # latent (512 output) WOULD trigger tiled_decode and its tile-blend
            # seams introduce visible patches/flicker (measured ~26/255). Keep
            # tiling explicitly off. Kill-switch: MUSETALK_LATENTSYNC_VAE_SLICING=0.
            if os.environ.get("MUSETALK_LATENTSYNC_VAE_SLICING", "1") != "0":
                try:
                    if hasattr(vae, "enable_slicing"):
                        vae.enable_slicing()
                    logger.info("LatentSync VAE slicing enabled (tiling kept off)")
                except Exception as exc:
                    logger.warning("LatentSync VAE slicing disabled: %s", exc)
            if hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
            logger.info(
                "[timing] latentsync vae load: %.3fs", time.perf_counter() - _t_load
            )
            _t_load = time.perf_counter()

            unet, load_info = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(cfg.model),
                self.paths.checkpoint,
                device="cpu",
            )
            unet = unet.to(dtype=dtype)
            logger.info(
                "[timing] latentsync unet weight load (%s): %.3fs",
                os.path.basename(os.path.dirname(self.paths.checkpoint)),
                time.perf_counter() - _t_load,
            )
            # Guard against a config/weights mismatch (e.g. 1.6 weights loaded
            # with the wrong resolution/arch yaml): surface any partial load.
            try:
                if isinstance(load_info, dict):
                    missing = list(load_info.get("missing_keys", []))
                    unexpected = list(load_info.get("unexpected_keys", []))
                else:
                    missing = list(getattr(load_info, "missing_keys", []) or [])
                    unexpected = list(getattr(load_info, "unexpected_keys", []) or [])
                if missing or unexpected:
                    logger.warning(
                        "LatentSync UNet load mismatch: missing=%d unexpected=%d "
                        "(missing[:5]=%s unexpected[:5]=%s)",
                        len(missing),
                        len(unexpected),
                        missing[:5],
                        unexpected[:5],
                    )
                else:
                    logger.info(
                        "LatentSync UNet weights loaded clean (0 missing / 0 unexpected)"
                    )
            except Exception as exc:
                logger.warning("Could not inspect LatentSync UNet load info: %s", exc)
            _t_load = time.perf_counter()

            pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to(self.device)
            logger.info(
                "[timing] latentsync pipeline build+to(gpu): %.3fs",
                time.perf_counter() - _t_load,
            )

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

            if self.compile_unet:
                try:
                    # dynamic=True lets one compilation serve the variable
                    # last-chunk frame counts without recompiling per shape.
                    pipeline.unet = torch.compile(unet, dynamic=True)
                    logger.info("LatentSync UNet torch.compile enabled (dynamic)")
                except Exception as exc:
                    pipeline.unet = unet
                    logger.warning("LatentSync torch.compile disabled: %s", exc)

            if self.seed >= 0:
                set_seed(self.seed)

            self.pipeline = pipeline
        finally:
            os.chdir(prev_cwd)

        logger.info("LatentSync 1.6 models loaded")

    def _get_face_detector(self):
        """Return the same InsightFace FaceDetector LatentSync affine uses."""
        fd = getattr(self.pipeline, "_face_detector", None)
        if fd is not None:
            return fd
        prev_cwd = os.getcwd()
        os.chdir(self.paths.repo_root)
        try:
            from latentsync.utils.face_detector import FaceDetector

            face_device = str(self.device) if self.device.type == "cuda" else "cuda"
            fd = FaceDetector(device=face_device)
            self.pipeline._face_detector = fd
            logger.info("LatentSync FaceDetector lazy-initialized for pre-filter")
            return fd
        except Exception as exc:
            logger.warning("LatentSync FaceDetector unavailable for pre-filter: %s", exc)
            return None
        finally:
            os.chdir(prev_cwd)

    def _filter_indices_with_faces(
        self,
        frame_list: List[np.ndarray],
        infer_indices: Sequence[int],
        coord_list: Optional[Sequence] = None,
        primary_iou_min: float = 0.3,
    ) -> List[int]:
        """Drop frames LatentSync cannot safely align to the MuseTalk primary.

        Uses RGB input to match lipsync_pipeline.read_video → affine_transform.
        When ``coord_list`` is provided, also require the InsightFace box to
        overlap the MuseTalk primary track (avoids lipsyncing a secondary face).
        Filtered indices keep timeline originals (no temporal jump).
        """
        ordered = sorted(int(i) for i in infer_indices)
        if not ordered:
            return []

        fd = self._get_face_detector()
        if fd is None:
            return ordered

        from musetalk.utils.preprocessing import _bbox_iou, coord_placeholder

        kept: List[int] = []
        dropped_noface: List[int] = []
        dropped_mismatch: List[int] = []
        min_iou = float(primary_iou_min) if primary_iou_min is not None else 0.0
        for idx in ordered:
            if idx < 0 or idx >= len(frame_list):
                dropped_noface.append(idx)
                continue
            rgb = cv2.cvtColor(frame_list[idx], cv2.COLOR_BGR2RGB)
            # Prefer raw detection bbox for IoU vs MuseTalk track.
            if hasattr(fd, "detect_primary_det_bbox"):
                bbox = fd.detect_primary_det_bbox(rgb)
            else:
                bbox, _ = fd(rgb)
            if bbox is None:
                dropped_noface.append(idx)
                continue
            if coord_list is not None and idx < len(coord_list) and min_iou > 0:
                hint = coord_list[idx]
                if hint is None or hint == coord_placeholder:
                    dropped_mismatch.append(idx)
                    continue
                if _bbox_iou(bbox, hint) < min_iou:
                    dropped_mismatch.append(idx)
                    continue
            kept.append(idx)

        if dropped_noface or dropped_mismatch:
            preview_nf = dropped_noface[:8]
            preview_mm = dropped_mismatch[:8]
            more_nf = (
                ""
                if len(dropped_noface) <= 8
                else f" …(+{len(dropped_noface) - 8})"
            )
            more_mm = (
                ""
                if len(dropped_mismatch) <= 8
                else f" …(+{len(dropped_mismatch) - 8})"
            )
            logger.info(
                "LatentSync FaceDetector pre-filter: keep %d/%d "
                "(no-face=%d %s%s, primary-mismatch=%d %s%s, iou_min=%.2f)",
                len(kept),
                len(ordered),
                len(dropped_noface),
                preview_nf,
                more_nf,
                len(dropped_mismatch),
                preview_mm,
                more_mm,
                min_iou,
            )
        return kept

    def lipsync_indices(
        self,
        frame_list: List[np.ndarray],
        audio_path: str,
        infer_indices: Sequence[int],
        fps: float,
        temp_dir: str,
        coord_list: Optional[Sequence] = None,
        primary_iou_min: float = 0.3,
        stream_cb: Optional[Callable[[int, int, Dict[int, np.ndarray]], None]] = None,
    ) -> Dict[int, np.ndarray]:
        """Run LatentSync on contiguous speaking runs; return full BGR frames.

        If ``stream_cb`` is provided, it is invoked right after each run finishes
        with ``(start, end, run_frames)`` so the caller can composite that run
        while the next run's GPU inference is still running (CPU/GPU overlap).
        The full ``out`` dict is still returned for backward compatibility.
        """
        _t_ff = time.perf_counter()
        faced_indices = self._filter_indices_with_faces(
            frame_list,
            infer_indices,
            coord_list=coord_list,
            primary_iou_min=primary_iou_min,
        )
        face_filter_sec = time.perf_counter() - _t_ff
        logger.info(
            "[timing] latentsync face pre-filter: %.3fs (%d/%d kept)",
            face_filter_sec,
            len(faced_indices),
            len(infer_indices),
        )
        runs = contiguous_runs(faced_indices)
        # Merge micro gaps + drop tiny runs: each pipeline call carries ~2-3s of
        # step-independent overhead, so 3-6 frame bursts are costly and can read
        # as a stray mouth flap. Merging is shot-gated (both boundary frames must
        # fall in the same camera shot per detect_shot_ids) so a cut / person
        # change inside the gap is NOT stitched into one clip, which would drive
        # the wrong mouth across the cut. Kill-switches (0 disables the pass):
        #   MUSETALK_LATENTSYNC_MIN_RUN_FRAMES   (default 6)
        #   MUSETALK_LATENTSYNC_MERGE_GAP_FRAMES (default 4)
        #   MUSETALK_LATENTSYNC_MERGE_USE_SHOT   (default 1; 0 -> bbox-IoU gate)
        #   MUSETALK_LATENTSYNC_MERGE_IOU_MIN    (fallback gate, default 0.5)
        _min_run = int(os.environ.get("MUSETALK_LATENTSYNC_MIN_RUN_FRAMES", "6") or 0)
        _merge_gap = int(os.environ.get("MUSETALK_LATENTSYNC_MERGE_GAP_FRAMES", "4") or 0)
        _merge_iou = float(os.environ.get("MUSETALK_LATENTSYNC_MERGE_IOU_MIN", "0.5") or 0)
        _use_shot = os.environ.get("MUSETALK_LATENTSYNC_MERGE_USE_SHOT", "1") != "0"
        shot_ids: Optional[List[int]] = None
        if _use_shot and _merge_gap > 0 and len(runs) > 1:
            try:
                from musetalk.utils.active_speaker import detect_shot_ids

                shot_ids = detect_shot_ids(frame_list)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("LatentSync merge shot-gate unavailable (%s); using bbox", exc)
                shot_ids = None
        _raw_frames = sum(e - s + 1 for s, e in runs)
        _n_raw = len(runs)
        runs = cleanup_runs(
            runs,
            min_run_frames=_min_run,
            merge_gap_frames=_merge_gap,
            coord_list=coord_list,
            shot_ids=shot_ids,
            merge_iou_min=_merge_iou,
        )
        _kept_frames = sum(e - s + 1 for s, e in runs)
        if _kept_frames != _raw_frames or len(runs) != _n_raw:
            logger.info(
                "LatentSync run cleanup: %d run(s)/%d frames -> %d run(s)/%d frames "
                "(min_run=%d merge_gap=%d gate=%s)",
                _n_raw,
                _raw_frames,
                len(runs),
                _kept_frames,
                _min_run,
                _merge_gap,
                ("shot" if shot_ids is not None else f"iou{_merge_iou:.2f}"),
            )
        if not runs:
            if infer_indices:
                logger.warning(
                    "LatentSync: all %d speaking frames lack a matching primary "
                    "face; keeping originals",
                    len(infer_indices),
                )
            return {}

        fps = float(fps) if fps and fps > 0 else 25.0
        os.makedirs(temp_dir, exist_ok=True)
        out: Dict[int, np.ndarray] = {}
        io_sec = 0.0
        infer_sec = 0.0
        logger.info(
            "LatentSync: %d speaking frames (%d after face filter) in %d contiguous run(s)",
            len(infer_indices),
            len(faced_indices),
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
            _t_io = time.perf_counter()
            with FFmpegRawVideoWriter(
                clip_video, width=w, height=h, fps=float(fps), crf=18, preset=self.clip_preset
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
            io_sec += time.perf_counter() - _t_io

            logger.info(
                "LatentSync run %d/%d: frames [%d, %d] (%d @ %.3ffps)",
                run_i + 1,
                len(runs),
                start,
                end,
                n,
                fps,
            )
            _t_inf = time.perf_counter()
            synced = self._infer_clip(
                clip_video, clip_audio, clip_out, clip_dir, fps=fps, expect_frames=n
            )
            infer_sec += time.perf_counter() - _t_inf
            if not synced:
                logger.warning(
                    "LatentSync produced no frames for run [%d, %d]; keeping originals",
                    start,
                    end,
                )
                continue

            # Never write more than the source run length; drop any pad/overshoot.
            use_n = min(n, len(synced))
            if len(synced) != n:
                logger.warning(
                    "LatentSync run [%d, %d] returned %d frames (expected %d); using %d",
                    start,
                    end,
                    len(synced),
                    n,
                    use_n,
                )
            run_frames: Dict[int, np.ndarray] = {}
            for offset in range(use_n):
                frame = synced[offset]
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
                run_frames[start + offset] = frame
            out.update(run_frames)
            if stream_cb is not None:
                stream_cb(start, end, run_frames)

        logger.info(
            "[timing] latentsync summary: face_filter=%.2fs clip_io=%.2fs "
            "pipeline_infer=%.2fs (runs=%d, sync_frames=%d)",
            face_filter_sec,
            io_sec,
            infer_sec,
            len(runs),
            len(out),
        )
        # VRAM: release all GPU-side cache blocks once generation is done so the
        # CPU composite/encode/mux phases (and idle time between requests) do not
        # keep the allocator holding memory on this shared GPU.
        if os.environ.get("MUSETALK_LATENTSYNC_EMPTY_CACHE", "1") != "0" \
                and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def _infer_clip(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        temp_dir: str,
        *,
        fps: float = 25.0,
        expect_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        prev_cwd = os.getcwd()
        os.chdir(self.paths.repo_root)
        try:
            # Keep whisper chunking on the same fps as the clip we wrote.
            video_fps = float(fps) if fps and fps > 0 else 25.0
            gen_res = self.gen_resolution or self.resolution
            _t_pipe = time.perf_counter()
            self.pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=video_out_path,
                num_frames=self.num_frames,
                stride=self.window_stride,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                weight_dtype=self.weight_dtype,
                width=gen_res,
                height=gen_res,
                mask_image_path=self.paths.mask_image,
                temp_dir=os.path.join(temp_dir, "pipeline_tmp"),
                video_fps=video_fps,
            )
            logger.info(
                "[timing] latentsync pipeline (%d frames, res=%d, steps=%d, stride=%s): %.3fs",
                expect_frames if expect_frames else -1,
                gen_res,
                self.inference_steps,
                self.window_stride if self.window_stride else "block",
                time.perf_counter() - _t_pipe,
            )
        except Exception:
            logger.exception("LatentSync inference failed for %s", video_path)
            return []
        finally:
            os.chdir(prev_cwd)
            # VRAM: hand this run's freed caching-allocator segments back to the
            # driver before the next run so our held peak does not ratchet up
            # (matters on this shared/multi-tenant GPU). Kill-switch env:
            # MUSETALK_LATENTSYNC_EMPTY_CACHE=0.
            if os.environ.get("MUSETALK_LATENTSYNC_EMPTY_CACHE", "1") != "0" \
                    and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not os.path.isfile(video_out_path):
            return []
        _t_dec = time.perf_counter()
        frames, _ = decode_video_frames(video_out_path)
        logger.info(
            "[timing] latentsync output decode (%d frames): %.3fs",
            len(frames),
            time.perf_counter() - _t_dec,
        )
        if expect_frames is not None and len(frames) > int(expect_frames):
            frames = frames[: int(expect_frames)]
        return frames
