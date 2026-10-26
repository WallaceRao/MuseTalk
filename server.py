import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from musetalk.service.engine import ServiceConfig
from musetalk.service.engine_pool import EnginePool
from musetalk.service.ffmpeg_env import ensure_ffmpeg_env
from musetalk.service.logging_setup import setup_logging

logger = setup_logging()

_engine_pool: Optional[EnginePool] = None


def _load_service_config() -> ServiceConfig:
    config = ServiceConfig()

    lipsync_backend = os.environ.get("MUSETALK_LIPSYNC_BACKEND")
    if lipsync_backend:
        config.lipsync_backend = lipsync_backend.strip().lower()

    # TBDub Student backend knobs (MUSETALK_LIPSYNC_BACKEND=tbdub, the default).
    tbdub_repo = os.environ.get("MUSETALK_TBDUB_REPO")
    if tbdub_repo:
        config.tbdub_repo = tbdub_repo
    tbdub_ckpt = os.environ.get("MUSETALK_TBDUB_CKPT_DIR")
    if tbdub_ckpt:
        config.tbdub_ckpt_dir = tbdub_ckpt
    tbdub_steps = os.environ.get("MUSETALK_TBDUB_STEPS")
    if tbdub_steps is not None:
        config.tbdub_num_student_steps = max(1, int(tbdub_steps))
    tbdub_sigma = os.environ.get("MUSETALK_TBDUB_SIGMA_SHIFT")
    if tbdub_sigma is not None:
        config.tbdub_sigma_shift = float(tbdub_sigma)
    tbdub_seed = os.environ.get("MUSETALK_TBDUB_SEED")
    if tbdub_seed is not None:
        config.tbdub_seed = int(tbdub_seed)
    tbdub_cpu_offload = os.environ.get("MUSETALK_TBDUB_CPU_OFFLOAD")
    if tbdub_cpu_offload is not None:
        config.tbdub_cpu_offload = tbdub_cpu_offload.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    tbdub_target_fps = os.environ.get("MUSETALK_TBDUB_TARGET_FPS")
    if tbdub_target_fps is not None:
        config.tbdub_target_fps = float(tbdub_target_fps)
    tbdub_min_run = os.environ.get("MUSETALK_TBDUB_MIN_RUN_FRAMES")
    if tbdub_min_run is not None:
        config.tbdub_min_run_frames = max(0, int(tbdub_min_run))
    tbdub_merge_gap = os.environ.get("MUSETALK_TBDUB_MERGE_GAP_FRAMES")
    if tbdub_merge_gap is not None:
        config.tbdub_merge_gap_frames = max(0, int(tbdub_merge_gap))
    # CFG branch weights. NOTE: measured no-ops on the shipped Student model --
    # TBDub's pipeline only reads them under `cfg_merge`, which is set to
    # `not is_student` (False for Student, since guidance is distilled into the
    # weights). Kept so a Teacher checkpoint can be swept without code changes.
    tbdub_ref_cfg = os.environ.get("MUSETALK_TBDUB_REF_CFG_SCALE")
    if tbdub_ref_cfg is not None:
        config.tbdub_ref_cfg_scale = float(tbdub_ref_cfg)
    tbdub_audio_cfg = os.environ.get("MUSETALK_TBDUB_AUDIO_CFG_SCALE")
    if tbdub_audio_cfg is not None:
        config.tbdub_audio_cfg_scale = float(tbdub_audio_cfg)

    # LatentSync model-version selector, evaluated at service startup: choose
    # the 1.6 (512, default) or the 1.5 (256, ~2.7x faster) UNet. Switching the
    # version swaps the unet config + checkpoint + whisper paths together; the
    # generation resolution is read from the config's data.resolution, so no
    # gen-resolution edit is needed. Set MUSETALK_LATENTSYNC_VERSION=1.5|1.6.
    ls_version = os.environ.get("MUSETALK_LATENTSYNC_VERSION", "").strip()
    if ls_version == "1.5":
        config.latentsync_unet_config = "./third_party/LatentSync/configs/unet/stage2.yaml"
        config.latentsync_ckpt = "./models/latentsync15/latentsync_unet.pt"
        config.latentsync_whisper = "./models/latentsync15/whisper/tiny.pt"
    elif ls_version == "1.6":
        config.latentsync_unet_config = "./third_party/LatentSync/configs/unet/stage2_512.yaml"
        config.latentsync_ckpt = "./models/latentsync16/latentsync_unet.pt"
        config.latentsync_whisper = "./models/latentsync16/whisper/tiny.pt"

    # Optional explicit ckpt override (e.g. a custom .pt), applied after version.
    ls_ckpt = os.environ.get("MUSETALK_LATENTSYNC_CKPT")
    if ls_ckpt:
        config.latentsync_ckpt = ls_ckpt

    # Inference steps are LOCKED to 15: validated visually lossless vs 20/30 at
    # guidance 1.2/1.5. Intentionally NOT env-overridable per production policy.
    config.latentsync_inference_steps = 15

    # Guidance default is 1.5 online (see ServiceConfig); still env-overridable.
    ls_guidance = os.environ.get("MUSETALK_LATENTSYNC_GUIDANCE")
    if ls_guidance is not None:
        config.latentsync_guidance_scale = float(ls_guidance)

    stream_composite = os.environ.get("MUSETALK_STREAM_COMPOSITE")
    if stream_composite is not None:
        config.stream_composite = stream_composite.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    max_concurrent = os.environ.get("MUSETALK_MAX_CONCURRENT")
    if max_concurrent is not None:
        config.max_concurrent_requests = max(2, int(max_concurrent))

    gpu_ids_env = os.environ.get("MUSETALK_GPU_IDS")
    if gpu_ids_env:
        config.gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]

    use_cf = os.environ.get("MUSETALK_USE_CODEFORMER")
    if use_cf is not None:
        config.use_codeformer = use_cf.strip().lower() in {"1", "true", "yes", "on"}

    fidelity = os.environ.get("MUSETALK_CODEFORMER_FIDELITY")
    if fidelity is not None:
        config.codeformer_fidelity = float(fidelity)

    cf_path = os.environ.get("MUSETALK_CODEFORMER_MODEL")
    if cf_path:
        config.codeformer_model_path = cf_path

    dilate = os.environ.get("MUSETALK_ASD_MASK_DILATE")
    if dilate is not None:
        config.asd_mask_dilate = max(0, int(dilate))

    vad_expand = os.environ.get("MUSETALK_ASD_VAD_EXPAND_MAX_SEC")
    if vad_expand is not None:
        config.asd_vad_expand_max_sec = max(0.0, float(vad_expand))

    pre_dilate_min = os.environ.get("MUSETALK_ASD_PRE_DILATE_MIN_SEC")
    if pre_dilate_min is not None:
        config.asd_pre_dilate_min_sec = max(0.0, float(pre_dilate_min))

    use_vsdlm = os.environ.get("MUSETALK_USE_VSDLM")
    if use_vsdlm is not None:
        config.use_vsdlm = use_vsdlm.strip().lower() in {"1", "true", "yes", "on"}

    vsdlm_path = os.environ.get("MUSETALK_VSDLM_MODEL")
    if vsdlm_path:
        config.vsdlm_model_path = vsdlm_path

    vsdlm_open = os.environ.get("MUSETALK_VSDLM_OPEN_THRESHOLD")
    if vsdlm_open is not None:
        # mean(open) in the activity window
        config.vsdlm_open_threshold = float(vsdlm_open)

    vsdlm_act = os.environ.get("MUSETALK_VSDLM_ACTIVITY_THRESHOLD")
    if vsdlm_act is not None:
        config.vsdlm_activity_threshold = float(vsdlm_act)

    vsdlm_min_speak = os.environ.get("MUSETALK_VSDLM_MIN_SPEAK_SEC")
    if vsdlm_min_speak is not None:
        config.vsdlm_min_speak_duration_sec = max(0.0, float(vsdlm_min_speak))

    mar_open = os.environ.get("MUSETALK_VSDLM_MAR_OPEN_THRESHOLD")
    if mar_open is not None:
        config.vsdlm_mar_open_threshold = float(mar_open)

    mar_act = os.environ.get("MUSETALK_VSDLM_MAR_ACTIVITY_THRESHOLD")
    if mar_act is not None:
        config.vsdlm_mar_activity_threshold = float(mar_act)

    soft_closed = os.environ.get("MUSETALK_VSDLM_SOFT_CLOSED_MAR")
    if soft_closed is not None:
        config.vsdlm_soft_closed_mar = max(0.0, float(soft_closed))

    # Hard MAR gate for the dubbing mouth-open speaking decision
    # (speaking_mask_op_mar: op>=speak_open_threshold AND mar>=speak_mar_threshold).
    # Lower it (together with MUSETALK_VSDLM_SOFT_CLOSED_MAR) to recover soft /
    # crying / side-face speech whose landmark MAR dips below the 0.12 default.
    speak_mar = os.environ.get("MUSETALK_VSDLM_SPEAK_MAR_THRESHOLD")
    if speak_mar is not None:
        config.vsdlm_speak_mar_threshold = max(0.0, float(speak_mar))

    speak_open = os.environ.get("MUSETALK_VSDLM_SPEAK_OPEN_THRESHOLD")
    if speak_open is not None:
        config.vsdlm_speak_open_threshold = max(0.0, float(speak_open))

    # Small-mouth adaptive MAR bar: lower the speak MAR threshold for frames
    # whose tight mouth pixel width < MUSETALK_VSDLM_SMALL_MOUTH_WIDTH_PX (0 = off).
    sm_width = os.environ.get("MUSETALK_VSDLM_SMALL_MOUTH_WIDTH_PX")
    if sm_width is not None:
        config.vsdlm_small_mouth_width_px = max(0, int(sm_width))
    sm_mar = os.environ.get("MUSETALK_VSDLM_SMALL_MOUTH_MAR_THRESHOLD")
    if sm_mar is not None:
        config.vsdlm_small_mouth_mar_threshold = max(0.0, float(sm_mar))

    # Loose lip-motion gate (AND on top of op+mar state gate): rolling std of
    # VSDLM open-prob >= OP_STD OR landmark MAR std >= MAR_STD within the
    # activity window. Suppresses frozen-open mouths (surprised gasp) that pass
    # the state gate but never move. 0/absent = disabled.
    motion_op = os.environ.get("MUSETALK_VSDLM_MOTION_OP_STD")
    if motion_op is not None:
        config.vsdlm_motion_op_std = max(0.0, float(motion_op))

    motion_mar = os.environ.get("MUSETALK_VSDLM_MOTION_MAR_STD")
    if motion_mar is not None:
        config.vsdlm_motion_mar_std = max(0.0, float(motion_mar))

    # Utterance-level fill: drive a whole (VAD∩shot) region when the primary
    # face is gated speaking for >= this fraction of its face-present frames.
    # Lower it to keep filling a shot whose motion-gated speaking ratio dropped
    # (e.g. soft/crying speech) so short holes like ~33s stay covered.
    shot_fill = os.environ.get("MUSETALK_VSDLM_SHOT_FILL_RATIO")
    if shot_fill is not None:
        config.vsdlm_shot_fill_ratio = max(0.0, min(1.0, float(shot_fill)))

    # Track-agreement cross-check: veto MuseTalk primary false-positives (a
    # closed mouth read as open) by requiring the independent SCRFD track scorer
    # to agree. Off unless MUSETALK_VSDLM_TRACK_AGREE=1.
    ta = os.environ.get("MUSETALK_VSDLM_TRACK_AGREE")
    if ta is not None:
        config.vsdlm_track_agree_check = ta.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    ta_opstd = os.environ.get("MUSETALK_VSDLM_TRACK_AGREE_OP_STD")
    if ta_opstd is not None:
        config.vsdlm_track_agree_op_std = max(0.0, float(ta_opstd))
    ta_opmean = os.environ.get("MUSETALK_VSDLM_TRACK_AGREE_OP_MEAN")
    if ta_opmean is not None:
        config.vsdlm_track_agree_op_mean = max(0.0, float(ta_opmean))
    ta_cov = os.environ.get("MUSETALK_VSDLM_TRACK_AGREE_MIN_COV")
    if ta_cov is not None:
        config.vsdlm_track_agree_min_cov = max(0.0, min(1.0, float(ta_cov)))
    ta_iou = os.environ.get("MUSETALK_VSDLM_TRACK_AGREE_MIN_IOU")
    if ta_iou is not None:
        config.vsdlm_track_agree_min_iou = max(0.0, float(ta_iou))

    use_vad_fusion = os.environ.get("MUSETALK_USE_VAD_FUSION")
    if use_vad_fusion is not None:
        config.use_vad_fusion = use_vad_fusion.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    vad_url = os.environ.get("MUSETALK_VAD_URL")
    if vad_url:
        config.vad_url = vad_url.strip()

    loose_speak = os.environ.get("MUSETALK_VSDLM_LOOSE_MIN_SPEAK_SEC")
    if loose_speak is not None:
        config.vsdlm_loose_min_speak_duration_sec = max(0.0, float(loose_speak))

    shot_min_speak = os.environ.get("MUSETALK_LIPSYNC_SHOT_MIN_SPEAK_SEC")
    if shot_min_speak is not None:
        config.lipsync_shot_min_speak_sec = max(0.0, float(shot_min_speak))

    short_shot = os.environ.get("MUSETALK_LIPSYNC_SHORT_SHOT_SEC")
    if short_shot is not None:
        config.lipsync_short_shot_sec = max(0.0, float(short_shot))

    short_shot_min = os.environ.get("MUSETALK_LIPSYNC_SHORT_SHOT_MIN_SPEAK_SEC")
    if short_shot_min is not None:
        config.lipsync_short_shot_min_speak_sec = max(0.0, float(short_shot_min))

    keep_partial = os.environ.get("MUSETALK_LIPSYNC_KEEP_PARTIAL_MIN_SEC")
    if keep_partial is not None:
        config.lipsync_keep_partial_min_sec = max(0.0, float(keep_partial))

    temporal_max = os.environ.get("MUSETALK_VSDLM_TEMPORAL_MAX_RADIUS")
    if temporal_max is not None:
        config.vsdlm_temporal_max_radius = max(0, int(temporal_max))

    assign_gap = os.environ.get("MUSETALK_VAD_ASSIGN_MAX_GAP_SEC")
    if assign_gap is not None:
        config.vad_assign_max_gap_sec = max(0.0, float(assign_gap))

    bbox_stride = os.environ.get("MUSETALK_BBOX_DETECT_STRIDE")
    if bbox_stride is not None:
        config.bbox_detect_stride = max(1, int(bbox_stride))

    face_stride = os.environ.get("MUSETALK_ASD_FACE_DETECT_STRIDE")
    if face_stride is not None:
        config.asd_face_detect_stride = max(1, int(face_stride))

    vsdlm_batch = os.environ.get("MUSETALK_VSDLM_BATCH_SIZE")
    if vsdlm_batch is not None:
        config.vsdlm_batch_size = max(1, int(vsdlm_batch))

    use_lr_asd = os.environ.get("MUSETALK_USE_LR_ASD")
    if use_lr_asd is not None:
        config.use_lr_asd = use_lr_asd.strip().lower() in {"1", "true", "yes", "on"}
        if config.use_lr_asd:
            config.use_vsdlm = False

    asd_threshold = os.environ.get("MUSETALK_ASD_THRESHOLD")
    if asd_threshold is not None:
        config.asd_threshold = float(asd_threshold)

    cf_stride = os.environ.get("MUSETALK_CODEFORMER_STRIDE")
    if cf_stride is not None:
        config.codeformer_stride = max(1, int(cf_stride))

    blend_ramp = os.environ.get("MUSETALK_BLEND_RAMP_FRAMES")
    if blend_ramp is not None:
        config.blend_ramp_frames = max(0, int(blend_ramp))

    color_match = os.environ.get("MUSETALK_COLOR_MATCH")
    if color_match is not None:
        config.use_color_match = color_match.strip().lower() in {"1", "true", "yes", "on"}

    min_face = os.environ.get("MUSETALK_MIN_FACE_AREA_RATIO")
    if min_face is not None:
        config.min_face_area_ratio = max(0.0, float(min_face))

    shot_aware = os.environ.get("MUSETALK_ASD_DILATE_RESPECT_SHOTS")
    if shot_aware is not None:
        config.asd_dilate_respect_shots = shot_aware.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    full_shots = os.environ.get("MUSETALK_LIPSYNC_FULL_SPEAKING_SHOTS")
    if full_shots is not None:
        config.lipsync_full_speaking_shots = full_shots.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    close_mouth = os.environ.get("MUSETALK_CLOSE_MOUTH_ON_SILENCE")
    if close_mouth is not None:
        config.lipsync_close_mouth_on_silence = close_mouth.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    shot_thr = os.environ.get("MUSETALK_SHOT_CUT_HIST_THRESHOLD")
    if shot_thr is not None:
        config.shot_cut_hist_threshold = float(shot_thr)

    # Yaw gate. Defaults (off=65/on=57) already sit where InsightFace's saturating
    # yaw actually separates a readable face from a profile; these exist so a
    # deployment can retune without a code change. ``apply_yaw_hysteresis_gate``
    # clamps on_deg down to off_deg when the pair is inverted.
    yaw_gate = os.environ.get("MUSETALK_LIPSYNC_YAW_GATE")
    if yaw_gate is not None:
        config.lipsync_yaw_gate = yaw_gate.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    yaw_off = os.environ.get("MUSETALK_LIPSYNC_YAW_OFF_DEG")
    if yaw_off is not None:
        config.lipsync_yaw_off_deg = max(0.0, float(yaw_off))
    yaw_on = os.environ.get("MUSETALK_LIPSYNC_YAW_ON_DEG")
    if yaw_on is not None:
        config.lipsync_yaw_on_deg = max(0.0, float(yaw_on))
    yaw_stride = os.environ.get("MUSETALK_LIPSYNC_YAW_STRIDE")
    if yaw_stride is not None:
        config.lipsync_yaw_stride = max(0, int(yaw_stride))

    return config


class LipSyncRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or relative path to the input video file")
    audio_path: str = Field(
        ...,
        description="Driving audio for lip-sync; also used as the audio track of the final output video",
    )
    output_path: str = Field(..., description="Absolute or relative path for the output video file")
    force_chunk: bool = Field(
        False,
        description="Force chunked processing even for short videos (useful for testing)",
    )
    chunk_duration_sec: Optional[float] = Field(
        None,
        description="Segment length in seconds for long videos (default: 60)",
    )
    VAD_result: Optional[str] = Field(
        None,
        description=(
            "Optional JSON string: array of {start,end} (seconds) speech segments. "
            "When set, skips remote VAD and uses these times (absolute to the full audio)."
        ),
    )


class LipSyncResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None
    frame_count: Optional[int] = None
    speaking_frames: Optional[int] = None
    lipsync_frames: Optional[int] = None
    chunked: Optional[bool] = None
    chunk_count: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine_pool
    config = _load_service_config()
    ensure_ffmpeg_env(config.ffmpeg_path)
    logger.info(
        "Starting MuseTalk HTTP service (max_concurrent=%d, gpu_ids=%s)...",
        config.max_concurrent_requests,
        config.gpu_ids or [config.gpu_id],
    )
    _engine_pool = EnginePool(config)
    logger.info("MuseTalk HTTP service ready")
    yield
    logger.info("Shutting down MuseTalk HTTP service")
    _engine_pool = None


app = FastAPI(title="MuseTalk Lip-Sync Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if _engine_pool is None:
        return {"status": "starting", "model_loaded": False}

    pool_status = _engine_pool.status()
    return {
        "status": "ok",
        "model_loaded": True,
        "max_concurrent": pool_status.max_concurrent,
        "total_engines": pool_status.total_engines,
        "available_engines": pool_status.available_engines,
        "busy_engines": pool_status.busy_engines,
    }


@app.post("/lipsync", response_model=LipSyncResponse)
async def lipsync(request: LipSyncRequest):
    logger.info(
        "Received lipsync request: video_path=%s audio_path=%s output_path=%s VAD_result=%s",
        request.video_path,
        request.audio_path,
        request.output_path,
        "yes" if request.VAD_result else "no",
    )

    if _engine_pool is None:
        logger.error("Inference engine pool is not initialized")
        raise HTTPException(status_code=503, detail="Inference engine is not ready")

    vad_segments = None
    if request.VAD_result is not None and str(request.VAD_result).strip():
        from musetalk.utils.vad_client import parse_vad_result

        try:
            vad_segments = parse_vad_result(request.VAD_result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with _engine_pool.borrow() as engine:
        try:
            result = await asyncio.to_thread(
                engine.run_lipsync,
                request.video_path,
                request.audio_path,
                request.output_path,
                force_chunk=request.force_chunk,
                chunk_duration_sec=request.chunk_duration_sec,
                vad_segments=vad_segments,
            )
            logger.info(
                "Lipsync request succeeded: output_path=%s frame_count=%s speaking_frames=%s lipsync_frames=%s chunked=%s chunk_count=%s",
                result["output_path"],
                result["frame_count"],
                result["speaking_frames"],
                result["lipsync_frames"],
                result.get("chunked"),
                result.get("chunk_count"),
            )
            return LipSyncResponse(
                success=True,
                message="Lip-sync completed successfully",
                output_path=result["output_path"],
                frame_count=result["frame_count"],
                speaking_frames=result["speaking_frames"],
                lipsync_frames=result["lipsync_frames"],
                chunked=result.get("chunked"),
                chunk_count=result.get("chunk_count"),
            )
        except Exception as exc:
            logger.error(
                "Lipsync request failed: video_path=%s audio_path=%s output_path=%s error=%s\n%s",
                request.video_path,
                request.audio_path,
                request.output_path,
                exc,
                traceback.format_exc(),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
