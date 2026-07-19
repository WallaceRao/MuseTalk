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

    shot_thr = os.environ.get("MUSETALK_SHOT_CUT_HIST_THRESHOLD")
    if shot_thr is not None:
        config.shot_cut_hist_threshold = float(shot_thr)

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
        "Received lipsync request: video_path=%s audio_path=%s output_path=%s",
        request.video_path,
        request.audio_path,
        request.output_path,
    )

    if _engine_pool is None:
        logger.error("Inference engine pool is not initialized")
        raise HTTPException(status_code=503, detail="Inference engine is not ready")

    async with _engine_pool.borrow() as engine:
        try:
            result = await asyncio.to_thread(
                engine.run_lipsync,
                request.video_path,
                request.audio_path,
                request.output_path,
                force_chunk=request.force_chunk,
                chunk_duration_sec=request.chunk_duration_sec,
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
