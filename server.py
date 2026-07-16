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
        config.max_concurrent_requests = max(1, int(max_concurrent))

    gpu_ids_env = os.environ.get("MUSETALK_GPU_IDS")
    if gpu_ids_env:
        config.gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]

    return config


class LipSyncRequest(BaseModel):
    video_path: str = Field(..., description="Absolute or relative path to the input video file")
    audio_path: str = Field(
        ...,
        description="Audio used to drive lip-sync only; output keeps the original video audio track",
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
