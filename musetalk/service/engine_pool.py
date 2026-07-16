"""Pool of MuseTalk engines for bounded concurrent inference."""

from __future__ import annotations

import asyncio
import copy
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Optional

from musetalk.service.engine import MuseTalkEngine, ServiceConfig

logger = logging.getLogger("musetalk_service")


@dataclass
class PoolStatus:
    max_concurrent: int
    total_engines: int
    available_engines: int
    busy_engines: int


class EnginePool:
    """One engine per concurrent slot; each slot is exclusive to a single request."""

    def __init__(self, base_config: ServiceConfig):
        self.max_concurrent = max(1, base_config.max_concurrent_requests)
        gpu_ids = base_config.gpu_ids or [base_config.gpu_id]

        self._engines: List[MuseTalkEngine] = []
        for slot in range(self.max_concurrent):
            cfg = copy.copy(base_config)
            cfg.gpu_id = gpu_ids[slot % len(gpu_ids)]
            logger.info(
                "Loading engine pool slot %d/%d on cuda:%d",
                slot + 1,
                self.max_concurrent,
                cfg.gpu_id,
            )
            self._engines.append(MuseTalkEngine(cfg))

        self._available: asyncio.Queue[MuseTalkEngine] = asyncio.Queue()
        for engine in self._engines:
            self._available.put_nowait(engine)

        logger.info(
            "Engine pool ready: %d engines, gpu_ids=%s",
            self.max_concurrent,
            [e.config.gpu_id for e in self._engines],
        )

    @property
    def size(self) -> int:
        return len(self._engines)

    def status(self) -> PoolStatus:
        available = self._available.qsize()
        return PoolStatus(
            max_concurrent=self.max_concurrent,
            total_engines=self.size,
            available_engines=available,
            busy_engines=self.size - available,
        )

    async def acquire(self) -> MuseTalkEngine:
        return await self._available.get()

    async def release(self, engine: MuseTalkEngine) -> None:
        await self._available.put(engine)

    @asynccontextmanager
    async def borrow(self):
        engine = await self.acquire()
        try:
            yield engine
        finally:
            await self.release(engine)
