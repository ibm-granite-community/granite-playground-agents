# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:
# - Allow max_workers to default via None
# - Configure semaphore via max_concurrent_tasks
# - Rate limiter

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from aiolimiter import AsyncLimiter

from granite_core.logging import get_logger, settings

logger = get_logger(__name__)


class WorkerPool:
    def __init__(
        self,
        name: str,
        max_workers: int = 8,
        max_concurrent_tasks: int = 8,
        rate_limit: int = 8,
        rate_period: float = 2,
    ) -> None:
        self.name = name
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.rate_limiter = AsyncLimiter(rate_limit, rate_period)
        self._semaphore_acquired_count = 0
        self._rate_limiter_acquired_count = 0
        self._counter_lock = asyncio.Lock()

    @asynccontextmanager
    async def throttle(self):  # noqa: ANN201
        logger.debug(
            f"[{self.name}] Before acquire - semaphore={self._semaphore_acquired_count}, "
            f"rate_limiter={self._rate_limiter_acquired_count}"
        )

        async with self.semaphore:
            async with self._counter_lock:
                self._semaphore_acquired_count += 1

            async with self.rate_limiter:
                async with self._counter_lock:
                    self._rate_limiter_acquired_count += 1

                logger.debug(
                    f"[{self.name}] After acquire - semaphore={self._semaphore_acquired_count}, "
                    f"rate_limiter={self._rate_limiter_acquired_count}"
                )

                try:
                    yield

                finally:
                    async with self._counter_lock:
                        self._rate_limiter_acquired_count -= 1
                        self._semaphore_acquired_count -= 1

                    logger.debug(
                        f"[{self.name}] After release - semaphore={self._semaphore_acquired_count}, "
                        f"rate_limiter={self._rate_limiter_acquired_count}"
                    )


# Control access to the chat backend
chat_pool = WorkerPool(
    name="inference",
    max_concurrent_tasks=settings.MAX_CONCURRENT_INFERENCE_TASKS,
    rate_limit=settings.RATE_LIMIT_INFERENCE_TASKS,
    rate_period=settings.RATE_PERIOD_INFERENCE_TASKS,
)

# Control access to the embeddings backend, share with chat for now
embeddings_pool = chat_pool

# General task control
task_pool = WorkerPool(
    name="general",
    max_concurrent_tasks=settings.MAX_CONCURRENT_TASKS,
    rate_limit=settings.RATE_LIMIT_TASKS,
    rate_period=settings.RATE_PERIOD_TASKS,
)
