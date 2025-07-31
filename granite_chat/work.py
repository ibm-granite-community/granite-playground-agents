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


class WorkerPool:
    def __init__(
        self,
        max_workers: int = 8,
        max_concurrent_tasks: int = 8,
        rate_limit: int = 8,
        rate_period: float = 2,
    ) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.rate_limiter = AsyncLimiter(rate_limit, rate_period)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # self._active = 0
        # self._lock = asyncio.Lock()  # Protect counter

    @asynccontextmanager
    async def throttle(self):  # noqa: ANN201
        async with self.semaphore, self.rate_limiter:
            # async with self._lock:
            #     self._active += 1
            # try:
            yield
            # finally:
            #     async with self._lock:
            #         self._active -= 1


# Control access to the chat backend
chat_pool = WorkerPool(max_concurrent_tasks=20)

# Control access to the embeddings backend
embeddings_pool = chat_pool

# General task control
task_pool = WorkerPool(max_concurrent_tasks=20, rate_limit=10, rate_period=2)
