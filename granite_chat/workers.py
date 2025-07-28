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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from aiolimiter import AsyncLimiter


class WorkerPool:
    def __init__(
        self,
        max_workers: int | None = None,
        max_concurrent_tasks: int | None = None,
        rate_limit: int = 8,
        rate_period: float = 2,
    ) -> None:
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks or 8)
        self.rate_limiter = AsyncLimiter(rate_limit, rate_period)
        self._active = 0
        self._lock = asyncio.Lock()  # Protect counter

    @asynccontextmanager
    async def throttle(self):  # noqa: ANN201
        async with self.rate_limiter, self.semaphore:
            async with self._lock:
                self._active += 1
                # print(f"[START] Active workers: {self._active}")
            try:
                yield
            finally:
                async with self._lock:
                    self._active -= 1
                    # print(f"[END]   Active workers: {self._active}")
