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


class WorkerPool:
    def __init__(self, max_workers: int | None = None, max_concurrent_tasks: int | None = None) -> None:
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks or 10)

    @asynccontextmanager
    async def throttle(self):  # noqa: ANN201
        async with self.semaphore:
            yield
