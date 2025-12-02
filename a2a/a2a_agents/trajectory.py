# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from agentstack_sdk.a2a.extensions import (
    TrajectoryExtensionServer,
)
from agentstack_sdk.a2a.types import AgentMessage, Metadata
from agentstack_sdk.server.context import RunContext


class TrajectoryHandler:
    def __init__(self, trajectory: TrajectoryExtensionServer, context: RunContext) -> None:
        self.trajectory = trajectory
        self.context = context
        self.log: list[Metadata[str, Any]] = []

    async def yield_trajectory(self, title: str | None = None, content: str | None = None) -> None:
        trajectory_metadata = self.trajectory.trajectory_metadata(title=title, content=content)
        await self.context.yield_async(trajectory_metadata)
        self.log.insert(0, trajectory_metadata)

    async def store(self) -> None:
        for metadata in self.log:
            await self.context.store(AgentMessage(metadata=metadata))
