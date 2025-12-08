# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from agentstack_sdk.a2a.extensions import (
    TrajectoryExtensionServer,
)
from agentstack_sdk.a2a.types import AgentMessage, Metadata
from agentstack_sdk.server.context import RunContext
from granite_core.logging import get_logger

logger = get_logger(__name__)


class TrajectoryHandler:
    def __init__(self, trajectory: TrajectoryExtensionServer, context: RunContext) -> None:
        self.trajectory = trajectory
        self.context = context
        self.log: list[Metadata[str, Any]] = []

    async def yield_trajectory(
        self, title: str | None = None, content: str | None = None, group_id: str | None = None
    ) -> None:
        if title is None and content is None:
            return
        log_msg = f"{title}: {content}"
        formatted_log_msg = log_msg[:77] + "..." if len(log_msg) > 77 else log_msg
        logger.debug(formatted_log_msg)
        trajectory_metadata = self.trajectory.trajectory_metadata(title=title, content=content, group_id=group_id)
        await self.context.yield_async(trajectory_metadata)
        self.log.append(trajectory_metadata)

    async def store(self) -> None:
        for metadata in self.log:
            await self.context.store(AgentMessage(metadata=metadata))
