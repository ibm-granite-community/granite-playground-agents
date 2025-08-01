from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Phase(str, Enum):
    active = "active"
    completed = "completed"


class BaseAgentStatus(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str
    phase: Phase


class SearchingWebStatus(BaseAgentStatus):
    status: str = "searching-web"


class GeneratingCitationsStatus(BaseAgentStatus):
    status: str = "generating-citations"
