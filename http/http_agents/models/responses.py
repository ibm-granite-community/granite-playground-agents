# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation metadata for sources."""

    url: str = Field(..., description="Source URL")
    title: str | None = Field(None, description="Source title")
    description: str | None = Field(None, description="Source description or context")
    start_index: int | None = Field(None, description="Start index in response text")
    end_index: int | None = Field(None, description="End index in response text")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")
    model_id: str = Field(..., description="Model identifier")


class Phase(BaseModel):
    """Phase information for tracking progress."""

    name: str = Field(..., description="Phase name (e.g., 'searching-web', 'generating-citations')")
    status: Literal["active", "completed"] = Field(..., description="Phase status")


class StreamEvent(BaseModel):
    """Server-Sent Event for streaming responses."""

    type: Literal["token", "phase", "citation", "usage", "heartbeat", "error", "done"] = Field(
        ..., description="Event type"
    )
    data: dict[str, Any] | str | None = Field(None, description="Event data")


class ChatResponse(BaseModel):
    """Response model for non-streaming chat."""

    response: str = Field(..., description="Generated response text")
    usage: UsageInfo | None = Field(None, description="Token usage information")
    session_id: str = Field(..., description="Session identifier")


class SearchResponse(BaseModel):
    """Response model for non-streaming search."""

    response: str = Field(..., description="Generated response text")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    phases: list[Phase] = Field(default_factory=list, description="Completed phases")
    usage: UsageInfo | None = Field(None, description="Token usage information")
    session_id: str = Field(..., description="Session identifier")


class ResearchResponse(BaseModel):
    """Response model for non-streaming research."""

    response: str = Field(..., description="Generated research response")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    trajectory: list[str] = Field(default_factory=list, description="Research trajectory/steps")
    phases: list[Phase] = Field(default_factory=list, description="Completed phases")
    usage: UsageInfo | None = Field(None, description="Token usage information")
    session_id: str = Field(..., description="Session identifier")


class Message(BaseModel):
    """Message in conversation history."""

    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: str | None = Field(None, description="Message timestamp")


class HistoryResponse(BaseModel):
    """Response model for session history."""

    session_id: str = Field(..., description="Session identifier")
    messages: list[Message] = Field(default_factory=list, description="Conversation history")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    version: str = Field(default="1.0.0", description="API version")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional health details")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    session_id: str | None = Field(None, description="Session identifier if applicable")
