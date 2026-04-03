# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User message content", min_length=1)
    stream: bool = Field(default=True, description="Enable streaming response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user-123-session-456",
                "message": "What is the capital of France?",
                "stream": True,
            }
        }
    }


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User message/query content", min_length=1)
    stream: bool = Field(default=True, description="Enable streaming response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user-123-session-456",
                "message": "What are the latest developments in quantum computing?",
                "stream": True,
            }
        }
    }


class ResearchRequest(BaseModel):
    """Request model for research endpoint."""

    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="Research topic or question", min_length=1)
    stream: bool = Field(default=True, description="Enable streaming response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user-123-session-456",
                "message": "Research the impact of artificial intelligence on healthcare",
                "stream": True,
            }
        }
    }
