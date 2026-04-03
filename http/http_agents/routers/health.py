# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from fastapi import APIRouter
from granite_core.logging import get_logger

from http_agents.models.responses import HealthResponse

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(status="healthy")
