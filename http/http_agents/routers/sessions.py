# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from fastapi import APIRouter, HTTPException
from granite_core.logging import get_logger

from http_agents.models.responses import HistoryResponse
from http_agents.services.session_manager import session_manager

logger = get_logger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get(
    "/{session_id}/history",
    response_model=HistoryResponse,
    summary="Get conversation history",
    description="Retrieve the conversation history for a specific session.",
)
async def get_history(session_id: str) -> HistoryResponse:
    """
    Get conversation history for a session.
    """
    try:
        messages = await session_manager.get_history(session_id)

        if not messages:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return HistoryResponse(session_id=session_id, messages=messages)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving history for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(
    "/{session_id}",
    summary="Delete session",
    description="Delete a session and its conversation history.",
)
async def delete_session(session_id: str) -> dict[str, str]:
    """
    Delete a session and its history.
    """
    try:
        await session_manager.delete_session(session_id)
        return {"message": f"Session {session_id} deleted successfully"}

    except Exception as e:
        logger.exception(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
