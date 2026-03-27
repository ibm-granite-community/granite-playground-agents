# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
import contextlib
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage
from granite_core.logging import get_logger

from http_agents.config import settings
from http_agents.models.responses import Message

logger = get_logger(__name__)


class SessionManager:
    """Manages conversation sessions and message history."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._session_timestamps: dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the session manager and cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
            logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            logger.info("Session manager stopped")

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to session history."""
        async with self._lock:
            message = {"role": role, "content": content, "timestamp": datetime.now(UTC).isoformat()}
            self._sessions[session_id].append(message)
            self._session_timestamps[session_id] = datetime.now(UTC)

            # Trim history if it exceeds max messages
            if len(self._sessions[session_id]) > settings.MAX_HISTORY_MESSAGES:
                self._sessions[session_id] = self._sessions[session_id][-settings.MAX_HISTORY_MESSAGES :]

            logger.debug(f"Added {role} message to session {session_id}")

    async def get_history(self, session_id: str) -> list[Message]:
        """Get conversation history for a session."""
        async with self._lock:
            messages = self._sessions.get(session_id, [])
            return [Message(**msg) for msg in messages]

    async def get_framework_messages(self, session_id: str) -> list[AnyMessage]:
        """Get conversation history as framework messages."""
        async with self._lock:
            messages = self._sessions.get(session_id, [])
            framework_messages: list[AnyMessage] = []

            for msg in messages:
                if msg["role"] == "user":
                    framework_messages.append(UserMessage(content=msg["content"]))
                else:
                    framework_messages.append(AssistantMessage(content=msg["content"]))

            return framework_messages

    async def clear_session(self, session_id: str) -> bool:
        """Clear a session's history."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._session_timestamps.pop(session_id, None)
                logger.info(f"Cleared session {session_id}")
                return True
            return False

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        async with self._lock:
            return session_id in self._sessions

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get raw messages for a session."""
        async with self._lock:
            return self._sessions.get(session_id, [])

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its history."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._session_timestamps.pop(session_id, None)
                logger.info(f"Deleted session {session_id}")

    async def _cleanup_expired_sessions(self) -> None:
        """Periodically cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                async with self._lock:
                    now = datetime.now(UTC)
                    ttl = timedelta(hours=settings.SESSION_TTL_HOURS)
                    expired_sessions = [
                        session_id
                        for session_id, timestamp in self._session_timestamps.items()
                        if now - timestamp > ttl
                    ]

                    for session_id in expired_sessions:
                        del self._sessions[session_id]
                        del self._session_timestamps[session_id]
                        logger.info(f"Cleaned up expired session {session_id}")

                    if expired_sessions:
                        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in session cleanup: {e}")


# Global session manager instance
session_manager = SessionManager()
