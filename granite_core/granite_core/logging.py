# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

from granite_core.config import settings


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.log_level)
    return logger


def get_logger_with_prefix(logger_name: str, tool_name: str, session_id: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.log_level)
    return LogContextAdapter(logger, {"tool": tool_name, "session_id": session_id})


class LogContextAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: Any) -> tuple[str, Any]:
        return (f"[{self.extra['tool']}:{self.extra['session_id']}] {msg}", kwargs)  # type: ignore
