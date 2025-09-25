import logging
import os
from typing import Any

from uvicorn.logging import DefaultFormatter

from granite_core.config import settings

# remove all handlers from the root logger
root_logger = logging.getLogger()
while root_logger.hasHandlers():
    root_logger.removeHandler(root_logger.handlers[0])

# create a handler that matches the ACP format but with additional info in brackets
handler = logging.StreamHandler()
handler.setFormatter(DefaultFormatter(fmt="%(levelprefix)s %(message)s (%(name)s:%(lineno)d)"))

# attach the handler and set the log level
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

os.environ["BEEAI_LOG_LEVEL"] = "INFO"


# Disable httpx INFO logging
# Get the logger for 'httpx'
httpx_logger = logging.getLogger("httpx")

# Set the logging level to WARNING to ignore INFO and DEBUG logs
httpx_logger.setLevel(logging.WARNING)


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


__all__ = ["get_logger"]
