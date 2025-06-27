import logging

from uvicorn.logging import DefaultFormatter

from granite_chat.config import settings

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


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.log_level)
    return logger


__all__ = ["get_logger"]
