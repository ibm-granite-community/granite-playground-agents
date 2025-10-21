# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import os

from uvicorn.logging import DefaultFormatter

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

# Disable httpx INFO logging by setting the log level to WARNING
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
