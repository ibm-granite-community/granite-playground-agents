# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

from granite_core.config import settings


class UserAgent:
    """
    Singleton class to provide user agent string
    """

    instance = None
    user_agent = "GranitePlayground/1.0 (https://www.ibm.com/granite/playground)"

    def __new__(cls, *args: tuple, **kwargs: dict[str, Any]) -> Self:
        if not cls.instance:
            cls.instance = super().__new__(cls, *args, **kwargs)
            if settings.USER_AGENT_CONTACT:
                cls.instance.user_agent = (
                    f"GranitePlayground/1.0 (https://www.ibm.com/granite/playground; {settings.USER_AGENT_CONTACT})"
                )
        return cls.instance
