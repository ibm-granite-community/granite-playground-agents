# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import random
from typing import ClassVar, Self


class UserAgent:
    _user_agents: ClassVar[list[str]] = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"  # noqa: E501
    ]

    @classmethod
    def user_agent(cls: type[Self]) -> str:
        return random.choice(cls._user_agents)
