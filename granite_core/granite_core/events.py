# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from granite_core.emitter import Event


class TextEvent(Event):
    text: str


class ThinkEvent(Event):
    text: str


class PassThroughEvent(Event):
    event: Any


class TrajectoryEvent(Event):
    title: str
    content: str | list[str] | None = None

    def to_markdown(self) -> str:
        # No content
        if not self.content:
            return f"**{self.title}**"

        # Content is a list
        if isinstance(self.content, list):
            bullets = "\n".join(f"- {item}" for item in self.content)
            return f"**{self.title}**  \n{bullets}"

        # Content is a string
        return f"**{self.title}**  \n{self.content}"


class GeneratingCitationsEvent(Event):
    pass


class GeneratingCitationsCompleteEvent(Event):
    pass
