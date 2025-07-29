from granite_chat.citations.types import Citation
from granite_chat.emitter import Event


class TextEvent(Event):
    text: str


class TrajectoryEvent(Event):
    step: str


class CitationEvent(Event):
    citation: Citation
