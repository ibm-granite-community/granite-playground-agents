from granite_chat.citations.types import Citation
from granite_chat.emitter import Event


class TextEvent(Event):
    text: str


class TrajectoryEvent(Event):
    step: str


class GeneratingCitationsEvent(Event):
    pass


class GeneratingCitationsCompleteEvent(Event):
    pass


class CitationEvent(Event):
    citation: Citation
