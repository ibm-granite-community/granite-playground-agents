from granite_chat.citations.types import Citation
from granite_chat.emitter import Event


class CitationEvent(Event):
    citation: Citation
