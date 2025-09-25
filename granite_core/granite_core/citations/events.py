from granite_core.citations.types import Citation
from granite_core.emitter import Event


class CitationEvent(Event):
    citation: Citation
