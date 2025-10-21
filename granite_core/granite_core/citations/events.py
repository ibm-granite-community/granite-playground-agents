# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from granite_core.citations.types import Citation
from granite_core.emitter import Event


class CitationEvent(Event):
    citation: Citation
