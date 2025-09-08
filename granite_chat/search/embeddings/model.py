from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer

from granite_chat.config import settings
from granite_chat.search.embeddings.tokenizer import EmbeddingsTokenizer
from granite_chat.search.embeddings.types import EmbeddingsModelType


class EmbeddingsModel:
    def __init__(self, embeddings: Embeddings, type: EmbeddingsModelType) -> None:
        self._embeddings = embeddings
        self._type = type

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def max_sequence_length(self) -> int:
        if self._type == "similarity":
            return settings.EMBEDDINGS_SIM_MAX_SEQUENCE
        else:
            return settings.EMBEDDINGS_MAX_SEQUENCE

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        return EmbeddingsTokenizer.get_instance().get_tokenizer(type=self._type)
