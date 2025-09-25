import threading
from typing import Literal, Optional, get_args

from transformers import AutoTokenizer

from granite_core import get_logger
from granite_core.config import settings

ModelType = Literal["retrieval", "similarity"]


class EmbeddingsTokenizer:
    _instance_lock = threading.Lock()
    _instance: Optional["EmbeddingsTokenizer"] = None

    def __init__(self) -> None:
        self.tokenizers: dict[ModelType, AutoTokenizer] = {}
        self.logger = get_logger(__name__)

    @classmethod
    def get_instance(cls) -> "EmbeddingsTokenizer":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._preload_tokenizers()
        return cls._instance

    def _preload_tokenizers(self) -> None:
        tokenizer_types: tuple[ModelType] = get_args(ModelType)

        for t in tokenizer_types:
            tokenizer_name = None

            if t == "similarity":
                tokenizer_name = settings.EMBEDDINGS_SIM_HF_TOKENIZER
            elif t == "retrieval":
                tokenizer_name = settings.EMBEDDINGS_HF_TOKENIZER

            if tokenizer_name:
                self.logger.info(f"Preloading tokenizer: {tokenizer_name}")
                self.tokenizers[t] = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_tokenizer(self, type: ModelType = "retrieval") -> AutoTokenizer | None:
        # Returns tokenizer or None if not set / not used
        return self.tokenizers.get(type, None)
