import threading
from typing import Optional

from transformers import AutoTokenizer

from granite_chat.config import settings  # type: ignore


class EmbeddingsTokenizer:
    _instance_lock = threading.Lock()
    _instance: Optional["EmbeddingsTokenizer"] = None

    def __init__(self) -> None:
        self.tokenizer = None

    @classmethod
    def get_instance(cls) -> "EmbeddingsTokenizer":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._preload_tokenizer()
        return cls._instance

    def _preload_tokenizer(self) -> None:
        tokenizer_name = settings.EMBEDDINGS_HF_TOKENIZER
        if tokenizer_name:
            # Lazy import here to avoid overhead if no tokenizer needed

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # No tokenizer used if env var unset
            self.tokenizer = None

    def get_tokenizer(self) -> AutoTokenizer | None:
        # Returns tokenizer or None if not set / not used
        return self.tokenizer
