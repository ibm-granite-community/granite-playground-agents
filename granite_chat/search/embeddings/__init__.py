from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from granite_chat.config import settings
from granite_chat.search.embeddings.watsonx import WatsonxEmbeddings
from granite_chat.workers import WorkerPool


def get_embeddings(provider: str, model_name: str, worker_pool: WorkerPool) -> Embeddings:
    """Return langchain embeddings based on provider and model name"""

    if provider == "watsonx":
        return WatsonxEmbeddings(model_id=model_name, worker_pool=worker_pool)

    elif provider == "openai":
        # Optional extra headers for openai api
        extra_headers = (
            dict(pair.split("=", 1) for pair in settings.EMBEDDINGS_OPENAI_API_HEADERS.strip('"').split(","))
            if settings.EMBEDDINGS_OPENAI_API_HEADERS
            else None
        )

        return OpenAIEmbeddings(
            model=model_name,
            api_key=SecretStr(secret_value=settings.EMBEDDINGS_OPENAI_API_KEY or ""),
            base_url=str(settings.EMBEDDINGS_OPENAI_API_BASE),
            check_embedding_ctx_length=False,
            default_headers=extra_headers,
        )

    elif provider == "ollama":
        return OllamaEmbeddings(
            model=model_name,
            base_url=str(settings.OLLAMA_BASE_URL),
        )

    else:
        raise Exception(f"Unsupported embeddings provider {provider}")
