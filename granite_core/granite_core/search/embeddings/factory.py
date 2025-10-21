# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from granite_core import utils
from granite_core.config import settings
from granite_core.search.embeddings.model import EmbeddingsModel
from granite_core.search.embeddings.types import EmbeddingsModelType
from granite_core.search.embeddings.watsonx import WatsonxEmbeddings
from granite_core.work import embeddings_pool


class EmbeddingsFactory:
    """Factory for Embeddings instances."""

    @staticmethod
    def create(model_type: EmbeddingsModelType = "retrieval") -> EmbeddingsModel:
        provider = settings.EMBEDDINGS_PROVIDER
        model_name = settings.EMBEDDINGS_MODEL

        if model_type == "similarity" and settings.EMBEDDINGS_SIM_MODEL:
            model_name = settings.EMBEDDINGS_SIM_MODEL

        if provider == "watsonx":
            return EmbeddingsModel(
                embeddings=WatsonxEmbeddings(
                    model_id=model_name,
                    worker_pool=embeddings_pool,
                    truncate_input_tokens=(
                        settings.EMBEDDINGS_SIM_MAX_SEQUENCE
                        if model_type == "similarity"
                        else settings.EMBEDDINGS_MAX_SEQUENCE
                    ),
                ),
                type=model_type,
            )

        elif provider == "openai":
            # Optional extra headers for openai api
            embeddings_openai_api_headers = utils.get_secret_value(settings.EMBEDDINGS_OPENAI_API_HEADERS)
            extra_headers = (
                dict(pair.split("=", 1) for pair in embeddings_openai_api_headers.strip('"').split(","))
                if embeddings_openai_api_headers
                else None
            )

            return EmbeddingsModel(
                embeddings=OpenAIEmbeddings(
                    model=model_name,
                    api_key=settings.EMBEDDINGS_OPENAI_API_KEY,
                    base_url=str(settings.EMBEDDINGS_OPENAI_API_BASE),
                    check_embedding_ctx_length=False,
                    default_headers=extra_headers,
                ),
                type=model_type,
            )

        elif provider == "ollama":
            return EmbeddingsModel(
                embeddings=OllamaEmbeddings(model=model_name, base_url=str(settings.OLLAMA_BASE_URL)),
                type=model_type,
            )

        else:
            raise Exception(f"Unsupported embeddings provider {provider}")
