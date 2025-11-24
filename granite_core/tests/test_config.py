# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
from collections.abc import Generator
from typing import Any
from unittest import mock

import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from granite_core.config import Settings


@pytest.fixture
def clean_env() -> Generator[None, Any, None]:
    """
    Ensure a clean environment for each test. Mock os.environ so that tests don't influence each other or the actual
    system environment.
    """

    with mock.patch.dict(os.environ, {}, clear=True):
        yield


def test_defaults(clean_env) -> None:  # noqa: ANN001
    """Test that the settings load with default values correctly."""

    settings = Settings()  # type: ignore[call-arg]

    assert settings.STREAMING is True  # boolean
    assert settings.LLM_PROVIDER == "ollama"  # literal
    assert settings.LLM_MODEL == "ibm/granite4"  # string
    assert settings.TEMPERATURE == 0.0  # float
    assert str(settings.OLLAMA_BASE_URL) == "http://localhost:11434/"  # HttpUrl


def test_granite_model_validation_success(clean_env) -> None:  # noqa: ANN001
    """Test that a model name containing 'granite' passes validation."""

    settings = Settings(LLM_MODEL="ibm/granite-13b-chat")  # type: ignore[call-arg]
    assert settings.LLM_MODEL == "ibm/granite-13b-chat"


def test_granite_model_validation_failure(clean_env) -> None:  # noqa: ANN001
    """Test that a model name NOT containing 'granite' raises ValueError."""

    with pytest.raises(ValidationError) as excinfo:
        Settings(LLM_MODEL="gpt-4")  # type: ignore[call-arg]

    errors = excinfo.value.errors()
    assert any("LLM_MODEL must be set to an IBM Granite model ID" in e["msg"] for e in errors)


def test_retriever_google_success(clean_env) -> None:  # noqa: ANN001
    """Test valid configuration for Google retriever."""

    settings = Settings(RETRIEVER="google", GOOGLE_API_KEY=SecretStr("test_key"), GOOGLE_CX_KEY=SecretStr("test_cx"))  # type: ignore[call-arg]
    assert settings.RETRIEVER == "google"
    assert settings.GOOGLE_API_KEY.get_secret_value() == "test_key"  # type: ignore[union-attr]


def test_retriever_google_missing_keys(clean_env) -> None:  # noqa: ANN001
    """Test failure when Google retriever is selected but keys are missing."""

    with pytest.raises(ValidationError) as excinfo:
        Settings(RETRIEVER="google")  # type: ignore[call-arg]

    assert "Google retriever requires GOOGLE_API_KEY and GOOGLE_CX_KEY" in str(excinfo.value)


def test_retriever_tavily_success(clean_env) -> None:  # noqa: ANN001
    """Test valid configuration for Tavily retriever."""

    settings = Settings(RETRIEVER="tavily", TAVILY_API_KEY=SecretStr("test_tavily_key"))  # type: ignore[call-arg]
    assert settings.RETRIEVER == "tavily"


def test_retriever_tavily_missing_key(clean_env) -> None:  # noqa: ANN001
    """Test failure when Tavily retriever is selected but key is missing."""

    with pytest.raises(ValidationError) as excinfo:
        Settings(RETRIEVER="tavily")  # type: ignore[call-arg]

    assert "Tavily retriever requires TAVILY_API_KEY" in str(excinfo.value)


def test_temperature_constraints(clean_env) -> None:  # noqa: ANN001
    """Test validation boundaries for TEMPERATURE."""

    # Valid
    assert Settings(TEMPERATURE=0.0).TEMPERATURE == 0.0  # type: ignore[call-arg]
    assert Settings(TEMPERATURE=2.0).TEMPERATURE == 2.0  # type: ignore[call-arg]

    # Too low
    with pytest.raises(ValidationError):
        Settings(TEMPERATURE=-0.1)  # type: ignore[call-arg]

    # Too high
    with pytest.raises(ValidationError):
        Settings(TEMPERATURE=2.1)  # type: ignore[call-arg]


def test_env_var_side_effects(clean_env) -> None:  # noqa: ANN001
    """
    Test that the set_secondary_env validator correctly updates os.environ.
    The code explicitly sets OLLAMA_BASE_URL and RETRIEVER in os.environ.
    """

    # Ensure environment is empty initially (handled by fixture, but double check)
    assert "RETRIEVER" not in os.environ

    Settings(RETRIEVER="duckduckgo")  # type: ignore[call-arg]

    # Check side effect
    assert os.environ["RETRIEVER"] == "duckduckgo"


def test_ollama_base_url_env_setup(clean_env) -> None:  # noqa: ANN001
    """Test OLLAMA_BASE_URL injection into os.environ."""

    custom_url = HttpUrl("http://custom-ollama:11434")
    Settings(EMBEDDINGS_PROVIDER="ollama", OLLAMA_BASE_URL=custom_url)  # type: ignore[call-arg]

    # The validator logic: if "OLLAMA_BASE_URL" not in os.environ and EMBEDDINGS_PROVIDER == "ollama"
    assert os.environ["OLLAMA_BASE_URL"] == f"{custom_url}"


def test_api_headers_env_setup(clean_env) -> None:  # noqa: ANN001
    """Test that LLM_API_HEADERS is pushed to OPENAI_API_HEADERS env var."""

    headers = SecretStr('{"Authorization": "Bearer token"}')
    Settings(LLM_API_HEADERS=headers)  # type: ignore[call-arg]

    assert os.environ["OPENAI_API_HEADERS"] == headers.get_secret_value()


def test_url_validation(clean_env) -> None:  # noqa: ANN001
    """Test that invalid URLs raise errors."""

    with pytest.raises(ValidationError):
        custom_url = HttpUrl("not-a-url")
        Settings(OLLAMA_BASE_URL=custom_url)  # type: ignore[call-arg]


def test_loading_from_environment_variables(clean_env) -> None:  # noqa: ANN001
    """Test that Pydantic loads values from environment variables correctly."""

    with mock.patch.dict(os.environ, {"LLM_MODEL": "ibm/granite-custom", "TEMPERATURE": "1.5", "STREAMING": "false"}):
        settings = Settings()  # type: ignore[call-arg]
        assert settings.LLM_MODEL == "ibm/granite-custom"
        assert settings.TEMPERATURE == 1.5
        assert settings.STREAMING is False
