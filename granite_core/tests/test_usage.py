# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend.types import ChatModelUsage

from granite_core.usage import UsageInfo, create_usage_info


def test_create_usage_info_with_usage() -> None:
    usage = ChatModelUsage(
        completion_tokens=100,
        prompt_tokens=200,
        total_tokens=300,
    )
    model_id = "ibm-granite/granite-4.0-h-small"

    result = create_usage_info(usage, model_id)

    assert isinstance(result, UsageInfo)
    assert result.completion_tokens == 100
    assert result.prompt_tokens == 200
    assert result.total_tokens == 300
    assert result.model_id == model_id
    assert result.type == "usage_info"


def test_create_usage_info_without_usage() -> None:
    model_id = "ibm-granite/granite-4.0-h-small"

    result = create_usage_info(None, model_id)

    assert isinstance(result, UsageInfo)
    assert result.completion_tokens is None
    assert result.prompt_tokens is None
    assert result.total_tokens is None
    assert result.model_id == model_id
    assert result.type == "usage_info"
