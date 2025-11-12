# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest
from beeai_framework.backend import UserMessage

from granite_core.chat_model import ChatModelFactory


@pytest.mark.asyncio
async def test_basic_chat() -> None:
    """Test basic chat infrastructure"""

    chat_model = ChatModelFactory.create()
    output = await chat_model.create(messages=[UserMessage("hello")])
    response = output.get_text_content()
    assert response is not None and len(response) > 0
