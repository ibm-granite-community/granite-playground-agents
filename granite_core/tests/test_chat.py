# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import pytest
from beeai_framework.backend import UserMessage
from pydantic import BaseModel, Field

from granite_core.chat_model import ChatModelFactory


@pytest.mark.asyncio
async def test_basic_chat() -> None:
    """Test basic chat infrastructure"""

    chat_model = ChatModelFactory.create()
    output = await chat_model.run([UserMessage("hello")])
    response = output.get_text_content()
    assert response is not None and len(response) > 0


@pytest.mark.asyncio
async def test_structured() -> None:
    """Test structured chat infrastructure"""

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)

    chat_model = ChatModelFactory.create(model_type="structured")
    response = await chat_model.run(
        [UserMessage("Who was the first known person to calculate the Earth's circumference.")],
        response_format=ProfileSchema,
    )
    assert isinstance(response.output_structured, ProfileSchema)
    assert response.output_structured.first_name == "Eratosthenes"
