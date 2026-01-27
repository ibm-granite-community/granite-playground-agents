# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Sequence

import pytest
from mcp.types import ContentBlock, TextContent, Tool

from granite_core_mcp.base import MCPService
from granite_core_mcp.internet_search import InternetSearchService


@pytest.fixture
async def mcp_service() -> MCPService:
    """Shared fixture for InternetSearchService instance."""
    service: InternetSearchService = InternetSearchService(
        transport="stdio", max_search_results=3
    )
    return service


@pytest.mark.asyncio
async def test_service_with_stdio_transport(mcp_service: MCPService) -> None:
    """Test creating the service with stdio transport."""
    assert mcp_service.name == "internet_search"
    # Verify tools are registered
    tools: list[Tool] = await mcp_service.mcp.list_tools()
    assert len(tools) > 0
    assert any(t.name == "internet_search" for t in tools)


@pytest.mark.asyncio
async def test_internet_search_tool_functionality(mcp_service: MCPService) -> None:
    """Test calling the internet_search tool directly."""

    # Get the tool instance
    tools: list[Tool] = await mcp_service.mcp.list_tools()
    internet_search_tool: Tool = next(t for t in tools if t.name == "internet_search")

    # Verify tool exists
    assert internet_search_tool is not None
    assert internet_search_tool.name == "internet_search"

    # Call the tool
    result: Sequence[ContentBlock] | dict[str, Any] = await mcp_service.mcp.call_tool(
        name="internet_search", arguments={"query": "Python"}
    )

    # Verify results exist
    assert result is not None
    assert len(result) > 0

    # Get the text content from the result
    # result is a list, result[0] is also a list containing TextContent
    content_list = result[0]  # type: ignore[index]
    assert isinstance(content_list, list), f"Expected list, got {type(content_list)}"
    assert len(content_list) > 0

    content_item = content_list[0]
    assert isinstance(content_item, TextContent), (
        f"Expected TextContent, got {type(content_item)}"
    )
    result_text: str = content_item.text

    # Parse the JSON result
    parsed_results = json.loads(result_text)

    # Verify it's an array
    assert isinstance(parsed_results, list)
    assert len(parsed_results) > 0

    # Verify first entry has required fields
    first_result: Any = parsed_results[0]
    assert "title" in first_result
    assert "url" in first_result
    assert "content" in first_result

    # Verify fields are not empty
    assert first_result["title"]
    assert first_result["url"]
    assert first_result["content"]
