# Granite Core MCP

Granite Core MCP provides Model Context Protocol (MCP) services built on top of the granite-core library. These services expose granite-core functionality as MCP tools that can be used by AI assistants and other MCP clients.

## Overview

This package provides a base framework for creating MCP services and includes an Internet Search service that enables AI assistants to search the web and scrape content from search results.

## Features

- **Base MCP Service Framework**: Abstract base class for building MCP services with common functionality
- **Internet Search Service**: Search the internet and retrieve scraped content from results
- **Multiple Transport Options**: Support for stdio, SSE, and streamable-http transports
- **Configurable**: Customizable search result limits and content length

## Installation

```bash
# Install with uv
uv --directory granite_core_mcp sync
```

## Services

### Internet Search Service

The Internet Search service provides a tool to search the internet.

#### Features

- Search the internet using various search engines (configured via granite-core)
- Automatically scrape content from search results
- Configurable maximum number of results
- Configurable content length limits
- Returns structured JSON with title, URL, and content

#### Usage

**Running as a standalone service:**

```bash
# Run with default settings (port 8001, streamable-http transport)
uv --directory granite_core -m granite_core_mcp.internet_search

# Run with custom settings
uv --directory granite_core -m granite_core_mcp.internet_search --transport stdio
```

**Command-line options:**

- `--port`: Port number for HTTP transport (default: 8001)
- `--transport`: Transport type - stdio, sse, or streamable-http (default: streamable-http)

**Using in Python code:**

```python
from granite_core_mcp.internet_search import InternetSearchService

# Create service instance
service = InternetSearchService(
    transport="streamable-http",
    port=8001,
    max_search_results=10,
    max_scraped=10,
    max_scraped_content_length=10000
)

# Run the service
service.run()
```

**Tool Interface:**

The service exposes an `internet_search` tool with the following interface:

```python
async def internet_search(query: str) -> str:
    """Search the internet for a query.
    
    Args:
        query: The search query string
    
    Returns:
        JSON string containing search results with title, url, and content
    """
```

**Example Response:**

```json
[
    {
        "title": "Python Programming Language",
        "url": "https://www.python.org",
        "content": "Python is a high-level, interpreted programming language..."
    },
    {
        "title": "Python Tutorial",
        "url": "https://docs.python.org/3/tutorial/",
        "content": "This tutorial introduces the reader informally to the basic concepts..."
    }
]
```

## Creating Custom MCP Services

You can create your own MCP services by extending the `MCPService` base class:

```python
from granite_core_mcp.base import MCPService, TransportType

class MyCustomService(MCPService):
    def __init__(self, port: int = 8000, transport: TransportType = "streamable-http"):
        super().__init__(name="my_service", port=port, transport=transport)
    
    def _register_tools(self) -> None:
        """Register tools for this service."""
        
        @self.mcp.tool(
            name="my_tool",
            description="Description of what my tool does"
        )
        async def my_tool(param: str) -> str:
            """Tool implementation."""
            # Your tool logic here
            return f"Result: {param}"

# Run the service
if __name__ == "__main__":
    service = MyCustomService()
    service.run()
```

### Running Tests

```bash
uv --directory granite_core_mcp run pytest  
```