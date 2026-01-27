# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""Base class for MCP services."""

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeAlias

from mcp.server.fastmcp import FastMCP

TransportType: TypeAlias = Literal["stdio", "sse", "streamable-http"]


class MCPService(ABC):
    """Base class for MCP services.

    Provides common functionality for MCP services including:
    - Server initialization
    - Transport configuration
    - Tool registration
    - Lifecycle management
    """

    def __init__(
        self,
        name: str,
        transport: TransportType = "streamable-http",
        port: int = 8000,
    ) -> None:
        """Initialize the MCP service.

        Args:
            name: Name of the MCP service
            transport: Transport type - "stdio", "sse", or "streamable-http" (default: "streamable-http")
            port: Port number for HTTP transport (default: 8000)
        """
        self.name = name
        self.port = port
        self.transport: TransportType = transport
        self.mcp = FastMCP(name=name, port=port)
        self._register_tools()

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools for this service.

        Subclasses must implement this method to register their tools
        using self.mcp.tool() decorator or self.register_tool() method.
        """
        pass

    def register_tool(
        self,
        func: Any,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a tool with the MCP server.

        Args:
            func: The function to register as a tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description for the tool

        Returns:
            The decorated function
        """
        return self.mcp.tool(name=name, description=description)(func)

    def run(self, transport: TransportType | None = None) -> None:
        """Run the MCP server.

        Args:
            transport: Optional transport override ("stdio", "sse", or "streamable-http")
        """
        transport_to_use: TransportType = transport or self.transport
        self.mcp.run(transport=transport_to_use)

    def get_mcp_instance(self) -> FastMCP:
        """Get the underlying FastMCP instance.

        Returns:
            The FastMCP instance
        """
        return self.mcp
