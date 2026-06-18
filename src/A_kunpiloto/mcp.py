"""Optional MCP (Model Context Protocol) server for A-kunpiloto.

Exposes the same tool registry as the REPL, but as an MCP server so
external MCP clients (Claude Desktop, opencode, etc.) can use A-module
tools.  Requires the ``mcp`` optional dependency.

Usage::

    A kunpiloto mcp          # start MCP server on stdio
    A kunpiloto mcp --tcp    # start MCP server on TCP (port 8765)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from A import info

from A_kunpiloto.tools.registry import ToolRegistry

logger = logging.getLogger("A-kunpiloto.mcp")

try:
    import mcp.server as mcp_server
    import mcp.types as mcp_types
    from mcp.server.stdio import stdio_server

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _convert_tool_schema(entry_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-compatible tool schema to MCP Tool format.

    MCP expects ``inputSchema`` instead of ``parameters``, and the
    schema is nested one level differently.

    Args:
        entry_schema: OpenAI-compatible tool schema.

    Returns:
        MCP-compatible Tool dict.
    """
    params = entry_schema.get("function", {}).get("parameters", {})
    return {
        "name": entry_schema.get("function", {}).get("name", ""),
        "description": entry_schema.get("function", {}).get("description", ""),
        "inputSchema": {
            "type": params.get("type", "object"),
            "properties": params.get("properties", {}),
        },
    }


def run_mcp_server(
    registry: ToolRegistry,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Start the MCP server.

    Args:
        registry: Tool registry with discovered modules.
        transport: Transport type ("stdio" or "sse").
        host: Host to bind (for TCP transport).
        port: Port to bind (for TCP transport).

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required for the MCP server. "
            "Install it with: uv pip install 'A-kunpiloto[mcp]'"
        )

    app = mcp_server.Server("A-kunpiloto")

    # Build schema list for tools/list
    schemas = registry.get_schemas()
    mcp_tools = [_convert_tool_schema(s) for s in schemas]

    @app.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in mcp_tools
        ]

    @app.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any] | None,
    ) -> list[mcp_types.TextContent]:
        """Handle a tools/call request."""
        if arguments is None:
            arguments = {}

        entry = registry.get_entry(name)
        if entry is None:
            raise ValueError(f"Unknown tool: {name}")

        from A_kunpiloto.tools.executor import execute_tool_call

        result = execute_tool_call(entry, arguments)

        # Assemble output text from stdout + stderr
        parts = []
        if result.get("output"):
            parts.append(result["output"])
        if result.get("error"):
            parts.append(f"[ERROR] {result['error']}")

        text = "\n".join(parts) if parts else "(empty result)"

        return [mcp_types.TextContent(type="text", text=text)]

    # Run
    import asyncio

    if transport == "stdio":
        info(
            "MCP server starting on stdio. "
            "Connect your MCP client to this process."
        )
        asyncio.run(app.run(stdio_server()))
    else:
        info(f"MCP server starting on {host}:{port}")
        # SSE transport not yet implemented in the server SDK
        # Fallback to stdio with a warning
        logger.warning("SSE transport not available, falling back to stdio.")
        asyncio.run(app.run(stdio_server()))
