"""Tests for the MCP server scaffold and tool registration."""
from __future__ import annotations

import asyncio
import importlib
import sys


def test_server_instantiates():
    """FastMCP("weka-app-store-mcp") creates without error and mcp attribute is a FastMCP instance."""
    from mcp.server.fastmcp import FastMCP
    import server  # noqa: F401

    assert hasattr(server, "mcp"), "server module must expose an 'mcp' attribute"
    assert isinstance(server.mcp, FastMCP), "server.mcp must be a FastMCP instance"


def test_mcp_server_name():
    """The MCP server is named 'weka-app-store-mcp'."""
    import server

    assert server.mcp.name == "weka-app-store-mcp"


def test_server_lists_5_tools():
    """MCP server has exactly 5 tools registered with correct names."""
    import server

    tools = asyncio.run(server.mcp.list_tools())
    tool_names = {t.name for t in tools}

    expected = {
        "inspect_cluster",
        "inspect_weka",
        "list_blueprints",
        "get_blueprint",
        "get_crd_schema",
    }

    assert len(tools) == 5, (
        f"Expected 5 tools, got {len(tools)}: {sorted(tool_names)}"
    )
    assert tool_names == expected, (
        f"Tool names mismatch.\nExpected: {sorted(expected)}\nGot: {sorted(tool_names)}"
    )


def test_all_tool_descriptions_have_sequencing():
    """Every tool docstring contains at least one sequencing keyword.

    Per RESEARCH.md Pitfall 5: agents need to know in what order to call tools.
    Each tool description must contain at least one of: before, after, first, sequencing.
    """
    import server

    tools = asyncio.run(server.mcp.list_tools())
    sequencing_keywords = {"before", "after", "first", "sequencing"}
    failures = []

    for tool in tools:
        description = tool.description or ""
        description_lower = description.lower()
        has_sequencing = any(kw in description_lower for kw in sequencing_keywords)
        if not has_sequencing:
            failures.append(
                f"Tool '{tool.name}' description missing sequencing guidance. "
                f"Add one of: {sequencing_keywords}. "
                f"Description: {description[:120]!r}"
            )

    assert not failures, "\n".join(failures)
