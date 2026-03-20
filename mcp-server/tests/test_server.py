"""Tests for the MCP server scaffold and tool registration."""
from __future__ import annotations

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
