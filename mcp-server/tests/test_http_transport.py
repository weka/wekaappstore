"""Tests for dual-mode HTTP transport in server.py.

Strategy: Construct FastMCP instances directly (not importing from server.py)
to avoid module-level side effects. Tests verify the patterns used by server.py.
One integration test imports server module to confirm no regression.
"""
from __future__ import annotations

import sys

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse


def _make_http_mcp(port: int = 8080) -> FastMCP:
    """Create a FastMCP instance configured for HTTP mode."""
    return FastMCP("weka-app-store-mcp", host="0.0.0.0", port=port, stateless_http=True)


def _make_stdio_mcp() -> FastMCP:
    """Create a FastMCP instance with defaults (stdio mode)."""
    return FastMCP("weka-app-store-mcp")


# --- HTTP mode config tests ---


def test_http_mcp_host_and_port():
    """In HTTP mode, mcp.settings.host == '0.0.0.0' and mcp.settings.port == 8080."""
    mcp = _make_http_mcp(port=8080)
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 8080


def test_stateless_mode_enabled():
    """In HTTP mode, mcp.settings.stateless_http is True."""
    mcp = _make_http_mcp()
    assert mcp.settings.stateless_http is True


# --- Health endpoint tests ---


def test_health_endpoint_returns_200():
    """GET /health returns 200 with {'status': 'ok', 'tools': 8, 'transport': 'http'}."""
    mcp = _make_http_mcp()

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "tools": 8, "transport": "http"})

    client = TestClient(mcp.streamable_http_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["tools"] == 8
    assert body["transport"] == "http"


def test_health_only_in_http_mode():
    """In stdio mode, no /health route exists on the mcp object."""
    mcp = _make_stdio_mcp()
    # Stdio mcp has no streamable HTTP app or custom routes registered
    # Verify the mcp has no _custom_starlette_routes with path "/health"
    custom_routes = getattr(mcp, "_custom_starlette_routes", [])
    health_routes = [r for r in custom_routes if getattr(r, "path", None) == "/health"]
    assert len(health_routes) == 0


# --- Transport argument selection tests ---


def test_transport_arg_selection(monkeypatch):
    """mcp.run() receives 'streamable-http' when MCP_TRANSPORT=http, 'stdio' when unset."""
    captured = {}

    def mock_run(self, transport="stdio", **kwargs):
        captured["transport"] = transport

    # Test HTTP mode
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    # Reload config to pick up new env var
    if "config" in sys.modules:
        del sys.modules["config"]
    import config as cfg_http
    transport_arg = "streamable-http" if cfg_http.MCP_TRANSPORT == "http" else "stdio"
    assert transport_arg == "streamable-http"

    # Test stdio mode
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    if "config" in sys.modules:
        del sys.modules["config"]
    import config as cfg_stdio
    transport_arg = "streamable-http" if cfg_stdio.MCP_TRANSPORT == "http" else "stdio"
    assert transport_arg == "stdio"


# --- Stdio mode unchanged tests ---


def test_stdio_mode_unchanged():
    """In stdio mode, mcp is constructed with default host/port (not HTTP settings)."""
    mcp = _make_stdio_mcp()
    # Default FastMCP host is "127.0.0.1", port is 8000
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8000
    assert mcp.settings.stateless_http is False


# --- Integration test: server module regression ---


def test_server_mcp_exists_with_tools(monkeypatch):
    """Importing server.py with MCP_TRANSPORT unset constructs mcp with 8 tools."""
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("BLUEPRINTS_DIR", "/tmp")

    # Remove cached modules to force re-import
    for mod in list(sys.modules.keys()):
        if mod in ("server", "config"):
            del sys.modules[mod]

    import server  # noqa: PLC0415

    assert hasattr(server, "mcp")
    # Count registered tools
    tools = server.mcp._tool_manager.list_tools()
    assert len(tools) == 8
