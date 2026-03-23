"""MCP server entry point for the WEKA App Store.

IMPORTANT: logging.basicConfig MUST be called before FastMCP import/init
to prevent the SDK from hijacking handlers (GitHub issue python-sdk#1656).
All log output goes to stderr; stdout carries only MCP protocol frames.
"""
from __future__ import annotations

import logging
import sys

import config  # noqa: E402 — read env vars before basicConfig

# Configure logging to stderr BEFORE any FastMCP import
logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from mcp.server.fastmcp import FastMCP  # noqa: E402 — must come after basicConfig
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

from tools.apply_tool import register_apply  # noqa: E402
from tools.blueprints import register_blueprint_tools  # noqa: E402
from tools.crd_schema import register_crd_schema  # noqa: E402
from tools.inspect_cluster import register_inspect_cluster  # noqa: E402
from tools.inspect_weka import register_inspect_weka  # noqa: E402
from tools.status_tool import register_status  # noqa: E402
from tools.validate_yaml import register_validate_yaml  # noqa: E402

_transport = config.MCP_TRANSPORT
_port = config.MCP_PORT

if _transport == "http":
    mcp = FastMCP("weka-app-store-mcp", host="0.0.0.0", port=_port, stateless_http=True)

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:  # noqa: RUF029
        return JSONResponse({"status": "ok", "tools": 8, "transport": "http"})
else:
    mcp = FastMCP("weka-app-store-mcp")

register_inspect_cluster(mcp)
register_inspect_weka(mcp)
register_blueprint_tools(mcp)
register_crd_schema(mcp)
register_validate_yaml(mcp)
register_apply(mcp)
register_status(mcp)

if __name__ == "__main__":
    from config import validate_required
    validate_required()
    _transport_arg = "streamable-http" if _transport == "http" else "stdio"
    mcp.run(transport=_transport_arg)
