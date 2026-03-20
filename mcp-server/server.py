"""MCP server entry point for the WEKA App Store.

IMPORTANT: logging.basicConfig MUST be called before FastMCP import/init
to prevent the SDK from hijacking handlers (GitHub issue python-sdk#1656).
All log output goes to stderr; stdout carries only MCP protocol frames.
"""
from __future__ import annotations

import logging
import sys

# Configure logging to stderr BEFORE any FastMCP import
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from mcp.server.fastmcp import FastMCP  # noqa: E402 — must come after basicConfig

from tools.blueprints import register_blueprint_tools  # noqa: E402
from tools.crd_schema import register_crd_schema  # noqa: E402
from tools.inspect_cluster import register_inspect_cluster  # noqa: E402
from tools.inspect_weka import register_inspect_weka  # noqa: E402

mcp = FastMCP("weka-app-store-mcp")

register_inspect_cluster(mcp)
register_inspect_weka(mcp)
register_blueprint_tools(mcp)
register_crd_schema(mcp)

if __name__ == "__main__":
    mcp.run()
