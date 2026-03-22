"""Generate openclaw.json registration config from server.py tool registrations.

Reads tool names and descriptions directly from the register_* functions in
tools/*.py using the _RegistryCapture stub pattern established in the mock
harness. This ensures openclaw.json stays in sync with actual tool docstrings.

Usage:
    cd mcp-server && PYTHONPATH=.:../app-store-gui python generate_openclaw_config.py

Output:
    mcp-server/openclaw.json (overwritten in place)

_comment in output:
    Best-effort format -- may need revision when NemoClaw alpha schema is published.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# _RegistryCapture — minimal MCP stub (matches harness pattern)
# ---------------------------------------------------------------------------


class _RegistryCapture:
    """Minimal MCP stub that captures @mcp.tool() registrations.

    When register_*(mcp) calls @mcp.tool(), this stub captures the decorated
    function's name and docstring into self.captured, in registration order.
    """

    def __init__(self) -> None:
        self.captured: dict[str, str] = {}

    def tool(self):
        """Return a decorator that records the function's name and docstring."""
        def decorator(fn):
            self.captured[fn.__name__] = (fn.__doc__ or "").strip()
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Tool registration order (matches server.py exactly)
# ---------------------------------------------------------------------------

TOOL_REGISTRATION_ORDER = [
    "inspect_cluster",
    "inspect_weka",
    "list_blueprints",
    "get_blueprint",
    "get_crd_schema",
    "validate_yaml",
    "apply",
    "status",
]


def collect_tool_descriptions() -> dict[str, str]:
    """Call each register_* function with a _RegistryCapture stub.

    Returns a dict of {tool_name: description} in registration order.
    Descriptions are taken verbatim from tool docstrings in tools/*.py.
    """
    from tools.inspect_cluster import register_inspect_cluster
    from tools.inspect_weka import register_inspect_weka
    from tools.blueprints import register_blueprint_tools
    from tools.crd_schema import register_crd_schema
    from tools.validate_yaml import register_validate_yaml
    from tools.apply_tool import register_apply
    from tools.status_tool import register_status

    cap = _RegistryCapture()
    register_inspect_cluster(cap)
    register_inspect_weka(cap)
    register_blueprint_tools(cap)
    register_crd_schema(cap)
    register_validate_yaml(cap)
    register_apply(cap)
    register_status(cap)

    return cap.captured


def build_openclaw_config(tool_descriptions: dict[str, str]) -> dict:
    """Build the openclaw.json structure from extracted tool descriptions.

    Structure follows the target format from RESEARCH.md / CONTEXT.md:
    - transport: stdio
    - startup: python -m server from mcp-server/ directory
    - env: required=[BLUEPRINTS_DIR], optional=[KUBERNETES_AUTH_MODE, LOG_LEVEL, KUBECONFIG]
    - tools: [{name, description}] in server.py registration order
    """
    # Build tools list in canonical registration order
    tools = []
    for name in TOOL_REGISTRATION_ORDER:
        description = tool_descriptions.get(name, "")
        tools.append({"name": name, "description": description})

    # Warn if any tool is missing
    missing = [n for n in TOOL_REGISTRATION_ORDER if n not in tool_descriptions]
    if missing:
        print(f"WARNING: Expected tools not found in registrations: {missing}", file=sys.stderr)

    # Warn if unexpected extra tools were registered
    extra = [n for n in tool_descriptions if n not in TOOL_REGISTRATION_ORDER]
    if extra:
        print(f"WARNING: Extra tools registered (not in expected list): {extra}", file=sys.stderr)
        # Append extras at the end so nothing is silently dropped
        for name in extra:
            tools.append({"name": name, "description": tool_descriptions[name]})

    return {
        "_comment": (
            "Best-effort format -- may need revision when NemoClaw alpha schema is published."
        ),
        "name": "weka-app-store-mcp",
        "description": (
            "MCP server for the WEKA App Store. Provides tools for inspecting cluster "
            "and WEKA storage resources, browsing blueprint catalogs, validating WekaAppStore "
            "YAML manifests, applying blueprints to the cluster, and monitoring deployment status."
        ),
        "transport": "stdio",
        "startup": {
            "command": "python",
            "args": ["-m", "server"],
            "cwd": "mcp-server/",
            "env": {
                "PYTHONPATH": ".:../app-store-gui",
            },
        },
        "env": {
            "required": ["BLUEPRINTS_DIR"],
            "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "KUBECONFIG"],
        },
        "container": "weka-app-store-mcp:latest",
        "skill": "mcp-server/SKILL.md",
        "tools": tools,
    }


def generate(output_path: Path | None = None) -> dict:
    """Generate openclaw.json and write it to disk.

    Args:
        output_path: Where to write the file. Defaults to
            <this script's directory>/openclaw.json (i.e. mcp-server/openclaw.json).

    Returns:
        The generated config dict (useful for testing).
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "openclaw.json"

    tool_descriptions = collect_tool_descriptions()
    config = build_openclaw_config(tool_descriptions)

    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return config


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "openclaw.json"
    config = generate(output_path)

    tool_count = len(config["tools"])
    print(f"Generated {output_path}")
    print(f"Registered tools ({tool_count}):")
    for tool in config["tools"]:
        print(f"  - {tool['name']}")
    print(f"\nTotal: {tool_count} tools")
