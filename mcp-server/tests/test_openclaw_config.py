"""Tests for mcp-server/openclaw.json validity and sync with generation script.

Verifies:
- File exists and is valid JSON
- Contains all 8 registered tools
- Tool names match server.py registrations
- Transport is 'streamable-http'
- URL points to localhost MCP endpoint
- No startup block (HTTP transport uses url, not subprocess)
- Required env vars are declared
- Optional env vars include MCP_TRANSPORT and MCP_PORT
- All tools have non-empty descriptions
- File matches what generate_openclaw_config.py would produce (drift detection)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Path to mcp-server/ directory — always relative to this file
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
OPENCLAW_JSON_PATH = MCP_SERVER_ROOT / "openclaw.json"

# Expected tool names — must match server.py TOOL_REGISTRATION_ORDER
EXPECTED_TOOL_NAMES = {
    "inspect_cluster",
    "inspect_weka",
    "list_blueprints",
    "get_blueprint",
    "get_crd_schema",
    "validate_yaml",
    "apply",
    "status",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openclaw_config() -> dict:
    """Load openclaw.json once for the module."""
    return json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_openclaw_json_exists():
    """openclaw.json exists at mcp-server/openclaw.json."""
    assert OPENCLAW_JSON_PATH.exists(), (
        f"openclaw.json not found at {OPENCLAW_JSON_PATH}. "
        "Run: cd mcp-server && PYTHONPATH=.:../app-store-gui python generate_openclaw_config.py"
    )


def test_openclaw_json_valid():
    """openclaw.json is valid JSON and loads without error."""
    content = OPENCLAW_JSON_PATH.read_text(encoding="utf-8")
    config = json.loads(content)  # raises JSONDecodeError if invalid
    assert isinstance(config, dict), "openclaw.json root must be a JSON object"


def test_openclaw_json_has_8_tools(openclaw_config: dict):
    """openclaw.json tools array has exactly 8 entries."""
    tools = openclaw_config.get("tools", [])
    assert len(tools) == 8, (
        f"Expected 8 tools in openclaw.json, got {len(tools)}: "
        f"{[t.get('name') for t in tools]}"
    )


def test_openclaw_json_tool_names_match_server(openclaw_config: dict):
    """Tool names in openclaw.json match the 8 tools registered in server.py."""
    tools = openclaw_config.get("tools", [])
    actual_names = {t.get("name") for t in tools}
    assert actual_names == EXPECTED_TOOL_NAMES, (
        f"Tool name mismatch.\n"
        f"Expected: {sorted(EXPECTED_TOOL_NAMES)}\n"
        f"Got:      {sorted(actual_names)}\n"
        f"Missing:  {sorted(EXPECTED_TOOL_NAMES - actual_names)}\n"
        f"Extra:    {sorted(actual_names - EXPECTED_TOOL_NAMES)}"
    )


def test_openclaw_json_has_http_transport(openclaw_config: dict):
    """openclaw.json transport field is 'streamable-http'."""
    transport = openclaw_config.get("transport")
    assert transport == "streamable-http", (
        f"Expected transport='streamable-http', got {transport!r}"
    )


def test_openclaw_json_has_url(openclaw_config: dict):
    """openclaw.json url points to localhost MCP endpoint."""
    assert openclaw_config.get("url") == "http://localhost:8080/mcp", (
        f"Expected url='http://localhost:8080/mcp', got {openclaw_config.get('url')!r}"
    )


def test_openclaw_json_no_startup_block(openclaw_config: dict):
    """openclaw.json has no startup block (HTTP transport uses url, not subprocess)."""
    assert "startup" not in openclaw_config, (
        "openclaw.json must not have a 'startup' block when using streamable-http transport"
    )


def test_openclaw_json_optional_env_includes_transport_vars(openclaw_config: dict):
    """openclaw.json env.optional includes MCP_TRANSPORT and MCP_PORT."""
    optional = openclaw_config.get("env", {}).get("optional", [])
    assert "MCP_TRANSPORT" in optional, (
        f"'MCP_TRANSPORT' must be in env.optional. Got: {optional}"
    )
    assert "MCP_PORT" in optional, (
        f"'MCP_PORT' must be in env.optional. Got: {optional}"
    )


def test_openclaw_json_has_required_env(openclaw_config: dict):
    """openclaw.json env.required contains 'BLUEPRINTS_DIR'."""
    env = openclaw_config.get("env", {})
    required = env.get("required", [])
    assert "BLUEPRINTS_DIR" in required, (
        f"'BLUEPRINTS_DIR' must be in env.required. Got: {required}"
    )


def test_openclaw_json_all_tools_have_descriptions(openclaw_config: dict):
    """Every tool entry in openclaw.json has a non-empty description string."""
    tools = openclaw_config.get("tools", [])
    missing_descriptions = []
    for tool in tools:
        name = tool.get("name", "<unnamed>")
        description = tool.get("description", "")
        if not isinstance(description, str) or not description.strip():
            missing_descriptions.append(name)

    assert not missing_descriptions, (
        f"These tools are missing descriptions in openclaw.json: {missing_descriptions}"
    )


def test_openclaw_json_matches_generation():
    """openclaw.json matches what generate_openclaw_config.py produces.

    Runs the generation script programmatically into a temp buffer (no file write)
    and compares the tool list. Detects drift when tool docstrings change but
    openclaw.json is not regenerated.
    """
    import sys

    # Import the generator — requires PYTHONPATH=.:../app-store-gui
    sys.path.insert(0, str(MCP_SERVER_ROOT))
    from generate_openclaw_config import collect_tool_descriptions, build_openclaw_config

    # Generate fresh config in memory (no file write)
    tool_descriptions = collect_tool_descriptions()
    fresh_config = build_openclaw_config(tool_descriptions)

    # Load the on-disk config
    on_disk_config = json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))

    # Compare tool lists (names and descriptions)
    fresh_tools = {t["name"]: t["description"] for t in fresh_config.get("tools", [])}
    disk_tools = {t["name"]: t["description"] for t in on_disk_config.get("tools", [])}

    assert fresh_tools == disk_tools, (
        "openclaw.json is out of sync with tool descriptions in tools/*.py.\n"
        "Run: cd mcp-server && PYTHONPATH=.:../app-store-gui python generate_openclaw_config.py\n"
        "Drifted tools: "
        + str([
            name for name in fresh_tools
            if fresh_tools.get(name) != disk_tools.get(name)
        ])
    )
