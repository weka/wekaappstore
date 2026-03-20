"""Tests for blueprint scanner, list_blueprints, and get_blueprint tools.

TDD — tests written BEFORE implementation. Run against blueprints.py.
Fixtures in tests/fixtures/sample_blueprints/ provide two WekaAppStore CRs:
  - ai-research.yaml  (2 components)
  - data-pipeline.yaml (1 component, 1 prerequisite)
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import FastMCP

# Path to sample fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_blueprints"


# ===========================================================================
# scan_blueprints tests
# ===========================================================================


def test_scan_blueprints_finds_yaml():
    """Given a dir with 2 WekaAppStore YAMLs, scanner returns 2 entries."""
    from tools.blueprints import scan_blueprints

    results = scan_blueprints(str(FIXTURES_DIR))
    assert len(results) == 2

    names = {r["manifest"]["metadata"]["name"] for r in results}
    assert names == {"ai-research", "data-pipeline"}


def test_scan_blueprints_ignores_non_wekaappstore(tmp_path: Path):
    """Given YAMLs that are NOT WekaAppStore kind, scanner returns empty."""
    from tools.blueprints import scan_blueprints

    # A ConfigMap — different kind
    (tmp_path / "configmap.yaml").write_text(
        textwrap.dedent("""\
            apiVersion: v1
            kind: ConfigMap
            metadata:
              name: not-a-blueprint
        """)
    )
    # A custom resource with warp.io apiVersion but wrong kind
    (tmp_path / "wrong-kind.yaml").write_text(
        textwrap.dedent("""\
            apiVersion: warp.io/v1alpha1
            kind: WekaCluster
            metadata:
              name: also-not-a-blueprint
        """)
    )

    results = scan_blueprints(str(tmp_path))
    assert results == []


def test_scan_blueprints_empty_dir(tmp_path: Path):
    """Given empty dir, returns empty list (no error)."""
    from tools.blueprints import scan_blueprints

    results = scan_blueprints(str(tmp_path))
    assert results == []


def test_scan_blueprints_missing_dir():
    """Given nonexistent dir, returns empty list (no error)."""
    from tools.blueprints import scan_blueprints

    results = scan_blueprints("/nonexistent/path/that/does/not/exist")
    assert results == []


# ===========================================================================
# list_blueprints tool response contract tests
# ===========================================================================


def test_list_blueprints_flat_response():
    """Response has captured_at, blueprints, warnings, and count fields."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-list-flat")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        # Call the function directly via the tool function — find it by name
        tool_fn = _get_tool_fn(mcp, "list_blueprints")
        result = tool_fn()

    assert "captured_at" in result
    assert "blueprints" in result
    assert "warnings" in result
    assert "count" in result
    assert isinstance(result["blueprints"], list)
    assert isinstance(result["warnings"], list)
    assert result["count"] == len(result["blueprints"])


def test_list_blueprints_each_entry_has_metadata():
    """Each blueprint entry has name, namespace, component_count, component_names, source_file."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-list-metadata")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        tool_fn = _get_tool_fn(mcp, "list_blueprints")
        result = tool_fn()

    assert result["count"] == 2
    for entry in result["blueprints"]:
        assert "name" in entry
        assert "namespace" in entry
        assert "component_count" in entry
        assert "component_names" in entry
        assert "source_file" in entry
        assert isinstance(entry["component_names"], list)
        # source_file should be basename only (no path)
        assert "/" not in entry["source_file"]
        assert "\\" not in entry["source_file"]

    # Verify specific blueprint metadata
    names = {e["name"] for e in result["blueprints"]}
    assert "ai-research" in names
    assert "data-pipeline" in names

    ai = next(e for e in result["blueprints"] if e["name"] == "ai-research")
    assert ai["component_count"] == 2
    assert set(ai["component_names"]) == {"vector-db", "research-api"}

    dp = next(e for e in result["blueprints"] if e["name"] == "data-pipeline")
    assert dp["component_count"] == 1
    assert dp["component_names"] == ["spark-operator"]


def test_list_blueprints_empty_dir_has_warning(tmp_path: Path):
    """When BLUEPRINTS_DIR is empty, response includes a warning."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-list-empty")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(tmp_path)
        tool_fn = _get_tool_fn(mcp, "list_blueprints")
        result = tool_fn()

    assert result["count"] == 0
    assert result["blueprints"] == []
    assert len(result["warnings"]) >= 1
    warning_text = " ".join(result["warnings"])
    assert "No blueprints found" in warning_text


# ===========================================================================
# get_blueprint tool tests
# ===========================================================================


def test_get_blueprint_known_name():
    """Given 'ai-research' name with fixtures, returns full detail including spec fields."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-get-known")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        tool_fn = _get_tool_fn(mcp, "get_blueprint")
        result = tool_fn(name="ai-research")

    assert "error" not in result
    assert result["name"] == "ai-research"
    assert result["namespace"] == "ai-platform"
    assert "captured_at" in result
    assert "components" in result
    assert "prerequisites" in result
    assert "warnings" in result
    assert len(result["components"]) == 2


def test_get_blueprint_unknown_name():
    """Given unknown name, returns error response with available_names list."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-get-unknown")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        tool_fn = _get_tool_fn(mcp, "get_blueprint")
        result = tool_fn(name="nonexistent-blueprint")

    assert "error" in result
    assert "available_names" in result
    assert "requested_name" in result
    assert "captured_at" in result
    assert isinstance(result["available_names"], list)
    assert "ai-research" in result["available_names"]
    assert "data-pipeline" in result["available_names"]


def test_get_blueprint_flat_response():
    """All values in response reachable in <=2 key traversals."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-get-flat")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        tool_fn = _get_tool_fn(mcp, "get_blueprint")
        result = tool_fn(name="ai-research")

    # Check no deeply nested objects — all values at top level or 1 list deep
    for key, value in result.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    # List-of-dicts items must themselves be flat (no nested dicts)
                    for sub_key, sub_val in item.items():
                        assert not isinstance(sub_val, dict), (
                            f"Depth violation: result[{key!r}][item][{sub_key!r}] is a dict"
                        )
        elif isinstance(value, dict):
            # No nested dicts at top level
            pytest.fail(f"Depth violation: result[{key!r}] is a nested dict")


def test_get_blueprint_components_flat():
    """Each component in response is flat dict — no nested helm_chart sub-dict."""
    from tools.blueprints import register_blueprint_tools

    mcp = FastMCP("test-get-components-flat")
    register_blueprint_tools(mcp)

    with patch("tools.blueprints.config") as mock_cfg:
        mock_cfg.BLUEPRINTS_DIR = str(FIXTURES_DIR)
        tool_fn = _get_tool_fn(mcp, "get_blueprint")
        result = tool_fn(name="ai-research")

    required_keys = {
        "name",
        "enabled",
        "target_namespace",
        "depends_on",
        "helm_chart_name",
        "helm_chart_version",
        "helm_chart_repository",
        "helm_chart_release_name",
        "wait_for_ready",
    }
    for component in result["components"]:
        # Must not have nested helm_chart sub-dict
        assert "helm_chart" not in component, "helm_chart sub-dict must be flattened"
        for key in required_keys:
            assert key in component, f"Component missing key: {key}"


# ===========================================================================
# Helper
# ===========================================================================


def _get_tool_fn(mcp: FastMCP, name: str):
    """Extract tool callable from FastMCP instance by name."""
    # FastMCP stores tools in _tool_manager or similar; access via tools dict
    for tool_name, tool in mcp._tool_manager._tools.items():
        if tool_name == name:
            return tool.fn
    raise KeyError(f"Tool {name!r} not found in FastMCP instance")
