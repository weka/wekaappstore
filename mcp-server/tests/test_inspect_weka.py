"""Tests for the inspect_weka MCP tool and flatten function."""
from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import patch

from tools.inspect_weka import flatten_inspect_weka_for_mcp


# --- flatten function tests ---------------------------------------------------

def test_inspect_weka_flat_response(sample_weka_snapshot: dict[str, Any]) -> None:
    """Given a mocked WEKA snapshot, flatten returns expected flat fields."""
    result = flatten_inspect_weka_for_mcp(sample_weka_snapshot)

    assert "captured_at" in result
    assert result["captured_at"] == "2026-03-20T00:00:00Z"

    assert "weka_cluster_name" in result
    assert result["weka_cluster_name"] == "weka-prod"

    assert "total_capacity_gib" in result
    assert isinstance(result["total_capacity_gib"], float)
    # 2199023255552 bytes = 2048.0 GiB
    assert result["total_capacity_gib"] == pytest.approx(2048.0, abs=0.1)

    assert "used_capacity_gib" in result
    assert isinstance(result["used_capacity_gib"], float)

    assert "filesystems" in result
    assert isinstance(result["filesystems"], list)
    assert len(result["filesystems"]) == 1

    fs = result["filesystems"][0]
    assert "name" in fs
    assert "size_gib" in fs
    assert "used_gib" in fs
    assert fs["name"] == "weka-home"

    assert "warnings" in result
    assert isinstance(result["warnings"], list)

    # No forbidden keys
    assert "domains" not in result


def test_inspect_weka_all_keys_depth_2(sample_weka_snapshot: dict[str, Any]) -> None:
    """Every value in response is reachable in <= 2 key traversals."""
    result = flatten_inspect_weka_for_mcp(sample_weka_snapshot)

    for key, value in result.items():
        if isinstance(value, dict):
            pytest.fail(
                f"Key '{key}' is a dict — violates 2-traversal depth constraint."
            )
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    for sub_key, sub_val in item.items():
                        assert not isinstance(sub_val, dict), (
                            f"result['{key}'][{idx}]['{sub_key}'] is a dict — "
                            f"exceeds 2-traversal depth constraint"
                        )


def test_inspect_weka_no_domains_key(sample_weka_snapshot: dict[str, Any]) -> None:
    """Response must not contain the domains key."""
    result = flatten_inspect_weka_for_mcp(sample_weka_snapshot)
    assert "domains" not in result


def test_inspect_weka_filesystem_shape(sample_weka_snapshot: dict[str, Any]) -> None:
    """Filesystem entries are flat {name, size_gib, used_gib} dicts."""
    result = flatten_inspect_weka_for_mcp(sample_weka_snapshot)
    for fs in result["filesystems"]:
        assert set(fs.keys()) == {"name", "size_gib", "used_gib"}, (
            f"Filesystem entry has unexpected keys: {set(fs.keys())}"
        )
        assert isinstance(fs["size_gib"], float)
        assert isinstance(fs["used_gib"], float)


def test_inspect_weka_unavailable_returns_warnings() -> None:
    """When WEKA CRD not found (ApiException 404), tool returns warnings not exception."""
    from kubernetes.client.rest import ApiException
    from tools.inspect_weka import register_inspect_weka
    from mcp.server.fastmcp import FastMCP

    test_mcp = FastMCP("test-mcp")
    register_inspect_weka(test_mcp)

    exc = ApiException(status=404, reason="Not Found")
    with patch("webapp.inspection.weka.collect_weka_inspection", side_effect=exc):
        tool_fn = None
        for tool in test_mcp._tool_manager._tools.values():
            if tool.name == "inspect_weka":
                tool_fn = tool.fn
                break
        assert tool_fn is not None, "inspect_weka tool not registered"

        result = tool_fn()
        assert isinstance(result, dict)
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert any("K8s API" in w for w in result["warnings"])


def test_inspect_weka_blockers_become_warnings() -> None:
    """Blockers in the weka domain appear as warnings in the flat response."""
    snapshot = {
        "captured_at": "2026-03-20T00:00:00Z",
        "domains": {
            "weka": {
                "status": "unavailable",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {"clusters": [], "filesystems": [], "cluster_total_bytes": None, "cluster_free_bytes": None},
                "notes": [],
                "blockers": [
                    {"code": "weka_cluster_missing", "message": "No WekaCluster resources were visible.", "domain": "weka"}
                ],
            }
        },
    }
    result = flatten_inspect_weka_for_mcp(snapshot)
    assert "No WekaCluster resources were visible." in result["warnings"]
    assert result["weka_cluster_name"] is None
    assert result["total_capacity_gib"] == 0.0
