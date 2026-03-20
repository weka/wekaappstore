"""Tests for the inspect_cluster MCP tool and flatten function."""
from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import patch

from tools.inspect_cluster import flatten_inspect_cluster_for_mcp


# --- flatten function tests ---------------------------------------------------

def test_inspect_cluster_flat_response(sample_cluster_snapshot: dict[str, Any]) -> None:
    """Given a mocked cluster snapshot, flatten returns expected flat fields."""
    result = flatten_inspect_cluster_for_mcp(sample_cluster_snapshot)

    assert "captured_at" in result
    assert result["captured_at"] == "2026-03-20T00:00:00Z"

    # GPU aggregates
    assert "gpu_total" in result
    assert isinstance(result["gpu_total"], int)
    assert result["gpu_total"] == 4  # 4x NVIDIA L40

    assert "gpu_models" in result
    assert isinstance(result["gpu_models"], list)
    assert "NVIDIA L40" in result["gpu_models"]

    assert "gpu_memory_total_gib" in result
    assert isinstance(result["gpu_memory_total_gib"], float)
    assert result["gpu_memory_total_gib"] == 192.0  # 4 * 48 GiB

    # Top-level warnings array
    assert "warnings" in result
    assert isinstance(result["warnings"], list)

    # No forbidden keys
    assert "inspection_snapshot" not in result
    assert "domains" not in result


def test_inspect_cluster_all_keys_depth_2(sample_cluster_snapshot: dict[str, Any]) -> None:
    """Every value in response is reachable in <= 2 key traversals."""
    result = flatten_inspect_cluster_for_mcp(sample_cluster_snapshot)

    for key, value in result.items():
        if isinstance(value, dict):
            # A nested dict would require 2 traversals just to get here — that's 3 total
            pytest.fail(
                f"Key '{key}' is a dict — violates 2-traversal depth constraint. "
                f"Value: {value!r}"
            )
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    # list[dict] is acceptable IF the dict itself contains no nested dicts
                    for sub_key, sub_val in item.items():
                        assert not isinstance(sub_val, dict), (
                            f"result['{key}'][{idx}]['{sub_key}'] is a dict — "
                            f"exceeds 2-traversal depth constraint"
                        )


def test_inspect_cluster_no_forbidden_keys(sample_cluster_snapshot: dict[str, Any]) -> None:
    """Response must not contain inspection_snapshot or domains key."""
    result = flatten_inspect_cluster_for_mcp(sample_cluster_snapshot)
    assert "inspection_snapshot" not in result
    assert "domains" not in result


def test_inspect_cluster_warnings_from_blockers() -> None:
    """Blockers in any domain produce warnings at the top level."""
    snapshot = {
        "captured_at": "2026-03-20T00:00:00Z",
        "k8s_version": "v1.30.1",
        "gpu_operator_installed": True,
        "app_store_crd_installed": True,
        "app_store_cluster_init_present": False,
        "app_store_crs": [],
        "default_storage_class": None,
        "domains": {
            "cpu": {
                "status": "unavailable",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {},
                "notes": [],
                "blockers": [
                    {"code": "node_list_failed", "message": "Could not list nodes", "domain": "cpu"}
                ],
            },
            "memory": {
                "status": "unavailable",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {},
                "notes": [],
                "blockers": [],
            },
            "gpu": {
                "status": "unavailable",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {},
                "notes": [],
                "blockers": [],
            },
            "namespaces": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {"names": []},
                "notes": [],
                "blockers": [],
            },
            "storage_classes": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {"names": []},
                "notes": [],
                "blockers": [],
            },
        },
    }
    result = flatten_inspect_cluster_for_mcp(snapshot)
    assert "Could not list nodes" in result["warnings"]


def test_inspect_cluster_gpu_operator_false_adds_warning() -> None:
    """gpu_operator_installed=False adds a GPU warning to the warnings array."""
    snapshot = {
        "captured_at": "2026-03-20T00:00:00Z",
        "k8s_version": "v1.30.1",
        "gpu_operator_installed": False,
        "app_store_crd_installed": True,
        "app_store_cluster_init_present": False,
        "app_store_crs": [],
        "default_storage_class": None,
        "domains": {
            "cpu": {"status": "complete", "freshness": {}, "observed": {}, "notes": [], "blockers": []},
            "memory": {"status": "complete", "freshness": {}, "observed": {}, "notes": [], "blockers": []},
            "gpu": {
                "status": "complete",
                "freshness": {},
                "observed": {"inventory": []},
                "notes": [],
                "blockers": [],
            },
            "namespaces": {"status": "complete", "freshness": {}, "observed": {"names": []}, "notes": [], "blockers": []},
            "storage_classes": {"status": "complete", "freshness": {}, "observed": {"names": []}, "notes": [], "blockers": []},
        },
    }
    result = flatten_inspect_cluster_for_mcp(snapshot)
    assert any("GPU operator" in w for w in result["warnings"])


def test_inspect_cluster_k8s_error_returns_warnings() -> None:
    """When K8s API raises ApiException, tool returns dict with warnings, not exception."""
    from kubernetes.client.rest import ApiException
    from tools.inspect_cluster import register_inspect_cluster
    from mcp.server.fastmcp import FastMCP
    import logging

    # Create a fresh FastMCP instance for testing
    test_mcp = FastMCP("test-mcp")
    register_inspect_cluster(test_mcp)

    exc = ApiException(status=500, reason="Internal Server Error")
    with patch("webapp.inspection.cluster.collect_cluster_inspection", side_effect=exc):
        # Call the inner function directly by finding it in the tool registry
        tool_fn = None
        for tool in test_mcp._tool_manager._tools.values():
            if tool.name == "inspect_cluster":
                tool_fn = tool.fn
                break
        assert tool_fn is not None, "inspect_cluster tool not registered"

        result = tool_fn()
        assert isinstance(result, dict)
        assert "warnings" in result
        assert len(result["warnings"]) > 0
        assert any("K8s API" in w for w in result["warnings"])
