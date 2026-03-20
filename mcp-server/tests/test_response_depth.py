"""Cross-tool response depth contract enforcer.

Validates that ALL tool responses satisfy the 2-key traversal depth contract:
  - Every value is reachable in <=2 key/index traversals from the response root
  - Lists of primitives count as depth 0 from their parent key
  - Lists of dicts add 1 traversal per level
  - Exception: 'schema' field in get_crd_schema is pass-through K8s CRD data
    (inherently deeply nested OpenAPI schema — not our structural decision)

This test file is intentionally cross-tool so that any future regression in
any tool's response shape is caught by a single failing test here.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# --- sys.path setup ---
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_ROOT))

SAMPLE_BLUEPRINTS_DIR = str(MCP_SERVER_ROOT / "tests" / "fixtures" / "sample_blueprints")


# ---------------------------------------------------------------------------
# Depth-check helper
# ---------------------------------------------------------------------------


def check_depth(
    obj: Any,
    max_depth: int = 2,
    path: str = "",
    exempt_keys: frozenset[str] | None = None,
    _current_depth: int = 0,
) -> None:
    """Recursively assert that no non-exempt value requires >max_depth key traversals.

    Depth counting rules:
    - Primitive (str, int, float, bool, None): depth = 0 from parent
    - List of primitives: depth = 0 from parent key (no additional traversal)
    - List of dicts: each dict adds 1 traversal level
    - Dict: each key adds 1 traversal level

    Args:
        obj: The value to check.
        max_depth: Maximum allowed depth from the response root.
        path: Dot-notation path for error messages.
        exempt_keys: Set of top-level key names whose values are exempt.
        _current_depth: Internal recursion counter (do not pass externally).

    Raises:
        AssertionError: If any non-exempt value exceeds max_depth.
    """
    if exempt_keys is None:
        exempt_keys = frozenset({"schema"})

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key

            # Check if this top-level key is exempt
            top_key = child_path.split(".")[0]
            if top_key in exempt_keys:
                continue  # Skip depth check for exempt keys

            if isinstance(value, dict):
                new_depth = _current_depth + 1
                assert new_depth <= max_depth, (
                    f"Depth violation at '{child_path}': dict at depth {new_depth} "
                    f"(max {max_depth}). Use flat fields instead of nested objects."
                )
                check_depth(
                    value,
                    max_depth=max_depth,
                    path=child_path,
                    exempt_keys=exempt_keys,
                    _current_depth=new_depth,
                )
            elif isinstance(value, list):
                check_depth(
                    value,
                    max_depth=max_depth,
                    path=child_path,
                    exempt_keys=exempt_keys,
                    _current_depth=_current_depth,
                )
            # Primitives at any depth within max_depth: OK

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            item_path = f"{path}[{idx}]"
            top_key = item_path.split(".")[0].split("[")[0]
            if top_key in exempt_keys:
                continue

            if isinstance(item, dict):
                new_depth = _current_depth + 1
                assert new_depth <= max_depth, (
                    f"Depth violation at '{item_path}': dict at depth {new_depth} "
                    f"(max {max_depth}). Use flat fields instead of nested objects."
                )
                check_depth(
                    item,
                    max_depth=max_depth,
                    path=item_path,
                    exempt_keys=exempt_keys,
                    _current_depth=new_depth,
                )
            elif isinstance(item, list):
                # Nested list — check contents
                check_depth(
                    item,
                    max_depth=max_depth,
                    path=item_path,
                    exempt_keys=exempt_keys,
                    _current_depth=_current_depth,
                )
            # Primitive list items: no depth added


# ---------------------------------------------------------------------------
# inspect_cluster depth test
# ---------------------------------------------------------------------------


def test_response_depth_inspect_cluster(sample_cluster_snapshot: dict[str, Any]) -> None:
    """inspect_cluster flat response has no value requiring >2 key traversals."""
    from tools.inspect_cluster import flatten_inspect_cluster_for_mcp

    result = flatten_inspect_cluster_for_mcp(sample_cluster_snapshot)
    check_depth(result, max_depth=2)


# ---------------------------------------------------------------------------
# inspect_weka depth test
# ---------------------------------------------------------------------------


def test_response_depth_inspect_weka(sample_weka_snapshot: dict[str, Any]) -> None:
    """inspect_weka flat response has no value requiring >2 key traversals."""
    from tools.inspect_weka import flatten_inspect_weka_for_mcp

    result = flatten_inspect_weka_for_mcp(sample_weka_snapshot)
    check_depth(result, max_depth=2)


# ---------------------------------------------------------------------------
# list_blueprints depth test
# ---------------------------------------------------------------------------


def test_response_depth_list_blueprints() -> None:
    """list_blueprints response has no value requiring >2 key traversals.

    blueprints[0].name = 2 traversals (list -> dict -> str): OK.
    """
    from mcp.server.fastmcp import FastMCP
    from tools.blueprints import register_blueprint_tools

    import config

    original_dir = config.BLUEPRINTS_DIR
    config.BLUEPRINTS_DIR = SAMPLE_BLUEPRINTS_DIR

    try:
        test_mcp = FastMCP("test")
        register_blueprint_tools(test_mcp)

        # Get list_blueprints by calling it directly via tool manager
        # We'll call the underlying function by re-importing with patched config
        from tools.blueprints import scan_blueprints, flatten_blueprint_summary, _utc_now
        entries = scan_blueprints(SAMPLE_BLUEPRINTS_DIR)
        result = {
            "captured_at": _utc_now(),
            "count": len(entries),
            "blueprints": [flatten_blueprint_summary(e) for e in entries],
            "warnings": [],
        }
        check_depth(result, max_depth=2)
    finally:
        config.BLUEPRINTS_DIR = original_dir


# ---------------------------------------------------------------------------
# get_blueprint depth test
# ---------------------------------------------------------------------------


def test_response_depth_get_blueprint() -> None:
    """get_blueprint response has no value requiring >2 key traversals.

    components[0].helm_chart_name = 2 traversals (list -> dict -> str): OK.
    """
    from tools.blueprints import scan_blueprints, flatten_blueprint_detail

    entries = scan_blueprints(SAMPLE_BLUEPRINTS_DIR)
    assert entries, "Need at least one blueprint fixture for depth test"

    result = flatten_blueprint_detail(entries[0])
    check_depth(result, max_depth=2)


# ---------------------------------------------------------------------------
# get_crd_schema depth test — schema field exempted
# ---------------------------------------------------------------------------


def _make_mock_crd() -> MagicMock:
    crd = MagicMock()
    crd.spec.group = "warp.io"
    crd.spec.names.kind = "WekaAppStore"
    schema_props = {"type": "object", "properties": {"spec": {"type": "object", "properties": {}}}}
    version_entry = MagicMock()
    version_entry.name = "v1alpha1"
    version_entry.schema.open_apiv3_schema.to_dict.return_value = schema_props
    crd.spec.versions = [version_entry]
    return crd


def test_response_depth_get_crd_schema() -> None:
    """get_crd_schema response: all fields within 2-key depth; 'schema' field exempted.

    The 'schema' key contains pass-through K8s CRD OpenAPI data that is
    inherently deeply nested. This is NOT our structural decision — it is
    raw K8s API output. The 2-key depth rule applies to OUR domain model
    fields only.
    """
    from tools.crd_schema import _get_crd_schema_impl

    crd = _make_mock_crd()
    api = MagicMock()
    api.read_custom_resource_definition.return_value = crd

    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    # All non-schema fields must pass the depth check
    check_depth(result, max_depth=2, exempt_keys=frozenset({"schema"}))

    # Verify schema IS present and is a dict (deeply nested is OK here)
    assert isinstance(result["schema"], dict)

    # Examples are a list of strings — depth 0 from 'examples' key
    assert isinstance(result["examples"], list)
    for example in result["examples"]:
        assert isinstance(example, str), "examples must contain strings, not nested objects"


# ---------------------------------------------------------------------------
# validate_yaml depth test
# ---------------------------------------------------------------------------


def test_response_depth_validate_yaml() -> None:
    """validate_yaml response has no value requiring >2 key traversals."""
    from tools.validate_yaml import _validate_yaml_impl

    valid_yaml = """
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: test-app
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-app
    version: 1.0.0
"""
    result = _validate_yaml_impl(valid_yaml)
    assert result["valid"] is True
    check_depth(result, max_depth=2)


# ---------------------------------------------------------------------------
# apply depth test
# ---------------------------------------------------------------------------


def test_response_depth_apply() -> None:
    """apply response has no value requiring >2 key traversals.

    Uses confirmed=False to get structured error without needing mocked K8s.
    """
    from tools.apply_tool import _apply_impl

    result = _apply_impl(
        yaml_text="apiVersion: warp.io/v1alpha1\nkind: WekaAppStore\n",
        namespace="default",
        confirmed=False,
    )
    assert result["applied"] is False
    assert result["error"] == "approval_required"
    check_depth(result, max_depth=2)


# ---------------------------------------------------------------------------
# status depth test
# ---------------------------------------------------------------------------


def test_response_depth_status() -> None:
    """status response has no value requiring >2 key traversals.

    conditions[0].type = depth 2 (conditions=list->dict->str): OK.
    component_status[0].name = depth 2: OK.
    """
    from tools.status_tool import _status_impl

    mock_api = MagicMock()
    mock_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "test-app", "namespace": "default"},
        "spec": {},
        "status": {
            "releaseStatus": "deployed",
            "releaseName": "test-app-release",
            "releaseVersion": 1,
            "appStackPhase": "Ready",
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "reason": "ReconcileSuccess",
                    "message": "All components healthy",
                    "lastTransitionTime": "2026-03-20T06:00:00Z",
                }
            ],
            "componentStatus": [
                {
                    "name": "nginx",
                    "phase": "Ready",
                    "releaseName": "nginx-release",
                    "releaseVersion": 1,
                    "message": "",
                    "lastTransitionTime": "2026-03-20T06:00:00Z",
                }
            ],
        },
    }

    result = _status_impl(name="test-app", namespace="default", custom_objects_api=mock_api)
    assert result["found"] is True
    check_depth(result, max_depth=2)
