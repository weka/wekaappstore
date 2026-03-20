"""Tests for the get_crd_schema MCP tool.

Tests the CRD schema tool in isolation using the _get_crd_schema_impl
helper that accepts injectable API clients for testing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# --- sys.path setup (also done in conftest.py, but explicit here for clarity) ---
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BLUEPRINTS_DIR = str(MCP_SERVER_ROOT / "tests" / "fixtures" / "sample_blueprints")


def make_mock_crd() -> MagicMock:
    """Build a MagicMock representing the wekaappstores.warp.io CRD."""
    crd = MagicMock()
    crd.spec.group = "warp.io"
    crd.spec.names.kind = "WekaAppStore"

    # Build a version entry with an openAPIV3Schema
    schema_props = {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "properties": {
                    "appStack": {"type": "object"},
                },
            }
        },
    }
    version_entry = MagicMock()
    version_entry.name = "v1alpha1"
    version_entry.schema.open_apiv3_schema.to_dict.return_value = schema_props

    crd.spec.versions = [version_entry]
    return crd


def make_mock_apiextensions_api(crd: MagicMock | None = None) -> MagicMock:
    """Return a mock ApiextensionsV1Api that returns *crd* when queried."""
    api = MagicMock()
    if crd is not None:
        api.read_custom_resource_definition.return_value = crd
    return api


def make_404_apiextensions_api() -> MagicMock:
    """Return a mock ApiextensionsV1Api that raises a 404 ApiException."""
    from kubernetes.client.rest import ApiException

    api = MagicMock()
    api.read_custom_resource_definition.side_effect = ApiException(status=404, reason="Not Found")
    return api


def make_500_apiextensions_api() -> MagicMock:
    """Return a mock ApiextensionsV1Api that raises a 500 ApiException."""
    from kubernetes.client.rest import ApiException

    api = MagicMock()
    api.read_custom_resource_definition.side_effect = ApiException(
        status=500, reason="Internal Server Error"
    )
    return api


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_crd_schema_returns_shape():
    """Response has all required top-level keys."""
    from tools.crd_schema import _get_crd_schema_impl

    crd = make_mock_crd()
    api = make_mock_apiextensions_api(crd)
    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    assert "captured_at" in result
    assert "group" in result
    assert "version" in result
    assert "kind" in result
    assert "schema" in result
    assert "examples" in result
    assert "warnings" in result


def test_get_crd_schema_returns_correct_values():
    """group, version, kind extracted correctly from mock CRD."""
    from tools.crd_schema import _get_crd_schema_impl

    crd = make_mock_crd()
    api = make_mock_apiextensions_api(crd)
    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    assert result["group"] == "warp.io"
    assert result["version"] == "v1alpha1"
    assert result["kind"] == "WekaAppStore"
    assert result["schema"] is not None
    assert isinstance(result["schema"], dict)
    assert result["warnings"] == []


def test_get_crd_schema_crd_not_installed():
    """When apiextensions raises 404, returns warning and null schema."""
    from tools.crd_schema import _get_crd_schema_impl

    api = make_404_apiextensions_api()
    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    assert result["schema"] is None
    assert any("WekaAppStore CRD not installed" in w for w in result["warnings"])
    assert result["captured_at"] is not None


def test_get_crd_schema_k8s_unavailable():
    """When apiextensions raises non-404 error, returns K8s unavailable warning."""
    from tools.crd_schema import _get_crd_schema_impl

    api = make_500_apiextensions_api()
    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    assert result["schema"] is None
    assert any("K8s" in w or "unavailable" in w.lower() for w in result["warnings"])


def test_get_crd_schema_includes_examples():
    """When blueprints exist, examples list has 1-2 entries that are valid YAML strings."""
    from tools.crd_schema import _get_crd_schema_impl

    crd = make_mock_crd()
    api = make_mock_apiextensions_api(crd)
    result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=SAMPLE_BLUEPRINTS_DIR)

    assert isinstance(result["examples"], list)
    assert 1 <= len(result["examples"]) <= 2

    for example in result["examples"]:
        assert isinstance(example, str)
        # Must be valid YAML
        parsed = yaml.safe_load(example)
        assert isinstance(parsed, dict)
        # Must be a WekaAppStore manifest
        assert parsed.get("apiVersion", "").startswith("warp.io")
        assert parsed.get("kind") == "WekaAppStore"


def test_get_crd_schema_no_examples_warning():
    """When BLUEPRINTS_DIR is empty or missing, examples is [] and warnings has note."""
    import tempfile

    from tools.crd_schema import _get_crd_schema_impl

    crd = make_mock_crd()
    api = make_mock_apiextensions_api(crd)

    with tempfile.TemporaryDirectory() as empty_dir:
        result = _get_crd_schema_impl(apiextensions_api=api, blueprints_dir=empty_dir)

    assert result["examples"] == []
    assert any("no example" in w.lower() for w in result["warnings"])


def test_get_crd_schema_missing_dir_warning():
    """When BLUEPRINTS_DIR does not exist, warnings includes note about missing dir."""
    from tools.crd_schema import _get_crd_schema_impl

    crd = make_mock_crd()
    api = make_mock_apiextensions_api(crd)
    result = _get_crd_schema_impl(
        apiextensions_api=api, blueprints_dir="/nonexistent/path/to/blueprints"
    )

    assert result["examples"] == []
    assert any("no example" in w.lower() for w in result["warnings"])
