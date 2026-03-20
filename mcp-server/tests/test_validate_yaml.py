"""Unit tests for validate_yaml tool.

Tests call _validate_yaml_impl() directly — no FastMCP needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- sys.path setup ---
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_HELMCHART_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-chart
    version: "1.0.0"
    releaseName: my-release
"""

VALID_APPSTACK_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-stack
spec:
  appStack:
    components:
      - name: frontend
        enabled: true
        helmChart:
          repository: https://charts.example.com
          name: frontend
          version: "1.0.0"
"""

SYNTAX_ERROR_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: bad
  broken: [unclosed
"""

WRONG_KIND_YAML = """\
apiVersion: warp.io/v1alpha1
kind: NotWekaAppStore
metadata:
  name: some-resource
spec:
  helmChart:
    name: foo
"""

WRONG_API_VERSION_YAML = """\
apiVersion: apps/v1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  helmChart:
    name: my-chart
"""

MISSING_NAME_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  namespace: default
spec:
  helmChart:
    name: my-chart
"""

V1_BLUEPRINT_FAMILY_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  blueprint_family: gpu-workloads
  helmChart:
    name: my-chart
"""

V1_FIT_FINDINGS_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  fit_findings:
    - ok
  helmChart:
    name: my-chart
"""

MISSING_DEPLOYMENT_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  targetNamespace: default
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_yaml_passes() -> None:
    """Valid WekaAppStore YAML with helmChart returns valid=True and empty errors."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(VALID_HELMCHART_YAML)

    assert result["valid"] is True
    assert result["errors"] == []


def test_valid_yaml_appstack() -> None:
    """Valid WekaAppStore YAML with appStack returns valid=True."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(VALID_APPSTACK_YAML)

    assert result["valid"] is True
    assert result["errors"] == []


def test_rejects_yaml_syntax_error() -> None:
    """Unparseable YAML returns valid=False with yaml_parse_error code."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(SYNTAX_ERROR_YAML)

    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    codes = [e["code"] for e in result["errors"]]
    assert "yaml_parse_error" in codes


def test_rejects_no_wekaappstore_doc() -> None:
    """YAML with wrong kind returns no_wekaappstore_doc error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(WRONG_KIND_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "no_wekaappstore_doc" in codes


def test_rejects_invalid_api_version() -> None:
    """Wrong apiVersion returns invalid_api_version error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(WRONG_API_VERSION_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "invalid_api_version" in codes


def test_rejects_missing_name() -> None:
    """Missing metadata.name returns missing_required_field error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(MISSING_NAME_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "missing_required_field" in codes


def test_rejects_v1_blueprint_family() -> None:
    """spec.blueprint_family returns v1_only_field error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(V1_BLUEPRINT_FAMILY_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "v1_only_field" in codes
    # Confirm the path points to blueprint_family
    paths = [e["path"] for e in result["errors"]]
    assert any("blueprint_family" in p for p in paths)


def test_rejects_v1_fit_findings() -> None:
    """spec.fit_findings returns v1_only_field error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(V1_FIT_FINDINGS_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "v1_only_field" in codes
    paths = [e["path"] for e in result["errors"]]
    assert any("fit_findings" in p for p in paths)


def test_rejects_missing_deployment_method() -> None:
    """spec with no helmChart/appStack/image returns missing_deployment_method error."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(MISSING_DEPLOYMENT_YAML)

    assert result["valid"] is False
    codes = [e["code"] for e in result["errors"]]
    assert "missing_deployment_method" in codes


def test_response_has_captured_at() -> None:
    """All responses include captured_at timestamp."""
    from tools.validate_yaml import _validate_yaml_impl

    # Test with valid YAML
    result_valid = _validate_yaml_impl(VALID_HELMCHART_YAML)
    assert "captured_at" in result_valid
    assert isinstance(result_valid["captured_at"], str)
    assert "T" in result_valid["captured_at"]

    # Test with invalid YAML
    result_invalid = _validate_yaml_impl(WRONG_KIND_YAML)
    assert "captured_at" in result_invalid
    assert isinstance(result_invalid["captured_at"], str)

    # Test with parse error YAML
    result_parse = _validate_yaml_impl(SYNTAX_ERROR_YAML)
    assert "captured_at" in result_parse
    assert isinstance(result_parse["captured_at"], str)
