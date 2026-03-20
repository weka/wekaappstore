"""Unit tests for apply MCP tool.

Tests call _apply_impl() directly — no FastMCP needed.

IMPORTANT: apply_gateway_deps must always be injected in tests to avoid
load_kube_config() being called during test execution, which would fail
in CI where no cluster is available.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --- sys.path setup ---
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
APP_STORE_GUI_ROOT = MCP_SERVER_ROOT.parent / "app-store-gui"
for path in (str(MCP_SERVER_ROOT), str(APP_STORE_GUI_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-chart
    version: "1.0.0"
"""


def _make_mock_deps(applied_kinds: list[str] | None = None) -> "ApplyGatewayDependencies":
    """Build a fully mocked ApplyGatewayDependencies with no real K8s calls."""
    from webapp.planning.apply_gateway import ApplyGatewayDependencies

    if applied_kinds is None:
        applied_kinds = ["WekaAppStore"]

    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.return_value = None

    return ApplyGatewayDependencies(
        load_kube_config=MagicMock(),
        ensure_namespace_exists=MagicMock(),
        is_cluster_scoped=MagicMock(return_value=False),
        crd_scope_for=MagicMock(return_value="Namespaced"),
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=MagicMock(return_value=object()),
        custom_objects_api_factory=MagicMock(return_value=mock_api),
        create_from_dict=MagicMock(return_value=[]),
        file_exists=MagicMock(return_value=True),
        logger=MagicMock(),
    )


def _make_mock_deps_k8s_error(status: int = 403, reason: str = "Forbidden") -> "ApplyGatewayDependencies":
    """Build mocked deps where create_namespaced_custom_object raises ApiException."""
    from webapp.planning.apply_gateway import ApplyGatewayDependencies
    from kubernetes.client.rest import ApiException

    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.side_effect = ApiException(
        status=status, reason=reason
    )

    return ApplyGatewayDependencies(
        load_kube_config=MagicMock(),
        ensure_namespace_exists=MagicMock(),
        is_cluster_scoped=MagicMock(return_value=False),
        crd_scope_for=MagicMock(return_value="Namespaced"),
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=MagicMock(return_value=object()),
        custom_objects_api_factory=MagicMock(return_value=mock_api),
        create_from_dict=MagicMock(return_value=[]),
        file_exists=MagicMock(return_value=True),
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_without_confirmation_returns_error() -> None:
    """confirmed=False returns applied=False, error=approval_required, no apply_gateway call."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps()

    result = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed=False,
        apply_gateway_deps=mock_deps,
    )

    assert result["applied"] is False
    assert result["error"] == "approval_required"
    assert result["applied_kinds"] == []
    # Confirm no K8s call was made
    mock_deps.load_kube_config.assert_not_called()


def test_apply_with_string_true_returns_error() -> None:
    """confirmed='true' (string) returns approval_required error (identity check, not truthiness)."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps()

    result = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed="true",  # type: ignore[arg-type]  # intentional type mismatch
        apply_gateway_deps=mock_deps,
    )

    assert result["applied"] is False
    assert result["error"] == "approval_required"
    mock_deps.load_kube_config.assert_not_called()


def test_apply_with_confirmation_succeeds() -> None:
    """confirmed=True with mocked deps returns applied=True and applied_kinds."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps()

    result = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed=True,
        apply_gateway_deps=mock_deps,
    )

    assert result["applied"] is True
    assert isinstance(result["applied_kinds"], list)
    assert result["namespace"] == "default"
    assert result["error"] is None


def test_apply_k8s_error_returns_structured() -> None:
    """ApiException from apply_gateway returns applied=False with k8s_api_error code."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps_k8s_error(status=403, reason="Forbidden")

    result = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed=True,
        apply_gateway_deps=mock_deps,
    )

    assert result["applied"] is False
    assert result["applied_kinds"] == []
    assert "k8s_api_error" in result["error"]
    assert "403" in result["error"]
    assert isinstance(result["message"], str)


def test_apply_response_has_captured_at() -> None:
    """All responses include captured_at timestamp."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps()

    # Without confirmation
    result_no_confirm = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed=False,
        apply_gateway_deps=mock_deps,
    )
    assert "captured_at" in result_no_confirm
    assert isinstance(result_no_confirm["captured_at"], str)
    assert "T" in result_no_confirm["captured_at"]

    # With confirmation (success)
    result_success = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="default",
        confirmed=True,
        apply_gateway_deps=mock_deps,
    )
    assert "captured_at" in result_success
    assert isinstance(result_success["captured_at"], str)


def test_apply_response_has_namespace() -> None:
    """Success response includes namespace field."""
    from tools.apply_tool import _apply_impl

    mock_deps = _make_mock_deps()

    result = _apply_impl(
        yaml_text=SAMPLE_YAML,
        namespace="weka-apps",
        confirmed=True,
        apply_gateway_deps=mock_deps,
    )

    assert result["namespace"] == "weka-apps"
