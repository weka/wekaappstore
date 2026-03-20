"""Tests for the status MCP tool."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kubernetes.client.rest import ApiException


# ---------------------------------------------------------------------------
# Test: status returns CR state
# ---------------------------------------------------------------------------


def test_status_returns_cr_state(mock_custom_objects_api):
    """Mocked CR with status fields returns found=true with all status fields."""
    from tools.status_tool import _status_impl

    mock_custom_objects_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "my-app", "namespace": "default"},
        "spec": {},
        "status": {
            "releaseStatus": "deployed",
            "releaseName": "my-app-release",
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

    result = _status_impl(
        name="my-app",
        namespace="default",
        custom_objects_api=mock_custom_objects_api,
    )

    assert result["found"] is True
    assert result["release_status"] == "deployed"
    assert result["release_name"] == "my-app-release"
    assert result["release_version"] == 1
    assert result["app_stack_phase"] == "Ready"
    assert len(result["conditions"]) == 1
    assert result["conditions"][0]["type"] == "Ready"
    assert len(result["component_status"]) == 1
    assert result["component_status"][0]["name"] == "nginx"
    assert result["warnings"] == []

    mock_custom_objects_api.get_namespaced_custom_object.assert_called_once_with(
        group="warp.io",
        version="v1alpha1",
        namespace="default",
        plural="wekaappstores",
        name="my-app",
    )


# ---------------------------------------------------------------------------
# Test: status 404 returns found=false with warning
# ---------------------------------------------------------------------------


def test_status_not_found(mock_custom_objects_api):
    """ApiException(404) returns found=false with warning message."""
    from tools.status_tool import _status_impl

    mock_custom_objects_api.get_namespaced_custom_object.side_effect = ApiException(
        status=404, reason="Not Found"
    )

    result = _status_impl(
        name="missing-app",
        namespace="production",
        custom_objects_api=mock_custom_objects_api,
    )

    assert result["found"] is False
    assert result["name"] == "missing-app"
    assert result["namespace"] == "production"
    assert result["release_status"] is None
    assert result["app_stack_phase"] is None
    assert result["conditions"] == []
    assert result["component_status"] == []
    assert len(result["warnings"]) == 1
    assert "missing-app" in result["warnings"][0]
    assert "production" in result["warnings"][0]


# ---------------------------------------------------------------------------
# Test: status empty warns about operator reconciliation
# ---------------------------------------------------------------------------


def test_status_empty_warns(mock_custom_objects_api):
    """CR exists but status={} returns found=true with warning about operator reconciliation."""
    from tools.status_tool import _status_impl

    mock_custom_objects_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "new-app", "namespace": "default"},
        "spec": {},
        "status": {},
    }

    result = _status_impl(
        name="new-app",
        namespace="default",
        custom_objects_api=mock_custom_objects_api,
    )

    assert result["found"] is True
    assert result["app_stack_phase"] is None
    assert len(result["warnings"]) >= 1
    reconcile_warning = any("reconcil" in w.lower() for w in result["warnings"])
    assert reconcile_warning, f"Expected reconciliation warning, got: {result['warnings']}"


# ---------------------------------------------------------------------------
# Test: status response includes captured_at
# ---------------------------------------------------------------------------


def test_status_response_has_captured_at(mock_custom_objects_api):
    """Response always includes captured_at timestamp."""
    from tools.status_tool import _status_impl

    mock_custom_objects_api.get_namespaced_custom_object.return_value = {
        "metadata": {"name": "my-app", "namespace": "default"},
        "spec": {},
        "status": {"appStackPhase": "Ready"},
    }

    result = _status_impl(
        name="my-app",
        namespace="default",
        custom_objects_api=mock_custom_objects_api,
    )

    assert "captured_at" in result
    assert result["captured_at"].endswith("Z")
