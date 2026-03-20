"""status MCP tool.

Reads the current deployment state of a named WekaAppStore CR from the cluster.

Provides:
  - _status_impl(): Testable implementation with injectable K8s API client
  - register_status(): Registers status tool with FastMCP instance
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _status_impl(
    name: str,
    namespace: str = "default",
    custom_objects_api: Any = None,
) -> dict:
    """Read WekaAppStore CR deployment state — injectable for testing.

    Args:
        name: Name of the WekaAppStore CR to inspect.
        namespace: Namespace where the CR lives. Defaults to "default".
        custom_objects_api: kubernetes.client.CustomObjectsApi instance.
            If None, creates a real one (load_incluster_config with fallback
            to load_kube_config).

    Returns:
        Dict with: captured_at, name, namespace, found (bool), release_status,
        release_name, release_version, app_stack_phase, conditions (list),
        component_status (list), warnings (list).
    """
    if custom_objects_api is None:
        from kubernetes import client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception:
                pass
        custom_objects_api = client.CustomObjectsApi()

    try:
        cr = custom_objects_api.get_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace=namespace,
            plural="wekaappstores",
            name=name,
        )
        status = cr.get("status") or {}
        warnings: list[str] = []

        app_stack_phase = status.get("appStackPhase")
        if not status or app_stack_phase is None:
            warnings.append(
                "Status not yet available -- operator may still be reconciling."
                " Try again in 10-30 seconds."
            )

        return {
            "captured_at": _utc_now(),
            "name": name,
            "namespace": namespace,
            "found": True,
            "release_status": status.get("releaseStatus"),
            "release_name": status.get("releaseName"),
            "release_version": status.get("releaseVersion"),
            "app_stack_phase": app_stack_phase,
            "conditions": status.get("conditions", []),
            "component_status": status.get("componentStatus", []),
            "warnings": warnings,
        }

    except ApiException as exc:
        if exc.status == 404:
            return {
                "captured_at": _utc_now(),
                "name": name,
                "namespace": namespace,
                "found": False,
                "release_status": None,
                "release_name": None,
                "release_version": None,
                "app_stack_phase": None,
                "conditions": [],
                "component_status": [],
                "warnings": [f"WekaAppStore '{name}' not found in namespace '{namespace}'"],
            }
        raise


# ---------------------------------------------------------------------------
# FastMCP registration
# ---------------------------------------------------------------------------


def register_status(mcp: Any) -> None:
    """Register the status tool with the given FastMCP instance."""

    @mcp.tool()
    def status(name: str, namespace: str = "default") -> dict:
        """Call this tool after apply to monitor deployment progress of a
        WekaAppStore CR. Pass the resource name and namespace from the applied
        manifest.

        Returns releaseStatus, releaseName, releaseVersion, appStackPhase,
        conditions, and componentStatus from the CR's .status subresource.

        When the CR was just created by apply, the operator may not have
        reconciled yet — appStackPhase will be null and a warning is included.
        Call again in 10-30 seconds and repeat until appStackPhase is 'Ready'
        or 'Failed'. Report the final status to the user.

        Returns found=false with a warning if the named CR does not exist in
        the specified namespace.

        Returns: captured_at, name, namespace, found (bool), release_status,
        release_name, app_stack_phase, conditions (list), component_status
        (list), warnings.

        Sequencing: apply -> status (repeat until appStackPhase is Ready or Failed).
        """
        return _status_impl(name=name, namespace=namespace)
