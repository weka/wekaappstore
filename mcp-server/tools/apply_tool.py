"""apply MCP tool.

Wraps apply_gateway.py to create WekaAppStore CRs with a hard approval gate.

The confirmation gate is enforced at the Python level (identity check on bool),
not just in SKILL.md instructions. An agent must pass confirmed=True (boolean)
explicitly — string "true" or any other truthy value is rejected.

Provides:
  - _apply_impl(): Testable implementation with injectable ApplyGatewayDependencies
  - register_apply(): Registers apply tool with FastMCP instance
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kubernetes.client.rest import ApiException

from webapp.planning.apply_gateway import (
    apply_yaml_content_with_namespace,
    ApplyGatewayDependencies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _apply_impl(
    yaml_text: str,
    namespace: str,
    confirmed: bool,
    apply_gateway_deps: ApplyGatewayDependencies | None = None,
) -> dict:
    """Apply WekaAppStore YAML to the cluster — injectable for testing.

    Args:
        yaml_text: YAML string to apply (should be validated first).
        namespace: Target namespace for namespaced resources.
        confirmed: Must be exactly True (boolean identity check) to proceed.
            Any other value (False, "true", 1, None) returns approval_required error.
        apply_gateway_deps: Injectable ApplyGatewayDependencies for testing.
            If None, production defaults are used (real K8s clients).
            NOTE: In tests, always inject mocked deps to avoid load_kube_config().

    Returns:
        Dict with: captured_at, applied (bool), applied_kinds (list), namespace,
        error (str|None), message (str|None), warnings (list).
    """
    # Hard approval gate — identity check, not truthiness
    if confirmed is not True:
        return {
            "captured_at": _utc_now(),
            "applied": False,
            "applied_kinds": [],
            "error": "approval_required",
            "message": (
                "apply requires confirmed=true. Call validate_yaml first, "
                "explain what will be created to the user, and only "
                "set confirmed=true after explicit user approval."
            ),
            "warnings": ["No resources were created — confirmation not provided"],
        }

    try:
        result = apply_yaml_content_with_namespace(
            yaml_text,
            namespace,
            dependencies=apply_gateway_deps,
        )
        return {
            "captured_at": _utc_now(),
            "applied": True,
            "applied_kinds": result.get("applied", []),
            "namespace": namespace,
            "error": None,
            "message": None,
            "warnings": [],
        }
    except ApiException as exc:
        return {
            "captured_at": _utc_now(),
            "applied": False,
            "applied_kinds": [],
            "namespace": namespace,
            "error": f"k8s_api_error_{exc.status}",
            "message": f"K8s API error: {exc.status} {exc.reason}",
            "warnings": [],
        }


# ---------------------------------------------------------------------------
# FastMCP registration
# ---------------------------------------------------------------------------


def register_apply(mcp: Any) -> None:
    """Register the apply tool with the given FastMCP instance."""

    @mcp.tool()
    def apply(yaml_text: str, namespace: str, confirmed: bool) -> dict:
        """Apply a WekaAppStore YAML manifest to the cluster.

        SAFETY RULES — all three must be satisfied before calling apply:
        1. validate_yaml must have returned valid=true for this YAML
        2. inspect_cluster must have been re-run AFTER validate_yaml passed to
           confirm cluster resources haven't changed since the initial inspection
        3. The user must have explicitly approved — show them the resource name,
           namespace, deployment method, and component list, then wait for approval

        confirmed must be exactly boolean true (not string "true" or integer 1).
        Any other value for confirmed returns error='approval_required' without
        creating any K8s resources.

        After a successful apply, call status to monitor deployment progress.

        Returns: captured_at, applied (bool), applied_kinds (list), namespace,
        error (str|None), message (str|None), warnings.

        Sequencing: validate_yaml -> inspect_cluster (re-run) ->
        (explicit user approval) -> apply (confirmed=True) -> status.
        """
        return _apply_impl(yaml_text=yaml_text, namespace=namespace, confirmed=confirmed)
