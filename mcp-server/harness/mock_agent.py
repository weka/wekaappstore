"""Mock agent harness: scripted tool chain runner for AGNT-02.

Calls tool implementation functions directly with mocked K8s dependencies.
No MCP stdio protocol needed — tests the tool logic, not framing.
No network calls are made — all K8s dependencies are fully mocked.

Exports:
  - build_mock_k8s_deps(): Build mocked ApplyGatewayDependencies + ops_log
  - build_mock_inspection_deps(): Build pre-built inspection snapshot dicts
  - run_happy_path(): Full inspect -> list -> get -> validate -> apply -> status chain
  - run_approval_bypass(): Apply without confirmed=True — structured error, no CR
  - run_validation_failure(): validate_yaml rejects v1.0 YAML before apply

Runnable standalone:
    cd mcp-server && PYTHONPATH=.:../app-store-gui python -m harness.mock_agent
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

from webapp.planning.apply_gateway import ApplyGatewayDependencies

from tools.blueprints import scan_blueprints, flatten_blueprint_summary, flatten_blueprint_detail
from tools.inspect_cluster import flatten_inspect_cluster_for_mcp
from tools.inspect_weka import flatten_inspect_weka_for_mcp
from tools.validate_yaml import _validate_yaml_impl
from tools.apply_tool import _apply_impl
from tools.status_tool import _status_impl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]

# Reuse Phase 6 blueprint fixtures
SAMPLE_BLUEPRINTS_DIR = str(MCP_SERVER_ROOT / "tests" / "fixtures" / "sample_blueprints")

# Valid WekaAppStore YAML — passes validate_yaml, applies cleanly
SAMPLE_VALID_YAML = """
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: test-app
  namespace: default
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-app
    version: 1.0.0
    releaseName: test-app-release
""".strip()

# v1.0 YAML with blueprint_family — must be rejected by validate_yaml
SAMPLE_V1_YAML = """
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: test-app
spec:
  blueprint_family: ai-inference
  helmChart:
    repository: https://charts.example.com
    name: my-app
    version: 1.0.0
""".strip()


# ---------------------------------------------------------------------------
# Mock dependency builders
# ---------------------------------------------------------------------------


def build_mock_k8s_deps(
    apply_should_succeed: bool = True,
    cr_status: dict | None = None,
) -> tuple[ApplyGatewayDependencies, list]:
    """Build fully mocked ApplyGatewayDependencies for harness scenarios.

    Args:
        apply_should_succeed: If False, create_namespaced_custom_object raises ApiException(403).
        cr_status: Status dict to return from get_namespaced_custom_object. Defaults to
            a Ready status with releaseStatus='deployed'.

    Returns:
        Tuple of (ApplyGatewayDependencies, ops_log). ops_log is a list of
        ("create", kwargs) or ("ensure_ns", ns) tuples — use to assert no side effects.
    """
    from kubernetes.client.rest import ApiException

    ops_log: list = []

    effective_status = cr_status if cr_status is not None else {
        "releaseStatus": "deployed",
        "releaseName": "test-app-release",
        "releaseVersion": 1,
        "appStackPhase": "Ready",
        "conditions": [
            {
                "type": "Ready",
                "status": "True",
                "reason": "ReconcileSuccess",
                "message": "Deployment successful",
                "lastTransitionTime": "2026-03-20T06:00:00Z",
            }
        ],
        "componentStatus": [],
    }

    class _MockCustomObjectsApi:
        def create_namespaced_custom_object(self, **kwargs):
            ops_log.append(("create", kwargs))
            if not apply_should_succeed:
                raise ApiException(status=403, reason="Forbidden")

        def patch_namespaced_custom_object(self, **kwargs):
            ops_log.append(("patch", kwargs))

        def get_namespaced_custom_object(self, **kwargs):
            return {
                "metadata": {"name": kwargs.get("name", "test-app"), "namespace": kwargs.get("namespace", "default")},
                "spec": {},
                "status": effective_status,
            }

    mock_custom_objects_api = _MockCustomObjectsApi()

    deps = ApplyGatewayDependencies(
        load_kube_config=lambda: None,
        ensure_namespace_exists=lambda ns: ops_log.append(("ensure_ns", ns)),
        is_cluster_scoped=lambda doc: False,
        crd_scope_for=lambda group, plural: "Namespaced",
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=lambda: object(),
        custom_objects_api_factory=lambda api_client: mock_custom_objects_api,
    )

    return deps, ops_log


def build_mock_inspection_deps() -> dict:
    """Build pre-built inspection snapshot dicts for inspect_cluster and inspect_weka.

    Returns a dict with 'cluster_snapshot' and 'weka_snapshot' keys.
    These match the shape that flatten_inspect_cluster_for_mcp() and
    flatten_inspect_weka_for_mcp() expect — no raw K8s API mocking needed.
    """
    cluster_snapshot = {
        "captured_at": "2026-03-20T06:00:00Z",
        "k8s_version": "v1.30.1",
        "gpu_operator_installed": True,
        "app_store_crd_installed": True,
        "app_store_cluster_init_present": True,
        "app_store_crs": ["default/app-store-cluster-init"],
        "default_storage_class": "wekafs",
        "domains": {
            "cpu": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 2,
                    "cpu_nodes": 2,
                    "gpu_nodes": 0,
                    "allocatable_cores": 32.0,
                    "used_cores": 8.0,
                    "free_cores": 24.0,
                },
                "notes": [],
                "blockers": [],
            },
            "memory": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 2,
                    "allocatable_gib": 128.0,
                    "used_gib": 32.0,
                    "free_gib": 96.0,
                },
                "notes": [],
                "blockers": [],
            },
            "gpu": {
                "status": "complete",
                "required": False,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 0,
                    "gpu_nodes": 0,
                    "total_devices": 0,
                    "used_devices": 0,
                    "free_devices": 0,
                    "inventory": [],
                },
                "notes": [],
                "blockers": [],
            },
            "namespaces": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {"names": ["default", "weka"]},
                "notes": [],
                "blockers": [],
            },
            "storage_classes": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {"names": ["wekafs"]},
                "notes": [],
                "blockers": [],
            },
        },
    }

    weka_snapshot = {
        "captured_at": "2026-03-20T06:00:00Z",
        "domains": {
            "weka": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T06:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "clusters": [
                        {
                            "name": "weka-prod",
                            "namespace": "weka",
                            "status": "Ready",
                            "cluster_total_bytes": 1099511627776,
                            "cluster_free_bytes": 549755813888,
                            "filesystem_capacity": None,
                        }
                    ],
                    "filesystems": [
                        {
                            "name": "weka-home",
                            "total_bytes": 549755813888,
                            "free_bytes": 274877906944,
                        }
                    ],
                    "cluster_total_bytes": 1099511627776,
                    "cluster_free_bytes": 549755813888,
                },
                "notes": [],
                "blockers": [],
            }
        },
    }

    return {
        "cluster_snapshot": cluster_snapshot,
        "weka_snapshot": weka_snapshot,
    }


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def run_happy_path(
    mock_inspection_deps: dict,
    mock_apply_deps: ApplyGatewayDependencies,
    mock_apply_ops: list,
) -> dict:
    """Run the complete inspect -> list -> get -> validate -> apply -> status chain.

    All steps use mocked dependencies — no network calls.

    Args:
        mock_inspection_deps: Dict from build_mock_inspection_deps().
        mock_apply_deps: ApplyGatewayDependencies from build_mock_k8s_deps().
        mock_apply_ops: ops_log list from build_mock_k8s_deps().

    Returns:
        Dict with results from each step: inspect_cluster, inspect_weka,
        list_blueprints, get_blueprint, validate, apply, status.
    """
    # Step 1: inspect_cluster (use pre-built snapshot, flatten it)
    cluster_result = flatten_inspect_cluster_for_mcp(mock_inspection_deps["cluster_snapshot"])

    # Step 2: inspect_weka (use pre-built snapshot, flatten it)
    weka_result = flatten_inspect_weka_for_mcp(mock_inspection_deps["weka_snapshot"])

    # Step 3: list_blueprints (use fixture dir)
    entries = scan_blueprints(SAMPLE_BLUEPRINTS_DIR)
    blueprints_result = {
        "count": len(entries),
        "blueprints": [flatten_blueprint_summary(e) for e in entries],
        "warnings": [],
    }

    # Step 4: get_blueprint (first blueprint from list)
    assert entries, "Need at least one blueprint fixture for happy path"
    first_entry = entries[0]
    blueprint_result = flatten_blueprint_detail(first_entry)

    # Step 5: validate_yaml
    validate_result = _validate_yaml_impl(SAMPLE_VALID_YAML)
    assert validate_result["valid"] is True, (
        f"Happy path YAML must be valid, got: {validate_result['errors']}"
    )

    # Step 6: apply (confirmed=True, mocked deps)
    apply_result = _apply_impl(
        yaml_text=SAMPLE_VALID_YAML,
        namespace="default",
        confirmed=True,
        apply_gateway_deps=mock_apply_deps,
    )
    assert apply_result["applied"] is True, (
        f"Happy path apply must succeed, got: {apply_result}"
    )

    # Step 7: status (use the mock CustomObjectsApi from apply deps)
    mock_status_api = mock_apply_deps.custom_objects_api_factory(None)
    status_result = _status_impl(
        name="test-app",
        namespace="default",
        custom_objects_api=mock_status_api,
    )

    return {
        "inspect_cluster": cluster_result,
        "inspect_weka": weka_result,
        "list_blueprints": blueprints_result,
        "get_blueprint": blueprint_result,
        "validate": validate_result,
        "apply": apply_result,
        "status": status_result,
    }


def run_approval_bypass(
    mock_apply_deps: ApplyGatewayDependencies,
    mock_apply_ops: list,
) -> dict:
    """Verify apply without confirmed=True returns structured error, no CR created.

    Args:
        mock_apply_deps: ApplyGatewayDependencies from build_mock_k8s_deps().
        mock_apply_ops: ops_log list from build_mock_k8s_deps() — asserted empty after call.

    Returns:
        apply result dict with applied=false and error='approval_required'.
    """
    result = _apply_impl(
        yaml_text=SAMPLE_VALID_YAML,
        namespace="default",
        confirmed=False,
        apply_gateway_deps=mock_apply_deps,
    )

    assert result["applied"] is False, (
        f"Approval bypass: apply must return applied=false, got: {result}"
    )
    assert result["error"] == "approval_required", (
        f"Approval bypass: error must be 'approval_required', got: {result}"
    )
    assert not any(op[0] == "create" for op in mock_apply_ops), (
        f"Approval bypass: no CR should be created, ops_log: {mock_apply_ops}"
    )

    return result


def run_validation_failure() -> dict:
    """Verify validate_yaml rejects v1.0 YAML before apply is called.

    Returns:
        validate_yaml result dict with valid=false and v1_only_field errors.
        Apply is NOT called in this scenario.
    """
    result = _validate_yaml_impl(SAMPLE_V1_YAML)

    assert result["valid"] is False, (
        f"Validation failure: v1.0 YAML must be rejected, got valid=true"
    )
    error_codes = [e["code"] for e in result.get("errors", [])]
    assert "v1_only_field" in error_codes, (
        f"Validation failure: must have v1_only_field error, got: {result['errors']}"
    )

    # Apply is NOT called — return validation result only
    return result


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Mock Agent Harness ===\n")

    # Build shared mocked dependencies
    mock_apply_deps, mock_apply_ops = build_mock_k8s_deps()
    mock_inspection_deps = build_mock_inspection_deps()

    print("--- Scenario 1: Happy Path ---")
    try:
        happy_results = run_happy_path(mock_inspection_deps, mock_apply_deps, mock_apply_ops)
        print(json.dumps({
            "validate_valid": happy_results["validate"]["valid"],
            "apply_applied": happy_results["apply"]["applied"],
            "status_found": happy_results["status"]["found"],
            "status_phase": happy_results["status"]["app_stack_phase"],
        }, indent=2))
        print("PASSED\n")
    except Exception as exc:
        print(f"FAILED: {exc}\n")
        sys.exit(1)

    print("--- Scenario 2: Approval Bypass ---")
    try:
        mock_apply_deps2, mock_apply_ops2 = build_mock_k8s_deps()
        bypass_result = run_approval_bypass(mock_apply_deps2, mock_apply_ops2)
        print(json.dumps({
            "applied": bypass_result["applied"],
            "error": bypass_result["error"],
            "no_cr_created": not any(op[0] == "create" for op in mock_apply_ops2),
        }, indent=2))
        print("PASSED\n")
    except Exception as exc:
        print(f"FAILED: {exc}\n")
        sys.exit(1)

    print("--- Scenario 3: Validation Failure ---")
    try:
        failure_result = run_validation_failure()
        print(json.dumps({
            "valid": failure_result["valid"],
            "error_codes": [e["code"] for e in failure_result.get("errors", [])],
        }, indent=2))
        print("PASSED\n")
    except Exception as exc:
        print(f"FAILED: {exc}\n")
        sys.exit(1)

    print("=== All 3 scenarios passed ===")
