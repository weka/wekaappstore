"""Mock agent harness: scripted tool chain runner for AGNT-02.

Calls tool implementation functions directly with mocked K8s dependencies.
No MCP stdio protocol needed — tests the tool logic, not framing.
No network calls are made — all K8s dependencies are fully mocked.

Tool selection uses description-based keyword matching (select_tool), proving
that tool descriptions contain sufficient sequencing guidance for an agent to
choose the right tool without hardcoded routing.

Exports:
  - build_tool_registry(): Build {tool_name: {description, callable}} registry
  - select_tool(intent_keywords, registry): Match keywords to tool descriptions
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
# Description-based tool registry
# ---------------------------------------------------------------------------


class _RegistryCapture:
    """Minimal MCP stub that captures tool registrations for registry building.

    When register_*(mcp) calls @mcp.tool(), this stub captures the decorated
    function's name and docstring into self.captured.
    """

    def __init__(self) -> None:
        self.captured: dict[str, str] = {}

    def tool(self):
        """Return a decorator that records the function's name and docstring."""
        def decorator(fn):
            self.captured[fn.__name__] = (fn.__doc__ or "").strip()
            return fn
        return decorator


def build_tool_registry(inspection_deps: dict | None = None) -> dict[str, dict]:
    """Build a registry mapping tool names to their descriptions and callables.

    Descriptions are extracted by calling each register_* function with a
    lightweight MCP stub that captures @mcp.tool() registrations. This ensures
    the descriptions in the registry exactly match the docstrings in tools/*.py.

    The callable for each tool wraps the corresponding _impl() or flatten_*()
    function with any required injected dependencies.

    Args:
        inspection_deps: Dict from build_mock_inspection_deps(), required for
            inspect_cluster and inspect_weka callables. If None, those callables
            will raise NotImplementedError.

    Returns:
        Dict of {tool_name: {"description": str, "callable": callable}}
    """
    from tools.inspect_cluster import register_inspect_cluster
    from tools.inspect_weka import register_inspect_weka
    from tools.blueprints import register_blueprint_tools
    from tools.crd_schema import register_crd_schema
    from tools.validate_yaml import register_validate_yaml
    from tools.apply_tool import register_apply
    from tools.status_tool import register_status

    # Capture descriptions from all register_* functions
    cap = _RegistryCapture()
    register_inspect_cluster(cap)
    register_inspect_weka(cap)
    register_blueprint_tools(cap)
    register_crd_schema(cap)
    register_validate_yaml(cap)
    register_apply(cap)
    register_status(cap)

    descriptions = cap.captured  # {tool_name: description_str}

    # Build callables that use injected deps where needed
    def _callable_inspect_cluster() -> dict:
        if inspection_deps is None:
            raise NotImplementedError("inspection_deps required for inspect_cluster")
        return flatten_inspect_cluster_for_mcp(inspection_deps["cluster_snapshot"])

    def _callable_inspect_weka() -> dict:
        if inspection_deps is None:
            raise NotImplementedError("inspection_deps required for inspect_weka")
        return flatten_inspect_weka_for_mcp(inspection_deps["weka_snapshot"])

    def _callable_list_blueprints() -> dict:
        entries = scan_blueprints(SAMPLE_BLUEPRINTS_DIR)
        return {
            "count": len(entries),
            "blueprints": [flatten_blueprint_summary(e) for e in entries],
            "warnings": [],
        }

    def _callable_get_blueprint(name: str) -> dict:
        entries = scan_blueprints(SAMPLE_BLUEPRINTS_DIR)
        for entry in entries:
            if entry["manifest"].get("metadata", {}).get("name", "") == name:
                return flatten_blueprint_detail(entry)
        return {"error": "Blueprint not found", "requested_name": name, "available_names": []}

    def _callable_get_crd_schema() -> dict:
        # Return a minimal schema response without live K8s
        return {
            "captured_at": "2026-03-20T06:00:00Z",
            "group": "warp.io",
            "version": "v1alpha1",
            "kind": "WekaAppStore",
            "schema": {"type": "object", "properties": {}},
            "examples": [],
            "warnings": ["Schema read from mock — no live cluster"],
        }

    def _callable_validate_yaml(yaml_text: str) -> dict:
        return _validate_yaml_impl(yaml_text)

    def _callable_apply(yaml_text: str, namespace: str, confirmed: bool,
                        apply_gateway_deps: ApplyGatewayDependencies | None = None) -> dict:
        return _apply_impl(
            yaml_text=yaml_text,
            namespace=namespace,
            confirmed=confirmed,
            apply_gateway_deps=apply_gateway_deps,
        )

    def _callable_status(name: str, namespace: str = "default",
                         custom_objects_api=None) -> dict:
        return _status_impl(name=name, namespace=namespace, custom_objects_api=custom_objects_api)

    callables = {
        "inspect_cluster": _callable_inspect_cluster,
        "inspect_weka": _callable_inspect_weka,
        "list_blueprints": _callable_list_blueprints,
        "get_blueprint": _callable_get_blueprint,
        "get_crd_schema": _callable_get_crd_schema,
        "validate_yaml": _callable_validate_yaml,
        "apply": _callable_apply,
        "status": _callable_status,
    }

    registry = {}
    for name, description in descriptions.items():
        registry[name] = {
            "description": description,
            "callable": callables.get(name),
        }

    return registry


def select_tool(intent_keywords: list[str], registry: dict) -> str:
    """Select a tool name by matching intent keywords against tool descriptions.

    Performs simple case-insensitive substring matching: counts how many keywords
    appear in each tool's description, then returns the tool with the most matches.

    This is intentionally simple — it proves that tool descriptions contain the
    right vocabulary for tool selection, not that NLP-based routing works.

    Args:
        intent_keywords: List of lowercase keywords expressing the intent
            (e.g., ["cluster", "resources", "cpu"] for inspect_cluster).
        registry: Dict from build_tool_registry().

    Returns:
        Tool name string of the best-matching tool.

    Raises:
        ValueError: If registry is empty or no tool matches any keyword.
    """
    if not registry:
        raise ValueError("Tool registry is empty")

    scores: dict[str, int] = {}
    for tool_name, entry in registry.items():
        description_lower = entry["description"].lower()
        score = sum(1 for kw in intent_keywords if kw.lower() in description_lower)
        scores[tool_name] = score

    best_tool = max(scores, key=lambda t: scores[t])
    if scores[best_tool] == 0:
        raise ValueError(
            f"No tool matched any of the keywords: {intent_keywords}. "
            f"Available tools: {list(registry.keys())}"
        )

    return best_tool


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

    Uses description-based tool selection via select_tool() for each step.
    All steps use mocked dependencies — no network calls.

    Args:
        mock_inspection_deps: Dict from build_mock_inspection_deps().
        mock_apply_deps: ApplyGatewayDependencies from build_mock_k8s_deps().
        mock_apply_ops: ops_log list from build_mock_k8s_deps().

    Returns:
        Dict with results from each step: inspect_cluster, inspect_weka,
        list_blueprints, get_blueprint, validate, apply, status.
    """
    registry = build_tool_registry(inspection_deps=mock_inspection_deps)

    # Step 1: inspect_cluster — intent: understand cluster resources first
    tool_name = select_tool(["cluster", "resources", "first", "cpu", "memory"], registry)
    assert tool_name == "inspect_cluster", f"Expected inspect_cluster, got {tool_name}"
    cluster_result = registry[tool_name]["callable"]()

    # Step 2: inspect_weka — intent: check WEKA storage capacity
    tool_name = select_tool(["weka", "storage", "capacity", "filesystem"], registry)
    assert tool_name == "inspect_weka", f"Expected inspect_weka, got {tool_name}"
    weka_result = registry[tool_name]["callable"]()

    # Step 3: list_blueprints — intent: discover available blueprints catalog
    tool_name = select_tool(["list", "blueprints", "catalog", "available"], registry)
    assert tool_name == "list_blueprints", f"Expected list_blueprints, got {tool_name}"
    blueprints_result = registry[tool_name]["callable"]()

    # Step 4: get_blueprint — intent: get full blueprint specification detail
    assert blueprints_result["count"] > 0, "Need at least one blueprint fixture for happy path"
    first_bp_name = blueprints_result["blueprints"][0]["name"]
    tool_name = select_tool(["get_blueprint", "full", "specification", "components"], registry)
    assert tool_name == "get_blueprint", f"Expected get_blueprint, got {tool_name}"
    blueprint_result = registry[tool_name]["callable"](name=first_bp_name)

    # Step 5: validate_yaml — intent: validate YAML structure, errors list, apiVersion check
    tool_name = select_tool(["structurally valid", "apiversion", "errors", "v1.0-only"], registry)
    assert tool_name == "validate_yaml", f"Expected validate_yaml, got {tool_name}"
    validate_result = registry[tool_name]["callable"](yaml_text=SAMPLE_VALID_YAML)
    assert validate_result["valid"] is True, (
        f"Happy path YAML must be valid, got: {validate_result['errors']}"
    )

    # Step 6: apply — intent: apply manifest to cluster with confirmation
    tool_name = select_tool(["apply", "manifest", "confirmed", "deploy"], registry)
    assert tool_name == "apply", f"Expected apply, got {tool_name}"
    apply_result = registry[tool_name]["callable"](
        yaml_text=SAMPLE_VALID_YAML,
        namespace="default",
        confirmed=True,
        apply_gateway_deps=mock_apply_deps,
    )
    assert apply_result["applied"] is True, (
        f"Happy path apply must succeed, got: {apply_result}"
    )

    # Step 7: status — intent: monitor deployment status after apply
    tool_name = select_tool(["status", "deployment", "after apply", "monitor"], registry)
    assert tool_name == "status", f"Expected status, got {tool_name}"
    mock_status_api = mock_apply_deps.custom_objects_api_factory(None)
    status_result = registry[tool_name]["callable"](
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

    Uses description-based tool selection to find the apply tool.

    Args:
        mock_apply_deps: ApplyGatewayDependencies from build_mock_k8s_deps().
        mock_apply_ops: ops_log list from build_mock_k8s_deps() — asserted empty after call.

    Returns:
        apply result dict with applied=false and error='approval_required'.
    """
    registry = build_tool_registry()

    # Select apply tool by description keywords
    tool_name = select_tool(["apply", "manifest", "confirmed", "cluster"], registry)
    assert tool_name == "apply", f"Expected apply tool, got {tool_name}"

    result = registry[tool_name]["callable"](
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

    Uses description-based tool selection to find the validate_yaml tool.

    Returns:
        validate_yaml result dict with valid=false and v1_only_field errors.
        Apply is NOT called in this scenario.
    """
    registry = build_tool_registry()

    # Select validate_yaml tool by description keywords unique to that tool
    tool_name = select_tool(["structurally valid", "apiversion", "errors", "v1.0-only"], registry)
    assert tool_name == "validate_yaml", f"Expected validate_yaml tool, got {tool_name}"

    result = registry[tool_name]["callable"](yaml_text=SAMPLE_V1_YAML)

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
