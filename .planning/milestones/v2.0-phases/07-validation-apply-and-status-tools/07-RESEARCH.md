# Phase 7: Validation, Apply, and Status Tools - Research

**Researched:** 2026-03-20
**Domain:** FastMCP tool wrappers over existing apply_gateway.py and validator.py; CRD-aware YAML validation; K8s status reads; mock agent harness
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCPS-07 | `validate_yaml` tool checks generated YAML against CRD and operator contract, returns structured errors | `validator.py` has `validate_structured_plan()` which validates v1.0 fields — Phase 7 needs a NEW validator targeting WekaAppStore YAML (CRD fields: `apiVersion`, `kind`, `metadata.name`, `spec`). Must reject v1.0-only fields `blueprint_family` and `fit_findings`. |
| MCPS-08 | `apply` tool creates `WekaAppStore` CRs with hard approval gate enforced in code | `apply_gateway.py` has `apply_yaml_content_with_namespace()` with injectable deps. Gate: require `confirmed: true` parameter at Python level — no SKILL.md-only constraint. |
| MCPS-09 | `status` tool returns deployment status of `WekaAppStore` resources | K8s `CustomObjectsApi.get_namespaced_custom_object()` or `list_namespaced_custom_object()` to read WekaAppStore CR status subresource. Pattern already in `apply_gateway.py` client setup. |
| AGNT-02 | Mock agent harness exercises full tool chain with scripted tool-use loops | Standalone Python script that calls MCP tool implementation functions directly (not via stdio), exercising inspect -> list -> get -> validate -> apply chain including approval bypass and validation failure paths. |
</phase_requirements>

---

## Summary

Phase 7 adds the final 3 MCP tools (`validate_yaml`, `apply`, `status`) and a mock agent harness. The critical constraint is that **`apply_gateway.py` and `validator.py` already exist and are injectable** — Phase 7 writes thin MCP tool wrappers around them, not new backend logic.

The `validator.py` in `app-store-gui/webapp/planning/validator.py` validates the v1.0 **StructuredPlan** model (`blueprint_family`, `fit_findings`, `components` etc.). The Phase 7 `validate_yaml` tool needs a **different validator** that checks WekaAppStore YAML documents against the CRD contract. The CRD schema is defined in `weka-app-store-operator-chart/templates/crd.yaml`. The v1.0 fields to explicitly reject are `blueprint_family` and `fit_findings` — these are StructuredPlan model fields that never appear in WekaAppStore CRD spec.

The `apply` tool wraps `apply_yaml_content_with_namespace()` from `apply_gateway.py`, which is fully injectable (all K8s clients passed as deps). The hard approval gate is a Python-level parameter check: if `confirmed` is not exactly `True`, return a structured error dict without calling `apply_gateway`. This is enforced in tool code, not SKILL.md.

The `status` tool reads WekaAppStore CR status via `CustomObjectsApi`. The CR's `.status` subresource contains `releaseStatus`, `releaseName`, `releaseVersion`, `conditions`, `componentStatus`, and `appStackPhase` fields per the CRD schema.

**Primary recommendation:** Use the same `_impl(injectable_deps) / register_*(mcp)` pattern from Phase 6. Create `mcp-server/tools/validate_yaml.py`, `mcp-server/tools/apply_tool.py`, `mcp-server/tools/status_tool.py`. The mock harness lives at `mcp-server/harness/mock_agent.py`.

---

## Standard Stack

### Core (all already in mcp-server/requirements.txt)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[cli]` | >=1.26.0 | FastMCP tool registration | Official SDK — already established in Phase 6 |
| `kubernetes` | >=27.0.0 | K8s API: CustomObjectsApi for apply and status | Already used |
| `PyYAML` | >=6.0.1 | Parse YAML string from agent for validation | Already used |

### No New Dependencies
Phase 7 adds zero new pip dependencies. All logic reuses:
- `mcp[cli]` (already installed)
- `kubernetes` (already installed)
- `PyYAML` (already installed)
- `apply_gateway.py` (already in `app-store-gui/webapp/planning/`)
- The new `validate_yaml` validator is custom Python — no external library needed

**Installation:** Nothing new — `requirements.txt` is unchanged.

---

## Architecture Patterns

### Updated Project Structure
```
mcp-server/
├── server.py                   # Add register_validate_yaml, register_apply, register_status
├── tools/
│   ├── inspect_cluster.py      # Phase 6 - complete
│   ├── inspect_weka.py         # Phase 6 - complete
│   ├── blueprints.py           # Phase 6 - complete
│   ├── crd_schema.py           # Phase 6 - complete
│   ├── validate_yaml.py        # Phase 7 - NEW
│   ├── apply_tool.py           # Phase 7 - NEW
│   └── status_tool.py          # Phase 7 - NEW
├── harness/
│   ├── __init__.py             # Phase 7 - NEW
│   └── mock_agent.py           # Phase 7 - NEW - scripted tool chain runner
├── tests/
│   ├── conftest.py             # Phase 6 - extend with apply/status fixtures
│   ├── test_validate_yaml.py   # Phase 7 - NEW
│   ├── test_apply_tool.py      # Phase 7 - NEW
│   ├── test_status_tool.py     # Phase 7 - NEW
│   ├── test_mock_agent.py      # Phase 7 - NEW
│   └── test_server.py          # Update: 5 -> 8 tools
├── config.py
└── requirements.txt
```

### Pattern 1: Injectable Implementation (established in Phase 6)

**What:** Every tool exposes a private `_impl()` function that accepts all K8s API clients as optional parameters. The `@mcp.tool()` decorated function calls `_impl()` with defaults. Tests call `_impl()` directly with mocks.

**When to use:** All three new tools follow this pattern exactly.

```python
# Source: Pattern from mcp-server/tools/crd_schema.py (Phase 6)
def _validate_yaml_impl(
    yaml_text: str,
    blueprints_dir: str | None = None,
) -> dict:
    """Testable implementation — no K8s needed for pure YAML validation."""
    ...

def register_validate_yaml(mcp: Any) -> None:
    @mcp.tool()
    def validate_yaml(yaml_text: str) -> dict:
        """Call this tool BEFORE apply to verify generated WekaAppStore YAML is valid.
        ...
        """
        return _validate_yaml_impl(yaml_text)
```

### Pattern 2: Hard Approval Gate (MCPS-08)

**What:** The `apply` tool MUST check `confirmed: true` as a Python-level function parameter before calling `apply_gateway`. This is NOT a SKILL.md constraint — it is enforced in code.

**When to use:** `apply` tool only.

**Critical implementation detail:** FastMCP extracts tool parameters from function signatures. A `confirmed: bool` parameter with no default will require the agent to pass it. The gate logic is:

```python
# Source: Pattern derived from MCPS-08 requirement + apply_gateway.py injectable pattern
def _apply_impl(
    yaml_text: str,
    namespace: str,
    confirmed: bool,
    apply_gateway_deps=None,
) -> dict:
    """Apply WekaAppStore YAML to cluster. confirmed must be True."""
    if confirmed is not True:
        return {
            "captured_at": _utc_now(),
            "applied": False,
            "error": "approval_required",
            "message": (
                "apply requires confirmed=true. Call validate_yaml first, "
                "explain what will be created to the user, and only set "
                "confirmed=true after explicit user approval."
            ),
            "warnings": ["No resources were created — confirmation not provided"],
        }
    # ... call apply_gateway ...
```

**Why `confirmed: bool` not `confirmed: str`:** FastMCP type hints map to JSON Schema types. `bool` maps to JSON boolean `true`/`false`. An agent that passes `"true"` (string) will be rejected by FastMCP validation before the function runs, forcing the agent to use the actual boolean `true`. This is correct behavior — it prevents accidental approvals from string coercion.

### Pattern 3: WekaAppStore YAML Validator (MCPS-07)

**What:** A new validator function that parses YAML text, checks structural requirements against the CRD contract, and explicitly rejects v1.0-only fields.

**Not reusing `validate_structured_plan()`:** The existing `validator.py` validates the v1.0 StructuredPlan model (blueprint_family, fit_findings, namespace_strategy, components, etc.) — none of these are WekaAppStore CRD spec fields. The two schemas are completely different. Phase 7 writes a new CRD-aware validator.

**WekaAppStore v2.0 valid spec fields** (from `crd.yaml`):
- `spec.image` (legacy pod deployment)
- `spec.binary` (legacy pod deployment)
- `spec.helmChart` (object with `repository`, `name`, `version`, `releaseName`, `crdsStrategy`)
- `spec.values` (free-form Helm values object)
- `spec.valuesFiles` (array of `{kind, name, key}`)
- `spec.targetNamespace`
- `spec.appStack.components[]` (each: `name`, `description`, `enabled`, `dependsOn`, `helmChart`, `kubernetesManifest`, `values`, `valuesFiles`, `targetNamespace`, `waitForReady`, `readinessCheck`)

**v1.0-only fields that must be rejected** (StructuredPlan model fields that never belong in a WekaAppStore YAML):
- `spec.blueprint_family` — StructuredPlan model field, not in CRD spec
- `spec.fit_findings` — StructuredPlan model field, not in CRD spec
- `spec.namespace_strategy` — StructuredPlan model field (CRD uses `spec.targetNamespace`)
- `spec.reasoning_summary` — StructuredPlan model field
- `spec.request_summary` — StructuredPlan model field
- `spec.unresolved_questions` — StructuredPlan model field

**Validation checks to implement:**

```python
# Source: CRD schema from weka-app-store-operator-chart/templates/crd.yaml
_V1_ONLY_SPEC_FIELDS = {
    "blueprint_family",
    "fit_findings",
    "namespace_strategy",
    "reasoning_summary",
    "request_summary",
    "unresolved_questions",
    # snake_case versions of camelCase CRD fields used in StructuredPlan
}

_REQUIRED_TOP_LEVEL = {"apiVersion", "kind", "metadata"}
_REQUIRED_METADATA = {"name"}
_VALID_API_VERSIONS = {"warp.io/v1alpha1"}
_VALID_KINDS = {"WekaAppStore"}

def _validate_yaml_impl(yaml_text: str) -> dict:
    errors = []
    warnings = []

    # 1. Parse YAML — catch syntax errors
    try:
        docs = list(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError as exc:
        return {
            "captured_at": _utc_now(),
            "valid": False,
            "errors": [{"code": "yaml_parse_error", "path": "$", "message": str(exc)}],
            "warnings": [],
        }

    # 2. Filter to WekaAppStore documents
    weka_docs = [d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"]
    if not weka_docs:
        errors.append({"code": "no_wekaappstore_doc", "path": "$",
                       "message": "No WekaAppStore document found in YAML"})

    for i, doc in enumerate(weka_docs):
        path_prefix = f"doc[{i}]"
        # 3. Check apiVersion
        if doc.get("apiVersion") not in _VALID_API_VERSIONS:
            errors.append({"code": "invalid_api_version", "path": f"{path_prefix}.apiVersion",
                           "message": f"apiVersion must be one of {sorted(_VALID_API_VERSIONS)}"})
        # 4. Check metadata.name
        if not (doc.get("metadata") or {}).get("name"):
            errors.append({"code": "missing_required_field", "path": f"{path_prefix}.metadata.name",
                           "message": "metadata.name is required"})
        # 5. Reject v1.0-only fields
        spec = doc.get("spec") or {}
        for v1_field in _V1_ONLY_SPEC_FIELDS:
            if v1_field in spec:
                errors.append({"code": "v1_only_field", "path": f"{path_prefix}.spec.{v1_field}",
                               "message": f"Field '{v1_field}' is a v1.0 planning model field and is not valid in WekaAppStore spec"})
        # 6. Must have spec.helmChart, spec.appStack, or spec.image (at least one deployment method)
        has_deployment = any(k in spec for k in ("helmChart", "appStack", "image"))
        if not has_deployment:
            errors.append({"code": "missing_deployment_method", "path": f"{path_prefix}.spec",
                           "message": "spec must include at least one of: helmChart, appStack, image"})

    return {
        "captured_at": _utc_now(),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
```

### Pattern 4: Status Tool (MCPS-09)

**What:** Read WekaAppStore CR status via `CustomObjectsApi`. Returns flattened status fields from `.status` subresource.

**CR status schema** (from `crd.yaml` `.status` section):
- `status.releaseStatus` — string (single-component)
- `status.releaseName` — string (single-component)
- `status.releaseVersion` — integer (single-component)
- `status.conditions[]` — array of `{type, status, reason, message, lastTransitionTime}`
- `status.componentStatus[]` — array of `{name, phase, releaseName, releaseVersion, message, lastTransitionTime}`
- `status.appStackPhase` — enum `Pending|Installing|Ready|Failed|Degraded`

```python
# Source: Pattern from apply_gateway.py CustomObjectsApi usage (already injectable)
def _status_impl(
    name: str,
    namespace: str = "default",
    custom_objects_api=None,
) -> dict:
    if custom_objects_api is None:
        from kubernetes import client, config as k8s_config
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
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
        return {
            "captured_at": _utc_now(),
            "name": name,
            "namespace": namespace,
            "found": True,
            "release_status": status.get("releaseStatus"),
            "release_name": status.get("releaseName"),
            "release_version": status.get("releaseVersion"),
            "app_stack_phase": status.get("appStackPhase"),
            "conditions": status.get("conditions", []),
            "component_status": status.get("componentStatus", []),
            "warnings": [],
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
```

**Depth note:** `conditions` and `component_status` are lists of dicts. Each dict inside is at depth 2 (status_response -> conditions[] -> dict). This is valid at 2-key depth. The `check_depth()` helper from Phase 6 will enforce this.

### Pattern 5: Mock Agent Harness (AGNT-02)

**What:** A standalone Python script at `mcp-server/harness/mock_agent.py` that calls tool implementation functions directly (not via stdio MCP protocol), simulating a scripted agent run. It exercises the complete `inspect -> list -> get -> validate_yaml -> apply` chain.

**Why not use actual MCP stdio for the harness:** The harness tests the tool logic, not the protocol framing. Calling `_impl()` functions directly is faster, easier to assert on, and requires no subprocess management.

**Required scenarios** (from success criteria):
1. **Happy path:** inspect_cluster -> list_blueprints -> get_blueprint -> validate_yaml (valid YAML) -> apply (confirmed=True) — full success
2. **Approval bypass:** apply called with confirmed=False — returns structured error, no CR created
3. **Validation failure:** validate_yaml called with YAML containing v1.0 fields (`blueprint_family`, `fit_findings`) — returns errors, apply never called

```python
# Source: mcp-server/harness/mock_agent.py pattern
"""Mock agent harness: scripted tool chain runner for AGNT-02.

Calls tool _impl() functions directly with mocked K8s dependencies.
No MCP stdio protocol needed — tests the tool logic, not framing.
"""
from unittest.mock import MagicMock
from tools.inspect_cluster import _inspect_cluster_impl
from tools.blueprints import _list_blueprints_impl, _get_blueprint_impl
from tools.validate_yaml import _validate_yaml_impl
from tools.apply_tool import _apply_impl
from tools.status_tool import _status_impl


def run_happy_path(mock_k8s_deps: dict) -> dict:
    """Run complete inspect -> validate -> apply chain with mocked backends."""
    ...

def run_approval_bypass(mock_k8s_deps: dict) -> dict:
    """Verify apply without confirmed=True returns error, no CR created."""
    ...

def run_validation_failure(mock_k8s_deps: dict) -> dict:
    """Verify validate_yaml rejects v1.0 fields before apply is called."""
    ...

if __name__ == "__main__":
    import json
    # Run all scenarios, print results
    ...
```

### Anti-Patterns to Avoid

- **Reusing `validate_structured_plan()` for WekaAppStore YAML validation:** The v1.0 plan model has nothing to do with the WekaAppStore CRD spec. Using it will cause spurious errors for valid CRD YAML and miss actual CRD violations.
- **SKILL.md-only apply gating:** The approval gate MUST be in Python code. SKILL.md instructions can be ignored by any agent. Code-level gating cannot.
- **`confirmed: str` instead of `confirmed: bool`:** String `"true"` != boolean `True` in Python. Use `bool` type hint to force agent to pass JSON `true`, not the string.
- **Raising exceptions from the apply tool:** When `apply_gateway` raises `ApiException`, catch it and return a structured error dict. Same pattern as Phase 6 tools.
- **Import-time K8s client initialization in apply/status tools:** The `apply_gateway.py` `_load_kube_config()` is called inside function bodies, not at import time. Keep this pattern — import-time calls fail in tests.
- **Status tool reading from `spec` instead of `status`:** The CR's `.spec` is the desired state. The `.status` subresource (returned by the same `get_namespaced_custom_object` call) is the actual deployment state.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| K8s CR apply with namespace handling | Custom kubectl wrapper | `apply_yaml_content_with_namespace()` from `apply_gateway.py` | Already handles WekaAppStore, namespace injection, create/patch 409 conflict, last-applied annotation |
| K8s client auth | Manual kubeconfig/token code | Existing injectable pattern in `apply_gateway.py` + `_load_kube_config()` | Handles both in-cluster and kubeconfig auth modes already |
| MCP protocol framing | Any custom JSON-RPC | `@mcp.tool()` decorator | FastMCP handles framing; already established |
| YAML parsing with multi-doc support | Custom parser | `yaml.safe_load_all()` | Already used in `apply_gateway.py` |
| v1.0 plan validation | Extending `validate_structured_plan()` | New `_validate_yaml_impl()` targeting CRD fields | Plans and CRD specs are different schemas |

**Key insight:** `apply_gateway.py` already has the tricky parts — WekaAppStore CR routing through `CustomObjectsApi`, 409 conflict handling (create-or-patch), namespace normalization, and last-applied annotation. The MCP tool is ~20 lines wrapping it.

---

## Common Pitfalls

### Pitfall 1: `confirmed` Parameter Type Coercion
**What goes wrong:** Agent passes `"true"` (string) or `1` (int) — Python `if confirmed is not True` catches string and int but `if not confirmed` does NOT catch `0` vs `False` distinction.
**Why it happens:** JSON serialization can produce strings if tool description says "pass 'true'".
**How to avoid:** Use `if confirmed is not True:` (identity check), not `if not confirmed:`. Type hint as `bool` — FastMCP will reject non-boolean inputs at the JSON Schema level before the function runs.
**Warning signs:** Apply succeeds when confirmed="true" (string) is passed.

### Pitfall 2: Validator Rejecting camelCase CRD Fields
**What goes wrong:** The validator checks for `blueprint_family` (snake_case) but the agent generates `helmChart` (camelCase per CRD spec). If the validator treats unknown field keys as errors, valid `helmChart` gets flagged.
**Why it happens:** CRD uses camelCase (`helmChart`, `appStack`, `targetNamespace`, `crdsStrategy`). The v1.0 StructuredPlan model used snake_case.
**How to avoid:** The validator should only explicitly reject the known-bad v1.0 snake_case fields (`blueprint_family`, `fit_findings`, etc.). Valid CRD fields in camelCase pass through. The CRD schema itself handles structural validation at apply time.
**Warning signs:** `validate_yaml` returns errors for `helmChart` or `appStack` keys.

### Pitfall 3: Status Tool Returns Empty When CR Has No Status Yet
**What goes wrong:** A freshly created CR has `status: {}` or no status subresource at all. Tool returns all nulls with no warning — agent incorrectly concludes deployment failed.
**Why it happens:** K8s operator hasn't reconciled yet (new CR). Status subresource is operator-populated.
**How to avoid:** When `status == {}` or `appStackPhase` is None, add a warning: `"Status not yet available — operator may still be reconciling. Try again in 10-30 seconds."`. Do not return an error for this case.
**Warning signs:** Status tool reports failure for a CR that was just applied.

### Pitfall 4: Mock Harness Coupling to Real K8s Fixtures
**What goes wrong:** Harness uses real kubeconfig or live cluster — harness fails in CI where no cluster is available.
**Why it happens:** Dependencies not fully mocked.
**How to avoid:** Harness `mock_k8s_deps` dict contains all MagicMock API clients. No test in the harness or `test_mock_agent.py` should call `config.load_kube_config()` or touch the network.
**Warning signs:** Harness tests fail with `FileNotFoundError: ~/.kube/config` or connection errors.

### Pitfall 5: `check_depth()` Failing on `conditions[]` or `component_status[]`
**What goes wrong:** Status tool response contains `conditions[0].lastTransitionTime` — that's 3 traversals: `conditions` -> `[0]` -> `lastTransitionTime`. Check_depth may fail if it counts list items as a traversal level.
**Why it happens:** The check_depth helper counts `list[dict]` as adding 1 depth. So `conditions[0].type` = depth 2 from root (conditions=1, dict inside list=2, then .type=primitive at depth 2). This is within the contract.
**How to avoid:** Verify check_depth count: `status_response.conditions` = depth 0, `status_response.conditions[0]` = depth 1 (entering list of dicts adds 1), `status_response.conditions[0].type` = depth 2 (entering dict adds 1). That is exactly 2 — within the max.
**Warning signs:** `test_response_depth_status` fails. Fix: re-read check_depth counting rules, don't flatten conditions further.

### Pitfall 6: Server.py Tool Count Test Fails After Adding New Tools
**What goes wrong:** `test_server_lists_5_tools` (Phase 6) asserts exactly 5 tools. Phase 7 adds 3 more.
**Why it happens:** Hardcoded expected count.
**How to avoid:** Update `test_server_lists_5_tools` to `test_server_lists_8_tools` and add the 3 new tool names to the expected set. This is a mandatory update in Phase 7.
**Warning signs:** Existing test fails with "Expected 5 tools, got 8".

---

## Code Examples

### validate_yaml Tool Registration

```python
# Source: Pattern from mcp-server/tools/crd_schema.py (Phase 6)
def register_validate_yaml(mcp: Any) -> None:
    @mcp.tool()
    def validate_yaml(yaml_text: str) -> dict:
        """Call this tool BEFORE apply to verify that generated WekaAppStore YAML
        is structurally valid per the CRD contract.

        Checks: correct apiVersion (warp.io/v1alpha1), kind (WekaAppStore),
        metadata.name present, no v1.0-only planning fields (blueprint_family,
        fit_findings), and at least one deployment method (helmChart/appStack/image).

        Returns valid=true with empty errors on success, or valid=false with
        a structured errors list on failure.

        Sequencing: get_crd_schema -> (generate YAML) -> validate_yaml -> apply.
        """
        return _validate_yaml_impl(yaml_text)
```

### apply Tool Registration

```python
# Source: Pattern combining apply_gateway.py and Phase 6 injectable pattern
def register_apply(mcp: Any) -> None:
    @mcp.tool()
    def apply(yaml_text: str, namespace: str, confirmed: bool) -> dict:
        """Apply a WekaAppStore YAML manifest to the cluster.

        IMPORTANT: confirmed must be true. You must call validate_yaml first,
        show the user what will be created (name, namespace, components), and
        only set confirmed=true after the user explicitly approves.

        Setting confirmed=false returns a structured error — no resources are created.

        Sequencing: validate_yaml -> (user approval) -> apply (confirmed=true).
        After apply, call status to monitor deployment progress.
        """
        return _apply_impl(yaml_text=yaml_text, namespace=namespace, confirmed=confirmed)
```

### apply_gateway Integration in apply Tool

```python
# Source: apply_gateway.py apply_yaml_content_with_namespace() + ApplyGatewayDependencies
from webapp.planning.apply_gateway import (
    apply_yaml_content_with_namespace,
    ApplyGatewayDependencies,
)

def _apply_impl(
    yaml_text: str,
    namespace: str,
    confirmed: bool,
    apply_gateway_deps: ApplyGatewayDependencies | None = None,
) -> dict:
    if confirmed is not True:
        return {
            "captured_at": _utc_now(),
            "applied": False,
            "error": "approval_required",
            "message": (
                "apply requires confirmed=true. Call validate_yaml first, "
                "explain what will be created to the user, and only "
                "set confirmed=true after explicit user approval."
            ),
            "applied_kinds": [],
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
            "warnings": [],
            "error": None,
            "message": None,
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
```

### Mock Agent Harness Scenario

```python
# Source: AGNT-02 requirement + Phase 6 injectable test patterns
def build_mock_k8s_deps(
    apply_should_succeed: bool = True,
    cr_status: dict | None = None,
) -> ApplyGatewayDependencies:
    """Build fully mocked ApplyGatewayDependencies for harness scenarios."""
    ops = []

    class MockCustomObjectsApi:
        def create_namespaced_custom_object(self, **kwargs):
            ops.append(("create", kwargs))
            if not apply_should_succeed:
                from kubernetes.client.rest import ApiException
                raise ApiException(status=403, reason="Forbidden")

        def get_namespaced_custom_object(self, **kwargs):
            return {"metadata": kwargs, "status": cr_status or {}}

    return ApplyGatewayDependencies(
        load_kube_config=lambda: None,
        ensure_namespace_exists=lambda ns: ops.append(("ensure_ns", ns)),
        is_cluster_scoped=lambda doc: False,
        crd_scope_for=lambda group, plural: "Namespaced",
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=lambda: object(),
        custom_objects_api_factory=lambda api_client: MockCustomObjectsApi(),
        create_from_dict=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("not expected")),
    ), ops
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.0: `validate_structured_plan()` checks StructuredPlan model | Phase 7: New `_validate_yaml_impl()` checks WekaAppStore CRD YAML directly | Phase 7 (this phase) | Validator matches what agents actually generate |
| v1.0: Approval gate in SKILL.md only (instructions) | Phase 7: `confirmed: bool` parameter in Python tool code | Phase 7 (this phase) | Gate cannot be bypassed by an agent that ignores SKILL.md |
| Phase 6: 5 tools, no write tools | Phase 7: 8 tools including validate_yaml, apply (write-gated), status | Phase 7 (this phase) | Complete inspect-validate-apply chain |

**Deprecated/outdated for Phase 7:**
- `validate_structured_plan()` and the v1.0 plan model: preserved for CLEAN-01 in Phase 8, but NOT used by any Phase 7 tool

---

## Open Questions

1. **apply_gateway.py `_load_kube_config()` called at function entry, not import time?**
   - What we know: `apply_yaml_documents_with_namespace()` calls `deps.load_kube_config()` immediately. The `ApplyGatewayDependencies` default for `load_kube_config` is `_load_kube_config` which calls `config.load_kube_config()`. In tests we override it with `lambda: None`.
   - What's unclear: Whether the MCP tool `_apply_impl()` calling `apply_yaml_content_with_namespace()` with `dependencies=None` (real default) would trigger `config.load_kube_config()` at tool call time in CI. **Answer: YES** — the tool will call `load_kube_config()` at apply time if no injectable deps provided. This is the expected production behavior. In tests, always pass mocked deps.
   - Recommendation: Document in tool test file that `apply_gateway_deps` must always be injected in tests.

2. **Does `CustomObjectsApi.get_namespaced_custom_object()` include the `.status` subresource?**
   - What we know: Standard K8s client `get_namespaced_custom_object()` returns the full CR object including `.status` if the operator has written to it. The CRD has `subresources: status: {}` which makes status a separate subresource endpoint, BUT the main GET endpoint still returns the full object including status.
   - Recommendation: Use `get_namespaced_custom_object()` directly. No need for a separate `/status` subresource endpoint call.

3. **Harness location: `mcp-server/harness/` vs `mcp-server/tests/`?**
   - Recommendation: `mcp-server/harness/mock_agent.py` as a standalone runnable script (has `if __name__ == "__main__":` block), with `mcp-server/tests/test_mock_agent.py` importing and calling it. This keeps the harness both runnable standalone and testable via pytest.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | None — tests run from `mcp-server/` with PYTHONPATH |
| Quick run command | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q` |
| Full suite command | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v` |

**Baseline:** Phase 6 ends with 41 passing tests. Phase 7 adds approximately 25-30 new tests.

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCPS-07 | `validate_yaml` accepts valid WekaAppStore YAML | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_valid_yaml_passes -x` | Wave 0 |
| MCPS-07 | `validate_yaml` rejects YAML with `blueprint_family` | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_rejects_v1_blueprint_family -x` | Wave 0 |
| MCPS-07 | `validate_yaml` rejects YAML with `fit_findings` | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_rejects_v1_fit_findings -x` | Wave 0 |
| MCPS-07 | `validate_yaml` rejects YAML with bad apiVersion | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_rejects_invalid_api_version -x` | Wave 0 |
| MCPS-07 | `validate_yaml` rejects YAML missing metadata.name | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_rejects_missing_name -x` | Wave 0 |
| MCPS-07 | `validate_yaml` rejects unparseable YAML syntax | unit | `pytest mcp-server/tests/test_validate_yaml.py::test_rejects_yaml_syntax_error -x` | Wave 0 |
| MCPS-08 | `apply` with `confirmed=False` returns structured error, no CR created | unit | `pytest mcp-server/tests/test_apply_tool.py::test_apply_without_confirmation_returns_error -x` | Wave 0 |
| MCPS-08 | `apply` with `confirmed=True` calls apply_gateway and returns applied_kinds | unit | `pytest mcp-server/tests/test_apply_tool.py::test_apply_with_confirmation_succeeds -x` | Wave 0 |
| MCPS-08 | `apply` with K8s error returns structured error dict (no exception raised) | unit | `pytest mcp-server/tests/test_apply_tool.py::test_apply_k8s_error_returns_structured -x` | Wave 0 |
| MCPS-09 | `status` returns current deployment state for named CR | unit | `pytest mcp-server/tests/test_status_tool.py::test_status_returns_cr_state -x` | Wave 0 |
| MCPS-09 | `status` returns found=False with warning for missing CR | unit | `pytest mcp-server/tests/test_status_tool.py::test_status_not_found -x` | Wave 0 |
| MCPS-09 | `status` returns warning when status subresource is empty | unit | `pytest mcp-server/tests/test_status_tool.py::test_status_empty_warns -x` | Wave 0 |
| AGNT-02 | Harness runs happy path inspect-validate-apply loop without errors | integration | `pytest mcp-server/tests/test_mock_agent.py::test_harness_happy_path -x` | Wave 0 |
| AGNT-02 | Harness approval bypass path returns structured error, asserts no CR created | integration | `pytest mcp-server/tests/test_mock_agent.py::test_harness_approval_bypass -x` | Wave 0 |
| AGNT-02 | Harness validation failure path: v1 YAML rejected before apply called | integration | `pytest mcp-server/tests/test_mock_agent.py::test_harness_validation_failure -x` | Wave 0 |

### Depth Contract Extension
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCPS-07 | `validate_yaml` response within 2-key depth | unit | Update `test_response_depth.py` | Exists (extend) |
| MCPS-08 | `apply` response within 2-key depth | unit | Update `test_response_depth.py` | Exists (extend) |
| MCPS-09 | `status` response within 2-key depth (conditions[] items at depth 2) | unit | Update `test_response_depth.py` | Exists (extend) |

### Server Tool Count Update
| File | Change |
|------|--------|
| `mcp-server/tests/test_server.py` | Update `test_server_lists_5_tools` to `test_server_lists_8_tools`, add `validate_yaml`, `apply`, `status` to expected set |

### Sampling Rate
- **Per task commit:** `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q`
- **Per wave merge:** `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v`
- **Phase gate:** All 41 existing tests plus all new Phase 7 tests green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `mcp-server/tools/validate_yaml.py` — validator tool (tests will fail RED until this exists)
- [ ] `mcp-server/tools/apply_tool.py` — apply tool wrapper (tests will fail RED until this exists)
- [ ] `mcp-server/tools/status_tool.py` — status tool (tests will fail RED until this exists)
- [ ] `mcp-server/harness/__init__.py` — harness package
- [ ] `mcp-server/harness/mock_agent.py` — scripted harness runner
- [ ] `mcp-server/tests/test_validate_yaml.py` — all validate_yaml tests
- [ ] `mcp-server/tests/test_apply_tool.py` — all apply tests
- [ ] `mcp-server/tests/test_status_tool.py` — all status tests
- [ ] `mcp-server/tests/test_mock_agent.py` — harness integration tests

*(All missing — no existing test infrastructure covers Phase 7 requirements)*

---

## Sources

### Primary (HIGH confidence)
- Project source: `mcp-server/tools/crd_schema.py` — `_impl(injectable)` + `register_*(mcp)` pattern confirmed
- Project source: `app-store-gui/webapp/planning/apply_gateway.py` — `apply_yaml_content_with_namespace()`, `ApplyGatewayDependencies` injectable pattern confirmed
- Project source: `app-store-gui/webapp/planning/validator.py` — `validate_structured_plan()` NOT used for Phase 7; confirmed wrong model for WekaAppStore YAML
- Project source: `weka-app-store-operator-chart/templates/crd.yaml` — full WekaAppStore spec fields and status schema confirmed
- Project source: `mcp-server/tests/test_response_depth.py` — `check_depth()` helper confirmed, schema exemption pattern confirmed
- Project source: `mcp-server/tests/conftest.py` — fixture patterns for mocked K8s APIs confirmed
- Project source: `mcp-server/server.py` — `register_*(mcp)` wiring pattern confirmed
- Phase 6 SUMMARY files — 41 tests passing at Phase 6 end, injectable patterns all established

### Secondary (MEDIUM confidence)
- Official FastMCP docs (modelcontextprotocol.io) — `@mcp.tool()` parameter type hints map to JSON Schema boolean for `confirmed: bool` — consistent with Phase 6 research
- Kubernetes Python client docs — `CustomObjectsApi.get_namespaced_custom_object()` returns full CR including `.status` on GET

### Tertiary (LOW confidence)
- None for critical claims — all key patterns verified from project source code

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; all patterns established in Phase 6 and confirmed in source
- Architecture (validate_yaml): HIGH — CRD schema fields read directly from crd.yaml; v1.0 fields identified from validator.py
- Architecture (apply): HIGH — apply_gateway.py fully read and injectable pattern confirmed
- Architecture (status): HIGH — K8s CustomObjectsApi pattern in apply_gateway.py; status fields from crd.yaml
- Architecture (harness): HIGH — injectable pattern confirmed; harness pattern is standard Python unittest.mock
- Pitfalls: HIGH — confirmed from direct code reading (type coercion, camelCase, depth contract)

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable codebase; mcp SDK evolves; re-verify if >30 days pass)
