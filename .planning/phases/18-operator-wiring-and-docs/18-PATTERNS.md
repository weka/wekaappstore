# Phase 18: Operator Wiring and Docs - Pattern Map

**Mapped:** 2026-05-08
**Files analyzed:** 7 (3 modified + 4 created/new directory)
**Analogs found:** 7 / 7 (1 with no direct in-repo analog — snapshots dir)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `operator_module/main.py` (`_render_or_raise` helper, `load_values_from_reference` rewrite, `handle_appstack_deployment` wiring, `field='spec'` decorator) | controller (kopf reconciler) | event-driven + request-response | Self-analogs in `main.py`: `render()` (line 253) for top-level helper placement; `handle_appstack_deployment` line 597, 620 for `kopf.PermanentError` raise idiom | exact (self-analog) |
| `weka-app-store-operator-chart/Chart.yaml` | config (Helm chart metadata) | static | Phase 17 commit `81d86ed` (0.1.61 → 0.1.62) | exact |
| `README.md` | docs | static | Existing `## Readiness checks for components` section (line 108) for top-level heading depth and prose tone | role-match |
| `operator_module/tests/test_appstack.py` (NEW) | test (unit, parametrized, mock-based) | request-response | `mcp-server/tests/test_apply_tool.py` (mocking pattern, `MagicMock`, `from unittest.mock import patch`); `operator_module/tests/test_render.py` (sys.path injection, naming, Phase 16 pattern); `mcp-server/tests/test_tool_descriptions.py:53` (parametrize idiom) | exact |
| `operator_module/tests/test_helm_non_wiring.py` (NEW) | test (mock + static introspection) | request-response | `operator_module/tests/test_render.py` for layout; `mcp-server/tests/test_apply_tool.py` for `unittest.mock.patch` style. `inspect.getsource` static-check has no exact analog — first of its kind in repo. | role-match |
| `operator_module/tests/test_backward_compat_snapshot.py` (NEW) | test (snapshot, file-IO) | file-IO | `mcp-server/tests/test_openclaw_config.py` lines 47, 65, 167 (`Path.read_text()` + plain `==` snapshot pattern, `@pytest.fixture(scope="module")` for fixture loading) | exact |
| `operator_module/tests/snapshots/ai-research/` (NEW directory) | test fixtures (baselines) | file-IO | No existing baseline directory in repo. Closest convention: `mcp-server/tests/fixtures/sample_blueprints/` for structured test data, but format differs (snapshots are deterministic outputs, not inputs). | NONE (first snapshots dir) |

## Pattern Assignments

### `operator_module/main.py` — `_render_or_raise` helper (NEW top-level function)

**Analog 1: top-level helper placement** — sibling to existing `render()` helper.
**Source:** `operator_module/main.py:253-287`

Existing top-level helper signature/docstring style (lines 232-287):
```python
def merge_values(base_values: Dict[str, Any], additional_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple values dictionaries, with later values taking precedence
    """
    result = base_values.copy()
    for values in additional_values:
        result = _deep_merge(result, values)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    ...


def render(text: str, variables: Optional[Dict[str, str]]) -> str:
    """Single-pass ${VAR} substitution via stdlib string.Template.

    Behavior contract (Phase 16, locked by CONTEXT.md D-01..D-06):
      ...
    """
    if not variables:
        return text
    ...
    try:
        return string.Template(text).substitute(variables)
    except KeyError as e:
        name = e.args[0] if e.args else ''
        raise ValueError(f"Undefined variable: ${{{name}}}") from e
    except ValueError as e:
        raise ValueError(f"Malformed placeholder in template: {e}") from e
```

**What to copy verbatim:**
- Top-level placement (NOT nested inside any class).
- `Optional[Dict[str, str]]` typing (consistent with `render()`'s param).
- Multi-line docstring with explicit "Behavior contract" header.
- `try ... except (KeyError, ValueError) as e: raise ... from e` chained-exception idiom.
- Place `_render_or_raise` IMMEDIATELY after `render()` (line 288 area) so the wrapper sits next to its dependency.

**What to adapt per CONTEXT.md D-15:**
- New signature uses keyword-only `*, source_desc: str`.
- Wraps `render()` and re-raises `kopf.PermanentError`, not `ValueError`.

---

### `operator_module/main.py` — `load_values_from_reference` rewrite (line 390 → expanded)

**Analog 1: existing function body to replace.**
**Source:** `operator_module/main.py:390-408`

```python
def load_values_from_reference(kind: str, name: str, key: str, namespace: str) -> Dict[str, Any]:
    """
    Load Helm values from a ConfigMap or Secret
    """
    try:
        if kind == "ConfigMap":
            cm = kr8s.objects.ConfigMap.get(name=name, namespace=namespace)
            values_yaml = cm.data.get(key, "")
        elif kind == "Secret":
            secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
            import base64
            values_yaml = base64.b64decode(secret.data.get(key, "")).decode('utf-8')
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        return yaml.safe_load(values_yaml) or {}
    except Exception as e:
        logging.error(f"Error loading values from {kind}/{name}: {str(e)}")
        return {}
```

**What to copy verbatim:**
- The kr8s fetch lines (`kr8s.objects.ConfigMap.get(name=name, namespace=namespace)` and `kr8s.objects.Secret.get(...)`) — kr8s call shape MUST NOT change. The new exception handling wraps them.
- Local `import base64` placement inside the Secret branch (existing convention).
- `yaml.safe_load(values_yaml) or {}` final-return idiom.

**What to adapt per CONTEXT.md D-01..D-05:**
- Signature evolves to `(kind, name, key, namespace, variables: Optional[Dict[str, str]] = None, *, comp_name: Optional[str] = None, ref_index: Optional[int] = None) -> Dict[str, Any]`.
- Replace broad `except Exception → return {}` with typed dispatch (`kr8s.NotFoundError`, `kr8s.APITimeoutError`, `kr8s.ServerError` with `e.response.status_code` disambiguator, `yaml.YAMLError`).
- Insert `if variables is not None: values_yaml = _render_or_raise(values_yaml, variables, source_desc=...)` BEFORE `yaml.safe_load`.
- Helm callsite at line 923 keeps positional 4-arg call — DO NOT modify.

**Analog 2: existing kopf error raise idiom.**
**Source:** `operator_module/main.py:597, 620, 868, 893, 965`

```python
# Line 597 (handle_appstack_deployment)
raise kopf.PermanentError("appStack.components is required and cannot be empty")

# Line 620 (chained idiom)
except ValueError as e:
    raise kopf.PermanentError(f"Dependency resolution failed: {str(e)}")

# Line 868
raise kopf.PermanentError(error_msg)

# Line 893
raise kopf.PermanentError("helmChart.name is required")

# Line 965 (TemporaryError with delay= kwarg)
raise kopf.TemporaryError(message, delay=30)
```

**What to copy verbatim:**
- `kopf.PermanentError(...)` and `kopf.TemporaryError(message, delay=30)` constructor shape — `delay=` is keyword.
- Error-wrapping pattern from line 620: `except ValueError as e: raise kopf.PermanentError(f"... {str(e)}")` — Phase 18 enhances this with `from e` chaining (per D-15) and richer context strings (per D-03/D-04).

---

### `operator_module/main.py` — `handle_appstack_deployment` variables-dict build (insert at ~line 600)

**Analog: existing function head structure.**
**Source:** `operator_module/main.py:589-622`

```python
def handle_appstack_deployment(body, spec, name, namespace, status, **kwargs):
    """Handle AppStack multi-component deployment with dependencies"""
    logging.info(f"Deploying AppStack {name}")

    app_stack = spec['appStack']
    components = app_stack.get('components', [])

    if not components:
        raise kopf.PermanentError("appStack.components is required and cannot be empty")

    # Filter enabled components
    enabled_components = [comp for comp in components if comp.get('enabled', True)]

    if not enabled_components:
        logging.warning(f"No enabled components in AppStack {name}")
        return {
            'appStackPhase': 'Ready',
            ...
        }

    # Resolve dependencies
    try:
        ordered_components = resolve_dependencies(enabled_components)
    except ValueError as e:
        raise kopf.PermanentError(f"Dependency resolution failed: {str(e)}")
```

**What to copy verbatim:**
- The variable dereference pattern `app_stack = spec['appStack']` — sibling reference is `app_stack.get('variables') or {}` per D-16.
- Early `kopf.PermanentError` raise (line 597) for invalid input — same pattern for invalid variable keys (D-16 step 2).

**What to adapt per CONTEXT.md D-16:**
- Insert variables build + key/type validation BETWEEN `enabled_components` filter (line 600) and `resolve_dependencies` call (line 618). Order is locked: filter → variables build → validate keys → validate values → resolve_dependencies.

---

### `operator_module/main.py` — kubernetesManifest render (insert at line ~779)

**Analog: existing tempfile + kubectl-apply block.**
**Source:** `operator_module/main.py:765-796`

```python
elif 'kubernetesManifest' in component and component['kubernetesManifest']:
    # Deploy raw Kubernetes manifest
    manifest_yaml = component['kubernetesManifest']
    target_namespace = component.get('targetNamespace', namespace)

    # Check if manifest is empty or contains only whitespace/comments
    manifest_stripped = manifest_yaml.strip()
    if not manifest_stripped or all(line.strip().startswith('#') or not line.strip()
                                   for line in manifest_stripped.split('\n')):
        logging.warning(f"Component {comp_name} has empty kubernetesManifest, skipping deployment")
        comp_status['phase'] = 'Ready'
        comp_status['message'] = 'Skipped: Empty manifest (placeholder component)'
    else:
        # Write manifest to temp file and apply
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(manifest_yaml)
            manifest_file = f.name
        ...
```

**What to copy verbatim:**
- The empty-manifest skip-check (must run BEFORE render, since pre-scan guard returns empty unchanged anyway).
- `tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)` line.
- The `try / finally: if os.path.exists(manifest_file): os.unlink(manifest_file)` cleanup.

**What to adapt per CONTEXT.md "Specific Ideas":**
- Insert `manifest_yaml = _render_or_raise(manifest_yaml, stack_vars, source_desc=f"Component '{comp_name}'.kubernetesManifest")` BETWEEN the empty-manifest check and the `with tempfile.NamedTemporaryFile(...)` call.

---

### `operator_module/main.py` — `field='spec'` decorator (line 1053)

**Analog: existing decorator surface.**
**Source:** `operator_module/main.py:1053-1054`

```python
@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores')
def update_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    logging.info(f"*** WarrpAppStore Updated: {name}")
    ...
```

**What to copy verbatim:**
- Decorator argument order: `('warp.io', 'v1alpha1', 'wekaappstores')`.
- Function signature with `**kwargs`.

**What to adapt per CONTEXT.md D-17:**
- Single-line change: append `, field='spec'` as the 4th positional/kwarg of `@kopf.on.update`.
- Final form: `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')`.

---

### `weka-app-store-operator-chart/Chart.yaml` (version bump)

**Analog: Phase 17 commit `81d86ed`.**
**Source:** Phase 17 chart bump commit message + diff
```
chore(17-01): bump chart version 0.1.61 -> 0.1.62

Patch bump signaling additive backward-compatible CRD change
(spec.appStack.variables added in the prior commit).

Per D-15: chart is NOT packaged or published in this phase — no helm
package, no helm repo index, no docs/ mutations. Per D-16: no
CHANGELOG.md, no description: edit, no appVersion: bump (only the
single version: line is touched).
```

Existing line to mutate (`Chart.yaml:18`):
```yaml
version: 0.1.62
```

**What to copy verbatim:**
- Touch ONLY the `version:` line on line 18.
- Do NOT bump `appVersion` (line 24).
- Do NOT modify `description:` (line 3).
- Do NOT package (`helm package`) or rebuild index (`helm repo index docs`) — Phase 18 is operator-image-affecting source change only.
- Commit message style: `chore(18-XX): bump chart version 0.1.62 -> 0.1.63` with rationale paragraph.

**What to adapt:**
- Bump from `0.1.62` to `0.1.63`.
- Rationale: signals operator-image-affecting wiring change (`render()` is now called from `handle_appstack_deployment` and `load_values_from_reference`).

---

### `README.md` — new top-level section

**Analog: existing top-level section structure.**
**Source:** `README.md:108-149` (`## Readiness checks for components`)

```markdown
## Readiness checks for components (pods or deployments)
When using AppStack components, you can control how the operator waits for a component to become ready after installation. The operator supports waiting on either pods (default) or deployments using `kubectl wait` under the hood.

Examples:

- Wait for a specific deployment by name (recommended when you know the resource name):

  appStack:
    components:
      - name: envoy-gateway
        enabled: true
        helmChart:
          repository: oci://example.registry/charts
          name: envoy-gateway
          version: 1.2.3
        readinessCheck:
          type: deployment
          name: envoy-gateway
          namespace: envoy-gateway-system
          timeout: 300

...

Notes:
- If `readinessCheck.name` is set, the operator waits for `type/name` in the specified namespace (or the component targetNamespace if omitted).
- ...
- Supported types: `pod`, `deployment`, `statefulset`, `job`.
```

**What to copy verbatim:**
- Heading depth: `## Heading` (top-level, like every other section).
- Indented YAML examples (2-space indent, no triple-backticks in some existing examples — Phase 18 should USE triple-backticks ```yaml fenced blocks for clarity since the section will have larger snippets).
- Notes-bullets pattern (`Notes:` followed by `- ` list).
- Backtick-quoted field references (e.g., `` `targetNamespace` ``).

**What to adapt per CONTEXT.md D-06..D-10:**
- Place between `## Common configuration` (ends ~line 78) and `## Upgrading` (line 80).
- Section ordering INSIDE: (1) why-paragraph → (2) syntax table → (3) worked example → (4) no-recursion callout → (5) operator-control callout → (6) error semantics.
- Worked example must use AIDP-style multi-component CR (milvus + ingress, `${namespace}` + `${milvusHost}`, valuesFiles ref) — NOT the broken PRD example with nested `${namespace}`.
- Use `> **Note:**` block-quote for D-08 callout (no recursive resolution).

---

### `operator_module/tests/test_appstack.py` (NEW — TST-02)

**Analog 1: file-header sys.path setup + naming convention.**
**Source:** `operator_module/tests/test_render.py:1-17`

```python
"""Unit tests for operator_module.main.render().

Tests cover: pre-scan guard, $$ escape, JSON-safety (plain + substitution),
undefined-variable error, malformed-placeholder error, no-op when variables
is None or {}, multi-occurrence, and the cluster_init shell-script regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))
```

**What to copy verbatim:**
- The header docstring style ("Unit tests for operator_module.main.X. Tests cover: ...") — Phase 18 should write `Unit tests for operator_module.main.handle_appstack_deployment substitution wiring (TST-02).`
- `from __future__ import annotations` (typing affordance).
- Defense-in-depth sys.path block — present in BOTH this file AND `conftest.py` per Phase 16's established pattern.
- Lazy `from main import X` inside each test function (NOT module-level) — see line 40, 47, 60, etc. of `test_render.py`. This pattern lets the sys.path injection complete before import resolution runs.

**Analog 2: `unittest.mock.patch` mocking idiom.**
**Source:** `mcp-server/tests/test_apply_tool.py:13, 49-63, 102-106`

```python
from unittest.mock import MagicMock
import pytest

def _make_mock_deps(applied_kinds: list[str] | None = None) -> "ApplyGatewayDependencies":
    """Build a fully mocked ApplyGatewayDependencies with no real K8s calls."""
    ...
    mock_api = MagicMock()
    mock_api.create_namespaced_custom_object.return_value = None
    return ApplyGatewayDependencies(
        load_kube_config=MagicMock(),
        ...
        custom_objects_api_factory=MagicMock(return_value=mock_api),
        ...
    )

# Inside a test:
result = _apply_impl(
    yaml_text=SAMPLE_YAML,
    namespace="default",
    confirmed=True,
    apply_gateway_deps=mock_deps,
)
assert result["applied"] is True
mock_deps.load_kube_config.assert_not_called()
```

**What to copy verbatim:**
- `from unittest.mock import MagicMock` and `from unittest.mock import patch` (NOT `pytest-mock`'s `mocker` fixture — Phase 18 keeps repo style).
- Helper-builder function pattern (`_make_mock_deps`) — Phase 18 can have `_make_cr(spec_overrides)` helper if test bodies repeat fixture construction.
- `mock.assert_not_called()` / `mock.call_args_list` post-execution assertions.
- Inline literal `SAMPLE_YAML = """\\n..."""` constants at top of file (line 29).

**Analog 3: parametrize idiom for multi-case TST-02.**
**Source:** `mcp-server/tests/test_tool_descriptions.py:53-60`

```python
EXPECTED_TOOLS = [
    "inspect_cluster",
    "inspect_weka",
    ...
]


@pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
def test_tool_has_non_empty_description(tool_descriptions, tool_name):
    """Each tool must have a non-empty description string."""
    desc = tool_descriptions.get(tool_name, "")
    assert desc, f"Tool '{tool_name}' has empty description"
```

**What to copy verbatim:**
- `@pytest.mark.parametrize("param_name", LIST_OF_VALUES)` — sole repo example.
- One-line docstring per parametrized test.
- Phase 18 TST-02 use case: parametrize over `(case_name, manifest_template, vars, expected_substring)` for the four manifest/valuesFiles cases per CONTEXT.md "Specific Ideas TST-02 outline".

**What to adapt per CONTEXT.md "Specific Ideas":**
- Mock `subprocess.run` (kubectl), `kr8s.objects.ConfigMap.get`, `kr8s.objects.Secret.get`, and the `HelmOperator` class (line 30 of main.py) — all reachable as `main.subprocess.run`, `main.kr8s.objects.ConfigMap.get`, `main.HelmOperator`.
- Capture manifest tempfile content via `side_effect=fake_run` that reads `Path(cmd[3]).read_text()` before `os.unlink` runs.
- Cases: (a) `${namespace}` auto-default in manifest, (b) explicit `namespace` user-key override, (c) `${milvusHost}` from CM valuesFiles, (d) `${milvusHost}` from Secret valuesFiles.

---

### `operator_module/tests/test_helm_non_wiring.py` (NEW — TST-05)

**Analog 1: file-header layout (mirror `test_render.py`).**
**Source:** `operator_module/tests/test_render.py:1-17` (same as above for `test_appstack.py`).

**Analog 2: `unittest.mock.patch` context-manager pattern.**
**Source:** `mcp-server/tests/test_apply_tool.py` (already cited).

**Analog 3: `inspect.getsource()` static check.**
**Source:** **No exact analog in repo** — Phase 18 introduces this pattern. Closest precedent is `mcp-server/tests/test_openclaw_config.py:149-181` which compares generator output to on-disk file (drift detection — same conceptual category as static-source check).

```python
def test_openclaw_json_matches_generation():
    """openclaw.json matches what generate_openclaw_config.py produces.

    Runs the generation script programmatically into a temp buffer (no file write)
    and compares the tool list. Detects drift when tool docstrings change but
    openclaw.json is not regenerated.
    """
    import sys
    sys.path.insert(0, str(MCP_SERVER_ROOT))
    from generate_openclaw_config import collect_tool_descriptions, build_openclaw_config

    tool_descriptions = collect_tool_descriptions()
    fresh_config = build_openclaw_config(tool_descriptions)

    on_disk_config = json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))
    fresh_tools = {t["name"]: t["description"] for t in fresh_config.get("tools", [])}
    disk_tools = {t["name"]: t["description"] for t in on_disk_config.get("tools", [])}

    assert fresh_tools == disk_tools, (
        "openclaw.json is out of sync with tool descriptions in tools/*.py.\n"
        ...
    )
```

**What to copy verbatim:**
- "Drift-detection" framing — TST-05's `inspect.getsource()` check serves the same purpose: locks behavior at the source level.
- Detailed assertion-error message (multi-line `f"...\n..."`) telling the reader HOW to fix the failure.

**What to adapt per CONTEXT.md D-14:**
- Two test functions:
  1. `test_handle_helm_deployment_does_not_pass_variables` — `with patch("main.load_values_from_reference", return_value={}) as mock_load` + `with patch("main.HelmOperator")` (note: per RESEARCH.md A4, `kr8s.NotFoundError` etc. are reachable via `kr8s.*`; mocks reach module-attached references via `main.X`).
  2. `test_handle_helm_deployment_source_has_no_render` — `import inspect; src = inspect.getsource(handle_helm_deployment); assert "render(" not in src`.
- Per RESEARCH.md Pitfall 3, `handle_helm_deployment` is NOT decorated, so `inspect.getsource()` returns the actual function body. Safe.

---

### `operator_module/tests/test_backward_compat_snapshot.py` (NEW — TST-03)

**Analog 1: hand-rolled snapshot via `Path.read_text()`.**
**Source:** `mcp-server/tests/test_openclaw_config.py:23-24, 44-47, 65, 167`

```python
# Path to mcp-server/ directory — always relative to this file
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
OPENCLAW_JSON_PATH = MCP_SERVER_ROOT / "openclaw.json"


@pytest.fixture(scope="module")
def openclaw_config() -> dict:
    """Load openclaw.json once for the module."""
    return json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))


# Inside test:
content = OPENCLAW_JSON_PATH.read_text(encoding="utf-8")
config = json.loads(content)
...
on_disk_config = json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))
```

**What to copy verbatim:**
- `Path(__file__).resolve().parents[1]` walk-up idiom for repo-relative paths.
- Module-level `Path` constant for fixture/baseline locations (e.g., `SNAPSHOTS = Path(__file__).parent / "snapshots" / "ai-research"`).
- `Path.read_text(encoding="utf-8")` for baseline reads.
- `@pytest.fixture(scope="module")` for one-time fixture loading.
- Plain `==` assertion for byte-identity (no third-party diff library).
- Detailed assertion-error message naming the file and the fix command (e.g., "Re-run with BASELINE_REGEN=1 to regenerate").

**Analog 2: fixture file path resolution to `mcp-server/tests/fixtures/`.**
**Source:** `operator_module/tests/test_render.py:49-52`

```python
fixture_path = (
    Path(__file__).resolve().parents[2] / "cluster_init" / "app-store-cluster-init.yaml"
)
content = fixture_path.read_text(encoding="utf-8")
```

**What to copy verbatim:**
- `Path(__file__).resolve().parents[2]` (2 levels up = repo root) for cross-directory fixture references.
- Phase 18 path: `Path(__file__).resolve().parents[2] / "mcp-server" / "tests" / "fixtures" / "sample_blueprints" / "ai-research.yaml"`.

**What to adapt per CONTEXT.md D-11..D-13 + RESEARCH.md Pitfall 6:**
- Add `_normalize_camel(d)` helper at top of file to convert snake_case → camelCase keys (`helm_chart→helmChart`, `target_namespace→targetNamespace`, `release_name→releaseName`, `crds_strategy→crdsStrategy`, `wait_for_ready→waitForReady`, `readiness_check→readinessCheck`, `depends_on→dependsOn`).
- Capture two outputs per component: (a) merged Helm values dict — patch `main.HelmOperator` and grab `mock_helm_cls.return_value.install_or_upgrade.call_args.kwargs['values']`; (b) manifest tempfile content — patch `main.subprocess.run` and read tempfile before unlink.
- Baseline format per RESEARCH.md Open Question 2: JSON with `indent=2, sort_keys=True` for values; raw YAML string for manifest.
- Re-generation gate: `if os.environ.get("BASELINE_REGEN") == "1": path.write_text(content)` (per RESEARCH.md Open Question 3).

---

### `operator_module/tests/snapshots/ai-research/` (NEW directory)

**No existing analog in repo.** First snapshot baseline directory.

**Pattern to establish (per CONTEXT.md D-12 + RESEARCH.md Open Question 2):**
- Files: `values_<comp>.json` (one per helm component — for `ai-research.yaml`, that's `values_vector-db.json` and `values_research-api.json`); `manifest_<comp>.yaml` (one per kubernetesManifest component — `ai-research.yaml` has none, so this fixture produces ONLY values baselines; the file pattern is established for future fixtures).
- Generation: `BASELINE_REGEN=1 pytest operator_module/tests/test_backward_compat_snapshot.py` produces these files on first run.
- Format conventions:
  - `values_*.json`: `json.dumps(values_dict, indent=2, sort_keys=True)` — deterministic, reviewable diff.
  - `manifest_*.yaml`: byte-identical content of the tempfile written by the manifest-render path (after `_render_or_raise` is a no-op).

---

## Shared Patterns

### Authentication / Cluster-config
**Source:** `operator_module/main.py:14-21`
**Apply to:** No new code in Phase 18 needs auth wiring (operator runs in-cluster; tests mock kr8s). Mention here for completeness — the existing optional `kubernetes` client import block is the project-wide pattern, and Phase 18's tests must NOT call `_load_kube_config_once()` (line 297).

### Error Handling — kopf typed exceptions
**Source:** `operator_module/main.py:597, 620, 868, 893, 965`
**Apply to:** All new error paths in `_render_or_raise`, the rewritten `load_values_from_reference`, and the variables-build validation.

```python
# PermanentError: bad CR / bad config / cannot proceed
raise kopf.PermanentError(f"<context-rich message>")

# PermanentError chained from underlying exception (Phase 18 D-15 pattern)
raise kopf.PermanentError(f"{source_desc}: {e}") from e

# TemporaryError: cluster wobble / will retry
raise kopf.TemporaryError(f"<context-rich message>", delay=30)
```

### Logging
**Source:** `operator_module/main.py:591, 622, 642, 804`
**Apply to:** No new logging is mandated by Phase 18's locked decisions, but if added, mirror existing `logging.info(f"...")` / `logging.error(f"...")` / `logging.warning(f"...")` patterns at module level (NOT class-instance loggers — those are reserved for `HelmOperator`).

### Testing — sys.path injection (defense-in-depth)
**Source:** `operator_module/tests/conftest.py` + repeated in `operator_module/tests/test_render.py:14-17`
**Apply to:** All three new test files in `operator_module/tests/`.

```python
# In conftest.py (already exists):
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))

# Repeated in each test file (defense-in-depth — direct invocation works):
import sys
from pathlib import Path
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))
```

### Testing — lazy import inside test functions
**Source:** `operator_module/tests/test_render.py:40, 47, 60, 67, 74, 87, 97, 106, 117, 126, 134, 141`
**Apply to:** All new test files. Reason: ensures sys.path setup runs before resolution.

```python
def test_X():
    """Docstring."""
    from main import handle_appstack_deployment   # lazy import
    ...
```

### Testing — mocking idiom
**Source:** `mcp-server/tests/test_apply_tool.py:13`
**Apply to:** All three new test files.

```python
from unittest.mock import patch, MagicMock
# NOT pytest-mock; NOT pytest_mock.mocker fixture
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `operator_module/tests/snapshots/ai-research/` | test fixtures (baselines) | file-IO | First snapshot directory in repo. Phase 18 establishes the convention; no prior analog to mirror. Closest precedent (`mcp-server/tests/fixtures/sample_blueprints/`) is for INPUTS, not deterministic OUTPUTS — different semantic. |

## Metadata

**Analog search scope:**
- `operator_module/main.py` (1140 lines, fully-mapped — read lines 1-12, 232-287, 290-340, 385-410, 589-810, 855-970, 1045-1075).
- `operator_module/tests/` (test_render.py, conftest.py — fully read).
- `mcp-server/tests/` (test_apply_tool.py, test_openclaw_config.py, test_blueprints.py, test_tool_descriptions.py — fully read for analog patterns).
- `weka-app-store-operator-chart/Chart.yaml` (fully read; Phase 17 commit `81d86ed` diff inspected).
- `README.md` (fully read).
- `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` (fully read; snake_case caveat per RESEARCH.md Pitfall 6 confirmed).

**Files scanned:** 9 source/test files + 2 commits.

**Pattern extraction date:** 2026-05-08

**Analog quality summary:**
- `operator_module/main.py` (modify): **HIGH** — self-analogs are exact (sibling helpers, sibling kopf.PermanentError sites).
- `weka-app-store-operator-chart/Chart.yaml` (modify): **HIGH** — Phase 17 commit `81d86ed` is a verbatim template (single-line bump, no packaging).
- `README.md` (modify): **MEDIUM** — existing top-level sections give heading depth + tone, but no existing section has the same structural needs (syntax table + worked example + side-by-side WRONG/CORRECT). CONTEXT.md D-10 fully specifies the structure.
- `operator_module/tests/test_appstack.py` (NEW): **HIGH** — `test_render.py` for layout, `test_apply_tool.py` for mocking, `test_tool_descriptions.py:53` for parametrize. Three strong analogs.
- `operator_module/tests/test_helm_non_wiring.py` (NEW): **MEDIUM** — file layout is HIGH (mirrors `test_render.py`); the `inspect.getsource()` static-check is a NEW pattern (no exact analog), but conceptually mirrors `test_openclaw_config.py:149-181` drift-detection.
- `operator_module/tests/test_backward_compat_snapshot.py` (NEW): **HIGH** — `test_openclaw_config.py` is an exact `Path.read_text() + ==` template; only addition is the `BASELINE_REGEN` env-var gate (per RESEARCH.md, no analog exists for that and it's a one-liner).
- `operator_module/tests/snapshots/ai-research/` (NEW dir): **LOW** — no analog in repo; convention established by Phase 18.

## PATTERN MAPPING COMPLETE

**Phase:** 18 - Operator Wiring and Docs
**Files classified:** 7
**Analogs found:** 6 HIGH/MEDIUM / 7 (1 file establishes new convention)

### Coverage
- Files with exact (HIGH) analog: 4 (main.py, Chart.yaml, test_appstack.py, test_backward_compat_snapshot.py)
- Files with role-match (MEDIUM) analog: 2 (README.md, test_helm_non_wiring.py)
- Files with no analog (LOW / convention-establishing): 1 (snapshots/ai-research/ directory)

### Key Patterns Identified
- **Self-analog dominance in main.py:** All four wiring sites (helper placement, kopf.PermanentError raise idiom, decorator argument order, function-head structure) have direct precedent within the same file. Phase 18 is plumbing; new abstractions are unwarranted.
- **Test-file convention is locked from Phase 16:** `from __future__ import annotations`, defense-in-depth sys.path block, lazy `from main import X` inside test bodies, `unittest.mock.patch` (NOT `pytest-mock`). Three new files mirror `test_render.py` layout.
- **Snapshot pattern is `Path.read_text()` + plain `==`:** No third-party snapshot library. `mcp-server/tests/test_openclaw_config.py` is the verbatim template; addition for Phase 18 is the `BASELINE_REGEN=1` env-var gate (one-liner).
- **Chart bump cadence is established:** Patch-bump signaling additive backward-compat operator-image change; no packaging, no docs/ mutation, no CHANGELOG. Phase 17's commit message is the template.
- **README structure inserts cleanly:** New `## Variable substitution in AppStack manifests` section between `## Common configuration` (line 78) and `## Upgrading` (line 80). Existing top-level sections supply heading depth and prose tone.

### Pitfalls to Carry Forward (from RESEARCH.md, relevant for planner)
- TST-03 fixture uses snake_case — planner adds 5-line `_normalize_camel(d)` helper at top of `test_backward_compat_snapshot.py` (RESEARCH.md Pitfall 6, Option 1).
- TST-05's `inspect.getsource()` is safe because `handle_helm_deployment` is NOT kopf-decorated (RESEARCH.md Pitfall 3).
- Helm-path callsite at `main.py:923` MUST stay positional 4-arg (RESEARCH.md Pitfall 7) — TST-05 locks this.

### File Created
`/Users/christopherjenkins/git/wekaappstore/.planning/phases/18-operator-wiring-and-docs/18-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns directly in PLAN.md files for Phase 18.
