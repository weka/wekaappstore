# Architecture Research

**Domain:** Kubernetes Operator — AppStack variable substitution integration
**Researched:** 2026-05-06
**Confidence:** HIGH (all findings from direct source inspection)

---

## System Overview

The substitution feature threads a stateless `render()` helper through exactly two execution paths inside `handle_appstack_deployment`. No new processes, controllers, or external services are involved. The entire change is confined to three files with no new runtime dependencies.

```
WekaAppStore CR
  spec.appStack.variables: {namespace: "prod", milvusHost: "milvus.prod.svc.cluster.local"}
        |
        v
kopf.on.create / kopf.on.update  (operator_module/main.py:807, 1015)
        |
        v
handle_appstack_deployment()  (main.py:551)
        |
        +-- BUILD variables dict once  (new -- line ~555)
        |     {'namespace': <CR namespace>, **(appStack.get('variables') or {})}
        |
        +-- for each component (ordered by resolve_dependencies):
        |
        |   +-- Helm path (main.py:616-725) ------------------------------------+
        |   |   load_values_from_reference(kind, name, key, ns,                |
        |   |                              variables=variables)  <- NEW ARG     |
        |   |     +-- render(raw_string, variables)  <- NEW CALL               |
        |   |         +-- yaml.safe_load(rendered_string)                      |
        |   |   deep_merge(inline_values, ref_values)                          |
        |   |   helm install/upgrade (values written to tempfile)              |
        |   +------------------------------------------------------------------+
        |
        |   +-- Manifest path (main.py:727-758) --------------------------------+
        |   |   render(component['kubernetesManifest'], variables)  <- NEW     |
        |   |   write rendered string to tempfile                              |
        |   |   kubectl apply -f <tempfile> -n <targetNamespace>              |
        |   +------------------------------------------------------------------+
        |
        v
componentStatus[].phase / .message  (already exists in status CRD)

mcp-server/tools/validate_yaml.py  (parallel -- no operator dependency)
  +-- Accept spec.appStack.variables as valid field  (soft, not strict check)
  +-- Emit soft-warning on *.svc.cluster.local / namespace: <literal>
```

---

## Integration Points at File:Line Precision

### New Component

**`operator_module/main.py` — `render()` helper (~line 13, after imports)**

```python
from string import Template

def render(text: str, variables: dict) -> str:
    """Substitute ${VAR} from variables dict. Raises KeyError on undefined keys."""
    if not variables or not text:
        return text
    return Template(text).substitute(variables)
```

- New function, no side effects, fully pure.
- `string.Template` is stdlib — no new dependency.
- `Template.substitute` (strict) raises `KeyError` on any `${UNRESOLVED}` token.
- `$$` in source text produces a literal `$` — documented Python behavior.

### Modified Component 1: `load_values_from_reference` (main.py:352-370)

Add `variables: dict = None` parameter. Render the raw string (ConfigMap data value or base64-decoded Secret content) before passing to `yaml.safe_load`.

Current call signature (line 352):
```
load_values_from_reference(kind, name, key, namespace) -> Dict
```

New call signature:
```
load_values_from_reference(kind, name, key, namespace, variables=None) -> Dict
```

Substitution site inside the function, between raw-string fetch and `yaml.safe_load`:
```python
values_yaml = render(values_yaml, variables)  # raises KeyError on undefined
```

Catch block re-raises as `kopf.PermanentError` naming `kind/name key=<key>`.

**Call site that must be updated:** main.py:672-677 — the `load_values_from_reference(...)` call inside the Helm-component `valuesFiles` loop. Add `variables=variables` argument.

NOTE: The single-component Helm path at main.py:883-891 (`handle_helm_deployment`) also calls `load_values_from_reference` without `variables`. This path is NOT touched by this PRD (it has no `appStack.variables` concept). The defaulted `variables=None` parameter ensures `render()` is a no-op when `variables` is `None`, so the single-component path is unaffected.

### Modified Component 2: manifest branch in `handle_appstack_deployment` (main.py:727-758)

Current code at line 729:
```python
manifest_yaml = component['kubernetesManifest']
```

New code:
```python
try:
    manifest_yaml = render(component['kubernetesManifest'], variables)
except KeyError as e:
    raise kopf.PermanentError(
        f"Undefined ${{{e.args[0]}}} in component "
        f"{component['name']}.kubernetesManifest")
```

No other changes to the manifest path. The rendered string is written to a tempfile at line 741 exactly as before.

### Modified Component 3: variables dict construction in `handle_appstack_deployment` (main.py:555-556)

After `components = app_stack.get('components', [])` (current line 556), add:
```python
variables = {'namespace': namespace, **(app_stack.get('variables') or {})}
```

This is the only construction site. It is stack-scoped, not per-component. Rationale confirmed below.

### Modified Component 4: CRD schema (weka-app-store-operator-chart/templates/crd.yaml:88-192)

Add `variables:` as a sibling of `components:` under `spec.properties.appStack.properties` (around line 89):

```yaml
variables:
  type: object
  description: |
    Map of variable name to string value. Substituted as ${VAR} into:
      - kubernetesManifest strings
      - The raw content of ConfigMap/Secret referenced by valuesFiles
    The variable ${namespace} auto-defaults to the CR metadata.namespace
    if not explicitly set. Use $$ to escape a literal dollar sign.
    Undefined references raise a permanent error at apply time.
  additionalProperties:
    type: string
```

No `required:` annotation — the field is optional. CRD schema is additive and backward-compatible.

### Modified Component 5: `mcp-server/tools/validate_yaml.py`

Two additions to `_validate_yaml_impl()`:

1. No new error raised for `spec.appStack.variables` — it is already allowed because the validator only checks for known-bad fields (v1-only snake_case fields) and presence of a deployment method. `variables` will pass silently with the current logic. However, add it to a comment or `_KNOWN_APPSTACK_FIELDS` constant so future reviewers know it is intentional.

2. Add soft-warning scan of `kubernetesManifest` strings. Walk `spec.appStack.components[].kubernetesManifest` and emit a `warnings` entry if any of:
   - `.svc.cluster.local` appears as a literal substring
   - `namespace:` followed by a non-`${` value appears

The warning is non-blocking: `valid` remains `True`, `errors` remains empty. Wording: `"Component '{name}' kubernetesManifest contains hardcoded value '{fragment}'; consider ${namespace} or a variables: substitution."` (abbreviated).

### New Component: `operator_module/tests/test_render.py`

New test file — directory `operator_module/tests/` does not currently exist and must be created along with an `__init__.py`. Coverage:

- `render()` with JSON payload containing `{` and `}` characters (no crash)
- undefined key raises `KeyError`
- empty string returns empty string
- `$$` produces literal `$`
- `variables=None` or empty dict returns text unchanged
- multi-occurrence: same `${VAR}` appears twice, both substituted

### New Component: `mcp-server/tests/fixtures/sample_blueprints/ai-research-portable.yaml`

New fixture demonstrating the recommended pattern with `variables:` and `${namespace}` usage. This fixture doubles as a regression guard: validate_yaml must return `valid=True` for a CR with `variables:`.

---

## Validated Architecture Questions

### 1. Variables-dict Construction Scope: Stack-Level Is Correct

The PRD proposes building `variables` once at the top of `handle_appstack_deployment`. This is confirmed correct.

**Evidence:** `variables` is read from `app_stack.get('variables')`, which is a single field at `spec.appStack.variables`. It is not a per-component field in the CRD schema. The `variables` dict is read-only after construction and passed by reference into both execution branches — no mutation risk.

**Implication:** `variables` is constructed once at ~line 555, before the `for component in ordered_components:` loop at line 602. All components in a single reconcile share the same variable values. This is the intended behavior.

### 2. Inline `values:` Not Substituted — The Hole Is Real but Acceptable

The PRD explicitly excludes `component.values:` (inline YAML object) from substitution. Confirmed: at main.py:668, `merged_values = component.get('values', {}).copy()` copies the already-parsed Python dict. By the time it reaches this line, the YAML has been parsed by Kubernetes admission into a Python object — the raw string form no longer exists. `string.Template` operates on strings only, so there is no insertion point.

**The hole:** A user who writes `${VAR}` directly in `component.values:` will not get substitution. The `$` will arrive at `helm install` as a literal string, which Helm does not interpret.

**Practical severity: LOW.** The AIDP case uses `valuesFiles:` (ConfigMap/Secret references), not inline `values:`. The workaround is documented: put substitution-bearing values in a ConfigMap referenced by `valuesFiles:`.

**Silent failure risk: MEDIUM.** A user who writes `${namespace}` in `component.values.someKey` expecting substitution will observe it arrive unexpanded in Helm with no error or warning. Recommend adding a README note: "`${VAR}` substitution does not apply to inline `values:` blocks. Use `valuesFiles:` instead."

### 3. Multi-Document YAML in `kubernetesManifest` — Confirmed Supported

`string.Template` is string-level and treats `---` separators as any other substring. The operator's manifest branch at main.py:740-748 writes `manifest_yaml` verbatim to a tempfile and runs `kubectl apply -f <tempfile>`. The `kubectl apply` command processes multi-document YAML files natively. The operator does not call `yaml.safe_load` on `kubernetesManifest` content — it passes the raw (now rendered) string directly to kubectl.

**Conclusion:** Multi-document manifests work today and continue to work after substitution is added. No special handling needed. `render()` produces a rendered multi-document string; kubectl applies it as-is.

**Edge case confirmed safe:** `${VAR}` tokens inside multi-line YAML string values (e.g., inside a ConfigMap data key) are substituted correctly because `string.Template` operates on the raw string, not on parsed YAML structure.

### 4. Idempotency Under kopf Reconcile

kopf fires `on.create` once and `on.update` on every spec change. Both handlers call `handle_appstack_deployment` identically (confirmed at main.py:813, 1021).

`render()` is a pure function: same input always produces same output. There is no state accumulated between reconcile runs. The operator does not cache substituted output.

**Drift risk: None.** `string.Template.substitute` is deterministic: `$$` always becomes `$`, `${VAR}` always becomes the variable value. The only scenario producing different output between reconciles is if the ConfigMap or Secret content changes externally — but re-rendering on that change is correct behavior.

**`kopf.PermanentError` idempotency:** If render fails with an undefined variable, kopf marks the CR permanently failed and stops retrying. The user must update the CR spec to fix the variable, which triggers `on.update` and a fresh render attempt. This is the correct UX.

### 5. Status and Observability

Per-component status is already confirmed in CRD schema at crd.yaml:224-249 (`componentStatus` array with `name`, `phase`, `message`, `lastTransitionTime`). The operator populates it at main.py:607-612 and the exception handler at main.py:762-765 catches all exceptions and sets `comp_status['phase'] = 'Failed'` and `comp_status['message'] = f"Error deploying component: {str(e)}"`.

**Substitution failure path:** A `kopf.PermanentError` raised inside the per-component `try` block (line 614) is caught by `except Exception as e:` at line 762. The error message — which names the undefined variable and component — lands in `componentStatus[i].message`. `appStackPhase` becomes `Failed`.

**What the user sees:** `kubectl describe wekaappstore <name>` shows the failed component with message: `"Undefined ${unset} in component foo.kubernetesManifest"`. No new status fields are required for v5.0 observability.

### 6. Build Order Recommendation

**Dependency graph:**

```
Phase 1: render() helper + operator_module/tests/ scaffolding
    |
    +-- Phase 2: CRD schema update (additive, safe to ship early)
    |
    +-- Phase 3: Operator wiring (depends on render() existing)
    |     +-- variables dict construction in handle_appstack_deployment
    |     +-- load_values_from_reference signature + call site update
    |     +-- kubernetesManifest render call
    |
    +-- Phase 4: validator soft-warning + fixture (parallel to 2 and 3)
          +-- validate_yaml.py soft-warning logic
          +-- test_validate_yaml.py extensions
          +-- ai-research-portable.yaml fixture

Phase 5: AIDP migration (separate repo, separate PR, after Phases 1-3 deployed)
```

**If I were planning the phases:**

**(1) render() helper + unit tests** — Pure Python function with no external dependencies. Write `operator_module/tests/__init__.py`, `operator_module/tests/test_render.py`, and the `render()` function in `main.py`. This phase is independently verifiable with `pytest` before anything touches live operator paths.

**(2) CRD schema update** — Ship `variables:` field to the CRD immediately after the helper exists. The CRD must be deployed before new CRs with `variables:` pass Kubernetes admission validation. CRD changes are additive and backward-compatible (optional field, no `required:` annotation). Can be merged and deployed in isolation with zero operator risk.

**(3) Operator wiring** — Wire variables construction, `load_values_from_reference` signature change, and the two render call sites. Safe to ship after the CRD schema exists in the cluster. Tests in `operator_module/tests/test_appstack.py` cover the wiring paths. This is the phase that delivers the actual feature.

**(4) Validator soft-warning + fixture** — Entirely parallel to Phases 2 and 3. Has no dependency on operator code. The validator does not call `render()`. Can be merged any time as a standalone PR. Blocking it on Phase 3 is unnecessary.

**(5) AIDP migration** — Separate repo (`aidp`), separate PR. Depends on Phases 1-3 being deployed to the cluster. This is the end-to-end smoke test.

**Justification for Phase 2 before Phase 3:** If operator wiring ships before the CRD schema update, existing CRs still work (the operator reads `app_stack.get('variables') or {}` which returns `{}` if the field is absent). However, new CRs authored with `variables:` will fail Kubernetes admission validation until the CRD schema is updated. In a single-PR atomic deployment (operator + chart), Phases 2 and 3 ship together. If split across PRs, the CRD must go first.

### 7. New vs Modified Summary

| File | Status | Change |
|------|--------|--------|
| `operator_module/main.py` | MODIFIED | `render()` helper added (~line 13); `variables` dict at line ~555; `load_values_from_reference` new `variables` param at line 352; call site at line ~675 gets `variables=variables`; manifest branch at line 729 calls `render()` |
| `weka-app-store-operator-chart/templates/crd.yaml` | MODIFIED | `variables:` property added ~line 89 under `spec.properties.appStack.properties` |
| `mcp-server/tools/validate_yaml.py` | MODIFIED | Soft-warning logic for `.svc.cluster.local` and `namespace: <literal>` inside `kubernetesManifest`; `variables:` documented as valid |
| `operator_module/tests/__init__.py` | NEW | Empty — creates the test package (directory is new) |
| `operator_module/tests/test_render.py` | NEW | Unit tests for `render()` helper covering JSON content, undefined keys, `$$` escape, empty input, multi-occurrence |
| `mcp-server/tests/fixtures/sample_blueprints/ai-research-portable.yaml` | NEW | Fixture with `variables:` block demonstrating portable pattern |
| `mcp-server/tests/test_validate_yaml.py` | MODIFIED | Tests for `variables:` accepted; soft-warning emitted on hardcoded DNS literals |
| `README.md` | MODIFIED | User-facing doc: `${VAR}` syntax, `$$` escape, `${namespace}` auto-default, inline `values:` exclusion caveat |

---

## Data Flow: Substitution in Each Path

### Manifest Path

```
CR spec.appStack.components[i].kubernetesManifest  (raw string with ${VAR} tokens)
        |
        v  render(text, variables)  [main.py:729 NEW]
        |  KeyError -> kopf.PermanentError naming variable + component
        v
rendered string (all ${VAR} resolved)
        |
        v  tempfile write  [main.py:741]
        |
        v  kubectl apply -f <tempfile> -n <targetNamespace>  [main.py:746]
        |
        v  comp_status.phase = Ready / Failed  [main.py:749-754]
```

### Helm / valuesFiles Path

```
CR spec.appStack.components[i].valuesFiles[j] -> {kind, name, key, namespace}
        |
        v  kr8s ConfigMap.get / base64.b64decode  [main.py:358-363]
        |
        v  raw string (YAML content with ${VAR} tokens)
        |
        v  render(raw_string, variables)  [load_values_from_reference NEW]
        |  KeyError -> kopf.PermanentError naming kind/name/key
        v
rendered YAML string
        |
        v  yaml.safe_load(rendered_string)  [main.py:367]
        |
        v  deep_merge(inline_values, ref_values)  [main.py:678]
        |
        v  HelmOperator.install_or_upgrade(values=merged_values)  [main.py:697]
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Per-Component Variables Construction

**What people do:** Construct the `variables` dict inside the `for component in ordered_components:` loop.

**Why it's wrong:** `variables` is stack-scoped in the CRD (`spec.appStack.variables`). Rebuilding per component is wasteful and obscures intent. The namespace auto-default does not vary per component.

**Do this instead:** Build once before the loop at line ~555 and pass by reference.

### Anti-Pattern 2: Applying `render()` to Parsed YAML Objects

**What people do:** Attempt to walk the `component.values` dict recursively and apply string substitution to leaf string values.

**Why it's wrong:** Requires recursive object traversal, type-checking at every leaf, and dict reassembly. The YAML has already been parsed to Python objects by Kubernetes admission. There is no raw string to apply `string.Template` to.

**Do this instead:** Accept the constraint. Document the workaround (use `valuesFiles:` for values that need substitution). The AIDP use case is fully covered without this complexity.

### Anti-Pattern 3: Using `safe_substitute` Instead of `substitute`

**What people do:** Call `Template(text).safe_substitute(variables)` which silently leaves undefined `${VAR}` tokens in place.

**Why it's wrong:** A typo like `${naemspace}` in a manifest silently passes through to `kubectl apply`, which either creates a resource with a literal `${naemspace}` namespace (rejected with a confusing API server error) or produces object names containing `${}` characters.

**Do this instead:** Use `substitute` (strict). Catch `KeyError`, re-raise as `kopf.PermanentError` with a message naming the offending token and component. The user gets a clear actionable error.

### Anti-Pattern 4: Touching the `on.create`/`on.update` Handler Signatures

**What people do:** Add a `variables` parameter to the kopf handler functions themselves.

**Why it's wrong:** kopf handler signatures are controlled by kopf's decorator injection system. Adding custom parameters that kopf does not inject will cause registration failure.

**Do this instead:** Construct the variables dict inside `handle_appstack_deployment` from `spec` and `namespace` — both already available as kopf-injected kwargs. No handler signature changes needed.

---

## Scaling Considerations

This feature has no meaningful scaling concerns. Substitution is O(n) on the length of the raw YAML string, performed once per component per reconcile. For the AIDP blueprint (the largest known AppStack), all `kubernetesManifest` strings combined are under 50KB. `string.Template` processing is in the microsecond range.

The only runtime cost is the same `kr8s` API call that already exists for each `valuesFiles:` reference. Substitution adds no extra API calls.

---

## Sources

All findings are from direct inspection of the codebase at HEAD (2026-05-06). No external sources were required.

- `operator_module/main.py` — lines 1-55 (imports), 352-370 (`load_values_from_reference`), 551-803 (`handle_appstack_deployment` and status handling), 807-830 (`on.create`), 1015-1027 (`on.update`)
- `weka-app-store-operator-chart/templates/crd.yaml` — full file (259 lines), specifically lines 85-192 (appStack schema) and 194-253 (status schema)
- `mcp-server/tools/validate_yaml.py` — full file (176 lines)
- `mcp-server/tests/test_validate_yaml.py` — full file (249 lines)
- `.planning/PRD-appstack-variable-substitution.md`
- `.planning/PROJECT.md`

---

*Architecture research for: WEKA App Store — AppStack variable substitution (v5.0)*
*Researched: 2026-05-06*
