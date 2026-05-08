# Phase 18: Operator Wiring and Docs - Research

**Researched:** 2026-05-08
**Domain:** Python kopf operator wiring, kr8s exception taxonomy, pytest snapshot testing, README authoring
**Confidence:** HIGH

## Summary

Phase 18 wires the Phase 16 `render()` helper into two real substitution sites in `operator_module/main.py` and ships a README section documenting the feature. CONTEXT.md locks 18 decisions covering helper extraction (`_render_or_raise`), error message formats, fetch-error classification, snapshot-test scope, and README structure. This research fills the external-lookup gaps the planner needs:

- **kr8s 0.20.10 exception taxonomy is fully mapped** — `NotFoundError` (404 with retry exhaustion), `ServerError` (other 4xx including 401/403/RBAC, AND 5xx — disambiguate via `e.response.status_code`), `APITimeoutError` (httpx timeout), `ConnectionClosedError` (websocket/portforward only — NOT relevant for `.get()`).
- **kopf 1.38.0 `field='spec'`** — accepted, documented to restrict update handlers to changes/adds/removes of that field, skipping all other patches (status patches included). Single-line change at line 1053.
- **`string.Template` error semantics confirmed empirically** — `KeyError` for undefined identifiers, `ValueError("Invalid placeholder...")` for `${}`, `${123}`, bare `$` at end. Phase 16's `render()` already catches both and re-raises `ValueError`.
- **Snapshot strategy: hand-rolled `Path.read_text()`** — the repo already uses this pattern (`mcp-server/tests/test_openclaw_config.py`). Adds no new dep, produces reviewable git diffs, matches the existing `requirements-dev.txt` minimalism (just `pytest>=8.0.0`).
- **`inspect.getsource(handle_helm_deployment)` is safe** — the function is NOT decorated (only the kopf entry handlers `create_warrpappstore_function` and `update_warrpappstore_function` are). Static `'render(' not in inspect.getsource(...)` will work reliably.
- **TST-03 fixture caveat** — `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` uses `snake_case` field names (`helm_chart`, `target_namespace`, etc.) which the operator code does NOT currently consume (`handle_appstack_deployment` reads `helmChart`, `targetNamespace`). The fixture as-is would short-circuit through the `else: raise ValueError` branch at main.py:798. This is a fixture-shape blocker that the planner must address (see `## Open Questions`).

**Primary recommendation:** Plan executes CONTEXT.md decisions D-01..D-18 verbatim. The five concrete external-data findings above unblock D-01 (kr8s class names), D-13/D-14 (mocking), D-12 (snapshot file format), D-17 (kopf field). The fixture caveat for TST-03 needs a 5-line decision before implementation begins.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Carried forward from prior phases (DO NOT re-litigate):**

- **L-01:** `render()` lives in `operator_module/main.py` (Phase 16 D-08); raises `ValueError` on undefined or malformed placeholder (Phase 16 D-04..06); pre-scan guard returns text unchanged when no `${` present (Phase 16 D-01).
- **L-02:** Variables dict shape is `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` — user-supplied `namespace` key wins over auto-default. (REQUIREMENTS OP-06 verbatim.)
- **L-03:** Key-name pattern is `^[_a-zA-Z][_a-zA-Z0-9]*$` (Python identifier). Phase 17 D-02 enforces this at CRD admission; Phase 18 OP-10 enforces it again at the operator (catches API-server bypass). Belt + suspenders.
- **L-04:** Single-pass substitution only; values are taken literally. README must NOT show `milvusHost: milvus.${namespace}.svc.cluster.local` (the broken PRD example). Use fully-resolved values.
- **L-05:** Both `KeyError` AND `ValueError` are caught at every `render()` call site. STATE.md invariant.
- **L-06:** `handle_helm_deployment` (single-chart path) MUST NOT receive variables wiring; `load_values_from_reference` keeps `variables=None` default. TST-05 locks this. (REQUIREMENTS OP-09.)
- **L-07:** Operator-control fields (`helmChart.*`, `releaseName`, `targetNamespace`, `readinessCheck.*`) are NOT templated. README must explicitly call this out and recommend dropping `targetNamespace`. (REQUIREMENTS DOC-06.)
- **L-08:** Out of scope (REQUIREMENTS): recursion into inline `component.values:`, conditionals/loops/Jinja, cross-component variable references, external sources (Vault/SM/env), variables in `dependsOn`, recursive resolution, resolved-vars in `status`.

**Phase 18 fetch-error classification (D-01..D-05):**

- **D-01:** `load_values_from_reference` upgrades broad `except Exception → return {}` to typed dispatch:
  - `kopf.TemporaryError(delay=30)` when resource is missing (404/`NotFoundError`), connection error, timeout, or 5xx from kr8s.
  - `kopf.PermanentError` when call fails for auth/RBAC reasons or when `yaml.safe_load` raises.
  - `kopf.PermanentError` when `render()` raises `ValueError` on the raw string.
- **D-02:** Apply fetch-error upgrade to BOTH callsites (AppStack and helm-only). Helm path's silent-`{}` gets fixed as bonus. Variables non-wiring preserved.
- **D-03:** TemporaryError format: `"Component '{comp_name}' valuesFiles[{idx}]: {Kind} {ns}/{name} not found (will retry in 30s)"`.
- **D-04:** Render-failure PermanentError format: `"Component '{comp_name}' valuesFiles[{idx}]: undefined variable ${{{name}}} in {Kind} {ns}/{name}[{key}]"`. Chained `raise ... from e`.
- **D-05:** `load_values_from_reference` signature evolves to `(kind, name, key, namespace, variables: Optional[Dict[str, str]] = None, *, comp_name: Optional[str] = None, ref_index: Optional[int] = None)`. Helm-path callsite at line 923 NOT touched.

**Phase 18 README structure (D-06..D-10):**

- **D-06:** New top-level `## Variable substitution in AppStack manifests` between `## Common configuration` and `## Upgrading`.
- **D-07:** Worked example: AIDP-style multi-component (milvus + ingress) with `${namespace}` + `${milvusHost}` + valuesFiles.
- **D-08:** No-recursion presentation: `> **Note:**` callout + WRONG/CORRECT side-by-side snippets.
- **D-09:** DOC-06: explicit "omit `targetNamespace`" recommendation.
- **D-10:** Section ordering: (1) why → (2) syntax table → (3) worked example → (4) no-recursion callout → (5) operator-control callout → (6) error semantics.

**Phase 18 backward-compat snapshot (D-11..D-14):**

- **D-11:** Snapshot fixture `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml`. Phase 19 adds the portable variant.
- **D-12:** Snapshot asserts BOTH (a) merged Helm values dict per helm component AND (b) manifest tempfile string per kubernetesManifest component. Baselines under `operator_module/tests/snapshots/ai-research/`.
- **D-13:** Mocking strategy: `unittest.mock.patch` on `subprocess.run`, `kr8s.objects.ConfigMap.get` / `Secret.get`, `HelmOperator`. Tests run in-process via `conftest.py` sys.path injection.
- **D-14:** TST-05 shape: mock `load_values_from_reference`, invoke `handle_helm_deployment` with single-helm fixture, assert no `variables=` kwarg. Plus `inspect.getsource()` static check that body has no `render(`.

**Phase 18 helper extraction & wiring (D-15..D-18):**

- **D-15:** Extract `_render_or_raise(text, variables, *, source_desc) -> str` helper. Three callsites with distinct `source_desc` strings.
- **D-16:** Variables dict build at top of `handle_appstack_deployment` after `enabled_components` filter (~line 600), BEFORE `resolve_dependencies`. Validate every key with `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)`; defensive isinstance(v, str) check too.
- **D-17:** `field='spec'` filter as-spec'd at line 1053. Single-line change.
- **D-18:** Variables dict scope is stack-level. Component-level overrides out of scope.

### Claude's Discretion

- Exact placement of `_render_or_raise` helper inside `operator_module/main.py` (top-level, near `render()`).
- Exact phrasing of error messages within the locked formats (e.g., `"will retry in 30s"` vs `"retrying in 30s"`).
- Snapshot baseline file format (`values_<comp>.json` vs `values_<comp>.yaml`).
- Whether the README syntax table uses pipes or code-block layout.
- Pinning vs. building the regex pattern (`re.compile` once at module scope vs. inline `re.fullmatch`).
- Chart.yaml version bump in this phase (likely `0.1.62` → `0.1.63`).

### Deferred Ideas (OUT OF SCOPE)

- **Per-component variable overrides** (D-18 alternative).
- **`when=` predicate on @kopf.on.update** (`field='spec'` already prevents the storm).
- **`field='spec'` audit on @kopf.on.create / @kopf.on.delete** (out of scope for v5.0).
- **Refactor `handle_appstack_deployment` into pure `_prepare_component_artifacts` helper** — defer.
- **Snapshot test for `cluster_init/app-store-cluster-init.yaml`** — Phase 16's regression test already covers it.
- **Migration walkthrough section in README** — Phase 20's PR description handles this.
- **CI wiring for `operator_module/tests/`** — future test-infra phase.
- **`status.conditions[type=VariablesResolved]` observability field** — V51-02.
- **Default-value syntax `${VAR:-default}`** — V51-03.
- **Templating `targetNamespace`** — V51-01.
- **Bare `$identifier` mixing landmine** — Phase 16 deferred; PermanentError message already names the offending bare identifier.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OP-06 | `handle_appstack_deployment` builds variables dict once at stack scope as `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` | D-16 locks placement at top-of-function after `enabled_components` filter; CONTEXT.md lines 200-214 give the verbatim build snippet. |
| OP-07 | `kubernetesManifest` strings rendered before `kubectl apply`; render failures → `kopf.PermanentError` with variable + component context | D-15 `_render_or_raise` helper wraps `render()` and converts `ValueError` to `kopf.PermanentError`; insertion point line 779 (before `tempfile.NamedTemporaryFile`). |
| OP-08 | `load_values_from_reference` renders raw ConfigMap data string and base64-decoded Secret string before `yaml.safe_load` | D-15 helper called inside the function after fetch + before parse; CONTEXT.md lines 233-241 give the snippet. Secret base64 decode happens BEFORE render (so the decoded string is what gets templated). |
| OP-09 | `load_values_from_reference` signature uses `variables=None` default; helm callsite at line 923 NOT wired | D-05 keyword-only `variables`/`comp_name`/`ref_index` params with `None` defaults preserve helm path's positional 4-arg call. TST-05 locks the non-wiring (D-14). |
| OP-10 | Variable key names validated as Python identifiers when dict is built; invalid → `kopf.PermanentError` early | D-16 `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)` validation BEFORE deployment work. Defense in depth with Phase 17's CRD `propertyNames`. |
| OP-11 | `load_values_from_reference` fetch failures surface as `kopf.TemporaryError(delay=30)` instead of silent `{}` | D-01 typed exception dispatch. **kr8s 0.20.10 exception taxonomy mapped below in `## Standard Stack`.** Verified empirically — `NotFoundError` for 404, `ServerError` for other 4xx/5xx (status_code disambiguates), `APITimeoutError` for httpx timeout. |
| OP-12 | `@kopf.on.update` decorator gets `field='spec'` filter | D-17 single-line change at line 1053. **Verified: kopf 1.38.0 accepts `field='spec'` and docs confirm "restricts update handlers to cases where the specified field is affected" — status patches skip the handler entirely.** |
| TST-02 | New `operator_module/tests/test_appstack.py` covers substitution behavior — manifest path, valuesFiles path, `${namespace}` auto-default, explicit override | D-13 `unittest.mock.patch` on `subprocess.run`, `kr8s.objects.*.get`, `HelmOperator`. Existing `mcp-server/tests/test_apply_tool.py` is the canonical pattern in this repo. |
| TST-03 | Backward-compat snapshot — existing AppStack fixture without `variables:` produces byte-identical merged values dict and manifest tempfile content pre/post change | D-11/D-12 use `ai-research.yaml`. **CAVEAT: fixture uses snake_case field names that don't match operator's camelCase reads — see Open Questions for resolution.** Snapshot baseline format: hand-rolled `Path.read_text()` (existing repo pattern in `test_openclaw_config.py`). |
| TST-05 | Test locks `handle_helm_deployment` non-wiring (`variables=None` passes through; substitution does not run on single-chart path) | D-14 two-layered: mock-based call assertion + `inspect.getsource()` static grep. **Verified: `handle_helm_deployment` is NOT kopf-decorated (only `create_warrpappstore_function` and `update_warrpappstore_function` carry decorators), so `inspect.getsource()` returns the actual function body, not a wrapper.** |
| DOC-01 | README section explaining `${VAR}` syntax with worked example | D-06/D-07/D-10 lock placement, content shape, and ordering. |
| DOC-02 | README documents `$$` literal-dollar escape with password example | Goes in syntax table (D-10 step 2). Phase 16 confirms `$$ → $` works via stdlib semantics. |
| DOC-03 | README documents `${namespace}` auto-defaulting to CR's `metadata.namespace` | Goes in syntax table (D-10 step 2) AND worked example (D-10 step 3). |
| DOC-04 | README documents strict failure on undefined references (`kopf.PermanentError` with named variable + component) | Goes in error semantics (D-10 step 6). |
| DOC-05 | README documents that variable values are NOT recursively resolved; documented examples must use fully-resolved values | D-08 callout block + side-by-side WRONG/CORRECT (D-10 step 4). |
| DOC-06 | README documents that operator-control fields are NOT templated; recommends dropping `targetNamespace` | D-09 hard recommendation (D-10 step 5). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `${VAR}` substitution in `kubernetesManifest` strings | Operator (Python kopf reconciler in `operator_module/main.py`) | — | Manifest is operator-owned text; substitution must happen before `kubectl apply` is invoked from the operator's tempfile path. |
| `${VAR}` substitution in ConfigMap/Secret-sourced values | Operator (`load_values_from_reference`) | — | Operator fetches and parses the referenced resources; substitution happens between raw fetch and `yaml.safe_load`. |
| Variables dict build (`namespace` auto-default + key validation) | Operator (`handle_appstack_deployment` top) | CRD (Phase 17 admission) | Operator builds the runtime dict; CRD admission rejects bad keys/types preemptively. Defense in depth. |
| Fetch-error classification (Temporary vs Permanent) | Operator (kr8s exception → kopf exception mapping) | — | Only the operator runs inside kopf and can decide retry semantics. |
| Reconcile-storm prevention | Operator (`@kopf.on.update` decorator filter) | — | Filtering is a kopf-handler concern; status patches issued by the operator must not re-trigger it. |
| User-facing documentation | README.md (repo root) | CRD description field (Phase 17) | README is the discovery surface; CRD `description` is the in-cluster surface (`kubectl explain`). |
| Test verification | `operator_module/tests/` (pytest, in-process, mocked) | — | Operator is single-file; tests live next to it mirroring `mcp-server/tests/`. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `kopf` | `>=1.38.0` (installed: 1.38.0) [VERIFIED: `pip show kopf`] | Kubernetes operator framework; provides `@kopf.on.update`, `kopf.PermanentError`, `kopf.TemporaryError`, `field=` filter | Already a pinned operator dep in `operator_module/requirements.txt`. No version bump needed. `field=` filter accepts `'spec'` and is documented to restrict handler invocation to that field's changes [CITED: docs.kopf.dev/en/stable/filters/]. |
| `kr8s` | `>=0.17.0` (installed: 0.20.10) [VERIFIED: `pip show kr8s`] | Kubernetes API client used to fetch ConfigMap and Secret resources | Already pinned. Exception taxonomy (below) drives D-01 typed-error dispatch. |
| `pytest` | `>=8.0.0` (installed locally: 9.0.2) [VERIFIED: `pip show pytest`] | Test framework for new TST-02/TST-03/TST-05 files | Phase 16 pinned this in `operator_module/requirements-dev.txt`. No new dep. |
| `unittest.mock` | stdlib | Mock kr8s/subprocess/HelmOperator interactions | Stdlib pattern used in `mcp-server/tests/test_apply_tool.py`; no new dep. |

### Supporting (no new deps required)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `string.Template` | stdlib | Used by Phase 16 `render()` | No direct use in Phase 18 — wrapped via `_render_or_raise`. |
| `re` | stdlib | Validate variable key names against Python identifier pattern | Used in D-16 key-name pre-validation. |
| `inspect` | stdlib | `inspect.getsource()` for TST-05 static check | Used in TST-05; verified safe for non-decorated `handle_helm_deployment`. |
| `pathlib.Path` | stdlib | Read snapshot baselines via `Path.read_text()` | TST-03 hand-rolled snapshot pattern (mirrors `test_openclaw_config.py`). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled `Path.read_text()` snapshots (D-12) | `syrupy>=5.1.0` | Adds a dev dep + auto-generates `__snapshots__/*.ambr` files. Stronger affordance for "regenerate baseline" but unfamiliar to repo. Lowest-friction path matches existing pattern in `mcp-server/tests/test_openclaw_config.py`. |
| Hand-rolled snapshots | `pytest-snapshot>=0.9.0` | Similar tradeoff — explicit `--snapshot-update` flag is nice. Still adds a dep for a 3-test surface. Skip. |
| Hand-rolled snapshots | `inline-snapshot` | Inline expectation values inside test code. Powerful but inverts the locked baseline-file approach in D-12 ("Baseline files live in `operator_module/tests/snapshots/ai-research/`..."). Skip. |
| `unittest.mock.patch` decorator | `pytest-mock` (`mocker` fixture) | Cleaner syntax but adds a dev dep. Existing repo uses `unittest.mock` directly. Match. |
| Mocking strategy via `patch` | Refactor `handle_appstack_deployment` into pure `_prepare_component_artifacts` | Better testability but pulls a 200-line refactor into Phase 18. Already deferred per CONTEXT.md "Deferred Ideas". |

**Installation:** No new packages required. Existing `operator_module/requirements-dev.txt` (`pytest>=8.0.0`) is sufficient for all three new test files.

**Version verification:**
```bash
pip3 show pytest kopf kr8s     # already verified during research
npm view ...                   # N/A — Python project
```

## Architecture Patterns

### System Architecture Diagram

```
                    ┌────────────────────────────────────────────────────┐
                    │  WekaAppStore CR (admitted by Phase 17 CRD schema) │
                    └────────────────────┬───────────────────────────────┘
                                         │ create / update event
                                         ▼
       ┌──────────────────────────────────────────────────────────┐
       │  @kopf.on.update(..., field='spec')   ← OP-12 (D-17)    │
       │  @kopf.on.create(...)                                    │
       │  └─> dispatch by spec shape (appStack | helmChart | pod) │
       └──────────────────────────────────────┬───────────────────┘
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                ▼                             ▼                             ▼
   handle_appstack_deployment         handle_helm_deployment      handle_pod_deployment
   (Phase 18 wiring target)           (Phase 18 LOCKED non-wiring) (untouched)
   │                                  │
   │  1. build stack_vars              │  positional 4-arg call to
   │     ${namespace} default + user   │  load_values_from_reference
   │     vars; key+type validation     │  (variables defaults to None)
   │     OP-06 / OP-10 (D-16)          │
   │                                   │
   │  2. for each component:           │  fetch-error upgrade still applies
   │     a. helmChart branch           │  (D-02 — bonus fix on helm path)
   │        - per valuesFiles ref:     │
   │          load_values_from_ref(    │
   │            ..., variables=        │
   │              stack_vars,          │
   │            comp_name=...,         │
   │            ref_index=...)         │
   │            ↓                      │
   │            kr8s fetch CM/Secret   │
   │            ↓ (raw string)         │
   │            _render_or_raise(...)  │
   │            ↓ (rendered string)    │
   │            yaml.safe_load         │
   │            ↓ (dict)               │
   │            _deep_merge into       │
   │              merged_values        │
   │     b. kubernetesManifest branch  │
   │        - manifest_yaml =          │
   │          _render_or_raise(        │
   │            component['kubernetes  │
   │            Manifest'], stack_vars,│
   │            source_desc=...)       │
   │            ↓                      │
   │          tempfile + kubectl apply │
   │                                   │
   ▼                                   ▼
   helm install / kubectl apply        helm install
                                       
   ┌─────────────────────────────────────────────────────────┐
   │  Error paths (D-01..D-04):                              │
   │   - render() KeyError/ValueError → kopf.PermanentError  │
   │     via _render_or_raise (D-15)                         │
   │   - kr8s.NotFoundError → kopf.TemporaryError(delay=30)  │
   │   - kr8s.APITimeoutError / 5xx ServerError              │
   │       → kopf.TemporaryError(delay=30)                   │
   │   - kr8s.ServerError 4xx (401/403/422...)               │
   │       → kopf.PermanentError                             │
   │   - yaml.YAMLError → kopf.PermanentError                │
   └─────────────────────────────────────────────────────────┘
```

### Recommended Project Structure (additions only)
```
operator_module/
├── main.py                     # MODIFIED: add _render_or_raise; wire 3 callsites; field='spec'
└── tests/                       # exists from Phase 16
    ├── __init__.py             # exists
    ├── conftest.py             # exists (sys.path injection)
    ├── test_render.py          # exists (Phase 16)
    ├── test_appstack.py        # NEW: TST-02
    ├── test_helm_non_wiring.py # NEW: TST-05
    ├── test_backward_compat_snapshot.py  # NEW: TST-03
    └── snapshots/              # NEW directory
        └── ai-research/        # NEW
            ├── values_<comp>.json     # baseline merged Helm values
            └── manifest_<comp>.yaml   # baseline manifest tempfile content
README.md                       # MODIFIED: new section between "Common configuration" and "Upgrading"
weka-app-store-operator-chart/
└── Chart.yaml                  # MODIFIED: 0.1.62 → 0.1.63 (operator-image-affecting change)
```

### Pattern 1: `_render_or_raise` helper (D-15 locked)
**What:** Single helper wraps Phase 16 `render()`, catches both `KeyError` and `ValueError`, re-raises as `kopf.PermanentError` with caller-supplied context.
**When to use:** All three callsites — kubernetesManifest, ConfigMap valuesFile, Secret valuesFile.
**Example:**
```python
# Source: CONTEXT.md D-15 (locked)
def _render_or_raise(
    text: str,
    variables: Optional[Dict[str, str]],
    *,
    source_desc: str,
) -> str:
    """Render text with variables; convert KeyError/ValueError to kopf.PermanentError."""
    try:
        return render(text, variables)
    except (KeyError, ValueError) as e:
        raise kopf.PermanentError(f"{source_desc}: {e}") from e
```

### Pattern 2: kr8s exception → kopf exception dispatch (D-01)
**What:** Replace today's broad `except Exception → return {}` with typed dispatch that distinguishes "cluster wobble" (retry) from "bad CR" (fail loudly).
**When to use:** Inside `load_values_from_reference` around the kr8s `.get()` call (BOTH 4-arg and new variables-aware paths).
**Example:**
```python
# Source: kr8s 0.20.10 verified taxonomy (this research) + CONTEXT.md D-01..D-04
import kr8s

try:
    if kind == "ConfigMap":
        cm = kr8s.objects.ConfigMap.get(name=name, namespace=namespace)
        values_yaml = cm.data.get(key, "")
    elif kind == "Secret":
        secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
        import base64
        values_yaml = base64.b64decode(secret.data.get(key, "")).decode("utf-8")
    else:
        raise kopf.PermanentError(f"Unsupported valuesFiles kind: {kind}")
except kr8s.NotFoundError as e:
    # 404 — retry: cluster may not have applied the CM/Secret yet
    raise kopf.TemporaryError(
        f"Component '{comp_name}' valuesFiles[{ref_index}]: "
        f"{kind} {namespace}/{name} not found (will retry in 30s)",
        delay=30,
    ) from e
except kr8s.APITimeoutError as e:
    # httpx timeout — retry
    raise kopf.TemporaryError(
        f"Component '{comp_name}' valuesFiles[{ref_index}]: "
        f"timeout fetching {kind} {namespace}/{name} (will retry in 30s)",
        delay=30,
    ) from e
except kr8s.ServerError as e:
    # ALL other 4xx and 5xx come through here. status_code disambiguates.
    status = (e.response.status_code if e.response is not None else None)
    if status is not None and status >= 500:
        raise kopf.TemporaryError(
            f"Component '{comp_name}' valuesFiles[{ref_index}]: "
            f"API server error {status} (will retry in 30s)",
            delay=30,
        ) from e
    # 401/403/422/etc. — bad CR or RBAC; permanent
    raise kopf.PermanentError(
        f"Component '{comp_name}' valuesFiles[{ref_index}]: "
        f"API error fetching {kind} {namespace}/{name}: {e}"
    ) from e

# render BEFORE yaml.safe_load
if variables is not None:
    values_yaml = _render_or_raise(
        values_yaml,
        variables,
        source_desc=(
            f"Component '{comp_name}' valuesFiles[{ref_index}] "
            f"{kind} {namespace}/{name}[{key}]"
        ),
    )

try:
    return yaml.safe_load(values_yaml) or {}
except yaml.YAMLError as e:
    raise kopf.PermanentError(
        f"Component '{comp_name}' valuesFiles[{ref_index}]: "
        f"malformed YAML in {kind} {namespace}/{name}[{key}]: {e}"
    ) from e
```

### Pattern 3: Variables dict build with key + type validation (D-16 locked)
**What:** Build `stack_vars` once at the top of `handle_appstack_deployment`, BEFORE `resolve_dependencies`, with per-key Python identifier check and per-value `isinstance(str)` check.
**When to use:** Once, immediately after the `enabled_components` filter (~line 600).
**Example:**
```python
# Source: CONTEXT.md "Specific Ideas" (locked)
import re

raw_user_vars = app_stack.get('variables') or {}
for key in raw_user_vars:
    if not re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key):
        raise kopf.PermanentError(
            f"Invalid variable key {key!r}: must match Python identifier "
            f"syntax [_a-zA-Z][_a-zA-Z0-9]*"
        )
    if not isinstance(raw_user_vars[key], str):
        raise kopf.PermanentError(
            f"Invalid variable value for {key!r}: must be a string"
        )
stack_vars = {'namespace': namespace, **raw_user_vars}
```

### Pattern 4: pytest mocking with `unittest.mock.patch` (D-13)
**What:** Use `unittest.mock.patch` as a context manager OR decorator to swap `kr8s.objects.ConfigMap.get`, `subprocess.run`, and `HelmOperator.install_or_upgrade` during a single test.
**When to use:** Every test in `test_appstack.py`, `test_backward_compat_snapshot.py`, and `test_helm_non_wiring.py`.
**Example (mirrors `mcp-server/tests/test_apply_tool.py`):**
```python
# Source: existing repo pattern in mcp-server/tests/test_apply_tool.py
from unittest.mock import patch, MagicMock
import pytest

def test_kubernetes_manifest_substitutes_namespace(tmp_path):
    """OP-07 + OP-06: ${namespace} in manifest renders to CR namespace."""
    cr_spec = {
        'appStack': {
            'components': [{
                'name': 'ingress',
                'kubernetesManifest': 'metadata:\n  namespace: ${namespace}\n',
            }],
        }
    }
    captured_files = []

    def fake_run(cmd, *args, **kwargs):
        # cmd = ["kubectl", "apply", "-f", manifest_file, "-n", target_ns]
        manifest_file = cmd[3]
        captured_files.append(Path(manifest_file).read_text())
        m = MagicMock()
        m.returncode = 0
        m.stderr = ""
        m.stdout = ""
        return m

    with patch('main.subprocess.run', side_effect=fake_run):
        from main import handle_appstack_deployment
        handle_appstack_deployment(
            body={'spec': cr_spec},
            spec=cr_spec,
            name='test-cr',
            namespace='aidp-test',
            status={},
        )

    assert len(captured_files) == 1
    assert 'namespace: aidp-test' in captured_files[0]
    assert '${namespace}' not in captured_files[0]
```

### Pattern 5: TST-03 snapshot via `Path.read_text()` (D-12)
**What:** Capture `merged_values` (passed to `HelmOperator.install_or_upgrade`) and the manifest tempfile content; compare byte-by-byte against checked-in baseline files. Re-generate baselines only when `BASELINE_REGEN=1` env var is set.
**Example:**
```python
# Source: existing repo pattern in mcp-server/tests/test_openclaw_config.py + D-12 locked
import json, os
from pathlib import Path
from unittest.mock import patch, MagicMock

SNAPSHOTS = Path(__file__).parent / "snapshots" / "ai-research"

def _maybe_write_baseline(path: Path, content: str) -> None:
    if os.environ.get("BASELINE_REGEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

def test_ai_research_no_op_substitution(monkeypatch):
    fixture = (
        Path(__file__).resolve().parents[2]
        / "mcp-server" / "tests" / "fixtures"
        / "sample_blueprints" / "ai-research.yaml"
    )
    cr = yaml.safe_load(fixture.read_text())
    cr_spec = cr["spec"]
    cr_namespace = cr["metadata"]["namespace"]

    captured_values = {}  # comp_name -> values dict
    captured_manifests = {}  # comp_name -> manifest text

    # ... patches for HelmOperator, subprocess, kr8s.objects ...

    handle_appstack_deployment(
        body=cr, spec=cr_spec, name=cr["metadata"]["name"],
        namespace=cr_namespace, status={},
    )

    for comp_name, values in captured_values.items():
        baseline_path = SNAPSHOTS / f"values_{comp_name}.json"
        rendered = json.dumps(values, indent=2, sort_keys=True)
        _maybe_write_baseline(baseline_path, rendered)
        assert baseline_path.exists(), f"Missing baseline {baseline_path}"
        assert rendered == baseline_path.read_text(), (
            f"Baseline drift in {baseline_path}. "
            f"Re-run with BASELINE_REGEN=1 to regenerate."
        )
```

### Pattern 6: TST-05 two-layered non-wiring assertion (D-14)
**What:** (a) Mock `load_values_from_reference`, invoke `handle_helm_deployment`, assert the mock received a positional 4-arg call (no `variables=` kwarg). (b) Static check `'render(' not in inspect.getsource(handle_helm_deployment)`.
**Example:**
```python
# Source: CONTEXT.md D-14 (locked)
import inspect
from unittest.mock import patch

def test_handle_helm_deployment_does_not_pass_variables():
    """OP-09: helm path's load_values_from_reference call has no variables kwarg."""
    cr_spec = {
        "helmChart": {"name": "qdrant", "repository": "https://charts.qdrant.tech"},
        "valuesFiles": [{"kind": "ConfigMap", "name": "cm", "key": "v.yaml"}],
    }
    with patch("main.load_values_from_reference", return_value={}) as mock_load:
        with patch("main.HelmOperator") as mock_helm_cls:
            mock_helm_cls.return_value.install_or_upgrade.return_value = (True, "ok")
            mock_helm_cls.return_value.get_release_info.return_value = None
            from main import handle_helm_deployment
            handle_helm_deployment(
                body={"spec": cr_spec}, spec=cr_spec,
                name="x", namespace="ns", status={},
            )
    assert mock_load.called
    # Variables kwarg must NOT appear in any call to load_values_from_reference.
    for call in mock_load.call_args_list:
        assert "variables" not in call.kwargs, (
            f"load_values_from_reference called with variables kwarg "
            f"from helm path: {call.kwargs}"
        )

def test_handle_helm_deployment_source_has_no_render():
    """OP-09 static guard: handle_helm_deployment body must not contain 'render('."""
    from main import handle_helm_deployment
    src = inspect.getsource(handle_helm_deployment)
    assert "render(" not in src, (
        "handle_helm_deployment must not call render() — "
        "single-chart path is locked non-wiring per OP-09."
    )
```

### Anti-Patterns to Avoid
- **Catching `Exception` broadly in the new fetch dispatch:** Re-introduces the silent-`{}` regression. Always catch by named class.
- **Passing `variables=stack_vars` from `handle_helm_deployment`:** Locked non-wiring per L-06/OP-09. TST-05 will fail.
- **Using `safe_substitute` instead of `substitute` in `render()`:** Phase 16 already forbids this; substitution is strict by design (OP-02).
- **Writing the baseline file in the test if missing:** Always require explicit `BASELINE_REGEN=1` so accidental drift fails CI / verification.
- **Mocking at module-name level the way pytest-mock does (`mocker.patch('main.kr8s')`):** Use `unittest.mock.patch` to match repo style.
- **Adding `field='spec'` to `@kopf.on.create` or `@kopf.on.delete`:** Out of scope per Deferred Ideas.
- **Templating the dockerconfigjson `$oauthtoken` literal:** Phase 16's bare-`$identifier` mixing landmine. Worked example must NOT mix bare `$shellvar` with `${VAR}` in the same component string.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| `${VAR}` substitution | Custom regex parser | Phase 16 `render()` (already shipped) | Stdlib `string.Template` handles `$$`, identifier rules, escaping. |
| HTTP-status → exception classification | Custom kr8s wrapper | kr8s 0.20.10's `NotFoundError` / `ServerError` / `APITimeoutError` | kr8s already maps httpx errors to typed exceptions; `ServerError.response.status_code` disambiguates 4xx vs 5xx. |
| Reconcile-storm prevention | Custom `when=` predicate comparing `old != new` | `field='spec'` filter | kopf 1.38.0's documented intent. Single-line. Avoids paranoid double-filter. |
| Mock harness for kr8s | Custom kr8s test double | `unittest.mock.patch('main.kr8s.objects.ConfigMap.get')` | stdlib pattern; matches repo. |
| Snapshot diff library | Custom dict-diff | `Path.read_text()` byte equality + `BASELINE_REGEN` env var | Existing repo pattern (`test_openclaw_config.py`); reviewable git diff; no new dep. |
| Variable-key validation | Custom `kebab-to-snake` allow logic | `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)` | string.Template's documented identifier rule; matches Phase 17 CRD `propertyNames` pattern verbatim. |

**Key insight:** Phase 16 + Phase 17 already paid the engineering cost for the hard parts (substitution semantics, CRD admission rules). Phase 18 is plumbing — `_render_or_raise`, exception mapping, kopf decorator filter. Resist any temptation to introduce abstractions; the locked CONTEXT.md decisions specify ~30 lines of new code in `main.py` plus three test files.

## Common Pitfalls

### Pitfall 1: kr8s `ServerError` swallows BOTH 4xx and 5xx
**What goes wrong:** Naive `except kr8s.ServerError → kopf.PermanentError` would mark transient 503 / 502 as permanent failures.
**Why it happens:** kr8s 0.20.10 wraps every `httpx.HTTPStatusError` with status_code ≥ 400 into `ServerError` [VERIFIED: read kr8s/_api.py:174-201]. The class itself doesn't differentiate 4xx (bad CR / RBAC) from 5xx (cluster wobble).
**How to avoid:** Inspect `e.response.status_code` (it's a documented attribute on `ServerError`). 5xx → `TemporaryError(delay=30)`; everything else → `PermanentError`.
**Warning signs:** Operator log spam from one bad CR getting `delay=30` retries forever (means 4xx wrongly mapped to TemporaryError) — OR — operator marking healthy CRs as permanently failed during cluster API hiccups (means 5xx wrongly mapped to PermanentError).

### Pitfall 2: kr8s `.get()` swallows 404 and retries internally
**What goes wrong:** Test that mocks `kr8s.objects.ConfigMap.get` to "raise 404" by raising `ServerError(response.status_code=404)` will not match what the production code path sees.
**Why it happens:** `APIObject.async_get` catches 404 internally and loops with backoff for `timeout=2` seconds, then raises `NotFoundError` once exhausted [VERIFIED: read kr8s/_objects.py:301-304, 320-332].
**How to avoid:** Tests that simulate "missing ConfigMap" must `raise kr8s.NotFoundError("...")` — NOT a 404 ServerError. Production code should `except kr8s.NotFoundError` first, then `except kr8s.ServerError`.
**Warning signs:** Test passes locally but operator behavior differs in real cluster.

### Pitfall 3: `inspect.getsource()` returns wrapper source for decorated functions
**What goes wrong:** `inspect.getsource(decorated_handler)` returns the inner `def wrapper(*a, **kw):` body, not the original function. The TST-05 `'render(' not in src` check would pass falsely.
**Why it happens:** Python tracks the function object's `__code__` attribute; decorators that return a new function shadow the original [VERIFIED empirically with a 2-line test in this research].
**How to avoid:** **`handle_helm_deployment` is NOT decorated** (verified by `grep -n -E "^@kopf|^def handle_helm_deployment" main.py` — only the kopf entry handlers carry decorators). The TST-05 static check is safe AS-IS. If a future refactor wraps `handle_helm_deployment`, switch to AST-based inspection (`ast.parse(inspect.getsource(...))`) or check `inspect.unwrap()`.
**Warning signs:** TST-05 passes after a refactor that adds a decorator to `handle_helm_deployment` even though `render(` was added to the body.

### Pitfall 4: `string.Template.substitute` raises on bare `$identifier` mixed with `${VAR}`
**What goes wrong:** A manifest containing `${namespace}` AND a literal `$oauthtoken` (e.g., AIDP's `dockerconfigjson`) raises `KeyError('oauthtoken')` at `Template.substitute()` time [VERIFIED empirically].
**Why it happens:** `string.Template`'s pattern matches BOTH `$identifier` and `${identifier}` — there is no way to opt out of bare-identifier matching without subclassing.
**How to avoid:** Phase 18's `_render_or_raise` catches `KeyError` and re-raises `kopf.PermanentError` naming the offending variable. The README worked example MUST NOT mix bare `$oauthtoken` (or any non-substituted bare identifier) with `${VAR}` in the same component's manifest. AIDP migration (Phase 20) addresses this by escaping (`$$oauthtoken`) or pre-resolving.
**Warning signs:** PermanentError message names a "variable" the user never declared (e.g., `Undefined variable: ${oauthtoken}`). README error semantics section (D-10 step 6) should explicitly mention this.

### Pitfall 5: Empty-vars short-circuit hides bugs in `render()`
**What goes wrong:** A test that sends `variables={}` always returns text unchanged even if the template has `${UNDEF}`. Tests for malformed-placeholder paths must use `variables={'x': 'y'}` to bypass the empty-vars short-circuit.
**Why it happens:** `render()` checks `if not variables: return text` first (Phase 16 D-02 ordering).
**How to avoid:** Phase 16's existing tests use `{'x': 'y'}` for malformed-placeholder cases. Phase 18 inherits this. Document in test file comments.
**Warning signs:** A test asserts an error type but the production code path never raises because the test path took the short-circuit.

### Pitfall 6: `ai-research.yaml` fixture uses `snake_case` field names
**What goes wrong:** The fixture has `helm_chart`, `target_namespace`, `release_name`, etc. — but `handle_appstack_deployment` reads `helmChart`, `targetNamespace`, `releaseName`. Loading the fixture as-is would short-circuit through `else: raise ValueError(f"Component {comp_name} must specify either helmChart or kubernetesManifest")` at main.py:798.
**Why it happens:** The fixture was written for a different consumer (mcp-server validator?) using snake_case spec fields; the operator code is the canonical camelCase consumer.
**How to avoid:** Three options for the planner to pick:
  1. **Convert in-test** — `yaml.safe_load(...)` then map snake → camel before passing to `handle_appstack_deployment`. Keeps fixture untouched.
  2. **Fix the fixture** — rename fields in `ai-research.yaml` to camelCase. Risk: breaks any other consumer (check `mcp-server/tools/` references).
  3. **Add a different fixture** — author a new `operator_module/tests/fixtures/ai-research-camel.yaml` mirroring the CR in cluster-init shape. Most isolated.
**Warning signs:** TST-03 baselines come back empty / no helm components captured.
**Recommendation:** Option 1 (in-test conversion). Lowest risk; planner adds 5-line `_normalize_camel(d)` helper inside `test_backward_compat_snapshot.py`. See `## Open Questions` for the explicit decision needed.

### Pitfall 7: Existing call site at line 923 NOT touched by D-05 signature change
**What goes wrong:** A planner who refactors all `load_values_from_reference` callers to use keyword arguments breaks D-05's "helm-path callsite is NOT touched" invariant and trips TST-05's mock-based assertion.
**Why it happens:** Python keyword-only parameters are added without breaking positional callers — but a planner intent on consistency might "modernize" the call site.
**How to avoid:** D-05 explicitly: "The helm-path callsite at line 923 is NOT touched (still uses positional 4-arg call)." Plan tasks must call this out.
**Warning signs:** TST-05 fails: "load_values_from_reference called with variables kwarg from helm path."

## Code Examples

Verified patterns from official sources and existing repo code:

### kr8s exception taxonomy (verified 2026-05-08)
```python
# Source: kr8s 0.20.10 _api.py:174-201 + _objects.py:301-332 (read directly)
import kr8s

# When you call: kr8s.objects.ConfigMap.get(name="cm", namespace="ns")
#
# HTTP 404 (resource not in API server):
#   → call_api raises ServerError(status_code=404)
#   → APIObject.async_get catches it and retries until timeout=2s exhausts
#   → raises NotFoundError("Could not find ConfigMap cm in namespace ns.")
#
# HTTP 401 / 403:
#   → call_api auto-retries auth 3 times, then raises ServerError
#   → ServerError.response.status_code in (401, 403)
#
# HTTP 5xx:
#   → call_api raises ServerError
#   → ServerError.response.status_code in (500..599)
#
# httpx.TimeoutException:
#   → APITimeoutError("Timeout while waiting for the Kubernetes API server")
#
# (ConnectionClosedError is portforward-only; not relevant for .get())

# All exception classes are reachable as:
#   kr8s.NotFoundError
#   kr8s.ServerError
#   kr8s.APITimeoutError
#   kr8s.ConnectionClosedError
# (or via kr8s._exceptions; kr8s.* is the public surface)
```

### `field='spec'` decorator (verified 2026-05-08)
```python
# Source: docs.kopf.dev/en/stable/filters/ + empirical decorator construction
import kopf

@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')
def update_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    ...

# Behavior (per official docs):
#   "The field= filter restricts the update handlers to cases where the
#    specified field is affected in any way: changed, added, or removed."
#   "When the specified field is not affected but something else is changed,
#    such update handlers are not invoked even if they do match the field criteria."
#
# So: status patches issued by the operator itself do NOT re-trigger this handler.
# Reconcile-storm prevention is the documented intent.
```

### `string.Template` empirical error semantics
```python
# Source: empirical verification via python3 (this research session)
import string

# Undefined identifier → KeyError
string.Template('${UNDEF}').substitute({'x': 'y'})
# KeyError: 'UNDEF'

# Empty placeholder → ValueError
string.Template('bad: ${}').substitute({'x': 'y'})
# ValueError: Invalid placeholder in string: line 1, col 6

# Numeric placeholder → ValueError
string.Template('bad: ${123}').substitute({'x': 'y'})
# ValueError: Invalid placeholder in string: line 1, col 6

# Bare $ at end → ValueError
string.Template('end: $').substitute({'x': 'y'})
# ValueError: Invalid placeholder in string: line 1, col 6

# Bare $identifier mixing — KeyError on the bare reference
string.Template('${ns} and $oauthtoken').substitute({'ns': 'foo'})
# KeyError: 'oauthtoken'

# $$ literal-dollar escape works
string.Template('price is $$5').substitute({'x': 'y'})
# 'price is $5'

# Phase 16 render() catches both KeyError and ValueError (D-04, D-05).
# Phase 18 _render_or_raise catches them again and raises kopf.PermanentError.
```

### Existing repo `Path.read_text()` snapshot pattern
```python
# Source: mcp-server/tests/test_openclaw_config.py:47, 65, 167
on_disk_config = json.loads(OPENCLAW_JSON_PATH.read_text(encoding="utf-8"))
content = OPENCLAW_JSON_PATH.read_text(encoding="utf-8")
# Pattern: load baseline, compare via plain ==. No snapshot library.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Silent `except Exception → return {}` in `load_values_from_reference` | Typed kr8s → kopf exception dispatch | Phase 18 (this phase) | Bad CRs / RBAC fail loudly; cluster wobble retries; observability via `componentStatus.message`. |
| `@kopf.on.update` with no field filter — handler fires on every status patch | `@kopf.on.update(..., field='spec')` filter | Phase 18 (this phase) | Eliminates reconcile-storm caused by operator's own status writes. |
| Hardcoded namespace in `kubernetesManifest` and `valuesFiles` content | `${namespace}` auto-default + user-defined `${VAR}` | v5.0 (Phase 16-18) | One-CR portability across namespaces. |
| `safe_substitute` (silent on undefined) — common in older Python templating | `Template.substitute` strict mode | Phase 16 | Typos surface as `kopf.PermanentError` at apply time, not as cryptic K8s errors hours later. |

**Deprecated/outdated:**
- **None within this phase's scope.** Phase 16/17 already shipped; their decisions are State of the Art for v5.0.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `ai-research.yaml` snake_case field names will be addressed via in-test conversion (Pitfall 6 Option 1) | Open Questions | If planner picks Option 2 (rename fixture), `mcp-server/tools/validate_yaml.py` consumers may break. |
| A2 | Chart.yaml bumps `0.1.62 → 0.1.63` for Phase 18's operator image change | Standard Stack / Specifics | Phase 17 D-14 anticipated this exact bump; SemVer says patch-bump is correct for additive backward-compat; user discretion may pick differently if AIDP timing shifts. |
| A3 | No new runtime deps required for Phase 18 (only stdlib + existing pinned `kopf`/`kr8s`/`pytest`) | Installation | If a future planner adds `pytest-mock` or `syrupy`, `requirements-dev.txt` changes — neutral cost. |
| A4 | The kr8s `>=0.17.0` minimum still ships `NotFoundError`, `ServerError`, `APITimeoutError`, `ConnectionClosedError` with the same names | Code Examples | Verified at 0.20.10. If user runs production with kr8s 0.17.x, names should match (per kr8s changelog, the public exception API has been stable since 0.10+) but not re-verified. Production operator images already pull whatever pip resolves — bump `requirements.txt` to `kr8s>=0.20.0` if the planner wants to lock the verified surface. |

## Open Questions (RESOLVED)

1. **TST-03 fixture mismatch (snake_case vs camelCase) — needs decision before TST-03 implementation**
   - What we know: `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` uses snake_case (`helm_chart`, `target_namespace`, `release_name`, `crds_strategy`, `wait_for_ready`, `readiness_check`, `depends_on`). The operator's `handle_appstack_deployment` reads camelCase (`helmChart`, `targetNamespace`, `releaseName`, `crdsStrategy`, `waitForReady`, `readinessCheck`, `dependsOn`).
   - What's unclear: Whether the planner will fix the fixture in place (touching mcp-server validator consumers) or normalize in-test.
   - **RESOLVED:** Recommendation: **Option 1 — in-test conversion**. Add a 5-line `_normalize_camel(d)` helper at the top of `test_backward_compat_snapshot.py` that recursively renames keys via a snake→camel map (`helm_chart→helmChart`, etc.). Rationale: (a) lowest blast radius; (b) keeps the validator's accepted surface unchanged; (c) Phase 19 will add `ai-research-portable.yaml` as a sibling — if the fixture format is going to change shape there, do it in the new file rather than retrofitting the existing one.
   - Action: Planner should add this 5-line helper to TST-03, OR the discuss-phase tool can re-engage if the user wants Option 2/3.

2. **Snapshot baseline file format: JSON vs YAML for merged Helm values dict (D-12 Claude's Discretion)**
   - What we know: D-12 says values + manifest baselines live under `operator_module/tests/snapshots/ai-research/`. Choice between `values_<comp>.json` and `values_<comp>.yaml` is explicit Claude's discretion.
   - What's unclear: Which produces the most reviewable PR diff.
   - **RESOLVED:** Recommendation: **JSON with `indent=2, sort_keys=True`**. Rationale: deterministic ordering eliminates spurious diffs from Python dict ordering changes; YAML's anchors/aliases/flow style add diff noise; `json.dumps` is stdlib. Manifest content stays as YAML (it IS YAML).

3. **Baseline regeneration UX — pytest fixture vs env var**
   - What we know: D-12 requires explicit re-generation gate.
   - What's unclear: Whether `BASELINE_REGEN=1 pytest ...` env var or `--update-snapshots` pytest option is more discoverable.
   - **RESOLVED:** Recommendation: **`BASELINE_REGEN=1` env var** (no plugin needed). Document at top of `test_backward_compat_snapshot.py`. A pytest option would require `conftest.py` parser changes.

4. **kr8s minimum version pin in `operator_module/requirements.txt`**
   - What we know: Currently `kr8s>=0.17.0`. Verified surface is 0.20.10. Exception class names appear stable across that range (per kr8s public API discipline).
   - What's unclear: Whether 0.17.x raises `NotFoundError` in `.get()` or used a different mechanism. (No re-verification done.)
   - **RESOLVED:** Recommendation: **Bump to `kr8s>=0.20.0`** in this phase ONLY IF the planner finds evidence of behavior drift. Otherwise leave the existing pin. Out-of-scope churn risk.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python 3.10+ | All operator code | ✓ | 3.10 (system) [VERIFIED] | — |
| `pytest` | TST-02, TST-03, TST-05 | ✓ | 9.0.2 [VERIFIED via `pip show pytest`] | — |
| `kopf` | Operator runtime | ✓ | 1.38.0 [VERIFIED] | — |
| `kr8s` | Operator runtime | ✓ | 0.20.10 [VERIFIED] | — |
| `kubernetes` (k8s client) | Operator runtime helpers (CRD discovery) | ✓ | already in `requirements.txt` | — |
| `helm` CLI | NOT used by tests; only by production operator | N/A for tests | — | tests mock `HelmOperator` |
| `kubectl` CLI | NOT used by tests; only by production operator | N/A for tests | — | tests mock `subprocess.run` |
| AIDP repo (for worked example reference) | DOC-01..06 (read-only reference) | ✓ | `/Users/christopherjenkins/git/aidp` [VERIFIED via `ls`] | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 (locally), pinned `pytest>=8.0.0` in `operator_module/requirements-dev.txt` |
| Config file | None (no `pytest.ini` / `pyproject.toml [tool.pytest]` per Phase 16 D-09) |
| Quick run command | `pytest operator_module/tests/test_appstack.py operator_module/tests/test_helm_non_wiring.py -x` |
| Full suite command | `pytest operator_module/tests/` |
| Phase gate | Full operator_module test suite green before `/gsd-verify-work` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OP-06 | `${namespace}` auto-default in stack_vars | unit | `pytest operator_module/tests/test_appstack.py::test_namespace_auto_defaults_to_cr_namespace -x` | ❌ Wave 0 |
| OP-06 | Explicit user `namespace` override wins | unit | `pytest operator_module/tests/test_appstack.py::test_explicit_namespace_override_wins -x` | ❌ Wave 0 |
| OP-07 | `${VAR}` in `kubernetesManifest` renders before `kubectl apply` | unit (mock subprocess) | `pytest operator_module/tests/test_appstack.py::test_kubernetes_manifest_substitutes_namespace -x` | ❌ Wave 0 |
| OP-07 | Render failure on manifest → `kopf.PermanentError` with var + component | unit | `pytest operator_module/tests/test_appstack.py::test_undefined_variable_in_manifest_raises_permanent_error -x` | ❌ Wave 0 |
| OP-08 | `${VAR}` in ConfigMap valuesFile renders before `yaml.safe_load` | unit (mock kr8s) | `pytest operator_module/tests/test_appstack.py::test_configmap_valuesfile_substitutes_variables -x` | ❌ Wave 0 |
| OP-08 | `${VAR}` in Secret valuesFile renders after base64 decode | unit (mock kr8s) | `pytest operator_module/tests/test_appstack.py::test_secret_valuesfile_substitutes_variables -x` | ❌ Wave 0 |
| OP-09 | `handle_helm_deployment` does NOT pass `variables=` kwarg | unit (mock-based call assertion) | `pytest operator_module/tests/test_helm_non_wiring.py::test_handle_helm_deployment_does_not_pass_variables -x` | ❌ Wave 0 |
| OP-09 | `handle_helm_deployment` body does NOT contain `render(` | static (inspect.getsource) | `pytest operator_module/tests/test_helm_non_wiring.py::test_handle_helm_deployment_source_has_no_render -x` | ❌ Wave 0 |
| OP-10 | Hyphenated key (`my-host`) → `kopf.PermanentError` early | unit | `pytest operator_module/tests/test_appstack.py::test_invalid_variable_key_raises_permanent_error -x` | ❌ Wave 0 |
| OP-10 | Non-string variable value → `kopf.PermanentError` early | unit | `pytest operator_module/tests/test_appstack.py::test_non_string_variable_value_raises_permanent_error -x` | ❌ Wave 0 |
| OP-11 | Missing ConfigMap → `kopf.TemporaryError(delay=30)` | unit (mock kr8s.NotFoundError) | `pytest operator_module/tests/test_appstack.py::test_missing_configmap_raises_temporary_error -x` | ❌ Wave 0 |
| OP-11 | RBAC denial (kr8s.ServerError 403) → `kopf.PermanentError` | unit | `pytest operator_module/tests/test_appstack.py::test_rbac_denied_raises_permanent_error -x` | ❌ Wave 0 |
| OP-11 | API timeout (kr8s.APITimeoutError) → `kopf.TemporaryError(delay=30)` | unit | `pytest operator_module/tests/test_appstack.py::test_api_timeout_raises_temporary_error -x` | ❌ Wave 0 |
| OP-11 | Malformed YAML in CM → `kopf.PermanentError` | unit | `pytest operator_module/tests/test_appstack.py::test_malformed_yaml_raises_permanent_error -x` | ❌ Wave 0 |
| OP-12 | `@kopf.on.update` decorator carries `field='spec'` | static (source-grep) | `pytest operator_module/tests/test_appstack.py::test_update_handler_has_field_spec_filter -x` | ❌ Wave 0 |
| TST-02 | Above unit-test surface delivered in `test_appstack.py` | meta | (covered by all `test_appstack.py::*`) | ❌ Wave 0 |
| TST-03 | Backward-compat snapshot — `ai-research.yaml` byte-identical merged values + manifest | snapshot | `pytest operator_module/tests/test_backward_compat_snapshot.py -x` | ❌ Wave 0 |
| TST-05 | Two-layered non-wiring assertion | unit + static | `pytest operator_module/tests/test_helm_non_wiring.py -x` | ❌ Wave 0 |
| DOC-01..06 | README section content | manual (rendered Markdown review) | `grep -A 200 'Variable substitution in AppStack manifests' README.md \| head -200` | ❌ Wave 0 (README edit) |

### Sampling Rate
- **Per task commit:** `pytest operator_module/tests/test_appstack.py operator_module/tests/test_helm_non_wiring.py -x` (skips snapshot for speed; ~2-5s)
- **Per wave merge:** `pytest operator_module/tests/` (full operator suite, ~3-7s)
- **Phase gate:** Full suite green + `python -m py_compile operator_module/main.py` + visual README review

### Wave 0 Gaps
- [ ] `operator_module/tests/test_appstack.py` — covers OP-06, OP-07, OP-08, OP-10, OP-11, OP-12 (TST-02 surface)
- [ ] `operator_module/tests/test_helm_non_wiring.py` — covers OP-09 (TST-05 surface, two-layered)
- [ ] `operator_module/tests/test_backward_compat_snapshot.py` — covers TST-03
- [ ] `operator_module/tests/snapshots/ai-research/` — directory + `values_<comp>.json` + `manifest_<comp>.yaml` baselines (initial baselines generated via `BASELINE_REGEN=1 pytest -k snapshot`)
- [ ] No new framework install needed (pytest already pinned)
- [ ] No new conftest fixtures needed (Phase 16's sys.path injection covers all three new files)

## Project Constraints (from CLAUDE.md)

No `./CLAUDE.md` exists in the repo root. No project-specific overrides apply beyond CONTEXT.md.

## Sources

### Primary (HIGH confidence)
- `/Users/christopherjenkins/git/wekaappstore/.planning/phases/18-operator-wiring-and-docs/18-CONTEXT.md` — locked decisions D-01..D-18 (the spec).
- `/Users/christopherjenkins/git/wekaappstore/.planning/REQUIREMENTS.md` — phase requirement IDs OP-06..12, TST-02/03/05, DOC-01..06.
- `/Users/christopherjenkins/git/wekaappstore/.planning/PRD-appstack-variable-substitution.md` — source PRD ("Operator Change" section).
- `/Users/christopherjenkins/git/wekaappstore/.planning/STATE.md` — non-negotiable invariants.
- `/Users/christopherjenkins/git/wekaappstore/.planning/phases/16-render-helper-and-test-scaffolding/16-CONTEXT.md` + `16-VERIFICATION.md` — render() shipped contract.
- `/Users/christopherjenkins/git/wekaappstore/.planning/phases/17-crd-schema-additive-update/17-CONTEXT.md` — CRD admission rules.
- `/Users/christopherjenkins/git/wekaappstore/operator_module/main.py` — wiring sites at lines 390, 589, 710, 779, 871, 923, 1053.
- `/Users/christopherjenkins/git/wekaappstore/operator_module/tests/conftest.py`, `requirements-dev.txt` — Phase 16 test scaffolding (verified to support new test files).
- `/Users/christopherjenkins/pythonProjects/lib/python3.10/site-packages/kr8s/_api.py:174-216` — read directly to map HTTPStatusError → ServerError/APITimeoutError.
- `/Users/christopherjenkins/pythonProjects/lib/python3.10/site-packages/kr8s/_objects.py:280-332` — read directly to confirm 404 → NotFoundError after retry.
- [Kopf Filtering documentation](https://docs.kopf.dev/en/stable/filters/) — `field=` filter semantics quoted verbatim.
- [Kopf Updating the objects](https://docs.kopf.dev/en/stable/walkthrough/updates/) — `@kopf.on.update` reference.
- Empirical Python 3.10 `string.Template` invocations performed in this research — error types for `${}`, `${123}`, bare `$`, mixed `${VAR}`+`$identifier`.
- Empirical kopf 1.38.0 `kopf.on.update(...field='spec')` decorator construction — accepts the kwarg without error.

### Secondary (MEDIUM confidence)
- `/Users/christopherjenkins/git/wekaappstore/mcp-server/tests/test_apply_tool.py` — canonical `unittest.mock.MagicMock` pattern used in the repo.
- `/Users/christopherjenkins/git/wekaappstore/mcp-server/tests/test_openclaw_config.py` — canonical `Path.read_text()` snapshot/baseline pattern.
- `/Users/christopherjenkins/git/aidp/appstack/weka-aidp-appstack.yaml` — real AIDP CR shape (1198 lines) referenced for the README worked example structure.
- [Syrupy snapshot plugin](https://github.com/syrupy-project/syrupy) — alternative considered, not chosen.
- [pytest-snapshot](https://pypi.org/project/pytest-snapshot/) — alternative considered, not chosen.
- [inline-snapshot](https://pypi.org/project/inline-snapshot/) — alternative considered, not chosen (inverts D-12's baseline-file model).

### Tertiary (LOW confidence)
- None — all critical claims were verifiable via direct package source reads, official docs, or empirical Python invocation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already pinned and version-verified locally; no new deps.
- Architecture: HIGH — CONTEXT.md locks the architecture in D-15..D-18; this research only fills exception-mapping detail.
- Pitfalls: HIGH — six pitfalls verified empirically (kr8s 4xx-vs-5xx, kr8s `.get()` 404 retry, `inspect.getsource` decorator hazard, `string.Template` mixing, empty-vars short-circuit, fixture snake_case mismatch). Pitfall 6 (fixture) is a meaningful blocker that needs an explicit planner decision.
- README structure: MEDIUM — D-06..D-10 lock the section ordering and content; the worked example's exact prose is Claude's discretion within the locked frame.
- Snapshot strategy: HIGH — repo already uses `Path.read_text()` for analogous baseline-file comparisons; matches D-13's "no new test deps" implicit constraint.

**Research date:** 2026-05-08
**Valid until:** 2026-06-07 (30 days — kopf 1.38.0 and kr8s 0.20.10 surfaces are stable; re-verify if kr8s major bumps)

## RESEARCH COMPLETE

**Phase:** 18 - Operator Wiring and Docs
**Confidence:** HIGH

### Key Findings
1. **kr8s 0.20.10 exception taxonomy is mapped** — `NotFoundError` (404 after `.get()` retry exhausted), `ServerError` (other 4xx + 5xx; disambiguate via `e.response.status_code`), `APITimeoutError` (httpx timeout), `ConnectionClosedError` (portforward only). D-01 typed dispatch is now fully implementable.
2. **kopf 1.38.0 accepts `field='spec'`** and the docs verbatim confirm "restricts the update handlers to cases where the specified field is affected" — status patches skip the handler. D-17 is a single-line change.
3. **Snapshot strategy: hand-rolled `Path.read_text()`** — repo already uses this pattern in `test_openclaw_config.py`; no new dev dep needed; `BASELINE_REGEN=1` env-var gate for re-generation.
4. **`inspect.getsource(handle_helm_deployment)` is safe** — verified that the function is NOT kopf-decorated; only entry handlers `create_warrpappstore_function` and `update_warrpappstore_function` carry `@kopf.on.*`. TST-05 static check works as locked.
5. **TST-03 fixture caveat** — `ai-research.yaml` uses snake_case field names that the operator does not consume. Planner needs an explicit decision (recommended: in-test snake→camel normalization, ~5 lines). This is the single open-ended item in an otherwise fully-locked phase.

### File Created
`/Users/christopherjenkins/git/wekaappstore/.planning/phases/18-operator-wiring-and-docs/18-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All libraries pinned, version-verified locally, no new deps. |
| Architecture | HIGH | CONTEXT.md locks 18 decisions; research only fills exception-class names. |
| Pitfalls | HIGH | All six pitfalls empirically verified (kr8s exception flow, decorator hazard, Template semantics, fixture mismatch). |
| README Structure | MEDIUM | Outline locked (D-06..D-10); prose is Claude's discretion within the frame. |
| Snapshot Strategy | HIGH | Existing repo pattern + locked D-12 baseline-file shape. |

### Open Questions
1. TST-03 fixture snake_case vs camelCase mismatch — **needs planner decision** (recommended: in-test normalization).
2. Snapshot baseline format JSON vs YAML — Claude's discretion (recommended: JSON with `sort_keys=True`).
3. Baseline regeneration UX — Claude's discretion (recommended: `BASELINE_REGEN=1` env var).
4. kr8s minimum-version bump in `requirements.txt` — out-of-scope unless drift evidence found.

### Ready for Planning
Research complete. Planner can now create PLAN.md files. The research surfaces one decision item (TST-03 fixture handling) that the planner should resolve in PLAN.md task definitions before implementation begins.
