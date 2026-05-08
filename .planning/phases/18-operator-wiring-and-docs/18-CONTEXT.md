# Phase 18: Operator Wiring and Docs - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the Phase 16 `render()` helper into the two real substitution sites in `operator_module/main.py`:

1. **`handle_appstack_deployment` (line 589)** — render `kubernetesManifest` strings before `kubectl apply`; pass a stack-scope `variables` dict to `load_values_from_reference` for each `valuesFiles` entry.
2. **`load_values_from_reference` (line 390)** — render the raw ConfigMap data string (and base64-decoded Secret string) BEFORE `yaml.safe_load`. Add `variables=None` default so the existing `handle_helm_deployment` callsite (line 923) keeps working unchanged.

Add: stack-scope variables dict build with `${namespace}` auto-default, key-name validation (Python identifier — belt + suspenders with Phase 17's CRD admission), `kopf.TemporaryError(delay=30)` on fetch failures (replaces today's silent `{}` return), `kopf.PermanentError` with named-variable + component context on render failures, and `field='spec'` filter on the `@kopf.on.update` decorator (line 1053) to prevent reconcile storms.

Update README with a new top-level `## Variable substitution in AppStack manifests` section (placed between "Common configuration" and "Upgrading"). Lock `handle_helm_deployment` non-wiring with a mock-based test (TST-05) and the AppStack no-op behavior with a backward-compat snapshot test (TST-03) using the existing `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` fixture.

**Requirements covered:** OP-06, OP-07, OP-08, OP-09, OP-10, OP-11, OP-12, TST-02, TST-03, TST-05, DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06.

</domain>

<decisions>
## Implementation Decisions

### Carried forward (locked by Phase 16/17 — DO NOT re-litigate)

- **L-01:** `render()` lives in `operator_module/main.py` (Phase 16 D-08); raises `ValueError` on undefined or malformed placeholder (Phase 16 D-04..06); pre-scan guard returns text unchanged when no `${` present (Phase 16 D-01).
- **L-02:** Variables dict shape is `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` — user-supplied `namespace` key wins over auto-default. (REQUIREMENTS OP-06 verbatim.)
- **L-03:** Key-name pattern is `^[_a-zA-Z][_a-zA-Z0-9]*$` (Python identifier). Phase 17 D-02 enforces this at CRD admission; Phase 18 OP-10 enforces it again at the operator (catches API-server bypass). Belt + suspenders.
- **L-04:** Single-pass substitution only; values are taken literally. README must NOT show `milvusHost: milvus.${namespace}.svc.cluster.local` (the broken PRD example). Use fully-resolved values.
- **L-05:** Both `KeyError` AND `ValueError` are caught at every `render()` call site (`Template.substitute()` raises `ValueError` for malformed placeholders like `${}` or `${123}`). STATE.md invariant.
- **L-06:** `handle_helm_deployment` (single-chart path) MUST NOT receive variables wiring; `load_values_from_reference` keeps `variables=None` default. TST-05 locks this. (REQUIREMENTS OP-09.)
- **L-07:** Operator-control fields (`helmChart.*`, `releaseName`, `targetNamespace`, `readinessCheck.*`) are NOT templated. README must explicitly call this out and recommend dropping `targetNamespace`. (REQUIREMENTS DOC-06.)
- **L-08:** Out of scope (REQUIREMENTS): recursion into inline `component.values:`, conditionals/loops/Jinja, cross-component variable references, external sources (Vault/SM/env), variables in `dependsOn`, recursive resolution, resolved-vars in `status`.

### Fetch-error classification (OP-11)

- **D-01:** `load_values_from_reference` upgrades the today's broad `except Exception → return {}` to typed dispatch:
  - **`kopf.TemporaryError(delay=30)`** when the resource is missing (404/`NotFoundError`), connection error, timeout, or 5xx from kr8s.
  - **`kopf.PermanentError`** when the call fails for auth/RBAC reasons or when `yaml.safe_load` raises (malformed YAML in the resource).
  - **`kopf.PermanentError`** when `render()` raises `ValueError` on the raw string (undefined or malformed `${VAR}`).
  - Rationale: "cluster wobble → retry; bad CR → fail loudly." Auth/RBAC retrying forever wastes operator time and hides misconfig.

- **D-02:** Apply the fetch-error upgrade to BOTH callsites — the AppStack path (line 710) AND the helm-only path (line 923). The helm path's pre-existing silent-`{}` behavior gets fixed as a bonus. Aligns with OP-11 wording ("fetch failures surface as TemporaryError"). The non-wiring of `variables` is preserved (helm path passes nothing — defaults to `None` — pre-scan guard short-circuits, no rendering).

- **D-03:** **TemporaryError message format:** `"Component '{comp_name}' valuesFiles[{idx}]: {Kind} {ns}/{name} not found (will retry in 30s)"`. AppStack path supplies `comp_name` and `idx`; helm path supplies `releaseName` (or spec name) and `idx`. Names the offending component and the valuesFiles index for fast debugging.

- **D-04:** **Render-failure (PermanentError) message format inside `load_values_from_reference`:** `"Component '{comp_name}' valuesFiles[{idx}]: undefined variable ${{{name}}} in {Kind} {ns}/{name}[{key}]"`. Names the variable, the component, the resource, and the key inside the resource (CMs/Secrets are multi-key — important for AIDP). Uses chained `raise ... from e`.

- **D-05:** **Backward-compat at the API surface:** `load_values_from_reference` signature evolves to `(kind, name, key, namespace, variables: Optional[Dict[str, str]] = None, *, comp_name: Optional[str] = None, ref_index: Optional[int] = None) -> Dict[str, Any]`. The two new params are keyword-only. The helm-path callsite at line 923 is NOT touched (still uses positional 4-arg call). The AppStack callsite at line 710 passes `variables=stack_vars, comp_name=comp_name, ref_index=idx`.

### README structure (DOC-01..06)

- **D-06:** New top-level section `## Variable substitution in AppStack manifests` placed between `## Common configuration` and `## Upgrading` in `README.md`. Discoverable in the TOC, doesn't bury the feature.

- **D-07:** Worked example is a single AIDP-style multi-component AppStack CR with `${namespace}` + `${milvusHost}` + a `valuesFiles` reference. Two components minimum (e.g., `milvus` helmChart with valuesFiles, `ingress` kubernetesManifest with `namespace: ${namespace}`). Mirrors the real AIDP migration so authors can copy-paste-adapt.

- **D-08:** **DOC-05 no-recursion presentation:** A `> **Note:** Variable values are taken literally — no recursive resolution.` callout block, IMMEDIATELY followed by a side-by-side `# WRONG (this does not work)` snippet showing `milvusHost: milvus.${namespace}.svc.cluster.local` and a `# CORRECT (use fully-resolved values)` snippet showing `milvusHost: milvus.aidp-prod.svc.cluster.local`. Highest signal — readers cannot miss it.

- **D-09:** **DOC-06 hard recommendation:** README states explicitly: "Recommendation: omit `targetNamespace` and let the operator default to `metadata.namespace`. Templating is not supported on this field; setting it pins the component to a specific namespace and defeats the portability the variables block provides." Strong nudge toward the portable pattern.

- **D-10:** Section ordering inside the new section: (1) one-paragraph why → (2) syntax reference table (`${VAR}`, `$$`, `${namespace}`) → (3) worked example CR → (4) no-recursion callout (D-08) → (5) operator-control-fields callout (D-09) → (6) error semantics (`kopf.PermanentError` on undefined; `kopf.TemporaryError` on missing CM/Secret).

### Backward-compat snapshot (TST-03 / OP-06)

- **D-11:** Snapshot fixture is `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` — already in tree, multi-component, no `variables:` block. Phase 19 will add `ai-research-portable.yaml` as a sibling (with variables) for VAL-01..05 / TST-04 — they pair naturally.

- **D-12:** Snapshot asserts BOTH (a) the merged Helm values dict per helm component AND (b) the manifest tempfile string per `kubernetesManifest` component, byte-identical to a captured baseline. Locks the entire substitution path as a no-op when no user variables are supplied. Baseline files live in `operator_module/tests/snapshots/ai-research/{values_<comp>.json, manifest_<comp>.yaml}` so regressions are reviewable in PRs.

- **D-13:** Mocking strategy: `unittest.mock.patch` on `subprocess.run` (kubectl), `kr8s.objects.ConfigMap.get` / `kr8s.objects.Secret.get`, and `HelmOperator` (capture `install`/`upgrade` call args). Mocks return safe defaults (success returncode for kubectl; minimal CMs that round-trip through render() unchanged when no vars). Test runs purely in-process via existing `operator_module/tests/conftest.py` `sys.path` injection.

- **D-14:** **TST-05 (handle_helm_deployment non-wiring) test shape:** Mock-based call assertion. Patch `load_values_from_reference`, invoke `handle_helm_deployment` with a single-helm CR fixture (containing a `valuesFiles` ref so the function actually gets called), and assert the mock was called WITHOUT a `variables=` kwarg (positional 4-arg call). Plus an `inspect.getsource(handle_helm_deployment)` static check that the function body does not contain the string `render(`. Two-layered safety net.

### Helper extraction & wiring style

- **D-15:** Extract one helper: `_render_or_raise(text: str, variables: Optional[Dict[str, str]], *, source_desc: str) -> str`. Behavior: pre-scan guard short-circuit; call `render(text, variables)` inside `try/except (KeyError, ValueError)`; on catch, re-raise `kopf.PermanentError(f"{source_desc}: {error}") from error`. Three call sites pass distinct `source_desc` strings:
  - `kubernetesManifest`: `f"Component '{comp_name}'.kubernetesManifest"`
  - ConfigMap render inside `load_values_from_reference`: `f"Component '{comp_name}' valuesFiles[{ref_index}] ConfigMap {namespace}/{name}[{key}]"` (when `comp_name` is set; else generic descriptor)
  - Secret render inside `load_values_from_reference`: `f"Component '{comp_name}' valuesFiles[{ref_index}] Secret {namespace}/{name}[{key}]"`
  
  Single error-format definition; ~10 lines of helper, ~3 lines per call site.

- **D-16:** Variables dict build site is the **top of `handle_appstack_deployment` immediately after the `enabled_components` filter (~line 600)**, BEFORE `resolve_dependencies` is called. Order:
  1. Build `stack_vars = {'namespace': namespace, **(app_stack.get('variables') or {})}`.
  2. Validate every key name with `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)` — on first invalid key, raise `kopf.PermanentError(f"Invalid variable key {key!r}: must match Python identifier syntax [_a-zA-Z][_a-zA-Z0-9]*")` BEFORE any deployment work starts.
  3. (Optional but recommended) Validate that every value is a string — non-string values would already be blocked by Phase 17's CRD `additionalProperties: { type: string }`, but defensive `if not isinstance(v, str): raise kopf.PermanentError(...)` handles the edge case where a CR was admitted by an older CRD or via raw API write.
  
  Pre-validation guarantees no partial deployment if keys are bad.

- **D-17:** **OP-12 `field='spec'` filter:** Apply as-spec'd at line 1053: `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')`. Single-line change. No additional `when=` predicate; field='spec' is sufficient and matches kopf's documented intent.

- **D-18:** **Variables dict scope is stack-level, not per-component.** Every component (helm and kubernetesManifest) sees the same `stack_vars`. Component-level variable overrides are out of scope (would require a precedence rule + new schema work). REQUIREMENTS confirms this with OP-06 verbatim.

### Claude's Discretion

- Exact placement of `_render_or_raise` helper inside `operator_module/main.py` (top-level, near `render()`).
- Exact phrasing of error messages within the formats locked above (e.g., `"will retry in 30s"` vs `"retrying in 30s"` — pick whichever sounds best).
- Snapshot baseline file format (`values_<comp>.json` vs `values_<comp>.yaml`) — pick whichever produces the most readable diff in PR review.
- Whether the README syntax table uses `|`-pipes or a code-block layout — pick whichever renders best on GitHub.
- Pinning vs. building the regex pattern (`re.compile` once at module scope vs. inline `re.fullmatch`) — micro-perf decision.
- Chart.yaml version bump in this phase (likely `0.1.62` → `0.1.63`) — Phase 17 D-14 anticipated this; bump if any operator-image-affecting change ships.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Specs
- `.planning/PRD-appstack-variable-substitution.md` — Source PRD; "Operator Change" section defines the wiring sites; Acceptance Criteria 1–7 are Phase 18's behavior contract.
- `.planning/REQUIREMENTS.md` — Phase 18 requirements OP-06..12, TST-02, TST-03, TST-05, DOC-01..06; Out-of-Scope table is non-negotiable.
- `.planning/ROADMAP.md` §"Phase 18: Operator Wiring and Docs" — 5 success criteria.
- `.planning/STATE.md` §"Key Architectural Decisions (v5.0)" — pre-scan guard, single-pass, both KeyError AND ValueError caught, handle_helm_deployment non-wiring.

### Prior phase deliverables (must be aware of)
- `.planning/phases/16-render-helper-and-test-scaffolding/16-CONTEXT.md` — locked decisions D-01..14 (render() shape, error types, test scaffolding).
- `.planning/phases/16-render-helper-and-test-scaffolding/16-VERIFICATION.md` — confirms render() signature and contract Phase 18 wires into.
- `.planning/phases/17-crd-schema-additive-update/17-CONTEXT.md` — locked decisions D-01..16 (CRD schema, key pattern, version bump cadence).
- `.planning/phases/17-crd-schema-additive-update/17-VERIFICATION.md` — confirms CRD admits the `variables:` block and rejects bad keys/types.

### Code to Touch
- `operator_module/main.py:390` — `load_values_from_reference` (signature change + render + typed error dispatch).
- `operator_module/main.py:589` — `handle_appstack_deployment` (variables dict build at top; render manifest before tempfile; pass variables to `load_values_from_reference`).
- `operator_module/main.py:710` — AppStack `valuesFiles` callsite of `load_values_from_reference` (pass variables/comp_name/ref_index).
- `operator_module/main.py:765–796` — kubernetesManifest deployment block (render before tempfile write at line 779).
- `operator_module/main.py:1053` — `@kopf.on.update` decorator (add `field='spec'`).
- `operator_module/tests/test_appstack.py` — NEW (TST-02 coverage: substitution behavior on manifest path, valuesFiles path, ${namespace} auto-default, explicit override).
- `operator_module/tests/test_helm_non_wiring.py` — NEW (TST-05 mock + inspect-based static check).
- `operator_module/tests/test_backward_compat_snapshot.py` — NEW (TST-03 byte-identical snapshot).
- `operator_module/tests/snapshots/ai-research/` — NEW directory with baseline `values_<comp>.json` and `manifest_<comp>.yaml` files.
- `README.md` — NEW top-level section `## Variable substitution in AppStack manifests` between "Common configuration" and "Upgrading".
- `weka-app-store-operator-chart/Chart.yaml:18` — version bump `0.1.62` → `0.1.63` (operator image change).

### Code to Reference (do NOT modify)
- `operator_module/main.py:871` — `handle_helm_deployment` (single-chart path; locked non-wiring per L-06; TST-05 asserts).
- `operator_module/main.py:923` — `load_values_from_reference` callsite from helm path (positional 4-arg call; do NOT change to keyword-args here).
- `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` — TST-03 fixture (read-only).
- `cluster_init/app-store-cluster-init.yaml` — backward-compat reference (Phase 16's regression test already covers this).

### Patterns to Mirror
- `mcp-server/tests/conftest.py` — sys.path injection (already mirrored by `operator_module/tests/conftest.py` from Phase 16).
- `mcp-server/tests/fixtures/sample_blueprints/` — fixture organization for portable variant in Phase 19.

### Codebase Maps
- `.planning/codebase/STRUCTURE.md` — directory layout; operator lives at `operator_module/main.py`.
- `.planning/codebase/TESTING.md` — confirms no CI exists; Phase 18 adds three new test files but does NOT add CI scaffolding.
- `.planning/codebase/CONVENTIONS.md` — type-hint and error-handling conventions used elsewhere in main.py.

### Operational Reference
- `README.md` — current top-level structure (Prerequisites → Quick start → Common configuration → Upgrading → Uninstalling → Troubleshooting → Readiness checks → Publishing). New variable-substitution section inserts between "Common configuration" and "Upgrading".

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `render()` (Phase 16) — already exists in `operator_module/main.py`; Phase 18 only consumes it via the new `_render_or_raise` wrapper.
- `load_values_from_reference` (line 390) — extends in place; signature change is backward-compatible via keyword-only new params + default None.
- `_deep_merge` (already used at line 716) — Phase 18 changes nothing here; the merge happens AFTER render+yaml.safe_load returns the parsed dict.
- `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` — reused as TST-03 fixture; no changes needed.
- `kopf.PermanentError` and `kopf.TemporaryError` — already imported and used elsewhere in main.py (e.g., line 597, 620); Phase 18 consumes existing imports.

### Established Patterns
- Type hints: `Dict[str, Any]`, `Optional[...]` (consistent throughout main.py). `_render_or_raise` uses `Dict[str, str]` for the variables map (string-only by CRD constraint).
- Error wrapping idiom: `raise kopf.PermanentError(...) from e` (chained exceptions) — D-15 follows this.
- Per-component split tests/`requirements-dev.txt` — Phase 16 established this for `operator_module/tests/`; Phase 18 only adds new test files in this directory, no new infra.
- `subprocess.run` for kubectl — line 785; Phase 18's TST-03 mocks this rather than refactoring.

### Integration Points
- Phase 17's CRD admission rejects non-string variable values and invalid key names BEFORE the operator sees the CR. Phase 18 still validates at the operator (defense in depth — handles raw API writes / apiserver bypass).
- Phase 19 will read CRs with `variables:` blocks via the validator. The schema admission contract is locked by Phase 17; Phase 19 adds soft-warning UX only. Phase 18's `_render_or_raise` is NOT consumed by the validator.
- Phase 20 (AIDP) consumes the wired-up operator end-to-end. The README example in Phase 18 is the canonical migration template AIDP authors will copy.

### Codebase Constraints
- No CI today (per `.planning/codebase/TESTING.md`). Tests are run manually: `pytest operator_module/tests/`. Phase 18 adds tests but does not add CI scaffolding.
- `operator_module/requirements.txt` is intentionally minimal (3 deps) for the operator container image. Phase 16 added `requirements-dev.txt` for pytest; Phase 18 does not add new runtime deps. New test deps (e.g., for snapshot comparison) — if any — go in `requirements-dev.txt`.
- The kr8s library's exact exception class names are version-dependent. Plan should `grep` the existing main.py and `mcp-server/tools/` for the canonical kr8s exception names already in use, then map to TemporaryError/PermanentError accordingly.

</code_context>

<specifics>
## Specific Ideas

- **Helper signature (locked):**
  ```python
  def _render_or_raise(
      text: str,
      variables: Optional[Dict[str, str]],
      *,
      source_desc: str,
  ) -> str:
      """Render text with variables; convert ValueError to kopf.PermanentError."""
      try:
          return render(text, variables)
      except (KeyError, ValueError) as e:
          raise kopf.PermanentError(f"{source_desc}: {e}") from e
  ```

- **Variables dict build (locked, place at top of handle_appstack_deployment):**
  ```python
  raw_user_vars = app_stack.get('variables') or {}
  for key in raw_user_vars:
      if not re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key):
          raise kopf.PermanentError(
              f"Invalid variable key {key!r}: must match Python identifier syntax [_a-zA-Z][_a-zA-Z0-9]*"
          )
      if not isinstance(raw_user_vars[key], str):
          raise kopf.PermanentError(
              f"Invalid variable value for {key!r}: must be a string"
          )
  stack_vars = {'namespace': namespace, **raw_user_vars}
  ```

- **kubernetesManifest render (locked, place at line ~779 before tempfile.NamedTemporaryFile):**
  ```python
  manifest_yaml = _render_or_raise(
      manifest_yaml,
      stack_vars,
      source_desc=f"Component '{comp_name}'.kubernetesManifest",
  )
  ```

- **load_values_from_reference rewrite (locked):**
  - Add `variables=None`, `comp_name=None`, `ref_index=None` keyword-only params.
  - Replace broad `except Exception → return {}` with typed dispatch:
    - `kr8s.NotFoundError` (or equivalent — plan must confirm exact class) → `kopf.TemporaryError(delay=30, msg=f"Component '{comp_name}' valuesFiles[{ref_index}]: {kind} {namespace}/{name} not found (will retry in 30s)")`.
    - Connection/timeout/5xx → `kopf.TemporaryError(delay=30, ...)` similar.
    - Auth/RBAC → `kopf.PermanentError(...)`.
    - `yaml.YAMLError` → `kopf.PermanentError(...)`.
  - After the raw fetch succeeds, BEFORE `yaml.safe_load`:
    ```python
    if variables is not None:
        values_yaml = _render_or_raise(
            values_yaml,
            variables,
            source_desc=f"Component '{comp_name}' valuesFiles[{ref_index}] {kind} {namespace}/{name}[{key}]",
        )
    return yaml.safe_load(values_yaml) or {}
    ```

- **field='spec' filter (locked):** `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')` at line 1053.

- **README section skeleton (locked outline; planner fills prose):**
  ```markdown
  ## Variable substitution in AppStack manifests
  
  Brief why-paragraph.
  
  ### Syntax
  | Syntax | Behavior |
  | --- | --- |
  | `${VAR}` | Substituted with the value of `VAR` from `spec.appStack.variables`. |
  | `$$` | Literal dollar sign. |
  | `${namespace}` | Auto-defaults to `metadata.namespace` if not explicitly set. |
  | undefined `${VAR}` | Raises `kopf.PermanentError` naming the variable and component. |
  
  ### Worked example
  [AIDP-style multi-component CR with milvus + ingress, ${namespace} + ${milvusHost}]
  
  ### Variable values are NOT recursively resolved
  > **Note:** Variable values are taken literally — no recursive resolution.
  
  # WRONG (does NOT work):
  ```yaml
  variables:
    milvusHost: milvus.${namespace}.svc.cluster.local  # nested ${} is NOT expanded
  ```
  
  # CORRECT (use fully-resolved values):
  ```yaml
  variables:
    milvusHost: milvus.aidp-prod.svc.cluster.local
  ```
  
  ### Operator-control fields are NOT templated
  Recommendation: omit `targetNamespace`. The fields below are NOT substituted:
  - `helmChart.repository`, `helmChart.name`, `helmChart.version`, `releaseName`
  - `targetNamespace`
  - `readinessCheck.*`
  
  ### Errors
  - Undefined variable → `kopf.PermanentError` (CR is bad; manual fix required).
  - Missing referenced ConfigMap/Secret → `kopf.TemporaryError(delay=30)` (operator retries every 30s).
  ```

- **TST-02 outline:** parametrize over (manifest path with `${namespace}`, manifest path with explicit `namespace` user-key override, valuesFiles path with `${milvusHost}` from CM, valuesFiles path with `${milvusHost}` from Secret). Each case asserts the resolved value lands in the right place (manifest tempfile content / merged values dict).

- **TST-03 outline:** load `ai-research.yaml`; mock subprocess.run + kr8s.objects.* + HelmOperator; capture manifest tempfile string + Helm install values; compare against `operator_module/tests/snapshots/ai-research/`. First run generates baselines (with explicit "BASELINE_REGEN=1 pytest ..." env-var gate); subsequent runs assert byte-identical.

- **TST-05 outline:** mock `load_values_from_reference`; invoke `handle_helm_deployment` with a single-helm CR fixture; assert `mock.call_args.kwargs.get('variables', None) is None` AND `'render(' not in inspect.getsource(handle_helm_deployment)`.

</specifics>

<deferred>
## Deferred Ideas

- **Per-component variable overrides** — Phase 18 D-18 locks stack-level scope only. Component-level overrides would need a precedence rule + new CRD schema; defer to a future phase if AIDP authors actually ask for it.
- **`when=` predicate on @kopf.on.update for old != new** (D-17 alternative B) — `field='spec'` already prevents the storm; an additional `when=` is paranoia. Defer unless future incident shows status-only patches still trigger reconciles.
- **field='spec' audit on @kopf.on.create / @kopf.on.delete** (D-17 alternative C) — out of scope for v5.0; create handlers already only fire once at admission (no storm risk).
- **Refactor handle_appstack_deployment into pure `_prepare_component_artifacts` helper** (D-13 alternative B) — testability win, but pulls a 200-line refactor into Phase 18. Defer to a future operator-cleanup phase. TST-03's mock-based approach is sufficient for now.
- **Snapshot test for cluster_init/app-store-cluster-init.yaml** (D-11 alternative B) — Phase 16's regression test already covers the shell-script content; duplicating here would add brittleness without coverage gain.
- **Migration walkthrough section in README** (D-07 alternative C) — would overlap with Phase 20's AIDP PR description. Phase 18 keeps the README focused on syntax + worked example; the PR description in `aidp` repo handles the before/after narrative.
- **CI wiring for the new operator_module/tests/** — repo has no CI today (per `.planning/codebase/TESTING.md`); Phase 16 deferred this and Phase 18 maintains that decision. Future test-infra phase.
- **`status.conditions[type=VariablesResolved]` observability field** — already deferred to V51-02 per REQUIREMENTS. Do NOT add in Phase 18.
- **Default-value syntax `${VAR:-default}`** — already deferred to V51-03. `string.Template` doesn't support natively; would require subclass.
- **Templating `targetNamespace`** — already deferred to V51-01. Workaround in DOC-06 is to drop `targetNamespace`.
- **Bare `$identifier` mixing landmine** (Phase 16 deferred) — Phase 18 inherits this. README's worked example must NOT mix bare `$shellvar` with `${VAR}` in the same string, and the kopf.PermanentError message Phase 18 emits will name the offending bare identifier so AIDP authors can locate it.

### Reviewed Todos (not folded)
None — discussion stayed within phase scope.

</deferred>

---

*Phase: 18-operator-wiring-and-docs*
*Context gathered: 2026-05-08*
