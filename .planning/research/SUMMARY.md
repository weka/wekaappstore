# Research Summary: v5.0 AppStack Variable Substitution

**Project:** WEKA App Store Operator — OpenClaw MCP Tools
**Milestone:** v5.0 AppStack Variable Substitution
**Domain:** Brownfield Kubernetes operator — additive feature to Kopf-based Python operator
**Researched:** 2026-05-06
**Confidence:** HIGH

---

## Executive Summary

The v5.0 milestone adds `spec.appStack.variables` to the `WekaAppStore` CRD and a `render()` helper using Python's stdlib `string.Template` to substitute `${VAR}` tokens into `kubernetesManifest` strings and into `valuesFiles` content (loaded from ConfigMaps/Secrets) before they reach `kubectl apply` or Helm. The change is confined to three files — `operator_module/main.py`, `weka-app-store-operator-chart/templates/crd.yaml`, and `mcp-server/tools/validate_yaml.py` — with no new runtime dependencies. The AIDP blueprint is the primary motivating case: 17 hardcoded `namespace: rag` literals and multiple DNS names in ConfigMap values cannot be portably overridden today without an external search-and-replace.

The recommended implementation uses `Template.substitute()` (strict mode) rather than `safe_substitute()`, making undefined variable references hard errors (`kopf.PermanentError`) surfaced at apply time with the offending variable name and component in the message. `string.Template` was confirmed correct via live Python 3.10 execution — it is the only option in the candidate set that is JSON-safe (leaves `{"auths": {}}` untouched) while requiring zero new dependencies. All 10 table-stakes features expected by engineers familiar with Flux, Kustomize, and ArgoCD substitution are covered by the PRD.

The most significant implementation risk is a backward-compatibility flaw in the PRD's own guard logic: the `${namespace}` auto-default makes the variables dict always non-empty, so the `if not variables` guard in `render()` is never True, which causes `Template.substitute()` to run on every existing `kubernetesManifest` string — including the shell scripts already in `cluster_init/app-store-cluster-init.yaml` (which contain bare `$CRDS`, `$CRD`, `$MISSING`, `$GATEWAY_API_URL`). This triggers `KeyError` on first reconcile after upgrade. The fix is a single pre-scan guard: `if not re.search(r'\$\{[^}]+\}', text): return text`. This makes the feature truly opt-in — only `${VAR}` tokens with braces trigger substitution — and simultaneously mitigates the NGC `$oauthtoken` bare-identifier risk. This guard is the foundation of the implementation and must be the first thing written.

---

## Key Findings

### Recommended Stack

The operator runtime is locked (Python/Kopf). No new technology is being introduced. `string.Template` (stdlib) is the only option satisfying all constraints simultaneously: JSON-safe (leaves `{"auths": {}}` untouched), zero new runtime dependency, `$$` escape, strict `KeyError` on undefined. `str.format()` crashes immediately on any `{`-containing string. Jinja2 has `{{ }}` collision with Kubernetes JSON content and adds a runtime dependency. `re.sub` custom regex has no documented escape mechanism. Verified by local Python 3.10.9 execution against real AIDP manifests.

**Core technologies:**
- `string.Template` (stdlib, Python 3.10): `${VAR}` substitution — JSON-safe, zero new dep, `$$` escape, strict `KeyError` on undefined. Verified by local execution.
- `kopf.PermanentError` (already imported): Non-retriable failure on undefined variable reference — correct kopf idiom already used elsewhere.
- `PyYAML` (already in use): No change — substitution runs on raw string *before* `yaml.safe_load`.

**Dev/test only:**
- `pytest-subprocess` 1.5.4: Mock `subprocess.run` calls to `kubectl`/`helm` in `operator_module/tests/`.
- `operator_module/requirements-dev.txt`: New file; keep test deps out of runtime `requirements.txt`.

**CRD schema:** `additionalProperties: { type: string }` confirmed safe — identical pattern already in production at `crd.yaml:184` for `matchLabels`. Do NOT add `x-kubernetes-preserve-unknown-fields: true`.

### Expected Features

**Must have (table stakes) — all covered by PRD:**
- `${VAR}` syntax resolves in `kubernetesManifest` strings and `valuesFiles` content
- Undefined variable raises `kopf.PermanentError` naming the variable AND component at apply time
- `$$` escapes a literal dollar sign (stdlib behavior, free)
- CRD schema declares `variables` with `additionalProperties: { type: string }`
- Existing CRs without `variables:` continue to work identically (non-negotiable constraint)
- Validator accepts `variables:` block without spurious error
- README docs: syntax, escape, auto-default, strict-failure
- Substitution in both `kubernetesManifest` AND `valuesFiles` content (including Secret-backed — Q1 resolved YES)
- `${namespace}` auto-defaults to CR's `metadata.namespace`

**Should have (differentiators):**
- Variables inline in CR — single-file portability, no external ConfigMap/Secret required
- Strict-fail from day one (Flux took years and a feature gate)
- JSON-safe by design — zero breakage on Docker registry auth payloads
- `PermanentError` with named variable + component — no log diving
- Validator soft-warning on hardcoded `.svc.cluster.local` DNS names and `namespace:` literals

**Defer to v5.1+:**
- `targetNamespace: ${namespace}` (Q2 resolved: DEFER — dropping `targetNamespace` achieves the same result; document the cliff)
- `status.conditions[type=VariablesResolved]` (Q3 resolved: NO resolved-values map in status — sensitive; optional condition boolean is v5.1)
- Default-value syntax, inline `values:` recursion, external variable sources

### Architecture Approach

The feature threads a stateless `render()` helper through exactly two execution paths inside `handle_appstack_deployment`. No new processes, controllers, or external services. Variables dict built once at stack scope (before the component loop at `main.py:~555`), passed by reference into both paths. Kopf decorators do not wrap handler functions, so `render()` and `load_values_from_reference()` are directly callable in pytest.

**Major components:**
1. `render(text, variables)` — new pure function; pre-scan guard is the backward-compat foundation; catches both `KeyError` (undefined) and `ValueError` (malformed placeholder `${}`)
2. `load_values_from_reference` signature extension — new `variables=None` default; render call between raw-string fetch and `yaml.safe_load`; `handle_helm_deployment` single-chart path at `main.py:~885` uses `variables=None` default — do NOT touch that call site
3. `handle_appstack_deployment` wiring — variables dict construction at `~line 555`; render call in manifest branch at line 727; call-site update at `~line 675`
4. CRD schema addition — `variables:` sibling of `components:`; optional, no `required:`
5. Validator soft-warning — non-blocking; `valid` stays `True`; also validates key names (identifier pattern) and value types
6. `operator_module/tests/` — new directory (does not currently exist); `__init__.py` + `test_render.py` + `test_appstack.py`

### Critical Pitfalls

1. **PRD backward-compat guard is broken (CRITICAL)** — `${namespace}` auto-default makes `variables` always non-empty; `if not variables` guard never triggers; `cluster_init/app-store-cluster-init.yaml` has shell scripts with `$CRDS`, `$CRD`, `$MISSING`, `$GATEWAY_API_URL` in `kubernetesManifest:` blocks that will raise `KeyError` on first reconcile after upgrade. Fix: `if not re.search(r'\$\{[^}]+\}', text): return text` as the FIRST thing in `render()`. Not optional.

2. **`$oauthtoken` in NGC credentials (CRITICAL — mitigated by fix for Pitfall 1)** — NGC Docker credential format uses `$oauthtoken` as username in JSON. `Template.substitute()` raises `KeyError('oauthtoken')` on bare `$identifier`. The Pitfall 1 pre-scan guard prevents this (no braces = no match = Template never called). Defense-in-depth: store NGC secrets in `data:` (base64) not `stringData:`.

3. **Variable values are NOT recursively resolved (PRD example is wrong)** — The PRD's `milvusHost: milvus.${namespace}.svc.cluster.local` example does NOT work. `Template.substitute()` is single-pass. `${milvusHost}` resolves to the literal string `milvus.${namespace}.svc.cluster.local` — the inner `${namespace}` is never further substituted. AIDP migration must use fully-resolved variable values. README must NOT ship the PRD's cross-referencing variable pattern as a documented example.

4. **Both `KeyError` AND `ValueError` must be caught** — `Template.substitute()` raises `ValueError` for malformed placeholders (`${}`, `${123}`), not `KeyError`. PRD only catches `KeyError`. A `ValueError` with no component name in scope produces a confusing generic error message. Both must be caught at every call site.

5. **Variable key names must be valid Python identifiers** — CRD permits arbitrary map keys; `string.Template` requires `[_a-zA-Z][_a-zA-Z0-9]*`. Key `my-host` produces `ValueError`. Add key-name validation when building the variables dict. Validator should also flag this.

6. **`on.update` reconcile storm (pre-existing, amplified)** — Bare `@kopf.on.update` fires on operator's own status patches. Add `field='spec'` filter to the decorator. Include in Phase 3.

7. **`handle_helm_deployment` single-chart path must NOT receive variables wiring** — `variables=None` default protects this call site. Lock it with a unit test.

---

## Implications for Roadmap

Recommended 5-phase split (Phase 4 parallel-shippable with Phases 2–3; Phase 5 separate repo):

### Phase 1: `render()` Helper + Test Scaffolding

**Rationale:** The pre-scan guard is the foundation of all backward-compat correctness. Write and test `render()` before wiring it into live handler paths. `operator_module/tests/` does not exist yet — create it here.

**Delivers:** `render()` with `_SUBST_RE` pre-scan guard; `operator_module/tests/__init__.py` + `test_render.py`; `operator_module/requirements-dev.txt`

**Addresses:** Pitfalls 1, 2, 3, 4

---

### Phase 2: CRD Schema Update

**Rationale:** Additive and independently deployable. Must land before new CRs with `variables:` can pass Kubernetes admission validation. Zero operator risk.

**Delivers:** `variables:` field under `spec.properties.appStack.properties` in `crd.yaml`; description documents syntax, auto-default, `$$` escape, and identifier requirement

---

### Phase 3: Operator Wiring

**Rationale:** With `render()` tested (Phase 1) and CRD schema deployed (Phase 2), wiring is safe. Three discrete edits to `main.py`. This is the phase that delivers the user-visible feature.

**Delivers:** Variables dict construction with key-name validation; `load_values_from_reference` signature extension + render call; manifest branch render call; both call sites wired; `test_appstack.py`; README user-facing doc; `field='spec'` filter on `@kopf.on.update`; fetch-error upgrade to `TemporaryError`

**PRD open questions resolved:** Q1=YES (Secret valuesFiles), Q2=DEFER (targetNamespace to v5.1), Q3=NO resolved-values map (sensitive); optional condition deferred to v5.1

---

### Phase 4: Validator Soft-Warning + New Fixture (parallel-shippable)

**Rationale:** No dependency on operator code. Standalone PR. Can merge any time after Phase 1 establishes what a valid `variables` block looks like.

**Delivers:** `validate_yaml.py` soft-warning for hardcoded DNS + `namespace:` literals; variable key/type validation raising `errors`; `test_validate_yaml.py` extensions; `ai-research-portable.yaml` fixture

---

### Phase 5: AIDP Migration (Separate Repo — after Phases 1–3 deployed)

**Rationale:** End-to-end smoke test. Separate PR against `aidp` repo. Only start after Phases 1–3 are deployed and cluster-verified.

**Delivers:** `weka-aidp-appstack.yaml` migrated with fully-resolved variable values; 17 `namespace: rag` literals → `${namespace}`; `aidp-site-config.yaml` DNS literals → variable references; NGC secrets in `data:` (base64) not `stringData:`

---

### Phase Ordering Rationale

- Phase 1 before Phase 3: `render()` must exist and be tested before live handler wiring
- Phase 2 before Phase 5: CRD schema must be in cluster before new CRs with `variables:` pass admission; in single-PR atomic deployment, Phase 2 and 3 ship together
- Phase 4 is parallel to 2 and 3: validator has no operator code dependency
- Phase 5 is a separate-repo follow-up: requires cluster deployment of Phases 1–3

The pre-scan guard in `render()` (Phase 1) is not optional polish — it is the architectural decision that determines whether the feature is backward-compatible. Every subsequent phase's correctness depends on it.

### Research Flags

No phases require `/gsd-research-phase` during planning. All unknowns were resolved during this research cycle with HIGH confidence from direct source inspection and local code execution.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | `string.Template` verified by local Python 3.10.9 execution; all alternatives disqualified by concrete test results |
| Features | HIGH | 10 table-stakes features verified against Flux/Kustomize/ArgoCD; all 3 PRD open questions resolved with rationale |
| Architecture | HIGH | All integration points at file:line precision from direct source inspection; data flow confirmed |
| Pitfalls | HIGH | All 13 pitfalls verified by executing code paths against real operator source; Pitfall 1 flaw confirmed by running Template against real `cluster_init/` manifests |

**Overall confidence: HIGH**

### Gaps to Address

- **PRD `milvusHost` example must be corrected in README:** The PRD's cross-referencing variable syntax does not work (single-pass). README must show fully-resolved values only.
- **Fetch-error handling in `load_values_from_reference`:** Upgrade transient fetch errors to `kopf.TemporaryError` (not silent `{}` return) to resolve Pitfall 5 asymmetry.
- **`@kopf.on.update` `field='spec'` filter is pre-existing debt:** Phase 3 should include this fix regardless of the variables feature.
- **Single-helm path behavior must be locked by test:** `handle_helm_deployment` at `main.py:~885` must NOT receive `variables` wiring. A unit test must lock this explicit non-behavior.

---

## Sources

**Primary (HIGH confidence):** Direct codebase inspection at `operator_module/main.py` (lines 352–803, 885, 1015–1027), `crd.yaml` (all 259 lines, specifically line 184), `mcp-server/tools/validate_yaml.py`, `cluster_init/app-store-cluster-init.yaml` (lines 143–158); Python 3.10.9 local execution confirming all Template behaviors; Python stdlib docs for `string.Template`; `pytest-subprocess` 1.5.4 PyPI metadata.

**Secondary (MEDIUM confidence):** Flux postBuildSubstitutions docs; kopf testing docs (KopfRunner is integration-only); Kubernetes issue #104137 (`additionalProperties` bug, pre-dates current cluster versions); ArgoCD, Kustomize docs for feature comparison.

---

*Research completed: 2026-05-06*
*Ready for roadmap: yes*
