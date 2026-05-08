---
phase: 18-operator-wiring-and-docs
plan: 01
subsystem: infra
tags: [kopf, kr8s, operator, helm, string-template, render]

requires:
  - phase: 16-render-helper-and-test-scaffolding
    provides: render() helper + ValueError contract
  - phase: 17-crd-schema-additive-update
    provides: spec.appStack.variables CRD schema with admission validation
provides:
  - _render_or_raise wrapper that converts ValueError to kopf.PermanentError with component context
  - load_values_from_reference signature evolved to (kind, name, key, namespace, *, variables=None, comp_name=None, ref_index=None)
  - Typed kr8s -> kopf exception dispatch (NotFoundError/APITimeoutError/ServerError 5xx -> TemporaryError, ServerError 4xx + yaml.YAMLError -> PermanentError)
  - stack-scope variables dict at top of handle_appstack_deployment with key/type validation (defense-in-depth alongside CRD admission)
  - kubernetesManifest ${VAR} substitution before kubectl apply
  - valuesFiles ${VAR} substitution before yaml.safe_load
  - field='spec' filter on @kopf.on.update to suppress reconcile storms from status patches
  - Chart bumped 0.1.62 -> 0.1.63
affects: [phase-18-tests, phase-19-validator, phase-20-aidp-migration]

tech-stack:
  added: []
  patterns:
    - "Typed kopf exception dispatch in load_values_from_reference (TemporaryError for transient, PermanentError for definitive failures)"
    - "_render_or_raise helper pattern: wrap render() in try/except and rewrap as kopf.PermanentError with source_desc context"
    - "Stack-scope variables-dict build at function top before any deployment work (fail-fast on invalid keys)"

key-files:
  created: []
  modified:
    - operator_module/main.py
    - weka-app-store-operator-chart/Chart.yaml

key-decisions:
  - "stack_vars built ONCE at top of handle_appstack_deployment after enabled_components filter and before resolve_dependencies — pre-validation prevents partial deployment on invalid keys"
  - "Helm-path callsite (line ~998 area) NOT touched — variables=None default preserves non-wiring per L-06; TST-05 will lock this in Plan 18-04"
  - "Both AppStack and helm paths get the typed kr8s -> kopf upgrade (D-02) — helm path's pre-existing silent-{} fallback is replaced with kopf.TemporaryError on transient failures"
  - "Chart patch bump (0.1.63) signals additive backward-compat change; appVersion unchanged"

patterns-established:
  - "_render_or_raise(text, variables, *, source_desc) — single helper used at all 3 render call sites with distinct source_desc strings"
  - "kr8s exception classification: NotFoundError + APITimeoutError + ServerError(status_code>=500) -> TemporaryError(delay=30); ServerError(<500) + yaml.YAMLError -> PermanentError"

requirements-completed: [OP-06, OP-07, OP-08, OP-09, OP-10, OP-11, OP-12]

duration: ~25min (incl. cherry-pick recovery from blocked worktree)
completed: 2026-05-08
---

# Phase 18 / Plan 01: Operator Wiring + Chart Bump Summary

**Phase 16 render() helper wired into handle_appstack_deployment (manifest path + valuesFiles loop) and load_values_from_reference (CM/Secret render-before-yaml.safe_load), with stack-scope variables dict, key/type validation, typed kr8s→kopf exception dispatch, field='spec' update guard, and Chart bump 0.1.62→0.1.63.**

## Performance

- **Duration:** ~25 min (cherry-pick recovery from worktree-blocked agent)
- **Started:** 2026-05-08 (executor agent ~10 min Tasks 1+2; orchestrator cherry-pick + Tasks 3+4 ~15 min)
- **Completed:** 2026-05-08
- **Tasks:** 4/4
- **Files modified:** 2

## Accomplishments
- `_render_or_raise(text, variables, *, source_desc)` helper added next to `render()` — wraps `(KeyError, ValueError)` into `kopf.PermanentError` with chained `from e`
- `load_values_from_reference` rewritten:
  - Signature: `(kind, name, key, namespace, *, variables=None, comp_name=None, ref_index=None) -> Dict[str, Any]`
  - Renders raw CM/Secret string via `_render_or_raise` before `yaml.safe_load` when `variables is not None`
  - Replaces broad `except Exception → return {}` with typed dispatch:
    - `kr8s.NotFoundError` / `kr8s.APITimeoutError` / `kr8s.ServerError(status>=500)` → `kopf.TemporaryError(delay=30)`
    - `kr8s.ServerError(status<500)` (auth/RBAC/4xx) → `kopf.PermanentError`
    - `yaml.YAMLError` → `kopf.PermanentError`
  - Helm-path callsite preserved (kwarg-only, no `variables=`)
- `handle_appstack_deployment`:
  - `import re` added at top of module
  - Stack-scope `stack_vars = {'namespace': namespace, **raw_user_vars}` built AFTER the `if not enabled_components: return {...}` early-return and BEFORE `resolve_dependencies(...)` — invalid keys/values raise `kopf.PermanentError` before any deployment work
  - `valuesFiles` loop now `for idx, values_ref in enumerate(...)` and passes `variables=stack_vars, comp_name=comp_name, ref_index=idx` on the AppStack path only
  - `kubernetesManifest` rendered via `_render_or_raise(..., source_desc=f"Component '{comp_name}'.kubernetesManifest")` immediately before `tempfile.NamedTemporaryFile` write
- `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')` — single-line decorator change suppresses reconcile storms from operator's own status patches
- `Chart.yaml`: `version: 0.1.62 → 0.1.63` (additive backward-compat patch bump)

## Task Commits

Each task was committed atomically:

1. **Task 1 — `_render_or_raise` helper** — `5cba201` (feat) — cherry-picked from worktree-agent-af11ab2085408d9f9
2. **Task 2 — `load_values_from_reference` rewrite (signature + render + typed dispatch)** — `cfeefd9` (feat) — cherry-picked from worktree-agent-af11ab2085408d9f9
3. **Task 3 — `handle_appstack_deployment` wiring (stack_vars + valuesFiles kwargs + manifest render + field='spec')** — `0a5c156` (feat)
4. **Task 4 — Chart.yaml version bump 0.1.62 → 0.1.63** — `efbe8b5` (chore)

## Files Created/Modified
- `operator_module/main.py` — `_render_or_raise` helper, `load_values_from_reference` rewrite, `handle_appstack_deployment` wiring (3 sub-steps), `import re`, `field='spec'` decorator
- `weka-app-store-operator-chart/Chart.yaml` — version bump

## Decisions Made
None during execution — all decisions came from CONTEXT.md D-01..D-18; only deviation was the worktree-recovery path described below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule: blocked-tool-fallback] Worktree executor blocked by Edit-tool denials → cherry-pick + orchestrator-direct execution**
- **Found during:** Task 3 (handle_appstack_deployment wiring) inside worktree-agent-af11ab2085408d9f9
- **Issue:** Edit tool was denied 4 times across 2 successive executor agents when targeting `operator_module/main.py` inside the `.claude/worktrees/agent-*/` path. Tasks 1+2 had completed and committed inside the worktree before the denials began. User confirmed they were not intentionally denying — likely a runtime-side path filter on `.claude/worktrees/`.
- **Fix:** Cherry-picked the two completed worktree commits (1106acc, 7363f14) onto main as `5cba201`, `cfeefd9`. Orchestrator then applied Tasks 3+4 directly via Edit on the main tree (Edit on main paths worked normally).
- **Files modified:** operator_module/main.py, weka-app-store-operator-chart/Chart.yaml
- **Verification:** All 11 acceptance gates from Plan 18-01 Task 3 + the Task 4 `git diff --shortstat` gate passed; `python -m py_compile operator_module/main.py` clean.
- **Committed in:** 5cba201, cfeefd9, 0a5c156, efbe8b5
- **Cleanup pending:** worktree-agent-af11ab2085408d9f9 branch and worktree directory now contain orphan commits superseded by the cherry-picks; orchestrator will force-remove during wave-merge sweep.

---

**Total deviations:** 1 auto-fixed (recovery path for runtime-level path block)
**Impact on plan:** No scope changes. End-state on main is byte-equivalent to what the original worktree agent would have produced — same commits, same code, same acceptance gates pass.

## Issues Encountered
- Edit-tool denials inside `.claude/worktrees/agent-*/` paths blocked the executor agent. Resolved by cherry-picking the agent's completed commits and finishing the remaining sub-steps via orchestrator-direct Edit on the main tree.
- Helm-path callsite NOT verified by an explicit grep gate in Plan 18-01 (it relies on TST-05 / Plan 18-04 to lock non-wiring). Manual inspection of lines ~998-1010 confirms the kwarg-only call shape is preserved.

## User Setup Required

None — no external service configuration required. Phase 18 changes are operator-image-internal; cluster will pick up the new image when the chart 0.1.63 is published (deferred to end-of-v5.0 per Phase 17 D-15).

## Next Phase Readiness

Wave 2 plans (18-03 test_appstack, 18-04 test_helm_non_wiring, 18-05 backward-compat snapshot) can now run — they all depend on Plan 18-01's wiring being landed on main. Plan 18-02 (README) is independent and ran in parallel as a separate worktree.

---
*Phase: 18-operator-wiring-and-docs*
*Completed: 2026-05-08*
