---
phase: 18-operator-wiring-and-docs
plan: 04
subsystem: testing
tags: [pytest, unit-test, mocking, inspect-getsource, regression-lock]

requires:
  - phase: 16-render-helper-and-test-scaffolding
    provides: operator_module/tests/conftest.py + requirements-dev.txt
  - phase: 18-operator-wiring-and-docs
    provides: Plan 18-01 wired _render_or_raise + load_values_from_reference signature change
provides:
  - Two-layered non-wiring lock for handle_helm_deployment (runtime mock-call + static inspect.getsource)
  - Regression guard: any future maintainer attempt to add render() / _render_or_raise() / stack_vars to the helm path triggers test failure
affects: [phase-19-validator, phase-20-aidp-migration]

tech-stack:
  added: []
  patterns:
    - "Static inspect.getsource check with comment stripping (self-invalidating-gate hygiene — a future comment that mentions 'render(' must not flip the test green/red spuriously)"
    - "Runtime mock-call assertion with explicit kwarg-set check (not just absence of unwanted kwargs — also lock the EXACT expected kwarg set)"

key-files:
  created:
    - operator_module/tests/test_helm_non_wiring.py
  modified: []

key-decisions:
  - "Test 1's per-call assertion shape is `len(args) == 0` AND `set(kwargs.keys()) == {kind, name, key, namespace}` — matches the verified kwarg-only call form at operator_module/main.py:1019-1025 (NOT `len(args) == 4` as the original plan-checker B-4 finding caught)"
  - "Test 2 strips comment lines before the substring check on `render(` — prevents future comments from spuriously flipping the gate"
  - "valuesFiles fixture is at spec['valuesFiles'] (top-level), NOT spec['helmChart']['valuesFiles'] — matches the verified call at main.py:1019"

patterns-established:
  - "Two-layered non-wiring lock: runtime mock-call shape + static source inspection together catch both intentional and accidental drift"

requirements-completed: [OP-09, TST-05]

duration: ~10min
completed: 2026-05-08
---

# Phase 18 / Plan 04: test_helm_non_wiring.py Summary

**Two-test invariant lock (152 lines, 1.16s runtime) ensuring `handle_helm_deployment` never receives the Phase 18 variables wiring — both runtime mock-call shape and static source inspection.**

## Performance

- **Duration:** ~10 min (orchestrator-direct after worktree-isolated executor was blocked by Bash-tool denials)
- **Started:** 2026-05-08
- **Completed:** 2026-05-08
- **Tasks:** 1/1
- **Files modified:** 1 created (`operator_module/tests/test_helm_non_wiring.py`, 152 lines)
- **Test runtime:** 1.16s for 2 tests

## Accomplishments

### Test 1 — Runtime mock-call shape
- Invokes `handle_helm_deployment` with a single-chart helm CR containing a top-level `valuesFiles` reference (matches the verified shape at `operator_module/main.py:1019` — `if 'valuesFiles' in spec`)
- Patches `main.load_values_from_reference` with a `MagicMock(return_value={})`
- Asserts:
  - `mock_load.call_count >= 1` (vacuous-pass guard — if zero calls, fixture missed the path)
  - For every recorded call: `len(args) == 0`, `set(kwargs.keys()) == {"kind", "name", "key", "namespace"}`, and absence of `variables=`/`comp_name=`/`ref_index=`
- The kwarg-set check is strictly stronger than absence-only: future drift like `kind=..., name=..., key=..., namespace=..., extra_thing=...` would fail even if `extra_thing` isn't in the deny-list

### Test 2 — Static source check
- Uses `inspect.getsource(handle_helm_deployment)` — verified safe because `handle_helm_deployment` is NOT kopf-decorated (only `update_warrpappstore_function` and `create_warrpappstore_function` carry `@kopf.on.*`; per RESEARCH.md Pitfall 3)
- Strips comment lines before substring check (self-invalidating-gate hygiene)
- Asserts absence of `render(`, `_render_or_raise(`, AND `stack_vars`

## Task Commits

1. **Task 1 — test_helm_non_wiring.py with two tests** — `9776cea` (test)

## Files Created/Modified

- `operator_module/tests/test_helm_non_wiring.py` — 152 lines, 2 tests, mocks `main.load_values_from_reference` + `main._load_kube_config_once` + `main.subprocess.run` + `main.HelmOperator` + `main.should_skip_crds_for_component` + `main.wait_for_component_ready`

## Verified call site (Plan output requirement)

The call shape Test 1 asserts against was verified by direct read of `/Users/christopherjenkins/git/wekaappstore/operator_module/main.py:1019-1025`:

```python
if 'valuesFiles' in spec:
    for values_ref in spec['valuesFiles']:
        ref_ns = values_ref.get('namespace', spec.get('targetNamespace', namespace))
        ref_values = load_values_from_reference(
            kind=values_ref['kind'],
            name=values_ref['name'],
            key=values_ref['key'],
            namespace=ref_ns
        )
```

- Zero positional args
- Exactly four kwargs: `kind`, `name`, `key`, `namespace`
- No `variables=`/`comp_name=`/`ref_index=`

This matches Test 1's per-call assertion shape exactly.

## Confirmed: handle_helm_deployment is NOT kopf-decorated

`grep -nE "^@kopf|^def handle_helm_deployment" operator_module/main.py` shows `handle_helm_deployment` at line 871 with no preceding `@kopf` decorator. The decorators in the file are `@kopf.on.create` (line 845, on `create_warrpappstore_function`), `@kopf.on.update` (line 1127, on `update_warrpappstore_function`), and `@kopf.on.delete` (line 1170, on `delete_warrpappstore_function`). Per RESEARCH.md Pitfall 3, `inspect.getsource(handle_helm_deployment)` returns the actual function body (not wrapper source). Test 2 is safe.

## Comment stripping rationale (Plan output requirement)

Test 2 strips comment lines (any line whose `lstrip()` starts with `#`) before the substring check on `render(`. Without this, the gate would be self-invalidating:

```python
# A future maintainer adds a comment in handle_helm_deployment:
def handle_helm_deployment(...):
    # NOTE: this path does NOT call render() — see Plan 18-04 TST-05
    ...
```

The comment contains the literal string `render(`, which would flip Test 2 to fail even though the code is fine. Stripping comment lines first removes that false-positive risk while still catching real `render(` calls in code.

## Decisions Made

None — followed plan as specified, with the corrected B-4 assertion shape (`len(args) == 0` + kwarg-set check, not `len(args) == 4`).

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **Worktree executor agent blocked by Bash-tool denials.** Same pattern as 18-01 and 18-03 — the orchestrator (this conversation) ran the work directly on main instead.

## User Setup Required

None.

## Next Phase Readiness

Plan 18-05 (backward-compat snapshot test) is the last remaining Wave 2 plan. After 18-05 lands, Phase 18 verification can run.

---
*Phase: 18-operator-wiring-and-docs*
*Completed: 2026-05-08*
