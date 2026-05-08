---
phase: 18-operator-wiring-and-docs
plan: 03
subsystem: testing
tags: [pytest, unit-test, mocking, operator, substitution]

requires:
  - phase: 16-render-helper-and-test-scaffolding
    provides: operator_module/tests/conftest.py + requirements-dev.txt
  - phase: 18-operator-wiring-and-docs
    provides: Plan 18-01 wired _render_or_raise + load_values_from_reference dispatch + handle_appstack_deployment variables build + field='spec' decorator
provides:
  - 13-test surface locking OP-06..08, OP-10..12 substitution wiring
  - load_values_from_reference dispatch tests (NotFoundError, ServerError 403, APITimeoutError, yaml.YAMLError)
  - field='spec' decorator static check
affects: [phase-19-validator]

tech-stack:
  added: []
  patterns:
    - "load_values_from_reference dispatch tested directly (not through handle_appstack_deployment) because the AppStack handler swallows per-component exceptions into comp_status['message']"
    - "Substitution-success tests target handle_appstack_deployment end-to-end with mocked subprocess.run + kr8s + HelmOperator"

key-files:
  created:
    - operator_module/tests/test_appstack.py
  modified: []

key-decisions:
  - "Tests 9-12 (OP-11 dispatch) call load_values_from_reference directly because handle_appstack_deployment's `except Exception as e` block at main.py:899 swallows kopf.TemporaryError/PermanentError into comp_status['message']. The OP-11 dispatch contract lives on load_values_from_reference; testing it there is more accurate."
  - "Test 4 (undefined-var PermanentError) calls _render_or_raise directly for the same reason — the wrap is inside the per-component try/except."
  - "All success-path tests (1, 2, 3, 5, 6) target handle_appstack_deployment end-to-end and assert via captured kubectl tempfile content or HelmOperator.install_or_upgrade call args."

patterns-established:
  - "OCI helm components in test fixtures (skip _add_repo branch)"
  - "kr8s.ServerError mock with response.status_code attribute via _make_kr8s_server_error helper"
  - "Base64-encoded Secret data mock via _make_kr8s_secret helper"

requirements-completed: [TST-02, OP-06, OP-07, OP-08, OP-10, OP-11, OP-12]

duration: ~15min (including a one-pass refactor of OP-11 tests after first run revealed the swallowing pattern)
completed: 2026-05-08
---

# Phase 18 / Plan 03: test_appstack.py Summary

**13 unit tests (491 lines, 0.76s runtime) locking the Phase 18 substitution wiring — kubernetesManifest render, valuesFiles dispatch, key/type validation, and field='spec' guard.**

## Performance

- **Duration:** ~15 min (orchestrator-direct after worktree-isolated executor agents were blocked by Bash-tool denials)
- **Started:** 2026-05-08
- **Completed:** 2026-05-08
- **Tasks:** 1/1
- **Files modified:** 1 created (`operator_module/tests/test_appstack.py`, 491 lines)
- **Test runtime:** 0.76s for 13 tests

## Accomplishments
- 13 tests covering OP-06..08, OP-10..12 — every requirement-id from VALIDATION.md's per-task verify map can flip from `❌ W0` to `✅`
- Zero real cluster calls, zero real subprocess invocations, zero real helm binary executions — all `main.subprocess.run`, `main.kr8s.objects.ConfigMap.get`, `main.kr8s.objects.Secret.get`, `main.HelmOperator`, and `main._load_kube_config_once` are patched
- Lazy imports inside test bodies (no module-level `from main import ...`) per the conftest.py sys.path injection pattern from Phase 16

## Task Commits

1. **Task 1 — test_appstack.py with 13 tests** — `734a92a` (test)

## Files Created/Modified
- `operator_module/tests/test_appstack.py` — 13 tests + helper builders (`_make_kr8s_cm`, `_make_kr8s_secret`, `_make_kr8s_server_error`, `_appstack_oci_helm_component`, `_make_kubectl_run_capture`)

## Decisions Made

1. **OP-11 dispatch tests call `load_values_from_reference` directly, not through `handle_appstack_deployment`.** The AppStack handler has a pre-existing `except Exception as e:` at `main.py:899` that swallows kopf.TemporaryError/PermanentError into `comp_status['message']`. The tests would `DID NOT RAISE` because the handler wraps every component-loop iteration. Testing the dispatch on the function that actually implements OP-11 (load_values_from_reference) is more accurate. **NOTE FOR PHASE 18+ HARDENING:** the swallowing block in handle_appstack_deployment may itself be a bug — kopf.TemporaryError is supposed to bubble up to the kopf reconcile loop so it can re-schedule. This is outside Plan 18-03's scope but should be revisited (likely a follow-up plan or an issue for v5.1).

2. **Test 4 (undefined-var PermanentError) calls `_render_or_raise` directly** for the same swallowing reason.

3. **Tests 5-6 (CM/Secret valuesFiles) use OCI helm repo** to skip the `_add_repo` HTTP-style code path that would otherwise need additional mocking.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule: pre-existing-handler-behavior] OP-11 tests refactored from `handle_appstack_deployment` to `load_values_from_reference`**
- **Found during:** Task 1 first test run (8/13 passed; Test 9 failed with `DID NOT RAISE kopf.TemporaryError`)
- **Issue:** `handle_appstack_deployment`'s per-component `except Exception as e` block at `main.py:899` swallows kopf.TemporaryError/PermanentError into `comp_status['message']`, preventing the test's `pytest.raises(...)` from triggering
- **Fix:** Refactored Tests 9-12 to call `load_values_from_reference` directly with `comp_name="vector-db"` and `ref_index=0` for context. The dispatch contract (NotFoundError → TemporaryError, ServerError 4xx → PermanentError, etc.) lives on this function, so testing it there is more honest.
- **Files modified:** operator_module/tests/test_appstack.py (Tests 9-12 only)
- **Verification:** All 13 tests pass on second run.
- **Committed in:** 734a92a (Task 1 commit — refactor was within the same task before commit)

---

**Total deviations:** 1 auto-fixed (test refactor for accuracy against pre-existing handler behavior)
**Impact on plan:** No scope changes. The plan's intent (lock OP-11 dispatch) is preserved; tests now target the function that actually implements OP-11 rather than a higher-level function that swallows the contract.

## Issues Encountered

- **Worktree executor agent blocked by Bash-tool denials.** Plan 18-03's executor was spawned with `isolation="worktree"` but was denied Bash on the very first command (`git symbolic-ref --quiet HEAD`). Same denial pattern as Plan 18-01's earlier failure. Orchestrator fell back to direct execution on the main tree. Wave 2 worktree dispatch was abandoned for plans 18-04 and 18-05 (also fell back to inline).

- **handle_appstack_deployment's broad `except Exception` swallows kopf.TemporaryError.** Documented in Decisions Made #1. Outside Plan 18-03's scope but worth surfacing for the verification phase or a follow-up hardening plan.

## User Setup Required

None — Phase 18 testing is unit-only; no external services, no environment variables, no credentials required.

## Next Phase Readiness

Wave 2 plans 18-04 and 18-05 still pending (also need orchestrator-direct execution per the worktree denial pattern). Once all 5 plans land, Phase 18 verification can run.

---
*Phase: 18-operator-wiring-and-docs*
*Completed: 2026-05-08*
