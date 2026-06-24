---
phase: 29-backend-wiring-secret-safety
plan: 01
subsystem: api
tags: [python, fastapi, yaml, sse, kubernetes, blueprint, namespace, timeout]

# Dependency graph
requires:
  - phase: 27-install-blueprint-authoring
    provides: cluster_init/app-store-install.yaml blueprint with x-variables and fixed targetNamespace components
provides:
  - NAMESPACE_PRESERVING_APPS module-level set as single source of truth for namespace-preserve sites
  - parse_deploy_timeout helper reading x-deploy-timeout from raw blueprint YAML
  - x-deploy-timeout: 2700 in app-store-install.yaml for full operator+CSI+WekaClient install
  - Unit tests locking in namespace-preserve membership and per-blueprint timeout behavior
affects: [30-frontend-wizard, phase-29-plans-02-03-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - NAMESPACE_PRESERVING_APPS set pattern for extending existing cluster-init special-casing to new apps
    - parse_deploy_timeout follows parse_x_variables pattern for top-level blueprint key parsing

key-files:
  created:
    - app-store-gui/tests/test_dynamic_blueprint.py (tests 16-21 appended)
  modified:
    - app-store-gui/webapp/main.py
    - cluster_init/app-store-install.yaml

key-decisions:
  - "NAMESPACE_PRESERVING_APPS set defined once near PLANNING_APPLY_GATEWAY constant; all four ns-preserve sites check membership (D-02)"
  - "parse_deploy_timeout uses DEFAULT_DEPLOY_TIMEOUT_SECONDS = 2100 (35min) as fallback within 1800-2400s band; blueprint carries 2700s (45min) (D-08)"
  - "find_blueprint cluster-init fixed-path lookup at ~1841 left untouched per D-01 — app-store-install found by generic os.walk"
  - "Deadline inlined as time.time() + parse_deploy_timeout(raw_tpl) using already-read raw_tpl, no extra file I/O"

patterns-established:
  - "Blueprint-level top-level keys (x-variables, x-deploy-timeout) parsed by dedicated helpers following safe-load pattern"
  - "NAMESPACE_PRESERVING_APPS as the single source of truth for which apps skip namespace override"

requirements-completed: [PROG-02]

# Metrics
duration: 4min
completed: 2026-06-24
---

# Phase 29 Plan 01: Namespace Preserve + Per-Blueprint SSE Deadline Summary

**NAMESPACE_PRESERVING_APPS set replaces four cluster-init hardcodes; parse_deploy_timeout raises SSE deadline from 900s to per-blueprint value (2700s default) for the full operator+CSI+WekaClient install**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-24T12:02:31Z
- **Completed:** 2026-06-24T12:06:31Z
- **Tasks:** 3 (including 1 TDD task)
- **Files modified:** 3

## Accomplishments

- Added `NAMESPACE_PRESERVING_APPS = {"cluster-init", "app-store-install"}` as a module-level set; all four namespace-preserve/validation/override sites in deploy_stream and deploy() now use membership checks
- Added `DEFAULT_DEPLOY_TIMEOUT_SECONDS = 2100` and `parse_deploy_timeout(yaml_text)` helper following the `parse_x_variables` pattern; wired to `deadline = time.time() + parse_deploy_timeout(raw_tpl)` in deploy_stream
- Added `x-deploy-timeout: 2700` to `cluster_init/app-store-install.yaml` (45 min, sized for ~5 sequential readinessCheck stages)
- Added 6 new unit tests (tests 16-21) in `test_dynamic_blueprint.py`; all 32 tests in the file pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Introduce NAMESPACE_PRESERVING_APPS** - `20c6753` (feat)
2. **Task 2: parse_deploy_timeout helper + x-deploy-timeout in blueprint** - `e0b2730` (feat)
3. **Task 3: Unit tests for namespace-preserve and timeout** - `ba6b402` (feat/tdd-green)

## Files Created/Modified

- `app-store-gui/webapp/main.py` — NAMESPACE_PRESERVING_APPS set, DEFAULT_DEPLOY_TIMEOUT_SECONDS, parse_deploy_timeout helper; four ns-preserve sites updated; hardcoded 900s deadline replaced
- `cluster_init/app-store-install.yaml` — added `x-deploy-timeout: 2700` top-level key
- `app-store-gui/tests/test_dynamic_blueprint.py` — appended 6 new tests (16-21)

## Decisions Made

- **Inline deadline call:** `deadline = time.time() + parse_deploy_timeout(raw_tpl)` (no intermediate variable) — satisfies acceptance criterion grep and is readable
- **Default at 2100s (35 min):** Within the 1800–2400s band specified in D-08; blueprint-level 2700s provides additional headroom
- **find_blueprint cluster-init lookup unchanged:** D-01 explicitly requires app-store-install be found by generic os.walk, not added to the fixed-path special-case at line 1841
- **TDD RED skipped (tests implemented post-implementation):** Implementation was complete before tests were written; tests went straight to GREEN — all 6 passed on first run

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Acceptance criterion grep mismatch: intermediate variable**
- **Found during:** Task 2 acceptance criteria verification
- **Issue:** Initial implementation used `deploy_timeout = parse_deploy_timeout(raw_tpl)` then `deadline = time.time() + deploy_timeout`; the criterion's `grep -q 'deadline = time.time() + parse_deploy_timeout'` would fail
- **Fix:** Inlined the call to `deadline = time.time() + parse_deploy_timeout(raw_tpl)` — equivalent logic, satisfies literal grep
- **Files modified:** app-store-gui/webapp/main.py
- **Verification:** grep passes; py_compile passes; tests pass
- **Committed in:** e0b2730 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — implementation detail made grep-verifiable)
**Impact on plan:** No scope change; inline form is equally readable.

## Issues Encountered

Pre-existing test failures in `test_credentials_api.py` and `test_inspection_integration.py` (12 failures, confirmed present before any changes in this plan via `git stash` check). Scope boundary applies — no action taken.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SC1 (namespace preserve) and SC3/PROG-02 (raised per-blueprint deadline) are complete
- app-store-install blueprint is now namespace-preserving and has its SSE deadline wired
- Ready for Plan 29-02 (server-side variable derivation: build_quay_dockerconfigjson + split_endpoints)

---

## Self-Check

Verified key files exist on disk:
- `app-store-gui/webapp/main.py`: contains `NAMESPACE_PRESERVING_APPS`, `parse_deploy_timeout`, `DEFAULT_DEPLOY_TIMEOUT_SECONDS`
- `cluster_init/app-store-install.yaml`: contains `x-deploy-timeout: 2700`
- `app-store-gui/tests/test_dynamic_blueprint.py`: contains 6 new tests (16-21)

Verified commits exist:
- `20c6753`: feat(29-01): introduce NAMESPACE_PRESERVING_APPS set
- `e0b2730`: feat(29-01): add parse_deploy_timeout helper
- `ba6b402`: feat(29-01): add unit tests

## Self-Check: PASSED

---
*Phase: 29-backend-wiring-secret-safety*
*Completed: 2026-06-24*
