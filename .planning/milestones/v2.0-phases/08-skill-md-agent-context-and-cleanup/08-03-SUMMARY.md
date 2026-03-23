---
phase: 08-skill-md-agent-context-and-cleanup
plan: 03
subsystem: cleanup
tags: [python, fastapi, mcp, planning, cleanup, deprecated-code]

# Dependency graph
requires:
  - phase: 07-validation-apply-and-status-tools
    provides: apply_gateway.py, validator.py, models.py as active MCP tool implementations
provides:
  - Clean codebase with only MCP-relevant planning modules (apply_gateway, validator, models)
  - main.py free of all v1.0 planning session routes and helpers
  - planning/__init__.py exports only apply_gateway symbols
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "planning/__init__.py acts as narrow re-export layer for only the symbols MCP tools need"
    - "Deleted files tracked with git rm to preserve history"

key-files:
  created: []
  modified:
    - app-store-gui/webapp/planning/__init__.py
    - app-store-gui/webapp/main.py
    - app-store-gui/tests/conftest.py
    - app-store-gui/tests/planning/test_apply_gateway.py
    - app-store-gui/tests/planning/test_plan_contract.py

key-decisions:
  - "Kept PLANNING_APPLY_GATEWAY and apply_blueprint_with_namespace/apply_blueprint_content_with_namespace in main.py — these are actively used by blueprint deploy routes (lines ~1391, 1727, 2137)"
  - "Deleted build_structured_plan_preview, execute_structured_plan_apply, apply_structured_plan as dead code (depended on deleted compiler.py, called from no routes)"
  - "Removed apply_structured_plan and build_structured_plan_preview tests from preserved test files — tests covered deleted compiler.py functionality"

requirements-completed: [CLEAN-01, CLEAN-02, CLEAN-03]

# Metrics
duration: 18min
completed: 2026-03-20
---

# Phase 8 Plan 3: Deprecated v1.0 Code Cleanup Summary

**Deleted 5 deprecated source files, 1 HTML template, and 6 test files; stripped all planning session routes and helpers from main.py; narrowed planning/__init__.py to apply_gateway exports only — 85 MCP + 31 planning tests pass**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-20T07:45:00Z
- **Completed:** 2026-03-20T08:03:00Z
- **Tasks:** 2
- **Files modified:** 5 (edited) + 12 (deleted)

## Accomplishments
- Deleted 5 v1.0 backend-brain source files: session_service.py, session_store.py, family_matcher.py, compiler.py, inspection_tools.py
- Deleted planning_session.html template and 6 obsolete test files
- Removed all planning session routes (6 handlers + SSE events endpoint), 3 helper groups, and dead code functions from main.py
- Narrowed planning/__init__.py to export only ApplyGateway symbols used by MCP server
- All 85 MCP server tests and 31 preserved planning tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete deprecated source files and their tests** - `631e301` (chore)
2. **Task 2: Clean up main.py and __init__.py, then verify all preserved tests pass** - `d2447b1` (chore)

## Files Created/Modified
- `app-store-gui/webapp/planning/__init__.py` - Stripped to apply_gateway exports only
- `app-store-gui/webapp/main.py` - Removed all planning session code; kept PLANNING_APPLY_GATEWAY for blueprint deploy routes
- `app-store-gui/tests/conftest.py` - Removed LocalPlanningSessionStore import and deleted fixtures
- `app-store-gui/tests/planning/test_apply_gateway.py` - Removed tests for deleted apply_structured_plan function
- `app-store-gui/tests/planning/test_plan_contract.py` - Removed tests for deleted build_structured_plan_preview function

## Decisions Made
- Kept `PLANNING_APPLY_GATEWAY` and the `apply_blueprint_*` helpers in main.py because they are actively used by blueprint deploy routes at lines ~1391, 1727, 2137 — only the planning session infrastructure was removed
- The functions `build_structured_plan_preview`, `execute_structured_plan_apply`, `apply_structured_plan` had no callers among active routes; they depended on deleted compiler.py and were dead code — removed
- Tests for deleted functions were found in the preserved test files (test_apply_gateway.py, test_plan_contract.py) and removed as Rule 1 auto-fixes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed conftest.py import breaking all test collection**
- **Found during:** Task 2 (verify tests after cleanup)
- **Issue:** `tests/conftest.py` still imported `LocalPlanningSessionStore` from `webapp.planning`, which no longer exports it after `__init__.py` cleanup; caused ImportError for all test collection
- **Fix:** Removed `LocalPlanningSessionStore` import, removed `planning_session_store` and `planning_test_client` fixtures that referenced deleted session code
- **Files modified:** `app-store-gui/tests/conftest.py`
- **Verification:** pytest collection succeeds; 31 preserved tests pass
- **Committed in:** `d2447b1` (Task 2 commit)

**2. [Rule 1 - Bug] Removed tests for deleted functions from preserved test files**
- **Found during:** Task 2 (run preserved test suite)
- **Issue:** `test_apply_gateway.py` had 2 tests calling `main.apply_structured_plan` and `test_plan_contract.py` had 2 tests calling `main.build_structured_plan_preview` — both functions deleted because they depended on deleted compiler.py
- **Fix:** Removed the 4 test functions from their respective files; kept all tests covering preserved functionality
- **Files modified:** `app-store-gui/tests/planning/test_apply_gateway.py`, `app-store-gui/tests/planning/test_plan_contract.py`
- **Verification:** All remaining 31 planning tests pass
- **Committed in:** `d2447b1` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes necessary for correctness — test suite would not run without them. No scope creep; all fixes were direct consequences of deleted modules.

## Issues Encountered
- `test_weka_inspection.py` was located at `tests/planning/test_weka_inspection.py` not `tests/test_weka_inspection.py` as listed in the plan — used the correct path

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- v1.0 backend-brain code fully removed
- Codebase now only contains MCP-relevant planning modules (apply_gateway, validator, models)
- Ready for Phase 9: SKILL.md and NemoClaw integration context work

---
*Phase: 08-skill-md-agent-context-and-cleanup*
*Completed: 2026-03-20*
