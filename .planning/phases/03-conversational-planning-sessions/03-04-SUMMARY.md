---
phase: 03-conversational-planning-sessions
plan: "04"
subsystem: testing
tags: [fastapi, planning-sessions, session-lifecycle, contract-validation, pytest]
requires:
  - phase: 03-01
    provides: typed planning-session contract, local persistence, and restart or abandon lifecycle semantics
  - phase: 03-02
    provides: supported-family routing, fit assessment, and draft replay service behavior
  - phase: 03-03
    provides: planning session routes, server-rendered chat surface, and SSE session-state rendering
provides:
  - lifecycle guards for invalid follow-up answers and non-active planning sessions
  - fail-closed contract coverage for session lifecycle markers at the structured-plan seam
  - deterministic end-to-end coverage for transcript replay, no-fit outcomes, and restart or abandon behavior
affects: [phase-04-review-approval-and-apply-gating, planning-ui, session-replay]
tech-stack:
  added: []
  patterns: [preflight session validation before replay, immutable non-active planning session history, fail-closed plan-contract boundary]
key-files:
  created: [app-store-gui/tests/planning/test_planning_session_integration.py]
  modified: [app-store-gui/webapp/main.py, app-store-gui/webapp/planning/session_service.py, app-store-gui/webapp/planning/session_store.py, app-store-gui/webapp/planning/__init__.py, app-store-gui/tests/planning/test_planning_session_service.py, app-store-gui/tests/conftest.py, app-store-gui/tests/planning/test_plan_contract.py]
key-decisions:
  - "Validate session state and pending follow-up eligibility before minting a new correlation ID or transcript turn."
  - "Treat restarted and abandoned planning sessions as immutable history so later phases cannot append review or apply behavior onto draft sessions."
  - "Reject session lifecycle markers at the structured-plan validator boundary so conversational state never leaks into YAML-generation inputs."
patterns-established:
  - "Planning turn submission fails with a conflict when the session is no longer active."
  - "Follow-up answers must target an existing pending question; replay does not accept stale or fabricated question IDs."
requirements-completed: [CHAT-03, CHAT-04, CHAT-05, PLAN-01]
duration: 9 min
completed: 2026-03-20
---

# Phase 3 Plan 04: Conversational Planning Lifecycle Hardening Summary

**Deterministic conversational planning coverage for follow-up replay, no-fit outcomes, and immutable draft-session lifecycle transitions**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-20T03:38:05Z
- **Completed:** 2026-03-20T03:47:18Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments
- Hardened planning-session replay so only active sessions accept new turns and only pending follow-up questions can be answered.
- Extended contract coverage so structured-plan validation rejects session lifecycle markers instead of letting conversational state drift into earlier planning seams.
- Added a deterministic end-to-end integration suite covering session creation, follow-up replay, transcript reload, supported-family routing, no-fit responses, and restart or abandon behavior.

## Task Commits

Each task was committed atomically:

1. **Task 1: Harden multi-turn draft regeneration and follow-up replay** - `ec739ab` (fix)
2. **Task 2: Lock the session lifecycle and contract edges** - `0726ac9` (test)
3. **Task 3: Add deterministic end-to-end coverage for the full Phase 3 flow** - `424e93e` (test)

**Plan metadata:** pending

## Files Created/Modified
- `app-store-gui/webapp/planning/session_service.py` - Added preflight validation so replay rejects non-active sessions and stale or unknown follow-up IDs before creating new correlation metadata.
- `app-store-gui/webapp/planning/session_store.py` - Added explicit planning session lifecycle and follow-up validation errors at the persistence seam.
- `app-store-gui/webapp/main.py` - Mapped invalid planning-session state transitions to HTTP 409 conflicts for the web routes.
- `app-store-gui/webapp/planning/__init__.py` - Exported the new planning-session error types through the package seam.
- `app-store-gui/tests/planning/test_planning_session_service.py` - Added service-level regression coverage for rejected stale follow-ups and rejected turns on abandoned sessions.
- `app-store-gui/tests/conftest.py` - Added structured-plan fixture coverage for leaked session lifecycle markers.
- `app-store-gui/tests/planning/test_plan_contract.py` - Locked the validator contract so session lifecycle fields stay outside the structured-plan seam.
- `app-store-gui/tests/planning/test_planning_session_integration.py` - Added deterministic end-to-end session lifecycle coverage through the FastAPI planning routes.

## Decisions Made
- Validated session state and follow-up eligibility before generating a new correlation ID so rejected retries remain side-effect free.
- Treated restarted and abandoned sessions as immutable planning history, preserving a clean boundary for later review and apply gating work.
- Kept the validator fail-closed on session lifecycle fields so conversational session state does not contaminate Phase 1 structured-plan inputs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented replay bookkeeping from advancing on invalid follow-up retries**
- **Found during:** Task 1 (Harden multi-turn draft regeneration and follow-up replay)
- **Issue:** Invalid follow-up retries could consume a new correlation ID before being rejected, creating incoherent replay bookkeeping for later turns.
- **Fix:** Added preflight session and follow-up validation in the session service before correlation generation or transcript mutation.
- **Files modified:** `app-store-gui/webapp/planning/session_service.py`, `app-store-gui/webapp/planning/session_store.py`, `app-store-gui/webapp/main.py`, `app-store-gui/webapp/planning/__init__.py`
- **Verification:** `cd app-store-gui && python -m pytest tests/planning/test_planning_session_service.py tests/planning/test_planning_routes.py tests/planning/test_planning_session_integration.py -q`
- **Committed in:** `ec739ab` (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix stayed within the planned session-replay seam and improved determinism without expanding scope.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 now has deterministic end-to-end coverage for transcript persistence, no-fit handling, and draft-session lifecycle boundaries.
- Phase 4 can layer review and approval semantics onto active draft sessions without reopening replay or structured-plan contract questions.

## Self-Check: PASSED

---
*Phase: 03-conversational-planning-sessions*
*Completed: 2026-03-20*
