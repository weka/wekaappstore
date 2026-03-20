---
phase: 03-conversational-planning-sessions
plan: "01"
subsystem: api
tags: [planning, sessions, json-storage, pytest, conversational-planning]
requires:
  - phase: 02-cluster-and-weka-inspection-signals
    provides: bounded inspection and fit-finding contracts reused by later conversational planning work
provides:
  - typed planning session, turn, follow-up, and draft revision models
  - replayable local session persistence seam for create, append, restart, and abandon flows
  - deterministic session-store tests proving restart and abandon behavior stays off the apply path
affects: [03-conversational-planning-sessions, review-and-apply-gating, planner-routes]
tech-stack:
  added: []
  patterns: [backend-owned session snapshots, replayable json persistence seam, deterministic repository fixtures]
key-files:
  created:
    - app-store-gui/webapp/planning/session_store.py
    - app-store-gui/tests/planning/test_planning_session_store.py
  modified:
    - app-store-gui/webapp/planning/models.py
    - app-store-gui/webapp/planning/__init__.py
    - app-store-gui/tests/conftest.py
key-decisions:
  - "Represent restart as a new active session linked back to the original so audit history remains intact."
  - "Keep the first durable seam file-backed and deterministic with injectable clocks and ID factories for pytest stability."
patterns-established:
  - "Planning session state lives in backend-owned typed snapshots instead of browser-only prompt history."
  - "Lifecycle actions update explicit session status and revision metadata without touching preview or apply code paths."
requirements-completed: [CHAT-04, CHAT-05]
duration: 17min
completed: 2026-03-20
---

# Phase 3 Plan 1: Conversational Planning Session Contract Summary

**Typed conversational planning sessions with replayable local persistence, explicit follow-up state, and audited restart or abandon lifecycle handling**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-20T02:53:00Z
- **Completed:** 2026-03-20T03:10:26Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added backend-owned dataclasses for planning sessions, turns, follow-up questions, draft revisions, and lifecycle status.
- Added a narrow local session repository that persists replayable JSON snapshots and supports create, append-turn, restart, and abandon operations.
- Added deterministic pytest coverage for create, reload, follow-up tracking, restart auditability, and abandon semantics.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define typed planning-session and follow-up contracts** - `6713bf9` (feat)
2. **Task 2: Implement a durable session-store seam with replay support** - `2174513` (feat)
3. **Task 3: Prove restart and abandon semantics with deterministic tests** - `1f7bf2e` (test)

## Files Created/Modified
- `app-store-gui/webapp/planning/models.py` - Adds the typed Phase 3 session, turn, follow-up, and revision contracts.
- `app-store-gui/webapp/planning/__init__.py` - Exposes the session models and store seam through the planning package.
- `app-store-gui/webapp/planning/session_store.py` - Implements the durable local planning session repository and lifecycle operations.
- `app-store-gui/tests/conftest.py` - Adds deterministic session-store fixtures with fixed timestamps and IDs.
- `app-store-gui/tests/planning/test_planning_session_store.py` - Verifies replay, unanswered follow-up tracking, restart, and abandon behavior.

## Decisions Made
- Restart creates a replacement session and marks the original as `restarted` so later plans can preserve audit history while giving users a clean draft.
- The first persistence seam is a local JSON repository with injected clocks and ID factories, keeping later storage backends replaceable and current tests deterministic.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- One abandon-flow test expected the wrong deterministic timestamp; the fixture expectation was corrected and the focused pytest target passed on rerun.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 now has an explicit backend session contract and persistence seam for upcoming chat routes and planner orchestration.
- Later plans can build on replayable session state instead of browser-only conversation history.

## Self-Check

PASSED

---
*Phase: 03-conversational-planning-sessions*
*Completed: 2026-03-20*
