---
phase: 07-validation-apply-and-status-tools
plan: 02
subsystem: mcp-tools
tags: [mcp, fastmcp, status, harness, kubernetes, wekaappstore, mock-agent, injectable]

# Dependency graph
requires:
  - phase: 07-validation-apply-and-status-tools
    plan: 01
    provides: "validate_yaml and apply tools with register_validate_yaml/register_apply"
  - phase: 06-mcp-scaffold-and-read-only-tools
    provides: "_impl(injectable)/register_*(mcp) pattern, conftest fixtures, test infrastructure"
provides:
  - "status MCP tool: _status_impl() reads WekaAppStore CR .status subresource, handles 404 and empty status"
  - "server.py updated: all 8 tools registered (inspect_cluster, inspect_weka, list_blueprints, get_blueprint, get_crd_schema, validate_yaml, apply, status)"
  - "mock agent harness: 3 scripted scenarios (happy path, approval bypass, validation failure)"
  - "depth contract tests extended to all 8 tools"
  - "67 tests passing (41 Phase 6 + 16 Phase 7 Plan 1 + 10 new)"
affects:
  - Phase 8 (cleanup) — full 8-tool server is the stable baseline
  - Phase 9 (NemoClaw config) — harness proves tool chain works end-to-end

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_impl(injectable)/register_*(mcp) pattern applied to status tool"
    - "Harness calls flatten_* functions directly with pre-built snapshot dicts (avoids mocking K8s collection stack)"
    - "build_mock_k8s_deps() returns (ApplyGatewayDependencies, ops_log) tuple for side-effect assertion"

key-files:
  created:
    - mcp-server/tools/status_tool.py
    - mcp-server/harness/__init__.py
    - mcp-server/harness/mock_agent.py
    - mcp-server/tests/test_status_tool.py
    - mcp-server/tests/test_mock_agent.py
  modified:
    - mcp-server/server.py
    - mcp-server/tests/test_server.py
    - mcp-server/tests/test_response_depth.py

key-decisions:
  - "Harness calls flatten_inspect_cluster_for_mcp() / flatten_inspect_weka_for_mcp() directly with pre-built snapshot dicts — avoids mocking the full webapp.inspection.cluster collection stack"
  - "build_mock_k8s_deps() returns a real _MockCustomObjectsApi class (not MagicMock) to avoid spec constraints on create_namespaced_custom_object"
  - "ops_log is a shared list appended by mock methods — enables assertion that no 'create' ops occurred in approval bypass scenario"
  - "status tool warns on empty status dict OR None appStackPhase — covers both freshly created CRs and partially reconciled ones"

patterns-established:
  - "ops_log pattern: mock side-effecting methods append (op_type, kwargs) tuples to a shared list for assertion"
  - "Harness module is both pytest-importable and standalone-runnable via __main__ block"

requirements-completed: [MCPS-09, AGNT-02]

# Metrics
duration: 4min
completed: 2026-03-20
---

# Phase 7 Plan 02: Status Tool, Server Wiring, and Mock Agent Harness Summary

**Injectable status tool for WekaAppStore CR state, 8-tool server registration complete, and mock agent harness proving the full inspect-validate-apply chain with 3 scripted scenarios**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T06:47:09Z
- **Completed:** 2026-03-20T06:51:11Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- `status` tool reads WekaAppStore CR `.status` subresource via injectable CustomObjectsApi; handles 404 (found=false) and empty status (reconciliation warning)
- `server.py` now registers all 8 tools; `test_server.py` updated to assert exactly 8 tools with correct names
- Mock agent harness at `harness/mock_agent.py` proves complete inspect-validate-apply chain with zero network calls; all 3 scenarios (happy path, approval bypass, validation failure) produce correct JSON
- Depth contract tests extended to cover `validate_yaml`, `apply`, and `status` responses — all 8 tools now explicitly verified at 2-key max depth
- 67 tests pass (57 Phase 6+7-01 baseline + 10 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: status tool, server wiring, and integration test updates** - `07792c4` (feat)
2. **Task 2: mock agent harness and integration tests** - `e5cb39a` (feat)

**Plan metadata:** (docs commit follows)

_Note: Both tasks followed TDD pattern: RED (tests written first, import failed) -> GREEN (implementation written, all tests pass)_

## Files Created/Modified

- `mcp-server/tools/status_tool.py` - `_status_impl()` + `register_status()` — injectable K8s CustomObjectsApi reader for WekaAppStore CR status
- `mcp-server/harness/__init__.py` - harness package marker
- `mcp-server/harness/mock_agent.py` - scripted tool chain runner: `build_mock_k8s_deps()`, `build_mock_inspection_deps()`, `run_happy_path()`, `run_approval_bypass()`, `run_validation_failure()`; standalone `__main__` block
- `mcp-server/tests/test_status_tool.py` - 4 unit tests: CR state, 404 handling, empty status warning, captured_at
- `mcp-server/tests/test_mock_agent.py` - 3 integration tests for all harness scenarios
- `mcp-server/server.py` - added `register_validate_yaml`, `register_apply`, `register_status` — 8 tools total
- `mcp-server/tests/test_server.py` - renamed test to `test_server_lists_8_tools`, added validate_yaml/apply/status to expected set
- `mcp-server/tests/test_response_depth.py` - added depth tests for validate_yaml, apply, and status

## Decisions Made

- **Harness uses pre-built snapshots, not raw K8s API mocks:** `inspect_cluster` and `inspect_weka` tools call `collect_cluster_inspection()` internally with no injectable API parameter. Rather than mocking the full K8s collection stack, the harness passes pre-built snapshot dicts directly to the `flatten_*_for_mcp()` functions. This is simpler and matches how the depth tests already work.
- **Real `_MockCustomObjectsApi` class instead of MagicMock:** Using a concrete class avoids issues with MagicMock spec constraints on keyword-argument-only calls like `create_namespaced_custom_object(**kwargs)`.
- **`ops_log` shared list pattern:** Mock methods append tuples to a shared list rather than using MagicMock call tracking. This is more readable in assertions and works cleanly with the `_MockCustomObjectsApi` class approach.
- **Status warning on both empty dict AND None appStackPhase:** A CR with `status: {}` is caught by `not status`. A CR with `status: {releaseStatus: ...}` but no `appStackPhase` is caught by `app_stack_phase is None`. Both indicate partial reconciliation.

## Deviations from Plan

None — plan executed exactly as written. The one adaptation was using pre-built snapshot dicts in the harness for the inspect tools (as the plan explicitly anticipated: "Check the actual _impl() signatures to determine the correct approach").

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 8 MCP tools complete with full test coverage (67 tests)
- Mock agent harness proves end-to-end tool chain works without a live cluster
- Phase 7 objectives met: validate_yaml, apply, status tools + server wiring + harness
- Phase 8 (cleanup) can safely remove v1.0 backend-brain code with stable 8-tool baseline in place
- Phase 9 (NemoClaw config) has a working harness to validate NemoClaw skill integration

---
*Phase: 07-validation-apply-and-status-tools*
*Completed: 2026-03-20*

## Self-Check: PASSED

- mcp-server/tools/status_tool.py: FOUND
- mcp-server/harness/__init__.py: FOUND
- mcp-server/harness/mock_agent.py: FOUND
- mcp-server/tests/test_status_tool.py: FOUND
- mcp-server/tests/test_mock_agent.py: FOUND
- mcp-server/server.py: FOUND (modified)
- mcp-server/tests/test_server.py: FOUND (modified)
- mcp-server/tests/test_response_depth.py: FOUND (modified)
- Commit 07792c4: FOUND
- Commit e5cb39a: FOUND
