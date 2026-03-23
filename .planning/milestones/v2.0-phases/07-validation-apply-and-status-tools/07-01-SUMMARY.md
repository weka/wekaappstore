---
phase: 07-validation-apply-and-status-tools
plan: 01
subsystem: mcp-tools
tags: [mcp, fastmcp, validate, apply, yaml, kubernetes, wekaappstore, crd]

# Dependency graph
requires:
  - phase: 06-mcp-scaffold-and-read-only-tools
    provides: "_impl(injectable)/register_*(mcp) pattern, conftest fixtures, test infrastructure"
provides:
  - "validate_yaml MCP tool: pure-Python CRD-aware YAML validator with _validate_yaml_impl()"
  - "apply MCP tool: confirmation-gated apply_gateway.py wrapper with _apply_impl()"
  - "16 new unit tests covering all error codes and confirmation gate enforcement"
affects:
  - 07-02 (server wiring plan — needs register_validate_yaml, register_apply)
  - 07-02 (test_server.py tool count update: 5 -> 8 tools after wiring)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_impl(injectable)/register_*(mcp) pattern extended to write-side tools"
    - "confirmed is not True identity check — bypasses string/int coercion for approval gate"
    - "ApiException caught in tool layer, returned as structured dict (no re-raise)"

key-files:
  created:
    - mcp-server/tools/validate_yaml.py
    - mcp-server/tools/apply_tool.py
    - mcp-server/tests/test_validate_yaml.py
    - mcp-server/tests/test_apply_tool.py
  modified: []

key-decisions:
  - "validate_yaml only rejects known v1.0 snake_case fields — camelCase CRD fields (helmChart, appStack) pass through to avoid false positives"
  - "confirmed is not True (identity) not confirmed is False (truthiness) — prevents string 'true' or int 1 from bypassing the approval gate"
  - "apply_tool.py catches ApiException and returns structured dict — no exception propagation to MCP layer"
  - "Tests always inject mocked ApplyGatewayDependencies — never call load_kube_config() in CI"

patterns-established:
  - "Write-tool pattern: validate before apply, explicit confirmed=True (bool) required"
  - "Error response pattern: {code, path, message} per error in errors list"

requirements-completed: [MCPS-07, MCPS-08]

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 7 Plan 01: validate_yaml and apply MCP Tools Summary

**CRD-aware WekaAppStore YAML validator and confirmation-gated apply tool using injectable pattern, rejecting v1.0 snake_case fields and enforcing bool identity check on confirmed parameter**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T06:41:40Z
- **Completed:** 2026-03-20T06:43:50Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `validate_yaml` tool validates WekaAppStore YAML against CRD contract: checks apiVersion, kind, metadata.name, v1.0 field rejection, and deployment method presence
- `apply` tool wraps `apply_yaml_content_with_namespace()` with a hard Python-level confirmation gate (`confirmed is not True`) that cannot be bypassed by string or truthy coercion
- 16 unit tests: 10 for validate_yaml (all error codes + timestamp) and 6 for apply_tool (gate, string bypass, success, K8s error, timestamp, namespace)
- All 57 tests pass (41 Phase 6 baseline + 16 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: validate_yaml tool and tests** - `d554f24` (feat)
2. **Task 2: apply tool and tests** - `2d076b3` (feat)

**Plan metadata:** (docs commit follows)

_Note: Both tasks followed TDD pattern: RED (tests written first, import failed) -> GREEN (implementation written, all tests pass)_

## Files Created/Modified

- `mcp-server/tools/validate_yaml.py` - `_validate_yaml_impl()` + `register_validate_yaml()` — pure-Python CRD-aware validator
- `mcp-server/tools/apply_tool.py` - `_apply_impl()` + `register_apply()` — confirmation-gated apply_gateway.py wrapper
- `mcp-server/tests/test_validate_yaml.py` - 10 unit tests for all validate_yaml rejection cases
- `mcp-server/tests/test_apply_tool.py` - 6 unit tests for apply tool confirmation gate and K8s error handling

## Decisions Made

- **Validator rejects only known v1.0 fields, not all unknown fields:** The CRD uses camelCase (`helmChart`, `appStack`) while v1.0 StructuredPlan used snake_case. Rejecting unknown fields would flag valid CRD YAML — instead, only the 6 explicitly known v1.0 snake_case fields are rejected.
- **`confirmed is not True` (identity check):** Prevents string `"true"`, integer `1`, or any other truthy value from bypassing the approval gate. FastMCP's `bool` type hint handles upstream JSON boolean validation.
- **ApiException caught at tool layer:** When `apply_gateway.py` raises `ApiException`, the apply tool catches it and returns a structured `k8s_api_error_{status}` dict. The MCP layer never sees a Python exception from K8s.
- **Tests always inject `ApplyGatewayDependencies`:** `load_kube_config()` is called at apply time when deps are `None`. Injecting fully mocked deps ensures CI never touches a kubeconfig.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `validate_yaml` and `apply` tools are complete and tested as standalone modules
- Plan 02 (server wiring) will call `register_validate_yaml(mcp)` and `register_apply(mcp)` in `server.py`
- `test_server.py` tool count test will need update from 5 to 8 tools in Plan 02

---
*Phase: 07-validation-apply-and-status-tools*
*Completed: 2026-03-20*

## Self-Check: PASSED

- mcp-server/tools/validate_yaml.py: FOUND
- mcp-server/tools/apply_tool.py: FOUND
- mcp-server/tests/test_validate_yaml.py: FOUND
- mcp-server/tests/test_apply_tool.py: FOUND
- Commit d554f24: FOUND
- Commit 2d076b3: FOUND
