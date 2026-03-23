---
phase: 07-validation-apply-and-status-tools
verified: 2026-03-20T00:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 7: Validation, Apply, and Status Tools Verification Report

**Phase Goal:** Create validate_yaml, apply, and status MCP tools with mock agent harness proving end-to-end chain
**Verified:** 2026-03-20
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | validate_yaml accepts valid WekaAppStore YAML and returns valid=true with empty errors | VERIFIED | `_validate_yaml_impl()` lines 132–137; tests `test_valid_yaml_passes`, `test_valid_yaml_appstack` pass |
| 2 | validate_yaml rejects YAML with v1.0-only fields (blueprint_family, fit_findings) with structured v1_only_field errors | VERIFIED | `_V1_ONLY_SPEC_FIELDS` frozenset lines 23–30; tests `test_rejects_v1_blueprint_family`, `test_rejects_v1_fit_findings` pass |
| 3 | validate_yaml rejects YAML with wrong apiVersion, missing metadata.name, or no deployment method | VERIFIED | Lines 91–130 of validate_yaml.py; 4 corresponding tests pass |
| 4 | apply with confirmed=False returns structured approval_required error and never calls apply_gateway | VERIFIED | `if confirmed is not True` gate at line 62 of apply_tool.py; `test_apply_without_confirmation_returns_error` and `test_apply_with_string_true_returns_error` pass; `mock_deps.load_kube_config.assert_not_called()` confirmed |
| 5 | apply with confirmed=True calls apply_gateway and returns applied_kinds on success | VERIFIED | Lines 77–90 of apply_tool.py; `test_apply_with_confirmation_succeeds` passes; `test_apply_response_has_namespace` passes |
| 6 | apply catches K8s ApiException and returns structured error dict instead of raising | VERIFIED | `except ApiException as exc` lines 91–100 of apply_tool.py; `test_apply_k8s_error_returns_structured` passes |
| 7 | status tool returns current deployment state for a named WekaAppStore CR | VERIFIED | `_status_impl()` reads `.status` subresource via `get_namespaced_custom_object()`; `test_status_returns_cr_state` passes |
| 8 | status tool returns found=false with warning when CR does not exist (404) | VERIFIED | `if exc.status == 404` branch lines 98–111 of status_tool.py; `test_status_not_found` passes |
| 9 | status tool adds a warning when status subresource is empty (operator not yet reconciled) | VERIFIED | `if not status or app_stack_phase is None` check lines 77–81 of status_tool.py; `test_status_empty_warns` passes |
| 10 | MCP server lists all 8 tools with correct names | VERIFIED | server.py lines 21–37 register all 8 tools; `test_server_lists_8_tools` passes with exact name set assertion |
| 11 | Mock agent harness runs happy path inspect-validate-apply loop without errors | VERIFIED | `run_happy_path()` in mock_agent.py lines 268–338; `test_harness_happy_path` passes; standalone run prints PASSED |
| 12 | Mock agent harness approval bypass returns structured error and no CR is created | VERIFIED | `run_approval_bypass()` lines 341–371; ops_log assertion; `test_harness_approval_bypass` passes |
| 13 | Mock agent harness validation failure rejects v1.0 YAML before apply is called | VERIFIED | `run_validation_failure()` lines 374–392; apply never called; `test_harness_validation_failure` passes |
| 14 | All 8 tool responses satisfy the 2-key depth contract | VERIFIED | `test_response_depth_validate_yaml`, `test_response_depth_apply`, `test_response_depth_status` pass alongside Phase 6 depth tests — all 8 tools covered |
| 15 | All 67 tests pass (41 Phase 6 baseline + 26 new Phase 7 tests) | VERIFIED | `pytest tests/ -v` output: 67 passed in 2.51s |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `mcp-server/tools/validate_yaml.py` | validate_yaml tool with `_validate_yaml_impl()` and `register_validate_yaml()` | VERIFIED | 166 lines; exports both functions; `_V1_ONLY_SPEC_FIELDS`, `_VALID_API_VERSIONS`, `_VALID_KINDS` constants present |
| `mcp-server/tools/apply_tool.py` | apply tool with `_apply_impl()` and `register_apply()` | VERIFIED | 127 lines; identity gate `confirmed is not True`; ApiException caught and returned as structured dict |
| `mcp-server/tools/status_tool.py` | status tool with `_status_impl()` and `register_status()` | VERIFIED | 140 lines; reads `.status` subresource; handles 404 and empty status |
| `mcp-server/tests/test_validate_yaml.py` | Unit tests covering valid, invalid, and v1.0 rejection cases | VERIFIED | 249 lines; 10 tests covering all 6+ error code categories; all pass |
| `mcp-server/tests/test_apply_tool.py` | Unit tests for apply covering confirmation gate, success, and K8s error | VERIFIED | 214 lines; 6 tests including string-true identity bypass check; all pass |
| `mcp-server/tests/test_status_tool.py` | Unit tests for status covering CR state, 404, empty status, captured_at | VERIFIED | 157 lines; 4 tests; all pass |
| `mcp-server/harness/mock_agent.py` | Scripted tool chain runner with happy path, approval bypass, and validation failure | VERIFIED | 448 lines; exports `run_happy_path`, `run_approval_bypass`, `run_validation_failure`; `__main__` block runs standalone; all 3 scenarios PASSED |
| `mcp-server/harness/__init__.py` | Package marker | VERIFIED | File exists (empty, as expected) |
| `mcp-server/server.py` | Updated server registering all 8 tools | VERIFIED | Lines 21–37: all 8 register calls present |
| `mcp-server/tests/test_server.py` | Updated test asserting 8 tools with name set | VERIFIED | `test_server_lists_8_tools` asserts `len(tools) == 8` and exact name set |
| `mcp-server/tests/test_response_depth.py` | Extended depth tests for validate_yaml, apply, and status | VERIFIED | Lines 260–348 add 3 new depth tests; all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `mcp-server/tools/apply_tool.py` | `app-store-gui/webapp/planning/apply_gateway.py` | `from webapp.planning.apply_gateway import apply_yaml_content_with_namespace, ApplyGatewayDependencies` | WIRED | Lines 20–23 of apply_tool.py; import present and `apply_yaml_content_with_namespace()` called at line 77 |
| `mcp-server/tools/validate_yaml.py` | CRD contract | `_VALID_API_VERSIONS` and `_V1_ONLY_SPEC_FIELDS` constants | WIRED | Constants at lines 23–36; used in validation logic lines 92, 112 |
| `mcp-server/tools/status_tool.py` | `kubernetes.client.CustomObjectsApi` | `get_namespaced_custom_object()` | WIRED | Line 66 calls `get_namespaced_custom_object(group="warp.io", version="v1alpha1", ...)` |
| `mcp-server/server.py` | `mcp-server/tools/validate_yaml.py` | `register_validate_yaml(mcp)` | WIRED | Line 27: import; line 35: call |
| `mcp-server/server.py` | `mcp-server/tools/apply_tool.py` | `register_apply(mcp)` | WIRED | Line 21: import; line 36: call |
| `mcp-server/server.py` | `mcp-server/tools/status_tool.py` | `register_status(mcp)` | WIRED | Line 26: import; line 37: call |
| `mcp-server/harness/mock_agent.py` | all tool `_impl()` functions | `from tools.validate_yaml import _validate_yaml_impl` etc. | WIRED | Lines 26–31: all 5 `_impl` imports; all called in scenario runners |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MCPS-07 | 07-01-PLAN | `validate_yaml` tool checks generated YAML against CRD and operator contract, returns structured errors | SATISFIED | `_validate_yaml_impl()` checks apiVersion, kind, metadata.name, v1.0 fields, deployment method; 10 tests pass |
| MCPS-08 | 07-01-PLAN | `apply` tool creates WekaAppStore CRs with hard approval gate enforced in code | SATISFIED | `confirmed is not True` identity gate in `_apply_impl()`; wraps `apply_gateway.py`; ApiException handled; 6 tests pass |
| MCPS-09 | 07-02-PLAN | `status` tool returns deployment status of WekaAppStore resources | SATISFIED | `_status_impl()` reads `.status` subresource; 404 and empty-status handling; 4 tests pass |
| AGNT-02 | 07-02-PLAN | Mock agent harness exercises full tool chain with scripted tool-use loops | SATISFIED | `harness/mock_agent.py` with 3 scenarios; standalone run confirmed all PASSED; `test_mock_agent.py` 3 tests pass |

No orphaned requirements — all 4 requirement IDs from the plans appear in REQUIREMENTS.md and are satisfied.

---

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no stub return values, no empty implementations found in any Phase 7 files.

---

### Human Verification Required

#### 1. FastMCP Tool Description Quality

**Test:** Connect a real MCP client (or Claude Desktop) to the server and inspect tool descriptions for `validate_yaml`, `apply`, and `status`.
**Expected:** Descriptions should clearly guide an agent to call them in the correct order (validate before apply, status after apply).
**Why human:** Tool description quality and agent guidance cannot be evaluated programmatically — requires reading them in context of real agent interaction.

#### 2. apply_gateway Integration Against Live Cluster

**Test:** With a real kubeconfig pointing at a cluster with the WekaAppStore CRD installed, call `apply` with confirmed=True and valid YAML.
**Expected:** A WekaAppStore CR is created in the target namespace; `status` then returns the CR's `.status` subresource.
**Why human:** CI has no live K8s cluster; all K8s paths are mocked. Real apply_gateway behavior requires network access.

---

### Gaps Summary

No gaps found. All truths verified, all artifacts substantive and wired, all key links confirmed, all 4 requirements satisfied. The test suite (67 tests, 2.51s) provides high confidence in correctness.

---

## Commit Verification

All 4 documented commits verified in git history:
- `d554f24` — feat(07-01): validate_yaml tool and tests
- `2d076b3` — feat(07-01): apply tool with confirmation gate and full test coverage
- `07792c4` — feat(07-02): status tool, server wiring, and integration test updates
- `e5cb39a` — feat(07-02): mock agent harness and integration tests

---

_Verified: 2026-03-20_
_Verifier: Claude (gsd-verifier)_
