---
phase: 06-mcp-scaffold-and-read-only-tools
plan: 03
subsystem: mcp-server
tags: [mcp, fastmcp, crd-schema, kubernetes, depth-contract, tools]
dependency_graph:
  requires:
    - phase: 06-01
      provides: server.py FastMCP entry point, register_*(mcp) pattern
    - phase: 06-02
      provides: scan_blueprints(), sample_blueprints fixtures
    - kubernetes.client.ApiextensionsV1Api
  provides:
    - mcp-server/tools/crd_schema.py (_get_crd_schema_impl, register_crd_schema)
    - mcp-server/tests/test_crd_schema.py (7 tests)
    - mcp-server/tests/test_response_depth.py (5 cross-tool depth tests)
    - mcp-server/tests/test_server.py (updated — 4 tests total)
    - All 5 read-only tools registered and validated
  affects:
    - Phase 07+ (any plan needing CRD schema for YAML generation)
    - Any future tool additions (depth contract auto-enforced)
tech_stack:
  added:
    - PyYAML yaml.dump() for manifest-to-YAML-string conversion
  patterns:
    - _get_crd_schema_impl(apiextensions_api, blueprints_dir) injectable pattern
    - Schema field depth exemption: pass-through K8s data is documented exception
    - check_depth() recursive helper for cross-tool depth contract enforcement
key_files:
  created:
    - mcp-server/tools/crd_schema.py
    - mcp-server/tests/test_crd_schema.py
    - mcp-server/tests/test_response_depth.py
  modified:
    - mcp-server/server.py (added register_crd_schema import and call)
    - mcp-server/tests/test_server.py (added 5-tool list and sequencing tests)
decisions:
  - "_get_crd_schema_impl() accepts injectable apiextensions_api and blueprints_dir — same testable pattern as inspect tools"
  - "'schema' field is documented exception to 2-key depth rule — pass-through K8s CRD OpenAPI data, not our domain model"
  - "check_depth() uses exempt_keys frozenset to skip schema depth check while enforcing all other fields"
  - "Examples extracted via scan_blueprints() reuse — no duplicate scanning logic"
metrics:
  duration: "~3 minutes"
  completed: "2026-03-20"
  tasks_completed: 2
  tests_added: 14
  files_created: 3
  files_modified: 2
requirements_completed: [MCPS-06, MCPS-10]
---

# Phase 6 Plan 3: get_crd_schema Tool and Cross-Tool Depth Contract Summary

**get_crd_schema tool reading live WekaAppStore CRD schema with example manifests, plus automated cross-tool depth contract enforcement across all 5 MCP tools via check_depth() helper.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-20T06:13:23Z
- **Completed:** 2026-03-20T06:16:33Z
- **Tasks:** 2
- **Files created:** 3
- **Files modified:** 2

## Accomplishments

- get_crd_schema reads WekaAppStore CRD via injectable ApiextensionsV1Api
- Returns group, version, kind, openAPIV3Schema dict, captured_at, warnings
- 404 ApiException -> structured warning "WekaAppStore CRD not installed", null schema
- Non-404 ApiException -> K8s unavailable warning, null schema
- 1-2 example manifests extracted from blueprints dir as YAML strings via scan_blueprints()
- Empty/missing blueprints dir -> examples=[] + warning (no exception)
- All 5 tools registered and confirmed via test_server_lists_5_tools
- Every tool description contains sequencing guidance (before/after/first/sequencing)
- check_depth() helper enforces 2-key traversal depth contract across all 5 tools
- 'schema' field documented and exempted as pass-through K8s OpenAPI data
- Full suite: 41 tests pass

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | get_crd_schema failing tests | 299f1d8 | tests/test_crd_schema.py |
| 1 (GREEN) | get_crd_schema implementation | bb7c7d3 | tools/crd_schema.py, server.py |
| 2 | Cross-tool depth contract + server validation | 3c3bb25 | tests/test_response_depth.py, tests/test_server.py |

## Files Created/Modified

- `mcp-server/tools/crd_schema.py` - _get_crd_schema_impl() + register_crd_schema()
- `mcp-server/tests/test_crd_schema.py` - 7 tests: shape, values, 404, 500, examples, no-examples, missing-dir
- `mcp-server/tests/test_response_depth.py` - 5 cross-tool depth tests with check_depth() helper
- `mcp-server/server.py` - Added register_crd_schema import and call (4 -> 5 tools)
- `mcp-server/tests/test_server.py` - Added test_server_lists_5_tools + test_all_tool_descriptions_have_sequencing

## Decisions Made

1. **Injectable pattern for testing** — `_get_crd_schema_impl(apiextensions_api, blueprints_dir)` accepts both K8s API client and blueprints directory as optional parameters. Tool function calls the impl with defaults. Tests call impl directly with mocks. Same pattern as inspect_cluster/inspect_weka.

2. **'schema' field depth exemption documented** — The CRD OpenAPI v3 schema is inherently deeply nested K8s data — not our structural choice. The 2-key depth rule applies to our domain model design. `check_depth()` uses `exempt_keys=frozenset({"schema"})` and this exception is documented inline in `test_response_depth.py`.

3. **check_depth() as shared test helper** — Rather than per-tool ad-hoc depth checks, a single recursive helper in `test_response_depth.py` enforces the contract for all tools. Future tools need only add one test call to the file.

4. **Reuse scan_blueprints()** — Examples extracted via the same `scan_blueprints()` function from Plan 02. No duplicate directory scanning logic. `yaml.dump()` converts manifest dicts to YAML strings for the examples list.

## Deviations from Plan

None — plan executed exactly as written. All 7 CRD schema tests and 5 depth contract tests written and passing. The TDD pattern was straightforward: task 2 tests passed GREEN immediately since the implementation (written in task 1) already satisfied all depth and tool-listing requirements.

## Verification

Full test suite output:
```
41 passed in 2.03s
```

5 tools confirmed registered:
```python
['inspect_cluster', 'inspect_weka', 'list_blueprints', 'get_blueprint', 'get_crd_schema']
```

## Phase 6 Completion

All 3 plans in Phase 6 complete:
- Plan 01: MCP server scaffold + inspect_cluster + inspect_weka (16 tests)
- Plan 02: Blueprint scanner + list_blueprints + get_blueprint (11 tests)
- Plan 03: get_crd_schema + depth contract + server validation (14 tests)

Total: 41 tests, 5 tools, flat response contract enforced globally.

## Self-Check: PASSED
