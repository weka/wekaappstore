---
phase: 06-mcp-scaffold-and-read-only-tools
plan: 02
subsystem: mcp-server
tags: [mcp, fastmcp, blueprints, yaml-scanner, wekaappstore, tools]

requires:
  - phase: 06-01
    provides: server.py FastMCP entry point, register_*(mcp) pattern, config.BLUEPRINTS_DIR

provides:
  - mcp-server/tools/blueprints.py (scan_blueprints, flatten_blueprint_summary, flatten_blueprint_detail, register_blueprint_tools)
  - mcp-server/tests/test_blueprints.py (11 tests covering scanner + list + get)
  - mcp-server/tests/fixtures/sample_blueprints/ (ai-research.yaml, data-pipeline.yaml)
  - list_blueprints and get_blueprint registered in server.py

affects:
  - Phase 06-03 (validate_yaml tool uses blueprint names from list_blueprints)
  - Phase 07+ (any tool that needs blueprint discovery or detail)

tech-stack:
  added:
    - PyYAML safe_load_all for multi-doc YAML parsing
    - pathlib.Path.rglob for recursive YAML discovery
  patterns:
    - scan_blueprints() returns internal {"source_file", "manifest"} dicts — never exposed directly to agents
    - flatten_blueprint_summary() and flatten_blueprint_detail() produce flat agent-facing JSON (<=2 key traversals)
    - register_blueprint_tools(mcp) pattern consistent with register_*(mcp) convention from Plan 01

key-files:
  created:
    - mcp-server/tools/blueprints.py
    - mcp-server/tests/test_blueprints.py
    - mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml
    - mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml
  modified:
    - mcp-server/server.py (added register_blueprint_tools import and call)

key-decisions:
  - "scan_blueprints() returns internal wrapper dicts, not raw manifests — flatten functions do the agent-facing shaping"
  - "flatten_blueprint_detail() flattens helm_chart sub-dict to helm_chart_name/version/repository/release_name fields — keeps 2-key depth contract"
  - "Missing or empty BLUEPRINTS_DIR produces warning in response, not exception — graceful degradation"
  - "get_blueprint unknown name returns structured error with available_names list — enables agent recovery"

patterns-established:
  - "Blueprint scanner: filter by apiVersion.startswith('warp.io') and kind == 'WekaAppStore'"
  - "Flat component pattern: no nested helm_chart sub-dicts; all fields hoisted to component top level"
  - "Empty-dir warning pattern: 'No blueprints found in BLUEPRINTS_DIR — directory may be empty or not yet synced'"

requirements-completed: [MCPS-04, MCPS-05]

duration: ~8min
completed: 2026-03-20
---

# Phase 6 Plan 2: Blueprint Scanner and list_blueprints/get_blueprint Tools Summary

**Directory-driven WekaAppStore blueprint scanner with two MCP tools — list_blueprints returning flat catalog metadata and get_blueprint returning full flattened component detail including hoisted helm_chart fields, both handling missing dirs gracefully.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-20T06:08:41Z
- **Completed:** 2026-03-20T06:16:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Blueprint scanner discovers WekaAppStore CRs from directory — not hardcoded, filter by apiVersion/kind
- list_blueprints returns flat catalog (count, per-entry name/namespace/component_count/component_names/source_file, captured_at, warnings)
- get_blueprint returns full flat detail with helm_chart fields hoisted — no nested sub-objects, 2-key depth contract holds
- Unknown blueprint name returns structured error with available_names list for agent recovery
- 11 tests pass covering scanner, list, get, flat-response depth, and component flattening

## Task Commits

Each task was committed atomically:

1. **Task 1: Blueprint scanner and list_blueprints tool (RED)** - `bb670f7` (test)
2. **Task 1: Blueprint scanner and list_blueprints tool (GREEN)** - `7b0bec2` (feat)

_Note: Task 2 (get_blueprint) was implemented in the same implementation commit as Task 1. Tests for both tasks were written together in the RED commit since get_blueprint depends on the same scan_blueprints() infrastructure._

## Files Created/Modified

- `mcp-server/tools/blueprints.py` - scan_blueprints(), flatten_blueprint_summary(), flatten_blueprint_detail(), register_blueprint_tools() with list_blueprints and get_blueprint
- `mcp-server/tests/test_blueprints.py` - 11 tests: scanner (4), list_blueprints (3), get_blueprint (4)
- `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` - WekaAppStore CR with 2 components (vector-db, research-api)
- `mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml` - WekaAppStore CR with 1 component (spark-operator), 1 prerequisite
- `mcp-server/server.py` - Added register_blueprint_tools import and call

## Decisions Made

1. **Internal vs. agent-facing shape separation** — scan_blueprints() returns {"source_file", "manifest"} internal dicts. Separate flatten functions do all agent-facing shaping. Same pattern as Plan 01's inspect tools.

2. **helm_chart sub-dict flattened** — flatten_blueprint_detail() hoists helm_chart fields to top-level component keys (helm_chart_name, helm_chart_version, etc.) to maintain 2-key traversal depth contract. No nested helm_chart object exposed to agent.

3. **source_file is basename only** — flatten_blueprint_summary() uses Path.name to strip directory path, keeping agent response clean and path-independent.

4. **Graceful degradation for missing dir** — Both tools call scan_blueprints() which returns [] for missing/non-directory paths. list_blueprints adds a warning. get_blueprint returns available_names=[] with structured error. No exceptions propagate to the agent.

## Deviations from Plan

None — plan executed exactly as written. Task 2 get_blueprint implementation was written in the same pass as Task 1 since they share the same scan_blueprints() infrastructure; test RED phase covered both tasks together.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required. BLUEPRINTS_DIR defaults to `/app/manifests/manifest` and falls back gracefully to warnings when not present.

## Next Phase Readiness

- list_blueprints and get_blueprint tools registered and tested — ready for Plan 03 (validate_yaml tool)
- Blueprint scanner infrastructure reusable for any tool that needs manifest access
- Fixture YAMLs in tests/fixtures/sample_blueprints/ available for subsequent plan tests

---
*Phase: 06-mcp-scaffold-and-read-only-tools*
*Completed: 2026-03-20*

## Self-Check: PASSED
