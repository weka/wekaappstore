---
phase: 08-skill-md-agent-context-and-cleanup
plan: 02
subsystem: mcp
tags: [openclaw, registration-config, tool-descriptions, drift-detection, stdio]

# Dependency graph
requires:
  - phase: 08-skill-md-agent-context-and-cleanup
    provides: "8-tool MCP server with SKILL.md and tuned tool descriptions"
provides:
  - "mcp-server/openclaw.json: OpenClaw registration config with all 8 tools, stdio transport, env var declarations"
  - "mcp-server/generate_openclaw_config.py: auto-generation script that reads server.py tool registrations"
  - "mcp-server/tests/test_openclaw_config.py: 8 tests including drift detection between file and live docstrings"
affects:
  - phase-09-openclaw-registration
  - any OpenClaw/NemoClaw agent integration consuming the MCP server

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_RegistryCapture reuse: generate_openclaw_config.py reuses the same _RegistryCapture stub pattern from mock_agent.py to extract tool names/descriptions without starting FastMCP"
    - "drift-detection test: test_openclaw_json_matches_generation() runs generator in-memory and compares tool descriptions against the on-disk file"

key-files:
  created:
    - mcp-server/openclaw.json
    - mcp-server/generate_openclaw_config.py
    - mcp-server/tests/test_openclaw_config.py
  modified: []

key-decisions:
  - "_RegistryCapture pattern reused from harness/mock_agent.py — same lightweight stub, same approach, descriptions stay in sync without FastMCP startup"
  - "openclaw.json includes skill field pointing to mcp-server/SKILL.md so OpenClaw agent can load workflow before calling tools"
  - "_comment top-level field documents best-effort format pending NemoClaw alpha schema publication"
  - "generate() function in generate_openclaw_config.py accepts optional output_path for testability (test_openclaw_json_matches_generation uses it in-memory)"

patterns-established:
  - "openclaw.json regeneration: run `cd mcp-server && PYTHONPATH=.:../app-store-gui python generate_openclaw_config.py` whenever tool docstrings change"
  - "drift detection: test_openclaw_json_matches_generation() CI gate prevents stale openclaw.json shipping"

requirements-completed: [AGNT-03]

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 8 Plan 02: OpenClaw Registration Config Summary

**openclaw.json auto-generated from server.py tool registrations using _RegistryCapture stub, with 8-test suite including drift detection that fails CI if docstrings change but openclaw.json is not regenerated**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T10:43:52Z
- **Completed:** 2026-03-20T10:45:36Z
- **Tasks:** 2
- **Files modified:** 3 (all created)

## Accomplishments
- Created `mcp-server/generate_openclaw_config.py` (118 lines): reuses `_RegistryCapture` stub pattern from mock harness to call all 7 `register_*` functions and extract tool names/descriptions without starting FastMCP; generates `openclaw.json` with stdio transport, startup command, env var declarations, and `skill` field pointing to SKILL.md
- Generated `mcp-server/openclaw.json`: 8 tools with full docstrings, transport=stdio, startup={command: python, args: [-m, server], cwd: mcp-server/}, env.required=[BLUEPRINTS_DIR], env.optional=[KUBERNETES_AUTH_MODE, LOG_LEVEL, KUBECONFIG], container and skill fields
- Created `mcp-server/tests/test_openclaw_config.py` (153 lines): 8 tests covering file existence, JSON validity, tool count, tool name matching, transport, required env vars, non-empty descriptions, and drift detection (fails if openclaw.json not regenerated after docstring changes)
- Full test suite: 93 passed (was 85) — all 8 new tests pass alongside the 85 existing tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create generate_openclaw_config.py and openclaw.json** - `5315761` (feat)
2. **Task 2: Add tests for openclaw.json validity** - `6848aab` (feat)

## Files Created/Modified
- `mcp-server/generate_openclaw_config.py` - Auto-generation script using _RegistryCapture to read tool registrations from tools/*.py
- `mcp-server/openclaw.json` - OpenClaw registration config: name, transport, startup, env, container, skill, 8 tools with descriptions
- `mcp-server/tests/test_openclaw_config.py` - 8 tests: structure, tool count, tool names, transport, env vars, descriptions, drift detection

## Decisions Made
- `_RegistryCapture` reused verbatim from `harness/mock_agent.py` pattern (no copy-paste duplication — both import from the same conceptual pattern but each file defines its own class locally for isolation)
- `openclaw.json` includes `"skill": "mcp-server/SKILL.md"` so an OpenClaw agent loading this config knows to read the workflow document before calling tools
- `_comment` at top level rather than `_schema_version` — documents format uncertainty without claiming schema conformance
- `generate()` function accepts `output_path` parameter so `test_openclaw_json_matches_generation` can call it in-memory without writing to disk during tests

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## Next Phase Readiness
- `openclaw.json` ready for Phase 9 OpenClaw registration
- `generate_openclaw_config.py` provides regeneration path when tool descriptions change
- `test_openclaw_json_matches_generation()` CI gate ensures config stays in sync
- All 93 mcp-server tests pass

---
*Phase: 08-skill-md-agent-context-and-cleanup*
*Completed: 2026-03-20*
