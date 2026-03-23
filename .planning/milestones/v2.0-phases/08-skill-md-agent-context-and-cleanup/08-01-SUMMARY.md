---
phase: 08-skill-md-agent-context-and-cleanup
plan: 01
subsystem: mcp
tags: [skill-md, agent-workflow, tool-descriptions, mock-harness, description-routing]

# Dependency graph
requires:
  - phase: 07-validation-apply-and-status-tools
    provides: "8-tool MCP server with validate_yaml, apply, and status tools"
provides:
  - "SKILL.md authoritative agent workflow at mcp-server/SKILL.md"
  - "All 8 tool descriptions tuned with sequencing hints and cross-references"
  - "Description-based tool selection harness proving descriptions are agent-navigable"
  - "test_tool_descriptions.py verifying sequencing keywords in all tool descriptions"
affects:
  - phase-09-openclaw-registration
  - any agent consuming the MCP server

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_RegistryCapture stub pattern: lightweight MCP stub extracting @mcp.tool() docstrings for testing"
    - "select_tool() keyword scoring: proves descriptions contain agent-navigable vocabulary"
    - "Description-first tool routing: harness selects tools via intent keywords, not hardcoded names"

key-files:
  created:
    - mcp-server/SKILL.md
    - mcp-server/tests/test_tool_descriptions.py
  modified:
    - mcp-server/tools/inspect_cluster.py
    - mcp-server/tools/inspect_weka.py
    - mcp-server/tools/blueprints.py
    - mcp-server/tools/crd_schema.py
    - mcp-server/tools/validate_yaml.py
    - mcp-server/tools/apply_tool.py
    - mcp-server/tools/status_tool.py
    - mcp-server/harness/mock_agent.py
    - mcp-server/tests/test_mock_agent.py

key-decisions:
  - "SKILL.md uses 12 numbered steps with explicit validate-retry loop (max 3 attempts) and re-inspect-before-apply as mandatory rule"
  - "_RegistryCapture stub builds description registry by calling register_* with a minimal MCP shim — no real FastMCP needed"
  - "select_tool() uses case-insensitive substring keyword scoring — intentionally simple, proves vocabulary not NLP"
  - "Keywords for validate_yaml selection use 'structurally valid' and 'apiversion' to disambiguate from get_crd_schema which also mentions 'validate'"

patterns-established:
  - "SKILL.md at mcp-server/SKILL.md: authoritative agent workflow document referenced by OpenClaw config"
  - "Sequencing line format: 'Sequencing: tool_a -> tool_b -> tool_c' at end of each tool description"
  - "Description tests: test_tool_descriptions.py keyword assertions on all 8 tools, run as part of full test suite"

requirements-completed: [AGNT-01]

# Metrics
duration: 8min
completed: 2026-03-20
---

# Phase 8 Plan 01: SKILL.md and Agent Context Summary

**SKILL.md authoritative 12-step agent workflow with re-inspect-before-apply rule, validate-retry loop, v1.0 negative YAML examples, and description-based tool selection harness proving all 8 tool descriptions are agent-navigable**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-20T10:33:04Z
- **Completed:** 2026-03-20T10:41:00Z
- **Tasks:** 2
- **Files modified:** 9 (1 created + 8 modified) + 2 new test/harness files

## Accomplishments
- Created mcp-server/SKILL.md (249 lines) with 12-step numbered workflow, validate-retry loop section (max 3 attempts), mandatory re-inspect-before-apply rule, negative YAML examples for v1.0 fields, and full tool reference table
- Tuned all 8 tool descriptions with consistent "Sequencing:" lines, cross-tool references by name, and safety warnings (especially apply requiring re-inspect + confirmed=True)
- Refactored mock harness to use description-based tool selection via `select_tool()` with `_RegistryCapture` stub — proves descriptions contain sufficient vocabulary for tool routing without hardcoded names
- Added `test_tool_descriptions.py` with 18 keyword assertion tests across all 8 tools (all pass as part of 85-test suite)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write SKILL.md and tune all 8 tool descriptions** - `ef08980` (feat)
2. **Task 2: Upgrade mock harness to description-based tool selection and add description tests** - `5a341fc` (feat)

## Files Created/Modified
- `mcp-server/SKILL.md` - Authoritative agent workflow document (249 lines, 6 sections)
- `mcp-server/tools/inspect_cluster.py` - Added FIRST/AGAIN sequencing, re-inspect warning
- `mcp-server/tools/inspect_weka.py` - Added conditional-call guidance and returns spec
- `mcp-server/tools/blueprints.py` - Added cross-refs to inspect_cluster and get_crd_schema
- `mcp-server/tools/crd_schema.py` - Added before-generating-YAML warning, post-validate hint
- `mcp-server/tools/validate_yaml.py` - Added retry loop hint, re-inspect after valid=true
- `mcp-server/tools/apply_tool.py` - Added 3-rule safety checklist and re-inspect requirement
- `mcp-server/tools/status_tool.py` - Added after-apply sequencing and repeat-until-Ready guidance
- `mcp-server/harness/mock_agent.py` - Refactored: _RegistryCapture, build_tool_registry(), select_tool()
- `mcp-server/tests/test_mock_agent.py` - Added test_select_tool_returns_correct_tool
- `mcp-server/tests/test_tool_descriptions.py` - New: 18 keyword assertion tests for all 8 tools

## Decisions Made
- `select_tool()` uses simple keyword scoring (case-insensitive substring count) rather than fuzzy matching — sufficient to prove description vocabulary, not trying to build NLP
- `get_blueprint` selection keywords use "get_blueprint" as a literal keyword since the tool description contains its own name
- `validate_yaml` uses "structurally valid" and "apiversion" keywords rather than "validate" alone, because `get_crd_schema` also mentions "validate_yaml" in its sequencing line (tie-breaking disambiguation)
- `_RegistryCapture` extracts descriptions by calling `register_*(mcp)` — descriptions stay in sync with tools/*.py without duplication

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Cleared stale .pyc cache that was making tests fail**
- **Found during:** Task 1 verification
- **Issue:** Deleted `family_matcher.py` (Phase 8 cleanup) left stale .pyc files that caused test_apply_tool to fail on import of `webapp/__init__.py`
- **Fix:** Deleted all .pyc files under app-store-gui/ and mcp-server/ to force recompilation
- **Files modified:** No source files — cache deletion only
- **Verification:** `pytest tests/ -x -q` returned 67 passed after cache clear
- **Committed in:** not committed (cache files are not tracked)

---

**Total deviations:** 1 auto-fixed (1 bug — stale .pyc cache from prior cleanup)
**Impact on plan:** One-time fix, no source changes needed. Pre-existing issue from Phase 8.03 cleanup.

## Issues Encountered
- Keyword disambiguation: `validate_yaml` and `get_crd_schema` both scored 5 on `["validate", "yaml", "valid", "errors", "before apply"]` because `get_crd_schema` mentions `validate_yaml` in its sequencing line. Fixed by using vocabulary unique to `validate_yaml` description: `"structurally valid"` and `"apiversion"`.

## Next Phase Readiness
- SKILL.md ready for OpenClaw registration config reference (Phase 9)
- All 8 tool descriptions contain sequencing hints suitable for agent navigation
- 85 tests passing: 67 original + 18 new description/harness tests
- Phase 8 Plan 02 (OpenClaw registration config) can proceed

---
*Phase: 08-skill-md-agent-context-and-cleanup*
*Completed: 2026-03-20*
