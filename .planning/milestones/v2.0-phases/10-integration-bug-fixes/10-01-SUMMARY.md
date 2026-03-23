---
phase: 10-integration-bug-fixes
plan: "01"
subsystem: mcp-server
tags: [mcp, logging, blueprints, openclaw, yaml, pythonpath]

# Dependency graph
requires:
  - phase: 09-deployment-and-registration
    provides: generate_openclaw_config.py and openclaw.json structure
  - phase: 06-mcp-scaffold-and-read-only-tools
    provides: scan_blueprints(), blueprints.py scanner, server.py logging setup
provides:
  - blueprints.py logger.warning() without TypeError on malformed YAML
  - LOG_LEVEL env var wired to logging.basicConfig in server.py
  - PYTHONPATH in openclaw.json startup block for non-container registration
  - Three regression tests covering all three defects
affects: [09-deployment-and-registration, 06-mcp-scaffold-and-read-only-tools]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "logger.warning() with %s placeholders only — no file= kwarg (print() only)"
    - "getattr(logging, config.LOG_LEVEL, logging.INFO) pattern for dynamic log level"
    - "startup.env.PYTHONPATH in openclaw.json for non-container MCP registration"

key-files:
  created:
    - mcp-server/tests/test_blueprints.py (test_scan_blueprints_skips_malformed_yaml added)
    - mcp-server/tests/test_logging.py (test_log_level_env_var added)
    - mcp-server/tests/test_openclaw_config.py (test_startup_env_includes_pythonpath added)
  modified:
    - mcp-server/tools/blueprints.py
    - mcp-server/server.py
    - mcp-server/generate_openclaw_config.py
    - mcp-server/openclaw.json

key-decisions:
  - "Remove import sys from blueprints.py entirely — no other usage existed"
  - "config imported before logging.basicConfig using noqa: E402 comment pattern consistent with FastMCP import"
  - "PYTHONPATH value '.:../app-store-gui' matches usage docstring in generate_openclaw_config.py"

patterns-established:
  - "logger.warning() signature: (msg, arg1, arg2) — never mix with print() kwargs"
  - "env var wiring: getattr(logging, config.LOG_LEVEL, logging.INFO) with safe fallback"

requirements-completed: [MCPS-04, MCPS-05, MCPS-10, MCPS-11, DEPLOY-03, AGNT-03, DEPLOY-04]

# Metrics
duration: 12min
completed: 2026-03-23
---

# Phase 10 Plan 01: Integration Bug Fixes Summary

**Three v2.0 audit defects eliminated: blueprints.py logger TypeError removed, LOG_LEVEL env var wired to basicConfig, PYTHONPATH added to openclaw.json startup block with 3 regression tests**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-22T23:18:00Z
- **Completed:** 2026-03-22T23:29:48Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Fixed `logger.warning("...", file=sys.stderr)` TypeError in `scan_blueprints()` — `file=` is a `print()` kwarg, not a `logging` kwarg; removed `import sys` as no longer needed
- Wired `LOG_LEVEL` environment variable to `logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))` in `server.py`; `config` is now imported before `basicConfig`
- Added `PYTHONPATH: ".:../app-store-gui"` to the `startup.env` block in both `generate_openclaw_config.py` and the regenerated `openclaw.json`
- Added 3 regression tests: `test_scan_blueprints_skips_malformed_yaml`, `test_log_level_env_var`, `test_startup_env_includes_pythonpath`
- Full test suite: 103 tests pass (100 pre-existing + 3 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix blueprints.py logger TypeError and wire LOG_LEVEL** - `67ef7da` (fix)
2. **Task 2: Add PYTHONPATH to openclaw.json startup and regenerate** - `9381879` (feat)

**Plan metadata:** `[pending final commit]` (docs: complete plan)

## Files Created/Modified
- `mcp-server/tools/blueprints.py` - Removed `file=sys.stderr` from `logger.warning()`, removed `import sys`
- `mcp-server/server.py` - Added `import config` before `basicConfig`; wired `config.LOG_LEVEL`
- `mcp-server/generate_openclaw_config.py` - Added `env: {PYTHONPATH: ".:../app-store-gui"}` to startup block
- `mcp-server/openclaw.json` - Regenerated to include `startup.env.PYTHONPATH`
- `mcp-server/tests/test_blueprints.py` - Added `test_scan_blueprints_skips_malformed_yaml`
- `mcp-server/tests/test_logging.py` - Added `test_log_level_env_var`
- `mcp-server/tests/test_openclaw_config.py` - Added `test_startup_env_includes_pythonpath`

## Decisions Made
- Removed `import sys` from `blueprints.py` entirely since no other usage existed after the fix
- Used `noqa: E402` comment pattern on `import config` in `server.py` consistent with how FastMCP import is annotated
- Set `PYTHONPATH` value to `".:../app-store-gui"` matching the usage docstring in `generate_openclaw_config.py`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three integration defects from the v2.0 milestone audit are closed
- 103 tests passing with zero regressions
- Ready for v2.0 milestone sign-off

---
*Phase: 10-integration-bug-fixes*
*Completed: 2026-03-23*
