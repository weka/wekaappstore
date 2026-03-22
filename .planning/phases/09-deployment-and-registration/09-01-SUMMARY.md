---
phase: 09-deployment-and-registration
plan: 01
subsystem: infra
tags: [docker, dockerfile, mcp-server, python, config, env-vars, startup-validation]

# Dependency graph
requires:
  - phase: 08-skill-md-agent-context-and-cleanup
    provides: "Cleaned-up mcp-server/ with all 8 tools and server.py entry point"
provides:
  - "Buildable Dockerfile for mcp-server container image using python:3.10-slim"
  - "config.validate_required() for fail-fast BLUEPRINTS_DIR startup validation"
  - "WEKA_ENDPOINT and KUBECONFIG optional env vars in config.py"
  - ".dockerignore excluding build artifacts, tests, .git, .planning"
  - "server.py calls validate_required() at startup before mcp.run()"
affects: [09-02-openclaw-registration, deployment, container-runtime]

# Tech tracking
tech-stack:
  added: [Dockerfile, .dockerignore]
  patterns: [fail-fast startup validation, TDD red-green for config tests]

key-files:
  created:
    - mcp-server/Dockerfile
    - mcp-server/.dockerignore
    - mcp-server/tests/test_config.py
  modified:
    - mcp-server/config.py
    - mcp-server/server.py

key-decisions:
  - "validate_required() NOT called at import time — avoids side effects during pytest collection"
  - "BLUEPRINTS_DIR defaults to empty string; validate_required() uses os.environ.get() check (not module-level constant) to detect unset env var correctly after monkeypatching"
  - "WEKA_ENDPOINT and KUBECONFIG use 'or None' idiom to normalize empty string to None"
  - "Dockerfile import sanity check (python -c 'import server') during build fails the image if deps are missing"
  - "Non-root user mcpuser (uid 10001) created and activated in Dockerfile for runtime security"

patterns-established:
  - "TDD for config validation: RED commit tests first, GREEN commit implementation, verify 100% pass"
  - "validate_required() pattern: explicit startup call from __main__, not import-time side effect"

requirements-completed: [DEPLOY-01, DEPLOY-02, DEPLOY-03]

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 9 Plan 01: Dockerfile, config startup validation, and .dockerignore Summary

**python:3.10-slim Dockerfile with PYTHONPATH for mcp-server + app-store-gui, fail-fast BLUEPRINTS_DIR validation via validate_required(), and WEKA_ENDPOINT/KUBECONFIG optional env vars**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T22:54:27Z
- **Completed:** 2026-03-22T22:56:04Z
- **Tasks:** 2 (Task 1 TDD: 3 commits; Task 2: 1 commit)
- **Files modified:** 5

## Accomplishments

- config.py extended with WEKA_ENDPOINT, KUBECONFIG, and validate_required() — no import-time side effects
- Dockerfile packages mcp-server + app-store-gui/webapp in python:3.10-slim with build-time import check and non-root user
- 7 new config tests pass alongside the full 100-test suite (no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Startup env var validation tests** - `1af49a7` (test)
2. **Task 1 GREEN: config.py implementation** - `8096f33` (feat)
3. **Task 2: Dockerfile, .dockerignore, server.py update** - `f7a0985` (feat)

_Note: TDD task has RED (test) + GREEN (feat) commits as required by TDD execution flow._

## Files Created/Modified

- `mcp-server/Dockerfile` - Container image definition using python:3.10-slim; copies mcp-server/ and app-store-gui/webapp/; sets PYTHONPATH; build-time import check; non-root user
- `mcp-server/.dockerignore` - Excludes __pycache__, tests/, .git, .planning, .env, .DS_Store
- `mcp-server/config.py` - Added validate_required(), WEKA_ENDPOINT (optional), KUBECONFIG (optional)
- `mcp-server/server.py` - __main__ block now calls validate_required() before mcp.run()
- `mcp-server/tests/test_config.py` - 7 tests covering validate_required() fail-fast and all env vars

## Decisions Made

- validate_required() checks `os.environ.get("BLUEPRINTS_DIR")` (not the module-level constant) so monkeypatching in tests works correctly after module reload
- WEKA_ENDPOINT and KUBECONFIG use `os.environ.get("VAR") or None` to normalize empty strings to None
- Dockerfile build from repo root (`docker build -f mcp-server/Dockerfile .`) per plan spec

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Container image is buildable: `docker build -f mcp-server/Dockerfile .`
- server.py enforces BLUEPRINTS_DIR at startup
- Phase 09-02 (OpenClaw registration) can proceed — openclaw.json already exists from Phase 08

---
*Phase: 09-deployment-and-registration*
*Completed: 2026-03-22*
