---
phase: 11-streamable-http-transport
plan: 02
subsystem: infra
tags: [mcp, openclaw, streamable-http, transport, dockerfile]

# Dependency graph
requires:
  - phase: 11-streamable-http-transport/11-01
    provides: MCP_TRANSPORT and MCP_PORT env var support in config.py
provides:
  - openclaw.json with transport=streamable-http and url=http://localhost:8080/mcp
  - generate_openclaw_config.py updated to emit HTTP transport structure
  - Dockerfile with EXPOSE 8080 directive
  - Updated test assertions for streamable-http transport
affects: [phase-12, phase-13, nemoclaw-registration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD for config schema changes: write failing assertions against current file, then update generator and regenerate"
    - "Drift detection test pattern: test_openclaw_json_matches_generation compares on-disk JSON to in-memory generator output"

key-files:
  created: []
  modified:
    - mcp-server/openclaw.json
    - mcp-server/generate_openclaw_config.py
    - mcp-server/tests/test_openclaw_config.py
    - mcp-server/Dockerfile

key-decisions:
  - "Removed startup block from openclaw.json entirely — HTTP transport uses url for discovery, subprocess spawn no longer needed"
  - "Added MCP_TRANSPORT and MCP_PORT to env.optional so NemoClaw knows these vars exist but doesn't require them"
  - "EXPOSE 8080 added to Dockerfile as documentation/K8s discovery signal, no runtime effect"

patterns-established:
  - "openclaw.json regenerated from generator, never hand-edited — drift detection test enforces this"

requirements-completed: [XPORT-01, XPORT-02]

# Metrics
duration: 12min
completed: 2026-03-23
---

# Phase 11 Plan 02: OpenClaw Config Streamable-HTTP Transport Summary

**openclaw.json migrated from stdio subprocess spawn to streamable-http URL registration, with EXPOSE 8080 in Dockerfile and all 11 config tests passing including drift detection**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-23T02:20:00Z
- **Completed:** 2026-03-23T02:32:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- openclaw.json now declares transport=streamable-http and url=http://localhost:8080/mcp with no startup block
- generate_openclaw_config.py updated to produce the new HTTP transport structure (generator and file stay in sync)
- MCP_TRANSPORT and MCP_PORT added to env.optional so NemoClaw can document their existence without requiring them
- Dockerfile updated with EXPOSE 8080 for K8s service discovery documentation
- Removed test_startup_env_includes_pythonpath (startup block gone); added 4 new HTTP-transport assertions; all 11 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Update tests (TDD RED)** - `67b41f8` (test)
2. **Task 1: Update generator and regenerate openclaw.json (TDD GREEN)** - `198c9c7` (feat)
3. **Task 2: Add EXPOSE 8080 to Dockerfile** - `a27c886` (feat)

## Files Created/Modified
- `mcp-server/openclaw.json` - Updated to streamable-http transport, url, no startup block, extended env.optional
- `mcp-server/generate_openclaw_config.py` - build_openclaw_config() produces HTTP transport structure; docstring updated
- `mcp-server/tests/test_openclaw_config.py` - 4 new HTTP assertions; stdio and startup tests removed; module docstring updated
- `mcp-server/Dockerfile` - Added EXPOSE 8080; CMD comment updated to reflect dual-mode transport

## Decisions Made
- Removed the entire startup block from openclaw.json. With streamable-http transport NemoClaw connects to the URL directly — no subprocess spawn needed.
- MCP_TRANSPORT and MCP_PORT go in env.optional (not required) because the server defaults to stdio when these vars are absent, preserving backward compatibility with CI and local dev.
- EXPOSE 8080 is documentation-only in Docker; it has no runtime effect but signals to K8s service meshes and developers which port the HTTP transport uses.

## Deviations from Plan

None - plan executed exactly as written. TDD flow followed: RED commit of updated tests against old openclaw.json, then GREEN commit after updating generator and regenerating.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- openclaw.json is NemoClaw-registration-ready with correct HTTP transport structure
- Dockerfile documents port 8080 for K8s service exposure in Phase 13 manifests
- Phase 12 (EKS topology validation) can proceed — the registration config format is locked

## Self-Check: PASSED

- FOUND: mcp-server/openclaw.json
- FOUND: mcp-server/generate_openclaw_config.py
- FOUND: mcp-server/tests/test_openclaw_config.py
- FOUND: mcp-server/Dockerfile
- FOUND: .planning/phases/11-streamable-http-transport/11-02-SUMMARY.md
- FOUND: commit 67b41f8 (TDD RED)
- FOUND: commit 198c9c7 (TDD GREEN)
- FOUND: commit a27c886 (Task 2)

---
*Phase: 11-streamable-http-transport*
*Completed: 2026-03-23*
