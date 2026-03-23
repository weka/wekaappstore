---
phase: 09-deployment-and-registration
plan: 02
subsystem: infra
tags: [github-actions, ci-cd, docker, readme, openclaw, nemoclaw, documentation]

# Dependency graph
requires:
  - phase: 09-deployment-and-registration
    plan: 01
    provides: "Buildable Dockerfile with python:3.10-slim, config.validate_required(), non-root mcpuser"

provides:
  - "GitHub Actions CI workflow running pytest on every PR touching mcp-server/"
  - "GitHub Actions build-push job publishing to wekachrisjen/weka-app-store-mcp on v* tag"
  - "mcp-server/README.md with Quick Start, env vars table, OpenClaw registration, NemoClaw TODO placeholder, CI/CD setup, and Troubleshooting"

affects: [openclaw-registration, nemoclaw-registration, deployment]

# Tech tracking
tech-stack:
  added: [GitHub Actions, docker/login-action@v3, docker/metadata-action@v5, docker/build-push-action@v6]
  patterns: [test-gate-before-push, semver-plus-latest tagging, stdio smoke test via echo-pipe]

key-files:
  created:
    - .github/workflows/mcp-server.yml
    - mcp-server/README.md

key-decisions:
  - "build-push job guarded by both needs: test and if: startsWith(github.ref, refs/tags/v) — tag push without passing tests cannot publish image"
  - "docker/metadata-action generates both semver tag and latest simultaneously — no manual re-tag step in CI"
  - "NemoClaw section uses best-effort fields with visible TODO marker — schema not yet published"
  - "README smoke test uses BLUEPRINTS_DIR=/tmp so it works without a real blueprint directory"

patterns-established:
  - "CI test-gate pattern: all deploy jobs list needs: test to enforce green-before-push"
  - "PYTHONPATH env in GitHub Actions matches local dev command exactly (mcp-server:app-store-gui)"

requirements-completed: [DEPLOY-01, DEPLOY-02, DEPLOY-04]

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 9 Plan 02: GitHub Actions CI/CD and mcp-server README Summary

**GitHub Actions workflow running pytest on every PR and building/pushing to wekachrisjen/weka-app-store-mcp on v* tag, plus complete README with copy-paste Quick Start, env vars table, OpenClaw registration from openclaw.json, NemoClaw TODO placeholder, and Troubleshooting guide**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T22:58:08Z
- **Completed:** 2026-03-22T23:00:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- CI workflow gates image publication behind passing tests — `build-push` job depends on `test` and only runs on `v*` tag push
- README covers every operator step end-to-end: build, tag, push, OpenClaw registration, NemoClaw placeholder, smoke test, GitHub Secrets setup, troubleshooting
- All 100 tests continue to pass after adding both files (no regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1: GitHub Actions CI/CD workflow** - `6cc16c9` (chore)
2. **Task 2: README with registration documentation** - `79edd27` (feat)

## Files Created/Modified

- `.github/workflows/mcp-server.yml` - Two-job pipeline: `test` (pytest on PR/push) and `build-push` (Docker Hub on v* tag, needs: test)
- `mcp-server/README.md` - 9-section operator guide: Quick Start, Building the Image, Environment Variables, OpenClaw Registration, NemoClaw Registration, Verify It Works, CI/CD, Troubleshooting

## Decisions Made

- `build-push` uses both `needs: test` and `if: startsWith(github.ref, 'refs/tags/v')` — tag push without passing tests cannot publish the image
- `docker/metadata-action` generates semver tag (`v2.0.0` → `2.0.0`) and `latest` simultaneously — no manual re-tag step
- NemoClaw section documents known fields (transport, command, env, skill) with a visible TODO marker per user decision
- README smoke test uses `BLUEPRINTS_DIR=/tmp` — server starts without error, returns empty blueprint list rather than failing, so the tools/list probe works without real blueprint data

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

One-time GitHub Secrets setup required before the `build-push` job can push to Docker Hub (documented in README CI/CD section):
- `DOCKERHUB_USERNAME` — Docker Hub username
- `DOCKERHUB_TOKEN` — Docker Hub access token

## Next Phase Readiness

- Phase 9 complete — all deployment and registration artifacts are in place
- Container image is buildable: `docker build -f mcp-server/Dockerfile .`
- CI pipeline triggers automatically on PR and v* tag
- OpenClaw registration: copy `mcp-server/openclaw.json`, update `container` field to versioned tag
- NemoClaw: placeholder documented, pending schema publication

## Self-Check

---
*Phase: 09-deployment-and-registration*
*Completed: 2026-03-22*
