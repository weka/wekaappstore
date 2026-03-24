---
phase: 13-kubernetes-manifests-and-sidecar-wiring
plan: "02"
subsystem: infra
tags: [kubernetes, mcp, sidecar, git-sync, openclaw, sandbox-cr, agent-sandbox, busybox, readiness-probe]

# Dependency graph
requires:
  - phase: 13-kubernetes-manifests-and-sidecar-wiring plan 01
    provides: mcp-rbac.yaml (ServiceAccount + RBAC), mcp-skill-configmap.yaml (SKILL.md ConfigMap)
  - phase: 12-nemoclaw-eks-topology
    provides: validated Sandbox CR structure, TOPOLOGY.md integration points
  - phase: 11-streamable-http-transport
    provides: MCP server image (wekachrisjen/weka-app-store-mcp) on port 8080 with /health endpoint
provides:
  - Complete openclaw-sandbox.yaml Sandbox CR with init container, MCP sidecar, git-sync, and all volumes wired
  - validate-phase13.sh: dry-run and live validation script for Phase 13 manifests
affects:
  - phase-14 (integration testing and live deployment)
  - any operator deploying OpenClaw with WEKA App Store MCP in EKS

# Tech tracking
tech-stack:
  added:
    - busybox:1.36 (init container for openclaw.json generation)
    - registry.k8s.io/git-sync/git-sync:v4.5.0 (continuous blueprint sync)
  patterns:
    - Init container writes runtime config (openclaw.json) via printf to shared emptyDir before regular containers start
    - subPath mount dereferences git-sync symlink so container sees plain directory
    - openclaw-config emptyDir is write-once (init container only), read-many (openclaw + MCP sidecar)
    - Readiness probe on /health:8080 gates MCP sidecar readiness status

key-files:
  created:
    - scripts/validate-phase13.sh
  modified:
    - k8s/agent-sandbox/openclaw-sandbox.yaml

key-decisions:
  - "printf used in init container (not heredoc) to avoid shell variable expansion when writing JSON"
  - "subPath: blueprints on MCP sidecar mount dereferences git-sync symlink at /blueprints/blueprints"
  - "openclaw-config mounted readOnly in weka-mcp-sidecar; write is exclusive to init container to prevent race conditions"
  - "git-sync runs without --one-time flag; continuous sync every 60s is the locked decision"
  - "runAsUser 1000 for init container matches fsGroup so openclaw.json is group-readable by MCP sidecar (uid 10001, fsGroup 1000)"

patterns-established:
  - "Phase 13 validate script pattern: dry-run mode always runs; live mode only with --live flag; WARN does not fail; exit 1 on any FAIL"
  - "Sandbox CR comments document WHY each addition was made (K8S-0x requirement references inline)"

requirements-completed: [K8S-01, K8S-03, K8S-04, K8S-05, NCLAW-02]

# Metrics
duration: 8min
completed: 2026-03-24
---

# Phase 13 Plan 02: Kubernetes Sidecar Wiring Summary

**MCP sidecar + git-sync + openclaw.json init container fully wired into OpenClaw Sandbox CR, with printf-based runtime config generation and subPath-dereferenced blueprint volume mount.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-24T05:49:00Z
- **Completed:** 2026-03-24T05:57:02Z
- **Tasks:** 2
- **Files modified:** 2 (1 modified, 1 created)

## Accomplishments

- Sandbox CR updated with all Phase 13 components: serviceAccountName, 1 init container, 3 regular containers (openclaw + weka-mcp-sidecar + git-sync), 3 volumes (openclaw-config + blueprints + skill-md), and 1 PVC template
- openclaw.json generated at pod startup by busybox init container using printf with env vars — transport=streamable-http, url=http://localhost:8080/mcp, skill path set via SKILL_MD_PATH env var
- git-sync runs continuously (no --one-time) syncing warp-blueprints repo every 60s; MCP sidecar accesses blueprints via subPath mount that dereferences git-sync symlink
- validate-phase13.sh created following validate-topology.sh pattern — dry-run mode passes all 4 checks against authored manifests; live mode checks defined for cluster verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Update Sandbox CR with init container, MCP sidecar, git-sync, and volumes** - `f60b683` (feat)
2. **Task 2: Create Phase 13 validation script** - `3e54e2c` (feat)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified

- `k8s/agent-sandbox/openclaw-sandbox.yaml` - Complete Sandbox CR with Phase 13 sidecar wiring; passes kubectl dry-run
- `scripts/validate-phase13.sh` - Dry-run and live validation script; all dry-run checks pass

## Decisions Made

- **printf over heredoc for openclaw.json**: Research pitfall documented in PLAN — heredoc triggers unexpected shell variable expansion when writing JSON. printf with positional args is safe.
- **subPath: blueprints on MCP sidecar mount**: git-sync writes to /blueprints/blueprints (a symlink); without subPath the container sees a dangling symlink. subPath dereferences it so BLUEPRINTS_DIR=/app/blueprints resolves to a real directory.
- **openclaw-config readOnly in weka-mcp-sidecar**: The init container is the sole writer. Marking readOnly in the sidecar prevents accidental overwrites and makes the design intent explicit.
- **runAsUser 1000 for init container**: Matches pod fsGroup (1000) so openclaw.json has correct group ownership and is readable by both OpenClaw (uid 1000) and MCP sidecar (uid 10001, which gets gid 1000 via fsGroup).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. kubectl dry-run passed on first attempt. All 6 structural checks in validate-phase13.sh passed immediately.

## User Setup Required

None — this plan only modifies manifests and creates a script. Cluster deployment is the Phase 14 concern.

## Next Phase Readiness

- `k8s/agent-sandbox/openclaw-sandbox.yaml` is deployment-ready; all Phase 13 structural requirements satisfied
- `bash scripts/validate-phase13.sh` confirms dry-run validity; `--live` flag ready for post-deployment verification
- Phase 14 (integration testing) can apply this manifest and run `bash scripts/validate-phase13.sh --live wekaappstore` to confirm sidecar health end-to-end

---
*Phase: 13-kubernetes-manifests-and-sidecar-wiring*
*Completed: 2026-03-24*
