---
gsd_state_version: 1.0
milestone: v3.0
milestone_name: Live EKS Deployment and Agent Testing
status: planning
stopped_at: Completed 13-02-PLAN.md
last_updated: "2026-03-24T05:58:14.824Z"
last_activity: 2026-03-23 — v3.0 roadmap created; Phases 11-14 defined
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 7
  completed_plans: 6
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-23)

**Core value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.
**Current focus:** Phase 11 — Streamable HTTP Transport

## Current Position

Milestone: v3.0 Live EKS Deployment and Agent Testing
Phase: 11 of 14 (Streamable HTTP Transport)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-23 — v3.0 roadmap created; Phases 11-14 defined

Progress: [░░░░░░░░░░] 0% (v3.0)

## Performance Metrics

**Velocity (v3.0):**
- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*
| Phase 11-streamable-http-transport P02 | 12 | 2 tasks | 4 files |
| Phase 11-streamable-http-transport P01 | 2min | 2 tasks | 4 files |
| Phase 12-nemoclaw-eks-topology P01 | 2min | 2 tasks | 4 files |
| Phase 12-nemoclaw-eks-topology P02 | 15min | 3 tasks | 4 files |
| Phase 13-kubernetes-manifests-and-sidecar-wiring P01 | 2min | 2 tasks | 2 files |
| Phase 13-kubernetes-manifests-and-sidecar-wiring P02 | 8min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

- [v3.0]: NemoClaw deployment uses experimental agent-sandbox CRD approach (user decision; not dedicated EC2 VM)
- [v3.0]: MCP server runs as native Kubernetes sidecar init container (restartPolicy: Always) to guarantee startup ordering
- [v3.0]: Streamable HTTP transport selected via MCP_TRANSPORT env var; stdio remains default for CI and local dev
- [v3.0]: openclaw.json generated at pod startup from env vars via init container; never baked into image
- [v2.0]: Pivot from backend-brain to OpenClaw-native MCP tool registration
- [v2.0]: Flat 2-key depth contract enforced by check_depth() across all 8 tools; 103 tests as regression safety net
- [Phase 11-02]: Removed startup block from openclaw.json — HTTP transport uses url for discovery, no subprocess spawn
- [Phase 11-02]: MCP_TRANSPORT and MCP_PORT in env.optional (not required) preserving stdio backward compat for CI and local dev
- [Phase 11]: FastMCP constructed conditionally at module level (not in __main__) so tests can import server.mcp without transport side effects
- [Phase 11]: stateless_http=True required in HTTP mode to avoid session ID forwarding issues with OpenClaw client (XPORT-04)
- [Phase 12-nemoclaw-eks-topology]: Sandbox CR has no hardcoded namespace — applied with kubectl -n <NAMESPACE> to match locked decision of same namespace as WEKA App Store components
- [Phase 12-nemoclaw-eks-topology]: Loopback :8080 probe in smoke test is WARN-only (exit 7 = PASS, exit 28 = WARN) since MCP sidecar does not exist until Phase 13
- [Phase 12-nemoclaw-eks-topology]: OpenClaw gateway uses --bind=loopback (not --bind=lan) for Sandbox CR deployment; non-loopback requires controlUi config not available in containerized mode
- [Phase 12-nemoclaw-eks-topology]: Phase 12 gate resolved: OpenClaw deployed via agent-sandbox Sandbox CRD in wekaappstore namespace; NCLAW-01 and NCLAW-03 validated; topology documented in TOPOLOGY.md
- [Phase 13-01]: delete verb included for wekaappstores resource — research audit includes it for operator lifecycle management
- [Phase 13-01]: subjects[0].namespace must be wekaappstore in ClusterRoleBinding — mismatch causes silent 403 failures
- [Phase 13-01]: SKILL.md ConfigMap targets OpenClaw container (not MCP sidecar) — OpenClaw reads SKILL.md at agent registration via skill field in openclaw.json
- [Phase 13-kubernetes-manifests-and-sidecar-wiring]: printf used in init container (not heredoc) to avoid shell variable expansion when writing openclaw.json
- [Phase 13-kubernetes-manifests-and-sidecar-wiring]: subPath: blueprints on MCP sidecar mount dereferences git-sync symlink at /blueprints/blueprints
- [Phase 13-kubernetes-manifests-and-sidecar-wiring]: openclaw-config readOnly in weka-mcp-sidecar; init container is sole writer to prevent race conditions

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 12 gate]: NemoClaw EKS topology must be validated before Phase 13 manifests are written — wrong topology = full Phase 13 rewrite
- [Phase 13 gate]: NemoClaw container image name on NVIDIA NGC is a placeholder; confirm before writing Deployment manifest
- [Phase 13 gate]: NemoClaw ConfigMap schema for mcpServers.url must be verified against the deployed version
- [Phase 13 gate]: Blueprint catalog size must be measured; if >1MB, ConfigMap strategy replaced with PVC or git-sync

## Session Continuity

Last session: 2026-03-24T05:58:14.822Z
Stopped at: Completed 13-02-PLAN.md
Resume file: None
