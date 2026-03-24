---
phase: 13-kubernetes-manifests-and-sidecar-wiring
plan: 01
subsystem: infra
tags: [kubernetes, rbac, serviceaccount, clusterrole, configmap, mcp, openclaw]

# Dependency graph
requires:
  - phase: 12-nemoclaw-eks-topology
    provides: Locked decisions for SA name (weka-mcp-server-sa), namespace (wekaappstore), and OpenClaw container mount path
provides:
  - k8s/agent-sandbox/mcp-rbac.yaml with ServiceAccount + ClusterRole + ClusterRoleBinding
  - k8s/agent-sandbox/mcp-skill-configmap.yaml with full SKILL.md content
affects: [13-02, 13-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ClusterRole rules annotated with inline comments mapping each rule to the MCP tool requiring it"
    - "subjects[0].namespace explicitly set in ClusterRoleBinding to prevent silent 403 failures"
    - "SKILL.md embedded verbatim in ConfigMap using YAML block scalar (|) for clean multiline content"

key-files:
  created:
    - k8s/agent-sandbox/mcp-rbac.yaml
    - k8s/agent-sandbox/mcp-skill-configmap.yaml
  modified: []

key-decisions:
  - "delete verb included for wekaappstores resource — research audit includes it for operator lifecycle management"
  - "subjects[0].namespace must be explicitly wekaappstore in ClusterRoleBinding to prevent silent 403 failures (common pitfall documented in research)"
  - "ConfigMap targets OpenClaw container (not MCP sidecar) — OpenClaw reads SKILL.md at agent registration time via skill field in openclaw.json"

patterns-established:
  - "RBAC: inline comments per-rule naming the tool(s) that require each permission"
  - "ClusterRoleBinding: subjects namespace is always explicitly set"

requirements-completed: [K8S-02, NCLAW-04]

# Metrics
duration: 2min
completed: 2026-03-24
---

# Phase 13 Plan 01: Kubernetes Manifests and Sidecar Wiring — RBAC and SKILL.md ConfigMap Summary

**RBAC manifest (SA + ClusterRole + ClusterRoleBinding) and SKILL.md ConfigMap for MCP sidecar and OpenClaw container, validated with kubectl dry-run**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-24T05:54:30Z
- **Completed:** 2026-03-24T05:56:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- ServiceAccount `weka-mcp-server-sa` with `automountServiceAccountToken: true` in wekaappstore namespace
- ClusterRole `weka-mcp-server-cr` covering minimum permissions for all 8 MCP tools with inline comments per rule
- ClusterRoleBinding `weka-mcp-server-crb` with explicit `subjects[0].namespace: wekaappstore` (prevents silent 403)
- ConfigMap `weka-mcp-skill-md` containing full 250-line SKILL.md content for OpenClaw agent tool registration

## Task Commits

Each task was committed atomically:

1. **Task 1: Create RBAC manifest with ServiceAccount, ClusterRole, and ClusterRoleBinding** - `704e13a` (feat)
2. **Task 2: Create SKILL.md ConfigMap manifest** - `e96715a` (feat)

## Files Created/Modified
- `k8s/agent-sandbox/mcp-rbac.yaml` - Three-document YAML: ServiceAccount + ClusterRole + ClusterRoleBinding for MCP sidecar
- `k8s/agent-sandbox/mcp-skill-configmap.yaml` - ConfigMap delivering full SKILL.md to OpenClaw container at registration

## Decisions Made
- Included `delete` verb for `wekaappstores` resource — research audit synthesized rules include it for operator lifecycle management
- `subjects[0].namespace` explicitly set to `wekaappstore` in ClusterRoleBinding — common pitfall from research: mismatch causes silent 403 failures without any error log
- ConfigMap is mounted into the OpenClaw container (not the MCP sidecar) because OpenClaw reads SKILL.md at registration time via the `skill` field in openclaw.json

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both YAML files pass `kubectl apply --dry-run=client` validation
- Plan 02 can now reference `serviceAccountName: weka-mcp-server-sa` in the Sandbox CR podTemplate
- Plan 02 can now mount `weka-mcp-skill-md` ConfigMap via `subPath: SKILL.md` at `/home/node/.openclaw/SKILL.md`
- No blockers for Plan 02 or Plan 03

---
*Phase: 13-kubernetes-manifests-and-sidecar-wiring*
*Completed: 2026-03-24*
