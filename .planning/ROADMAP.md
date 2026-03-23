# Roadmap: OpenClaw MCP Tools For WEKA App Store

## Milestones

- ✅ **v2.0 OpenClaw MCP Tool Integration** - Phases 6-10 (shipped 2026-03-22)
- 🚧 **v3.0 Live EKS Deployment and Agent Testing** - Phases 11-14 (in progress)

## Phases

<details>
<summary>✅ v2.0 OpenClaw MCP Tool Integration (Phases 6-10) - SHIPPED 2026-03-22</summary>

8-tool MCP server shipped with 103 tests, SKILL.md, mock agent harness, Dockerfile, CI/CD, and deprecated v1.0 code removed.

See MILESTONES.md for full v2.0 summary.

</details>

### 🚧 v3.0 Live EKS Deployment and Agent Testing (In Progress)

**Milestone Goal:** Deploy OpenClaw/NemoClaw and the MCP server to EKS, register tools via Streamable HTTP sidecar, and validate the full agent chat experience with a happy-path blueprint deployment.

- [ ] **Phase 11: Streamable HTTP Transport** - Add HTTP transport mode to MCP server (code-only, no cluster needed)
- [ ] **Phase 12: NemoClaw EKS Topology** - Deploy NemoClaw/OpenClaw to EKS using agent-sandbox CRD; validate topology before manifests
- [ ] **Phase 13: Kubernetes Manifests and Sidecar Wiring** - Author complete K8s manifest set; wire MCP sidecar into OpenClaw pod
- [ ] **Phase 14: End-to-End Validation** - Validate full happy-path blueprint deployment through live agent chat

## Phase Details

### Phase 11: Streamable HTTP Transport
**Goal**: MCP server runs in dual-mode: stdio (default) and Streamable HTTP, selected by env var, fully validated locally before any cluster work begins
**Depends on**: Nothing (Phase 10 complete)
**Requirements**: XPORT-01, XPORT-02, XPORT-03, XPORT-04
**Success Criteria** (what must be TRUE):
  1. `curl localhost:8080/health` returns HTTP 200 when server starts with `MCP_TRANSPORT=http`
  2. `MCP_TRANSPORT=stdio` (default) starts the server exactly as before; all 103 existing tests pass unchanged
  3. `MCP_TRANSPORT=http` starts the server in Streamable HTTP mode on the port set by `MCP_PORT`
  4. Tool calls over HTTP return the same flat JSON responses as stdio (depth contract preserved)
  5. `openclaw.json` points to `http://localhost:8080/mcp` with `"transport": "streamable-http"` replacing the stdio startup block
**Plans**: TBD

Plans:
- [ ] 11-01: TBD

### Phase 12: NemoClaw EKS Topology
**Goal**: NemoClaw/OpenClaw is running and reachable on EKS using the experimental agent-sandbox CRD approach; topology confirmed and documented before any manifests are written
**Depends on**: Phase 11
**Requirements**: NCLAW-01, NCLAW-03
**Success Criteria** (what must be TRUE):
  1. NemoClaw/OpenClaw pod is Running in EKS cluster (`kubectl get pods` shows Ready)
  2. NemoClaw egress policy explicitly allows loopback access so sidecar port is reachable
  3. GPU node group and NVIDIA GPU Operator confirmed operational (agent container starts without GPU errors)
  4. Topology decision (agent-sandbox CRD approach) documented as a Key Decision in PROJECT.md
**Plans**: TBD

Plans:
- [ ] 12-01: TBD

### Phase 13: Kubernetes Manifests and Sidecar Wiring
**Goal**: Complete Kubernetes manifest set authored and applied; MCP sidecar running inside the OpenClaw pod with correct RBAC, startup ordering, and runtime-generated openclaw.json
**Depends on**: Phase 12
**Requirements**: K8S-01, K8S-02, K8S-03, K8S-04, K8S-05, NCLAW-02, NCLAW-04
**Success Criteria** (what must be TRUE):
  1. MCP sidecar container starts after NemoClaw pod readiness; `kubectl logs` shows no startup race errors
  2. `kubectl logs <mcp-sidecar>` shows `/health` returning 200 before OpenClaw attempts tool registration
  3. Blueprint YAML files are accessible inside the sidecar at `BLUEPRINTS_DIR` via volume mount
  4. `openclaw.json` is generated at pod startup from env vars (not baked into the image); correct URL and transport visible in pod logs
  5. `weka-mcp-server-sa` ServiceAccount exists with scoped ClusterRole (not reusing operator's service account)
**Plans**: TBD

Plans:
- [ ] 13-01: TBD

### Phase 14: End-to-End Validation
**Goal**: Agent completes the full SKILL.md tool chain against a live EKS cluster and live WEKA storage; happy-path blueprint deployment succeeds through chat
**Depends on**: Phase 13
**Requirements**: E2E-01, E2E-02, E2E-03, E2E-04
**Success Criteria** (what must be TRUE):
  1. Agent returns real cluster resource data (GPU, CPU, RAM, namespaces) when asked about cluster state through chat
  2. Agent lists and describes blueprints from the live catalog through chat
  3. Agent generates, validates, and applies a WekaAppStore CR through the full SKILL.md workflow (inspect → validate → apply); CR appears in `kubectl get wekaappstores`
  4. Agent reports deployment status after apply (operator reconciliation outcome visible in chat)
**Plans**: TBD

Plans:
- [ ] 14-01: TBD

## Progress

**Execution Order:** 11 → 12 → 13 → 14

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 6. MCP Scaffold and Read-Only Tools | v2.0 | 3/3 | Complete | 2026-03-22 |
| 7. Validation, Apply, and Status Tools | v2.0 | 2/2 | Complete | 2026-03-22 |
| 8. SKILL.md, Agent Context, and Cleanup | v2.0 | 3/3 | Complete | 2026-03-22 |
| 9. Deployment and Registration | v2.0 | 2/2 | Complete | 2026-03-22 |
| 10. Integration Bug Fixes | v2.0 | 1/1 | Complete | 2026-03-22 |
| 11. Streamable HTTP Transport | v3.0 | 0/TBD | Not started | - |
| 12. NemoClaw EKS Topology | v3.0 | 0/TBD | Not started | - |
| 13. Kubernetes Manifests and Sidecar Wiring | v3.0 | 0/TBD | Not started | - |
| 14. End-to-End Validation | v3.0 | 0/TBD | Not started | - |
