# Requirements: OpenClaw MCP Tools For WEKA App Store

**Defined:** 2026-03-23
**Core Value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## v3.0 Requirements

Requirements for the Live EKS Deployment and Agent Testing milestone. Each maps to roadmap phases.

### MCP Transport

- [x] **XPORT-01**: MCP server supports Streamable HTTP transport on a configurable port alongside existing stdio
- [x] **XPORT-02**: `MCP_TRANSPORT` env var selects transport mode (`stdio` default, `http` for sidecar deployment)
- [x] **XPORT-03**: Health endpoint (`/health`) returns 200 when server is ready for tool calls
- [x] **XPORT-04**: HTTP transport operates in stateless mode (no session ID dependency)

### Kubernetes Deployment

- [x] **K8S-01**: Sidecar container spec defines the MCP server container for inclusion in OpenClaw pod
- [x] **K8S-02**: ServiceAccount and RBAC ClusterRole grant minimum permissions for all 8 tools (read cluster/WEKA, create WekaAppStore CRs)
- [x] **K8S-03**: Readiness probe on health endpoint prevents tool discovery before server is ready
- [x] **K8S-04**: Blueprint manifests are available to the sidecar via shared volume mount
- [x] **K8S-05**: `openclaw.json` is generated at pod startup from environment variables (not baked into image)

### NemoClaw/OpenClaw Setup

- [x] **NCLAW-01**: NemoClaw/OpenClaw deployed to EKS using experimental agent-sandbox CRD approach
- [x] **NCLAW-02**: MCP server registered with OpenClaw via Streamable HTTP transport (`http://localhost:<port>/mcp`)
- [x] **NCLAW-03**: NemoClaw egress policy explicitly allows loopback access to MCP sidecar port
- [x] **NCLAW-04**: SKILL.md loaded by agent at registration time

### E2E Validation

- [ ] **E2E-01**: Agent can inspect cluster resources and WEKA storage through chat
- [ ] **E2E-02**: Agent can list and describe blueprints through chat
- [ ] **E2E-03**: Agent can generate, validate, and apply a WekaAppStore CR through the full SKILL.md workflow
- [ ] **E2E-04**: Agent reports deployment status after apply

## Future Requirements

Deferred to v3.1+ after live agent behavior is observed.

- **LIVE-02**: SKILL.md tuning based on real agent behavior
- **LIVE-03**: Multi-blueprint coexistence assessment tool
- **LIVE-04**: Maintainer-facing draft blueprint authoring through agent
- **PROD-01**: Production hardening (TLS, auth, rate limiting for HTTP transport)
- **PROD-02**: Monitoring and alerting for MCP server health

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom chat UI | OpenClaw/NemoClaw provides conversation interface natively |
| SSE transport | Deprecated April 2026; Streamable HTTP is the correct transport |
| Multi-tenant MCP server | Single-operator deployment; multi-client is v4.0+ |
| NemoClaw production hardening | Early preview software; focus on proving the chain works |
| Dedicated EC2 VM deployment | Using agent-sandbox CRD approach instead |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| XPORT-01 | Phase 11 | Complete |
| XPORT-02 | Phase 11 | Complete |
| XPORT-03 | Phase 11 | Complete |
| XPORT-04 | Phase 11 | Complete |
| K8S-01 | Phase 13 | Complete |
| K8S-02 | Phase 13 | Complete |
| K8S-03 | Phase 13 | Complete |
| K8S-04 | Phase 13 | Complete |
| K8S-05 | Phase 13 | Complete |
| NCLAW-01 | Phase 12 | Complete |
| NCLAW-02 | Phase 13 | Complete |
| NCLAW-03 | Phase 12 | Complete |
| NCLAW-04 | Phase 13 | Complete |
| E2E-01 | Phase 14 | Pending |
| E2E-02 | Phase 14 | Pending |
| E2E-03 | Phase 14 | Pending |
| E2E-04 | Phase 14 | Pending |

**Coverage:**
- v3.0 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-23*
*Last updated: 2026-03-23 after v3.0 roadmap creation*
