# Requirements: OpenClaw MCP Tools For WEKA App Store

**Defined:** 2026-03-20
**Core Value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## v2.0 Requirements

Requirements for the OpenClaw MCP Tool Integration milestone. Each maps to roadmap phases.

### MCP Server

- [ ] **MCPS-01**: MCP server scaffold using official `mcp[cli]` SDK with FastMCP, stdio transport
- [ ] **MCPS-02**: `inspect_cluster` tool returns flat GPU, CPU, RAM, namespace, and storage class data
- [ ] **MCPS-03**: `inspect_weka` tool returns WEKA capacity, filesystems, and mount data
- [ ] **MCPS-04**: `list_blueprints` tool returns blueprint catalog with names, descriptions, and resource requirements
- [ ] **MCPS-05**: `get_blueprint` tool returns full blueprint detail including Helm values schema and defaults
- [ ] **MCPS-06**: `get_crd_schema` tool returns the `WekaAppStore` CRD spec for agent YAML generation
- [ ] **MCPS-07**: `validate_yaml` tool checks generated YAML against CRD and operator contract, returns structured errors
- [ ] **MCPS-08**: `apply` tool creates `WekaAppStore` CRs with hard approval gate enforced in code
- [ ] **MCPS-09**: `status` tool returns deployment status of `WekaAppStore` resources
- [ ] **MCPS-10**: All tool responses use flat agent-friendly JSON, not nested v1.0 planning models
- [ ] **MCPS-11**: All logging goes to stderr, never stdout (stdio transport requirement)

### Agent Integration

- [ ] **AGNT-01**: SKILL.md defines the agent workflow with validate-before-apply constraint and negative examples
- [ ] **AGNT-02**: Mock agent harness exercises full tool chain with scripted tool-use loops
- [ ] **AGNT-03**: OpenClaw registration config (`openclaw.json` / NemoClaw equivalent) generated for the MCP server

### Deployment

- [ ] **DEPLOY-01**: Dockerfile packages MCP server as a container image
- [ ] **DEPLOY-02**: Container includes all dependencies and runs MCP server on stdio
- [ ] **DEPLOY-03**: Configuration interface for NemoClaw sandbox (environment variables for K8s/WEKA endpoints, credentials)
- [ ] **DEPLOY-04**: Documentation for registering MCP server with OpenClaw/NemoClaw

### Code Cleanup

- [ ] **CLEAN-01**: Remove deprecated `planning/session_service.py`, `planning/session_store.py`, `planning/family_matcher.py`, `planning/compiler.py`
- [ ] **CLEAN-02**: Remove deprecated planning session routes and `planning_session.html` template from `main.py`
- [ ] **CLEAN-03**: Preserve `inspection/cluster.py`, `planning/apply_gateway.py`, `planning/validator.py` as tool implementations

## v3.0 Requirements

Deferred to future milestone after live OpenClaw is available.

- **LIVE-01**: End-to-end testing with live NemoClaw/OpenClaw instance
- **LIVE-02**: SKILL.md tuning based on real agent behavior
- **LIVE-03**: Multi-blueprint coexistence assessment tool
- **LIVE-04**: Maintainer-facing draft blueprint authoring through agent

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom chat UI | OpenClaw provides conversation interface natively |
| Backend planning logic (session management, family matching, YAML compilation) | OpenClaw handles reasoning, follow-ups, and YAML generation |
| CRD or operator changes | MCP tools work with the existing runtime model |
| Live OpenClaw/NemoClaw deployment | Development uses mock harness; live integration is v3.0 |
| HTTP/SSE MCP transport | Stdio is correct for single-operator OpenClaw; HTTP only needed for multi-client |
| Unrestricted kubectl/helm execution | Tools are bounded and read-only except approval-gated apply |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MCPS-01 | — | Pending |
| MCPS-02 | — | Pending |
| MCPS-03 | — | Pending |
| MCPS-04 | — | Pending |
| MCPS-05 | — | Pending |
| MCPS-06 | — | Pending |
| MCPS-07 | — | Pending |
| MCPS-08 | — | Pending |
| MCPS-09 | — | Pending |
| MCPS-10 | — | Pending |
| MCPS-11 | — | Pending |
| AGNT-01 | — | Pending |
| AGNT-02 | — | Pending |
| AGNT-03 | — | Pending |
| DEPLOY-01 | — | Pending |
| DEPLOY-02 | — | Pending |
| DEPLOY-03 | — | Pending |
| DEPLOY-04 | — | Pending |
| CLEAN-01 | — | Pending |
| CLEAN-02 | — | Pending |
| CLEAN-03 | — | Pending |

**Coverage:**
- v2.0 requirements: 21 total
- Mapped to phases: 0
- Unmapped: 21 ⚠️

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after initial definition*
