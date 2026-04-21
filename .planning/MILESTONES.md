# Milestones

## v3.0 Live EKS Deployment (Shipped: 2026-04-21, rescoped from "...and Agent Testing")

**Phases completed:** 3 full (11, 12, 13) + 1 partial (14-01 infra prep)
**Commits across v3.0 phases:** 40+
**Requirements delivered:** 13/17 — XPORT-01..04, K8S-01..05, NCLAW-01..04

**Key accomplishments:**
- Streamable HTTP transport for MCP server (port 8080, stateless, `/health` endpoint)
- OpenClaw/NemoClaw deployed to EKS GPU node (NVIDIA A10G) via experimental agent-sandbox CRD
- Complete Kubernetes manifest set: dedicated RBAC (`weka-mcp-server-sa` + scoped ClusterRole), SKILL.md ConfigMap, MCP sidecar wired into the Sandbox CR with init-container-generated openclaw.json, git-sync blueprint catalog
- Plan 14-01 infrastructure prep shipped: prereq validation script, evidence capture script, Gateway API Service/HTTPRoute manifests

**Rescoped out:** E2E-01..04 agent chat validation. A 2026-04-21 retry session proved infrastructure works (direct MCP tool invocation returns structured responses) but surfaced code bug in inspect tool wrappers, config gap in init container, and small-model reliability issues. See `.planning/v3.0-KNOWN-ISSUES.md` for full analysis. Moved to v3.1 along with four prerequisite fixes (FIX-01..04).

---

## v2.0 OpenClaw MCP Tool Integration (Shipped: 2026-03-22)

**Phases completed:** 5 phases (6-10), 11 plans
**Lines of code:** 4,628 Python (mcp-server/)
**Commits:** 63 across v2.0 phases
**Tests:** 103 passing

**Key accomplishments:**
- 8-tool MCP server with flat agent-friendly JSON responses (inspect_cluster, inspect_weka, list_blueprints, get_blueprint, get_crd_schema, validate_yaml, apply, status)
- SKILL.md: 12-step agent workflow with validate-retry loop, re-inspect-before-apply, and negative YAML examples
- Mock agent harness with description-based tool selection exercising full inspect→validate→apply chain
- Approval-gated apply tool with boolean identity check preventing unapproved CR creation
- Deprecated v1.0 backend-brain code removed (12 files, planning routes, pruned exports)
- Container-ready: Dockerfile, GitHub Actions CI/CD to wekachrisjen Docker Hub, README with OpenClaw/NemoClaw registration docs

---

