# OpenClaw MCP Tools For WEKA App Store

## What This Is

This project extends the WEKA App Store with an MCP tool server that lets OpenClaw/NemoClaw agents inspect cluster resources, browse blueprints, validate YAML, and apply installations through the existing WEKA operator. The agent handles all conversation, reasoning, and YAML generation — we provide the tools it calls.

The primary users are platform users and cluster admins who interact with OpenClaw to deploy blueprints, plus maintainers who use the agent to draft new blueprint definitions.

## Core Value

OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## Current Milestone: v3.0 Live EKS Deployment and Agent Testing

**Goal:** Deploy OpenClaw/NemoClaw and the MCP server to Chris's EKS cluster, register the MCP tools via Streamable HTTP sidecar, and validate the full agent chat experience with a happy-path blueprint deployment.

**Target features:**
- Streamable HTTP transport added to MCP server (alongside existing stdio)
- NemoClaw/OpenClaw deployed to EKS
- MCP server running as sidecar container alongside NemoClaw pod
- Agent can chat, inspect cluster, browse blueprints, validate YAML, and deploy a blueprint end-to-end
- Happy-path E2E validation against real K8s and WEKA resources

## Current State (after v2.0)

The MCP server is complete and container-ready. 8 tools are registered and tested (103 tests). SKILL.md defines the agent workflow. Deprecated v1.0 backend-brain code has been removed.

**What's shipped:**
- 8-tool MCP server (`mcp-server/`) with flat agent-friendly JSON responses
- SKILL.md with 12-step workflow, validate-retry loop, re-inspect-before-apply
- Mock agent harness with description-based tool selection (3 scenarios)
- Dockerfile, GitHub Actions CI/CD to `wekachrisjen/weka-app-store-mcp`
- OpenClaw registration config (`openclaw.json`) with drift detection
- README with full registration docs for OpenClaw and NemoClaw (placeholder)

## Requirements

### Validated

- ✓ Users can browse and deploy blueprint content through the existing WEKA App Store web UI and backend apply flow. — existing
- ✓ The system can create and reconcile `WekaAppStore` resources through the current operator-driven execution model. — existing
- ✓ The operator supports multi-component `appStack` deployments with Helm charts, raw manifests, dependencies, target namespaces, and readiness checks. — existing
- ✓ Blueprint content can be sourced externally and applied as rendered YAML strings or file-backed manifests. — existing
- ✓ MCP server exposes bounded read-only tools for Kubernetes cluster inspection (GPU, CPU, RAM, namespaces, storage classes) — v2.0
- ✓ MCP server exposes bounded read-only tools for WEKA storage inspection (capacity, filesystems, mounts) — v2.0
- ✓ MCP server exposes blueprint catalog and schema tools for agent consumption — v2.0
- ✓ MCP server exposes a YAML validation tool that checks against CRD and operator contract — v2.0
- ✓ MCP server exposes an approval-gated apply tool that creates `WekaAppStore` CRs through the existing operator path — v2.0
- ✓ SKILL.md defines the agent workflow for blueprint planning, validation, and installation — v2.0
- ✓ Mock agent harness can exercise the full tool chain without a live OpenClaw instance — v2.0
- ✓ Deprecated v1.0 backend-brain code (session service, family matcher, compiler, session routes) is removed — v2.0

### Active

- [ ] Streamable HTTP transport for MCP server (sidecar deployment pattern)
- [ ] NemoClaw/OpenClaw deployed and running in EKS
- [ ] MCP server registered with OpenClaw as sidecar via Streamable HTTP
- [ ] Agent can complete happy-path blueprint deployment through chat
- [ ] SKILL.md and openclaw.json updated for HTTP transport and real deployment

### Out of Scope

- Building a custom chat UI — OpenClaw provides the conversation interface natively
- Reimplementing agent reasoning in Python — OpenClaw handles intent, follow-ups, and plan generation
- Session/conversation state management — OpenClaw has built-in memory
- Autonomous unrestricted kubectl/helm execution — tools are bounded and read-only except for the gated apply
- Replacing the `WekaAppStore` CRD or operator contract — the MCP tools work with the existing runtime model

## Context

The codebase is a brownfield Kubernetes application bundle with a FastAPI/Jinja web UI, a Kopf-based operator, and a Helm chart. The v1.0 milestone built backend-owned planning logic under the assumption that NemoClaw was a thin model API. After discovering that OpenClaw is a full agentic framework, the architecture was pivoted to tool-registration-first in v2.0.

v2.0 shipped the MCP server at `mcp-server/` (4,628 LOC Python) with 8 tools, a mock agent harness, SKILL.md, and container deployment. The v1.0 backend-brain code was removed in Phase 8; reusable modules (`inspection/cluster.py`, `planning/apply_gateway.py`, `planning/validator.py`) are preserved as tool implementations.

OpenClaw connects via WebSocket Gateway, uses OpenAI-compatible model API, and registers tools through MCP servers, TypeScript plugins, or SKILL.md files. NemoClaw adds NVIDIA inference (NIM/Nemotron) and sandboxed execution on top.

## Constraints

- **Tech stack**: MCP server in Python (reuses existing inspection/apply code), runs as standalone process alongside the FastAPI backend
- **Compatibility**: No CRD-breaking changes — MCP tools produce standard `WekaAppStore` resources the operator already handles
- **Safety**: All inspection tools read-only; apply tool requires approval gate; no unrestricted cluster exec
- **Development**: Fully testable without a live OpenClaw/NemoClaw instance via mock harness
- **MCP protocol**: Tools conform to MCP server specification for OpenClaw compatibility
- **Container**: Image published to `wekachrisjen/weka-app-store-mcp` on Docker Hub

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on the existing WEKA App Store repo as a brownfield project | The current GUI, operator, and CRD already implement the safe apply path and runtime contract | ✓ Good |
| Pivot from backend-brain to OpenClaw-native tool registration | OpenClaw is a full agent framework — reimplementing its reasoning in Python duplicates its capabilities | ✓ Good — v2.0 shipped |
| Provide tools via MCP server, not custom API endpoints | OpenClaw has native MCP support; MCP is the standard tool integration path | ✓ Good — 8 tools registered |
| Remove v1.0 backend-brain code instead of keeping it | Clean break avoids confusion about which code path is authoritative | ✓ Good — 12 files removed |
| Develop with mock harness before live OpenClaw | NemoClaw not yet available in environment; mock proves tool chain works | ✓ Good — 3 scenarios pass |
| Let OpenClaw generate YAML from CRD schema context | The agent reasons about YAML structure natively; validate_yaml tool catches errors | ✓ Good — validate-retry loop in SKILL.md |
| Flat 2-key depth contract for all tool responses | Prevents agents from needing deep key traversal; enforced by check_depth() test | ✓ Good — 103 tests enforce |
| Description-based tool selection in harness | Proves tool descriptions are sufficient for agent routing without hardcoded names | ✓ Good — keyword matching works |
| Container image on wekachrisjen Docker Hub | Chris's corporate Docker Hub account for all WEKA images | ✓ Good — CI/CD wired |
| Deploy NemoClaw/OpenClaw via agent-sandbox CRD on EKS | Experimental CRD provides sandbox isolation, GPU scheduling, and volume management out of the box; avoids hand-rolling Deployment + RBAC + PV scaffolding. Gateway must use --bind=loopback (not lan) since non-loopback requires controlUi.allowedOrigins config; loopback is correct for sidecar deployment | Validated in Phase 12 — pod Running with NVIDIA A10G on EKS in wekaappstore namespace |

---
*Last updated: 2026-03-23 after v3.0 milestone start*
