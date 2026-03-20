# OpenClaw MCP Tools For WEKA App Store

## What This Is

This project extends the WEKA App Store with an MCP tool server that lets OpenClaw/NemoClaw agents inspect cluster resources, browse blueprints, validate YAML, and apply installations through the existing WEKA operator. The agent handles all conversation, reasoning, and YAML generation — we provide the tools it calls.

The primary users are platform users and cluster admins who interact with OpenClaw to deploy blueprints, plus maintainers who use the agent to draft new blueprint definitions.

## Core Value

OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## Current Milestone: v2.0 OpenClaw MCP Tool Integration

**Goal:** Build an MCP server exposing WEKA App Store capabilities as tools OpenClaw can call, with a mock harness for development without a live agent.

**Target features:**
- MCP server with read-only cluster and WEKA inspection tools
- Blueprint catalog and CRD schema tools for agent context
- YAML validation tool against CRD/operator contract
- Approval-gated apply tool that creates `WekaAppStore` CRs
- SKILL.md defining the agent's blueprint planning workflow
- Mock agent harness for end-to-end testing without live OpenClaw
- Removal of deprecated backend-brain code from v1.0

## Requirements

### Validated

- ✓ Users can browse and deploy blueprint content through the existing WEKA App Store web UI and backend apply flow. — existing
- ✓ The system can create and reconcile `WekaAppStore` resources through the current operator-driven execution model. — existing
- ✓ The operator supports multi-component `appStack` deployments with Helm charts, raw manifests, dependencies, target namespaces, and readiness checks. — existing
- ✓ Blueprint content can be sourced externally and applied as rendered YAML strings or file-backed manifests. — existing

### Active

- [ ] MCP server exposes bounded read-only tools for Kubernetes cluster inspection (GPU, CPU, RAM, namespaces, storage classes)
- [ ] MCP server exposes bounded read-only tools for WEKA storage inspection (capacity, filesystems, mounts)
- [ ] MCP server exposes blueprint catalog and schema tools for agent consumption
- [ ] MCP server exposes a YAML validation tool that checks against CRD and operator contract
- [ ] MCP server exposes an approval-gated apply tool that creates `WekaAppStore` CRs through the existing operator path
- [ ] SKILL.md defines the agent workflow for blueprint planning, validation, and installation
- [ ] Mock agent harness can exercise the full tool chain without a live OpenClaw instance
- [ ] Deprecated v1.0 backend-brain code (session service, family matcher, compiler, session routes) is removed

### Out of Scope

- Building a custom chat UI — OpenClaw provides the conversation interface natively
- Reimplementing agent reasoning in Python — OpenClaw handles intent, follow-ups, and plan generation
- Session/conversation state management — OpenClaw has built-in memory
- Autonomous unrestricted kubectl/helm execution — tools are bounded and read-only except for the gated apply
- Replacing the `WekaAppStore` CRD or operator contract — the MCP tools work with the existing runtime model
- Live OpenClaw/NemoClaw deployment — development uses mock harness; live integration is a follow-on milestone

## Context

The codebase is a brownfield Kubernetes application bundle with a FastAPI/Jinja web UI, a Kopf-based operator, and a Helm chart. The v1.0 milestone built backend-owned planning logic (typed plan contracts, cluster/WEKA inspection, session management, family matching, YAML compilation) under the assumption that NemoClaw was a thin model API. After discovering that OpenClaw is a full agentic framework with native tool-use, conversation management, and MCP support, the architecture was pivoted to tool-registration-first.

Reusable from v1.0: `inspection/cluster.py` (K8s inspection logic), `planning/apply_gateway.py` (apply path), `planning/validator.py` (CRD validation rules). These become implementations behind MCP tools.

OpenClaw connects via WebSocket Gateway, uses OpenAI-compatible model API, and registers tools through MCP servers, TypeScript plugins, or SKILL.md files. NemoClaw adds NVIDIA inference (NIM/Nemotron) and sandboxed execution on top.

## Constraints

- **Tech stack**: MCP server in Python (reuses existing inspection/apply code), must work as a standalone process alongside the FastAPI backend
- **Compatibility**: No CRD-breaking changes — MCP tools produce standard `WekaAppStore` resources the operator already handles
- **Safety**: All inspection tools read-only; apply tool requires approval gate; no unrestricted cluster exec
- **Development**: Must be fully testable without a live OpenClaw/NemoClaw instance via mock harness
- **MCP protocol**: Tools must conform to MCP server specification for OpenClaw compatibility

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on the existing WEKA App Store repo as a brownfield project | The current GUI, operator, and CRD already implement the safe apply path and runtime contract | ✓ Good |
| Pivot from backend-brain to OpenClaw-native tool registration | OpenClaw is a full agent framework — reimplementing its reasoning in Python duplicates its capabilities | — Pending |
| Provide tools via MCP server, not custom API endpoints | OpenClaw has native MCP support; MCP is the standard tool integration path | — Pending |
| Remove v1.0 backend-brain code instead of keeping it | Clean break avoids confusion about which code path is authoritative | — Pending |
| Develop with mock harness before live OpenClaw | NemoClaw not yet available in environment; mock proves tool chain works | — Pending |
| Let OpenClaw generate YAML from CRD schema context | The agent reasons about YAML structure natively; validate_yaml tool catches errors | — Pending |

---
*Last updated: 2026-03-20 after v2.0 milestone start (OpenClaw pivot)*
