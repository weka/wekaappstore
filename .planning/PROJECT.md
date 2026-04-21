# OpenClaw MCP Tools For WEKA App Store

## What This Is

This project extends the WEKA App Store with an MCP tool server that lets OpenClaw/NemoClaw agents inspect cluster resources, browse blueprints, validate YAML, and apply installations through the existing WEKA operator. The agent handles all conversation, reasoning, and YAML generation — we provide the tools it calls.

The primary users are platform users and cluster admins who interact with OpenClaw to deploy blueprints, plus maintainers who use the agent to draft new blueprint definitions.

## Core Value

OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## Current State (after v4.0)

No active milestone. v4.0 shipped 2026-04-21. Two natural next directions: pick up the v3.1 deferred E2E chat work (see `.planning/v3.0-KNOWN-ISSUES.md` for tool-wrapper bug, init-container config gap, and model reliability findings), or start a new product direction.

**What's shipped end-to-end:**
- 8-tool MCP server (`mcp-server/`, v2.0) with flat agent-friendly JSON responses, 103 tests, Dockerfile, CI/CD to `wekachrisjen/weka-app-store-mcp`
- SKILL.md defining the 12-step agent workflow (v2.0)
- OpenClaw/NemoClaw deployed to EKS GPU node via agent-sandbox CRD (v3.0) — infrastructure functional, E2E chat deferred to v3.1
- Kubernetes manifest set: dedicated RBAC, SKILL.md ConfigMap, MCP sidecar wiring, init-container-generated openclaw.json, git-sync blueprint catalog (v3.0)
- WEKA App Store home page with 3-card category filter row (AIDP, WARP, Partner) above the catalog grid, URL hash deep-link support, full keyboard accessibility, mobile responsive (v4.0)

## Recent Milestones

**v4.0 App Categories on Home Screen — Shipped 2026-04-21**
Single-file frontend feature: 3-card category filter (AIDP=1 app, WARP=4, Partner=0) above the App Catalog grid, URL hash deep-links, keyboard a11y, mobile responsive. Delivered in 1 phase / 3 plans / ~1h40min in `app-store-gui/webapp/templates/index.html`. All 14 requirements (CAT/FIL/VIS/URL/A11Y) verified. No new dependencies, no build step.

**v3.0 Live EKS Deployment — Shipped 2026-04-21 (rescoped from "...and Agent Testing")**
Infrastructure delivered: HTTP transport, EKS deployment via agent-sandbox CRD, K8s manifests, sidecar wiring, RBAC, SKILL.md ConfigMap, init container openclaw.json, git-sync blueprint catalog. Direct MCP tool invocation verified functional. E2E chat validation + 4 prerequisite fixes (inspect-tool config loader, init container config gap, model reliability, OpenClaw upgrade) deferred to v3.1 — see `.planning/v3.0-KNOWN-ISSUES.md`.

**v2.0 OpenClaw MCP Tool Integration — Shipped 2026-03-22**
8-tool MCP server, SKILL.md, mock agent harness, deprecated v1.0 backend-brain code removed.

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
- ✓ MCP server supports Streamable HTTP transport (port 8080, stateless, `/health` endpoint) — v3.0 (XPORT-01..04)
- ✓ OpenClaw/NemoClaw deployed to EKS GPU node via experimental agent-sandbox CRD — v3.0 (NCLAW-01, NCLAW-03)
- ✓ Kubernetes manifest set with dedicated RBAC, SKILL.md ConfigMap, MCP sidecar wiring, init-container-generated openclaw.json, git-sync blueprint catalog — v3.0 (K8S-01..05, NCLAW-02, NCLAW-04)
- ✓ Three top-level app categories (AIDP, WARP, Partner) as selectable filter cards on the home screen with URL hash deep-links, keyboard a11y, mobile responsive — v4.0 (CAT-01..03, FIL-01..03, VIS-01..02, URL-01..03, A11Y-01..03)

### Active

(No active milestone — ready to start next. Two tracked directions: v3.1 E2E chat validation work — see `.planning/v3.0-KNOWN-ISSUES.md` — or a new product milestone. Run `/gsd:new-milestone` to start.)

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
| Single React root (PRD Option A) for v4.0 Categories feature | Lifting `ThemeProvider` out of `Catalog` into a new `AppShell` keeps both category cards and catalog grid under one theme provider, shares `selectedCategory` state via plain `useState` (no context needed for 2 consumers), preserves single-file CDN-React constraint | ✓ Good — shipped v4.0 in 1h40min; zero new deps; all 5 critical pitfalls mitigated via grep-level verification |
| No count Chip on category cards in v4.0 | Deferred to v4.1 polish milestone to keep v4.0 scope tight and ship in one session | ✓ Good — cleanly deferred; worth reconsidering if catalog grows beyond 5 items |
| Acronym-first category labels ("AIDP" not "NeuralMesh AIDP") | Shorter horizontal footprint on the 3-card row; description beneath carries the full name for accessibility | — Pending — revisit after user feedback on whether "AIDP" alone is recognizable |

---
*Last updated: 2026-04-21 after v4.0 milestone shipped*
