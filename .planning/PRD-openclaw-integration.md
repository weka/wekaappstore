# PRD: OpenClaw Tool Integration For WEKA App Store

**Project:** WEKA App Store
**Date:** 2026-03-20
**Status:** Draft — replaces PRD-nemoclaw-integration.md after architectural pivot
**Primary Goal:** Register the WEKA App Store's cluster inspection, blueprint catalog, validation, and apply capabilities as MCP tools that an OpenClaw/NemoClaw agent can call, so the agent handles all planning, resource reasoning, and user conversation natively.

## Architectural Pivot

The original PRD assumed the backend would own the agent's reasoning: session state, fit assessment, family matching, follow-up question logic, and plan compilation. That is wrong.

OpenClaw is a full agentic framework with:
- Its own tool-use loop and reasoning
- Built-in conversation/session management and memory
- WebSocket Gateway protocol for clients
- MCP server support for tool registration
- Built-in approval gating for dangerous operations
- NemoClaw adds NVIDIA inference (NIM/Nemotron), sandboxed execution, and enterprise guardrails

**The correct architecture:** We provide tools. OpenClaw provides the brain.

## Problem Statement

The current app store is form-centric. Users pick blueprints and fill in parameters. There is no way to:
- Describe goals in natural language
- Get agent-driven resource assessment before install
- Have the agent reason about whether GPUs, CPU, RAM, and WEKA storage can support a request
- Get the agent to synthesize valid `WekaAppStore` YAML from intent

OpenClaw/NemoClaw can solve all of this — IF we give it the right tools to inspect, validate, and apply.

## Product Outcome

Users interact with OpenClaw through its native chat channels (web UI, Slack, etc.). The agent:
1. Understands what the user wants to install
2. Calls our MCP tools to inspect the cluster and WEKA environment
3. Reasons about whether resources are sufficient
4. Generates `WekaAppStore` YAML using the CRD schema we provide as context
5. Validates the YAML via our validation tool
6. Presents the plan and requests approval
7. Applies via our apply tool (gated by OpenClaw's approval system)
8. Reports status by calling our status inspection tool

## What We Build vs. What OpenClaw Handles

### We Build (MCP Tools)
- `weka_appstore_inspect_cluster` — returns GPU inventory, CPU/RAM availability, namespaces, storage classes
- `weka_appstore_inspect_weka` — returns WEKA capacity, filesystems, existing mounts
- `weka_appstore_list_blueprints` — returns available blueprint catalog with schemas and default values
- `weka_appstore_get_blueprint` — returns full blueprint detail including Helm values schema
- `weka_appstore_validate_yaml` — validates generated YAML against CRD and operator contract
- `weka_appstore_apply` — creates the `WekaAppStore` CR (requires approval)
- `weka_appstore_status` — returns current status of deployed `WekaAppStore` resources
- `weka_appstore_get_crd_schema` — returns the `WekaAppStore` CRD spec so the agent can generate valid YAML

### OpenClaw Handles (We Do NOT Build)
- Conversation state and session management
- Natural language understanding and intent extraction
- Resource reasoning and fit decisions
- Follow-up question logic
- Blueprint family selection
- YAML generation (guided by CRD schema context)
- Plan presentation and user approval flow
- Memory across sessions

## MCP Tool Specifications

### weka_appstore_inspect_cluster
Returns a bounded snapshot of Kubernetes cluster state relevant to blueprint deployment.
- GPU inventory: count, model, memory per node
- CPU: allocatable vs requested per node
- RAM: allocatable vs requested per node
- Namespaces: list with labels
- Storage classes: list with provisioner and parameters
- Read-only. No cluster mutation.

### weka_appstore_inspect_weka
Returns WEKA storage state from WekaCluster CRs and/or WEKA API.
- Cluster name, status, endpoints
- Filesystem list with capacity and used space
- Existing CSI storage classes
- Read-only. No storage mutation.

### weka_appstore_list_blueprints
Returns the blueprint catalog.
- Blueprint name, description, category
- Required and optional parameters
- Resource requirements (GPU, CPU, RAM, storage minimums)
- Supported configurations

### weka_appstore_get_blueprint
Returns full detail for a specific blueprint.
- Helm chart reference or manifest template
- Full values schema
- Default values
- Dependencies
- Known prerequisites

### weka_appstore_validate_yaml
Validates a `WekaAppStore` YAML document.
- CRD schema conformance
- Required fields present
- Component naming uniqueness
- Helm chart field completeness
- Namespace resolution
- Dependency reference validity
- Returns pass/fail with actionable errors

### weka_appstore_apply
Creates or updates a `WekaAppStore` CR in the cluster.
- Accepts validated YAML
- MUST be gated by OpenClaw approval system
- Submits through the existing operator reconciliation path
- Returns the created resource name and namespace

### weka_appstore_status
Returns status of deployed `WekaAppStore` resources.
- appStackPhase
- Per-component status
- Error messages if any
- Read-only.

### weka_appstore_get_crd_schema
Returns the `WekaAppStore` CRD specification.
- Full CRD spec including appStack.components schema
- Supported component types (helmChart, kubernetesManifest)
- All supported fields: values, dependsOn, targetNamespace, waitForReady, readinessCheck
- This is context for the agent to generate valid YAML

## Implementation Approach

### MCP Server
Build a single MCP server (Python, using the MCP SDK) that exposes all tools above. The server:
- Runs as a sidecar or standalone process alongside the WEKA App Store backend
- Connects to the Kubernetes API (read-only for inspection, write for apply)
- Connects to WEKA API for storage inspection
- Reuses existing code from `inspection/cluster.py` and `planning/apply_gateway.py`
- Is registered in OpenClaw's `openclaw.json` configuration

### Development Without Live OpenClaw
Since NemoClaw/OpenClaw is not yet running in the environment:
1. Build the MCP server with full tool implementations
2. Write a mock OpenClaw client that simulates the tool-use loop for testing
3. Test each tool independently against mocked Kubernetes/WEKA responses
4. Build a simple CLI harness that calls tools in sequence to prove the flow
5. When OpenClaw becomes available, register the MCP server — no code changes needed

### Skill Definition
Create a `SKILL.md` that tells OpenClaw how to use the tools together:
- When to inspect before planning
- How to reason about resource fit
- When to ask the user for clarification
- How to generate valid `WekaAppStore` YAML
- When to validate before applying
- That apply ALWAYS requires user approval

## Functional Requirements

### MCP Tools
- [ ] **OC-01**: MCP server exposes `weka_appstore_inspect_cluster` with GPU, CPU, RAM, namespace, and storage class data
- [ ] **OC-02**: MCP server exposes `weka_appstore_inspect_weka` with WEKA capacity and filesystem data
- [ ] **OC-03**: MCP server exposes `weka_appstore_list_blueprints` with catalog metadata
- [ ] **OC-04**: MCP server exposes `weka_appstore_get_blueprint` with full blueprint detail
- [ ] **OC-05**: MCP server exposes `weka_appstore_validate_yaml` with CRD and operator contract checks
- [ ] **OC-06**: MCP server exposes `weka_appstore_apply` with approval requirement
- [ ] **OC-07**: MCP server exposes `weka_appstore_status` for deployment status
- [ ] **OC-08**: MCP server exposes `weka_appstore_get_crd_schema` for YAML generation context

### Skill & Context
- [ ] **OC-09**: SKILL.md defines the agent's workflow for blueprint planning and installation
- [ ] **OC-10**: CRD schema and operator contract are provided as agent context
- [ ] **OC-11**: Blueprint catalog metadata is structured for agent consumption

### Safety
- [ ] **OC-12**: All inspection tools are read-only
- [ ] **OC-13**: Apply tool requires OpenClaw approval gate
- [ ] **OC-14**: No tool exposes unrestricted kubectl or helm execution
- [ ] **OC-15**: Tool responses include freshness timestamps

### Testability
- [ ] **OC-16**: Each tool is independently testable with mocked Kubernetes/WEKA backends
- [ ] **OC-17**: A mock agent harness can exercise the full tool chain without live OpenClaw
- [ ] **OC-18**: Integration tests prove the apply tool creates valid `WekaAppStore` CRs

## Implementation Phases

### Phase 1: MCP Server Scaffold And Read-Only Tools
- Set up MCP server project structure
- Implement `inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`
- Reuse existing `inspection/cluster.py` code
- All tools testable with mocked K8s/WEKA

### Phase 2: Validation And Apply Tools
- Implement `validate_yaml` using existing validator logic
- Implement `apply` tool with approval gate semantics
- Implement `status` tool
- Full tool chain exercisable via mock harness

### Phase 3: Skill Definition And Agent Context
- Write SKILL.md with workflow instructions
- Package CRD schema and operator contract as context
- Structure blueprint catalog for agent consumption
- Test with mock agent harness end-to-end

### Phase 4: OpenClaw Integration And Live Testing
- Register MCP server in OpenClaw configuration
- Test with live NemoClaw/OpenClaw instance
- Tune skill instructions based on real agent behavior
- Validate approval gating works end-to-end

## What Happens To Existing Code

### Keep and reuse as tool implementations
- `inspection/cluster.py` → behind `inspect_cluster` tool
- `planning/apply_gateway.py` → behind `apply` tool
- `planning/validator.py` → behind `validate_yaml` tool
- `planning/models.py` → partially, for tool input/output schemas

### Deprecate (OpenClaw handles this)
- `planning/session_service.py` — OpenClaw manages conversation state
- `planning/session_store.py` — OpenClaw has built-in memory
- `planning/family_matcher.py` — OpenClaw reasons about this from catalog context
- `planning/compiler.py` — OpenClaw generates YAML from CRD schema context
- Planning session routes in `main.py` — replaced by OpenClaw Gateway
- `planning_session.html` — replaced by OpenClaw chat interface

## Risks

### OpenClaw YAML Quality
The agent may generate invalid YAML even with CRD schema context.
Mitigation: `validate_yaml` tool catches errors before apply; the agent can iterate.

### Approval Bypass
The agent could attempt to apply without approval if poorly configured.
Mitigation: The `apply` tool itself enforces approval at the MCP level, independent of agent behavior.

### Inspection Staleness
Cluster state changes between inspection and apply.
Mitigation: Tools include timestamps; apply tool can re-inspect before execution.

### Development Without Live Agent
Building tools without testing against real OpenClaw may produce interface mismatches.
Mitigation: Mock agent harness simulates the tool-use loop; MCP protocol is standardized.

## Definition of Done

- MCP server with all 8 tools running and testable
- SKILL.md defining the agent workflow
- Mock harness proving the full inspect → reason → validate → apply flow
- Documentation for registering the MCP server with OpenClaw/NemoClaw
- Existing inspection and apply code reused, not duplicated

---
*Prepared for GSD workflow*
*Replaces: PRD-nemoclaw-integration.md*
