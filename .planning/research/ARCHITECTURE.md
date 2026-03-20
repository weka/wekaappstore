# Architecture Research: MCP Tool Server for OpenClaw Integration

**Domain:** MCP tool server wrapping existing Python inspection/apply business logic
**Researched:** 2026-03-20
**Confidence:** HIGH (official MCP SDK docs, FastMCP docs, PRD analysis, codebase inspection)

---

## Architectural Pivot Summary

The previous ARCHITECTURE.md described the old approach: NemoClaw embedded inside the FastAPI
backend, which would own conversation state, session management, and plan compilation.

The new PRD (PRD-openclaw-integration.md) reverses this. OpenClaw is the brain. We are a
toolbox. This document describes the MCP tool server architecture only.

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          OpenClaw / NemoClaw                         │
│  (Conversation, reasoning, YAML generation, approval UI, memory)     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  MCP Protocol (JSON-RPC over stdio or HTTP)
                           │  tools/list → tools/call
┌──────────────────────────▼───────────────────────────────────────────┐
│                    WEKA App Store MCP Server                         │
│              mcp-server/server.py  (FastMCP instance)                │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │ inspect_cluster │  │  inspect_weka    │  │  list_blueprints   │  │
│  ├─────────────────┤  ├──────────────────┤  ├────────────────────┤  │
│  │  get_blueprint  │  │  get_crd_schema  │  │  validate_yaml     │  │
│  ├─────────────────┤  └──────────────────┘  └────────────────────┘  │
│  │     apply       │  (destructiveHint=True, REQUIRES APPROVAL)      │
│  ├─────────────────┤                                                  │
│  │     status      │                                                  │
│  └────────┬────────┘                                                  │
└───────────┼──────────────────────────────────────────────────────────┘
            │  Direct Python import (same virtualenv, same process)
┌───────────▼──────────────────────────────────────────────────────────┐
│                  Existing App Store Business Logic                   │
│                                                                      │
│  app-store-gui/webapp/inspection/cluster.py   (K8s inspection)       │
│  app-store-gui/webapp/inspection/weka.py      (WEKA inspection)      │
│  app-store-gui/webapp/planning/validator.py   (YAML validation)      │
│  app-store-gui/webapp/planning/apply_gateway.py  (CR submission)     │
│  app-store-gui/webapp/planning/models.py      (shared types)         │
└───────────┬──────────────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────────────────────────┐
│                        External Systems                              │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐   ┌────────────────┐  │
│  │  Kubernetes API  │    │    WEKA API       │   │  Blueprint     │  │
│  │  (read + write)  │    │  (read-only)      │   │  Catalog Files │  │
│  └──────────────────┘    └──────────────────┘   └────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| OpenClaw / NemoClaw | Conversation, intent, reasoning, YAML gen, approval gate | External agent system — we do not build this |
| MCP Server (`mcp-server/server.py`) | Expose tools via MCP protocol; thin adapter only; no business logic | Python, FastMCP (`mcp[cli]`) |
| Tool modules (`mcp-server/tools/`) | One file per tool group; import existing business logic; build tool response schema | Python, structured dataclasses |
| `inspection/cluster.py` | Kubernetes GPU/CPU/RAM/namespace/storageclass reads | Existing — reused unmodified |
| `inspection/weka.py` | WEKA capacity and filesystem reads | Existing — reused unmodified |
| `planning/validator.py` | CRD schema and operator contract validation | Existing — reused unmodified |
| `planning/apply_gateway.py` | Submit `WekaAppStore` CR to Kubernetes | Existing — reused unmodified |
| `planning/models.py` | Shared dataclasses for inspection/plan types | Existing — reused, may need minor extensions |
| Blueprint catalog | YAML/JSON files describing available apps | Existing helm chart values and weka-app-store-operator-chart structure |
| Mock harness (`mcp-server/tests/mock_agent.py`) | Simulate OpenClaw tool-use loop for testing without live agent | New — test tooling only |

---

## Recommended Project Structure

```
wekaappstore/
├── app-store-gui/
│   └── webapp/
│       ├── inspection/          # EXISTING — reused as-is
│       │   ├── cluster.py       #   K8s inspection functions
│       │   └── weka.py          #   WEKA inspection functions
│       ├── planning/            # EXISTING — reused as-is
│       │   ├── apply_gateway.py #   CR submission
│       │   ├── validator.py     #   Plan/YAML validation
│       │   └── models.py        #   Shared dataclasses
│       └── main.py              # EXISTING FastAPI app — not modified
│
├── mcp-server/                  # NEW — entire MCP server lives here
│   ├── pyproject.toml           #   Dependencies (mcp[cli], kubernetes, etc.)
│   ├── server.py                #   FastMCP instance creation + tool registration
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── inspect_cluster.py   #   weka_appstore_inspect_cluster tool
│   │   ├── inspect_weka.py      #   weka_appstore_inspect_weka tool
│   │   ├── blueprints.py        #   list_blueprints + get_blueprint + get_crd_schema tools
│   │   ├── validate.py          #   weka_appstore_validate_yaml tool
│   │   ├── apply.py             #   weka_appstore_apply tool (destructive)
│   │   └── status.py            #   weka_appstore_status tool
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── responses.py         #   Pydantic models for tool return types
│   ├── context/
│   │   ├── __init__.py
│   │   └── kube_client.py       #   Shared kubernetes client init (read vs write)
│   └── tests/
│       ├── conftest.py          #   Shared fixtures (mock K8s, mock WEKA)
│       ├── mock_agent.py        #   Simulates OpenClaw tool-use loop for CI
│       ├── test_inspect_cluster.py
│       ├── test_inspect_weka.py
│       ├── test_blueprints.py
│       ├── test_validate.py
│       ├── test_apply.py
│       └── test_status.py
│
└── SKILL.md                     # NEW — agent workflow instructions for OpenClaw
```

### Structure Rationale

- **`mcp-server/` at repo root:** The MCP server is a peer process to `app-store-gui/`, not
  a submodule of it. It has its own `pyproject.toml` and dependency set but shares the Python
  path to import from `app-store-gui/webapp/`.

- **`mcp-server/tools/` one file per tool group:** Keeps each tool's wrapper small and
  independently testable. The wrapper never contains business logic — it only calls existing
  code and transforms the result into the tool response schema.

- **`mcp-server/schemas/responses.py`:** Pydantic models for what each tool returns. These
  are the contracts OpenClaw reasons against. Stable schemas here mean the agent's prompts
  don't break when internal implementation changes.

- **`mcp-server/context/kube_client.py`:** Single shared Kubernetes client initialized once
  per server process. Read-only and read-write clients are separate objects to make the
  permission boundary explicit and auditable.

- **`mcp-server/tests/mock_agent.py`:** Critical for development without a live OpenClaw
  instance. Calls tools directly through the FastMCP test interface in sequence: inspect
  cluster → inspect weka → list blueprints → validate → apply.

---

## Architectural Patterns

### Pattern 1: Separate Process, Shared Python Path

**What:** The MCP server runs as an independent process (`python mcp-server/server.py`) but
imports from `app-store-gui/webapp/` using the same virtualenv and PYTHONPATH. The FastAPI
app and MCP server are not aware of each other at runtime.

**When to use:** Always. This is the correct architecture for this project.

**Trade-offs:**
- Pro: No coupling between FastAPI routes and MCP tools. Either can restart independently.
- Pro: OpenClaw spawns the MCP server as a child process via stdio; it does not know about
  FastAPI at all.
- Pro: Existing business logic is imported directly — no HTTP round-trip between MCP and
  FastAPI.
- Con: Both processes share the same K8s credentials and cluster state. This is acceptable
  here because both represent the same operator, not a multi-tenant setup.
- Con: If `inspection/cluster.py` or `planning/apply_gateway.py` have initialization side
  effects, they must be safe to call from two processes. (Review before Phase 1.)

**Example registration in `~/.openclaw/openclaw.json`:**
```json
{
  "mcpServers": {
    "weka-appstore": {
      "command": "python",
      "args": ["/path/to/wekaappstore/mcp-server/server.py"],
      "env": {
        "KUBECONFIG": "/path/to/kubeconfig"
      }
    }
  }
}
```

### Pattern 2: Thin Tool Wrapper (Never Reimplement Logic)

**What:** Each tool function in `mcp-server/tools/` is a thin wrapper. It validates the
incoming tool arguments, calls one or more existing functions from the business logic layer,
and maps the result to the tool response schema. No business logic lives in the wrapper.

**When to use:** Every tool, without exception.

**Trade-offs:**
- Pro: Business logic tested once in existing unit tests. Tool wrapper test only needs to
  verify the mapping and error handling.
- Pro: Future changes to inspection logic automatically propagate to the MCP tool.
- Con: If the existing function's return shape is messy, the wrapper must normalize it. This
  is a one-time cost.

**Example:**
```python
# mcp-server/tools/inspect_cluster.py
from webapp.inspection.cluster import collect_cluster_inspection
from mcp-server.schemas.responses import ClusterInspectionResult

@mcp.tool()
def weka_appstore_inspect_cluster() -> ClusterInspectionResult:
    """
    Returns GPU inventory, CPU/RAM availability, namespaces, and storage classes.
    Read-only. No cluster mutation.
    """
    raw = collect_cluster_inspection()
    return ClusterInspectionResult.from_raw(raw)
```

### Pattern 3: Destructive Tool Annotation for Apply

**What:** The `weka_appstore_apply` tool is annotated with MCP's `destructiveHint=True` (via
`annotations` in the tool definition). This signals to OpenClaw that the tool modifies state
and MUST NOT be called without user approval.

**When to use:** Only the `apply` tool. All other tools are read-only.

**Trade-offs:**
- Pro: OpenClaw's built-in approval gate fires automatically for annotated tools. We do not
  need to build UI approval logic.
- Pro: Even if a poorly-configured agent tried to skip approval, the annotation is visible in
  the tools/list response and compliant clients enforce it.
- Con: Depends on OpenClaw respecting the annotation. Defense-in-depth: the `apply` tool
  should also validate internally that the input YAML passed through `validate_yaml` first.

**MCP tool annotation pattern (FastMCP):**
```python
from mcp.types import Tool

@mcp.tool(annotations={"destructiveHint": True, "requiresApproval": True})
def weka_appstore_apply(wekaappstore_yaml: str, namespace: str) -> ApplyResult:
    """
    Creates or updates a WekaAppStore CR. REQUIRES USER APPROVAL before calling.
    Input must be pre-validated via weka_appstore_validate_yaml.
    """
    ...
```

### Pattern 4: Structured Output Schemas

**What:** All tool return types are Pydantic models defined in `mcp-server/schemas/responses.py`.
These become the JSON schema the agent reasons against via the MCP `outputSchema` field.

**When to use:** All tools. The agent needs stable, typed outputs to reason correctly.

**Trade-offs:**
- Pro: Stable contracts mean SKILL.md instructions remain valid even as internal
  implementations evolve.
- Pro: FastMCP automatically generates `outputSchema` from Pydantic return type hints.
- Con: Requires discipline to not change field names without reviewing agent prompts.

### Pattern 5: STDIO Transport for OpenClaw Integration

**What:** The MCP server uses stdio transport (default for FastMCP). OpenClaw spawns it as a
child process and communicates via stdin/stdout JSON-RPC.

**When to use:** OpenClaw integration. Also for the mock agent harness in CI.

**Trade-offs:**
- Pro: No network port needed. Simpler deployment, no TLS concerns.
- Pro: OpenClaw manages the server lifecycle — no separate daemon to manage.
- Con: One server process per OpenClaw session. Not multi-tenant. Acceptable here.
- Note: If the MCP server ever needs to serve multiple concurrent agents (e.g., team-wide
  deployment), switch to HTTP/SSE transport. That requires a port and network access but
  no code changes to tools themselves — only the `mcp.run()` call changes.

---

## Data Flow

### Read-Only Tool Call (inspect_cluster)

```
OpenClaw agent decides it needs cluster state
    ↓
tools/call { name: "weka_appstore_inspect_cluster", arguments: {} }
    ↓
MCP server receives call → routes to inspect_cluster.py wrapper
    ↓
wrapper calls collect_cluster_inspection() from webapp/inspection/cluster.py
    ↓
cluster.py issues Kubernetes API reads (nodes, pods, namespaces, storage classes)
    ↓
wrapper maps result to ClusterInspectionResult Pydantic model
    ↓
MCP server serializes to JSON, returns via tools/call response
    ↓
OpenClaw receives structured JSON, reasons about GPU/CPU/RAM availability
```

### Destructive Tool Call (apply) with Approval Gate

```
OpenClaw has validated YAML, presents plan to user
    ↓
User approves in OpenClaw chat UI
    ↓
OpenClaw sees apply tool has destructiveHint=True — shows confirmation prompt
    ↓
User confirms
    ↓
tools/call { name: "weka_appstore_apply", arguments: { wekaappstore_yaml: "...", namespace: "..." } }
    ↓
MCP server receives call → routes to apply.py wrapper
    ↓
apply.py internally re-validates YAML via validate_yaml logic (defense-in-depth)
    ↓
apply.py calls apply_gateway.py to submit WekaAppStore CR to Kubernetes
    ↓
Kubernetes stores CR → Kopf operator reconciles → Helm releases deployed
    ↓
MCP server returns ApplyResult { resource_name, namespace, timestamp }
    ↓
OpenClaw reports success to user, can follow up with weka_appstore_status calls
```

### Full Agent Workflow (Inspect → Reason → Validate → Apply)

```
User: "Install NIM LLM inference with WEKA storage on this cluster"
    ↓
[OpenClaw invokes tools in sequence]
    ↓
1. weka_appstore_inspect_cluster    → GPU/CPU/RAM/namespace/storageclass snapshot
2. weka_appstore_inspect_weka       → WEKA capacity + filesystem list
3. weka_appstore_list_blueprints    → Available blueprint catalog
4. weka_appstore_get_blueprint      → Full detail for chosen blueprint
5. weka_appstore_get_crd_schema     → WekaAppStore CRD spec for YAML generation
    ↓
[OpenClaw reasons: does cluster fit? which values to set? generate YAML]
    ↓
6. weka_appstore_validate_yaml      → Validation pass/fail + errors
    ↑ (agent iterates if errors returned)
    ↓
[OpenClaw presents plan + YAML to user, requests approval]
    ↓
7. weka_appstore_apply              → Creates WekaAppStore CR (approval-gated)
    ↓
[OpenClaw polls for completion]
    ↓
8. weka_appstore_status             → Reports appStackPhase and component status
```

---

## Integration Points

### MCP Server → Existing Business Logic

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `mcp-server/tools/` → `webapp/inspection/cluster.py` | Direct Python import | Server must have `app-store-gui/` on PYTHONPATH |
| `mcp-server/tools/` → `webapp/inspection/weka.py` | Direct Python import | Same PYTHONPATH requirement |
| `mcp-server/tools/` → `webapp/planning/validator.py` | Direct Python import | Validator is pure Python, no side effects at import |
| `mcp-server/tools/` → `webapp/planning/apply_gateway.py` | Direct Python import | Apply gateway initializes K8s client at call time, not import time — verify this |
| `mcp-server/tools/` → `webapp/planning/models.py` | Direct Python import | Models are dataclasses — safe to share |

### MCP Server → External Systems

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Kubernetes API | `kubernetes` Python client, initialized in `context/kube_client.py` | Read client and write client are separate objects |
| WEKA API | Called via `webapp/inspection/weka.py` existing implementation | MCP server does not add new WEKA API calls |
| Blueprint catalog | Read from filesystem — same chart directories existing FastAPI app reads | Must agree on catalog root path between FastAPI and MCP server |

### OpenClaw → MCP Server

| Boundary | Communication | Notes |
|----------|---------------|-------|
| OpenClaw registers server | Entry in `~/.openclaw/openclaw.json` under `mcpServers` | OpenClaw spawns as child process via stdio |
| OpenClaw discovers tools | `tools/list` JSON-RPC call on startup | FastMCP handles this automatically |
| OpenClaw calls tool | `tools/call` JSON-RPC call | Tool response returned synchronously |
| OpenClaw approval gate | Triggered by `destructiveHint: true` annotation on apply tool | OpenClaw-native — MCP server does not implement UI |

---

## New vs. Modified Components

### New (must be built)

| Component | Location | Description |
|-----------|----------|-------------|
| MCP server entry point | `mcp-server/server.py` | FastMCP instance, registers all tools |
| Tool wrappers | `mcp-server/tools/*.py` | Thin adapters calling existing business logic |
| Response schemas | `mcp-server/schemas/responses.py` | Pydantic models for stable tool output contracts |
| K8s client context | `mcp-server/context/kube_client.py` | Shared client init, read/write separation |
| Mock agent harness | `mcp-server/tests/mock_agent.py` | Simulates OpenClaw tool-use loop for CI |
| Tool unit tests | `mcp-server/tests/test_*.py` | Per-tool tests with mocked K8s and WEKA |
| `pyproject.toml` | `mcp-server/pyproject.toml` | MCP server dependencies |
| SKILL.md | `SKILL.md` (repo root) | Agent workflow instructions for OpenClaw |

### Existing (reused, not modified in Phase 1)

| Component | Location | Reuse Pattern |
|-----------|----------|---------------|
| Cluster inspection | `app-store-gui/webapp/inspection/cluster.py` | Called directly from `tools/inspect_cluster.py` |
| WEKA inspection | `app-store-gui/webapp/inspection/weka.py` | Called directly from `tools/inspect_weka.py` |
| Plan validator | `app-store-gui/webapp/planning/validator.py` | Called directly from `tools/validate.py` |
| Apply gateway | `app-store-gui/webapp/planning/apply_gateway.py` | Called directly from `tools/apply.py` |
| Shared models | `app-store-gui/webapp/planning/models.py` | Imported for type references |

### Existing (to be deprecated after MCP tools prove stable)

| Component | Location | Reason |
|-----------|----------|--------|
| `planning/session_service.py` | `app-store-gui/webapp/planning/` | OpenClaw owns conversation state |
| `planning/session_store.py` | `app-store-gui/webapp/planning/` | OpenClaw has built-in memory |
| `planning/family_matcher.py` | `app-store-gui/webapp/planning/` | OpenClaw reasons about this |
| `planning/compiler.py` | `app-store-gui/webapp/planning/` | OpenClaw generates YAML from CRD schema context |
| Planning session routes in `main.py` | `app-store-gui/webapp/main.py` | Replaced by OpenClaw Gateway |

---

## Build Order and Phase Dependencies

The following order respects code dependencies and testability gates.

### Phase 1: MCP Server Scaffold + Read-Only Tools

**Prerequisite:** None. These tools have no mutation risk.

**Build order within phase:**
1. `mcp-server/pyproject.toml` and `mcp-server/context/kube_client.py` — foundation
2. `mcp-server/schemas/responses.py` — define response contracts before writing wrappers
3. `mcp-server/tools/inspect_cluster.py` — wrap `inspection/cluster.py`
4. `mcp-server/tools/inspect_weka.py` — wrap `inspection/weka.py`
5. `mcp-server/tools/blueprints.py` — list_blueprints, get_blueprint, get_crd_schema
6. `mcp-server/server.py` — register all Phase 1 tools
7. Tests for all Phase 1 tools with mocked K8s/WEKA

**Gate to Phase 2:** All Phase 1 tools callable via `mcp dev server.py` inspector. Mocked
K8s responses produce correct structured output.

### Phase 2: Validation and Apply Tools

**Prerequisite:** Phase 1 complete. Tools 1-5 testable.

**Build order within phase:**
1. `mcp-server/tools/validate.py` — wrap `planning/validator.py`
2. `mcp-server/tools/status.py` — wrap K8s CR status reads
3. `mcp-server/tools/apply.py` — wrap `planning/apply_gateway.py`; add destructiveHint
4. `mcp-server/tests/mock_agent.py` — full tool-use loop test
5. Register Phase 2 tools in `server.py`

**Critical:** `apply.py` wrapper MUST internally re-validate YAML via `validate.py` logic
before calling the apply gateway. This is defense-in-depth independent of OpenClaw approval.

**Gate to Phase 3:** mock_agent.py can run full loop — inspect → validate → apply against
mocked backends — without errors.

### Phase 3: Skill Definition and Agent Context

**Prerequisite:** Phase 2 complete. All 8 tools working and tested.

**Build order within phase:**
1. `SKILL.md` — workflow instructions for OpenClaw agent
2. Structure blueprint catalog for agent consumption (verify `list_blueprints` output is
   LLM-friendly)
3. Package CRD schema as returned by `get_crd_schema` tool (verify completeness)
4. End-to-end test with mock_agent.py proving inspect → reason → validate → apply flow

### Phase 4: Live OpenClaw Integration

**Prerequisite:** Phase 3 complete. Server stable.

**Build order within phase:**
1. Register MCP server in `~/.openclaw/openclaw.json`
2. Test tools/list discovery
3. Test each tool via OpenClaw chat
4. Tune SKILL.md based on actual agent behavior
5. Validate approval gating fires for apply tool

---

## Scaling Considerations

This is a single-operator tool server, not a multi-tenant service. Scaling is not a
primary concern.

| Scale | Architecture Notes |
|-------|--------------------|
| Single operator, single cluster | STDIO transport. OpenClaw spawns server per session. |
| Team use, shared OpenClaw instance | Switch to HTTP transport. Server runs as persistent process. Blueprint catalog reads become concurrent — add file-level locking or cache. |
| Multi-cluster | Separate MCP server per cluster, or add `cluster_context` argument to each tool and switch kubeconfig dynamically. |

---

## Anti-Patterns

### Anti-Pattern 1: Embedding the MCP Server Inside the FastAPI Process

**What people do:** Mount the MCP server as a FastAPI sub-app using `app.mount()` or
`FastApiMCP(app)`.

**Why it's wrong for this project:** OpenClaw connects to MCP servers via stdio process
spawning, not HTTP. Embedding inside FastAPI requires HTTP transport and introduces the
FastAPI app as a dependency for the agent to work. The MCP server would not be runnable
standalone. It also couples the MCP server's lifecycle to the web UI's lifecycle.

**Do this instead:** Keep the MCP server as a standalone Python process in `mcp-server/`.
The FastAPI app and MCP server share business logic via Python imports, not via HTTP.

### Anti-Pattern 2: Reimplementing Business Logic in Tool Wrappers

**What people do:** Copy inspection code or validation logic into the MCP tool file because
it is easier than figuring out the right import path.

**Why it's wrong:** Duplicated logic diverges. When `cluster.py` is updated, the copy in
the MCP tool does not get the fix. This has already caused bugs in the existing codebase
(see PITFALLS.md: duplicated apply logic).

**Do this instead:** Import and call. Tools are wrappers, not re-implementations.

### Anti-Pattern 3: Skipping the Validation Tool Before Apply

**What people do:** Let the agent call `apply` directly after generating YAML, skipping
`validate_yaml` because the agent "should know the schema."

**Why it's wrong:** LLM YAML generation is unreliable. The agent will produce invalid YAML.
Without the validate step, the operator fails at reconcile time with an opaque error. The
agent cannot recover without structured validation feedback.

**Do this instead:** The SKILL.md must instruct the agent to always call `validate_yaml`
before `apply`. The `apply` wrapper should also defensively re-validate internally.

### Anti-Pattern 4: Writing to stdout in the STDIO MCP Server

**What people do:** Add `print()` statements or standard logging to stdout for debugging.

**Why it's wrong:** In stdio transport, stdout is the JSON-RPC channel. Any text written to
stdout corrupts the MCP protocol messages and breaks the server.

**Do this instead:** Use `logging` configured to write to stderr or a file. FastMCP routes
its own output correctly; tool code must not print to stdout.

### Anti-Pattern 5: Using the Apply Tool Without the Approval Annotation

**What people do:** Omit `destructiveHint: True` from the apply tool definition because it
feels redundant — "OpenClaw will ask for approval anyway."

**Why it's wrong:** The annotation is the machine-readable signal that triggers the approval
gate. Without it, OpenClaw may call apply inline without user confirmation, depending on its
configuration. The annotation must be present in the tool definition.

**Do this instead:** Apply tool must have both `destructiveHint: True` and clear docstring
language stating that user approval is required.

---

## Sources

- [MCP Python SDK — Official GitHub](https://github.com/modelcontextprotocol/python-sdk) — HIGH confidence
- [MCP Tools Specification (2025-06-18)](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — HIGH confidence
- [Build an MCP Server — Official Docs](https://modelcontextprotocol.io/docs/develop/build-server) — HIGH confidence
- [FastMCP — Running Server / Deployment](https://gofastmcp.com/deployment/running-server) — HIGH confidence
- [FastMCP — FastAPI Integration](https://gofastmcp.com/integrations/fastapi) — HIGH confidence
- [OpenClaw MCP Server Registration](https://safeclaw.io/blog/openclaw-mcp) — MEDIUM confidence (third-party guide, consistent with MCP spec)
- [OpenClaw `openclaw.json` configuration](https://openclawvps.io/blog/add-mcp-openclaw) — MEDIUM confidence (third-party guide)
- [MCP Transport Comparison — stdio vs SSE vs HTTP](https://dev.to/zrcic/understanding-mcp-server-transports-stdio-sse-and-http-streamable-5b1p) — MEDIUM confidence
- [DEV Community: MCP into existing FastAPI backend](https://dev.to/hiteshchawla/create-mcp-into-an-existing-fastapi-backend-3onp) — MEDIUM confidence
- [MCP Security — Tool Annotations and Approval](https://towardsdatascience.com/the-mcp-security-survival-guide-best-practices-pitfalls-and-real-world-lessons/) — MEDIUM confidence

---

*Architecture research for: WEKA App Store MCP tool server (OpenClaw integration milestone)*
*Researched: 2026-03-20*
*Replaces: previous ARCHITECTURE.md (NemoClaw-in-backend approach)*
