# Feature Research

**Domain:** MCP Tool Server for Agentic Blueprint Deployment (OpenClaw/NemoClaw integration)
**Researched:** 2026-03-20
**Confidence:** HIGH — PRD is precise, MCP spec is current, existing code is directly inspectable

## Context: What Is Already Built

This milestone adds to an existing system. The following already exist and must NOT be rebuilt:

- Blueprint catalog browsing UI (`app-store-gui/webapp/`)
- Cluster inspection (`inspection/cluster.py`, `inspection/weka.py`)
- YAML validation (`planning/validator.py`)
- Apply gateway (`planning/apply_gateway.py`)
- WekaAppStore CR operator (`operator_module/main.py`)
- `PlanningInspectionTools` class (`planning/inspection_tools.py`)

**What we are building:** An MCP server that wraps these capabilities as tools an OpenClaw agent can call.

---

## Feature Landscape

### Table Stakes (Users Expect These)

"Users" here means the OpenClaw agent and the developer registering the server. Missing any of these means the agent loop breaks.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `weka_appstore_inspect_cluster` tool | Agent must know GPU/CPU/RAM/namespaces before reasoning about fit. Without this it cannot assess viability. | LOW | Wraps existing `inspection/cluster.py`. New code is only the MCP decorator and response shaping. |
| `weka_appstore_inspect_weka` tool | Agent needs WEKA filesystem capacity to validate storage-related blueprints. | LOW | Wraps existing `inspection/weka.py`. Same pattern as above. |
| `weka_appstore_list_blueprints` tool | Agent must know what can be deployed before it can plan. Discovery is the entry point. | MEDIUM | Catalog metadata must be structured for machine consumption, not human UI. Needs a defined JSON schema for blueprint descriptors. |
| `weka_appstore_get_blueprint` tool | Agent needs full values schema and defaults to generate valid YAML. Cannot generate without this. | MEDIUM | Blueprint detail includes Helm values schema. Must include resource requirements so agent can reason without calling inspect again. |
| `weka_appstore_validate_yaml` tool | Agent will attempt YAML generation and may get it wrong. Validation closes the loop — agent sees errors and iterates. | MEDIUM | Wraps existing `planning/validator.py`. Must return structured errors, not a flat string, so agent can target specific fields. |
| `weka_appstore_apply` tool | Without apply, the whole pipeline ends at planning. The workflow has no endpoint. | MEDIUM | Wraps `planning/apply_gateway.py`. Must enforce approval gate regardless of how it is called. |
| `weka_appstore_status` tool | Agent needs to report back to user post-apply. Without status it cannot confirm success or surface errors. | LOW | Read-only K8s query. Wraps existing CR status fields from the operator. |
| `weka_appstore_get_crd_schema` tool | Agent must have the CRD spec to generate valid `WekaAppStore` YAML. Without it the agent is guessing field names. | LOW | Serves the static CRD spec. Can be embedded in the server or read from cluster at startup. |
| Tool response format: JSON with `content` array | MCP protocol requirement. Clients expect `{"content": [{"type": "text", "text": "..."}]}`. Non-conforming responses break all agents. | LOW | Python MCP SDK handles this if you use `@mcp.tool()` decorator and return dicts or strings. |
| Tool errors in result object, not protocol exceptions | MCP spec: tool errors must be returned as `{"isError": true, "content": [...]}`, not raised as Python exceptions. Exceptions produce protocol-level errors agents cannot recover from. | LOW | Explicitly required by MCP spec. Must be enforced by convention across all 8 tools. |
| `readOnlyHint` and `destructiveHint` annotations | Agents and approval systems use these to decide whether to auto-execute or gate for approval. Missing = agent may prompt for approval on every read, or skip approval on writes. | LOW | Set `readOnlyHint=True` on all inspect/list/get/status tools. Set `readOnlyHint=False, destructiveHint=False` on apply. Shipped in MCP spec 2025-03-26. |
| Freshness timestamps on all tool responses | PRD requirement OC-15. Cluster state changes; agent needs to know when the snapshot was taken to assess staleness. | LOW | `_utc_timestamp()` already exists in `inspection/cluster.py`. Apply it to every response envelope. |
| SKILL.md skill definition file | OpenClaw requires a SKILL.md to understand when to invoke the tools, the workflow order, and constraints like "always validate before apply." Without this, the agent may call tools in the wrong order. | MEDIUM | Not code — a structured Markdown file with YAML frontmatter. Requires careful authoring of the agent instructions section. See format details below. |
| Mock harness for development testing | NemoClaw/OpenClaw is not yet running in the environment. Without a harness, there is no way to verify the tool chain works end-to-end during development. | MEDIUM | Pytest fixtures + in-process MCP client. Uses FastMCP in-memory transport. No subprocess, no network. |

### Differentiators (Competitive Advantage)

These make the MCP server more reliable and the agent more capable than a bare wrapper would be.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Structured validation errors from `validate_yaml` | If errors are just a flat string, the agent may not know which YAML field to fix. Field-keyed errors let the agent do targeted correction without human prompting. | MEDIUM | `planning/validator.py` returns `ValidationResult` with `PlanValidationError` objects. The MCP tool must preserve this structure in response JSON, not flatten to a single message. |
| CRD schema embedded as inline context in SKILL.md | Providing the `WekaAppStore` CRD spec both as a callable tool (`get_crd_schema`) AND as inline reference context in SKILL.md gives the agent dual access — reduces how often agent needs to call `get_crd_schema` mid-conversation. | LOW | Static content. No extra logic needed. |
| Resource requirement hints in `list_blueprints` | If blueprint descriptors include minimum GPU/CPU/RAM/storage, the agent can prefilter options without calling `get_blueprint` for each one. Reduces tool call count and latency. | LOW-MEDIUM | Requires catalog metadata to be enriched with a `requirements` section if not already present. |
| Preflight re-inspect in apply (freshness guard) | Apply tool optionally re-runs cluster inspection before submitting the CR. Guards against stale state between plan and apply — a risk flagged explicitly in the PRD. | MEDIUM | Not in PRD spec but addresses the "Inspection Staleness" risk. Optional via a `skip_preflight` bool parameter. Adds latency but prevents silent failures on stale plans. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Session/conversation state in MCP server | Developers familiar with REST APIs want server-side context. Seems like it would improve multi-turn behavior. | OpenClaw owns conversation state. Building it in the MCP server creates dual state that diverges, causes amnesia bugs, and duplicates what OpenClaw already provides. This is the exact mistake the PRD's architectural pivot corrects. | Let OpenClaw handle all session state. Tools are stateless — each call gets full context in parameters. |
| YAML generation inside the MCP server | Seems natural since the server knows the CRD schema. Would reduce agent load. | YAML generation requires reasoning about user intent, resource fit, and blueprint constraints simultaneously. That is exactly what LLM reasoning is for. Server-side generation reintroduces the planning logic that was removed in the architectural pivot. | Expose `get_crd_schema` and structured blueprint details. Let the agent generate YAML. Validate with `validate_yaml`. |
| Natural language tool inputs ("install Jupyter with big storage") | Would make tools feel friendlier. Seems to improve UX. | NLP parsing in tool inputs creates an ambiguous contract. "Big" is undefined. Ambiguous intent pushes reasoning into the tool layer, where errors are silent and unrecoverable by the agent. | Tools accept precise typed parameters. The agent does NLP and passes structured inputs. That is the correct layer separation. |
| Streaming tool responses | Appealing for long-running operations like apply. Avoids timeout feel. | MCP streaming tool responses are not part of the stable spec for tool invocation as of 2025-03-26. Partial responses can break agent parsers. Apply via K8s CR submission is fast — async work happens in the operator. | Return immediately with the created CR name. Use `weka_appstore_status` for follow-up polling. |
| Unrestricted `kubectl exec` or `helm install` passthrough tool | Would give agent full cluster control. Requested as an escape hatch. | PRD requirement OC-14 explicitly forbids this. Unrestricted execution bypasses validation, approval gating, and audit trails. Agent could be manipulated into destructive operations. | All cluster mutations go through `apply` which submits only `WekaAppStore` CRs through the validated operator path. |
| Per-tool authentication tokens in request parameters | Seems secure — each caller provides its own token. | Tokens in tool parameters appear in MCP call logs and LLM context. This exposes credentials to the model. | Configure authentication at the MCP server transport level (OAuth or API key in headers), not in tool schemas. |
| Auto-apply without approval gate | Speeds up the workflow. Fewer round-trips. | Removes the safety boundary the PRD requires. PRD requirement OC-13 is explicit: apply requires approval. The agent could be instructed by malicious input to apply without the user's knowledge. | Apply tool enforces approval token at the MCP level, independent of agent behavior. |

---

## Feature Dependencies

```
weka_appstore_get_crd_schema
    └──informs──> SKILL.md (CRD spec embedded as inline context)

weka_appstore_inspect_cluster  ──parallel──> weka_appstore_inspect_weka
    └──feeds──> agent resource reasoning
                    └──enables──> agent YAML generation
                                      └──requires──> weka_appstore_get_blueprint (values schema)
                                      └──requires──> weka_appstore_get_crd_schema

weka_appstore_list_blueprints
    └──narrows choices for──> weka_appstore_get_blueprint

agent YAML generation
    └──requires──> weka_appstore_validate_yaml (before apply)
                       └──gates──> weka_appstore_apply

weka_appstore_apply
    └──requires──> OpenClaw approval gate (external to our code)
    └──produces──> CR name + namespace
                       └──enables──> weka_appstore_status

SKILL.md
    └──orchestrates order of all tools above
    └──requires──> weka_appstore_get_crd_schema content as inline context

Mock harness
    └──requires all 8 tools to be callable
    └──simulates──> SKILL.md workflow order
```

### Dependency Notes

- **`validate_yaml` must precede `apply`:** The apply tool should require a validated YAML document. SKILL.md must instruct the agent that calling apply without prior validation is not permitted.
- **`get_crd_schema` feeds SKILL.md authoring:** CRD schema content must be finalized before SKILL.md can be written, because SKILL.md embeds it as inline reference.
- **`inspect_cluster` and `inspect_weka` are independent:** They can be called in parallel by the agent. Neither depends on the other.
- **Mock harness depends on all 8 tools being callable:** It cannot simulate the workflow until the full tool surface is exposed. This makes the harness a Phase 3 deliverable (after all tools exist), not Phase 1.
- **Blueprint catalog schema must be shared:** `list_blueprints` and `get_blueprint` must return data in the same schema. Define this schema once in a shared models file to avoid response drift.

---

## MVP Definition

### Launch With (v1 — this milestone)

- [ ] MCP server scaffold using the official Python `mcp` SDK or FastMCP
- [ ] `weka_appstore_inspect_cluster` — wraps existing `inspection/cluster.py`
- [ ] `weka_appstore_inspect_weka` — wraps existing `inspection/weka.py`
- [ ] `weka_appstore_list_blueprints` — catalog with resource requirement hints
- [ ] `weka_appstore_get_blueprint` — full values schema and defaults
- [ ] `weka_appstore_get_crd_schema` — static CRD spec delivery
- [ ] `weka_appstore_validate_yaml` — wraps `planning/validator.py` with structured error output
- [ ] `weka_appstore_apply` — wraps `planning/apply_gateway.py` with approval gate enforcement
- [ ] `weka_appstore_status` — CR status query
- [ ] Tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) on all 8 tools
- [ ] Freshness timestamps on all tool responses
- [ ] SKILL.md with workflow instructions, approval requirements, and embedded CRD context
- [ ] Mock harness (pytest fixtures + in-process MCP client) exercising the full tool chain
- [ ] `openclaw.json` registration snippet documented for OpenClaw integration

### Add After Validation (v1.x)

- [ ] Preflight re-inspect in `apply` tool — add `skip_preflight: bool = False` parameter; only after confirming the latency budget with OpenClaw team
- [ ] Blueprint catalog hot-reload — refresh catalog without restarting the MCP server; only needed if blueprint updates are frequent
- [ ] Pagination in `list_blueprints` — only needed if catalog exceeds approximately 50 entries

### Future Consideration (v2+)

- [ ] Async status streaming — only if operator adds webhook-style notifications
- [ ] Multi-cluster support — `cluster_id` parameter on inspect tools; requires OpenClaw session to track context across clusters
- [ ] Audit log tool (`weka_appstore_audit`) — returns apply history; requires a persistent store not currently in the system

---

## Feature Prioritization Matrix

| Feature | Agent Value | Implementation Cost | Priority |
|---------|-------------|---------------------|----------|
| `inspect_cluster` tool | HIGH | LOW (wraps existing) | P1 |
| `inspect_weka` tool | HIGH | LOW (wraps existing) | P1 |
| `list_blueprints` tool | HIGH | MEDIUM (catalog schema) | P1 |
| `get_blueprint` tool | HIGH | MEDIUM (values schema) | P1 |
| `get_crd_schema` tool | HIGH | LOW (static delivery) | P1 |
| `validate_yaml` tool | HIGH | MEDIUM (structured errors) | P1 |
| `apply` tool | HIGH | MEDIUM (approval gate) | P1 |
| `status` tool | HIGH | LOW (simple CR query) | P1 |
| Tool annotations (hints) | MEDIUM | LOW (decorator metadata) | P1 |
| Freshness timestamps | MEDIUM | LOW (already in codebase) | P1 |
| SKILL.md | HIGH | MEDIUM (careful authoring) | P1 |
| Mock harness | HIGH | MEDIUM (pytest + in-process) | P1 |
| Structured validation errors | HIGH | MEDIUM (preserve existing models) | P1 |
| Resource hints in catalog | MEDIUM | LOW-MEDIUM | P2 |
| Preflight re-inspect in apply | MEDIUM | MEDIUM | P2 |
| Catalog hot-reload | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for this milestone to be usable by an OpenClaw agent
- P2: Should have once core is stable
- P3: Nice to have, future milestone

---

## MCP Tool Server Behavior Patterns (Verified)

These are confirmed behaviors from the current MCP spec and Python SDK, not training-data assumptions.

### Response Format

All tools return a `content` array. The Python MCP SDK serializes dict returns automatically:

```python
# Success — return dict, SDK wraps in content array
return {"result": "...", "timestamp": "2026-03-20T12:00:00Z"}

# Error — use isError flag so the agent can see and handle the error
return mcp.types.CallToolResult(
    isError=True,
    content=[mcp.types.TextContent(type="text", text="Error: namespace 'foo' not found")]
)
```

Tool errors go in the result object, NOT raised as Python exceptions. Exceptions produce MCP protocol-level errors that agents cannot inspect or recover from. [Source: modelcontextprotocol.info/docs/concepts/tools]

### Tool Annotations

```python
from mcp.types import ToolAnnotations

@mcp.tool(annotations=ToolAnnotations(
    readOnlyHint=True,
    idempotentHint=True,
    openWorldHint=True  # talks to external K8s API
))
async def weka_appstore_inspect_cluster() -> dict:
    ...

@mcp.tool(annotations=ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,   # additive, not destructive
    idempotentHint=False,    # second apply is NOT a no-op
    openWorldHint=True
))
async def weka_appstore_apply(yaml_document: str, approval_token: str) -> dict:
    ...
```

Annotations are hints for the agent and client UX layer. They do NOT enforce security. Actual approval gating must be enforced both at the OpenClaw configuration layer AND in the tool code itself. [Source: blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations]

### Approval Gate Pattern for `apply`

OpenClaw's approval system operates at the client layer. The tool code must also enforce approval independently, as a defense-in-depth measure:

```python
@mcp.tool()
async def weka_appstore_apply(yaml_document: str, approval_token: str) -> dict:
    if not approval_token:
        return mcp.types.CallToolResult(
            isError=True,
            content=[mcp.types.TextContent(type="text",
                text="Apply requires explicit user approval. Provide an approval_token.")]
        )
    ...
```

The SKILL.md must instruct the agent: "Always request user approval before calling apply. Never call apply without first showing the user the validated YAML and receiving explicit confirmation." [Source: PRD OC-12/OC-13]

### Mock Harness Pattern (Verified)

Use FastMCP in-process client to eliminate network dependencies. All dependencies injected via constructor:

```python
# conftest.py
import pytest
from unittest.mock import AsyncMock
from mcp import Client
from mcp_server import build_server

@pytest.fixture
async def mcp_client(mock_k8s, mock_weka):
    server = build_server(k8s_client=mock_k8s, weka_client=mock_weka)
    async with Client(server) as client:
        yield client

# test_workflow.py
async def test_full_tool_chain(mcp_client):
    cluster = await mcp_client.call_tool("weka_appstore_inspect_cluster", {})
    blueprints = await mcp_client.call_tool("weka_appstore_list_blueprints", {})
    blueprint = await mcp_client.call_tool("weka_appstore_get_blueprint",
                                           {"name": blueprints[0]["name"]})
    # agent generates YAML here (simulated in test)
    validation = await mcp_client.call_tool("weka_appstore_validate_yaml",
                                            {"yaml_document": SAMPLE_YAML})
    assert not validation["isError"]
    result = await mcp_client.call_tool("weka_appstore_apply",
                                        {"yaml_document": SAMPLE_YAML,
                                         "approval_token": "test-approved"})
    assert result["name"] is not None
```

`mock_k8s` and `mock_weka` are `AsyncMock` objects. Everything runs in-process — no subprocess, no network. Tests are deterministic. [Source: mcpcat.io/guides/writing-unit-tests-mcp-servers, gofastmcp.com/servers/testing]

### SKILL.md Format for OpenClaw (Verified from Official Docs and Examples)

```markdown
---
name: weka-appstore
description: Install and manage WEKA-optimized AI/ML applications on Kubernetes using natural language requests.
homepage: https://github.com/weka/wekaappstore
metadata: {"openclaw":{"emoji":"🐳","requires":{"bins":[],"env":["KUBECONFIG"]}}}
---

## For Agents

[Agent system prompt instructions: workflow order, tool call sequence,
validation requirements before apply, approval gate instructions,
CRD field reference, what to do on validation failures, disambiguation]

## For Humans

[Slash command reference and quick start]

## Resources

- WekaAppStore CRD spec (inline or linked file)
- Blueprint catalog reference
```

Required authoring in the "For Agents" section:
1. Inspect cluster and WEKA before reasoning about resource fit
2. Call `list_blueprints` to discover options, `get_blueprint` for full details
3. Use `get_crd_schema` (or the embedded CRD spec) to generate valid `WekaAppStore` YAML
4. Always call `validate_yaml` before `apply`; on errors, fix and retry
5. Always request explicit user approval before calling `apply`
6. After apply, poll `status` to confirm deployment progress

[Source: docs.openclaw.ai/tools/skills, github.com/openclaw/skills agents-manager SKILL.md]

---

## Dependencies on Existing Code

| New MCP Tool | Existing File to Wrap | Existing Class / Function |
|---|---|---|
| `inspect_cluster` | `app-store-gui/webapp/inspection/cluster.py` | cluster collection functions |
| `inspect_weka` | `app-store-gui/webapp/inspection/weka.py` | WEKA collection functions |
| `validate_yaml` | `app-store-gui/webapp/planning/validator.py` | `ValidationResult`, `PlanValidationError` |
| `apply` | `app-store-gui/webapp/planning/apply_gateway.py` | apply gateway functions |
| `status` | `operator_module/main.py` + K8s API | WekaAppStore CR status fields |
| `list_blueprints`, `get_blueprint` | Blueprint catalog source (TBD) | Catalog metadata models |
| `get_crd_schema` | CRD YAML in `weka-app-store-operator-chart/` | Static file delivery |

**Code to deprecate after this milestone:**
- `planning/session_service.py` — OpenClaw owns conversation state
- `planning/session_store.py` — OpenClaw has built-in memory
- `planning/family_matcher.py` — OpenClaw reasons about blueprint families
- `planning/compiler.py` — OpenClaw generates YAML from CRD context
- Planning session routes in `app-store-gui/webapp/main.py`
- `planning_session.html` frontend template

---

## Sources

- [MCP Tools Concept — Model Context Protocol Spec](https://modelcontextprotocol.info/docs/concepts/tools/) — tool response format, `isError` error handling pattern (HIGH confidence — official spec)
- [Tool Annotations as Risk Vocabulary — MCP Blog, 2026-03-16](https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/) — `readOnlyHint`, `destructiveHint`, `idempotentHint` behavior and limitations (HIGH confidence — official MCP blog)
- [Testing FastMCP Servers — gofastmcp.com](https://gofastmcp.com/servers/testing) — in-process client test pattern, pytest-asyncio setup (HIGH confidence)
- [Unit Testing MCP Servers — MCPcat](https://mcpcat.io/guides/writing-unit-tests-mcp-servers/) — mock dependency injection, in-memory transport pattern (MEDIUM confidence — third-party guide, consistent with official SDK)
- [OpenClaw Skills Documentation](https://docs.openclaw.ai/tools/skills) — SKILL.md frontmatter fields, metadata structure (HIGH confidence — official docs)
- [OpenClaw agents-manager SKILL.md — GitHub](https://github.com/openclaw/skills/blob/main/skills/agentandbot-design/agents-manager/SKILL.md) — real SKILL.md format example (HIGH confidence — official skill registry)
- [PRD-openclaw-integration.md](/.planning/PRD-openclaw-integration.md) — tool specifications OC-01 through OC-18, approval requirements, code reuse plan (authoritative for this project)
- Python `mcp` SDK — [PyPI](https://pypi.org/project/mcp/), [GitHub](https://github.com/modelcontextprotocol/python-sdk) — `@mcp.tool()` decorator, `ToolAnnotations` class (HIGH confidence — official SDK)

---

*Feature research for: MCP Tool Server — OpenClaw/NemoClaw Agent Integration*
*Researched: 2026-03-20*
