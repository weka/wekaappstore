# Project Research Summary

**Project:** WEKA App Store — OpenClaw MCP Tool Server Integration
**Domain:** Brownfield MCP server addition to an existing Python/Kubernetes operator backend
**Researched:** 2026-03-20
**Confidence:** HIGH

## Executive Summary

This project adds an MCP (Model Context Protocol) tool server to an existing WEKA App Store operator system, enabling an OpenClaw AI agent to inspect Kubernetes clusters, discover blueprints, generate and validate WekaAppStore CRDs, and deploy them through a gated apply workflow. The key architectural decision — already made in the PRD — is that OpenClaw owns all reasoning, YAML generation, conversation state, and approval UX. The MCP server is a stateless toolbox of 8 thin wrappers over already-built business logic. No new business logic belongs in the MCP layer.

The recommended implementation uses the official Anthropic Python `mcp` SDK (v1.9.x, FastMCP API) with stdio transport, deployed as a standalone process that shares a Python path with the existing `app-store-gui/webapp/` modules. All 8 tools are P1 for this milestone — none can be deferred — because the agent workflow breaks without any single one of them. The complete feature set (tools + SKILL.md + mock harness) should be delivered together before live OpenClaw registration is attempted.

The highest-risk area is not implementation complexity — the wrappers are straightforward — but contract correctness. Three critical errors can silently corrupt the agent loop: passing deeply nested internal inspection models as tool responses (agent wastes context budget decoding structure), routing agent YAML through the old `validate_structured_plan()` entrypoint (wrong contract), and implementing the apply approval gate only in SKILL.md rather than in tool code (single point of failure). These must be resolved in Phase 1 and Phase 2 respectively, before any tools are wired up to a live agent.

---

## Key Findings

### Recommended Stack

The only new dependencies required are `mcp[cli]>=1.9,<2` (the official Anthropic SDK with FastMCP bundled) and `pytest-asyncio>=1.3.0` for async test support. All existing stack components — FastAPI, kubernetes-client, PyYAML, pytest — are reused unmodified. The MCP server is a new top-level module (`mcp-server/`) at the repo root, not wired into the FastAPI app. OpenClaw spawns it as a child process via stdio using a `mcpServers` entry in `~/.openclaw/openclaw.json`.

**Core technologies:**
- `mcp[cli]>=1.9,<2`: MCP server framework — official SDK, FastMCP bundled, `@mcp.tool()` decorator, stdio transport by default; pin `<2` because v2 is pre-alpha as of March 2026
- `pytest-asyncio>=1.3.0`: Async test execution — required because FastMCP in-memory test client calls are coroutines; configure `asyncio_mode = strict` to avoid event loop conflict with anyio transitive dependency
- `mcp dev server.py` (CLI tool via `mcp[cli]`): Browser-based MCP Inspector for interactive tool testing — use before writing automated tests to verify tool schemas and response shapes

**What NOT to add:**
- `pip install fastmcp` (PyPI community fork — not the official SDK)
- `mcp>=2` (pre-alpha, breaking API changes)
- HTTP/SSE transport (adds operational complexity; stdio is correct for single-agent stdio spawn)
- Session or conversation state in any tool (OpenClaw owns this; duplicating it causes divergence bugs)

### Expected Features

All 8 tools are table stakes — every one is P1. Missing any breaks the agent loop. See `FEATURES.md` for full dependency graph.

**Must have (table stakes — this milestone):**
- `weka_appstore_inspect_cluster` — cluster GPU/CPU/RAM/namespace snapshot; wraps existing `inspection/cluster.py`
- `weka_appstore_inspect_weka` — WEKA filesystem capacity; wraps existing `inspection/weka.py`
- `weka_appstore_list_blueprints` — machine-consumable catalog with resource requirement hints; NOT a copy of the human UI
- `weka_appstore_get_blueprint` — full values schema and defaults for a named blueprint
- `weka_appstore_get_crd_schema` — static WekaAppStore CRD spec for agent YAML generation context
- `weka_appstore_validate_yaml` — CRD-contract validation with structured, field-keyed errors; wraps (but does NOT pass through) `planning/validator.py`
- `weka_appstore_apply` — submits WekaAppStore CR with hard approval gate enforced in tool code; wraps `planning/apply_gateway.py`
- `weka_appstore_status` — read-only CR status query
- Tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) on all 8 tools — machine-readable approval signals
- Freshness timestamps (`captured_at`) at the top level of every inspection response
- `SKILL.md` — OpenClaw workflow instructions with embedded CRD reference, validate-retry loop, re-inspect-before-apply instruction
- Mock harness — pytest fixtures + in-process FastMCP client exercising the full tool chain

**Should have (v1.x after core is stable):**
- Preflight re-inspect inside `apply` tool (`skip_preflight: bool = False`) — guards against stale inspection state
- Resource requirement hints in `list_blueprints` — lets agent prefilter without calling `get_blueprint` for each entry
- Blueprint catalog hot-reload — only if blueprint updates are frequent

**Defer (v2+):**
- Async status streaming — requires operator webhook support
- Multi-cluster support — requires OpenClaw session tracking across cluster contexts
- `weka_appstore_audit` — requires a persistent store not currently in the system

**Anti-features to avoid:**
- Session/conversation state in the MCP server (OpenClaw owns this)
- YAML generation inside the MCP server (agent's job)
- Natural language tool inputs (ambiguous contract)
- Streaming tool responses (not stable in MCP spec for tool invocation)
- Unrestricted `kubectl exec` or `helm install` passthrough (PRD explicitly forbids)
- Auto-apply without approval gate (PRD requirement OC-13)

### Architecture Approach

The MCP server runs as a standalone process that imports the existing `webapp/` business logic via a shared Python path. OpenClaw spawns it as a child process via stdio. The FastAPI app and MCP server run independently and are unaware of each other at runtime. All 8 tool implementations are thin wrappers — they validate incoming arguments, call existing functions, and map results to Pydantic response schemas defined in `mcp-server/schemas/responses.py`. Zero business logic lives in the wrapper layer.

**Major components:**
1. `mcp-server/server.py` — FastMCP instance; registers all 8 tools via `@mcp.tool()` decorator
2. `mcp-server/tools/*.py` — one file per tool group; thin adapters over existing business logic; no business logic here
3. `mcp-server/schemas/responses.py` — Pydantic models for stable tool output contracts; what the agent reasons against
4. `mcp-server/context/kube_client.py` — shared Kubernetes client with explicit read-only vs. read-write separation
5. `mcp-server/tests/mock_agent.py` — simulates OpenClaw tool-use loop for CI without a live agent
6. `SKILL.md` (repo root) — agent workflow instructions for OpenClaw; orchestrates tool call sequence

**Existing code to deprecate after MCP tools stabilize:**
- `planning/session_service.py`, `session_store.py` — OpenClaw owns state
- `planning/family_matcher.py`, `compiler.py` — OpenClaw reasons about this
- Planning session routes in `main.py` and `planning_session.html` template

### Critical Pitfalls

1. **Tool responses expose internal domain model structure** — `collect_cluster_inspection()` returns a deeply nested domain structure designed for the old backend contract, not agent consumption. Use `flatten_cluster_status()` (already exists) as tool output. Define flat, agent-facing Pydantic schemas in `responses.py` before writing any wrapper. Guard: answer extractable in 2 key traversals or fewer; response JSON under 2000 tokens.

2. **Apply approval gate lives only in SKILL.md** — a SKILL.md instruction is a single point of failure bypassed by misconfigured agents, direct test calls, or future model versions. The `apply` tool must enforce a hard code-level check (`confirmed: true` parameter or `approval_token`) that rejects any call without the signal. Guard: a direct test call to apply without the approval signal must return an error, never a created CR.

3. **Validator reused against the wrong contract** — `validate_structured_plan()` validates the v1.0 `StructuredPlan` contract (with `blueprint_family`, `fit_findings`, etc.), not a `WekaAppStore` CRD. Routing agent YAML through the old entrypoint rejects valid CRDs and accepts wrong-format documents. Write a new `validate_wekaappstore_yaml()` function; extract and reuse only the component-level helpers from the old validator. Guard: validator accepts valid WekaAppStore YAML and rejects YAML with `blueprint_family` or `fit_findings` fields.

4. **Deprecated session code creates parallel authority** — leaving `session_service.py`, `family_matcher.py`, and `compiler.py` active means two deployment paths exist. Developers will add features to the old path. Treat deprecation as a phased milestone deliverable: mark in Phase 1, disable FastAPI routes in Phase 2, delete in Phase 3.

5. **Tool descriptions written for developers, not agents** — descriptions that explain what a tool does ("queries Kubernetes CoreV1Api") cause wrong tool selection and wrong call sequencing. Each description must start with when and why to call it, include sequencing guidance, and stay under 200 tokens. Guard: mock harness agent simulation selects the correct tool from the description alone.

---

## Implications for Roadmap

All research converges on a 4-phase structure that matches the architectural dependency graph exactly. The phases are ordered to defer all mutation risk until the read path is validated, and to ensure the approval gate and correct validator contract are in place before any write operation is enabled.

### Phase 1: MCP Server Scaffold and Read-Only Tools

**Rationale:** Foundation must be in place before any tool is callable. Read-only tools carry zero mutation risk and can be tested immediately with mocked K8s responses. The output schema contracts defined here constrain every subsequent phase — getting them right first prevents retrofitting all wrappers later. The approval gate and validator contract issues (Pitfalls 1, 3, 5) must be resolved before any code is written.

**Delivers:** Runnable MCP server with 5 read-only tools (`inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`), per-tool tests with mocked backends, `mcp dev server.py` verification of all tool schemas and response shapes.

**Addresses features:** `inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`, tool annotations on all 5 tools, freshness timestamps.

**Avoids pitfalls:** Tool response over-engineering (define flat Pydantic schemas in `responses.py` first); tool descriptions written for humans (draft agent-facing descriptions with sequencing guidance before implementation); deprecated code authority conflict (add `# DEPRECATED` markers to `session_service.py`, `family_matcher.py`, `compiler.py`).

**Build order within phase:** `pyproject.toml` + `kube_client.py` → `schemas/responses.py` → tool wrappers → `server.py` registration → tests.

**Gate to Phase 2:** All 5 read-only tools callable via `mcp dev server.py`; mocked K8s responses produce correct agent-facing output; answer extractable in 2 key traversals or fewer.

### Phase 2: Validation, Apply, and Status Tools

**Rationale:** Mutation tools (`validate_yaml`, `apply`) depend on the read tools being stable — the agent must be able to inspect and discover before validating or applying. The apply approval gate and the correct validator contract are the highest-risk items in the project and must be resolved before the tool is callable by anyone. Status tool is read-only but logically follows apply.

**Delivers:** Remaining 3 tools (`validate_yaml`, `apply`, `status`), full mock harness (`mock_agent.py`) exercising the complete inspect → validate → apply chain including error paths, FastAPI planning session routes disabled (return 410).

**Addresses features:** `validate_yaml` with structured field-keyed errors, `apply` with hard approval gate, `status`, `destructiveHint` annotation on apply, all 8 tools callable.

**Avoids pitfalls:** Approval gate in code not SKILL.md (tool-level `confirmed` parameter check); wrong validator contract (`validate_wekaappstore_yaml()` against CRD, not `StructuredPlan`); stale inspection at apply time (staleness check / `inspection_age_warning` in apply tool); mock harness exercises error paths and approval bypass rejection.

**Gate to Phase 3:** Direct call to apply without `confirmed: true` returns error. `validate_yaml` accepts valid WekaAppStore YAML. Mock harness runs full loop against mocked backends without errors.

### Phase 3: Skill Definition and Agent Context

**Rationale:** SKILL.md cannot be written authoritatively until all 8 tools exist and their response shapes are finalized. CRD schema content (from `get_crd_schema`) must be verified complete before embedding in SKILL.md. Tool descriptions should be reviewed and tuned based on mock harness evidence before an agent ever sees them.

**Delivers:** `SKILL.md` with full workflow instructions, embedded CRD reference, validate-retry loop, re-inspect-before-apply instruction, explicit negative YAML examples; tool description review pass; deprecated code deleted (`session_service.py`, `session_store.py`, `family_matcher.py`, `compiler.py` removed from repo).

**Addresses features:** SKILL.md, structured validation errors, agent YAML hallucination prevention, blueprint catalog verified as LLM-friendly.

**Avoids pitfalls:** Agent YAML hallucination (SKILL.md includes negative examples and retry loop); stale inspection (SKILL.md instructs re-inspect immediately before apply); tool descriptions tuned based on harness evidence; deprecated code fully deleted (not just marked).

**Gate to Phase 4:** Mock harness agent simulation selects correct tools from descriptions. SKILL.md includes validate-retry instruction. Deprecated files absent from repo.

### Phase 4: Live OpenClaw Integration

**Rationale:** Only attempt live registration after all tools are tested end-to-end with the mock harness. Registration is configuration, not code. SKILL.md will need tuning based on actual agent behavior — expect 1-2 iteration cycles.

**Delivers:** Server registered in `~/.openclaw/openclaw.json`; tools/list discovery verified; each tool exercised via OpenClaw chat; approval gate firing confirmed; SKILL.md tuned based on live agent behavior; `openclaw.json` registration snippet documented.

**Addresses features:** Live OpenClaw integration, approval gate verification, NemoClaw alpha compatibility check.

**Avoids pitfalls:** Registering before end-to-end test passes (gate enforced by Phase 3 completion criterion).

**Note on NemoClaw:** NemoClaw alpha (announced March 16 2026) does not yet have a published configuration schema. Plan for a possible second SKILL.md format iteration when NemoClaw reaches stable release.

### Phase Ordering Rationale

- Read tools precede write tools because the agent cannot reason about fit without inspection data, and because read tools carry zero mutation risk — a safe way to validate the stack and schema contracts.
- Output schemas are defined in Phase 1 before any wrapper is written — this prevents retroactive flattening across all 8 tools.
- Apply approval gate and validator contract are resolved in Phase 2, not deferred — these are the highest-consequence errors and have HIGH recovery cost if discovered later.
- SKILL.md is written in Phase 3 after tools are stable — authoring it earlier would require revision every time a tool response shape changes.
- Deprecated code deletion is a Phase 3 hard deliverable, not cleanup — this closes the dual-authority risk permanently.
- Live OpenClaw registration is last — it is configuration that depends on all code being proven correct.

### Research Flags

Phases with well-documented patterns (skip research-phase):
- **Phase 1:** Tool scaffold, FastMCP `@mcp.tool()` decorator, stdio transport, pytest-asyncio setup — all well-documented in official MCP SDK and FastMCP docs; patterns are explicit in STACK.md.
- **Phase 2:** Apply approval gate pattern, K8s RBAC scoping — documented in PITFALLS.md and ARCHITECTURE.md with code examples.
- **Phase 3:** SKILL.md format — official OpenClaw docs and real skill registry examples available; format is explicit in FEATURES.md.

Phases likely needing deeper research during planning:
- **Phase 4 (NemoClaw):** NemoClaw alpha does not yet publish an official config schema as of March 16 2026. The OpenClaw YAML registration format is MEDIUM confidence (community docs only). Plan a research step before Phase 4 planning if NemoClaw stabilizes.
- **Phase 2 (blueprint catalog source):** `list_blueprints` and `get_blueprint` depend on a catalog source identified as "TBD" in FEATURES.md. The exact file path and schema for the existing blueprint catalog must be confirmed from the live codebase before Phase 2 planning completes.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official Anthropic MCP SDK confirmed on PyPI (v1.9.4); FastMCP API verified against official docs; pytest-asyncio v1.3.0 confirmed; one LOW-confidence item: NemoClaw config schema not yet published |
| Features | HIGH | PRD is precise and directly inspectable; MCP spec is current; existing codebase is inspectable; 8-tool surface is unambiguous |
| Architecture | HIGH | Official MCP SDK docs, FastMCP deployment guide, PRD analysis, and direct codebase inspection all converge on the same pattern |
| Pitfalls | HIGH | Verified against MCP specification, Anthropic engineering guidance, and the actual codebase; contract boundary issues identified from direct code inspection |

**Overall confidence:** HIGH

### Gaps to Address

- **Blueprint catalog schema (resolve before Phase 1 ends):** `list_blueprints` and `get_blueprint` wrap a catalog source identified as "TBD" in FEATURES.md. Before Phase 1 wraps these tools, confirm what files constitute the blueprint catalog, where they live in the repo, and whether existing metadata includes resource requirements or needs enrichment. Action: read `weka-app-store-operator-chart/` and any existing catalog-browsing code in `webapp/` before Phase 1 planning.

- **NemoClaw config schema (LOW priority for current milestone):** NemoClaw alpha has no published configuration schema as of March 16 2026. The OpenClaw registration format in STACK.md is MEDIUM confidence (community docs). This does not block Phase 1-3 development but may require a SKILL.md format revision for NemoClaw in Phase 4. Monitor NemoClaw release notes.

- **`apply_gateway.py` initialization side effects (resolve in Phase 1):** ARCHITECTURE.md notes that `apply_gateway.py` K8s client initialization behavior at import time vs. call time should be verified before the MCP server imports it. If it has import-time side effects, the MCP server startup will attempt to reach the K8s API in CI. Action: read `apply_gateway.py` initialization pattern before writing the apply tool wrapper.

- **RBAC service account (resolve before Phase 4):** The MCP server should use a separate service account with least-privilege RBAC — read-only for inspect tools, write scoped to `wekaappstores` for apply. This is a pre-Phase 4 deployment prerequisite, not a code change.

---

## Sources

### Primary (HIGH confidence)
- [PyPI `mcp` package](https://pypi.org/project/mcp/) — v1.9.4 confirmed latest stable, v2 pre-alpha
- [GitHub `modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk) — FastMCP API, `@mcp.tool()` decorator, in-process test client
- [MCP Tools Specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — response format, `isError`, output schema requirements
- [FastMCP testing guide](https://gofastmcp.com/servers/testing) — in-process client test pattern
- [Tool Annotations — MCP Blog 2026-03-16](https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/) — `readOnlyHint`, `destructiveHint`, `idempotentHint`
- [OpenClaw Skills Documentation](https://docs.openclaw.ai/tools/skills) — SKILL.md frontmatter fields and format
- [Anthropic Engineering: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — tool description quality, response design
- [MCP Security — Elastic Security Labs](https://www.elastic.co/security-labs/mcp-tools-attack-defense-recommendations) — approval bypass risk
- [pytest-asyncio PyPI](https://pypi.org/project/pytest-asyncio/) — v1.3.0 released November 2025
- `.planning/PRD-openclaw-integration.md` — tool specifications OC-01 through OC-18, authoritative for this project
- Codebase direct inspection: `inspection/cluster.py`, `planning/validator.py`, `planning/apply_gateway.py`, `planning/models.py`, `planning/session_service.py`

### Secondary (MEDIUM confidence)
- [OpenClaw MCP server YAML config format](https://www.clawctl.com/blog/mcp-server-setup-guide) — registration format (community guide, consistent with MCP spec)
- [OpenClaw `openclaw.json` configuration](https://openclawvps.io/blog/add-mcp-openclaw) — mcpServers format
- [MCPcat unit testing guide](https://mcpcat.io/guides/writing-unit-tests-mcp-servers/) — mock dependency injection, in-memory transport
- [MCP Tool Schema Bloat](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/) — token bloat patterns
- [NearForm: MCP tips and pitfalls](https://nearform.com/digital-community/implementing-model-context-protocol-mcp-tips-tricks-and-pitfalls/) — approval gates, testing without live agents

### Tertiary (LOW confidence — needs validation)
- NemoClaw alpha announcement — NVIDIA, March 16 2026 — NemoClaw config schema not yet published; OpenClaw compatibility details TBD

---
*Research completed: 2026-03-20*
*Ready for roadmap: yes*
