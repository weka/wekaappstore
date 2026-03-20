# Pitfalls Research

**Domain:** MCP Tool Server Integration — adding agentic tool registration to an existing backend (WEKA App Store / OpenClaw)
**Researched:** 2026-03-20
**Confidence:** HIGH (verified against MCP specification, Anthropic engineering guidance, and the actual codebase)

---

## Critical Pitfalls

### Pitfall 1: Tool Response Over-Engineering Confuses the Agent

**What goes wrong:**
Tool responses that mirror the existing internal Python dataclass structure — nested `InspectionDomainFinding`, `FitFindings`, `InspectionSnapshot`, `ValidationResult` — are designed for backend-to-backend contracts, not agent consumption. When an LLM receives a deeply nested, multi-level object with internal codes, correlation IDs, normalization warnings, and domain-status enums, it spends context budget decoding structure instead of reasoning about content. The model may miss the key answer (e.g., "12 free GPUs") buried inside `domains.gpu.observed.free_devices`.

**Why it happens:**
The natural shortcut is to serialize existing Python models directly into tool responses. `inspection/cluster.py` already returns a rich domain-structured dict. Passing that raw output through the MCP tool feels like code reuse. But the schema was designed for the old planning session contract (where the backend parsed it), not for an agent that needs to act on the number.

**How to avoid:**
Define a separate, flattened agent-facing response shape for each tool. For `inspect_cluster`, return a flat summary: `{ "free_gpus": 4, "gpu_model": "H100", "free_cpu_cores": 48, "free_memory_gib": 192, "namespaces": [...], "captured_at": "..." }`. Expose `flatten_cluster_status()` (which already exists in `inspection/cluster.py`) as the tool output, not `collect_cluster_inspection()`. The rich domain format can be retained internally for logging. Keep a separate `weka_appstore_inspect_cluster_detailed` tool if the agent ever needs the structured form.

**Warning signs:**
- Mock harness calls the tool and the useful answer requires 4+ key traversals to extract.
- Tool response JSON exceeds 2000 tokens for a single snapshot.
- The SKILL.md guidance has to tell the agent how to parse the response structure.

**Phase to address:**
Phase 1 (MCP Server Scaffold and Read-Only Tools) — define the output shape contract before writing tool implementations.

---

### Pitfall 2: Approval Gate Lives Only in Trust, Not in Code

**What goes wrong:**
The `weka_appstore_apply` tool accepts YAML and creates a `WekaAppStore` CR. If the approval gate is implemented as "the agent should ask the user first" (a prompt instruction in SKILL.md) rather than as a hard check inside the tool itself, there is nothing preventing:
- A misconfigured OpenClaw skill that skips the confirmation step
- A future agent version or model that reasons around the instruction
- A direct MCP call during testing that bypasses all agent-level logic

The PRD explicitly identifies this risk: "The agent could attempt to apply without approval if poorly configured."

**Why it happens:**
Trusting OpenClaw's built-in approval system is the right architectural intent, but delegating all gating to the agent layer creates a single point of failure. Developers building the tool layer assume the agent layer is always present and always correct.

**How to avoid:**
The `apply` tool must enforce a gate that is independent of agent behavior. Options:
1. Require a `confirmed: true` parameter that the tool checks before proceeding — the agent must explicitly set it, and SKILL.md instructs the agent never to set it without user confirmation.
2. Require an `approval_token` that OpenClaw generates after its own approval flow, and the tool validates it.
3. Implement a two-phase apply: `weka_appstore_prepare_apply` (returns a preview and a short-lived token), `weka_appstore_confirm_apply` (consumes the token).

Option 1 is the minimum. The key is that the tool itself rejects a call without the approval signal — it does not trust that no call will arrive without one.

**Warning signs:**
- The mock harness can call `apply` in a single step with no confirmation parameter.
- SKILL.md says "ask the user" but the tool code has no guard.
- A test that calls the apply tool directly (bypassing all agent logic) succeeds and creates a CR.

**Phase to address:**
Phase 2 (Validation and Apply Tools) — approval gate enforced in tool implementation, not deferred to SKILL.md.

---

### Pitfall 3: Mock Harness Tests the Tool API, Not the Agent-Tool Interaction

**What goes wrong:**
The mock harness proves each tool returns valid JSON when called with known inputs. But when a real OpenClaw agent calls the tools:
- Tool descriptions are ambiguous and the agent picks the wrong tool for the job
- The agent constructs parameter values the tool doesn't expect (e.g., passes a `blueprint_name` where a `blueprint_id` is required)
- The agent misinterprets a response flag (`"status": "partial"`) and draws a wrong conclusion
- The agent calls tools in an unintended sequence (e.g., calls `apply` before `validate`)
- The agent loops on a tool call when it should stop

A mock harness that calls `tool_function(known_args)` and asserts `response == expected` does not catch any of these issues because the agent is not part of the loop.

**Why it happens:**
Development without a live OpenClaw instance is the explicit project constraint. The temptation is to test what is measurable: tool inputs and outputs. The agent-facing contract (descriptions, parameter naming, response semantics) goes untested because there is no agent to test it with.

**How to avoid:**
Build two levels of mock testing:
1. **Unit tests**: call each tool directly, assert response shape — this is necessary but not sufficient.
2. **Simulated tool-use loop**: write a harness that sequences tool calls as a real agent would, driven by a scripted decision tree. The harness should: call `inspect_cluster`, then `list_blueprints`, then `get_blueprint`, then `validate_yaml` on a generated example, then attempt `apply` — and verify the full chain behaves correctly including the approval gate.

Separately, write explicit tests for the descriptions and parameter names: read the tool description and check it would unambiguously route an agent to the right tool for each scenario. This is a human review step, not automated.

**Warning signs:**
- All harness tests call tools with hardcoded kwargs, never constructing calls from natural language intent.
- The harness never exercises an error path (e.g., `validate_yaml` returning errors → agent retries).
- No test checks that the approval confirmation cannot be bypassed.

**Phase to address:**
Phase 1 (tool structure), Phase 2 (harness exercises apply and approval), Phase 3 (end-to-end chain test).

---

### Pitfall 4: Validator Ported from v1.0 Validates the Wrong Contract

**What goes wrong:**
`planning/validator.py` validates a `StructuredPlan` — the v1.0 backend-brain contract that includes `blueprint_family`, `fit_findings`, `unresolved_questions`, `reasoning_summary`, and deeply nested domain findings. In v2.0, the `weka_appstore_validate_yaml` tool validates **YAML that the agent generates for the WekaAppStore CRD**, not a planning session contract.

If `validator.py` is reused without modification, the `validate_yaml` tool will:
- Reject valid `WekaAppStore` YAML because it doesn't match `StructuredPlan` fields
- Accept invalid YAML because the CRD contract is different from the planning session contract
- Return errors referencing internal fields (`blueprint_family`, `fit_findings`) that the agent has no context for

**Why it happens:**
The instruction to "reuse existing `planning/validator.py`" is correct in spirit (the CRD-level checks like component naming, dependency references, and Helm fields are still needed), but the top-level schema validation is v1.0-specific. The reuse boundary is at the component-level helper functions, not the top-level `validate_structured_plan()` entrypoint.

**How to avoid:**
Write a new `validate_wekaappstore_yaml()` function that validates against the actual `WekaAppStore` CRD shape (`apiVersion`, `kind`, `metadata.name`, `spec.appStack.components[]`). Extract and reuse the component-level helpers from `validator.py` (`_validate_component_contracts`, `_parse_helm_chart`, `_parse_readiness_check`) since those rules apply to both contracts. Do not pass YAML from the agent through `validate_structured_plan()`.

**Warning signs:**
- The `validate_yaml` tool returns errors about `blueprint_family`, `reasoning_summary`, or `fit_findings` — fields that have no meaning in a `WekaAppStore` CR.
- Agent-generated YAML that creates a valid CR passes validation, but valid-looking YAML is rejected with "unsupported field" errors.
- The validator imports `SUPPORTED_BLUEPRINT_FAMILIES` from `models.py` and uses it to block the agent's output.

**Phase to address:**
Phase 2 (Validation and Apply Tools) — clarify the validation contract boundary before implementation.

---

### Pitfall 5: Deprecated Code Lives Alongside New Tool Code, Creating Two Authority Sources

**What goes wrong:**
`planning/session_service.py`, `planning/family_matcher.py`, and `planning/compiler.py` are scheduled for deletion. If deletion is deferred ("we'll clean it up later"), two things happen:
- Developers maintaining the codebase write new logic against the old service layer instead of the new MCP tools, because the old code is still there and wired up
- The FastAPI routes that use the session service continue to work, creating a parallel deploy path that doesn't go through the OpenClaw gate and approval system
- Integration bugs are masked because both paths "work" in isolation

This is the classic brownfield migration failure mode: the new system is built, but the old system is never fully turned off, and the two drift apart.

**Why it happens:**
Removing working code feels risky. The old session service has test coverage; removing it means those tests disappear. The planning session HTML is still in `templates/` and still renders. Nobody explicitly breaks anything, so nobody removes anything.

**How to avoid:**
Phase the deprecation explicitly in the roadmap:
1. In Phase 1: mark deprecated modules with a top-level `# DEPRECATED: see MCP tool implementation` comment.
2. In Phase 2: disable the FastAPI planning session routes (return 410 Gone) and remove the session HTML template. The code can remain but must not be callable.
3. In Phase 3 (or when the mock harness proves the MCP path works end-to-end): delete `session_service.py`, `session_store.py`, `family_matcher.py`, `compiler.py`, and their tests.

Treat deletion as a milestone deliverable, not cleanup.

**Warning signs:**
- A developer adds a feature to `session_service.py` during v2.0 work.
- The planning session routes still return 200 responses after Phase 2.
- The old planning session HTML is still linked from the main UI.

**Phase to address:**
Phase 1 (mark deprecated), Phase 2 (disable routes), Phase 3 (delete).

---

### Pitfall 6: Agent YAML Generation Hallucinates Fields the CRD Does Not Support

**What goes wrong:**
When the agent is given the `WekaAppStore` CRD schema as context and asked to generate YAML, it will sometimes produce fields that look plausible but do not exist in the schema:
- Inventing a `resources:` block at the component level
- Using `image:` or `tag:` as top-level component fields instead of inside `values:`
- Generating `apiVersion: warp.io/v1` instead of `warp.io/v1alpha1`
- Producing valid Helm values syntax that the operator does not process (e.g., `helmChart.valueFiles` instead of `helmChart.values_files`)

The agent doesn't hallucinate randomly — it generalizes from Kubernetes YAML patterns it has seen in training data, which is broader than the specific `WekaAppStore` contract.

**Why it happens:**
The CRD schema context provided via `weka_appstore_get_crd_schema` is not enough to suppress generalization. The agent knows more Kubernetes patterns than the CRD allows, and when uncertain, fills gaps with plausible-but-wrong fields.

**How to avoid:**
- `validate_yaml` must catch and return actionable errors for every hallucinated field — not just "unknown field" but "field `X` is not valid; use `Y` instead where applicable."
- The SKILL.md must include explicit negative examples: "Do NOT add `resources:` to components. Do NOT add `image:` at the component level."
- Provide 2-3 complete valid YAML examples in the CRD schema context, not just the schema spec.
- The validate-then-retry loop must be explicitly scripted in SKILL.md: "If validate_yaml returns errors, fix them and validate again before applying."

**Warning signs:**
- `validate_yaml` receives YAML with Kubernetes-standard fields that are not in the `WekaAppStore` spec.
- The agent succeeds on the first validation attempt consistently in mock testing but fails on the first real agent attempt.
- SKILL.md does not include a validate-retry instruction.

**Phase to address:**
Phase 2 (validate_yaml returns actionable errors), Phase 3 (SKILL.md includes examples and retry loop).

---

### Pitfall 7: Inspection Tools Return Stale Data, Agent Applies Without Re-Checking

**What goes wrong:**
The agent calls `inspect_cluster` at the start of a conversation and uses those values throughout the planning session. By the time it calls `apply`, the cluster state may have changed: a GPU was allocated by another workload, a namespace was deleted, a StorageClass was removed. The apply succeeds at the MCP tool level (creates the CR), but the operator reconcile fails because the resources assumed to be available are no longer there.

**Why it happens:**
`inspect_cluster` returns a snapshot with a `captured_at` timestamp, but nothing in the protocol forces the agent to re-inspect before applying. The SKILL.md may instruct the agent to do this, but agents follow SKILL.md loosely when the instructions are long.

**How to avoid:**
- The `weka_appstore_apply` tool should optionally re-inspect critical resources before creating the CR and include an `inspection_age_warning` in its response if the most recent snapshot is older than a threshold (e.g., 5 minutes).
- SKILL.md should state: "Call `inspect_cluster` and `inspect_weka` immediately before calling `apply`, not only at the start of the conversation."
- Tool responses must include `captured_at` prominently at the top level so the agent can reason about staleness.
- This is partly addressed by the existing `InspectionFreshness` design in the codebase — ensure it surfaces to the agent, not just to internal validators.

**Warning signs:**
- The mock harness calls `inspect_cluster` once and then calls `apply` directly without re-inspection.
- `captured_at` is nested inside `domains.cpu.freshness` rather than at the response root.
- SKILL.md does not mention staleness or re-inspection.

**Phase to address:**
Phase 2 (apply tool includes staleness check), Phase 3 (SKILL.md includes re-inspection instruction).

---

### Pitfall 8: Tool Descriptions Are Written for Humans, Not Agents

**What goes wrong:**
Tool descriptions that work well for human API docs cause agent confusion:
- Descriptions that explain what the tool does rather than when to use it ("Returns GPU, CPU, RAM, namespace, and storage class data" vs "Call this first when assessing whether the cluster can support a blueprint deployment")
- Parameter names that are ambiguous or internally-named (`blueprint_id` vs `blueprint_name`)
- Missing guidance on what to do with the response
- No indication of which tools depend on which other tools

The agent must infer call sequencing and interpretation from descriptions alone during live operation. Poorly written descriptions mean the agent picks the wrong tool, passes wrong parameters, or calls tools in the wrong order.

**Why it happens:**
Tool descriptions are written by developers who understand the codebase. They document what the tool does, not how an agent should reason about using it. The distinction becomes visible only when a real agent misuses the tool.

**How to avoid:**
Follow Anthropic's engineering guidance on tool descriptions:
- Start with the decision context: "Call this tool to determine if the cluster has sufficient GPU, CPU, and RAM to run a blueprint before generating a deployment plan."
- Include sequencing: "Call `inspect_cluster` and `inspect_weka` before `list_blueprints` or `validate_yaml`."
- Include response guidance: "Use `free_gpus` and `gpu_model` to assess GPU fit. Use `free_memory_gib` for RAM requirements."
- Keep descriptions under 200 tokens. Remove tutorial-style prose.
- Name parameters for what the agent will have, not internal identifiers (`blueprint_name`, not `blueprint_id`).

**Warning signs:**
- Description of any tool exceeds 300 tokens.
- The description explains implementation details ("queries the Kubernetes API using CoreV1Api").
- SKILL.md has to re-explain what each tool does because the tool description is inadequate.

**Phase to address:**
Phase 1 (write initial descriptions), Phase 3 (review and tune after end-to-end harness testing).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Pass `collect_cluster_inspection()` output directly as tool response | No extra serialization code | Agent receives nested domain structure it must traverse; context bloat; descriptions need to explain the format | Never — `flatten_cluster_status()` already exists |
| Validate agent YAML through `validate_structured_plan()` | Reuses existing tested validator | Wrong contract — rejects valid CRDs, accepts wrong format | Never — write `validate_wekaappstore_yaml()` |
| Implement approval gate only in SKILL.md | No code change to apply tool | Single point of failure; bypassed by direct calls or misconfigured agent | Never for production; acceptable only in initial Phase 1 stub |
| Keep deprecated session service code while building MCP tools | No immediate breakage | Two authority sources; developers add features to old path; parallel deploy path | Phase 1 only with explicit deprecation markers; must be removed by Phase 3 |
| Hardcode blueprint family list in the validator | Simple to check | Validator breaks every time a new blueprint is added; not data-driven | Never — load from catalog dynamically |
| Return all inspection domains even when not needed | Complete data | Bloats every response with irrelevant domains; wasted context tokens | Never for standard calls; add `domains` parameter to filter |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| OpenClaw MCP registration | Registering the MCP server before testing tool call/response semantics with mock agent | Test tool chain end-to-end with scripted mock harness first; only register with OpenClaw after mock tests pass |
| `inspection/cluster.py` reuse | Calling `collect_cluster_inspection()` in the tool handler, serializing its full output | Call `flatten_cluster_status(collect_cluster_inspection(...))` to get the agent-facing flat dict |
| `planning/validator.py` reuse | Passing agent-generated YAML through `validate_structured_plan()` | Extract component-level helpers; write new top-level `validate_wekaappstore_yaml()` for the CRD contract |
| `planning/apply_gateway.py` reuse | Calling `apply_yaml_content_with_namespace()` directly from the tool | Wrap in a gated function that checks the `confirmed` parameter before delegating |
| Kubernetes RBAC for the MCP server | Running the MCP server with the same service account as the operator (full cluster write access) | Create a separate service account for the MCP server with read-only access plus write permission scoped to `wekaappstores` resources only |
| WEKA API connection | Using the same WEKA API credentials used by the operator | Use a read-only WEKA API token for inspection tools; apply tool does not need WEKA write access |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full cluster inspection on every tool call | Every `list_blueprints` call also queries the K8s API for no reason | Inspection is only triggered by `inspect_cluster` and `inspect_weka` — other tools are static | First tool call in a chain |
| Passing full CRD spec as static context in every tool response | Every tool response includes the 10KB CRD spec | Only `get_crd_schema` returns the CRD spec; other tools do not include it | At scale when multiple tools are registered |
| Tool response without pagination or filtering | `list_blueprints` returns every field for every blueprint including full Helm values schemas | Return name, description, category, and resource requirements only; use `get_blueprint` for full detail | When blueprint catalog grows beyond ~10 entries |
| All 8 tools preloaded into agent context simultaneously | Agent context budget consumed by tool schemas before any conversation | Keep tool descriptions concise; group read-only tools under a shared `instructions` prefix; total tool schema budget should be under 5000 tokens | When OpenClaw loads tools at session start |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| MCP server runs with operator service account | A compromised MCP server or agent prompt injection can delete CRDs, nodes, or operator state | Separate service account with least-privilege RBAC: read-only for inspect tools, write scoped to `wekaappstores` for apply |
| `apply` tool accepts arbitrary YAML strings without validation | Prompt injection could construct YAML that creates non-`WekaAppStore` resources | Always run `validate_wekaappstore_yaml()` inside the apply tool before passing to `apply_gateway.py`, regardless of whether the caller already called `validate_yaml` |
| No audit log for tool calls | Cannot trace which agent call created which CR or whether the approval gate was exercised | Log every tool call with: tool name, parameters hash, timestamp, caller identity (OpenClaw session ID if available), and result status |
| Tool descriptions expose internal implementation details | Description-based prompt injection can trick agent into extracting internal state | Descriptions should state behavior and usage, not implementation (no mentions of `CoreV1Api`, `warp.io` API group internals, or file paths) |

---

## "Looks Done But Isn't" Checklist

- [ ] **`validate_yaml` tool:** Often missing validation of `apiVersion` and `kind` values — verify it rejects `warp.io/v1` (wrong version) and `WekaAppStore` with wrong casing.
- [ ] **`apply` tool:** Often missing the approval gate in the tool body — verify a direct call without `confirmed: true` returns an error, not a created CR.
- [ ] **`inspect_cluster` response:** Often missing `captured_at` at the top level — verify it is present and not only inside `domains.cpu.freshness`.
- [ ] **Tool descriptions:** Often written in "what it does" style — verify each description starts with when/why to call it, not what it calls internally.
- [ ] **Deprecated code removal:** Often scheduled but not enforced — verify `session_service.py`, `family_matcher.py`, and `compiler.py` are unreachable (routes disabled or deleted) before Phase 3 ends.
- [ ] **SKILL.md validate-retry loop:** Often omitted — verify SKILL.md explicitly instructs the agent to re-validate after fixing errors before applying.
- [ ] **Blueprint catalog tool:** Often returns full Helm schema in `list_blueprints` — verify it returns summary only and delegates full detail to `get_blueprint`.
- [ ] **Namespace resolution:** The existing `apply_gateway.py` normalizes `targetNamespace` only when the field already exists. Verify agent-generated YAML always includes `targetNamespace` on each component, or the gateway falls back to the CR namespace.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Tool response over-engineering discovered after Phase 1 | MEDIUM | Add a flattening layer in the tool handler without changing `cluster.py`; no client changes needed since MCP is server-side |
| Approval gate missing from apply tool | HIGH | Requires adding required parameter to tool signature, updating SKILL.md, re-registering tool with OpenClaw; any existing test cases that call apply without confirmation must be updated |
| Wrong validator reused for YAML validation | MEDIUM | Write new validator function alongside old one; redirect `validate_yaml` tool to new function; old validator remains for any legacy paths still using it |
| Deprecated code not removed | LOW-MEDIUM | Disable routes first (immediate, low risk), then schedule deletion with test removal; no data migration needed |
| Agent hallucinates CRD fields | LOW | Add negative examples to SKILL.md and improve `validate_yaml` error messages to be prescriptive; no code architecture change needed |
| Tool descriptions cause wrong tool selection | LOW | Update descriptions (no API change); re-test with mock harness |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Tool response over-engineering | Phase 1 | Agent-facing response shape defined and reviewed before implementation; mock harness extracts answer in ≤2 key traversals |
| Approval gate only in trust | Phase 2 | Direct call to apply tool without `confirmed: true` returns error, not a created CR |
| Mock harness tests tool API only | Phase 1-3 | Harness exercises full inspect → validate → apply chain including error paths |
| Validator validates wrong contract | Phase 2 | `validate_yaml` tool accepts valid `WekaAppStore` YAML and rejects invalid CRD YAML (not planning session fields) |
| Deprecated code creates dual authority | Phase 1 (mark), Phase 2 (disable routes), Phase 3 (delete) | Planning session routes return 410; deprecated module files removed from repo |
| Agent YAML hallucination | Phase 2 (validator), Phase 3 (SKILL.md) | SKILL.md includes examples and retry loop; `validate_yaml` returns field-specific prescriptive errors |
| Stale inspection at apply time | Phase 2 (apply tool), Phase 3 (SKILL.md) | Apply tool includes `inspection_age_warning`; SKILL.md instructs re-inspect before apply |
| Tool descriptions written for humans | Phase 1 (draft), Phase 3 (review) | Mock harness agent simulation selects correct tools from descriptions alone |

---

## Sources

- [Anthropic Engineering: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — tool description quality, response design, context efficiency (HIGH confidence)
- [MCP Specification: Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) — structured response format, output schema requirements (HIGH confidence)
- [MCP Tool Schema Bloat: The Hidden Token Tax](https://layered.dev/mcp-tool-schema-bloat-the-hidden-token-tax-and-how-to-fix-it/) — token bloat patterns, verbose descriptions, schema redundancy (HIGH confidence)
- [NearForm: Implementing MCP — Tips, Tricks, and Pitfalls](https://nearform.com/digital-community/implementing-model-context-protocol-mcp-tips-tricks-and-pitfalls/) — approval gates, testing without live agents, version drift (MEDIUM confidence)
- [MCP Security for Agent-Tool Interactions — Elastic Security Labs](https://www.elastic.co/security-labs/mcp-tools-attack-defense-recommendations) — approval bypass, autonomous action risks (HIGH confidence)
- [Building Least-Privilege AI Agent Gateway — InfoQ](https://www.infoq.com/articles/building-ai-agent-gateway-mcp/) — RBAC scoping, policy-as-code for MCP gateways (MEDIUM confidence)
- [MCP Context Window Problem — Junia AI](https://www.junia.ai/blog/mcp-context-window-problem) — context bloat from tool schema loading (MEDIUM confidence)
- [Anthropic Engineering: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — token efficiency, dynamic tool loading patterns (HIGH confidence)
- Codebase review: `inspection/cluster.py`, `planning/validator.py`, `planning/apply_gateway.py`, `planning/models.py`, `planning/session_service.py` — contract analysis and reuse boundary identification (HIGH confidence — direct inspection)
- PRD: `PRD-openclaw-integration.md` — architecture intent, risks identified by the project (HIGH confidence)

---
*Pitfalls research for: MCP tool server integration into existing WEKA App Store backend*
*Researched: 2026-03-20*
