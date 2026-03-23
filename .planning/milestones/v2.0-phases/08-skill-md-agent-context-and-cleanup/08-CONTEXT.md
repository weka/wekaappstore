# Phase 8: SKILL.md, Agent Context, and Cleanup - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

SKILL.md authoritatively defines the agent workflow for blueprint planning, tool descriptions are tuned with sequencing hints, the OpenClaw registration config is generated, and all deprecated v1.0 backend-brain files are deleted from the repo. The 8-tool MCP server (built in Phases 6-7) is unchanged — this phase adds the agent guidance layer and cleans up legacy code.

</domain>

<decisions>
## Implementation Decisions

### SKILL.md Workflow
- Full step-by-step workflow: inspect_cluster -> inspect_weka -> list_blueprints -> get_blueprint -> get_crd_schema -> generate YAML -> validate_yaml -> fix if invalid -> re-inspect cluster -> confirm with user -> apply -> status
- Validate-retry loop: agent reads validation errors, fixes YAML, re-validates. Max 3 attempts before asking user for help
- Re-inspect-before-apply is mandatory: after YAML passes validation and before calling apply, always re-run inspect_cluster to confirm resources haven't changed
- Negative examples include v1.0 field mistakes (blueprint_family, fit_findings) AND common pitfalls (wrong apiVersion, missing metadata.name, skipping validation, applying without inspecting cluster first)

### Tool Description Tuning
- Sequencing hints in tool descriptions: each tool says when to use it relative to others (e.g., "Call this BEFORE apply", "Use AFTER inspect_cluster")
- Explicit cross-references by tool name in descriptions (e.g., "After validate_yaml passes, call apply")
- Key safety warnings on tools where misuse could cause problems (e.g., apply without validation, skipping confirmation)
- Mock harness updated to select tools via keyword matching on descriptions — proves descriptions are sufficient for tool selection without hardcoded tool names

### Cleanup Strategy
- Full removal of planning session routes from main.py — no 410 Gone stubs, delete entirely
- Delete the 4 explicit files: session_service.py, session_store.py, family_matcher.py, compiler.py
- Also remove models.py and inspection_tools.py if they have no remaining imports elsewhere
- Clean up __init__.py exports of deleted modules (leave __init__.py itself since package still needed for apply_gateway, validator)
- Remove planning_session.html template
- Full test suite verification after cleanup to confirm nothing broke
- Preserve: inspection/cluster.py, planning/apply_gateway.py, planning/validator.py and their tests

### OpenClaw Registration Config
- Best-effort openclaw.json based on MCP server spec conventions, with README noting it may need revision when NemoClaw alpha schema is published
- Full deployment spec: tool names with descriptions, stdio startup command, required env vars (BLUEPRINTS_DIR, KUBECONFIG), optional env vars, container image reference, resource limits, sidecar deployment pattern
- Located at mcp-server/openclaw.json
- Auto-generated from server.py tool registrations via a script — stays in sync as tools change

### Claude's Discretion
- SKILL.md prose style and formatting
- Exact keyword matching strategy for harness tool selection
- Which env vars are required vs optional in openclaw.json
- Order of cleanup operations (files first vs routes first)
- Whether to create a generate_openclaw_config.py script or inline generation in tests

</decisions>

<specifics>
## Specific Ideas

- The mock harness currently in mcp-server/harness/mock_agent.py imports _impl() functions directly with hardcoded tool selection — needs update to use description-based selection
- main.py has ~15 lines of planning session route code (lines 379-920 area) including create_planning_session_service(), helper functions, and 3+ route handlers
- NemoClaw alpha config schema not yet published as of 2026-03-20 (STATE.md blocker) — openclaw.json is best-effort

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/server.py`: Has all 8 tool registrations — source of truth for openclaw.json generation
- `mcp-server/harness/mock_agent.py`: Existing harness with 3 scenarios (happy path, approval bypass, validation failure) — needs description-based tool selection upgrade
- `mcp-server/tools/*.py`: Each tool has `register_*(mcp)` with @mcp.tool() descriptions — these are the descriptions to tune

### Established Patterns
- `register_*(mcp)` pattern: @mcp.tool() decorator with description string — descriptions already include some sequencing guidance from Phases 6-7
- `_impl(injectable)` pattern: all tool logic is injectable and testable without MCP framing
- `check_depth()` in test_response_depth.py: shared depth contract enforcer

### Integration Points
- SKILL.md goes at repo root or mcp-server/ — needs to be accessible to OpenClaw when registered
- openclaw.json references server.py startup command and tool names
- Cleanup touches app-store-gui/webapp/planning/ and app-store-gui/webapp/main.py

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-skill-md-agent-context-and-cleanup*
*Context gathered: 2026-03-20*
