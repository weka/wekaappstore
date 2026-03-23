# Phase 08: SKILL.md, Agent Context, and Cleanup - Research

**Researched:** 2026-03-20
**Domain:** Agent documentation, MCP tool description tuning, legacy code removal, OpenClaw registration config
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**SKILL.md Workflow**
- Full step-by-step workflow: inspect_cluster -> inspect_weka -> list_blueprints -> get_blueprint -> get_crd_schema -> generate YAML -> validate_yaml -> fix if invalid -> re-inspect cluster -> confirm with user -> apply -> status
- Validate-retry loop: agent reads validation errors, fixes YAML, re-validates. Max 3 attempts before asking user for help
- Re-inspect-before-apply is mandatory: after YAML passes validation and before calling apply, always re-run inspect_cluster to confirm resources haven't changed
- Negative examples include v1.0 field mistakes (blueprint_family, fit_findings) AND common pitfalls (wrong apiVersion, missing metadata.name, skipping validation, applying without inspecting cluster first)

**Tool Description Tuning**
- Sequencing hints in tool descriptions: each tool says when to use it relative to others (e.g., "Call this BEFORE apply", "Use AFTER inspect_cluster")
- Explicit cross-references by tool name in descriptions (e.g., "After validate_yaml passes, call apply")
- Key safety warnings on tools where misuse could cause problems (e.g., apply without validation, skipping confirmation)
- Mock harness updated to select tools via keyword matching on descriptions — proves descriptions are sufficient for tool selection without hardcoded tool names

**Cleanup Strategy**
- Full removal of planning session routes from main.py — no 410 Gone stubs, delete entirely
- Delete the 4 explicit files: session_service.py, session_store.py, family_matcher.py, compiler.py
- Also remove models.py and inspection_tools.py if they have no remaining imports elsewhere
- Clean up __init__.py exports of deleted modules (leave __init__.py itself since package still needed for apply_gateway, validator)
- Remove planning_session.html template
- Full test suite verification after cleanup to confirm nothing broke
- Preserve: inspection/cluster.py, planning/apply_gateway.py, planning/validator.py and their tests

**OpenClaw Registration Config**
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AGNT-01 | SKILL.md defines the agent workflow with validate-before-apply constraint and negative examples | Full workflow confirmed from tool descriptions in tools/*.py; negative examples grounded in actual v1.0 fields (blueprint_family, fit_findings) and validation codes |
| AGNT-03 | OpenClaw registration config (openclaw.json / NemoClaw equivalent) generated for the MCP server | 8 tools confirmed in server.py with register_*(mcp) pattern; env vars BLUEPRINTS_DIR and KUBERNETES_AUTH_MODE confirmed in config.py; stdio startup command is `python -m server` from mcp-server/ |
| CLEAN-01 | Remove deprecated planning/session_service.py, planning/session_store.py, planning/family_matcher.py, planning/compiler.py | All 4 files confirmed present; dependency analysis shows models.py and inspection_tools.py are also removal candidates; validator.py depends on models.py (must keep) |
| CLEAN-02 | Remove deprecated planning session routes and planning_session.html template from main.py | 6 route handlers confirmed (lines ~834-1028), 8 helper functions (lines ~270-417), and template confirmed at app-store-gui/webapp/templates/planning_session.html |
| CLEAN-03 | Preserve inspection/cluster.py, planning/apply_gateway.py, planning/validator.py as tool implementations | apply_gateway.py is imported by mcp-server/tools/apply_tool.py and harness; validator.py imports models.py; apply_gateway.py has no internal planning imports |
</phase_requirements>

---

## Summary

Phase 8 is a documentation-and-cleanup phase. The MCP server (8 tools, fully functional from Phases 6-7) is unchanged. The work has three independent streams: (1) write SKILL.md as the authoritative agent workflow document, (2) tune tool descriptions in `tools/*.py` to include sequencing hints and update the mock harness to prove descriptions drive tool selection, and (3) delete deprecated v1.0 backend-brain code from the app-store-gui.

The cleanup scope is well-defined but has one dependency subtlety: `validator.py` imports `models.py`, so models.py must be preserved even though it's a v1.0 data-model file. `inspection_tools.py` has no imports from deleted files but is used only by planning session code; it can be removed once main.py no longer imports it. The `__init__.py` needs surgical editing to remove 20+ exports from deleted modules while keeping the 5 apply_gateway exports intact.

**Primary recommendation:** Execute in three waves — Wave 1: SKILL.md + tool description tuning, Wave 2: openclaw.json generation, Wave 3: cleanup + verification. The waves are independent enough to parallelize within a plan but the final test verification pass must come last.

---

## Standard Stack

### Core (already installed — no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mcp[cli] | >=1.26.0 | FastMCP tool registration and stdio transport | Already in mcp-server/requirements.txt |
| PyYAML | >=6.0.1 | YAML parsing in validate_yaml and blueprints tools | Already installed |
| pytest | >=8.0.0 | Test runner for both mcp-server and app-store-gui | Already in requirements |
| kubernetes | >=27.0.0 | K8s client used in apply/status/inspect tools | Already in requirements |

### No New Libraries Needed
This phase is pure documentation, configuration, and deletion — no new packages required.

**Installation:** None required.

---

## Architecture Patterns

### Current Tool Registration Pattern
All 8 tools follow the same pattern established in Phase 6:

```python
# Source: mcp-server/tools/inspect_cluster.py (and all other tools/*.py)
def register_inspect_cluster(mcp: Any) -> None:
    @mcp.tool()
    def inspect_cluster() -> dict:
        """<description string — THIS is what agents read>"""
        return _impl_function()
```

The description string inside `@mcp.tool()` is the agent's primary signal for tool selection. This is the string to tune.

### SKILL.md Structure Pattern
SKILL.md is a plain Markdown file placed at repo root or `mcp-server/`. It is consumed by the OpenClaw agent registration and loaded as system-level context. Based on context decisions, it must contain:

1. **Workflow section** — numbered step-by-step tool sequence
2. **Validate-retry loop** — explicit max-3-attempt rule
3. **Re-inspect-before-apply rule** — mandatory re-run after validation passes
4. **Negative examples** — YAML snippets with annotations showing what NOT to do
5. **Tool reference** — brief per-tool description linking to the workflow steps

### Description-Based Tool Selection Pattern for Harness
The CONTEXT.md decision is to upgrade `mock_agent.py` so it selects tools via keyword matching on descriptions, proving descriptions alone are sufficient. The current harness (Phase 7) calls `_impl()` functions directly with hardcoded tool selection. The new pattern:

```python
# Pattern: build a tool registry from descriptions, select by keyword match
TOOL_REGISTRY = {
    "inspect_cluster": {
        "description": "...",
        "fn": flatten_inspect_cluster_for_mcp,
    },
    # ... all 8 tools
}

def select_tool(intent_keywords: list[str], registry: dict) -> str:
    """Select tool name by matching intent keywords against description text."""
    for tool_name, entry in registry.items():
        if any(kw.lower() in entry["description"].lower() for kw in intent_keywords):
            return tool_name
    raise ValueError(f"No tool matched keywords: {intent_keywords}")
```

The 3 existing scenarios (happy path, approval bypass, validation failure) remain — they just use description-driven selection instead of hardcoded function calls.

### openclaw.json Structure Pattern
No official NemoClaw schema published yet (STATE.md blocker confirmed). Based on MCP server conventions, the JSON must capture:

```json
{
  "name": "weka-app-store-mcp",
  "description": "...",
  "transport": "stdio",
  "startup": {
    "command": "python",
    "args": ["-m", "server"],
    "cwd": "mcp-server/"
  },
  "env": {
    "required": ["BLUEPRINTS_DIR"],
    "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "KUBECONFIG"]
  },
  "tools": [
    {
      "name": "inspect_cluster",
      "description": "<from @mcp.tool() decorator>"
    }
    // ... all 8 tools
  ]
}
```

### __init__.py Surgical Edit Pattern
The `planning/__init__.py` currently exports ~130 symbols from 7 modules. After cleanup, only the 5 apply_gateway exports survive in __init__.py:

**Keep in __init__.py:**
```python
from .apply_gateway import (
    ApplyGateway,
    ApplyGatewayDependencies,
    apply_yaml_content_with_namespace,
    apply_yaml_documents_with_namespace,
    apply_yaml_file_with_namespace,
)
```

**Also keep** (validator.py imports models.py — both must be preserved):
```python
from .validator import validate_structured_plan
# models.py: imported by validator.py — file stays but NOT re-exported from __init__.py
```

**Remove from __init__.py** (all imports from deleted modules):
- All imports from `.models` (exported at top of current __init__.py)
- All imports from `.family_matcher`
- All imports from `.compiler`
- All imports from `.session_store`
- All imports from `.session_service`
- All imports from `.inspection_tools` (if file deleted)

### Recommended Project Structure Post-Cleanup
```
app-store-gui/webapp/planning/
├── __init__.py          # 5 apply_gateway exports only (+ validator if needed)
├── apply_gateway.py     # PRESERVED — MCP server imports this
├── models.py            # PRESERVED — validator.py imports this
└── validator.py         # PRESERVED — but only used by legacy tests after cleanup

mcp-server/
├── server.py            # Unchanged
├── tools/*.py           # 8 tool files with tuned descriptions
├── harness/
│   └── mock_agent.py    # Updated: description-based tool selection
├── SKILL.md             # NEW — authoritative agent workflow
└── openclaw.json        # NEW — OpenClaw registration config
```

### Anti-Patterns to Avoid
- **Editing SKILL.md as code:** SKILL.md is prose documentation — no unit tests should validate its content, only the harness selection test proves descriptions work
- **Removing models.py:** validator.py has a hard `from .models import (...)` — deleting models.py breaks validator.py and apply_gateway tests
- **Deleting planning/__init__.py:** The apply_gateway and validator are still needed; only the exports change
- **Adding 410 Gone stubs:** Context decision is full deletion of planning routes — no stub endpoints

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenClaw config schema | Custom schema inference | Use server.py tool registration as source of truth, generate from it | Server.py already has all 8 tool names and descriptions in register_*(mcp) functions |
| Tool description validation | Custom linter | Write a harness test that fails if descriptions don't contain sequencing keywords | Proves the content, not just presence |
| Import graph analysis for cleanup | Manual tracing | Read __init__.py + grep for all cross-module imports | The dependency chain is shallow — 3 layers max |

---

## Common Pitfalls

### Pitfall 1: models.py Hidden Dependency
**What goes wrong:** Deleting models.py because "it's a v1.0 file" — breaks `validator.py` which does `from .models import (...)` on line 6.
**Why it happens:** models.py looks like v1.0 backend-brain; easy to assume it's obsolete.
**How to avoid:** Read validator.py imports before touching models.py. The CONTEXT.md decision says "remove models.py IF no remaining imports elsewhere" — validator.py IS a remaining import.
**Warning signs:** If `pytest app-store-gui/tests/planning/test_plan_contract.py` passes after deletion, models.py was safe; if it fails with ImportError, it wasn't.

### Pitfall 2: inspection_tools.py Transitive Dependency
**What goes wrong:** Deleting inspection_tools.py before removing the `from webapp.planning.inspection_tools import PlanningInspectionTools` import in main.py (line 41).
**Why it happens:** inspection_tools.py is conceptually v1.0, but main.py still imports it at module load time.
**How to avoid:** Delete inspection_tools.py only AFTER removing the main.py import and PLANNING_INSPECTION_TOOLS usage (lines ~270-290).
**Warning signs:** `ImportError: cannot import name 'PlanningInspectionTools'` from main.py on startup.

### Pitfall 3: __init__.py __all__ List Desync
**What goes wrong:** Removing `from .session_service import ...` from __init__.py but leaving the symbol name in the `__all__` list — Python doesn't error on this but any code doing `from webapp.planning import *` will get a NameError.
**Why it happens:** __init__.py has a large `__all__` list (lines 70-132) that mirrors all the imports. Easy to miss during surgical edit.
**How to avoid:** After removing import lines, also remove the corresponding strings from `__all__`.
**Warning signs:** `AttributeError: module 'webapp.planning' has no attribute '...'`

### Pitfall 4: Planning Route Line Range
**What goes wrong:** Deleting only the `@app.post("/planning/sessions")` route handlers but leaving the helper functions (lines ~270-418).
**Why it happens:** The route handlers (lines 834-1028) and their supporting helpers (lines 270-418) are in two separate blocks. Deleting only routes leaves orphaned functions that still import from deleted modules.
**How to avoid:** The full planning removal scope in main.py is:
  - Lines 25-40: `from webapp.planning import (...)` block
  - Line 41: `from webapp.planning.inspection_tools import PlanningInspectionTools`
  - Line 195: `PLANNING_APPLY_GATEWAY = ApplyGateway(project_root=PROJECT_ROOT)`
  - Line 196: `PLANNING_SESSIONS_DIR = os.path.join(PROJECT_ROOT, ".planning-sessions")`
  - Lines ~270-418: 5 helper functions (build_planning_inspection_snapshot, build_fit_findings_from_inspection, build_default_planning_draft, create_planning_session_service, get_planning_session_service, _planning_session_not_found, _planning_session_conflict, _planning_session_context)
  - Lines 834-1028: 6 route handlers + events endpoint

### Pitfall 5: Harness Description-Matching Over-Engineering
**What goes wrong:** Building a complex NLP-based tool selector when the test only needs to prove that keyword strings from intent phrases appear in tool descriptions.
**Why it happens:** "description-based selection" sounds sophisticated.
**How to avoid:** Keyword matching is correct — the test only needs to confirm that intent-to-tool routing is possible from descriptions alone without hardcoded names. Simple substring or set-intersection match on 3-5 keywords is sufficient.

---

## Code Examples

### Current Tool Description (inspect_cluster — baseline for tuning)
```python
# Source: mcp-server/tools/inspect_cluster.py (current state)
@mcp.tool()
def inspect_cluster() -> dict:
    """Call this tool FIRST when you need to understand what cluster resources are
    available before blueprint selection. Returns a flat snapshot of CPU cores,
    memory, GPU devices, namespaces, and storage classes. Call before
    list_blueprints to know which blueprints can fit the cluster. Call again after
    time passes to refresh — results are not cached.

    Sequencing: inspect_cluster -> list_blueprints -> get_blueprint ->
    validate_yaml -> apply.
    """
```

The descriptions already have sequencing hints from Phases 6-7. The Phase 8 task is to ensure ALL 8 tools have consistent sequencing hints and safety warnings — particularly `apply` (re-inspect warning) and `get_crd_schema` (must appear in the validate sequence).

### Current apply Tool Description (has safety warning — baseline)
```python
# Source: mcp-server/tools/apply_tool.py (current state)
@mcp.tool()
def apply(yaml_text: str, namespace: str, confirmed: bool) -> dict:
    """Apply a WekaAppStore YAML manifest to the cluster.

    IMPORTANT: confirmed must be true (boolean). You must call validate_yaml
    first to verify the YAML is valid, then show the user what will be created
    (resource name, namespace, deployment method), and only set confirmed=true
    after the user explicitly approves the apply.

    Setting confirmed=false returns a structured error — no resources are
    created and no K8s API calls are made.

    Sequencing: validate_yaml -> (user approval) -> apply (confirmed=true).
    After apply, call status to monitor deployment progress.
    """
```

Missing: the re-inspect-before-apply instruction decided in CONTEXT.md. Phase 8 adds: "Before calling apply, re-run inspect_cluster to confirm cluster resources haven't changed since initial inspection."

### Harness Tool Selection (current — hardcoded, must be replaced)
```python
# Source: mcp-server/harness/mock_agent.py (current — hardcoded)
# Step 1: inspect_cluster (use pre-built snapshot, flatten it)
cluster_result = flatten_inspect_cluster_for_mcp(mock_inspection_deps["cluster_snapshot"])
# Step 2: inspect_weka (use pre-built snapshot, flatten it)
weka_result = flatten_inspect_weka_for_mcp(mock_inspection_deps["weka_snapshot"])
```

The new pattern wraps tool selection through a registry keyed on description keywords, so the scenario runner asks "which tool handles cluster inspection?" and the answer comes from matching the description, not from hardcoded function names.

### Dependency Chain: What Survives Cleanup
```python
# mcp-server/tools/apply_tool.py — still needed, unchanged
from webapp.planning.apply_gateway import (
    apply_yaml_content_with_namespace,
    ApplyGatewayDependencies,
)

# apply_gateway.py — no internal planning imports, only stdlib + kubernetes
# validator.py — imports from models.py (models.py must stay)
# models.py — only stdlib (dataclasses, typing) — no planning cross-imports

# Safe to delete (confirmed no downstream imports from preserved modules):
# - session_service.py
# - session_store.py
# - family_matcher.py
# - compiler.py
# - inspection_tools.py (after removing from main.py)
```

### planning/__init__.py After Cleanup (target state)
```python
# Target: app-store-gui/webapp/planning/__init__.py
from .apply_gateway import (
    ApplyGateway,
    ApplyGatewayDependencies,
    apply_yaml_content_with_namespace,
    apply_yaml_documents_with_namespace,
    apply_yaml_file_with_namespace,
)

__all__ = [
    "ApplyGateway",
    "ApplyGatewayDependencies",
    "apply_yaml_content_with_namespace",
    "apply_yaml_documents_with_namespace",
    "apply_yaml_file_with_namespace",
]
```

Note: `validator.py` can be imported directly by tests that need it (`from webapp.planning.validator import validate_structured_plan`) without re-exporting from __init__.py.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1.0 backend-brain: session_service.py handles YAML generation | OpenClaw agent generates YAML; MCP tools provide bounded inspection + apply | Phase 6-7 | planning/ module is now legacy — safe to clean |
| Harness uses hardcoded tool function calls | Harness will use description-based selection | Phase 8 (this phase) | Proves tool descriptions are agent-navigable |
| No SKILL.md | SKILL.md as authoritative agent workflow | Phase 8 (this phase) | Required for AGNT-01 |
| No openclaw.json | openclaw.json generated from server.py | Phase 8 (this phase) | Required for AGNT-03 |

**Deprecated/outdated after this phase:**
- `planning/session_service.py`: Replaces backend reasoning with OpenClaw
- `planning/session_store.py`: No session state needed — OpenClaw manages conversation
- `planning/family_matcher.py`: Family detection is agent reasoning, not backend logic
- `planning/compiler.py`: YAML generation is agent output, not compiled from planning model
- `planning/inspection_tools.py`: Inspection now goes through MCP tools, not this wrapper
- Planning routes in main.py (`/planning/sessions/*`): Entire route family deleted

---

## Open Questions

1. **models.py retention scope**
   - What we know: `validator.py` imports `models.py`; `validator.py` is preserved per CLEAN-03; `models.py` is not in the explicit delete list
   - What's unclear: Does anything in the test suite test validator.py directly after cleanup? (`app-store-gui/tests/planning/test_plan_contract.py` likely does)
   - Recommendation: Preserve models.py silently (don't re-export it from __init__.py, just leave it on disk). Confirm by running `pytest app-store-gui/tests/planning/test_plan_contract.py` after cleanup.

2. **inspection_tools.py deletion path**
   - What we know: Context decision says "remove models.py and inspection_tools.py if they have no remaining imports elsewhere"; `test_weka_inspection.py` imports `PlanningInspectionTools` from it; main.py imports it too
   - What's unclear: Can the test that uses it (`test_weka_inspection.py`) be deleted too? It tests behavior that no longer exists.
   - Recommendation: Delete both `inspection_tools.py` and `test_weka_inspection.py` after verifying test_weka_inspection.py only tests the deleted PlanningInspectionTools class (confirmed from grep — it does).

3. **SKILL.md file location**
   - What we know: Context says "SKILL.md goes at repo root or mcp-server/"
   - What's unclear: OpenClaw registration convention (NemoClaw alpha schema not published)
   - Recommendation: Place SKILL.md at `mcp-server/SKILL.md` co-located with the server code. Reference it by relative path in openclaw.json. Can be moved when NemoClaw schema is published.

4. **app-store-gui tests for deleted planning session code**
   - What we know: `test_planning_routes.py`, `test_planning_session_integration.py`, `test_planning_session_service.py`, `test_planning_session_store.py`, `test_compiler.py` all test deleted functionality
   - What's unclear: Context.md says "full test suite verification after cleanup" — but the test suite for deleted code will necessarily fail after deletion
   - Recommendation: Delete the 5 planning session test files alongside the source files. "Full test suite" means the remaining tests (apply_gateway, cluster_inspection, weka_inspection, inspection_contract, plan_contract) all pass — NOT that deleted-code tests pass.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | none — run from directory with PYTHONPATH set |
| Quick run command (mcp-server) | `cd mcp-server && PYTHONPATH=.:../app-store-gui pytest tests/ -x -q` |
| Quick run command (app-store-gui) | `cd app-store-gui && PYTHONPATH=. pytest tests/ -x -q` |
| Full suite command | `cd mcp-server && PYTHONPATH=.:../app-store-gui pytest tests/ -q && cd ../app-store-gui && PYTHONPATH=. pytest tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AGNT-01 | SKILL.md exists and contains required workflow sections | smoke/manual | `test -f mcp-server/SKILL.md && grep -q "validate_yaml" mcp-server/SKILL.md` | Wave 0 |
| AGNT-01 | Tool descriptions contain sequencing keywords | unit | `pytest mcp-server/tests/test_server.py -x -q` (extend with description checks) | Partial — test_server.py exists but needs new assertions |
| AGNT-02 (existing) | Harness 3 scenarios pass with description-based selection | unit | `pytest mcp-server/tests/test_mock_agent.py -x -q` | Yes — needs harness refactor |
| AGNT-03 | openclaw.json exists and contains all 8 tool names | smoke | `python mcp-server/generate_openclaw_config.py && python -c "import json; d=json.load(open('mcp-server/openclaw.json')); assert len(d['tools'])==8"` | Wave 0 |
| CLEAN-01 | Deleted files are absent from repo | smoke | `git -C . ls-files -- app-store-gui/webapp/planning/session_service.py | wc -l | grep -q '^0$'` (repeat for all 4) | Wave 0 — post-deletion |
| CLEAN-02 | Planning routes are absent from main.py | smoke | `grep -c "planning/sessions" app-store-gui/webapp/main.py | grep -q '^0$'` | Wave 0 — post-deletion |
| CLEAN-03 | apply_gateway.py tests pass after cleanup | unit | `cd app-store-gui && PYTHONPATH=. pytest tests/planning/test_apply_gateway.py -x -q` | Yes |
| CLEAN-03 | MCP server apply tool test passes after cleanup | unit | `cd mcp-server && PYTHONPATH=.:../app-store-gui pytest tests/test_apply_tool.py -x -q` | Yes |

### Sampling Rate
- **Per task commit:** `cd mcp-server && PYTHONPATH=.:../app-store-gui pytest tests/ -x -q`
- **Per wave merge:** full suite (both mcp-server and app-store-gui remaining tests)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `mcp-server/generate_openclaw_config.py` — script to auto-generate openclaw.json from server.py registrations
- [ ] `mcp-server/SKILL.md` — new file (no tests, but harness description-selection serves as a functional proof)
- [ ] `mcp-server/openclaw.json` — generated output artifact

*(Existing test infrastructure covers CLEAN-03 validation — test_apply_gateway.py and test_apply_tool.py already exist)*

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `mcp-server/tools/*.py` — all 8 tool descriptions read verbatim
- Direct code inspection: `mcp-server/harness/mock_agent.py` — current hardcoded tool selection confirmed
- Direct code inspection: `app-store-gui/webapp/main.py` — planning route range confirmed (lines 834-1028), helpers confirmed (lines 270-418), imports confirmed (lines 25-41)
- Direct code inspection: `app-store-gui/webapp/planning/__init__.py` — all 130 exports confirmed; apply_gateway exports identified
- Direct code inspection: `app-store-gui/webapp/planning/validator.py` — confirmed `from .models import (...)` dependency
- Direct code inspection: `app-store-gui/webapp/planning/apply_gateway.py` — confirmed no internal planning module imports
- Direct code inspection: `mcp-server/server.py` — all 8 register_*(mcp) calls confirmed as source of truth for openclaw.json generation
- Direct code inspection: `mcp-server/config.py` — env vars BLUEPRINTS_DIR, KUBERNETES_AUTH_MODE, LOG_LEVEL confirmed

### Secondary (MEDIUM confidence)
- STATE.md: NemoClaw alpha config schema not published as of 2026-03-20 — openclaw.json is best-effort
- CONTEXT.md decisions: tool description strategy, cleanup scope, SKILL.md content requirements

### Tertiary (LOW confidence)
- OpenClaw/NemoClaw JSON registration schema conventions — no official spec available; structure inferred from MCP stdio transport conventions

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies confirmed from requirements.txt; no new packages
- Architecture patterns: HIGH — tool registration pattern, harness structure, and __init__.py exports all confirmed from direct code reading
- Cleanup scope: HIGH — line numbers and file paths confirmed from direct inspection; dependency chain traced
- Pitfalls: HIGH — all grounded in actual code (models.py dependency confirmed on line 6 of validator.py; inspection_tools.py line 41 of main.py)
- openclaw.json structure: LOW — no official NemoClaw schema; structure is best-effort based on MCP conventions

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable codebase; LOW-confidence openclaw.json format may need revision when NemoClaw alpha schema publishes)
