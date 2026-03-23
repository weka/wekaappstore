# Roadmap: OpenClaw MCP Tools For WEKA App Store

**Mode:** YOLO
**Granularity:** Standard

## Milestones

- ✅ **v1.0 NemoClaw Agent Planning** - Phases 1-5 (completed 2026-03-20)
- 🚧 **v2.0 OpenClaw MCP Tool Integration** - Phases 6-10 (in progress)

## Phases

<details>
<summary>✅ v1.0 NemoClaw Agent Planning (Phases 1-5) - COMPLETED 2026-03-20</summary>

### Phase 1: Plan Contract And YAML Translation
**Goal:** Establish the deterministic structured-plan contract, YAML translation, and safe existing-runtime handoff before NemoClaw-generated plans can submit anything.
**Requirements**: PLAN-02, PLAN-03, PLAN-06, PLAN-07, PLAN-08, APPLY-06, APPLY-07
**Success criteria:**
1. Backend accepts a structured planning payload and validates it without any model dependency.
2. Backend renders canonical `WekaAppStore` YAML from validated plan data.
3. Validated plans still hand off to the existing apply and operator execution path.
4. Invalid plans are rejected with deterministic errors tied to repo and operator contract rules.

Plans:
- [x] 01-01: Pytest harness and planning/apply gateway fixtures
- [x] 01-02: Typed structured-plan contract and layered validation rules
- [x] 01-03: Shared YAML apply gateway and WekaAppStore runtime-path compatibility
- [x] 01-04: Plan compilation to canonical WekaAppStore YAML wired through shared gateway

### Phase 2: Cluster And WEKA Inspection Signals
**Goal:** Build the bounded inspection layer and safety signals needed for trustworthy cluster-fit and storage-fit decisions.
**Requirements**: CLSTR-01, CLSTR-02, CLSTR-03, CLSTR-04, CLSTR-05, PLAN-04, SAFE-01, SAFE-02, SAFE-03, SAFE-04
**Success criteria:**
1. Backend can produce bounded inspection snapshots for namespaces, storage classes, GPU, CPU, RAM, and WEKA state.
2. Fit signals include freshness and confidence metadata and fail closed when required data is incomplete.
3. NemoClaw-facing tool calls are bounded, auditable, and read-only.
4. The system can classify failures by stage: inspection, validation, YAML generation, or apply handoff.

Plans:
- [x] 02-01: Typed inspection-domain contract and fail-closed validation rules
- [x] 02-02: Bounded Kubernetes inspection service
- [x] 02-03: Read-only WEKA inspection seam and planning tool wrapper
- [x] 02-04: Merged cluster and WEKA inspection into planner-facing fit findings

### Phase 3: Conversational Planning Sessions
**Goal:** Deliver the chat-first planning workflow with follow-up turns, supported-family matching, and safe session management.
**Requirements**: CHAT-01, CHAT-02, CHAT-03, CHAT-04, CHAT-05, PLAN-01
**Success criteria:**
1. Users can start a planning session from the UI and submit free-text requests.
2. The system can match a supported blueprint family or explicitly say none fit.
3. Users can answer follow-ups, review session history, and restart or abandon drafts safely.

Plans:
- [x] 03-01: Typed planning-session contract and replayable local persistence
- [x] 03-02: Supported-family matching and backend session service
- [x] 03-03: Planning-session routes and server-rendered chat workspace
- [x] 03-04: Replay lifecycle guards and deterministic end-to-end session coverage

### Phase 4: Review, Approval, And Apply Gating
**Goal:** Expose fit, validation, and YAML review to users and keep all applies behind an explicit approval gate.
**Requirements**: PLAN-05, APPLY-01, APPLY-02, APPLY-03, APPLY-04, APPLY-05
**Success criteria:**
1. Users can review the plan, YAML, validation results, and fit rationale before apply.
2. Multi-blueprint oversubscription is detected and blocks apply.
3. No apply occurs without explicit approval.

### Phase 5: Maintainer Draft Authoring And Test Hardening
**Goal:** Add maintainer draft blueprint generation and the mocked test coverage required for safer rollout.
**Requirements**: AUTHR-01, AUTHR-02, AUTHR-03, SAFE-05
**Success criteria:**
1. Maintainers can request draft blueprints without applying them.
2. Draft output is reviewable separately and follows repo conventions.
3. The workflow is testable with mocked NemoClaw, Kubernetes, and WEKA inputs.

</details>

### v2.0 OpenClaw MCP Tool Integration

**Milestone Goal:** Build an MCP server exposing WEKA App Store capabilities as tools OpenClaw can call, deliver all 8 tools with a mock harness for testing without a live agent, define the agent workflow in SKILL.md, and remove deprecated v1.0 backend-brain code.

- [x] **Phase 6: MCP Scaffold and Read-Only Tools** - Runnable MCP server with 5 read-only tools and flat agent-facing response schemas (completed 2026-03-20)
- [ ] **Phase 7: Validation, Apply, and Status Tools** - Remaining 3 tools with approval gate, correct validator contract, and full mock harness
- [x] **Phase 8: SKILL.md, Agent Context, and Cleanup** - Agent workflow definition, tool description tuning, deprecated code deleted (completed 2026-03-20)
- [x] **Phase 9: Deployment and Registration** - Container image, OpenClaw/NemoClaw registration config, deployment documentation (completed 2026-03-22)

## Phase Details

### Phase 6: MCP Scaffold and Read-Only Tools
**Goal**: A runnable MCP server exposes 5 read-only tools with flat, agent-facing response schemas that set the output contract for all subsequent tools
**Depends on**: Phase 5 (reuses `inspection/cluster.py`, `planning/apply_gateway.py`, `planning/validator.py`)
**Requirements**: MCPS-01, MCPS-02, MCPS-03, MCPS-04, MCPS-05, MCPS-06, MCPS-10, MCPS-11
**Success Criteria** (what must be TRUE):
  1. `mcp dev server.py` starts and lists all 5 read-only tools (`inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`) with correct MCP schemas
  2. Each tool returns a flat JSON response with all values reachable in 2 key traversals or fewer and a top-level `captured_at` timestamp
  3. Calling any inspection tool with mocked K8s/WEKA backends produces agent-readable output with no nested domain model structures
  4. All tool descriptions start with when and why to call the tool and include sequencing guidance
  5. All server logging goes to stderr; stdout carries only MCP protocol frames
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — MCP server scaffold, test harness, and inspection tools (inspect_cluster + inspect_weka)
- [x] 06-02-PLAN.md — Blueprint catalog tools (list_blueprints + get_blueprint)
- [x] 06-03-PLAN.md — CRD schema tool (get_crd_schema) and cross-tool integration validation

### Phase 7: Validation, Apply, and Status Tools
**Goal**: The remaining 3 tools are callable, the apply approval gate is enforced in tool code (not just SKILL.md), the validator works against the WekaAppStore CRD contract, and the mock harness exercises the complete inspect-validate-apply chain
**Depends on**: Phase 6
**Requirements**: MCPS-07, MCPS-08, MCPS-09, AGNT-02
**Success Criteria** (what must be TRUE):
  1. Calling `apply` without `confirmed: true` returns a structured error and no CR is created
  2. `validate_yaml` accepts valid WekaAppStore YAML and rejects YAML containing v1.0-only fields (`blueprint_family`, `fit_findings`)
  3. `status` returns current deployment state of a named WekaAppStore CR
  4. The mock harness runs a scripted inspect-validate-apply loop against mocked backends without errors, including approval bypass and validation failure paths
**Plans**: 2 plans

Plans:
- [ ] 07-01-PLAN.md — validate_yaml and apply tools with unit tests
- [ ] 07-02-PLAN.md — status tool, server wiring, mock agent harness, and integration tests

### Phase 8: SKILL.md, Agent Context, and Cleanup
**Goal**: SKILL.md authoritatively defines the agent workflow, tool descriptions are tuned based on harness evidence, the OpenClaw registration config is generated, and all deprecated v1.0 backend-brain files are deleted from the repo
**Depends on**: Phase 7
**Requirements**: AGNT-01, AGNT-03, CLEAN-01, CLEAN-02, CLEAN-03
**Success Criteria** (what must be TRUE):
  1. SKILL.md contains the full blueprint planning workflow including a validate-retry loop, re-inspect-before-apply instruction, and negative YAML examples
  2. The mock harness agent simulation selects the correct tool from tool descriptions alone without hardcoded tool selection
  3. `session_service.py`, `session_store.py`, `family_matcher.py`, and `compiler.py` are absent from the repo (git-removed, not just commented out)
  4. Planning session routes return 410 Gone or are fully removed from `main.py`
  5. `inspection/cluster.py`, `planning/apply_gateway.py`, and `planning/validator.py` remain intact and their tests pass
**Plans**: 3 plans

Plans:
- [ ] 08-01-PLAN.md — SKILL.md, tool description tuning, and description-based harness upgrade
- [ ] 08-02-PLAN.md — OpenClaw registration config (openclaw.json) generation
- [ ] 08-03-PLAN.md — Deprecated v1.0 backend-brain code cleanup

### Phase 9: Deployment and Registration
**Goal**: The MCP server ships as a container image and OpenClaw/NemoClaw operators can register and invoke it using documented, repeatable configuration steps
**Depends on**: Phase 8
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04
**Success Criteria** (what must be TRUE):
  1. `docker build` produces an image that starts the MCP server on stdio with no missing dependencies
  2. All K8s endpoints, WEKA endpoints, and credentials are configurable via environment variables with no hardcoded values in the image
  3. An `openclaw.json` registration snippet is provided that, when placed in the OpenClaw config, causes `tools/list` to return all 8 tools
  4. Documentation describes every step required to register the MCP server with both OpenClaw and NemoClaw
**Plans**: 2 plans

Plans:
- [ ] 09-01-PLAN.md — Dockerfile, .dockerignore, and startup env var validation
- [ ] 09-02-PLAN.md — GitHub Actions CI/CD workflow and README registration documentation

### Phase 10: Integration Bug Fixes
**Goal**: Fix 3 integration defects found by milestone audit — blueprints.py logger crash, LOG_LEVEL env var not wired, PYTHONPATH missing from openclaw.json startup
**Depends on**: Phase 9
**Requirements**: MCPS-04, MCPS-05, MCPS-10, MCPS-11, DEPLOY-03, AGNT-03, DEPLOY-04
**Gap Closure**: Closes gaps from v2.0 audit
**Success Criteria** (what must be TRUE):
  1. `scan_blueprints()` skips malformed YAML files with a warning instead of crashing
  2. Setting `LOG_LEVEL=DEBUG` at runtime changes the MCP server's logging verbosity
  3. `openclaw.json` startup block includes PYTHONPATH so non-container registration works
  4. All 100+ existing tests still pass
**Plans**: 1 plan

Plans:
- [ ] 10-01-PLAN.md — Fix blueprints.py logger crash, wire LOG_LEVEL, add PYTHONPATH to openclaw.json

## Progress

**Execution Order:**
v2.0 phases execute in numeric order: 6 → 7 → 8 → 9 → 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Plan Contract And YAML Translation | v1.0 | 4/4 | Complete | 2026-03-20 |
| 2. Cluster And WEKA Inspection Signals | v1.0 | 4/4 | Complete | 2026-03-20 |
| 3. Conversational Planning Sessions | v1.0 | 4/4 | Complete | 2026-03-20 |
| 4. Review, Approval, And Apply Gating | v1.0 | -/- | Complete | 2026-03-20 |
| 5. Maintainer Draft Authoring And Test Hardening | v1.0 | -/- | Complete | 2026-03-20 |
| 6. MCP Scaffold and Read-Only Tools | 3/3 | Complete   | 2026-03-20 | - |
| 7. Validation, Apply, and Status Tools | 1/2 | In Progress|  | - |
| 8. SKILL.md, Agent Context, and Cleanup | 3/3 | Complete   | 2026-03-20 | - |
| 9. Deployment and Registration | 2/2 | Complete   | 2026-03-22 | - |
| 10. Integration Bug Fixes | 1/1 | Complete    | 2026-03-22 | - |

---
*Roadmap created: 2026-03-20 (v1.0)*
*v2.0 phases added: 2026-03-20*
*All v2.0 requirements covered: yes (21/21)*
