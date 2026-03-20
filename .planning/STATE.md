---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: OpenClaw MCP Tool Integration
status: planning
stopped_at: Completed 07-02-PLAN.md
last_updated: "2026-03-20T06:55:35.536Z"
last_activity: 2026-03-20 — v2.0 roadmap created; Phases 6-9 defined
progress:
  total_phases: 9
  completed_phases: 5
  total_plans: 21
  completed_plans: 17
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-20)

**Core value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.
**Current focus:** Phase 6 — MCP Scaffold and Read-Only Tools

## Current Position

Phase: 6 of 9 (MCP Scaffold and Read-Only Tools)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-20 — v2.0 roadmap created; Phases 6-9 defined

Progress: [░░░░░░░░░░] 0% (v2.0)

## Performance Metrics

**Velocity:**
- Total plans completed: 0 (v2.0)
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

*Updated after each plan completion*
| Phase 06-mcp-scaffold-and-read-only-tools P06-01 | 4min | 2 tasks | 13 files |
| Phase 06-mcp-scaffold-and-read-only-tools P06-02 | 8min | 2 tasks | 5 files |
| Phase 06-mcp-scaffold-and-read-only-tools P06-03 | 3min | 2 tasks | 5 files |
| Phase 07-validation-apply-and-status-tools P01 | 2min | 2 tasks | 4 files |
| Phase 07-validation-apply-and-status-tools PP02 | 4min | 2 tasks | 8 files |

## Accumulated Context

### Decisions

- [v2.0]: Pivot from backend-brain to OpenClaw-native MCP tool registration
- [v2.0]: MCP server in Python reusing `inspection/cluster.py`, `planning/apply_gateway.py`, `planning/validator.py`
- [v2.0]: Develop with mock agent harness since NemoClaw not yet available in environment
- [v2.0]: Remove deprecated v1.0 backend-brain code in Phase 8 (not earlier — needs stable tools first)
- [v2.0]: Output schemas defined in Phase 6 before any wrapper written — prevents retroactive flattening
- [Phase 06]: mcp-server/ placed at repo root for clean container separation; imports app-store-gui via PYTHONPATH
- [Phase 06]: Separate flatten_inspect_*_for_mcp() functions never expose inspection_snapshot or domain wrappers to agents
- [Phase 06]: register_*(mcp) pattern for tool registration enables isolated test instances
- [Phase Phase 06-02]: scan_blueprints() returns internal wrapper dicts — flatten functions shape agent-facing JSON; helm_chart sub-dict hoisted to flat component fields for 2-key depth contract
- [Phase Phase 06-03]: _get_crd_schema_impl() injectable pattern matches inspect tools — same testability approach
- [Phase Phase 06-03]: 'schema' field is documented exception to 2-key depth rule — pass-through K8s CRD OpenAPI data, not our domain model
- [Phase Phase 06-03]: check_depth() shared helper in test_response_depth.py enforces depth contract globally across all tools
- [Phase Phase 07-01]: validate_yaml only rejects known v1.0 snake_case fields (blueprint_family, fit_findings, etc.) — camelCase CRD fields pass through to avoid false positives
- [Phase Phase 07-01]: confirmed is not True (identity check) in apply tool — prevents string 'true' or int 1 from bypassing confirmation gate
- [Phase Phase 07-02]: Harness calls flatten_* functions directly with pre-built snapshot dicts (avoids mocking K8s collection stack)
- [Phase Phase 07-02]: ops_log pattern: mock side-effecting methods append (op_type, kwargs) tuples to shared list for assertion

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 6]: Blueprint catalog source (`list_blueprints`/`get_blueprint`) is TBD — confirm file path and schema from `weka-app-store-operator-chart/` before Phase 6 planning completes
- [Phase 6]: Verify `apply_gateway.py` K8s client initialization is not import-time (would fail in CI) before writing apply tool wrapper
- [Phase 9]: NemoClaw alpha config schema not yet published as of 2026-03-20 — may require SKILL.md format revision; monitor release notes

## Session Continuity

Last session: 2026-03-20T06:52:36.726Z
Stopped at: Completed 07-02-PLAN.md
Resume file: None
