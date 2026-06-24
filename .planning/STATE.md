---
gsd_state_version: 1.0
milestone: v8.0
milestone_name: Guided Install Wizard — WEKA Operator, CSI & Storage Classes
status: planning
last_updated: "2026-06-24T02:24:31.392Z"
last_activity: 2026-06-24
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-17 — milestone v7.0 archived)

**Core value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.
**Current focus:** Planning next milestone

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-06-24 — Milestone v8.0 started

## Accumulated Context

### Key Architectural Decisions

- **Pre-scan guard is non-negotiable:** `render()` must return text unchanged when no `${...}` pattern is present (regex check). Without the pre-scan guard, `cluster_init/app-store-cluster-init.yaml` shell scripts raise `KeyError` on first reconcile after upgrade.
- **Single-pass only — no recursive resolution:** Variable values are taken literally. AIDP migration must use fully-resolved values. README must NOT show the cross-referencing pattern.
- **WarpCredential multi-instance:** One CR per named credential (not per type). Multiple NGC keys, multiple WEKA clusters per namespace are all supported.
- **Derived secrets not deleted on WarpCredential delete:** Admin must clean up `warp-<name>-*` secrets manually; avoids accidental removal of secrets in use by running workloads.
- **`[[var]]` Jinja2 delimiters for dynamic blueprint substitution:** Avoids conflicts with shell `$VAR` expansion in YAML and operator `render()` function's `${VAR}` syntax.
- **`parse_x_variables` returns `{}` on all failure paths:** Blueprint pages degrade to static install form rather than 500 on malformed `x-variables` block.
- **Phase 20 is a separate repo** — all deliverables in `/Users/christopherjenkins/git/aidp`, not `wekaappstore`.

### Open Blockers / Tracked Work

- **v3.1 deferred** — E2E chat validation (E2E-01..04) plus four prerequisite fixes (inspect-tool `load_incluster_config`, init-container openclaw.json config gap, NIM model reliability, OpenClaw version upgrade). Full root-cause and fix plan in `.planning/v3.0-KNOWN-ISSUES.md`.
- **v5.0 Phases 19-20 unstarted** — Validator Soft-Warning and AIDP Migration Smoke Test. Phase 19 has no code dependency; Phase 20 requires cluster with Phases 16-18 deployed.
- **DYN-07 external repo** — Production blueprints (oss-rag, openfold, nvidia) in external `warp-blueprints` repo need `x-variables` migration; tracked as follow-on.

### Pending Todos

None.

## Session Continuity

Last session: 2026-06-17
Stopped at: v7.0 milestone archived
Resume: `/gsd-new-milestone` to start next milestone
