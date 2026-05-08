---
phase: 18-operator-wiring-and-docs
plan: 05
subsystem: testing
tags: [pytest, snapshot, backward-compat, regression-lock]

requires:
  - phase: 16-render-helper-and-test-scaffolding
    provides: render() pre-scan guard (the source-level invariant this snapshot tests)
  - phase: 18-operator-wiring-and-docs
    provides: Plan 18-01 wired _render_or_raise + handle_appstack_deployment variables build
provides:
  - Byte-identical merged Helm values lock for the inline-values + helm-install path
  - Reviewable JSON baseline files at operator_module/tests/snapshots/ai-research/
  - BASELINE_REGEN=1 regeneration UX
affects: [phase-19-validator, phase-20-aidp-migration]

tech-stack:
  added: []
  patterns:
    - "Hand-rolled Path.read_text() snapshot pattern (matches mcp-server/tests/test_openclaw_config.py — no third-party snapshot library)"
    - "JSON serialization with sort_keys=True + indent=2 for deterministic, PR-reviewable baselines"
    - "snake_case -> camelCase fixture normalization (in-test, ~5 lines)"

key-files:
  created:
    - operator_module/tests/test_backward_compat_snapshot.py
    - operator_module/tests/snapshots/ai-research/values_qdrant.json
    - operator_module/tests/snapshots/ai-research/values_research-api.json
  modified: []

key-decisions:
  - "Parametrize list uses release_name values ['qdrant', 'research-api'] (the helm release names from helm_chart.release_name in the fixture) NOT component name values ['vector-db', 'research-api'] — because the mock helper keys captured Helm calls by kwargs.get('name') which is populated from release_name at main.py:829 (W-2 fix from plan-checker revision)"
  - "Coverage scope is intentionally narrow: inline-values + helm-install path only. The valuesFiles render path is locked by Plan 18-03 tests 5+6; kubernetesManifest no-op path is locked indirectly by Plan 18-03 tests 1+3 (W-3/W-4 disclosure from plan-checker revision)"
  - "Baselines committed as plain JSON (not YAML) for sort_keys determinism and easier diff review"

patterns-established:
  - "BASELINE_REGEN=1 env var gate for regeneration (recommended over a separate regen test or CLI tool — simpler, discoverable in failure messages)"

requirements-completed: [TST-03, OP-06]

duration: ~10min
completed: 2026-05-08
---

# Phase 18 / Plan 05: test_backward_compat_snapshot.py Summary

**Byte-identical Helm values dict snapshot test (~210 lines, 0.78s runtime) plus committed baselines locking the inline-values + helm-install path against future regression.**

## Performance

- **Duration:** ~10 min (orchestrator-direct after worktree-isolated executor was blocked by Bash-tool denials)
- **Started:** 2026-05-08
- **Completed:** 2026-05-08
- **Tasks:** 1/1
- **Files modified:** 3 created
  - `operator_module/tests/test_backward_compat_snapshot.py` (~210 lines, 2 parametrized cases)
  - `operator_module/tests/snapshots/ai-research/values_qdrant.json` (24 bytes — `{"replicaCount": 1}`)
  - `operator_module/tests/snapshots/ai-research/values_research-api.json` (41 bytes — `{"image": {"tag": "latest"}}`)
- **Test runtime:** 0.78s for 2 parametrized cases
- **Regen runtime (`BASELINE_REGEN=1`):** identical, plus baseline files written

## Accomplishments

- 2 parametrized cases (one per release_name) capture and assert byte-identical merged Helm values dicts via `mock_helm.install_or_upgrade.call_args_list`
- `_normalize_camel(node)` recursive helper renames the 8 snake_case fields to camelCase (`helm_chart` → `helmChart`, etc.) so the fixture flows through `handle_appstack_deployment` unchanged
- `BASELINE_REGEN=1` env var gates regeneration; default behavior asserts equality with actionable failure messages including the regen command verbatim
- Both baselines pass `pytest -x` on second run (without env var) — confirms determinism

## Task Commits

1. **Task 1 — test_backward_compat_snapshot.py + baselines** — `09a4f1c` (test)

## Coverage scope (W-3/W-4 disclosure)

This snapshot intentionally has a narrow scope:

| Path | Status | Locked by |
|---|---|---|
| Inline `values:` + helm-install | **Locked here** | This file |
| `valuesFiles:` render | Locked elsewhere | Plan 18-03 Tests 5+6 (`test_configmap_valuesfile_substitutes_variables`, `test_secret_valuesfile_substitutes_variables`) |
| `kubernetesManifest:` no-op | Locked elsewhere | Plan 18-03 Tests 1+3 (rendered string equals source string when no `${...}` tokens are present — pre-scan guard short-circuits) |

The fixture (`ai-research.yaml`) has only `helm_chart` components with inline `values:` — no `valuesFiles`, no `kubernetesManifest`. Extending the fixture to cover those paths was deferred to keep TST-03 churn-minimal; the hard contract from REQUIREMENTS.md "Out of Scope" ("any CR without `variables:` must produce byte-identical Helm values as before") is still enforced via the combination of:

- Plan 18-03 tests (substitution behavior)
- Plan 18-05 snapshot (this file — inline-values byte-identity)
- Phase 16's locked `render()` pre-scan guard at the source level

## Decisions Made

None — followed the revised plan as specified, including the W-2 release_name parametrize fix and the W-3/W-4 coverage disclosure.

## Deviations from Plan

None — plan executed exactly as written after the iteration-1 plan-checker revisions.

## Issues Encountered

- **Worktree executor agent blocked by Bash-tool denials.** Same pattern as 18-01, 18-03, 18-04 — orchestrator-direct on main.

## User Setup Required

None — pure unit tests; no cluster, kubectl, or helm binary required.

## Next Phase Readiness

All 5 plans of Phase 18 are now landed on `main`. Ready for goal-backward verification:

- Run `pytest operator_module/tests/ -x` to confirm full suite green (Phase 16 + Phase 18)
- Spawn `gsd-verifier` agent for goal-backward verification against the 5 ROADMAP success criteria
- If verification passes, mark Phase 18 complete in STATE.md and ROADMAP.md

---
*Phase: 18-operator-wiring-and-docs*
*Completed: 2026-05-08*
