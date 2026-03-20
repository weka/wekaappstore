---
phase: 01-plan-contract-and-yaml-translation
plan: "02"
subsystem: planning-contract
tags: [python, validation, dataclasses, wekaappstore, planning]
requires:
  - phase: 01-01
    provides: pytest planning fixtures and seeded contract tests
provides:
  - typed structured-plan contract models for Phase 1
  - layered validator for deterministic plan acceptance and rejection
  - expanded contract-fixture coverage for supported and unsupported plan variants
affects: [phase-01-04, planning-validator, canonical-yaml]
tech-stack:
  added: []
  patterns: [typed plan models, layered validation, explicit blocker versus warning separation]
key-files:
  created:
    - app-store-gui/webapp/planning/models.py
    - app-store-gui/webapp/planning/validator.py
  modified:
    - app-store-gui/webapp/planning/__init__.py
    - app-store-gui/tests/conftest.py
    - app-store-gui/tests/planning/test_plan_contract.py
key-decisions:
  - "Keep the plan contract narrower than raw YAML and model only the fields needed to compile one canonical WekaAppStore resource."
  - "Treat unsupported top-level fields, unsupported blueprint families, invalid readiness types, and invalid values file kinds as deterministic contract failures."
patterns-established:
  - "Validator output separates blocking PlanValidationError objects from NormalizationWarning objects."
  - "Phase 1 fixtures enumerate explicit invalid-plan variants for regression-style contract coverage."
requirements-completed: [PLAN-02, PLAN-03, PLAN-06, PLAN-07]
duration: 8min
completed: 2026-03-20
---

# Phase 1 Plan 02: Contract And Validator Summary

**Typed structured-plan models and layered validator that make the backend authoritative for Phase 1 plan shape and repo/operator rule enforcement**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-20T01:18:20Z
- **Completed:** 2026-03-20T01:26:40Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added typed Phase 1 planning models for blueprint family, namespace strategy, components, readiness checks, validation issues, and structured plans.
- Implemented deterministic structured-plan validation with explicit blocking errors for unsupported families, malformed components, dependency mistakes, invalid deployment-method combinations, and invalid unresolved-question states.
- Expanded shared fixtures and contract tests so the Phase 1 suite covers rich valid plans, warning-producing defaults, unsupported top-level fields, unsupported readiness types, invalid values file kinds, and unsupported CRD strategies.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add typed planning contract models and validation result types** - `99a2675` (feat)
2. **Task 2: Implement layered validator entry points** - `afad2c1` (feat)
3. **Task 3: Establish pytest contract tests for plan validation** - `to-be-recorded in current branch after fixture expansion commit`

## Files Created/Modified

- `app-store-gui/webapp/planning/models.py` - dataclass-based structured plan contract and validation issue types.
- `app-store-gui/webapp/planning/validator.py` - layered validation logic enforcing Phase 1 contract rules and safe normalization warnings.
- `app-store-gui/webapp/planning/__init__.py` - planning package exports for models and validator entry points.
- `app-store-gui/tests/conftest.py` - shared valid, invalid, and warning fixture payloads covering the richer validation matrix.
- `app-store-gui/tests/planning/test_plan_contract.py` - deterministic contract tests for acceptance, warnings, and rejections.

## Decisions Made

- Used dataclasses instead of ad hoc dict helpers for the initial contract surface to keep the models lightweight and repo-native.
- Kept the validator Phase 1-scoped: no chat/session fields, no inspection tooling assumptions, and no generic YAML-AST behavior.

## Deviations from Plan

None - plan stayed within the intended contract and validation boundary.

## Issues Encountered

- The original fixture set under-covered some unsupported contract cases, so the shared fixture module was expanded before finalizing the plan.

## User Setup Required

None.

## Next Phase Readiness

- The compiler plan can now consume typed validated plans instead of raw dict payloads.
- Backend integration work can rely on deterministic validator behavior and a broader invalid-plan regression set.

## Self-Check

PASSED

- Verified `app-store-gui/webapp/planning/models.py` and `app-store-gui/webapp/planning/validator.py` exist.
- Verified `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py -q` passes.

---
*Phase: 01-plan-contract-and-yaml-translation*
*Completed: 2026-03-20*
