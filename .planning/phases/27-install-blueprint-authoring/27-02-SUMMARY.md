---
phase: 27-install-blueprint-authoring
plan: "02"
subsystem: testing
tags: [pytest, jinja2, yaml, resolve_dependencies, blueprint, WekaAppStore]

# Dependency graph
requires:
  - phase: 27-01
    provides: cluster_init/app-store-install.yaml blueprint with [[ ]] Jinja2 tokens and all appStack components

provides:
  - Cluster-free pytest verifying the install blueprint renders, parses, and satisfies D-01 topo order
  - Quay dockerconfigjson round-trip test using the real operator resolve_dependencies
  - Single-default-StorageClass assertion (INST-08) and stringData-only assertion (INST-09)

affects:
  - Phase 29 (server-side quay builder must close trailing-newline bug class via SC2)
  - Future phases referencing operator_module/tests/test_install_blueprint.py

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Blueprint test via Jinja2 Environment(variable_start_string='[[', variable_end_string=']]') reproducing GUI render path"
    - "resolve_dependencies called directly from test (no reimplementation) via conftest sys.path"
    - "Index-based topo-order assertions (a < b) rather than rigid single sequence"

key-files:
  created:
    - operator_module/tests/test_install_blueprint.py
  modified:
    - cluster_init/app-store-install.yaml

key-decisions:
  - "Use single-quoted YAML string for dockerconfigjson value in blueprint so JSON content (with {}) is valid YAML after Jinja2 substitution"
  - "Test accesses kubernetesManifest string from parsed WekaAppStore CR components, then parses each manifest individually for Secret/StorageClass assertions"
  - "quay_roundtrip test carries inline NOTE comment pointing to Phase 29 SC2 as the authoritative trailing-newline guard"

patterns-established:
  - "Blueprint contract tests: render blueprint via SAMPLE_VARS, parse CR, extract components, call real operator functions"
  - "D-01 edge assertions: index(a) < index(b) for each required ordering constraint"

requirements-completed: [INST-01, INST-02, INST-03, INST-05, INST-08, INST-09]

# Metrics
duration: 4min
completed: "2026-06-24"
---

# Phase 27 Plan 02: Install Blueprint Contract Tests Summary

**Cluster-free pytest verifying the install blueprint renders via [[ ]] Jinja2 path, satisfies D-01 topo order through the operator's real resolve_dependencies, and enforces stringData-only secrets and single-default StorageClass**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-24T03:41:52Z
- **Completed:** 2026-06-24T03:46:08Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Five-test pytest in `operator_module/tests/test_install_blueprint.py` covering all Phase 27 success criteria without a live cluster
- Tests import and call `resolve_dependencies` from `operator_module/main.py` directly (not reimplemented) and reproduce the GUI's exact Jinja2 environment
- D-01 topo order verified via 12 index-of edge assertions across all dependency chains
- Blueprint bug fixed: quay secret manifests now use YAML single-quoted strings so JSON dockerconfigjson content is valid YAML after rendering

## Task Commits

1. **Task 1: Render + topo-sort + secret-encoding + default-StorageClass assertions** - `508cf3e` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `operator_module/tests/test_install_blueprint.py` — Five-test suite: test_render_parses, test_topo_order, test_quay_roundtrip, test_single_default_sc, test_stringdata_only (266 lines)
- `cluster_init/app-store-install.yaml` — Bug fix: changed two `.dockerconfigjson:` YAML values from double-quoted to single-quoted strings

## Decisions Made

- **YAML quoting for dockerconfigjson:** Changed blueprint template from `"[[ quay_dockerconfigjson ]]"` to `'[[ quay_dockerconfigjson ]]'`. JSON content contains `{`, `}`, and `:` which are invalid inside YAML double-quoted scalars without escaping. Single-quoted YAML scalars accept these characters natively (JSON strings have no single quotes so there is no escaping conflict).
- **Inner manifest parse strategy:** The test accesses `kubernetesManifest` as a string from the parsed WekaAppStore CR, then calls `yaml.safe_load_all()` on each component's manifest individually. This mirrors how `kubectl apply` would process the manifests.
- **quay_roundtrip scope annotation:** Added inline NOTE comment in the test stating that this validates a synthetic test-local builder, and that the authoritative quay-builder trailing-newline guard lives in Phase 29 (ROADMAP Phase 29 SC2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid YAML in blueprint quay secret kubernetesManifest**

- **Found during:** Task 1 (test_quay_roundtrip and test_stringdata_only)
- **Issue:** The blueprint template used YAML double-quoted strings `".dockerconfigjson: "[[ quay_dockerconfigjson ]]""`. After Jinja2 renders the JSON value (`{"auths": {"quay.io": {"auth": "..."}}}`) into this field, the inner `{`, `}`, and `:` characters break the YAML double-quoted scalar context. Running `yaml.safe_load_all()` on the kubernetesManifest raised `yaml.parser.ParserError: expected <block end>, but found '<scalar>'`. This means `kubectl apply` would also fail on the rendered manifest in production.
- **Fix:** Changed `.dockerconfigjson: "[[ quay_dockerconfigjson ]]"` to `.dockerconfigjson: '[[ quay_dockerconfigjson ]]'` in both `quay-secret-operator-ns` and `quay-secret-default-ns` components. Single-quoted YAML scalars accept JSON content natively.
- **Files modified:** `cluster_init/app-store-install.yaml`
- **Verification:** All five tests pass after fix; `yaml.safe_load_all()` succeeds on the rendered kubernetesManifest
- **Committed in:** `508cf3e` (same task commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was necessary for correctness — the unpatched blueprint would produce invalid YAML that kubectl apply would reject. No scope creep.

## Issues Encountered

None beyond the auto-fixed blueprint bug above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All Phase 27 success criteria are machine-verified: SC1 (render + topo order), SC2 (quay roundtrip + stringData), SC4 (single default StorageClass)
- Phase 28 (operator helm registry auth) can proceed with confidence that the blueprint contract is proven
- Phase 29 must implement the authoritative server-side quay-builder round-trip guard (ROADMAP Phase 29 SC2) — this test validates only the synthetic test-local builder

## Known Stubs

None — test uses real SAMPLE_VARS and the real `resolve_dependencies` function. No hardcoded empty values or placeholder data flows to any rendering path.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Test file only; threat model T-27-06 (sample credentials) applies — all credentials are obvious throwaway literals (`user:pass`, `admin`/`secret`).

## Self-Check

Files exist:
- `operator_module/tests/test_install_blueprint.py`: FOUND
- `cluster_init/app-store-install.yaml`: FOUND (modified)

Commits exist:
- `508cf3e`: FOUND (test(27-02): add cluster-free blueprint contract tests)

## Self-Check: PASSED

---
*Phase: 27-install-blueprint-authoring*
*Completed: 2026-06-24*
