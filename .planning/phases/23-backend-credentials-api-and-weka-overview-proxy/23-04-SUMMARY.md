---
phase: 23
plan: "04"
subsystem: app-store-gui/tests
tags: [testing, credentials-api, weka-overview, regression, security]
dependency_graph:
  requires: ["23-02", "23-03"]
  provides: ["test-coverage-credentials", "test-coverage-weka-overview"]
  affects: ["app-store-gui/tests/test_credentials_api.py"]
tech_stack:
  added: []
  patterns: ["monkeypatch.setattr injection", "asyncio.run direct coroutine invocation", "class stub pattern", "SimpleNamespace stubs"]
key_files:
  created:
    - app-store-gui/tests/test_credentials_api.py
  modified: []
decisions:
  - "Wrote both Task 1 and Task 2 tests in a single file creation pass (23 tests total) rather than create-then-append, since the full spec was known upfront. All 23 tests were committed in one atomic commit."
  - "Used asyncio.run(handler(kwargs...)) for direct coroutine invocation per D-15 (no FastAPI TestClient)"
  - "For login-failure test used generator throw pattern to raise RuntimeError from lambda"
  - "test_weka_overview_bust_query_bypasses_cache monkeypatches datetime.datetime via subclass to control fetchedAt without real sleep"
metrics:
  duration: "~8 minutes"
  completed_date: "2026-06-11"
  tasks_completed: 2
  files_created: 1
  tests_written: 23
  test_runtime: "0.62s"
---

# Phase 23 Plan 04: Credentials API Test Suite Summary

Created the behavioral regression test file `app-store-gui/tests/test_credentials_api.py` with 23 tests covering all acceptance criteria from Plans 02 and 03.

## Final Test List

### Helper Unit Tests (no I/O)
1. `test_make_credential_slug_normalizes_and_truncates`
2. `test_build_credential_response_item_omits_secret_fields`
3. `test_build_credential_response_item_weka_storage_exposes_endpoint_only_from_status`

### GET /api/credentials Handler Tests
4. `test_list_credentials_returns_shape_without_secret_values`
5. `test_list_credentials_type_filter_returns_only_ready`

### POST /api/credentials Handler Tests
6. `test_post_credential_nvidia_creates_secret_and_cr`
7. `test_post_credential_slug_collision_appends_suffix`
8. `test_post_credential_weka_storage_persists_three_keys`
9. `test_post_credential_invalid_type_returns_400`
10. `test_post_credential_weka_missing_username_returns_400`

### DELETE /api/credentials Handler Tests
11. `test_delete_credential_deletes_cr_then_raw_secret_preserves_derived`
12. `test_delete_credential_idempotent_on_secret_404`
13. `test_delete_credential_invalid_name_returns_400_without_io`

### _assemble_weka_overview Pure Function Tests
14. `test_assemble_weka_overview_pure_transform_shape`
15. `test_assemble_weka_overview_tolerates_alt_field_names`
16. `test_assemble_weka_overview_capacity_fallback_when_no_cluster_capacity_dict`

### WEKA Overview Cache / Security Tests
17. `test_weka_overview_cache_hit_avoids_refetch`
18. `test_weka_overview_bust_query_bypasses_cache`
19. `test_weka_overview_namespace_scoped_cache`
20. `test_weka_overview_invalid_credential_name_returns_400_without_io`
21. `test_weka_overview_login_failure_returns_502_without_leak`

### _resolve_weka_credential_secret Tests
22. `test_resolve_weka_credential_secret_decodes_base64_secret`
23. `test_resolve_weka_credential_secret_missing_key_raises_runtime`

## Deviations from Plan

### Implementation Approach
- **Deviation:** Wrote both Task 1 (13 tests) and Task 2 (10 tests) in a single file creation, then committed the complete file in one commit rather than create-then-append in two phases.
- **Reason:** The full specification was known before writing; creating the file atomically guaranteed consistency between Task 1 and Task 2 stubs (shared factory functions, shared `_patch_weka_overview` helper).
- **Impact:** Both Task 1 and Task 2 acceptance criteria met in the single commit `f8e96e7`. Task 2 acceptance criteria verified by second test run.

### Extra Tests Beyond Spec Minimum
- 23 tests delivered vs. minimum of 20 (spec said "~20 tests"). The three extra tests are `test_resolve_weka_credential_secret_decodes_base64_secret` and `test_resolve_weka_credential_secret_missing_key_raises_runtime` (2 from spec D-16) plus `test_assemble_weka_overview_tolerates_alt_field_names` (spec listed as required). Count matches spec exactly.

## Pytest Invocation and Runtime

```bash
PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints \
  pytest app-store-gui/tests/test_credentials_api.py -v
```

**Runtime:** 0.62 seconds (23 tests, all passed, 0 failures)

## Known Stubs

None — all production behavior covered. The test file stubs K8s clients and WEKA HTTP helpers (correct: these are external I/O dependencies, not implementation stubs).

## Threat Flags

No new network endpoints, auth paths, or schema changes introduced by this test file.

## Self-Check: PASSED

- `app-store-gui/tests/test_credentials_api.py` exists: FOUND
- Commit `f8e96e7` exists: FOUND
- All 23 tests pass: CONFIRMED (0.62s runtime)
- No FastAPI TestClient usage: CONFIRMED
- `monkeypatch.setattr(main` count ≥ 3: CONFIRMED (29 occurrences)
- `grep -c '^def test_'` returns 23: CONFIRMED (≥ 20 required)
