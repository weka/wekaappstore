---
phase: 28-operator-helm-auth-crd-discovery
plan: 02
subsystem: operator
tags: [tests, helm, oci, quay, auth, crd-discovery, registry-config, caching, security]
requires:
  - "operator_module/main.py: discover_chart_crds, _chart_crds_cache, handle_appstack_deployment, HelmOperator (from 28-01)"
provides:
  - "operator_module/tests/test_operator_helm_auth.py: regression tests for OPA-01 / OPA-02 invariants"
  - "automated evidence for Success Criteria 2 (credential never in argv) and 3 (failure not memoized)"
affects:
  - "operator test suite (now 67 tests, +7)"
tech-stack:
  added: []
  patterns:
    - "patch('main.subprocess.run' / 'main.subprocess.check_output') with argv-capturing side_effect to observe the exact helm command line"
    - "real HelmOperator (NOT mocked) so install_or_upgrade builds the actual argv; only the subprocess layer is mocked"
    - "helm status side_effect returns rc 1 to drive the install path; install rc toggled to exercise the failure cleanup branch"
key-files:
  created:
    - "operator_module/tests/test_operator_helm_auth.py"
  modified: []
decisions:
  - "Mocked subprocess layer (run/check_output) rather than HelmOperator so the credential-secrecy assertions observe the real argv (Task 2 plan instruction)."
  - "Patched main._patch_appstack_progress and main.wait_for_component_ready to keep the appstack handler cluster-free."
  - "Drove the install (not upgrade) path via a helm-status rc-1 side_effect; both paths append --registry-config identically so this is sufficient."
metrics:
  duration: "~6m"
  completed: 2026-06-24
  tasks: 2
  files: 1
---

# Phase 28 Plan 02: Operator Helm Auth & CRD Discovery Tests Summary

Added a cluster-free pytest module (`operator_module/tests/test_operator_helm_auth.py`, 7 tests) that locks the 28-01 behaviors: the success-only CRD cache (no negative memoization), the `--registry-config <path>` argv threading for OCI/quay charts, the guarantee that the quay docker auth never appears as a helm argv element, and the temp registry-config file cleanup on both success and install-failure paths.

## What Was Built

**Task 1 (OPA-02) — success-only CRD cache tests** (commit b358a73)
- `test_failed_helm_show_crds_is_not_cached`: patches `main.subprocess.check_output` with `CalledProcessError`; asserts `discover_chart_crds` returns `set()`, the `(chart_ref, version, None)` key is absent from `main._chart_crds_cache`, and a second call re-invokes the subprocess (`call_count == 2`) — proving no negative memoization (T-28-05).
- `test_successful_helm_show_crds_is_cached`: patches `check_output` to return a one-document CRD yaml; asserts the returned set contains `wekaclients.weka.weka.io`, the cache key is populated, and a second identical call is a cache hit (`call_count == 1`).
- `test_registry_config_path_in_cache_key`: asserts a call with `registry_config_path="/tmp/x.json"` produces a distinct cache entry from `None` (two keys, D-07).

**Task 2 (OPA-01) — registry-config argv / secrecy / temp-file lifecycle tests** (commit c1aaacf)
- `test_registry_config_flag_present_for_oci_quay_chart`: builds an AppStack with one `oci://quay.io/weka.io/helm` `weka-operator` component and a sentinel `quay_dockerconfigjson` in `appStack.variables`; runs the REAL `HelmOperator` against an argv-capturing `subprocess.run`/`check_output`; asserts `--registry-config <path>` appears in the captured helm argv, the file at that path held the raw dockerconfigjson (read during the run), and the `U0VOVElORUw=` sentinel + full dockerconfigjson appear in NO captured argv element (T-28-01 / Success Criterion 2).
- `test_no_registry_config_flag_without_quay_credential`: same spec without the credential; asserts `--registry-config` appears in no captured argv (D-05 backward-compat).
- `test_temp_registry_config_file_removed_after_return`: captures the registry-config path from the argv and asserts `os.path.exists(path)` is False after the handler returns (D-04).
- `test_temp_registry_config_file_removed_on_install_failure`: install side_effect returns rc 1; asserts the temp file is STILL removed (try/finally exception-safe cleanup, T-28-03).

## Verification

- `PYTHONPATH=operator_module pytest operator_module/tests/test_operator_helm_auth.py -v` → 7 passed.
- `PYTHONPATH=operator_module pytest operator_module/tests/ -q` → 67 passed (60 prior + 7 new; no regressions).
- `python -m py_compile operator_module/tests/test_operator_helm_auth.py` exits 0.

## Deviations from Plan

None - plan executed exactly as written. (The plan suggested capturing the temp-file content during the side_effect; the credential-secrecy test reads the registry-config file inside the `check_output` side_effect while it still exists, then asserts the credential is absent from all argv — both options offered by the plan are satisfied.)

## Threat Surface Notes

No new security surface. The tests are observation-only: they mock the subprocess boundary to assert the operator's existing 28-01 mitigations (T-28-01 argv non-disclosure, T-28-03 temp-file cleanup, T-28-05 success-only cache). All three `mitigate`-disposition threats in the plan's register now have automated evidence.

## Self-Check: PASSED
- FOUND: operator_module/tests/test_operator_helm_auth.py
- FOUND: commit b358a73 (Task 1)
- FOUND: commit c1aaacf (Task 2)
