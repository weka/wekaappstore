---
phase: 28-operator-helm-auth-crd-discovery
plan: 01
subsystem: operator
tags: [helm, oci, quay, auth, crd-discovery, registry-config, caching]
requires:
  - "operator_module/main.py: HelmOperator, discover_chart_crds, should_skip_crds_for_component, handle_appstack_deployment (existing)"
provides:
  - "registry_config_path-threaded OCI quay auth across discover_chart_crds / should_skip_crds_for_component / install_or_upgrade / _install_chart / _upgrade_chart"
  - "module-level _chart_crds_cache dict (success-only memoization, replaces @lru_cache)"
  - "per-OCI-component temp registry-config write + try/finally cleanup in handle_appstack_deployment"
affects:
  - "operator OCI helm install/upgrade path (appStack components)"
tech-stack:
  added: []
  patterns:
    - "success-only manual dict cache (replaces functools.lru_cache to avoid memoizing failures)"
    - "tempfile.NamedTemporaryFile(delete=False) + try/finally os.unlink (mirrors existing values_file pattern)"
    - "credential passed only as --registry-config <tempfile-path>, never as argv element"
key-files:
  created: []
  modified:
    - "operator_module/main.py"
decisions:
  - "D-06: removed @lru_cache from discover_chart_crds; replaced with _chart_crds_cache dict written only on subprocess success"
  - "D-07: cache key is (chart_ref, version, registry_config_path)"
  - "D-01/D-04: --registry-config <tempfile> (not helm registry login); param threaded as Optional[str]=None"
  - "D-05: temp registry-config written only when chart_repo startswith oci:// AND quay_dockerconfigjson in stack_vars"
  - "D-03: handle_helm_deployment left untouched"
metrics:
  duration: "~3m"
  completed: 2026-06-24
  tasks: 2
  files: 1
---

# Phase 28 Plan 01: Operator Helm Auth & CRD Discovery Summary

Threaded an optional `registry_config_path` through the operator's OCI helm path so a fresh-cluster operator authenticates to quay via a per-component `--registry-config <tempfile>`, and replaced `discover_chart_crds`'s `@lru_cache` with a success-only `_chart_crds_cache` dict so an auth/network failure is never memoized as "no CRDs."

## What Was Built

**Task 1 (OPA-02) — `discover_chart_crds` cache fix** (commit d959c3e)
- Removed `@lru_cache(maxsize=128)` from `discover_chart_crds`; the `@lru_cache(maxsize=1)` on `list_existing_crds` is preserved (out of scope per D-06).
- Added module-level `_chart_crds_cache: dict[tuple, set]` keyed by `(chart_ref, version, registry_config_path)` (D-07).
- Cache is written ONLY after `subprocess.check_output` succeeds — including a genuine empty CRD set. `CalledProcessError` and the generic `Exception` branch return `set()` WITHOUT writing the dict, so the next call re-attempts the subprocess.
- Added `registry_config_path: Optional[str] = None` to the signature (consumed in Task 2; defined here so the cache key is correct) and appended `--registry-config <path>` to the `helm show crds` argv when set.

**Task 2 (OPA-01) — registry-config threading + temp-file lifecycle** (commit 0d9829d)
- Added `registry_config_path: Optional[str] = None` to `should_skip_crds_for_component`, `install_or_upgrade`, `_install_chart`, and `_upgrade_chart`.
- `should_skip_crds_for_component` forwards the param to `discover_chart_crds`; `install_or_upgrade` forwards to `_install_chart` / `_upgrade_chart`; both `_install_chart` and `_upgrade_chart` append `--registry-config <path>` to their helm argv when set (mirroring the existing `--values` extend pattern).
- In `handle_appstack_deployment`'s helmChart block: when `chart_repo.startswith("oci://")` AND `"quay_dockerconfigjson" in stack_vars` (D-05), the raw docker auth JSON (D-02) is written to a `tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)`, its `.name` captured as `registry_config_path`, and a DEBUG log records the file path only (never the content). The CRD-strategy evaluation, `install_or_upgrade` call, and readiness handling are wrapped in a `try/finally`; the `finally` unlinks the temp file if present. When the guard is false, `registry_config_path` stays `None` and no `--registry-config` flag appears (backward-compatible).
- `handle_helm_deployment` is untouched (D-03).

## Verification

- `python -m py_compile operator_module/main.py` exits 0.
- Inline `python -c` assertions from both tasks pass: `discover_chart_crds` has no `cache_info` (lru_cache gone), `_chart_crds_cache` is a dict, `registry_config_path` is a parameter of all five functions/methods, `handle_appstack_deployment` source references `registry_config_path` / `quay_dockerconfigjson` / `oci://`, `list_existing_crds` cache retained, and `handle_helm_deployment` contains no `registry_config_path` (D-03).
- `PYTHONPATH=operator_module pytest operator_module/tests/ -q` → 60 passed (existing suite green; default `None` param keeps non-OCI callers unaffected).
- Threat check: credential is written only via `rcf.write(stack_vars["quay_dockerconfigjson"])` to the temp file; no `logging` statement emits the credential value (T-28-01 / T-28-02).

## Deviations from Plan

None - plan executed exactly as written.

## Threat Surface Notes

All five threat-register mitigations (T-28-01 argv non-disclosure, T-28-02 log non-disclosure, T-28-03 temp-file `try/finally` cleanup, T-28-04 per-operation registry-config over persistent login, T-28-05 success-only cache) are implemented as specified. No new security surface introduced beyond the planned threat model. Behavioral test assertions for argv contents, temp-file lifecycle, and cache-on-success-only are deferred to Plan 28-02 per the plan's verification note.

## Self-Check: PASSED
- FOUND: operator_module/main.py (modified)
- FOUND: commit d959c3e (Task 1)
- FOUND: commit 0d9829d (Task 2)
