# Phase 28: Operator Helm Auth & CRD Discovery - Context

**Gathered:** 2026-06-24 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Modify `operator_module/main.py` so the operator pod can authenticate to quay before any `helm show crds` or `helm install/upgrade` of an OCI chart, and fix `discover_chart_crds` to never memoize an empty-CRD failure. Scope is `operator_module/main.py` only — no frontend, no blueprint authoring, no Dockerfile changes.
</domain>

<decisions>
## Implementation Decisions

### Auth Mechanism (OPA-01)

- **D-01:** Use `--registry-config <tempfile>` (not `helm registry login`) for all OCI quay helm operations. Write the auth JSON to a temp file per OCI component, pass the file path to all relevant helm subprocess calls, and delete the file in a `try/finally` block. This keeps credentials out of process args and avoids writing persistent state to `~/.config/helm/registry/config.json` (which would create shared state across concurrent reconciles).

- **D-02:** Credential source is `stack_vars["quay_dockerconfigjson"]` — the complete docker auth JSON already present in `appStack.variables` (merged into `stack_vars` at line 1042 of `operator_module/main.py`). Write this string directly to the temp registry-config file; helm's `--registry-config` file format is identical to docker's `config.json` format.

- **D-03:** Auth applies only to the `handle_appstack_deployment` code path. The `handle_helm_deployment` single-chart path is NOT touched — no quay OCI charts use `spec.helmChart` CRs in this milestone. Future phases can extend to that path if needed.

### Auth Injection Point (OPA-01 threading)

- **D-04:** The temp registry-config file is written once per OCI helmChart component block. A new `registry_config_path` parameter is added to `discover_chart_crds`, `should_skip_crds_for_component`, and `HelmOperator.install_or_upgrade` / `_install_chart` / `_upgrade_chart`. The parameter is optional (`None` by default) so non-OCI callers are unaffected. The file is cleaned up in a `try/finally` after `install_or_upgrade` returns.

- **D-05:** The registry-config file is written only when `chart_repo.startswith("oci://")` AND `"quay_dockerconfigjson" in stack_vars`. If `quay_dockerconfigjson` is absent from `stack_vars` (e.g., a different OCI registry), no registry-config file is written and helm commands proceed without `--registry-config` (existing behavior preserved).

### discover_chart_crds Cache Fix (OPA-02)

- **D-06:** Remove the `@lru_cache(maxsize=128)` decorator from `discover_chart_crds`. Replace with a module-level `_chart_crds_cache: dict[tuple, set]`. Write to the dict ONLY when `subprocess.check_output` succeeds (no exception). Failed runs (auth error, network blip, any `Exception`) return `set()` uncached — the next call re-attempts the subprocess. Successful runs with no CRDs (chart genuinely has none) cache `set()` normally, so a successful "no-CRDs" result is not retried on every call.

- **D-07:** The cache key includes the `registry_config_path` parameter (from D-04) so that an auth-failed call (`registry_config_path=None`) and a later successful call (`registry_config_path="/tmp/..."`) are not treated as the same cache entry.

### Scope

- **D-08:** All changes are confined to `operator_module/main.py`. No new files, no Dockerfile changes, no Helm chart changes. The operator single-file design is preserved.

### Claude's Discretion

- Exact temp file location (`tempfile.NamedTemporaryFile` with `delete=False` then explicit unlink in `finally` — matching the existing `values_file` pattern in `_install_chart`).
- Whether to add a helper function (e.g., `_write_registry_config(quay_dockerconfigjson: str) -> str`) or inline the write logic — either is fine given the single-file design.
- Whether to log a debug message when registry-config is applied (recommended: yes, at DEBUG level, logging the file path but NOT the content).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `.planning/REQUIREMENTS.md` — OPA-01, OPA-02 acceptance criteria
- `.planning/ROADMAP.md` — Phase 28 success criteria (4 items)
- `.planning/PRD-install-wizard-weka-storage-stack.md` — authoritative spec; Decision C REVISED
- `operator_module/main.py` — entire file (single-module design):
  - `discover_chart_crds` (~line 673) — `@lru_cache` and empty-set failure paths
  - `should_skip_crds_for_component` (~line 724) — calls `discover_chart_crds` at line 740
  - `handle_appstack_deployment` (~line 1042) — `stack_vars` population, helmChart component processing (lines 1090–1197)
  - `HelmOperator._install_chart` (~line 136) — subprocess call pattern for `--registry-config` injection
  - `HelmOperator._upgrade_chart` (~line 180) — same
  - `HelmOperator.install_or_upgrade` (~line 81) — entry point for both install/upgrade paths
- `operator_module/tests/` — existing test patterns for mocking subprocess calls
- `.planning/phases/27-install-blueprint-authoring/27-CONTEXT.md` — Phase 27 D-06 (x-variables including `quay_dockerconfigjson`), canonical blueprint structure
- `.planning/STATE.md` — Decision C REVISED (load-bearing architectural decision)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `HelmOperator._install_chart` (line 136): temp file pattern (`tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)` + `try/finally` unlink) — use this same pattern for the registry-config temp file.
- `stack_vars` at line 1042: `{'namespace': namespace, **raw_user_vars}` where `raw_user_vars = (app_stack.get('variables') or {})` — `quay_dockerconfigjson` is available here when the blueprint includes it.
- `should_skip_crds_for_component` (line 724): thin wrapper over `discover_chart_crds` and `list_existing_crds` — the natural injection point for `registry_config_path`.

### Established Patterns

- Single-file operator module — no new files or imports beyond stdlib `tempfile` (already used).
- `subprocess.run` / `subprocess.check_output` with explicit arg lists — never shell=True.
- `@lru_cache` used for `discover_chart_crds` and `list_existing_crds` — OPA-02 requires removing cache from `discover_chart_crds` and replacing with manual dict; `list_existing_crds` cache is unaffected.
- `skip_crds=False` fallback on any exception in CRD strategy evaluation (line 1163–1166) — maintain this safety net.

### Integration Points

- `handle_appstack_deployment` at line 1152–1166 calls `should_skip_crds_for_component` then `install_or_upgrade` — registry-config file must be live for both calls.
- `discover_chart_crds` is also imported/called from `should_skip_crds_for_component` at line 724 (module-level function, not a method) — parameter threading goes through `should_skip_crds_for_component`.
</code_context>

<specifics>
## Specific Ideas

- The `quay_dockerconfigjson` value is a raw JSON string (not base64-encoded) in docker config.json format: `{"auths":{"quay.io":{"auth":"base64(user:pass)"}}}`. This is exactly what helm's `--registry-config` expects — write it directly to a temp file with `.json` suffix.

- The `--registry-config` flag is supported by `helm show crds`, `helm install`, and `helm upgrade`. All three subcommands need it.

- When `quay_dockerconfigjson` is absent from `stack_vars` (non-quay OCI chart, or blueprint doesn't include the variable), the OCI path falls through to existing behavior with no `--registry-config` — backwards-compatible.

- For the cache key in `_chart_crds_cache`, using `(chart_ref, version, registry_config_path)` as the key means: a successful auth'd call caches a result keyed to that config path. On a fresh operator pod (new reconcile loop, new temp file path), the cache miss triggers a fresh `helm show crds` — this is acceptable and correct.
</specifics>

<deferred>
## Deferred Ideas

- Extending `handle_helm_deployment` (single-chart path) with OCI quay auth — not in Phase 28 scope; no current CRs use `spec.helmChart` with quay OCI.
- Persistent helm registry login across operator lifetime (e.g., at startup if quay credentials are injected as env vars) — deferred; the per-operation temp-file approach is sufficient for v8.0.
- `list_existing_crds` cache invalidation — `@lru_cache(maxsize=1)` on `list_existing_crds` is a separate concern; not touched in Phase 28.
</deferred>
