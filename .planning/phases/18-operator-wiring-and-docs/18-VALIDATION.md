---
phase: 18
slug: operator-wiring-and-docs
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-08
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x (locally installed); `pytest>=8.0.0` pinned in `operator_module/requirements-dev.txt` (Phase 16) |
| **Config file** | None (no `pytest.ini` / `pyproject.toml [tool.pytest]` per Phase 16 D-09) |
| **Quick run command** | `pytest operator_module/tests/test_appstack.py operator_module/tests/test_helm_non_wiring.py -x` |
| **Full suite command** | `pytest operator_module/tests/` |
| **Estimated runtime** | ~3–7 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest operator_module/tests/test_appstack.py operator_module/tests/test_helm_non_wiring.py -x` (skips snapshot for speed)
- **After every plan wave:** Run `pytest operator_module/tests/`
- **Before `/gsd-verify-work`:** Full suite must be green + `python -m py_compile operator_module/main.py` + visual README rendered review
- **Max feedback latency:** ~7 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 18-XX-XX | XX | 1 | OP-06 | — | `${namespace}` auto-defaults to CR namespace; explicit user `namespace` key overrides | unit | `pytest operator_module/tests/test_appstack.py::test_namespace_auto_defaults_to_cr_namespace -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-06 | — | Explicit user `namespace` override wins over auto-default | unit | `pytest operator_module/tests/test_appstack.py::test_explicit_namespace_override_wins -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-07 | — | `${VAR}` in `kubernetesManifest` renders before `kubectl apply` | unit (mock subprocess.run) | `pytest operator_module/tests/test_appstack.py::test_kubernetes_manifest_substitutes_namespace -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-07 | — | Render failure on manifest → `kopf.PermanentError` naming variable + component | unit | `pytest operator_module/tests/test_appstack.py::test_undefined_variable_in_manifest_raises_permanent_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-08 | — | `${VAR}` in ConfigMap valuesFile renders before `yaml.safe_load` | unit (mock kr8s) | `pytest operator_module/tests/test_appstack.py::test_configmap_valuesfile_substitutes_variables -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-08 | — | `${VAR}` in Secret valuesFile renders after base64 decode | unit (mock kr8s) | `pytest operator_module/tests/test_appstack.py::test_secret_valuesfile_substitutes_variables -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-09, TST-05 | — | `handle_helm_deployment` does NOT receive `variables=` kwarg | unit (mock-call assertion) | `pytest operator_module/tests/test_helm_non_wiring.py::test_handle_helm_deployment_does_not_pass_variables -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-09, TST-05 | — | `handle_helm_deployment` body does NOT contain `render(` | static (inspect.getsource) | `pytest operator_module/tests/test_helm_non_wiring.py::test_handle_helm_deployment_source_has_no_render -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-10 | — | Hyphenated variable key (`my-host`) raises `kopf.PermanentError` at dict-build time | unit | `pytest operator_module/tests/test_appstack.py::test_invalid_variable_key_raises_permanent_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-10 | — | Non-string variable value raises `kopf.PermanentError` at dict-build time | unit | `pytest operator_module/tests/test_appstack.py::test_non_string_variable_value_raises_permanent_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-11 | — | Missing ConfigMap (kr8s.NotFoundError) → `kopf.TemporaryError(delay=30)` | unit | `pytest operator_module/tests/test_appstack.py::test_missing_configmap_raises_temporary_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-11 | — | RBAC denial (kr8s.ServerError status_code=403) → `kopf.PermanentError` | unit | `pytest operator_module/tests/test_appstack.py::test_rbac_denied_raises_permanent_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-11 | — | API timeout (kr8s.APITimeoutError) → `kopf.TemporaryError(delay=30)` | unit | `pytest operator_module/tests/test_appstack.py::test_api_timeout_raises_temporary_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-11 | — | Malformed YAML in ConfigMap → `kopf.PermanentError` | unit | `pytest operator_module/tests/test_appstack.py::test_malformed_yaml_raises_permanent_error -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 1 | OP-12 | — | `@kopf.on.update` decorator carries `field='spec'` filter | static (source-grep) | `pytest operator_module/tests/test_appstack.py::test_update_handler_has_field_spec_filter -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 2 | TST-02 | — | Substitution-behavior surface delivered in `test_appstack.py` | meta | (covered by all `test_appstack.py::*` tests) | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 2 | TST-03 | — | Backward-compat snapshot: `ai-research.yaml` (snake→camel normalized) produces byte-identical merged values dict + manifest tempfile content | snapshot | `pytest operator_module/tests/test_backward_compat_snapshot.py -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 2 | TST-05 | — | Two-layered non-wiring assertion (mock-call + static source-grep) | unit + static | `pytest operator_module/tests/test_helm_non_wiring.py -x` | ❌ W0 | ⬜ pending |
| 18-XX-XX | XX | 3 | DOC-01..06 | — | README contains the new section with syntax table, worked AIDP-style example, `$$` escape, `${namespace}` auto-default, no-recursion callout (WRONG/CORRECT pair), operator-control-fields callout (drop targetNamespace recommendation) | manual (rendered Markdown review) + grep | `grep -c '## Variable substitution in AppStack manifests' README.md` returns `1`; `grep -c '${namespace}' README.md` ≥ 3; `grep -c 'WRONG' README.md` ≥ 1 | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `operator_module/tests/test_appstack.py` — NEW; covers OP-06, OP-07, OP-08, OP-10, OP-11, OP-12 (TST-02 surface)
- [ ] `operator_module/tests/test_helm_non_wiring.py` — NEW; covers OP-09 / TST-05 (two-layered: mock-call + inspect.getsource)
- [ ] `operator_module/tests/test_backward_compat_snapshot.py` — NEW; covers TST-03
- [ ] `operator_module/tests/snapshots/ai-research/` — NEW directory; `values_<comp>.json` (sort_keys=True) + `manifest_<comp>.yaml` baselines generated via `BASELINE_REGEN=1 pytest operator_module/tests/test_backward_compat_snapshot.py`
- [ ] In-test snake_case→camelCase fixture normalization helper (~5 lines) to bridge `ai-research.yaml` field convention to `handle_appstack_deployment`'s camelCase reads
- [ ] No new framework install (pytest already pinned ≥8.0.0 in `operator_module/requirements-dev.txt`)
- [ ] No new shared fixtures (Phase 16's `operator_module/tests/conftest.py` `sys.path` injection covers all three new test files)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README rendered structure (TOC placement, table styling, callout block rendering on GitHub) | DOC-01..06 | Visual rendering on GitHub differs from local Markdown; tables and `> Note:` callouts must look right in PR preview | Open PR; click "Files changed"; visually inspect README.md preview; confirm new section sits between "Common configuration" and "Upgrading"; confirm syntax table renders; confirm `> Note:` callout renders distinctly; confirm WRONG/CORRECT code blocks have a `# WRONG` / `# CORRECT` heading comment so the contrast is unmistakable |
| Worked-example CR is realistic vs AIDP shape | DOC-01..06 | Phase 20 will migrate AIDP using these patterns; the README example must be copy-paste-adaptable for those authors | After README change, read `aidp/appstack/weka-aidp-appstack.yaml` (in separate aidp repo) and confirm the README example uses the same field shape (helmChart + valuesFiles + ${namespace} + ${milvusHost}) — even if values differ |
| Live-cluster smoke (full E2E render path) | OP-06..12 | Unit tests mock kr8s + subprocess; an end-to-end smoke against a real cluster verifies the whole render+apply chain | Deferred — Phase 20 (AIDP migration smoke test) covers this. Phase 18 ships unit + static + snapshot tests only; live verification is owned by Phase 20. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (the 3 new test files + snapshots dir)
- [ ] No watch-mode flags (`pytest -x` halts on first failure; no `--watch`)
- [ ] Feedback latency < 7s (quick command runs `test_appstack.py` + `test_helm_non_wiring.py` only)
- [ ] `nyquist_compliant: true` set in frontmatter once plans land

**Approval:** pending
