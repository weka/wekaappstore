---
phase: 18-operator-wiring-and-docs
verified: 2026-05-08T00:00:00Z
status: human_needed
score: 5/5 success criteria verified at function level (16/16 requirement IDs satisfied)
overrides_applied: 0
human_verification:
  - test: "Decide whether the pre-existing per-component `except Exception as e:` block at operator_module/main.py:899 is acceptable for v5.0, or must be fixed before Phase 18 closes"
    expected: "Either (a) accept and document in REQUIREMENTS.md/STATE.md as a known v5.0 limitation deferred to a later phase (operator-hardening), or (b) require a closure plan that removes the broad swallow so kopf.TemporaryError/PermanentError reach the kopf reconcile loop and SC-2/SC-3 are observable end-to-end."
    why_human: "The function-level OP-07/OP-08/OP-11 contracts are met (render failures raise kopf.PermanentError; fetch failures raise kopf.TemporaryError); tests verify this directly on `_render_or_raise` and `load_values_from_reference`. However, the per-component `except Exception` at main.py:899 catches both kopf.PermanentError and kopf.TemporaryError raised inside the component loop and converts them to `comp_status['message']`. As a result, a missing ConfigMap does NOT trigger kopf's 30-second retry behavior — the loop logs the error, marks the component Failed, and proceeds. SC-2 ('PermanentError... is raised when ${unset} appears in a manifest') and SC-3 ('a missing CM/Secret surfaces as kopf.TemporaryError(delay=30) rather than silent empty dict') are technically satisfied at the function level (which is what OP-07/OP-11 wording requires) but are not observable at the kopf reconcile boundary. This is pre-existing behavior (not introduced by Phase 18) and is explicitly flagged in 18-03-SUMMARY.md decision #1. The verification context invites a developer decision: accept-and-defer or block-until-fixed."
  - test: "Visually inspect the README.md ## Variable substitution in AppStack manifests section in a GitHub Markdown previewer"
    expected: "Section sits between '## Common configuration' and '## Upgrading' in the rendered TOC; syntax table renders as a 2-column table; `> **Note:**` block renders as a callout; WRONG/CORRECT YAML blocks render with syntax highlighting; the section is discoverable and copy-paste-adapt-able by an AIDP author."
    why_human: "Markdown rendering on GitHub differs from local previewers; tables, callouts, and code-block syntax highlighting need a human eye to confirm DOC-01..06 are presented well. Automated grep confirms anchor presence (all 16 grep gates green) but cannot confirm visual quality."
  - test: "Live-cluster smoke test of full ${VAR} → kubectl apply chain"
    expected: "Apply a real WekaAppStore CR with `${namespace}` and `${milvusHost}` against a live cluster; confirm rendered manifests apply into the correct namespace; confirm a CR with `${unset}` is rejected at reconcile time; confirm a CR pointing at a missing ConfigMap retries every 30s."
    why_human: "Unit tests mock kr8s + subprocess + HelmOperator. End-to-end behavior at the kopf reconcile boundary is owned by Phase 20 (AIDP migration smoke test in the separate aidp repo). Phase 18 ships the wiring + tests; Phase 20 verifies the contract end-to-end."
---

# Phase 18: Operator Wiring and Docs — Verification Report

**Phase Goal:** The render() helper is wired into both substitution sites in handle_appstack_deployment and load_values_from_reference; ${namespace} auto-defaults to CR namespace; key-name validation, fetch-error upgrade, and field='spec' guard are in place; README documents the feature; the non-wiring of handle_helm_deployment is locked by a test.

**Verified:** 2026-05-08
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Phase 18 Success Criteria)

| #   | Truth   | Status     | Evidence       |
| --- | ------- | ---------- | -------------- |
| SC-1   | A WekaAppStore CR with `metadata.namespace: staging` and no `variables:` field applies with byte-identical Helm values dict and manifest tempfile content compared to pre-Phase-18 — backward-compat snapshot test passes (TST-03, OP-06 verified) | VERIFIED | `operator_module/tests/test_backward_compat_snapshot.py::test_helm_values_byte_identical_to_baseline[qdrant]` and `[research-api]` both PASS. Baselines committed at `operator_module/tests/snapshots/ai-research/values_qdrant.json` (`{"replicaCount": 1}`) and `values_research-api.json` (`{"image": {"tag": "latest"}}`). Phase 16 render() pre-scan guard short-circuits on the no-variables fixture, locking source-level no-op. **Coverage scope:** inline-values + helm-install path only (fixture has no valuesFiles or kubernetesManifest components — disclosed in 18-05-SUMMARY.md). |
| SC-2   | A CR with `metadata.namespace: staging` and a `kubernetesManifest:` containing `namespace: ${namespace}` causes resources to be created in `staging`; a `kopf.PermanentError` naming the variable and component is raised when `${unset}` appears in a manifest (OP-07, DOC-04) | VERIFIED at function level (see human verification #1 for reconcile-boundary concern) | `test_kubernetes_manifest_substitutes_namespace` (PASS) confirms `namespace: ${namespace}` renders to `namespace: staging` in the captured kubectl tempfile. `test_undefined_variable_in_manifest_raises_permanent_error` (PASS) confirms `_render_or_raise` raises `kopf.PermanentError` whose message contains `unset`, `Component`, and `ingress`. Implementation at `operator_module/main.py:870-876` runs `_render_or_raise(manifest_yaml, stack_vars, source_desc=...)` immediately before the `tempfile.NamedTemporaryFile` write. **Caveat:** the per-component `except Exception` at `main.py:899` catches the kopf.PermanentError and writes it to `comp_status['message']` — see human-verification item #1. |
| SC-3   | A ConfigMap referenced via `valuesFiles:` containing `host: ${milvusHost}` deep-merges into Helm values with the resolved value; a missing CM/Secret surfaces as `kopf.TemporaryError(delay=30)` rather than silent empty dict (OP-08, OP-11) | VERIFIED at function level (see human verification #1 for reconcile-boundary concern) | `test_configmap_valuesfile_substitutes_variables` and `test_secret_valuesfile_substitutes_variables` (both PASS) confirm the rendered host value (`milvus.aidp-prod.svc.cluster.local`) lands in the HelmOperator install_or_upgrade call args. `test_missing_configmap_raises_temporary_error` (PASS) confirms `kr8s.NotFoundError → kopf.TemporaryError(delay=30)` with `delay==30` and message naming the missing CM and component. Implementation at `operator_module/main.py:411-483` (`load_values_from_reference`) renders before yaml.safe_load and dispatches kr8s exceptions to typed kopf errors. **Caveat:** same `except Exception` swallow at `main.py:899` applies. |
| SC-4   | A CR with `variables: {my-host: foo}` raises `kopf.PermanentError` at variables-dict build time naming `my-host` as invalid; `handle_helm_deployment` single-chart path does not receive variables wiring and its unit test passes (OP-09, OP-10, TST-05) | VERIFIED | `test_invalid_variable_key_raises_permanent_error` (PASS) confirms hyphenated key triggers `kopf.PermanentError` with message naming `my-host` and matching `Python identifier` text. The validation runs at `main.py:694-704` AFTER the `if not enabled_components: return {...}` early-return and BEFORE `resolve_dependencies()` — fail-fast without partial deployment. `test_handle_helm_deployment_does_not_pass_variables` and `test_handle_helm_deployment_source_has_no_render` (both PASS) confirm zero `variables=`/`comp_name=`/`ref_index=` kwargs and zero `render(`/`_render_or_raise(`/`stack_vars` references in `handle_helm_deployment` source. The helm-path callsite at `main.py:1022-1027` is byte-identical kwarg form `kind=, name=, key=, namespace=`. |
| SC-5   | README contains worked `${VAR}` example, `$$` password example, `${namespace}` auto-default explanation, strict-failure documentation using fully-resolved values, explicit callout that operator-control fields are not templated (DOC-01..06) | VERIFIED | All 16 grep anchors from 18-02-SUMMARY.md pass. Section sits at `README.md:80` between `## Common configuration` (line 39) and `## Upgrading` (line 201). `${milvusHost}` count=2, `${namespace}` count=4, `` `$$` `` count=1, `kopf.PermanentError` count=4, `kopf.TemporaryError` count=1, `# WRONG` count=1, `# CORRECT` count=1, broken `milvus.${namespace}.svc.cluster.local` appears ONLY in WRONG snippet (count=1), `> **Note:** Variable values are taken literally` callout=1, `Omit \`targetNamespace\``=1 (case-insensitive). Visual rendering deferred to human verification #2. |

**Score:** 5/5 success criteria verified at function level. SC-2 and SC-3 carry a reconcile-boundary caveat (see human verification #1).

### Required Artifacts

| Artifact | Expected    | Status | Details |
| -------- | ----------- | ------ | ------- |
| `operator_module/main.py` | `_render_or_raise` helper, `load_values_from_reference` rewrite, `handle_appstack_deployment` wiring, `field='spec'` decorator, `import re` | VERIFIED | 1239 lines; py_compile clean; `_render_or_raise` at line 291; `load_values_from_reference` at line 411 with new signature; `handle_appstack_deployment` at line 664 with stack_vars build at line 691-704, valuesFiles loop at 798-808, manifest render at 871-876; `@kopf.on.update(...field='spec')` at line 1152; `import re` at line 9 |
| `operator_module/tests/test_appstack.py` | 13 tests covering OP-06..08, OP-10..12 | VERIFIED | 491 lines; 13 named tests all PASS in 0.76s; mocks subprocess.run / kr8s.objects.* / HelmOperator / _load_kube_config_once |
| `operator_module/tests/test_helm_non_wiring.py` | 2 tests: runtime mock-call + static inspect.getsource | VERIFIED | 152 lines; both tests PASS in 1.16s; runtime test asserts `len(args)==0` AND `set(kwargs.keys())=={kind,name,key,namespace}`; static test asserts no `render(`, `_render_or_raise(`, or `stack_vars` (with comment-stripping for self-invalidation hygiene) |
| `operator_module/tests/test_backward_compat_snapshot.py` | Parametrized over release_names with BASELINE_REGEN gate | VERIFIED | 215 lines; 2 parametrized cases (`qdrant`, `research-api`) both PASS in 0.78s; `_normalize_camel` helper handles snake_case→camelCase fixture conversion |
| `operator_module/tests/snapshots/ai-research/values_qdrant.json` | Baseline merged Helm values for vector-db (release_name: qdrant) | VERIFIED | Present; content `{"replicaCount": 1}` |
| `operator_module/tests/snapshots/ai-research/values_research-api.json` | Baseline merged Helm values for research-api component | VERIFIED | Present; content `{"image": {"tag": "latest"}}` |
| `README.md` | New `## Variable substitution in AppStack manifests` section between `## Common configuration` and `## Upgrading` | VERIFIED | Section at line 80; Common at line 39; Upgrading at line 201; ordering correct |
| `weka-app-store-operator-chart/Chart.yaml` | version 0.1.63 | VERIFIED | `version: 0.1.63` confirmed; `appVersion: "1.16.0"` unchanged |

### Key Link Verification

| From | To  | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| `_render_or_raise` (main.py:291) | `render()` (main.py:254) | wraps `render(text, variables)` in `try/except (KeyError, ValueError)` and re-raises `kopf.PermanentError(f"{source_desc}: {e}") from e` | WIRED | Source at lines 305-308 verified |
| `handle_appstack_deployment` kubernetesManifest branch (line 870-876) | `_render_or_raise` | `manifest_yaml = _render_or_raise(manifest_yaml, stack_vars, source_desc=f"Component '{comp_name}'.kubernetesManifest")` | WIRED | grep returns 1 match for `manifest_yaml = _render_or_raise(`; placement is BEFORE the `tempfile.NamedTemporaryFile` write at line 878 |
| `load_values_from_reference` (main.py:411) | `_render_or_raise` | `values_yaml = _render_or_raise(values_yaml, variables, source_desc=...)` runs when `variables is not None` BEFORE yaml.safe_load | WIRED | Source at lines 471-476 verified |
| `handle_appstack_deployment` valuesFiles loop (line 798-808) | `load_values_from_reference` | kwargs `variables=stack_vars, comp_name=comp_name, ref_index=idx` appended to existing kwarg call | WIRED | Source at lines 800-808 verified; `for idx, values_ref in enumerate(...)` at line 798 |
| `@kopf.on.update` decorator at line 1152 | `field='spec'` filter | single-line `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')` | WIRED | Exactly one match for the field='spec' form; zero matches for the unfiltered form |
| `handle_helm_deployment` (line 970-1064) | `load_values_from_reference` | kwarg-only call `kind=, name=, key=, namespace=` at lines 1022-1027 | WIRED (intentional non-wiring of variables) | Verified zero `variables=`/`comp_name=`/`ref_index=` kwargs; zero `render(`/`_render_or_raise(`/`stack_vars` references — TST-05 locked |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `_render_or_raise` | text | caller-supplied via render() (Phase 16 helper, locked) | Yes — string.Template.substitute strict mode | FLOWING |
| `load_values_from_reference` | values_yaml | kr8s.objects.ConfigMap.get() / kr8s.objects.Secret.get() (real cluster fetch) → optional render → yaml.safe_load | Yes — DB equivalent (real K8s API call) | FLOWING |
| `handle_appstack_deployment` | stack_vars | `{'namespace': namespace, **(app_stack.get('variables') or {})}` — namespace from kopf-supplied CR metadata; user vars from CR spec | Yes — real CR-supplied data | FLOWING |
| `handle_appstack_deployment` | manifest_yaml | `component['kubernetesManifest']` from CR → `_render_or_raise(...)` → tempfile → kubectl apply | Yes — real CR-supplied manifest text | FLOWING |
| README section | static documentation | author-supplied prose | N/A — documentation, not dynamic | N/A |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full operator test suite green | `pytest operator_module/tests/ -v` | 29 passed in 0.75s (Phase 16: 12 render tests + Phase 18: 17 new tests) | PASS |
| main.py compiles | `python -m py_compile operator_module/main.py` | exits 0 | PASS |
| `import re` is present | `grep -c "^import re" operator_module/main.py` | 1 | PASS |
| Field='spec' decorator is the ONLY @kopf.on.update | `grep -nE "^@kopf.on.update" operator_module/main.py` | 1 line at 1152, includes `field='spec'` | PASS |
| No yaml.load regression | `grep -nE "yaml.load\(" operator_module/main.py` | 0 (only yaml.safe_load and yaml.safe_load_all) | PASS |
| Helm-path zero variable wiring | `awk 'NR>=970 && NR<=1070' operator_module/main.py \| grep -E "render\(|_render_or_raise\|stack_vars\|variables="` | 0 matches | PASS |
| Chart.yaml version bump | `grep -c "^version: 0.1.63$" weka-app-store-operator-chart/Chart.yaml` | 1 | PASS |
| README section heading present | `grep -c "^## Variable substitution in AppStack manifests$" README.md` | 1 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| OP-06 | 18-01 | Variables dict built once at stack scope: `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` before component loop | SATISFIED | `main.py:704` `stack_vars = {'namespace': namespace, **raw_user_vars}`; tests `test_namespace_auto_defaults_to_cr_namespace` + `test_explicit_namespace_override_wins` PASS |
| OP-07 | 18-01 | kubernetesManifest rendered before kubectl apply; render failures raise kopf.PermanentError naming variable + component | SATISFIED at function level | `main.py:870-876` runs `_render_or_raise` before tempfile write; tests `test_kubernetes_manifest_substitutes_namespace` + `test_undefined_variable_in_manifest_raises_permanent_error` PASS. **See human verification #1.** |
| OP-08 | 18-01 | load_values_from_reference renders raw CM data string and base64-decoded Secret string before yaml.safe_load | SATISFIED | `main.py:471-476` runs `_render_or_raise` before yaml.safe_load when variables is not None; tests `test_configmap_valuesfile_substitutes_variables` + `test_secret_valuesfile_substitutes_variables` PASS |
| OP-09 | 18-01, 18-04 | load_values_from_reference signature uses variables=None default; handle_helm_deployment single-chart callsite NOT wired with variables | SATISFIED | Signature at `main.py:411-419` has `variables: Optional[Dict[str, str]] = None`; helm-path callsite at `main.py:1022-1027` uses kwargs only with NO `variables=`/`comp_name=`/`ref_index=`; tests `test_handle_helm_deployment_does_not_pass_variables` + `test_handle_helm_deployment_source_has_no_render` PASS |
| OP-10 | 18-01 | Variable key names validated as Python identifiers when dict is built; invalid keys raise kopf.PermanentError early | SATISFIED | `main.py:694-703` validates each key against `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)` AND each value against `isinstance(v, str)`; both raise `kopf.PermanentError` with named-key context; tests `test_invalid_variable_key_raises_permanent_error` + `test_non_string_variable_value_raises_permanent_error` PASS |
| OP-11 | 18-01 | load_values_from_reference fetch failures surface as kopf.TemporaryError(delay=30) instead of silent {} return | SATISFIED at function level | `main.py:449-468` typed dispatch: NotFoundError → TemporaryError(delay=30); APITimeoutError → TemporaryError(delay=30); ServerError(>=500) → TemporaryError(delay=30); ServerError(<500) → PermanentError; yaml.YAMLError → PermanentError; tests `test_missing_configmap_raises_temporary_error`, `test_rbac_denied_raises_permanent_error`, `test_api_timeout_raises_temporary_error`, `test_malformed_yaml_raises_permanent_error` all PASS. **See human verification #1 — reconcile-boundary observability concern.** |
| OP-12 | 18-01 | @kopf.on.update decorator gets field='spec' filter to prevent reconcile storms | SATISFIED | `main.py:1152` `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')`; test `test_update_handler_has_field_spec_filter` PASS |
| TST-02 | 18-03 | operator_module/tests/test_appstack.py covers handle_appstack_deployment substitution behavior | SATISFIED | 491-line test file with 13 tests covering manifest path, valuesFiles path, ${namespace} auto-default, explicit override, key/value validation, fetch-error dispatch, field='spec' decorator |
| TST-03 | 18-05 | Backward-compat snapshot test — existing AppStack fixture without variables: produces byte-identical output pre/post change | SATISFIED for inline-values + helm-install path | 215-line test file with 2 parametrized cases over release_names; baselines committed; both regen and assert runs green. **Coverage scope:** inline-values only (fixture has no valuesFiles/kubernetesManifest — disclosed in plan and summary) |
| TST-05 | 18-04 | Test locks handle_helm_deployment non-wiring (variables=None passes through; substitution does not run on single-chart path) | SATISFIED | 152-line test file with two-layered lock: runtime mock-call shape (kwargs-only, exact set) + static inspect.getsource (no render(/_render_or_raise(/stack_vars; comment-stripped) |
| DOC-01 | 18-02 | README section explaining ${VAR} syntax with worked example | SATISFIED | README.md:80-200 has section with AIDP-style multi-component CR worked example using `${milvusHost}` (count=2) |
| DOC-02 | 18-02 | README documents $$ literal-dollar escape with password example | SATISFIED | README.md syntax table includes `` `$$` `` row referencing database-password use case (count=1) |
| DOC-03 | 18-02 | README documents ${namespace} auto-defaulting to CR's metadata.namespace | SATISFIED | README.md `${namespace}` count=4; auto-default explained in syntax table and worked example |
| DOC-04 | 18-02 | README documents strict failure on undefined references (kopf.PermanentError with named variable + component) | SATISFIED | README.md has `kopf.PermanentError` count=4 in syntax table + Errors section; explicitly names "variable, component, source location" |
| DOC-05 | 18-02 | README documents variables NOT recursively resolved; documented examples use fully-resolved values | SATISFIED | `> **Note:** Variable values are taken literally` callout (count=1); WRONG snippet shows broken `milvus.${namespace}.svc.cluster.local` (count=1); CORRECT snippet shows fully-resolved `milvus.aidp-prod.svc.cluster.local` (count=3) |
| DOC-06 | 18-02 | README documents operator-control fields NOT templated; recommends dropping targetNamespace | SATISFIED | README.md names `helmChart.repository`, `releaseName`, `readinessCheck`, `targetNamespace` explicitly; "**Recommendation:** Omit `targetNamespace`" present (case-insensitive grep count=1) |

**Coverage:** 16/16 requirement IDs satisfied. No orphaned requirements (Phase 18 row in REQUIREMENTS.md status table covers exactly OP-06..12, TST-02, TST-03, TST-05, DOC-01..06; TST-04 is correctly mapped to Phase 19).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `operator_module/main.py` | 899 | `except Exception as e: comp_status['message'] = ...` | WARNING (pre-existing) | Catches kopf.PermanentError and kopf.TemporaryError raised by `_render_or_raise` and `load_values_from_reference` inside the per-component loop; converts them to `comp_status['message']` rather than letting them reach the kopf reconcile boundary. Means OP-11's "instead of silent empty dict" intent is met at the function-level (no silent {} return) but not fully at the reconcile boundary (kopf doesn't see TemporaryError → no 30s retry). Pre-existing — not introduced by Phase 18. Documented in 18-03-SUMMARY.md decision #1. **Routed to human verification #1.** |
| (no other anti-patterns found) | — | — | — | — |

No TODO/FIXME/PLACEHOLDER comments introduced by Phase 18. No empty implementations, hardcoded empty data, or console.log-only handlers in the new code.

### Human Verification Required

#### 1. Decide on per-component `except Exception` swallowing (BLOCKING-CANDIDATE → ROUTED TO HUMAN)

**Test:** Read `operator_module/main.py:899` and decide whether the broad `except Exception as e: comp_status['message'] = f"Error deploying component: {str(e)}"` block is acceptable for v5.0 closure.

**Expected:** Either (a) accept and document the pre-existing limitation in REQUIREMENTS.md / STATE.md as a known v5.0 issue deferred to a future operator-hardening phase, OR (b) require a closure plan that narrows the swallow so kopf.TemporaryError/PermanentError propagate to the kopf reconcile loop. Option (b) would change the test surface (Tests 4, 9-12 in test_appstack.py would assert pytest.raises directly on `handle_appstack_deployment` instead of bypassing into `_render_or_raise` / `load_values_from_reference`).

**Why human:** The function-level OP-07/OP-08/OP-11 wording IS satisfied; tests verify the function-level contract. But the reconcile-boundary observability of those errors is a separate concern and is needed for the kopf retry loop (OP-11's stated intent). The verification context note says: *"The OP-11 contract is verified at the FUNCTION level (load_values_from_reference) but the per-component except Exception may make it ineffective at the kopf reconcile boundary. This is a pre-existing bug (not introduced by Phase 18) but worth flagging in the verification report. Recommend: either accept (deferred to v5.1) or block (Phase 18 cannot pass until fixed)."* This is an explicit invitation for developer decision.

#### 2. Visually inspect README rendering on GitHub

**Test:** Open `README.md` in GitHub PR preview (or `glow README.md` locally) and confirm the new `## Variable substitution in AppStack manifests` section renders correctly:
- Section sits between `## Common configuration` and `## Upgrading` in the rendered TOC
- Syntax table renders as a 2-column table (Syntax | Behavior)
- `> **Note:**` block renders as a GitHub callout (gray-background block-quote)
- WRONG and CORRECT YAML blocks render with syntax highlighting and visible `# WRONG` / `# CORRECT` comments

**Expected:** All five visual checks pass; section is discoverable and AIDP authors can copy-paste-adapt the worked example for `aidp/appstack/weka-aidp-appstack.yaml` migration.

**Why human:** GitHub Markdown rendering differs from local previewers; tables, callouts, and code-block syntax highlighting need visual confirmation. Automated grep verifies all 16 anchors are present (DOC-01..06) but cannot confirm visual quality. Per the plan's Manual-Only Verifications table.

#### 3. Live-cluster smoke (deferred to Phase 20)

**Test:** Apply a real WekaAppStore CR with `${namespace}` and `${milvusHost}` against a live cluster; confirm rendered manifests apply into the correct namespace; confirm a CR pointing at a missing ConfigMap eventually retries (or fails depending on outcome of human verification #1); confirm a CR with `${unset}` is rejected.

**Expected:** End-to-end behavior matches Phase 18 contract.

**Why human:** Unit tests mock kr8s + subprocess + HelmOperator. Phase 20 (AIDP migration smoke test in the separate `aidp` repo) owns the live verification. Phase 18 ships the wiring + tests; Phase 20 verifies end-to-end.

### Gaps Summary

**No blocking gaps for the literal Phase 18 ROADMAP success criteria.** All 5 SCs are verified at the function level, all 16 requirement IDs (OP-06..12, TST-02, TST-03, TST-05, DOC-01..06) are satisfied with passing tests and code anchors, all 8 required artifacts exist with substantive content and proper wiring, and the full operator test suite is green (29/29 passing in 0.75s).

**One human-decision item is escalated** (per the verification context's explicit recommendation): the pre-existing `except Exception` block at `main.py:899` swallows kopf.PermanentError/TemporaryError raised by Phase 18's wiring before they can reach the kopf reconcile boundary. The function-level contracts in OP-07/OP-08/OP-11 are met (which is what the requirement wording says), but the reconcile-boundary observability — which is the *spirit* of OP-11's "instead of silent empty dict" — is degraded. This is pre-existing behavior, predates Phase 18, and is explicitly documented in 18-03-SUMMARY.md decision #1. The developer must decide:
- **Accept** as a known v5.0 limitation, document in REQUIREMENTS.md/STATE.md, and defer fix to a future operator-hardening phase. Rationale: requirement wording is met at the function level; tests verify that level; behavior is unchanged from pre-Phase-18 for non-substitution code paths; Phase 20 smoke test will surface the practical impact when it lands. Phase 18 closes.
- **Block** Phase 18 closure until a closure plan narrows the swallow. Rationale: OP-11's stated intent is reconcile-boundary observability; without that, the new typed-error dispatch is wasted on the AppStack path. Add a Plan 18-06 that catches narrower exception types in the per-component loop, lets kopf errors propagate, and updates Tests 4 + 9-12 to assert pytest.raises directly on `handle_appstack_deployment`.

Two additional human-verification items (visual README inspection, live-cluster smoke via Phase 20) are routine for this phase and are noted but not blocking.

---

*Verified: 2026-05-08*
*Verifier: Claude (gsd-verifier)*
