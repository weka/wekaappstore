---
phase: 18-operator-wiring-and-docs
verified: 2026-05-08T00:00:00Z
re_verified: 2026-05-08T17:30:00Z
status: passed
score: 5/5 success criteria verified end-to-end (16/16 requirement IDs satisfied; reconcile-boundary now observable)
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 5/5 at function level (1 BLOCKING-CANDIDATE routed to human)
  closure_commit: d0b6d52
  gaps_closed:
    - "Pre-existing `except Exception as e:` at main.py:899 swallowing kopf.PermanentError/TemporaryError before reconcile boundary — closure fix at d0b6d52 adds explicit `except (kopf.PermanentError, kopf.TemporaryError): raise` BEFORE the broad except"
  gaps_remaining: []
  regressions: []
  new_tests:
    - "test_handle_appstack_propagates_permanent_error_to_kopf_boundary (test_appstack.py:457)"
    - "test_handle_appstack_propagates_temporary_error_to_kopf_boundary (test_appstack.py:489)"
  refactored_tests:
    - "test_undefined_variable_in_manifest_raises_permanent_error (test_appstack.py:195) — narrowed to target _render_or_raise only; end-to-end propagation moved to new boundary test"
human_verification:
  - test: "Visually inspect the README.md ## Variable substitution in AppStack manifests section in a GitHub Markdown previewer"
    expected: "Section sits between '## Common configuration' and '## Upgrading' in the rendered TOC; syntax table renders as a 2-column table; `> **Note:**` block renders as a callout; WRONG/CORRECT YAML blocks render with syntax highlighting; the section is discoverable and copy-paste-adapt-able by an AIDP author."
    why_human: "Markdown rendering on GitHub differs from local previewers; tables, callouts, and code-block syntax highlighting need a human eye to confirm DOC-01..06 are presented well. Automated grep confirms anchor presence (all 16 grep gates green) but cannot confirm visual quality. NON-BLOCKING — routine manual inspection."
  - test: "Live-cluster smoke test of full ${VAR} → kubectl apply chain"
    expected: "Apply a real WekaAppStore CR with `${namespace}` and `${milvusHost}` against a live cluster; confirm rendered manifests apply into the correct namespace; confirm a CR with `${unset}` is rejected at reconcile time and surfaces a kopf.PermanentError event on the CR; confirm a CR pointing at a missing ConfigMap retries every 30s with kopf.TemporaryError events on the CR."
    why_human: "Unit tests mock kr8s + subprocess + HelmOperator. End-to-end behavior at the kopf reconcile boundary is owned by Phase 20 (AIDP migration smoke test in the separate aidp repo). Phase 18 ships the wiring + tests + closure fix; Phase 20 verifies the contract end-to-end. NON-BLOCKING — ROADMAP explicitly assigns live-cluster verification to Phase 20."
---

# Phase 18: Operator Wiring and Docs — Verification Report

**Phase Goal:** The render() helper is wired into both substitution sites in handle_appstack_deployment and load_values_from_reference; ${namespace} auto-defaults to CR namespace; key-name validation, fetch-error upgrade, and field='spec' guard are in place; README documents the feature; the non-wiring of handle_helm_deployment is locked by a test.

**Verified:** 2026-05-08
**Re-verified:** 2026-05-08 (post-closure)
**Status:** PASSED
**Re-verification:** Yes — after closure fix `d0b6d52` for the `except Exception` reconcile-boundary swallow

---

## Re-verification (closure landed)

### What changed

Closure commit `d0b6d52` (`fix(18-closure): propagate kopf.* errors to reconcile boundary`) addressed the single human-needed item flagged in the initial verification.

**Code change at `operator_module/main.py:899-910`:**

```python
        except (kopf.PermanentError, kopf.TemporaryError):
            # Phase 18 OP-07/OP-08/OP-11: kopf-typed errors must reach the reconcile
            # boundary so kopf can re-schedule (TemporaryError) or fail loudly
            # (PermanentError). Don't swallow into comp_status['message'] — that
            # hides transient cluster failures and undefined-variable bugs from
            # operators monitoring the CR's status conditions.
            raise
        except Exception as e:
            comp_status['phase'] = 'Failed'
            comp_status['message'] = f"Error deploying component: {str(e)}"
            failed = True
            logging.error(f"Error deploying component {comp_name}: {str(e)}")
```

The narrow `except (kopf.PermanentError, kopf.TemporaryError): raise` runs FIRST, so kopf-typed exceptions raised by `_render_or_raise` (manifest path) or `load_values_from_reference` (valuesFiles path) propagate UP to the kopf reconcile boundary unchanged. Non-kopf exceptions (e.g., `ValueError("must specify either helmChart or kubernetesManifest")`, subprocess exec failures) still flow to the broad except — backward-compat for non-substitution failure paths is preserved.

**Test additions (`operator_module/tests/test_appstack.py`):**

| # | Test | Asserts | Result |
| --- | --- | --- | --- |
| New | `test_handle_appstack_propagates_permanent_error_to_kopf_boundary` (line 457) | Calling `handle_appstack_deployment` with `kubernetesManifest: "namespace: ${unset}"` raises `kopf.PermanentError` whose message contains `unset` AND `ingress` (the component name) | PASS |
| New | `test_handle_appstack_propagates_temporary_error_to_kopf_boundary` (line 489) | Calling `handle_appstack_deployment` with a `valuesFiles` ref pointing at a `kr8s.NotFoundError` ConfigMap raises `kopf.TemporaryError` with `delay==30` and message containing `missing-cm` | PASS |
| Refactored | `test_undefined_variable_in_manifest_raises_permanent_error` (line 195) | Narrowed to target `_render_or_raise` directly (function-level contract); the end-to-end propagation aspect moved to the new boundary test above | PASS |

### Re-verification spot-checks

| Check | Command | Result | Status |
| --- | --- | --- | --- |
| Closure commit landed | `git log --oneline d0b6d52` | `d0b6d52 fix(18-closure): propagate kopf.* errors to reconcile boundary` | PASS |
| `except (kopf.PermanentError, kopf.TemporaryError): raise` is in place at main.py:899 | `grep -n "except (kopf.PermanentError, kopf.TemporaryError)" operator_module/main.py` | line 899 | PASS |
| Re-raise is BEFORE broad `except Exception` | Read `main.py:899-906` | line 899 narrow except → line 905 `raise` → line 906 broad except | PASS |
| Both new boundary tests present | `grep -n "test_handle_appstack_propagates_" operator_module/tests/test_appstack.py` | line 457 (Permanent) + line 489 (Temporary) | PASS |
| Full operator test suite green | `pytest operator_module/tests/ -v` | **31 passed in 1.28s** (12 Phase 16 render + 13 Phase 18 substitution + 2 reconcile-boundary new + 2 helm non-wiring + 2 backward-compat snapshot) | PASS |
| main.py compiles | `python -m py_compile operator_module/main.py` | exits 0 | PASS |
| `@kopf.on.update` field='spec' filter still single-occurrence | `grep -nE "^@kopf.on.update" operator_module/main.py` | line 1159 (shifted +7 from 1152 due to closure diff), still one match with `field='spec'` | PASS |
| Helm path stays unwired (TST-05) | `awk 'NR>=977 && NR<=1077' main.py \| grep -E "render\(\|_render_or_raise\|stack_vars\|variables="` | 0 matches | PASS |
| Chart.yaml version still 0.1.63 | `grep -c "^version: 0.1.63$" weka-app-store-operator-chart/Chart.yaml` | 1 | PASS |
| README section heading still present | `grep -c "^## Variable substitution in AppStack manifests$" README.md` | 1 | PASS |

### Verdict on the prior BLOCKING-CANDIDATE

**RESOLVED.** The pre-existing per-component `except Exception as e:` no longer swallows `kopf.PermanentError` or `kopf.TemporaryError` raised by Phase 18's wiring. The reconcile-boundary observability concern (which was the *spirit* of OP-11's "instead of silent empty dict" requirement) is now satisfied directly:

- **OP-07 reconcile-boundary:** `_render_or_raise` raises `kopf.PermanentError` → propagates through `handle_appstack_deployment` → kopf records a permanent failure on the CR. Asserted by `test_handle_appstack_propagates_permanent_error_to_kopf_boundary`.
- **OP-11 reconcile-boundary:** `load_values_from_reference` raises `kopf.TemporaryError(delay=30)` → propagates through `handle_appstack_deployment` → kopf reschedules the reconcile after 30s. Asserted by `test_handle_appstack_propagates_temporary_error_to_kopf_boundary`.
- **Backward-compat preserved:** Non-kopf exceptions (e.g., the existing `ValueError` for components missing both `helmChart` and `kubernetesManifest`, or subprocess timeouts) still flow into the broad except and update `comp_status['message']` as before. The `failed = True; break` flow on the original component-failure path is unchanged.

### Regressions

**None.** All 12 Phase 16 render tests, all 13 Phase 18 substitution tests (now refactored to function-level for `test_undefined_variable_in_manifest_raises_permanent_error`), both helm non-wiring tests, and both backward-compat snapshot tests still pass. Net delta: +2 tests, 0 removed, 1 narrowed in scope (with its end-to-end aspect now covered by a more direct boundary test).

---

## Goal Achievement (final, post-closure)

### Observable Truths (ROADMAP Phase 18 Success Criteria)

| #   | Truth   | Status     | Evidence       |
| --- | ------- | ---------- | -------------- |
| SC-1   | A WekaAppStore CR with `metadata.namespace: staging` and no `variables:` field applies with byte-identical Helm values dict and manifest tempfile content compared to pre-Phase-18 — backward-compat snapshot test passes (TST-03, OP-06 verified) | VERIFIED | `operator_module/tests/test_backward_compat_snapshot.py::test_helm_values_byte_identical_to_baseline[qdrant]` and `[research-api]` both PASS. Baselines committed at `operator_module/tests/snapshots/ai-research/values_qdrant.json` (`{"replicaCount": 1}`) and `values_research-api.json` (`{"image": {"tag": "latest"}}`). Phase 16 render() pre-scan guard short-circuits on the no-variables fixture, locking source-level no-op. **Coverage scope:** inline-values + helm-install path only (fixture has no valuesFiles or kubernetesManifest components — disclosed in 18-05-SUMMARY.md). |
| SC-2   | A CR with `metadata.namespace: staging` and a `kubernetesManifest:` containing `namespace: ${namespace}` causes resources to be created in `staging`; a `kopf.PermanentError` naming the variable and component is raised when `${unset}` appears in a manifest (OP-07, DOC-04) | VERIFIED end-to-end | `test_kubernetes_manifest_substitutes_namespace` (PASS) confirms `namespace: ${namespace}` renders to `namespace: staging` in the captured kubectl tempfile. `test_undefined_variable_in_manifest_raises_permanent_error` (PASS) confirms `_render_or_raise` raises `kopf.PermanentError` whose message contains `unset`, `Component`, and `ingress`. **Reconcile-boundary now confirmed:** `test_handle_appstack_propagates_permanent_error_to_kopf_boundary` (PASS) confirms the error propagates OUT of `handle_appstack_deployment` to kopf. Implementation at `operator_module/main.py:870-876` runs `_render_or_raise(manifest_yaml, stack_vars, source_desc=...)` immediately before the `tempfile.NamedTemporaryFile` write; closure fix at `main.py:899-905` re-raises kopf.* before the broad except. |
| SC-3   | A ConfigMap referenced via `valuesFiles:` containing `host: ${milvusHost}` deep-merges into Helm values with the resolved value; a missing CM/Secret surfaces as `kopf.TemporaryError(delay=30)` rather than silent empty dict (OP-08, OP-11) | VERIFIED end-to-end | `test_configmap_valuesfile_substitutes_variables` and `test_secret_valuesfile_substitutes_variables` (both PASS) confirm the rendered host value (`milvus.aidp-prod.svc.cluster.local`) lands in the HelmOperator install_or_upgrade call args. `test_missing_configmap_raises_temporary_error` (PASS) confirms `kr8s.NotFoundError → kopf.TemporaryError(delay=30)` with `delay==30` and message naming the missing CM and component. **Reconcile-boundary now confirmed:** `test_handle_appstack_propagates_temporary_error_to_kopf_boundary` (PASS) confirms `kopf.TemporaryError(delay=30)` propagates OUT of `handle_appstack_deployment` to kopf so the reconcile is rescheduled. Implementation at `operator_module/main.py:411-483` (`load_values_from_reference`) renders before yaml.safe_load and dispatches kr8s exceptions to typed kopf errors; closure fix at `main.py:899-905` lets them through. |
| SC-4   | A CR with `variables: {my-host: foo}` raises `kopf.PermanentError` at variables-dict build time naming `my-host` as invalid; `handle_helm_deployment` single-chart path does not receive variables wiring and its unit test passes (OP-09, OP-10, TST-05) | VERIFIED | `test_invalid_variable_key_raises_permanent_error` (PASS) confirms hyphenated key triggers `kopf.PermanentError` with message naming `my-host` and matching `Python identifier` text. The validation runs at `main.py:694-704` AFTER the `if not enabled_components: return {...}` early-return and BEFORE `resolve_dependencies()` — fail-fast without partial deployment. `test_handle_helm_deployment_does_not_pass_variables` and `test_handle_helm_deployment_source_has_no_render` (both PASS) confirm zero `variables=`/`comp_name=`/`ref_index=` kwargs and zero `render(`/`_render_or_raise(`/`stack_vars` references in `handle_helm_deployment` source. The helm-path callsite at `main.py:1029-1034` (post-closure shift) is byte-identical kwarg form `kind=, name=, key=, namespace=`. |
| SC-5   | README contains worked `${VAR}` example, `$$` password example, `${namespace}` auto-default explanation, strict-failure documentation using fully-resolved values, explicit callout that operator-control fields are not templated (DOC-01..06) | VERIFIED | All 16 grep anchors from 18-02-SUMMARY.md pass. Section sits at `README.md:80` between `## Common configuration` (line 39) and `## Upgrading` (line 201). `${milvusHost}` count=2, `${namespace}` count=4, `` `$$` `` count=1, `kopf.PermanentError` count=4, `kopf.TemporaryError` count=1, `# WRONG` count=1, `# CORRECT` count=1, broken `milvus.${namespace}.svc.cluster.local` appears ONLY in WRONG snippet (count=1), `> **Note:** Variable values are taken literally` callout=1, `Omit \`targetNamespace\``=1 (case-insensitive). Visual rendering deferred to human verification #1 (non-blocking). |

**Score:** 5/5 success criteria verified end-to-end. SC-2 and SC-3's reconcile-boundary observability — previously caveated as function-level only — is now satisfied directly by the closure fix and the two new boundary tests.

### Required Artifacts

| Artifact | Expected    | Status | Details |
| -------- | ----------- | ------ | ------- |
| `operator_module/main.py` | `_render_or_raise` helper, `load_values_from_reference` rewrite, `handle_appstack_deployment` wiring, `field='spec'` decorator, `import re`, **closure-fix narrow except** | VERIFIED | 1246 lines (+7 from initial verification due to closure diff); py_compile clean; `_render_or_raise` at line 291; `load_values_from_reference` at line 411 with new signature; `handle_appstack_deployment` at line 664 with stack_vars build at line 691-704, valuesFiles loop at 798-808, manifest render at 871-876; **closure narrow except at lines 899-905**; `@kopf.on.update(...field='spec')` at line 1159; `import re` at line 9 |
| `operator_module/tests/test_appstack.py` | 13 tests covering OP-06..08, OP-10..12 + **2 new reconcile-boundary tests** | VERIFIED | 532 lines (+41 from initial verification); **15 named tests all PASS in 0.76s**; mocks subprocess.run / kr8s.objects.* / HelmOperator / _load_kube_config_once; new tests at lines 457 and 489 |
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
| `@kopf.on.update` decorator at line 1159 | `field='spec'` filter | single-line `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')` | WIRED | Exactly one match for the field='spec' form; zero matches for the unfiltered form |
| `handle_helm_deployment` (line 977-1071, post-closure shift) | `load_values_from_reference` | kwarg-only call `kind=, name=, key=, namespace=` | WIRED (intentional non-wiring of variables) | Verified zero `variables=`/`comp_name=`/`ref_index=` kwargs; zero `render(`/`_render_or_raise(`/`stack_vars` references — TST-05 locked |
| **`handle_appstack_deployment` per-component except chain (NEW, post-closure)** | **kopf reconcile loop** | `except (kopf.PermanentError, kopf.TemporaryError): raise` BEFORE broad `except Exception` | **WIRED (closure fix)** | **`main.py:899-905`; both reconcile-boundary tests PASS; resolves the prior BLOCKING-CANDIDATE from initial verification** |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `_render_or_raise` | text | caller-supplied via render() (Phase 16 helper, locked) | Yes — string.Template.substitute strict mode | FLOWING |
| `load_values_from_reference` | values_yaml | kr8s.objects.ConfigMap.get() / kr8s.objects.Secret.get() (real cluster fetch) → optional render → yaml.safe_load | Yes — DB equivalent (real K8s API call) | FLOWING |
| `handle_appstack_deployment` | stack_vars | `{'namespace': namespace, **(app_stack.get('variables') or {})}` — namespace from kopf-supplied CR metadata; user vars from CR spec | Yes — real CR-supplied data | FLOWING |
| `handle_appstack_deployment` | manifest_yaml | `component['kubernetesManifest']` from CR → `_render_or_raise(...)` → tempfile → kubectl apply | Yes — real CR-supplied manifest text | FLOWING |
| **`handle_appstack_deployment` exception path (NEW)** | **kopf-typed exceptions raised inside per-component loop** | **`_render_or_raise` (manifest path) and `load_values_from_reference` (valuesFiles path)** | **Yes — kopf-typed exceptions reach the reconcile boundary unchanged** | **FLOWING (closure fix)** |
| README section | static documentation | author-supplied prose | N/A — documentation, not dynamic | N/A |

### Behavioral Spot-Checks (post-closure)

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full operator test suite green | `pytest operator_module/tests/ -v` | **31 passed in 1.28s** (12 Phase 16 render + 13 Phase 18 substitution + 2 reconcile-boundary NEW + 2 helm non-wiring + 2 backward-compat snapshot) | PASS |
| main.py compiles | `python -m py_compile operator_module/main.py` | exits 0 | PASS |
| Closure narrow except is in place BEFORE broad except | Read `main.py:899-906` | line 899 narrow except → line 905 `raise` → line 906 broad except `Exception` | PASS |
| `import re` is present | `grep -c "^import re" operator_module/main.py` | 1 | PASS |
| Field='spec' decorator is the ONLY @kopf.on.update | `grep -nE "^@kopf.on.update" operator_module/main.py` | 1 line at 1159, includes `field='spec'` | PASS |
| No yaml.load regression | `grep -nE "yaml.load\(" operator_module/main.py \| grep -v safe_load` | 0 (only yaml.safe_load and yaml.safe_load_all) | PASS |
| Helm-path zero variable wiring | `awk 'NR>=977 && NR<=1077' main.py \| grep -E "render\(\|_render_or_raise\|stack_vars\|variables="` | 0 matches | PASS |
| Chart.yaml version bump | `grep -c "^version: 0.1.63$" weka-app-store-operator-chart/Chart.yaml` | 1 | PASS |
| README section heading present | `grep -c "^## Variable substitution in AppStack manifests$" README.md` | 1 | PASS |
| Closure commit landed | `git log --oneline d0b6d52` | `d0b6d52 fix(18-closure): propagate kopf.* errors to reconcile boundary` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| OP-06 | 18-01 | Variables dict built once at stack scope: `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` before component loop | SATISFIED | `main.py:704` `stack_vars = {'namespace': namespace, **raw_user_vars}`; tests `test_namespace_auto_defaults_to_cr_namespace` + `test_explicit_namespace_override_wins` PASS |
| OP-07 | 18-01 | kubernetesManifest rendered before kubectl apply; render failures raise kopf.PermanentError naming variable + component | **SATISFIED end-to-end (closure fix applied)** | `main.py:870-876` runs `_render_or_raise` before tempfile write; tests `test_kubernetes_manifest_substitutes_namespace` + `test_undefined_variable_in_manifest_raises_permanent_error` PASS; **`test_handle_appstack_propagates_permanent_error_to_kopf_boundary` PASS confirms reconcile-boundary observability via closure fix at `main.py:899-905`** |
| OP-08 | 18-01 | load_values_from_reference renders raw CM data string and base64-decoded Secret string before yaml.safe_load | SATISFIED | `main.py:471-476` runs `_render_or_raise` before yaml.safe_load when variables is not None; tests `test_configmap_valuesfile_substitutes_variables` + `test_secret_valuesfile_substitutes_variables` PASS |
| OP-09 | 18-01, 18-04 | load_values_from_reference signature uses variables=None default; handle_helm_deployment single-chart callsite NOT wired with variables | SATISFIED | Signature at `main.py:411-419` has `variables: Optional[Dict[str, str]] = None`; helm-path callsite (post-closure shift) uses kwargs only with NO `variables=`/`comp_name=`/`ref_index=`; tests `test_handle_helm_deployment_does_not_pass_variables` + `test_handle_helm_deployment_source_has_no_render` PASS |
| OP-10 | 18-01 | Variable key names validated as Python identifiers when dict is built; invalid keys raise kopf.PermanentError early | SATISFIED | `main.py:694-703` validates each key against `re.fullmatch(r'^[_a-zA-Z][_a-zA-Z0-9]*$', key)` AND each value against `isinstance(v, str)`; both raise `kopf.PermanentError` with named-key context; tests `test_invalid_variable_key_raises_permanent_error` + `test_non_string_variable_value_raises_permanent_error` PASS |
| OP-11 | 18-01 | load_values_from_reference fetch failures surface as kopf.TemporaryError(delay=30) instead of silent {} return | **SATISFIED end-to-end (closure fix applied)** | `main.py:449-468` typed dispatch: NotFoundError → TemporaryError(delay=30); APITimeoutError → TemporaryError(delay=30); ServerError(>=500) → TemporaryError(delay=30); ServerError(<500) → PermanentError; yaml.YAMLError → PermanentError; tests `test_missing_configmap_raises_temporary_error`, `test_rbac_denied_raises_permanent_error`, `test_api_timeout_raises_temporary_error`, `test_malformed_yaml_raises_permanent_error` all PASS; **`test_handle_appstack_propagates_temporary_error_to_kopf_boundary` PASS confirms `delay=30` reaches the kopf reconcile loop via closure fix at `main.py:899-905`** |
| OP-12 | 18-01 | @kopf.on.update decorator gets field='spec' filter to prevent reconcile storms | SATISFIED | `main.py:1159` `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')`; test `test_update_handler_has_field_spec_filter` PASS |
| TST-02 | 18-03 | operator_module/tests/test_appstack.py covers handle_appstack_deployment substitution behavior | SATISFIED (now expanded by closure fix) | 532-line test file with **15** tests covering manifest path, valuesFiles path, ${namespace} auto-default, explicit override, key/value validation, fetch-error dispatch, field='spec' decorator, **and reconcile-boundary propagation (2 new boundary tests)** |
| TST-03 | 18-05 | Backward-compat snapshot test — existing AppStack fixture without variables: produces byte-identical output pre/post change | SATISFIED for inline-values + helm-install path | 215-line test file with 2 parametrized cases over release_names; baselines committed; both regen and assert runs green. **Coverage scope:** inline-values only (fixture has no valuesFiles/kubernetesManifest — disclosed in plan and summary) |
| TST-05 | 18-04 | Test locks handle_helm_deployment non-wiring (variables=None passes through; substitution does not run on single-chart path) | SATISFIED | 152-line test file with two-layered lock: runtime mock-call shape (kwargs-only, exact set) + static inspect.getsource (no render(/_render_or_raise(/stack_vars; comment-stripped) |
| DOC-01 | 18-02 | README section explaining ${VAR} syntax with worked example | SATISFIED | README.md:80-200 has section with AIDP-style multi-component CR worked example using `${milvusHost}` (count=2) |
| DOC-02 | 18-02 | README documents $$ literal-dollar escape with password example | SATISFIED | README.md syntax table includes `` `$$` `` row referencing database-password use case (count=1) |
| DOC-03 | 18-02 | README documents ${namespace} auto-defaulting to CR's metadata.namespace | SATISFIED | README.md `${namespace}` count=4; auto-default explained in syntax table and worked example |
| DOC-04 | 18-02 | README documents strict failure on undefined references (kopf.PermanentError with named variable + component) | SATISFIED | README.md has `kopf.PermanentError` count=4 in syntax table + Errors section; explicitly names "variable, component, source location" |
| DOC-05 | 18-02 | README documents variables NOT recursively resolved; documented examples use fully-resolved values | SATISFIED | `> **Note:** Variable values are taken literally` callout (count=1); WRONG snippet shows broken `milvus.${namespace}.svc.cluster.local` (count=1); CORRECT snippet shows fully-resolved `milvus.aidp-prod.svc.cluster.local` (count=3) |
| DOC-06 | 18-02 | README documents operator-control fields NOT templated; recommends dropping targetNamespace | SATISFIED | README.md names `helmChart.repository`, `releaseName`, `readinessCheck`, `targetNamespace` explicitly; "**Recommendation:** Omit `targetNamespace`" present (case-insensitive grep count=1) |

**Coverage:** 16/16 requirement IDs satisfied end-to-end. No orphaned requirements (Phase 18 row in REQUIREMENTS.md status table covers exactly OP-06..12, TST-02, TST-03, TST-05, DOC-01..06; TST-04 is correctly mapped to Phase 19).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `operator_module/main.py` | 906 | `except Exception as e: comp_status['message'] = ...` (broad catch-all) | INFO (no longer blocking) | The broad except remains for non-kopf exceptions (existing `ValueError("must specify either helmChart or kubernetesManifest")`, subprocess timeouts, unexpected runtime errors). The closure fix at lines 899-905 ensures kopf.PermanentError and kopf.TemporaryError are re-raised BEFORE this broad catch, so OP-07/OP-11 reconcile-boundary observability is preserved. The broad except's behavior on non-kopf exceptions is intentional Phase-pre-18 behavior preserved by the surgical closure fix; it's not a regression and not a goal-blocking concern. |

No TODO/FIXME/PLACEHOLDER comments introduced by Phase 18 or the closure fix. No empty implementations, hardcoded empty data, or console.log-only handlers in the new code.

### Human Verification Required (non-blocking, routine)

#### 1. Visually inspect README rendering on GitHub

**Test:** Open `README.md` in GitHub PR preview (or `glow README.md` locally) and confirm the new `## Variable substitution in AppStack manifests` section renders correctly:
- Section sits between `## Common configuration` and `## Upgrading` in the rendered TOC
- Syntax table renders as a 2-column table (Syntax | Behavior)
- `> **Note:**` block renders as a GitHub callout (gray-background block-quote)
- WRONG and CORRECT YAML blocks render with syntax highlighting and visible `# WRONG` / `# CORRECT` comments

**Expected:** All five visual checks pass; section is discoverable and AIDP authors can copy-paste-adapt the worked example for `aidp/appstack/weka-aidp-appstack.yaml` migration.

**Why human:** GitHub Markdown rendering differs from local previewers; tables, callouts, and code-block syntax highlighting need visual confirmation. Automated grep verifies all 16 anchors are present (DOC-01..06) but cannot confirm visual quality. Per the plan's Manual-Only Verifications table. **NON-BLOCKING.**

#### 2. Live-cluster smoke (deferred to Phase 20)

**Test:** Apply a real WekaAppStore CR with `${namespace}` and `${milvusHost}` against a live cluster; confirm rendered manifests apply into the correct namespace; confirm a CR with `${unset}` is rejected at reconcile time and surfaces a `kopf.PermanentError` event on the CR's status conditions; confirm a CR pointing at a missing ConfigMap retries every 30s with `kopf.TemporaryError` events on the CR's status conditions.

**Expected:** End-to-end behavior matches Phase 18 contract; the closure fix's reconcile-boundary observability translates to visible CR status conditions and event timelines.

**Why human:** Unit tests mock kr8s + subprocess + HelmOperator. Phase 20 (AIDP migration smoke test in the separate `aidp` repo) owns the live verification. Phase 18 ships the wiring + tests + closure fix; Phase 20 verifies end-to-end. **NON-BLOCKING — ROADMAP explicitly assigns live-cluster verification to Phase 20.**

### Gaps Summary

**No blocking gaps remain.** The single BLOCKING-CANDIDATE from initial verification — pre-existing `except Exception` swallowing kopf-typed errors at `main.py:899` — has been resolved by closure commit `d0b6d52`. The narrow `except (kopf.PermanentError, kopf.TemporaryError): raise` runs BEFORE the broad except, so kopf.PermanentError (OP-07) and kopf.TemporaryError(delay=30) (OP-11) propagate from the per-component loop UP to the kopf reconcile boundary unchanged. Two new tests (`test_handle_appstack_propagates_permanent_error_to_kopf_boundary` and `test_handle_appstack_propagates_temporary_error_to_kopf_boundary`) lock this contract end-to-end.

All 5 ROADMAP success criteria are now VERIFIED end-to-end (no longer "at function level"); 16/16 requirement IDs are SATISFIED end-to-end; the full operator test suite is green at 31/31 passing in 1.28s (+2 tests vs initial verification, no regressions, 1 narrowing refactor in scope).

The two remaining human-verification items (visual README inspection + live-cluster Phase 20 smoke) are non-blocking and routine for this phase — they were never escalated as goal-blockers in the initial verification. They remain noted for completeness but the phase goal is achieved without them.

---

*Verified: 2026-05-08*
*Re-verified: 2026-05-08 (post-closure d0b6d52)*
*Verifier: Claude (gsd-verifier)*

## VERIFICATION PASSED
