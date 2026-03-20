# Phase 1 Research: Plan Contract And YAML Translation

**Date:** 2026-03-20
**Status:** Complete

## Objective

Research what is needed to plan Phase 1 well: define a deterministic structured-plan contract, validate that contract against the existing `WekaAppStore` runtime, produce canonical YAML, and hand off to the current apply path without introducing a second execution model.

## Key Findings

### 1. The existing runtime contract is narrower than the current GUI apply logic

- The authoritative runtime shape is the `WekaAppStore` CRD in [crd.yaml](/Users/christopherjenkins/git/wekaappstore/weka-app-store-operator-chart/templates/crd.yaml) plus the operator behavior in [main.py](/Users/christopherjenkins/git/wekaappstore/operator_module/main.py).
- The GUI currently accepts multiple blueprint sources and mutates documents at apply time in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py), but that logic is permissive and duplicated.
- Phase 1 should target the runtime contract first, not preserve every quirk of the current apply helpers.

### 2. A typed structured plan should be narrower than full YAML

The plan contract should not be a generic YAML AST. It should represent exactly the information needed to generate one canonical `WekaAppStore` resource.

Recommended top-level plan fields:
- `request_summary`
- `blueprint_family`
- `namespace_strategy`
- `components`
- `prerequisites`
- `fit_findings`
- `unresolved_questions`
- `reasoning_summary`
- `normalization_warnings`

Recommended component fields:
- `name`
- `enabled`
- exactly one of `helm_chart` or `kubernetes_manifest`
- `depends_on`
- `target_namespace`
- `values`
- `values_files`
- `wait_for_ready`
- `readiness_check`

This mirrors the CRD and operator semantics closely enough to validate deterministically before YAML generation.

### 3. Validation must be split into layers

Phase 1 should treat validation as layered checks instead of one monolithic pass:

1. **Schema validation**
   - required top-level plan fields
   - required component fields
   - exactly one deployment method per component

2. **Repo and operator contract validation**
   - `appStack.components[]` is non-empty
   - component names are unique
   - `dependsOn` references existing component names
   - Helm chart config has required fields
   - readiness check types match supported operator values

3. **Normalization validation**
   - safe inferred defaults are permitted only where explicitly allowed
   - normalization emits warnings, never silent rewrites
   - explicit valid user intent is preserved

4. **Apply handoff validation**
   - compiler output is exactly one canonical `WekaAppStore`
   - output can be consumed by the existing apply path without planner-specific branches

### 4. The biggest code risk is duplicated apply behavior

The two GUI helpers in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py):
- `apply_blueprint_with_namespace()`
- `apply_blueprint_content_with_namespace()`

duplicate:
- YAML loading
- namespace override rules
- `WekaAppStore` custom-resource handling
- CRD scope checks
- built-in resource apply fallback

Phase 1 planning should include extracting a shared document-application service before planner-generated YAML is allowed to join the same flow.

### 5. Canonical YAML should optimize for determinism, not minimal diff from planner output

Given the Phase 1 context decisions, the compiler should:
- emit one `WekaAppStore` object
- preserve explicit valid values
- inject only runtime-relevant defaults
- maintain stable field ordering and shape for equivalent plans

This reduces prompt drift and makes testing practical.

## Recommended Implementation Shape

### Backend modules

Recommended new backend units inside `app-store-gui/webapp/`:
- `planning/models.py` for typed plan, validation, and compiler models
- `planning/validator.py` for layered validation
- `planning/compiler.py` for `WekaAppStore` translation
- `planning/apply_gateway.py` for shared apply handoff logic

This is compatible with the repo’s current single-entrypoint style while still isolating the new contract logic.

### Compiler target

The compiler should output a Python dict representing:
- `apiVersion: warp.io/v1alpha1`
- `kind: WekaAppStore`
- `metadata`
- `spec.appStack`

YAML should be rendered from that dict as the final preview artifact.

### Normalization boundaries

Safe normalization candidates:
- default CR namespace when contract allows fallback
- Helm release name fallback to component name
- omitted `enabled` defaulting to `true`
- omitted `waitForReady` defaulting to `true`

Do not silently normalize:
- unsupported blueprint family
- missing deployment method
- invalid `dependsOn`
- missing install-critical values
- conflicting namespace intent

## Testing Implications

The repo currently has no Python test framework configured. Phase 1 will likely need Wave 0 work for:
- `pytest`
- a `tests/` tree
- shared fixtures for plan payloads and expected YAML

Highest-value first tests:
- valid plan -> canonical YAML
- invalid plan -> deterministic error set
- normalization with warning payload
- plan preserving explicit namespace and component intent
- handoff into shared apply gateway without altering runtime contract

## Validation Architecture

Nyquist-relevant validation strategy for this phase:

- Treat typed plan validation and compiler determinism as the core automated verification surface.
- Add a fast unit-style suite that validates plan parsing, contract rejection, normalization warnings, and YAML compilation.
- Add one integration-style seam test around the shared apply gateway to prove canonical compiler output still routes through the existing `WekaAppStore` apply path.
- Keep Helm/operator runtime validation out of full end-to-end scope for this phase except where needed to assert contract compatibility.

## Open Planning Risks

- `phase_req_ids` were not populated by the GSD init tool, so planning should rely on the roadmap and requirements traceability explicitly.
- The repo has no existing test harness, so planner work must account for test-infrastructure bootstrapping.
- The current GUI file is large and imperative, so extraction work should stay narrowly scoped to Phase 1 boundaries.

## Planning Recommendations

1. Start with typed plan models and layered validator behavior.
2. Build the compiler from typed plan -> canonical `WekaAppStore` dict -> YAML.
3. Refactor the current file/string apply duplication behind one shared handoff.
4. Add tests for contract validity, normalization warnings, and compiler output.
5. Keep chat, NemoClaw orchestration, and capacity tooling out of scope for this phase.

---
*Phase 1 research completed: 2026-03-20*
