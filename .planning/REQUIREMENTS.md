# Requirements: WEKA App Store v5.0 — AppStack Variable Substitution

**Defined:** 2026-05-06
**Core Value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.
**Milestone Goal:** Add `spec.appStack.variables` to the `WekaAppStore` CR. Operator performs `${VAR}` substitution over `kubernetesManifest:` strings and `valuesFiles:` content (loaded from ConfigMaps/Secrets) before they are applied or merged into Helm values. Blueprint authors get one-CR portability across namespaces and environments without external pre-render tooling.
**Source PRD:** `.planning/PRD-appstack-variable-substitution.md`
**Source Research:** `.planning/research/SUMMARY.md`

## v1 Requirements

Requirements for milestone v5.0. Each maps to exactly one roadmap phase.

### OPERATOR

Render helper and wiring inside `operator_module/main.py`.

- [ ] **OP-01**: `render(text, variables)` pre-scan guard returns text unchanged when no `${...}` pattern is present (CRITICAL backward-compat — without this, existing `cluster_init/app-store-cluster-init.yaml` shell scripts containing bare `$CRDS`/`$CRD`/`$MISSING`/`$GATEWAY_API_URL` would `KeyError` on first reconcile after upgrade)
- [ ] **OP-02**: `render()` uses `string.Template.substitute()` strict mode, not `safe_substitute()` — undefined references must fail loudly
- [ ] **OP-03**: `render()` catches BOTH `KeyError` AND `ValueError` and re-raises a descriptive error naming the variable and component context
- [ ] **OP-04**: `render()` preserves `$$` literal-dollar escape semantics (stdlib behavior; lock with unit test)
- [ ] **OP-05**: `render()` returns text unchanged when the variables dict is `None` or empty (backward-compat)
- [ ] **OP-06**: `handle_appstack_deployment` builds the variables dict once at stack scope as `{'namespace': cr_namespace, **(spec.appStack.variables or {})}` before the component loop
- [ ] **OP-07**: `kubernetesManifest` strings are rendered before `kubectl apply`; render failures raise `kopf.PermanentError` whose message names the variable and component
- [ ] **OP-08**: `load_values_from_reference` renders the raw ConfigMap data string and the base64-decoded Secret string before `yaml.safe_load`
- [ ] **OP-09**: `load_values_from_reference` signature uses `variables=None` default; the `handle_helm_deployment` single-chart call site (`main.py:~885`) is NOT wired with variables
- [ ] **OP-10**: Variable key names are validated as Python identifiers when the dict is built; invalid keys (e.g., `my-host`) raise `kopf.PermanentError` early instead of cryptic `ValueError` from `Template.substitute`
- [ ] **OP-11**: `load_values_from_reference` fetch failures (ConfigMap/Secret missing or API error) surface as `kopf.TemporaryError(delay=30)` instead of silent `{}` return
- [ ] **OP-12**: `@kopf.on.update` decorator gets `field='spec'` filter to prevent reconcile storms triggered by the operator's own status patches

### CRD

CRD schema additions in `weka-app-store-operator-chart/templates/crd.yaml`.

- [ ] **CRD-01**: `spec.appStack.variables` added as an optional map under `spec.versions[0].schema.openAPIV3Schema.properties.spec.properties.appStack.properties`
- [ ] **CRD-02**: Schema `description` documents `${VAR}` syntax, `$$` escape, `${namespace}` auto-default, and the identifier-name requirement
- [ ] **CRD-03**: `additionalProperties: { type: string }` enforces string-only values at admission (no `x-kubernetes-preserve-unknown-fields`)

### VALIDATOR

Soft-warning UX in `mcp-server/tools/validate_yaml.py`.

- [ ] **VAL-01**: Validator accepts a `spec.appStack.variables` block without raising a spurious schema error
- [ ] **VAL-02**: Validator soft-warns on hardcoded `*.svc.cluster.local` DNS literals inside `kubernetesManifest` strings, suggesting a `${VAR}` replacement
- [ ] **VAL-03**: Validator soft-warns on inline `namespace: <literal>` lines inside `kubernetesManifest` when the literal differs from `metadata.namespace`, suggesting `${namespace}`
- [ ] **VAL-04**: Validator errors (not soft-warns) on invalid variable key names that do not match `[_a-zA-Z][_a-zA-Z0-9]*`
- [ ] **VAL-05**: Validator errors (not soft-warns) on non-string variable values

### TEST

Test scaffolding and coverage. `operator_module/tests/` does not currently exist.

- [ ] **TST-01**: `operator_module/tests/__init__.py` + `operator_module/tests/test_render.py` with unit coverage for pre-scan guard, `$$`, JSON-safety, undefined-variable error, malformed-placeholder error, and no-op when variables dict is empty/None
- [ ] **TST-02**: `operator_module/tests/test_appstack.py` covers `handle_appstack_deployment` substitution behavior — manifest path, `valuesFiles` path, `${namespace}` auto-default, explicit override
- [ ] **TST-03**: Backward-compat snapshot test — an existing AppStack fixture without `variables:` produces byte-identical merged values dict and manifest tempfile content pre/post change
- [ ] **TST-04**: `mcp-server/tests/fixtures/sample_blueprints/ai-research-portable.yaml` fixture demonstrating `${namespace}` and `${milvusHost}` portable pattern
- [ ] **TST-05**: Test locks `handle_helm_deployment` non-wiring (`variables=None` passes through; substitution does not run on the single-chart path)

### DOCS

User-facing documentation in `README.md`.

- [ ] **DOC-01**: README section explaining `${VAR}` syntax with a worked example
- [ ] **DOC-02**: README documents `$$` literal-dollar escape with a password example
- [ ] **DOC-03**: README documents `${namespace}` auto-defaulting to the CR's `metadata.namespace`
- [ ] **DOC-04**: README documents strict failure on undefined references (`kopf.PermanentError` with named variable + component)
- [ ] **DOC-05**: README documents that variable values are NOT recursively resolved — the PRD's `milvusHost: milvus.${namespace}.svc.cluster.local` example does not work; documented examples must use fully-resolved values
- [ ] **DOC-06**: README documents that operator-control fields (`helmChart.*`, `releaseName`, `targetNamespace`, `readinessCheck.*`) are NOT templated, and recommends dropping `targetNamespace` so the existing namespace fallback chain handles it

### MIGRATION

End-to-end smoke test via the AIDP blueprint. Lives in the separate `aidp` repo and ships in a follow-up PR; tracked here so v5.0 is "really done" only when AIDP confirms the feature works in production.

- [ ] **MIG-01**: `aidp/appstack/weka-aidp-appstack.yaml` declares `spec.appStack.variables` with `milvusHost`, `postgresHost`, plus any other DNS literals (fully-resolved values, not cross-referencing)
- [ ] **MIG-02**: 17 inline `namespace: rag` literals across `kubernetesManifest` blocks → `namespace: ${namespace}`
- [ ] **MIG-03**: PV/PVC `claimRef.namespace: rag` → `${namespace}`
- [ ] **MIG-04**: `aidp/appstack/aidp-site-config.yaml` DNS literals replaced — `milvus.rag.svc.cluster.local` → `${milvusHost}`, etc.
- [ ] **MIG-05**: `kubectl apply -f appstack/weka-aidp-appstack.yaml` with `metadata.namespace: aidp-test` deploys cleanly into `aidp-test` with no other file changes (acceptance evidence captured as command output)

## v2 Requirements

Acknowledged but deferred to a future release.

### V51

- **V51-01**: Allow `${VAR}` substitution in `targetNamespace` (PRD Open Q2). Workaround for v5.0: drop `targetNamespace` and rely on the existing fallback chain. Reconsider if customers ask.
- **V51-02**: Optional `status.conditions[type=VariablesResolved]` boolean condition for observability (PRD Open Q3 partial). Do NOT publish resolved-values map — sensitive content.
- **V51-03**: Default-value syntax (e.g., `${VAR:-default}`) — Python `string.Template` does not support this natively; would require subclass.

## Out of Scope

Explicitly excluded for v5.0. Documented to prevent scope creep and to make AIDP authoring expectations explicit.

| Feature | Reason |
|---------|--------|
| Recursion into inline `component.values:` objects | At the wiring point the values block is already parsed YAML, not a raw string. Workaround: route substitution-bearing values through `valuesFiles:`. |
| Templating of operator-control fields (`helmChart.*`, `releaseName`, `targetNamespace`, `readinessCheck.*`) | These are operator-control fields; templating invites surprising behavior. Add explicitly per-field if ever needed. |
| Conditionals, loops, or full template engines (Jinja, Go templates, sprig) | Substitution-only is the entire point; richer templating belongs in a wrapper Helm chart. |
| Cross-component variable references (one component's output → another's input) | Out of scope; users compose with shared stack-level variables instead. |
| Variable resolution from external sources (Vault, AWS Secrets Manager, env vars) | Variables are static strings declared in the CR. |
| Variables in `spec.appStack.components[].dependsOn` arrays | Hardcoded; templating component identity creates ordering ambiguity. |
| Recursive variable resolution (`${a}` in the value of `${b}`) | `string.Template.substitute()` is single-pass. Variable values are taken literally. AIDP migration uses fully-resolved values. |
| Resolved-variables map exposed in `status.appStackVariables` | Variables can be sensitive; status is broadly readable. Failure observability already lands in `componentStatus[].message`. |
| Backward-incompatible changes to existing CRs | Hard gate: any CR without `variables:` must produce byte-identical Helm values, manifest tempfile content, and `kubectl apply` invocations as before. |

## Traceability

Mapping of requirements to roadmap phases. Populated by the roadmapper.

| Requirement | Phase | Status |
|-------------|-------|--------|
| OP-01 | Phase 16 | Pending |
| OP-02 | Phase 16 | Pending |
| OP-03 | Phase 16 | Pending |
| OP-04 | Phase 16 | Pending |
| OP-05 | Phase 16 | Pending |
| OP-06 | Phase 18 | Pending |
| OP-07 | Phase 18 | Pending |
| OP-08 | Phase 18 | Pending |
| OP-09 | Phase 18 | Pending |
| OP-10 | Phase 18 | Pending |
| OP-11 | Phase 18 | Pending |
| OP-12 | Phase 18 | Pending |
| CRD-01 | Phase 17 | Pending |
| CRD-02 | Phase 17 | Pending |
| CRD-03 | Phase 17 | Pending |
| VAL-01 | Phase 19 | Pending |
| VAL-02 | Phase 19 | Pending |
| VAL-03 | Phase 19 | Pending |
| VAL-04 | Phase 19 | Pending |
| VAL-05 | Phase 19 | Pending |
| TST-01 | Phase 16 | Pending |
| TST-02 | Phase 18 | Pending |
| TST-03 | Phase 18 | Pending |
| TST-04 | Phase 19 | Pending |
| TST-05 | Phase 18 | Pending |
| DOC-01 | Phase 18 | Pending |
| DOC-02 | Phase 18 | Pending |
| DOC-03 | Phase 18 | Pending |
| DOC-04 | Phase 18 | Pending |
| DOC-05 | Phase 18 | Pending |
| DOC-06 | Phase 18 | Pending |
| MIG-01 | Phase 20 | Pending |
| MIG-02 | Phase 20 | Pending |
| MIG-03 | Phase 20 | Pending |
| MIG-04 | Phase 20 | Pending |
| MIG-05 | Phase 20 | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-06*
*Last updated: 2026-05-06 — traceability filled in by roadmapper (Phases 16-20)*
