# Feature Research

**Domain:** `${VAR}` substitution feature for a Kopf-based Kubernetes operator CR (WEKA App Store v5.0)
**Researched:** 2026-05-06
**Confidence:** HIGH (table stakes, anti-features, edge-case decisions, open questions Q1/Q2/Q3);
MEDIUM (differentiator framing vs. Helm/Kustomize/ArgoCD)

---

## Feature Landscape

### Table Stakes (Users Expect These)

These are behaviors that any engineer familiar with Helm, Kustomize, Flux, or ArgoCD substitution
will expect by default. Shipping without them creates friction and support tickets.

| # | Feature | Why Expected | PRD Status | Complexity | Dependency |
|---|---------|--------------|------------|------------|------------|
| TS-1 | `${VAR}` syntax resolves variable values | Universal pattern from Bash envsubst, Flux postBuildSubstitutions, Kustomize. Engineers reach for this syntax automatically. | Covered | SMALL | operator-only |
| TS-2 | Undefined variable raises a clear, named error | Flux `StrictPostBuildSubstitutions` exists _because_ silent empty-string substitution causes obscure downstream failures. Users who have been burned by silent empty strings (e.g., a namespace field becoming `""` and failing on server-side validation 10 minutes later) strongly prefer fail-fast. | Covered (strict `KeyError` → `kopf.PermanentError`) | SMALL | operator-only |
| TS-3 | Error message names the specific variable and the component it came from | Without this, debugging a failed CR requires reading YAML manually to find the typo. Helm's `--debug` flag addresses this; Kustomize's replacements transformer names the source field. | Covered (PRD `f"Undefined ${{{e.args[0]}}} in component {comp_name}.kubernetesManifest"`) | SMALL | operator-only |
| TS-4 | `$$` escapes a literal dollar sign | `string.Template` stdlib behavior. Any blueprint with a password, a Bash script, or a Docker registry URL that contains `$` will break without escape support. AIDP's `dockerconfigjson` Secret does not need this today but the pattern is expected. | Covered | SMALL | operator-only |
| TS-5 | CRD schema declares the `variables` field with type constraints | Without a schema entry, Kubernetes API server may reject the CR or strip unknown fields depending on pruning settings. Also required for `kubectl explain` to surface the field. | Covered (CRD `additionalProperties: { type: string }`) | SMALL | CRD-only |
| TS-6 | Existing CRs without `variables:` continue to work identically | Backward-compatibility is table stakes for any operator change. Silent regression = customer outage. | Covered (substitution is a no-op when `variables` is absent and no `${...}` patterns exist) | SMALL | operator-only |
| TS-7 | Validator accepts `variables:` block without error | Today `validate_yaml` passes schema checks for `helmChart`/`appStack`/`image`. A CR with a new `variables:` field must not get a spurious validation error or the OpenClaw agent will loop on false failures. | Covered (validator change in PRD) | SMALL | operator-only |
| TS-8 | User-facing doc covers syntax, escape, auto-default, and strict-failure | Without docs, blueprint authors write Jinja-style `{{ }}` or `$VAR` (no braces) and are confused when it silently does nothing. Flux and Kustomize both have explicit callouts for this. | Covered (README section) | SMALL | docs-only |
| TS-9 | Substitution applies to both `kubernetesManifest` strings and `valuesFiles` content | Users mentally model "variables substitute everywhere data appears." If it works in manifests but not in ConfigMap-backed Helm values, authors will spend hours searching for why half the substitutions worked. | Covered (both substitution sites wired) | MEDIUM | operator-only |
| TS-10 | `${namespace}` auto-defaults to CR namespace without explicit declaration | The single most common variable. Requiring users to set `variables: { namespace: rag }` manually defeats the goal. Flux does this implicitly for cluster-level variables. Kustomize does this via replacements source fields. | Covered (auto-default `namespace → CR metadata.namespace`) | SMALL | operator-only |

**Table stakes not met (gap analysis):**

None of the above are unmet by the PRD. The implementation covers the full expected set for this
scoped feature.

---

### Differentiators (vs. Comparable Tooling)

The goal is not competition — it is **intuitive feel** for operators of Helm/Kustomize/ArgoCD/Crossplane.
Each item below names what users from those tools will notice.

| # | Feature | vs. Comparable Tool | Our Approach | User Impact | Complexity | Dependency |
|---|---------|---------------------|--------------|-------------|------------|------------|
| D-1 | Variables live inside the CR itself, no external resource required | Kustomize replacements require a `ConfigMapGenerator` or patch source; Flux requires a separate `ConfigMap`/`Secret` in `.substituteFrom`. ArgoCD parameter overrides require a separate Application resource edit. | `spec.appStack.variables:` is inline in the CR — one file, one apply, done. | Blueprint authors shipping a self-contained CR appreciate not having to coordinate a separate ConfigMap. Field engineers doing site-specific deploys don't need cluster pre-work. | SMALL | CRD-only |
| D-2 | Default-value syntax intentionally NOT supported | Flux supports `${var:=default}`. Kustomize v2 had `$(VAR)` with defaults. | `string.Template` does not support `${var:=default}`. The operator provides `${namespace}` as the only auto-default; all other undefined variables are hard errors. | This is a "worse by design" differentiator but defensible: default values hide typos. The one legitimate use case (namespace) is covered by the auto-default. If this proves painful, add per-variable `default:` metadata to the CRD schema in v6.0 rather than adopting envsubst syntax. |  SMALL (intentional omission) | operator-only |
| D-3 | JSON-safe by design | Helm's `{{ }}` template syntax collides with JSON in multiline values (a known and frequently-reported pain). Kustomize replacements operate on parsed YAML fields, not raw strings, so JSON is never a collision. Flux `envsubst` replaces `$VAR` not `${VAR}` with braces in JSON — but only works in string context. | `string.Template` only matches `$identifier` and `${identifier}`. Literal `{"auths": {}}` JSON is untouched. This is why `string.Template` was chosen over `str.format`. | Zero breakage on AIDP's `dockerconfigjson` Secret (the hardest real-world case). Blueprint authors with Docker config or Prometheus rule JSON don't need special escaping. | SMALL | operator-only |
| D-4 | Strict failure surfaced as `kopf.PermanentError` with named variable + component | Kustomize replacement failures are often confusing; Flux strict mode (behind a feature gate) only recently became default; ArgoCD param override failures surface as sync errors without variable-level detail. | Error message format: `"Undefined ${unset} in component aidp-bootstrap-secrets.kubernetesManifest"`. Enters `Failed` phase immediately, no retry. | Blueprint authors see exactly which variable is undefined and in which component — no log diving required. | SMALL | operator-only |
| D-5 | Substitution in Secret-backed `valuesFiles` content | Kustomize does not template Secret data (it replaces values in Kubernetes manifests, not raw content). Flux substitutes from Secrets as sources but not into Secret content. ArgoCD does not template Secret values at all. | We decode the Secret, apply `render()`, then parse YAML — identical pipeline to ConfigMap. | Lets blueprint authors store site-specific Helm values with secrets-class sensitivity (RBAC-gated) while still benefiting from variable substitution. | SMALL | operator-only |

---

### Anti-Features (Excluded by PRD — Defensibility Assessment)

Each exclusion is evaluated: is the PRD rationale sound, or does excluding it materially harm UX?

| # | Feature | PRD Rationale | Defensibility | UX Harm if Excluded | Phase |
|---|---------|---------------|---------------|---------------------|-------|
| AF-1 | Recursion into inline `component.values:` objects | Walking arbitrary YAML dicts adds non-trivial code; inline `values:` can be resolved at authoring time; workaround exists (`valuesFiles:` via ConfigMap). | **Defensible.** The workaround is low-friction for field engineers (one ConfigMap apply). Inline values are typically static (image tags, replica counts) not namespace-variable content. Only problematic if an author wants to substitute a namespace-derived DNS name into an inline Helm value — the AIDP case shows this is real but the workaround is acceptable. | LOW — workaround is one ConfigMap away. MEDIUM if this blocks migration of Helm-only blueprints that can't use valuesFiles. | Deferred to v6.0 if workaround proves insufficient. |
| AF-2 | Conditionals, loops, Jinja, Go templates | Out-of-scope by design: "substitution only." Adding conditional logic would require a template engine dependency (Jinja2, Chevron, Go text/template) and full template rendering semantics. | **Strongly defensible.** The explicit niche of this feature is portability (namespace, DNS names). Operators who need conditionals have Helm chart templating available. Adding conditionals to operator CRs creates a "half a template engine" that satisfies nobody. | LOW — users who need loops/conditionals should be authoring a Helm chart, not a raw manifest CR. Document this clearly in README. |
| AF-3 | Substitution into operator-control fields (`helmChart.*`, `releaseName`, `targetNamespace`, `readinessCheck.*`) | "These are operator-control fields; templating them invites surprising behavior." | **Defensible with nuance.** `targetNamespace: ${namespace}` in particular is a legitimate ask (see Q2 below). However, allowing `helmChart.version: ${appVersion}` creates a footgun where the chart cannot be looked up until runtime. The PRD correctly defers this per-field — if `targetNamespace` is the only ask, add it explicitly in v5.1, not generically now. | MEDIUM specifically for `targetNamespace`. LOW for all others. The Q2 analysis below recommends a targeted v5.1 addition. |
| AF-4 | Cross-component variable references (output of one component as input to another) | Would require a two-pass reconciliation model where component A's output (e.g. a generated ServiceAccount name) populates a variable for component B. | **Strongly defensible.** This is Crossplane composition territory. Adding output-to-input wiring would require the operator to marshal `kubectl get` output into a variable store between components — significant architectural complexity. | LOW for v5.0. Users with cross-component references should use shared `variables:` (author the value directly), not dynamic references. |
| AF-5 | External variable sources (Vault, env vars, AWS Secrets Manager) | Variables are static strings declared in the CR. External resolution would require sidecar/init-container patterns or external-secrets operator integration. | **Strongly defensible.** The operator has no Vault client, no AWS SDK. Adding external sources is an integration project, not a substitution feature. | LOW — users with external secrets management already have ExternalSecretOperator syncing into cluster Secrets, which can already be referenced via `valuesFiles: [{kind: Secret, ...}]`. |
| AF-6 | Variables in `dependsOn` arrays | Hardcoded. Component names are structural identifiers, not environment-variable content. | **Strongly defensible.** A `dependsOn: ["${dbComponent}"]` pattern makes dependency graphs dynamic and breaks topological sort validation. No real-world use case requires this. | NEGLIGIBLE. |
| AF-7 | Backward-incompatible changes | Any existing CR without `variables:` must behave identically. | **Non-negotiable.** The entire WEKA App Store catalog, MCP server fixture set, and existing customer deployments depend on this. | N/A — this is a constraint, not a feature exclusion. |

---

## Feature Dependencies

```
[TS-5: CRD schema update]
    └──required-by──> [TS-1: ${VAR} syntax recognized by API server]

[TS-10: ${namespace} auto-default]
    └──enables──> [TS-2: Strict failure on truly undefined vars]
                  (auto-default means ${namespace} is never "undefined")

[TS-9: Both substitution sites wired]
    └──requires──> [TS-1: render() helper]
    └──requires──> [D-5: Secret-backed valuesFiles support]

[TS-7: Validator accepts variables block]
    └──enables──> [TS-8: Docs credible — validator will not contradict them]

[D-1: Inline CR variables]
    └──enables──> [AIDP migration: single-file portability]

[D-3: JSON-safe design]
    └──enables──> [TS-6: Backward compat on existing dockerconfigjson secrets]
```

### Dependency Notes

- CRD schema update (TS-5) must land in the same PR as the operator code changes (TS-1, TS-9) to avoid a window where the operator accepts `variables:` but the API server strips it.
- Validator change (TS-7) can land after operator/CRD if needed, but should be in the same PR to keep the validate-retry loop in SKILL.md consistent.
- AIDP migration is a follow-up PR (different repo), not a blocker for the operator PR.

---

## PRD Open Questions — Recommended Answers

### Q1: Should `${VAR}` substitute into Secret-backed `valuesFiles` content?

**PRD says:** "Initial design says yes... confirm during implementation."

**Recommendation: YES, confirm it.**

Rationale:
- The code path is identical to ConfigMap: base64-decode → raw string → `render()` → `yaml.safe_load()`. No additional code required.
- The security concern is log leakage of resolved secret _values_. The PRD already handles this correctly: the `KeyError` message in the `kopf.PermanentError` contains only the _variable name_, never the value. The rendered string (which may contain a resolved password) is passed directly to `yaml.safe_load()` and then into the Helm values tempfile — it is never logged.
- The `load_values_from_reference()` function currently logs `f"Error loading values from {kind}/{name}: {str(e)}"` on exception — `str(e)` for a `kopf.PermanentError` (which we re-raise) will contain the variable name, not any values. That is safe.
- One audit is needed before merge: confirm `logging.error(f"Error loading values from {kind}/{name}: {str(e)}")` at line 369 does not expose a resolved value. Since we re-raise `kopf.PermanentError` from inside `render()` (before `yaml.safe_load`), the only value that can appear in `str(e)` is the variable _name_ from `KeyError.args[0]`. Safe.
- **Implementation note:** the `render()` call for Secret-backed content should happen before any logging of the raw content. Current code logs nothing on the happy path, so there is no risk as written.

**Complexity:** SMALL. **Dependency:** operator-only.

---

### Q2: Should `${namespace}` be substitutable in `targetNamespace` for Helm components?

**PRD says:** "Recommend deferring to v2" (v5.1 in our numbering).

**Recommendation: DEFER, but document the cliff.**

Rationale:
- The PRD correctly identifies this as adding substitution to "operator-control fields." The risk is different from manifest/valuesFiles substitution: if `targetNamespace` resolves incorrectly, the Helm release lands in the wrong namespace with no error — a silent namespace escape.
- The current fallback chain (component.targetNamespace → component.namespace → namespaces map → defaultNamespace → CR namespace) already handles the most common case: dropping `targetNamespace` entirely causes the CR namespace to be used. That is functionally equivalent to `targetNamespace: ${namespace}` for the standard use case.
- The only case where `targetNamespace: ${namespace}` would add value is when a blueprint author needs to document the intent _explicitly_ in the CR. That is a documentation/readability concern, not a deployment concern.
- **Action for v5.0:** In the README section, explicitly note that `${namespace}` is NOT substituted into `targetNamespace` and explain that dropping `targetNamespace` achieves the same result. This prevents the "why doesn't this work?" question.
- **v5.1 addition** (after v5.0 is stable): add targeted `targetNamespace` substitution if field engineers report friction. Do NOT add it generically across all operator-control fields.

**Complexity:** SMALL to add in v5.1, NEGLIGIBLE to document in v5.0. **Dependency:** docs-only for v5.0.

---

### Q3: Should the operator publish `status.appStackVariables: {...}` with resolved values?

**PRD says:** "Optional; useful for debugging but adds a status field."

**Recommendation: NO for v5.0, with a specific reason.**

Rationale:
- Variables _values_ can be sensitive (DNS names, internal host addresses, and potentially — if a user miscategorizes something — credentials). Publishing them to `status` makes them readable by anyone with `kubectl get wekaappstore` RBAC access, which is a broader audience than the Secrets they came from.
- The industry standard for this class of risk is to NOT echo resolved values into status. External Secrets Operator, for example, exposes `status.conditions` (state machine) but does not echo resolved secret values. Helm's `helm get values` is access-gated.
- For debugging, the correct tool is `kubectl describe wekaappstore <name>` which surfaces the `conditions` and `componentStatus[].message` — both already written by the operator. A bad substitution (undefined var) surfaces as `Failed` with the variable name. That is sufficient for debugging without exposing values.
- **What IS useful to publish:** `status.conditions[type=VariablesResolved]` as a condition (True/False/Unknown) with a `reason` field (e.g. `AllVariablesResolved` or `UndefinedVariable`). This gives observability without value exposure.
- **v5.0 recommendation:** add `status.conditions[type=VariablesResolved]` only if the operator already writes structured conditions (it does — see line 569 in main.py, `'type': 'Ready'` condition pattern). If adding one condition is zero marginal complexity, do it. If it requires a new status subresource, defer to v5.1.

**Complexity:** SMALL for a condition boolean. MEDIUM for a full resolved-variables map (and not recommended). **Dependency:** operator-only.

---

## Migration UX — Before/After for AIDP

The single happy-path example from the PRD, to confirm the customer-facing diff is clean:

**BEFORE (hardcoded, not portable):**
```yaml
# weka-aidp-appstack.yaml
metadata:
  name: weka-aidp
  namespace: rag          # <-- customer must change this AND hunt every other occurrence
spec:
  appStack:
    components:
      - name: aidp-bootstrap-secrets
        kubernetesManifest: |
          apiVersion: v1
          kind: Secret
          metadata:
            name: space-manager-secrets
            namespace: rag   # <-- hardcoded (1 of 17)
          stringData:
            SM_POSTGRES_DSN: "postgresql+asyncpg://space_manager:Pass@space-manager-postgres.rag.svc.cluster.local:5432/space_manager"
                                                                                                # ^^^^^^^^^^^^^^^^^^^^ hardcoded DNS

# aidp-site-config.yaml (ConfigMap data)
milvus_host: "milvus.rag.svc.cluster.local:19530"
postgres_host: "space-manager-postgres.rag.svc.cluster.local:5432"
```

**AFTER (portable, single-field change to redeploy to any namespace):**
```yaml
# weka-aidp-appstack.yaml
metadata:
  name: weka-aidp
  namespace: aidp-prod    # <-- customer changes only this
spec:
  appStack:
    variables:
      milvusHost: milvus.${namespace}.svc.cluster.local
      postgresHost: space-manager-postgres.${namespace}.svc.cluster.local
    components:
      - name: aidp-bootstrap-secrets
        kubernetesManifest: |
          apiVersion: v1
          kind: Secret
          metadata:
            name: space-manager-secrets
            namespace: ${namespace}   # <-- resolves to aidp-prod
          stringData:
            SM_POSTGRES_DSN: "postgresql+asyncpg://space_manager:Pass@${postgresHost}:5432/space_manager"
                                                                       # ^^^^^^^^^^^^^^^ resolves to space-manager-postgres.aidp-prod.svc.cluster.local

# aidp-site-config.yaml (ConfigMap data — valuesFiles source)
milvus_host: "${milvusHost}:19530"
postgres_host: "${postgresHost}:5432"
```

Customer action to move from `rag` to `aidp-prod`: change `metadata.namespace: rag` to `metadata.namespace: aidp-prod`. Done.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Phase | Priority |
|---------|------------|---------------------|-------|----------|
| TS-1 `${VAR}` render helper | HIGH | LOW | Phase 1 — Operator Core | P1 |
| TS-5 CRD schema update | HIGH | LOW | Phase 1 — CRD Schema | P1 |
| TS-10 `${namespace}` auto-default | HIGH | LOW | Phase 1 — Operator Core | P1 |
| TS-2/TS-3 Strict failure + named error | HIGH | LOW | Phase 1 — Operator Core | P1 |
| TS-4 `$$` escape | MEDIUM | LOW | Phase 1 — Operator Core | P1 (free with `string.Template`) |
| TS-6 Backward compat | HIGH | LOW | Phase 1 — Operator Core | P1 (architectural constraint) |
| TS-9 Both substitution sites | HIGH | LOW | Phase 1 — Operator Core | P1 |
| D-5 Secret-backed valuesFiles (Q1 answer) | MEDIUM | LOW | Phase 1 — Operator Core | P1 (confirmed in Q1) |
| TS-7 Validator accepts `variables:` | MEDIUM | LOW | Phase 2 — Validator | P1 |
| TS-8 README docs | HIGH | LOW | Phase 3 — Docs | P1 |
| Validator soft-warn on hardcoded DNS | LOW | LOW | Phase 2 — Validator | P2 |
| `status.conditions[VariablesResolved]` (Q3) | LOW | LOW-MEDIUM | Phase 1 or defer | P2 |
| AIDP migration follow-up | HIGH (smoke test) | LOW | Phase 4 — Migration | P1 (validation) |
| `targetNamespace: ${namespace}` (Q2) | LOW-MEDIUM | LOW | v5.1 follow-up | P3 |

---

## Competitor Feature Analysis

This feature is not a competitive product — it is a parity/ergonomics feature. The comparison
is against tooling the target users already operate.

| Feature | Flux postBuildSubstitutions | Kustomize replacements | ArgoCD params | Our Approach |
|---------|---------------------------|----------------------|---------------|--------------|
| Syntax | `${var}` (envsubst) | source field → target field path | `--helm-set-string 'k=${ARGOCD_APP_NAME}'` | `${VAR}` (string.Template) |
| Undefined vars default | empty string (opt-in strict) | N/A — fields must exist | N/A — per-param | strict `KeyError` always |
| Default value syntax | `${var:=default}` | N/A | N/A | none (auto-default for `${namespace}` only) |
| Escape syntax | `$${var}` → `${var}` | N/A | N/A | `$$` → `$` |
| Variable source | inline + ConfigMap/Secret | ConfigMap generator | ApplicationSet params | inline CR only |
| Applies to Secret-backed content | as source, not target | no | no | yes (decoded → render → parse) |
| JSON safety | yes (string context) | yes (parsed YAML fields) | yes | yes (string.Template) |
| Operator-control field substitution | all manifest fields | all YAML fields | Helm values only | manifest + valuesFiles only |
| Strict-fail as default | no (behind feature gate) | no | no | yes |

**Key insight:** Our strict-fail-as-default is the most opinionated design choice. Flux took years
and a feature gate to get there because existing users relied on silent empty-string behavior.
We start strict from day one because we have zero existing users of the substitution feature.
This is the right call.

---

## Sources

- Flux Kustomization postBuildSubstitutions docs: https://fluxcd.io/flux/components/kustomize/kustomizations/
- Flux issue: Option to error on missing postBuild substitution variable: https://github.com/fluxcd/flux2/issues/4694
- Flux discussion: StrictPostBuildSubstitutions request (unanswered): https://github.com/fluxcd/flux2/discussions/4459
- Kustomize variable substitution issues: https://github.com/kubernetes-sigs/kustomize/issues/2052
- ArgoCD parameter overrides: https://argo-cd.readthedocs.io/en/stable/user-guide/parameters/
- Python `string.Template` stdlib: implicit (training data, HIGH confidence, stdlib unchanged since Python 3.0)
- kopf PermanentError behavior: https://kopf.readthedocs.io/en/latest/errors/
- Kubernetes operator observability best practices: https://sdk.operatorframework.io/docs/best-practices/observability-best-practices/
- CWE-209 sensitive data in error messages: https://cwe.mitre.org/data/definitions/209.html
- PRD source: `.planning/PRD-appstack-variable-substitution.md`
- Operator code inspected: `operator_module/main.py` (lines 352-369, 551-766)
- Existing validator inspected: `mcp-server/tools/validate_yaml.py`

---
*Feature research for: WEKA App Store v5.0 AppStack Variable Substitution*
*Researched: 2026-05-06*
