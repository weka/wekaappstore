# PRD: AppStack Variable Substitution (`spec.appStack.variables`)

**Project:** WEKA App Store Operator
**Date:** 2026-05-06
**Status:** Draft for GSD intake
**Primary Goal:** Add a `spec.appStack.variables:` map to the `WekaAppStore` CR. The operator performs a single `${VAR}` substitution pass over `kubernetesManifest:` strings and over `valuesFiles:` content (loaded from ConfigMaps/Secrets) before they are applied or merged into Helm values. This makes an AppStack blueprint portable across namespaces and environments without external pre-render tooling.

## Problem Statement

Today the `WekaAppStore` CR has zero string-interpolation support. Every component must hardcode every namespace, DNS name, and environment-specific value as a literal string. This is fine for a single-tenant first-party install but fails as soon as a blueprint needs to be:

- Deployed by a customer into a non-default namespace
- Reused across dev/staging/prod with different host names
- Released as a public AppStore catalog entry where the install namespace is unknown at authoring time

Concrete symptom from the AIDP blueprint (`/Users/christopherjenkins/git/aidp/appstack/weka-aidp-appstack.yaml`): until the cleanup commit on 2026-05-06, the literal namespace `rag` appeared 30+ times across the AppStack file (per-component `targetNamespace`, `readinessCheck.namespace`, and inside every `kubernetesManifest:` string for Secrets, ConfigMaps, PVCs, and a PV `claimRef.namespace`). The cleanup removed the per-component `targetNamespace` and `readinessCheck.namespace` literals — those now correctly fall back to the CR's `metadata.namespace` via the existing operator fallback chain (`operator_module/main.py:657-665`). What remains stuck:

1. **17 `namespace: rag` literals inside `kubernetesManifest:` strings.** The operator passes `kubectl apply -n <targetNamespace>` (`main.py:746`), but resources with explicit `namespace:` in their YAML override the `-n` flag. So Secrets, the realm ConfigMap, PVCs, and the PV's `claimRef.namespace` stay in `rag` regardless of CR namespace.
2. **All cross-service DNS literals inside `aidp-site-config.yaml`.** Strings like `milvus.rag.svc.cluster.local:19530` and `space-manager-postgres.rag.svc.cluster.local:5432` live inside ConfigMap data that the operator loads via `valuesFiles:` (`main.py:352-369`). Loaded as parsed YAML and deep-merged. No substitution.

Customers currently have to do a search-and-replace on `rag` across two files before applying. That is the opposite of "one-click from the App Store."

## Product Outcome

Authors of `WekaAppStore` blueprints can declare:

```yaml
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
            namespace: ${namespace}
          stringData:
            SM_POSTGRES_DSN: "postgresql+asyncpg://space_manager:Pass@${postgresHost}:5432/space_manager"
```

…and a customer changes exactly one field (`metadata.namespace`) to deploy the same blueprint anywhere. Every `${VAR}` resolves before the manifest hits the API server or before values are merged into a Helm release.

## Users

### Primary Users
- Blueprint authors (WEKA platform engineers shipping AIDP, RAG, and partner AppStacks)
- Customers deploying a WEKA App Store blueprint into a non-default namespace
- WEKA field engineers customizing site values per deployment

### Secondary Users
- Partner engineers contributing AppStack blueprints to the catalog
- The `mcp-server` validator (`mcp-server/tools/validate_yaml.py`) which today flags hardcoded namespaces — it can soft-promote `${...}` references as the recommended replacement

## In Scope

- New optional `spec.appStack.variables:` map on the `WekaAppStore` CRD: `additionalProperties: { type: string }`
- `${VAR}` substitution syntax via Python `string.Template`
- Substitution applied to:
  - `component.kubernetesManifest` strings (in `handle_appstack_deployment`, before `kubectl apply`)
  - The raw string returned by `load_values_from_reference` (ConfigMap or Secret), before `yaml.safe_load`
- Auto-default `${namespace}` to the CR's `metadata.namespace` when the user has not set it explicitly
- Strict failure (raise `kopf.PermanentError`) on undefined `${VAR}` references — surfaces typos at admission, not silently at runtime
- Updated CRD `openAPIV3Schema` in `weka-app-store-operator-chart/templates/crd.yaml`
- Unit tests in `mcp-server/tests/` (validator) and a new `operator_module/tests/` (substitution)
- A short user-facing doc in `README.md` describing the syntax, the auto-default, and the `$$` literal-dollar escape

## Out of Scope

- Recursive substitution into inline `component.values:` objects. Workaround: put substitution-bearing values in a ConfigMap and reference via `valuesFiles:` (which IS rendered). Rationale: walking arbitrary YAML adds non-trivial code; users authoring inline `values:` can resolve before authoring.
- Conditional logic, loops, or full template engines (Jinja, Go templates). This PRD is *substitution only*.
- Substitution into `helmChart.repository`, `helmChart.name`, `helmChart.version`, `releaseName`, `targetNamespace`, `readinessCheck.*`. These are operator-control fields; templating them invites surprising behavior. If needed in the future, add explicitly per-field.
- Cross-component variable references (e.g. one component's output becoming another's input). Out of scope; users compose with shared variables instead.
- Variable resolution from external sources (Vault, AWS Secrets Manager, env vars). Variables are static strings declared in the CR.
- Variables in `spec.appStack.components[].dependsOn` arrays. Hardcoded.
- Backward-incompatible changes: any existing CR without `variables:` MUST behave identically.

## Existing System Facts This PRD Locks In

Confirmed by direct inspection of the operator code (commit at `/Users/christopherjenkins/git/wekaappstore` HEAD):

- **CRD path:** `weka-app-store-operator-chart/templates/crd.yaml`. Schema location: `spec.versions[0].schema.openAPIV3Schema.properties.spec.properties.appStack`.
- **Reconciler entry point:** `operator_module/main.py:551 — handle_appstack_deployment(body, spec, name, namespace, status, **kwargs)`. The CR's namespace arrives as the `namespace` kwarg from kopf.
- **Helm-component branch:** `main.py:616-725`. Namespace fallback chain (`main.py:657-665`):
  ```
  component.targetNamespace
  → component.namespace            (alias, undocumented)
  → spec.appStack.namespaces[name] (per-component map, undocumented)
  → spec.appStack.defaultNamespace (stack-wide default, undocumented)
  → CR's metadata.namespace
  ```
- **Manifest-component branch:** `main.py:727-758`. Reads `component.kubernetesManifest` raw, writes to a tempfile, runs `kubectl apply -f <tempfile> -n <targetNamespace>`. Resources with explicit `namespace:` in the YAML ignore the `-n` flag.
- **Values loading from ConfigMap/Secret:** `main.py:352-369 — load_values_from_reference()`. Returns parsed YAML (dict). Deep-merged into the inline `values:` at `main.py:667-678`.
- **Helm install:** `helm_operator.install_or_upgrade()` (`main.py:40-100+`) writes the merged values dict to a tempfile and shells out to `helm`.
- **No existing templating:** `grep -E 'template|substitut|render|jinja|\${'` in `operator_module/` returns zero matches.
- **CR namespace is the only field that today thread-routes through every component.** Removing per-component `targetNamespace` literals (as AIDP did 2026-05-06) makes the CR namespace the natural sole source of truth for Helm components — but cannot reach the two stuck locations (manifest strings, ConfigMap values content) without this PRD.

## Proposed Solution

### Substitution Syntax

`${VAR}` via Python `string.Template`. Rationale:

- **JSON-safe.** `string.Template` only matches `$identifier` and `${identifier}`. It leaves `{...}` alone. AIDP's `aidp-bootstrap-secrets` contains literal Docker config JSON (`{"auths": {"nvcr.io": {…}}}`) — Python's `str.format(**vars)` would crash on every `{`.
- **Stdlib.** No new dependency.
- **`$$` literal-dollar escape.** Documented behavior of `string.Template`. Users with passwords containing `$` write `$$`.
- **`Template.substitute` (strict).** Raises `KeyError` on undefined keys. The operator catches and re-raises as `kopf.PermanentError` with a message naming the offending variable and the component. Choosing strict over `safe_substitute` so typos surface at apply time, not as cryptic Kubernetes errors hours later.

### CRD Change

`weka-app-store-operator-chart/templates/crd.yaml`, under `spec.properties.appStack.properties` (sibling of `components`):

```yaml
variables:
  type: object
  description: |
    Map of variable name → string value. Substituted as ${VAR} into:
      - kubernetesManifest strings
      - The raw content of ConfigMap/Secret referenced by valuesFiles
    The variable ${namespace} auto-defaults to the CR's metadata.namespace
    if not explicitly set. Use $$ to escape a literal dollar sign.
    Undefined references raise a permanent error.
  additionalProperties:
    type: string
```

### Operator Change (`operator_module/main.py`)

Three discrete edits:

**1. New helper near top of file (~10 lines):**

```python
from string import Template

def render(text: str, variables: dict) -> str:
    """Substitute ${VAR} from variables. Raises KeyError on undefined keys."""
    if not variables or not text:
        return text
    return Template(text).substitute(variables)
```

**2. In `handle_appstack_deployment`, build the variables dict once (~line 555):**

```python
app_stack = spec['appStack']
components = app_stack.get('components', [])
# ${namespace} defaults to CR namespace; user-defined variables override.
variables = {'namespace': namespace, **(app_stack.get('variables') or {})}
```

**3. Thread `variables` through to two substitution sites:**

a) `load_values_from_reference()` — render the raw string before parsing:
```python
def load_values_from_reference(kind, name, key, namespace, variables=None):
    ...
    values_yaml = cm.data.get(key, "")              # or base64-decoded secret
    try:
        values_yaml = render(values_yaml, variables)
    except KeyError as e:
        raise kopf.PermanentError(
            f"Undefined ${{{e.args[0]}}} in {kind}/{name} key={key}")
    return yaml.safe_load(values_yaml) or {}
```
And update the call site at `main.py:672-677` to pass `variables=variables`.

b) `kubernetesManifest` branch (`main.py:727-746`):
```python
elif 'kubernetesManifest' in component and component['kubernetesManifest']:
    try:
        manifest_yaml = render(component['kubernetesManifest'], variables)
    except KeyError as e:
        raise kopf.PermanentError(
            f"Undefined ${{{e.args[0]}}} in component "
            f"{component['name']}.kubernetesManifest")
    target_namespace = component.get('targetNamespace', namespace)
    ...
```

### Validator change (`mcp-server/tools/validate_yaml.py`)

- Recognize `spec.appStack.variables` as a valid optional field.
- Soft-warn when a CR contains hardcoded `*.svc.cluster.local` strings or repeated `namespace: <literal>` lines inside `kubernetesManifest`, suggesting `${namespace}` as the replacement. Non-blocking informational.

## Acceptance Criteria

1. **Backward compatibility.** Every existing AppStack CR in `mcp-server/tests/fixtures/sample_blueprints/*.yaml` continues to deploy with no changes. CRs without `variables:` produce byte-identical Helm values, manifest tempfiles, and `kubectl apply` invocations as before this change. (Test: snapshot of merged values and rendered manifest string for a representative existing fixture.)
2. **Auto-default namespace.** A CR with `metadata.namespace: foo`, no `variables:`, and a manifest containing `namespace: ${namespace}` deploys resources to `foo`.
3. **Explicit override.** A CR with `variables: {namespace: bar}` and `metadata.namespace: foo` deploys `${namespace}`-tagged resources to `bar`.
4. **Substitution in valuesFiles.** A ConfigMap referenced via `valuesFiles:` containing `URL: "http://${milvusHost}:19530"` deep-merges into the Helm release with the resolved value, given `variables: {milvusHost: milvus.foo.svc.cluster.local}`.
5. **Strict failure on undefined.** A manifest with `${unset}` and no matching key raises `kopf.PermanentError` whose message names `unset` and the component. The CR enters `Failed` phase with a clear `componentStatus.message`.
6. **JSON content untouched.** A `kubernetesManifest:` containing the AIDP `dockerconfigjson` Secret (with literal `{"auths": {…}}` payload) renders without error and the JSON is byte-identical pre/post.
7. **`$$` escapes.** A value `passw0rd$$abc` survives substitution as `passw0rd$abc`.
8. **End-to-end smoke (manual).** AIDP's `weka-aidp-appstack.yaml` and `aidp-site-config.yaml` are migrated to use `${namespace}`, `${milvusHost}`, `${postgresHost}` (sample diff produced as part of the PR description). Applying the migrated CR with `metadata.namespace: aidp-prod` deploys every component into `aidp-prod` with no other field changes.

## Test Plan

- **Unit (new file `operator_module/tests/test_render.py`):** `render` over JSON, undefined key, empty string, `$$`, multi-occurrence, missing variables dict (no-op).
- **Unit (extend `operator_module/tests/test_appstack.py` if exists, else new):** `load_values_from_reference` with substitution; `handle_appstack_deployment` namespace auto-default; explicit override.
- **Validator (`mcp-server/tests/test_validate_yaml.py`):** valid `variables:` block accepted; soft-warning emitted on hardcoded `.svc.cluster.local` strings.
- **Fixture (`mcp-server/tests/fixtures/sample_blueprints/`):** add `ai-research-portable.yaml` showing the recommended pattern with `${namespace}`-based DNS.

## Risk and Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| String-Template strict mode breaks an existing CR that uses literal `$` in unquoted positions | Low | The feature is opt-in: substitution only runs when `appStack.variables` is set OR the CR uses `${namespace}` (always set via auto-default). Existing CRs that don't reference `${...}` are unaffected because no `${VAR}` patterns exist in their content. The `${namespace}` auto-default has no impact on CRs that never reference it. |
| User accidentally introduces `${}` patterns in JSON content (unlikely but possible) | Low | `string.Template` only matches `$identifier`, not bare `${}`. Documented in README. |
| Variables in Secret content leak through error messages | Medium | The `KeyError` only contains the variable name, never values. `kopf.PermanentError` message is hand-crafted. Audit before merge. |
| Performance regression on large `valuesFiles` ConfigMaps | Negligible | `string.Template` is O(n) regex; values content is < 1MB in practice. |
| Customers expect Jinja-style features (loops, conditionals) | Medium | README explicitly scopes the feature as "substitution only" and links to a wrapper-Helm-chart pattern for users who need richer templating. |

## Migration Path for AIDP

Once shipped, AIDP applies a follow-up commit to:

1. Add `spec.appStack.variables:` to `appstack/weka-aidp-appstack.yaml` with `milvusHost`, `postgresHost`, plus any others.
2. Replace 17 inline `namespace: rag` literals with `namespace: ${namespace}`.
3. Replace `pvc.claimRef.namespace: rag` with `${namespace}`.
4. Replace DNS literals in `appstack/aidp-site-config.yaml`:
   - `milvus.rag.svc.cluster.local` → `${milvusHost}`
   - `space-manager-postgres.rag.svc.cluster.local` → `${postgresHost}` (where applicable)
5. Verify `kubectl apply -f appstack/weka-aidp-appstack.yaml` with `metadata.namespace: aidp-test` deploys cleanly into `aidp-test` with no other file changes.

## Estimated Effort

| Item | Estimate |
|---|---|
| Operator code (`render` + 3 wire-up edits) | ~30 lines |
| CRD schema update | ~10 lines |
| Unit tests | ~80 lines |
| Validator soft-warning | ~30 lines |
| README user-facing doc | ~40 lines |
| AIDP migration follow-up | ~50 line diff across 2 files |

**Target:** single PR against `wekaappstore`, single follow-up PR against `aidp`.

## Open Questions

1. Should `${VAR}` also substitute into the Secret-backed `valuesFiles:` content (currently base64-decoded at `main.py:363`)? Initial design says yes — the decoded string is rendered before YAML parse, identical path as ConfigMap. Confirm during implementation.
2. Do we want to support `${namespace}` as the default *target* namespace for `helmChart` components, allowing manifests to literally reference `targetNamespace: ${namespace}`? Cleaner for some use cases but adds substitution to operator-control fields. Recommend deferring to v2.
3. Should the operator publish the resolved `variables` map onto `status.appStackVariables` for observability? Optional; useful for debugging but adds a status field.

---

*PRD owner:* Christopher Jenkins
*Codebase references:* `wekaappstore/operator_module/main.py`, `wekaappstore/weka-app-store-operator-chart/templates/crd.yaml`
*Triggering blueprint:* `aidp/appstack/weka-aidp-appstack.yaml` (AIDP v3.0)
