# Pitfalls Research: AppStack Variable Substitution

**Domain:** Adding `${VAR}` substitution to a Kopf-based Kubernetes operator
**Researched:** 2026-05-06
**Confidence:** HIGH — all pitfalls verified by executing the actual code paths against the real operator source

---

## Critical Pitfalls

### Pitfall 1: Auto-Default Destroys Backward Compatibility

**What goes wrong:** The PRD claims substitution is opt-in and "existing CRs without `variables:` produce byte-identical output." This claim is false as implemented. The proposed code builds `variables = {'namespace': namespace, **(app_stack.get('variables') or {})}` and then calls `render(text, variables)`. Because `variables` is always non-empty (it always contains at least `{'namespace': cr_namespace}`), the guard `if not variables` in `render()` is never True. `Template(text).substitute(variables)` runs on every `kubernetesManifest` string and every `valuesFiles` content — regardless of whether the CR uses `${...}` at all.

**Root cause:** The PRD conflates "user did not set `variables:`" with "variables dict is empty." The auto-default injection (`${namespace}` to CR namespace) is correct behavior, but it invalidates the backward-compat guard.

**Concrete break already in the repo:** `cluster_init/app-store-cluster-init.yaml` has shell scripts in multiple `kubernetesManifest:` blocks (lines 143-158) containing `$CRDS`, `$CRD`, `$MISSING`, `$GATEWAY_API_URL`. These are shell environment variables — not substitution references. `string.Template.substitute()` raises `KeyError('CRDS')` on the first one. This CR will fail to deploy the instant the operator upgrade is applied, with no user action required to trigger the break.

**Prevention:** Replace `if not variables` with a pre-scan for `${...}` patterns using `re.search(r'\$\{[^}]+\}', text)`. This makes render a pure no-op on texts that contain no `${...}` patterns, regardless of what is in the variables dict.

```python
_SUBST_RE = re.compile(r'\$\{[^}]+\}')

def render(text: str, variables: dict) -> str:
    if not text or not _SUBST_RE.search(text):
        return text  # fast path: no ${...} patterns present
    return Template(text).substitute(variables)
```

Shell scripts, PromQL, regex, env references all pass through untouched unless they contain `${...}` brace syntax. Note: bare `$IDENTIFIER` (without braces) is NOT a substitution pattern in this design — only `${IDENTIFIER}` is.

**Warning signs:** `KeyError` on an ALL-CAPS name in `kopf.PermanentError` message (shell env vars are uppercase by convention). Any component that embeds bash, PromQL series queries, or annotation values containing `$`.

**Phase to address:** Operator core — before any code is written. The guard logic is the foundation of backward compat.

---

### Pitfall 2: `$oauthtoken` in NVIDIA NGC Credentials Breaks the "JSON is Safe" Claim

**What goes wrong:** The PRD states "`string.Template` only matches `$identifier` and `${identifier}`. It leaves `{...}` alone." This is incomplete. The full truth is that `string.Template` leaves bare `{...}` alone but DOES match bare `$identifier` patterns — and `$oauthtoken` is a valid Python identifier. NVIDIA NGC's Docker credential format uses the literal string `$oauthtoken` as the username: `{"auths":{"nvcr.io":{"username":"$oauthtoken","password":"TOKEN"}}}`. If this JSON is stored in a `kubernetesManifest:` using `stringData:` (plaintext), `Template.substitute()` raises `KeyError('oauthtoken')`.

**Confirmed by experiment:** `Template('{"auths":{"nvcr.io":{"username":"$oauthtoken"}}}').substitute({'namespace': 'foo'})` raises `KeyError: 'oauthtoken'`.

**Root cause:** The PRD's "JSON-safe" reasoning only accounts for curly-brace interference, not for `$identifier` patterns embedded in JSON string values.

**Prevention (two layers):**
1. If the fast-path guard from Pitfall 1 is implemented (only run Template when `${...}` is present), this issue is fully mitigated — `$oauthtoken` without braces is a bare `$identifier` pattern which the `_SUBST_RE` fast-path does not match, so Template is never called on such content.
2. As a defense-in-depth migration strategy: store the `.dockerconfigjson` value as base64 in the Secret's `data:` field rather than plaintext in `stringData:`. Base64 character set (A-Za-z0-9+/=) contains no `$`, so there is zero collision regardless of engine choice.

**Warning signs:** A `kubernetesManifest` component creating a `kubernetes.io/dockerconfigjson` Secret using `stringData:` and NGC registry URL. Add a unit test: render the full AIDP `aidp-bootstrap-secrets` manifest with `variables = {'namespace': 'foo'}` and assert no exception and byte-identical output.

**Phase to address:** Operator core (fast-path guard covers this) + AIDP migration (prefer `data:` not `stringData:` for NGC secrets as defense-in-depth).

---

### Pitfall 3: Variable VALUES Containing `${namespace}` Are Not Recursively Resolved

**What goes wrong:** The PRD Product Outcome shows `variables: {milvusHost: "milvus.${namespace}.svc.cluster.local"}` as a recommended pattern. Users reading this assume `milvusHost` resolves to `milvus.rag.svc.cluster.local`. It does not. The operator builds the variables dict once — `{'namespace': 'rag', 'milvusHost': 'milvus.${namespace}.svc.cluster.local'}` — and substitutes into the TEXT. When `${milvusHost}` in the text is replaced, it becomes the literal string `milvus.${namespace}.svc.cluster.local`. The `${namespace}` inside the value is never resolved because `Template.substitute` is single-pass.

**Confirmed by experiment:** `Template('url: ${milvusHost}').substitute({'namespace': 'rag', 'milvusHost': 'milvus.${namespace}.svc.cluster.local'})` returns `'url: milvus.${namespace}.svc.cluster.local'` — the inner `${namespace}` is not resolved.

**Root cause:** `string.Template.substitute` is single-pass, not recursive. The PRD's usage example implies cross-variable resolution that the implementation does not provide.

**Prevention:** Either (a) implement a pre-resolution pass on the variables dict itself before substituting into text, or (b) remove the cross-referencing example from the PRD and README and require fully-resolved values: `milvusHost: milvus.rag.svc.cluster.local`. Option (b) is correct for v1. Cross-variable resolution is a v2 feature. Add a unit test asserting single-pass behavior so the limitation is intentional and documented.

**Warning signs:** A user reports their DNS names contain literal `${namespace}` after deployment. The AIDP migration must use fully-resolved variable values or use `${namespace}` directly in the text, not inside another variable's value.

**Phase to address:** Operator core + docs. README example must show fully-resolved variable values.

---

### Pitfall 4: Bare `$identifier` Patterns Raise `ValueError` (Not `KeyError`) — Catch Both

**What goes wrong:** When a manifest contains a malformed placeholder like `${}` or `${123}` (digit-start), `Template.substitute()` raises `ValueError: Invalid placeholder`, not `KeyError`. The proposed error handler only catches `KeyError` and re-raises as `kopf.PermanentError`. A `ValueError` propagates as a bare `Exception`, gets caught by the outer `except Exception as e` in the component loop (line 762), and produces `comp_status['message'] = "Error deploying component: Invalid placeholder in string: line X, col Y"` — no component name, no guidance on what to fix.

Also affects: `$[regex]` patterns in PromQL queries embedded in ConfigMaps (e.g., the Prometheus adapter config in `cluster_init/`) — `$[A-Z]+` raises `ValueError` not `KeyError`. However, the Pitfall 1 fast-path guard prevents this if the ConfigMap content contains no `${...}` patterns.

**Confirmed by experiment:** `Template('pattern: $[A-Z]+').substitute({'namespace': 'foo'})` raises `ValueError: Invalid placeholder in string: line 1, col 10`.

**Root cause:** `string.Template` has two failure modes (`KeyError` for undefined vars, `ValueError` for syntactically invalid placeholders), and the PRD's error handler only accounts for one.

**Prevention:** Catch both at the call sites where component name is in scope:

```python
try:
    manifest_yaml = render(component['kubernetesManifest'], variables)
except KeyError as e:
    raise kopf.PermanentError(
        f"Undefined ${{{e.args[0]}}} in component {comp_name}.kubernetesManifest")
except ValueError as e:
    raise kopf.PermanentError(
        f"Invalid placeholder syntax in component {comp_name}.kubernetesManifest: {e}")
```

**Warning signs:** `PermanentError` whose message contains "Invalid placeholder" without a variable name. Unit test: `render('$[A-Z]+', {'namespace': 'foo'})` where text contains `${foo}` (so fast-path activates) should raise a catchable error, not silently return.

**Phase to address:** Operator core.

---

### Pitfall 5: `load_values_from_reference` Silently Swallows Fetch Errors While Render Errors Are Permanent — Asymmetry

**What goes wrong:** Currently `load_values_from_reference` catches all exceptions and returns `{}`. After this change, a `KeyError` from `render()` inside `load_values_from_reference` is re-raised as `kopf.PermanentError`. But a transient Kubernetes API error (ConfigMap temporarily unavailable, network blip, Secret not yet created) is still caught and returns `{}` silently — it does not raise. This creates an asymmetry: a typo in a variable name causes the CR to enter permanent Failed state with no retries, while a missing ConfigMap silently no-ops and produces a Helm install with empty values. The Helm install may succeed with wrong configuration.

**Root cause:** The existing error handling in `load_values_from_reference` prioritizes leniency over correctness. Adding strict substitution on top of lenient fetch creates a confusing failure mode.

**Prevention:** Distinguish fetch errors from render errors:

```python
# In load_values_from_reference, after the fetch:
try:
    values_yaml = render(values_yaml, variables)
except KeyError as e:
    raise kopf.PermanentError(
        f"Undefined ${{{e.args[0]}}} in {kind}/{name} key={key}")
except Exception as fetch_e:
    logging.error(f"Error loading {kind}/{name}: {fetch_e}")
    raise kopf.TemporaryError(
        f"Could not fetch {kind}/{name}: {fetch_e}", delay=30)
```

`TemporaryError` causes kopf to retry with backoff — appropriate for transient Kubernetes API issues.

**Warning signs:** Operator log showing `Error loading values from ConfigMap/X: ...` followed by a successful but wrong Helm install. Unit test: mock `kr8s.NotFound` exception and assert `TemporaryError` is raised, not silent `{}` return.

**Phase to address:** Operator core.

---

## Moderate Pitfalls

### Pitfall 6: Variable Key Names With Hyphens or Leading Digits Produce `ValueError`

**What goes wrong:** CRD `additionalProperties: { type: string }` constrains values to strings but imposes no restriction on key names. A user who names a variable `my-host` (hyphen) and writes `${my-host}` in a manifest gets `ValueError: Invalid placeholder` from `string.Template`, not `KeyError`. The variable is declared but unreachable by the template engine. Python `string.Template` identifiers must match `[_a-zA-Z][_a-zA-Z0-9]*`.

**Root cause:** CRD schema permits arbitrary YAML map keys; `string.Template` is restricted to Python-valid identifiers.

**Prevention:** Add key-name validation in the operator when building the variables dict:

```python
_VALID_VAR_NAME = re.compile(r'^[_a-zA-Z][_a-zA-Z0-9]*$')
for k in (app_stack.get('variables') or {}):
    if not _VALID_VAR_NAME.match(k):
        raise kopf.PermanentError(
            f"Variable name '{k}' is not a valid identifier "
            f"(must match [_a-zA-Z][_a-zA-Z0-9]*)")
```

**Warning signs:** User reports `ValueError: Invalid placeholder` on a manifest that has no obvious bad placeholders. Unit test: variables dict with key `my-host` triggers PermanentError before any substitution.

**Phase to address:** Operator core + docs. Mention valid identifier requirement prominently in README.

---

### Pitfall 7: `$$` Escape Semantics Are Counter-Intuitive — Doc Strategy

**What goes wrong:** `string.Template` treats `$$` as an escape for a single literal `$`. So a YAML value `p$$w0rd` renders to `p$w0rd`. Users with a real password containing `$$` must write `p$$$$w0rd` to get `p$$w0rd`. Users who have a password `p$w0rd` and don't know about escaping will write it literally — `$w0rd` matches as a `$identifier` pattern, raising `KeyError('w0rd')`. The behavior is correct per the stdlib spec but non-obvious.

**Confirmed by experiment:** `Template('password: p$$word').substitute({'namespace': 'foo'})` returns `'password: p$word'`.

**Root cause:** The `$$` escape is standard Python `string.Template` behavior but is not widely known among Kubernetes operators.

**Prevention:** The README must include an explicit escape table:

| In YAML source | Rendered value |
|----------------|----------------|
| `p$$word` | `p$word` (one dollar — the intended result for most passwords) |
| `p$$$$word` | `p$$word` (two dollars) |
| `p$word` | `KeyError('word')` — must escape the `$` |

Recommend that `valuesFiles:` Secret content with literal `$` characters use the Secret's `data:` field (base64-encoded), which has no `$` and requires no escaping.

Unit tests needed: `render('p$$w0rd', {'namespace': 'x'}) == 'p$w0rd'` and `render('p$$$$w0rd', {'namespace': 'x'}) == 'p$$w0rd'`.

**Phase to address:** Docs + tests. No code change required — behavior is correct, only documentation and test coverage needed.

---

### Pitfall 8: Validator Soft-Warning on `*.svc.cluster.local` Triggers on Legitimate External Hostnames

**What goes wrong:** The PRD proposes a soft-warning when a CR contains hardcoded `*.svc.cluster.local` strings, suggesting `${namespace}` as a replacement. A customer whose cluster domain is not `cluster.local` (e.g., `cluster.internal`, common in GKE) or who has an intentional literal hostname will receive a spurious warning on a correctly-authored CR.

**Root cause:** The pattern `*.svc.cluster.local` is a reliable heuristic for in-cluster DNS but not universal.

**Prevention:** Tighten the warning message to be clearly advisory:

> "Informational: detected what appears to be an in-cluster service DNS name (`milvus.rag.svc.cluster.local`). If this namespace portion should vary per deployment, consider replacing it with `${namespace}`. Safe to ignore if intentional or if your cluster DNS domain differs from `cluster.local`."

Keep this strictly in the `warnings` list, never `errors`. Never block `valid=True` on this heuristic.

**Phase to address:** Validator.

---

### Pitfall 9: Variables Map Change Between Reconciles Orphans Resources in the Prior Namespace

**What goes wrong:** When a user changes `variables: {namespace: rag}` to `variables: {namespace: rag-prod}`, kopf fires `on.update` which calls `handle_appstack_deployment` with the new variables. The operator re-applies manifest components with namespace `rag-prod`. Resources previously created in `rag` remain — the operator has no record of prior deployment namespace and does not clean up. For `helmChart` components, Helm releases are scoped per-namespace; changing the namespace effectively creates a new release in the new namespace while the old release persists.

**Root cause:** The operator re-renders on every reconcile but has no "previous state" for cleanup. This is a pre-existing behavior amplified by a namespace variable.

**Prevention (scoped to this PRD):** Add a clear warning to the README: "Changing `variables.namespace` after initial deployment does not migrate or clean up resources in the prior namespace. Manual cleanup is required." Optionally record `status.appStackVariablesSnapshot` on each successful reconcile so users can see what the last applied values were. Full migration support is out of scope for v5.0.

**Phase to address:** Docs (immediate). Status snapshot field is optional enhancement.

---

### Pitfall 10: Bare `@kopf.on.update` Fires on Status Patches — Reconcile Storm Risk

**What goes wrong:** `@kopf.on.update` (line 1015) fires on any update to the CR, including status subresource patches made by the operator itself. When `handle_appstack_deployment` writes to `patch.status`, kopf issues a status patch, which fires `on.update`, which calls `handle_appstack_deployment` again. With a fast-path guard and idempotent `helm upgrade`, this may be tolerable, but it wastes resources and creates confusing log noise. Adding a `status.appStackVariablesSnapshot` field (from Pitfall 9) worsens this if not addressed.

**Root cause:** Kopf distinguishes `spec` updates from `status` updates only when the handler is filtered with `field=`. The current bare `@kopf.on.update` does not filter.

**Prevention:** Add `field='spec'` to the update decorator:

```python
@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')
def update_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
```

This is a pre-existing issue but the variables feature adds new status writes that amplify it.

**Warning signs:** Operator logs showing rapid successive reconcile cycles for the same CR after a deploy completes. Unit/integration test: perform a `kubectl patch --subresource status` and assert the update handler is NOT triggered.

**Phase to address:** Operator core. Worthwhile to fix regardless of the variables feature.

---

## Minor Pitfalls

### Pitfall 11: Validator Does Not Inspect `variables:` Keys or Values — False Negatives Before Apply

**What goes wrong:** `_validate_yaml_impl` in `validate_yaml.py` has no awareness of `spec.appStack.variables`. A CR with `variables: {my-host: foo}` (invalid identifier key), `variables: {namespace: null}` (null value), or `variables: [list]` (wrong type) passes `validate_yaml` with `valid: True`. The user receives no feedback until the operator fails at runtime.

**Prevention:** Add to `_validate_yaml_impl`:

```python
variables = (app_stack or {}).get('variables') or {}
if not isinstance(variables, dict):
    errors.append({'code': 'invalid_variables_type', 'path': '...variables',
                   'message': 'spec.appStack.variables must be a map'})
else:
    for k, v in variables.items():
        if not re.match(r'^[_a-zA-Z][_a-zA-Z0-9]*$', k):
            errors.append({'code': 'invalid_variable_name', 'path': f'...variables.{k}',
                           'message': f"Variable name '{k}' must match [_a-zA-Z][_a-zA-Z0-9]*"})
        if not isinstance(v, str):
            errors.append({'code': 'invalid_variable_value_type', 'path': f'...variables.{k}',
                           'message': f"Variable value for '{k}' must be a string"})
```

**Phase to address:** Validator.

---

### Pitfall 12: Tempfile Contains Rendered Secret Values — Verify Call Order

**What goes wrong:** The manifest component branch writes `component['kubernetesManifest']` to a tempfile. After this change, the RENDERED manifest (with substituted values) is written. If a future refactor writes to the tempfile BEFORE calling `render()` (e.g., to avoid re-rendering on retry), unrendered content with `${...}` placeholders persists on disk without cleanup if `render()` then raises.

**Prevention:** As proposed, call `render()` first, then write to tempfile, with tempfile creation inside the `try` block. Add a code comment: "render() must precede tempfile creation — prevents plaintext secret values persisting on render failure." Add a unit test that induces a `KeyError` in `render()` and asserts no tempfile exists after the exception.

**Phase to address:** Operator core — code review checkpoint.

---

### Pitfall 13: Single-Helm Path (`handle_helm_deployment`) Not Patched for Substitution

**What goes wrong:** `handle_helm_deployment` (line 833+) calls `load_values_from_reference` for top-level `spec.valuesFiles` (the non-AppStack, single-chart path). The variables dict is built only inside `handle_appstack_deployment`. If a user applies `${...}` in a non-AppStack `valuesFiles:` ConfigMap, no substitution occurs and the unresolved placeholder string is passed to Helm as-is — no error, silent wrong configuration.

**Prevention:** Either (a) thread `variables` through to `handle_helm_deployment` as well (extend the scope of v5.0), or (b) explicitly document and test that `${VAR}` substitution in `valuesFiles:` is only supported inside `appStack:` components. Add a unit test that locks whichever behavior is chosen.

**Phase to address:** Operator core. Scope decision should be explicit in the plan.

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `kubectl apply` + `kubernetesManifest` | Rendering after tempfile write creates a race where rendered secrets could persist | Call `render()` to get a string, then write the rendered string to tempfile, then `kubectl apply` |
| Helm + `valuesFiles` | Assuming Helm does `{{ }}` expansion on values content | Helm does NOT expand values; our `${VAR}` render happens before `yaml.safe_load`, Helm receives a resolved dict — zero collision |
| Kubernetes API + CRD validation | `variables: {port: 5432}` — integer passes CRD `type: string` if schema is malformed | Use `additionalProperties: {type: string}` correctly; add operator-side `isinstance(v, str)` check as defense |
| Kopf status patches + `on.update` | Status updates trigger re-deployment loop | Add `field='spec'` filter on `@kopf.on.update` |
| NGC Docker secrets | `$oauthtoken` in `stringData:` is a bare `$identifier` pattern | Fast-path guard prevents this; defense-in-depth: store in `data:` as base64 |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| `KeyError.args[0]` leaks variable name in error message | Variable NAME leaks (e.g., `postgresPassword`) — not the value. Acceptable. | Verified: `Template.substitute` raises `KeyError(key_name)`, not `KeyError(value)`. The error message `f"Undefined ${{{e.args[0]}}} in component X"` is safe. |
| Rendered Secret content in `comp_status['message']` | If `ValueError` from bad placeholder propagates to the generic handler, `str(e)` contains only positional info (line/col), never variable values | Verified safe — `ValueError.__str__()` is positional only |
| Rendered manifest written to `/tmp` | Rendered Secret `stringData` values on disk during `kubectl apply` | Python `tempfile.NamedTemporaryFile` creates mode 0600 on Linux; the `finally` block deletes the file. Acceptable risk. In-memory pipe to `kubectl apply --stdin` is a future hardening option. |
| Logging `str(e)` for exception from `render()` | If exception carries a value reference (not possible with Template), it could log secrets | `KeyError.__str__()` returns the key name quoted. `ValueError.__str__()` returns position text. Neither contains variable values. Safe. |

---

## "Looks Done But Isn't" Checklist

- [ ] **Backward compat gate:** Run `render(manifest, {'namespace': 'kube-system'})` on each of the 8 `kubernetesManifest` blocks in `cluster_init/app-store-cluster-init.yaml` and assert byte-identical output
- [ ] **NGC secret:** Render the full `aidp-bootstrap-secrets` manifest with `variables={'namespace': 'rag'}` — assert no exception and byte-identical output (covers `$oauthtoken` in any `stringData:` form)
- [ ] **Shell script passthrough:** Unit test `render('for X in $LIST; do echo $X; done', {'namespace': 'foo'})` returns the input unchanged (fast-path guard because no `${...}`)
- [ ] **`$$` escape:** Unit test `render('p$$w0rd', {'namespace': 'x'}) == 'p$w0rd'`
- [ ] **Nested variable values:** Unit test that `${milvusHost}` where `milvusHost='milvus.${namespace}.svc'` produces `'milvus.${namespace}.svc'` in output (inner `${namespace}` unresolved — single-pass documented)
- [ ] **TypeError guard:** Unit test `variables = {'namespace': 'x', 'port': 5432}` (int value) does not crash; Template str()-coerces it
- [ ] **Kopf update handler scope:** Verify status patch to the CR does NOT trigger re-deployment (requires `field='spec'` filter)
- [ ] **Single-helm path decision:** Unit test or explicit doc locks whether `handle_helm_deployment` also substitutes or explicitly does not

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Auto-default breaks backward compat (shell scripts, env vars) | Operator core | Unit test: 8 existing manifests from `cluster_init/` return unchanged |
| 2. `$oauthtoken` in NGC stringData | Operator core (fast-path guard covers) + AIDP migration | Unit test: AIDP bootstrap secret manifest renders unchanged |
| 3. Variable values not recursively resolved | Docs + tests | Unit test: single-pass behavior asserted; README shows fully-resolved examples |
| 4. `ValueError` on invalid placeholders not caught | Operator core | Unit test: `render()` call site catches `ValueError` and converts to `PermanentError` with component name |
| 5. Fetch vs. render error asymmetry | Operator core | Unit test: mock `kr8s.NotFound` raises `TemporaryError`, not silent `{}` |
| 6. Hyphen/digit variable key names | Operator core + docs | Unit test: `{'my-host': 'x'}` raises `PermanentError` before substitution |
| 7. `$$` escape semantics | Docs + tests | README escape table; unit tests for `p$$w0rd` and `p$$$$w0rd` |
| 8. Validator false-positives on svc.cluster.local | Validator | Warning message copy review; test external hostname does not produce error |
| 9. Orphan resources on namespace change | Docs | README warning section; optional status snapshot field |
| 10. `on.update` storm from status patches | Operator core | Add `field='spec'` filter; test that status patch does not trigger handler |
| 11. Validator blind to `variables:` field | Validator | Unit test: `variables: {my-host: foo}` produces `errors` not just `warnings` |
| 12. Tempfile call order | Operator core | Code review; unit test render failure leaves no tempfile |
| 13. Single-helm path not patched | Operator core or docs | Decision explicit in plan; unit test locks chosen behavior |

---

## Sources

- Direct code inspection: `operator_module/main.py` lines 340-370, 551-803, 1015-1027 — HIGH confidence
- Direct code inspection: `cluster_init/app-store-cluster-init.yaml` lines 107-161 (shell scripts in `kubernetesManifest`) — HIGH confidence
- Direct code inspection: `mcp-server/tools/validate_yaml.py` full file — HIGH confidence
- Executed experiments: Python 3.10 `string.Template` behavior verification (all pitfalls tested against actual CPython source) — HIGH confidence
- PRD: `.planning/PRD-appstack-variable-substitution.md` (risk table cross-referenced and extended) — HIGH confidence
- Python stdlib: `string.Template` identifier pattern confirmed via `Template.pattern.pattern` inspection: `[_a-z][_a-z0-9]*` (case-insensitive) — HIGH confidence

---
*Pitfalls research for: AppStack Variable Substitution (`${VAR}`) in Kopf-based Kubernetes operator*
*Researched: 2026-05-06*
