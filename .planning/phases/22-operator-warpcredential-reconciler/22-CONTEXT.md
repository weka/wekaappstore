# Phase 22: Operator WarpCredential Reconciler - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Add kopf handlers to `operator_module/main.py` that watch `WarpCredential` CRs (group `warp.io/v1alpha1`), read the raw credential from the referenced Secret, derive the type-appropriate Kubernetes Secrets (`warp-<name>-*`), and maintain `status.conditions`, `status.derivedSecrets`, and `status.lastSyncTime`. All derivation is idempotent — a manually-deleted derived secret is restored on the next reconcile cycle or operator restart.

Three credential types:
- `nvidia-ngc`: derives `warp-<name>-apikey` (Opaque, `NGC_API_KEY`) + `warp-<name>-docker` (`kubernetes.io/dockerconfigjson` for `nvcr.io`)
- `huggingface`: derives `warp-<name>-token` (Opaque, `HF_API_KEY`)
- `weka-storage`: derives `warp-<name>-token` (Opaque, `WEKA_API_USERNAME` + `WEKA_API_TOKEN` + `WEKA_API_ENDPOINT`); copies `spec.endpoint` → `status.wekaEndpoint`

Delete handler: logs a warning, does NOT delete derived secrets (OPS-08).

</domain>

<decisions>
## Implementation Decisions

### Secret Write Mechanism

- **D-01:** Use **kr8s** for creating and patching derived secrets — consistent with how Secrets are already read (`kr8s.objects.Secret.get`), pure Python, mockable in unit tests without subprocess.
- **D-02:** Idempotency pattern (OPS-09): `try secret.create() / except AlreadyExists → secret.patch()`. Standard Kubernetes controller create-or-update pattern. No delete-and-recreate.
- **D-03:** Secret values (raw key, derived payloads) are NEVER logged at any level (API-08). Pass key values through memory only; no tempfiles.

### Handler Registration Pattern

- **D-04:** Register `reconcile_warpcredential()` with three decorators: `@kopf.on.create`, `@kopf.on.update`, `@kopf.on.resume`. All three fire the same reconcile function — no code duplication.
  - `@kopf.on.resume` fires when the operator pod restarts, re-checking each existing `WarpCredential` CR and restoring any missing derived secrets. Required for OPS-09 idempotency after operator restart.
- **D-05:** Register a separate `@kopf.on.delete` handler (`delete_warpcredential`) that logs a warning and returns — does NOT delete derived secrets (OPS-08).
- **D-06:** All handlers target group `warp.io`, version `v1alpha1`, plural `warpcredentials`.

### Error Classification (extends Phase 18 pattern)

- **D-07:** If `spec.secretRef` Secret does not exist: `kopf.TemporaryError(delay=30)` — OPS-02 explicit. Updates `status.conditions[KeyReady].status = "False"` with `reason = "KeyMissing"` before raising.
- **D-08:** If `spec.type` is not one of `nvidia-ngc`, `huggingface`, `weka-storage`: `kopf.PermanentError` — OPS-01. (CRD enum already blocks this at admission, but the handler is the belt-and-suspenders catch.)
- **D-09:** If the key value read from `secretRef` is empty or whitespace-only: `kopf.PermanentError` naming the credential — OPS-03.
- **D-10:** kr8s network failures (timeout, 5xx) during Secret read: `kopf.TemporaryError(delay=30)`. Auth/RBAC failures (4xx): `kopf.PermanentError`. Consistent with Phase 18 D-01.

### Derivation Logic Structure

- **D-11:** Extract private helper functions per type — NOT inline in the handler:
  - `_derive_ngc_payloads(key: str) -> tuple[dict, dict]` — returns `(apikey_data, docker_data)` as plain dicts (Secret `.data` values already base64-encoded per Kubernetes Secret spec)
  - `_derive_hf_payload(key: str) -> dict` — returns `{HF_API_KEY: <b64>}`
  - `_derive_weka_payload(username: str, token: str, endpoint: str) -> dict` — returns three-key dict
  - Matches the `_render_or_raise` helper pattern established in Phase 18.
- **D-12:** The docker `auth` field in the NGC dockerconfigjson Secret is `base64("$oauthtoken:<key>")` using **standard base64 with padding** (`base64.b64encode(...).decode()`). Username is the literal string `$oauthtoken` (nvcr.io convention). Full payload: `{"auths":{"nvcr.io":{"username":"$oauthtoken","password":"<key>","auth":"<base64>"}}}`.
- **D-13:** All Secret `.data` values must be base64-encoded strings (Kubernetes stores Secret data as base64). Helpers return already-encoded values so the kr8s Secret object is built directly.

### Status Updates

- **D-14:** After successful derivation: set `status.conditions` (type `KeyReady`, status `True`, reason `KeyPresent`; for nvidia-ngc also `DockerSecretReady = True`), `status.derivedSecrets` list (name + type per derived secret), `status.lastSyncTime` (ISO 8601 UTC). Via kopf `patch` object.
- **D-15:** On failure paths (missing Secret, empty key): set `status.conditions[KeyReady].status = "False"` with appropriate reason (`KeyMissing`, `EmptyKey`, `DerivationFailed`) before raising the kopf error.

### Test Structure

- **D-16:** Test file: `operator_module/tests/test_warp_credential.py`. Tests call the helper functions directly — no kr8s mocking required for derivation tests. Separate test class for handler-level tests if needed.
- **D-17:** Test scope: verify derivation payloads for all three types, verify base64 encoding of NGC auth field, verify `KeyMissing` error path, verify no key values appear in any log output at any level (API-08 compliance).

### Claude's Discretion

- Exact placement of helper functions inside `operator_module/main.py` (near the top with other helpers, before kopf handlers — consistent with `_render_or_raise`)
- Exact error message wording for `PermanentError` and `TemporaryError` (within the format established by Phase 18: component/resource context, "will retry in 30s")
- Whether `weka-storage` `secretRef` carries all three values (username, token) in a single secret with multiple keys, or two keys — defer to requirements (OPS-06 says `secretRef` points to a single secret; the key referenced provides the token; username comes from a separate field or the same secret). **Researcher must clarify:** does `spec.secretRef.key` provide ONLY the token, or does the weka-storage secretRef secret contain all three keys? ROADMAP success criterion 3 says `warp-<name>-token` has `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT` — but the CRD only has `secretRef.name` + `secretRef.key`. Researcher should check the PRD for weka-storage source secret structure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §OPS-01 through OPS-09 — all operator reconciler requirements for this phase
- `.planning/REQUIREMENTS.md` §API-08 — no key values logged at any level (applies to operator)

### CRD Schema (locked by Phase 21)
- `weka-app-store-operator-chart/templates/crd.yaml` lines 279–382 — full `WarpCredential` CRD schema including `spec`, `status.conditions`, `status.derivedSecrets`, `status.lastSyncTime`, `status.wekaEndpoint`

### Existing Operator Patterns
- `operator_module/main.py` lines 297–308 — `_render_or_raise` helper pattern to follow for `_derive_*` helpers
- `operator_module/main.py` lines 421–481 — `load_values_from_reference` for kr8s error dispatch pattern (TemporaryError/PermanentError classification)
- `operator_module/main.py` lines 951–975 — existing `@kopf.on.create` handler shape to mirror for `WarpCredential`

### PRD (for weka-storage source secret structure clarification)
- `.planning/PRD-secret-management-overhaul.md` — check for weka-storage credential input schema (what keys does the raw secretRef Secret contain for username vs token)

### Test Patterns
- `operator_module/tests/conftest.py` — sys.path setup, import conventions
- `operator_module/tests/test_render.py` — example of testing a pure helper function directly

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_render_or_raise(text, variables, *, source_desc)` at `operator_module/main.py:297` — helper pattern to replicate for derivation helpers
- `kr8s.objects.Secret.get(name, namespace)` — already used at line 444 for reads; same API for write path
- `kopf.TemporaryError` / `kopf.PermanentError` — established error hierarchy; Phase 18 error dispatch at lines 449–481 is the canonical reference

### Established Patterns
- Single large `operator_module/main.py` — all new code goes here, no new submodules
- kopf `patch` object for status updates — used in existing handlers at lines 720–724, 967–975
- Error dispatch: `kr8s.NotFoundError → TemporaryError`, `kr8s.APITimeoutError → TemporaryError`, `kr8s.ServerError(5xx) → TemporaryError`, `kr8s.ServerError(4xx) → PermanentError`

### Integration Points
- Three new kopf decorators registered on `warp.io/v1alpha1/warpcredentials` (after existing `wekaappstores` handlers at lines 951–1200)
- Phase 23 will call `GET /api/credentials` which reads `WarpCredential` CRs and their `status` — the `status.conditions`, `status.derivedSecrets`, `status.wekaEndpoint` fields populated here are consumed there

</code_context>

<specifics>
## Specific Ideas

- NGC docker pull secret `auth` field: `base64.b64encode(f"$oauthtoken:{key}".encode()).decode()` — standard encoding, padding included
- Handler function name: `reconcile_warpcredential` for the create/update/resume target; `delete_warpcredential` for delete
- `warp-<name>-*` secret names use the `WarpCredential` CR's `metadata.name` verbatim — no slug transformation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 22-operator-warpcredential-reconciler*
*Context gathered: 2026-06-11*
