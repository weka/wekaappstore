# Phase 22: Operator WarpCredential Reconciler - Research

**Researched:** 2026-06-11
**Domain:** Kubernetes operator (kopf + kr8s) — CRD reconciliation, Secret derivation, status conditions
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Secret Write Mechanism**
- **D-01:** Use **kr8s** for creating and patching derived secrets — consistent with how Secrets are already read (`kr8s.objects.Secret.get`), pure Python, mockable in unit tests without subprocess.
- **D-02:** Idempotency pattern (OPS-09): `try secret.create() / except AlreadyExists → secret.patch()`. Standard Kubernetes controller create-or-update pattern. No delete-and-recreate.
- **D-03:** Secret values (raw key, derived payloads) are NEVER logged at any level (API-08). Pass key values through memory only; no tempfiles.

**Handler Registration Pattern**
- **D-04:** Register `reconcile_warpcredential()` with three decorators: `@kopf.on.create`, `@kopf.on.update`, `@kopf.on.resume`. All three fire the same reconcile function — no code duplication.
- **D-05:** Register a separate `@kopf.on.delete` handler (`delete_warpcredential`) that logs a warning and returns — does NOT delete derived secrets (OPS-08).
- **D-06:** All handlers target group `warp.io`, version `v1alpha1`, plural `warpcredentials`.

**Error Classification (extends Phase 18 pattern)**
- **D-07:** If `spec.secretRef` Secret does not exist: `kopf.TemporaryError(delay=30)` — OPS-02 explicit. Updates `status.conditions[KeyReady].status = "False"` with `reason = "KeyMissing"` before raising.
- **D-08:** If `spec.type` is not one of `nvidia-ngc`, `huggingface`, `weka-storage`: `kopf.PermanentError` — OPS-01.
- **D-09:** If the key value read from `secretRef` is empty or whitespace-only: `kopf.PermanentError` naming the credential — OPS-03.
- **D-10:** kr8s network failures (timeout, 5xx) during Secret read: `kopf.TemporaryError(delay=30)`. Auth/RBAC failures (4xx): `kopf.PermanentError`. Consistent with Phase 18 D-01.

**Derivation Logic Structure**
- **D-11:** Extract private helper functions per type — NOT inline in the handler:
  - `_derive_ngc_payloads(key: str) -> tuple[dict, dict]` — returns `(apikey_data, docker_data)` as plain dicts (Secret `.data` values already base64-encoded per Kubernetes Secret spec)
  - `_derive_hf_payload(key: str) -> dict` — returns `{HF_API_KEY: <b64>}`
  - `_derive_weka_payload(username: str, token: str, endpoint: str) -> dict` — returns three-key dict
  - Matches the `_render_or_raise` helper pattern established in Phase 18.
- **D-12:** The docker `auth` field in the NGC dockerconfigjson Secret is `base64("$oauthtoken:<key>")` using **standard base64 with padding** (`base64.b64encode(...).decode()`). Username is the literal string `$oauthtoken`. Full payload: `{"auths":{"nvcr.io":{"username":"$oauthtoken","password":"<key>","auth":"<base64>"}}}`.
- **D-13:** All Secret `.data` values must be base64-encoded strings. Helpers return already-encoded values so the kr8s Secret object is built directly.

**Status Updates**
- **D-14:** After successful derivation: set `status.conditions` (type `KeyReady`, status `True`, reason `KeyPresent`; for nvidia-ngc also `DockerSecretReady = True`), `status.derivedSecrets` list, `status.lastSyncTime` (ISO 8601 UTC). Via kopf `patch` object.
- **D-15:** On failure paths: set `status.conditions[KeyReady].status = "False"` with appropriate reason (`KeyMissing`, `EmptyKey`, `DerivationFailed`) before raising the kopf error.

**Test Structure**
- **D-16:** Test file: `operator_module/tests/test_warp_credential.py`. Tests call the helper functions directly — no kr8s mocking required for derivation tests.
- **D-17:** Test scope: verify derivation payloads for all three types, verify base64 encoding of NGC auth field, verify `KeyMissing` error path, verify no key values appear in any log output at any level.

### Claude's Discretion

- Exact placement of helper functions inside `operator_module/main.py` (near the top with other helpers, before kopf handlers — consistent with `_render_or_raise`)
- Exact error message wording for `PermanentError` and `TemporaryError` (within the format established by Phase 18: component/resource context, "will retry in 30s")
- **OPEN QUESTION** about weka-storage source-secret structure — **resolved below in Section 5** (single source Secret containing all three keys: `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`).

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OPS-01 | `kopf.on.create`/`on.update` watches WarpCredential; PermanentError on unknown `spec.type` | Section 4 (decorator syntax + multi-decorator stacking) + Section 3 (PermanentError pattern) |
| OPS-02 | Missing `spec.secretRef` Secret → `TemporaryError(delay=30)` | Section 3 (kr8s.NotFoundError → TemporaryError pattern at main.py:449–453) |
| OPS-03 | Empty/whitespace key → `PermanentError` naming the credential | Section 3 (PermanentError pattern at main.py:466–468) |
| OPS-04 | Create/patch `warp-<name>-apikey` (Opaque, NGC_API_KEY) + `warp-<name>-docker` (dockerconfigjson) | Section 1 (kr8s Secret create + patch; Section 5 lists exact data keys) |
| OPS-05 | Create/patch `warp-<name>-token` (Opaque, HF_API_KEY) | Section 1 (kr8s pattern) |
| OPS-06 | Create/patch `warp-<name>-token` (Opaque, three keys: USERNAME+TOKEN+ENDPOINT); copy `spec.endpoint` → `status.wekaEndpoint` | Section 5 (resolved source-secret structure: all three keys live in single source Secret, indexed by hard-coded constants) |
| OPS-07 | Update `status.conditions`, `status.derivedSecrets`, `status.lastSyncTime` after reconcile | Section 2 (status patch shape mirroring existing CR pattern at main.py:720–728) |
| OPS-08 | Delete handler does NOT delete derived secrets; logs warning | Section 4 (`@kopf.on.delete(..., optional=True)` to avoid finalizer; handler body is just logger.warning + return) |
| OPS-09 | Idempotency: deleted derived secret restored on next reconcile | Section 1 (create-or-patch loop) + Section 4 (`@kopf.on.resume` fires on operator restart re-checking every CR) |
| API-08 | No raw key values logged at any level | Section 7 (caplog-based assertion pattern; logger context tagging) |

</phase_requirements>

## Summary

Phase 22 adds three kopf handlers (`reconcile_warpcredential` stacked on `on.create`/`on.update`/`on.resume`, plus a separate `delete_warpcredential` on `on.delete`) to `operator_module/main.py` watching `warp.io/v1alpha1/warpcredentials`. The reconciler reads the raw credential from `spec.secretRef`, dispatches on `spec.type` to one of three private `_derive_*` helpers, and writes derived `warp-<name>-*` Secrets via the kr8s create-or-patch idempotency pattern. Status conditions, `derivedSecrets`, and `lastSyncTime` are written through kopf's `patch.status` object.

The standard stack is already pinned by `operator_module/requirements.txt`: `kopf>=1.38.0`, `kr8s>=0.17.0` (locally installed 0.20.10). The substitution-helper pattern (`_render_or_raise`) and the kr8s exception dispatch matrix (`NotFoundError` / `APITimeoutError` / `ServerError(5xx)` → `TemporaryError`, `ServerError(4xx)` → `PermanentError`) are established by Phase 18 at `operator_module/main.py:449–481` and must be mirrored — not reinvented — for the new handler.

**Primary recommendation:** Add ~150 lines of code below the existing `wekaappstores` handlers (after line 1246) — three `_derive_*` helpers grouped with `_render_or_raise` (around line 308), one `_apply_secret_idempotent(secret_obj)` helper (one place to encapsulate the create/409→patch pattern, single test surface for OPS-09), one `reconcile_warpcredential(...)` body with type dispatch, one `delete_warpcredential(...)` warning-only body, plus shared status-condition builders. Tests live in `operator_module/tests/test_warp_credential.py` using the same `MagicMock`-based kr8s mocking already used in `test_appstack.py`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| WarpCredential CR watch / dispatch | Operator (kopf event loop) | — | Only the operator runs the kopf loop; GUI/API do not handle CRs |
| Raw Secret read via `spec.secretRef` | Operator (kr8s) | — | Operator's namespace-scoped Role grants Secret get/list/patch (Phase 21) |
| Type-specific payload derivation (`_derive_*`) | Operator (pure Python) | — | Pure functions; testable without cluster |
| Derived Secret create/patch | Operator (kr8s) | — | kr8s.objects.Secret create/patch with merge-patch — operator-only RBAC |
| Status condition + `derivedSecrets` write | Operator (kopf `patch.status`) | — | kopf framework manages status subresource patches |
| Cache of status data for API consumers (Phase 23) | API tier (GUI backend) | Operator writes status | Operator publishes; backend reads — clean handoff |
| Raw key validation / form input | API tier (Phase 23) | — | Operator only reads existing Secrets; it does not handle form data |

## Standard Stack

### Core
| Library | Version (pinned) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `kopf` | `>=1.38.0` (installed 1.38.0) | CRD event loop, decorators, patch object, error classes | [VERIFIED: operator_module/requirements.txt:4] — already used by all existing operator handlers (main.py:951, 1159, 1175). |
| `kr8s` | `>=0.17.0` (installed 0.20.10) | Sync Kubernetes API client for Secret read/create/patch | [VERIFIED: operator_module/requirements.txt:7] + [VERIFIED: `pip show kr8s` returns 0.20.10] — already used for Secret reads (main.py:444) and Pod create (main.py:1119). |
| `kubernetes` | `>=27.0.0` | Official k8s client (used for CRD discovery only) | [VERIFIED: operator_module/requirements.txt:10] — NOT used by reconcile_warpcredential; lazy-loaded for CRD-strategy helpers only. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `base64` (stdlib) | builtin | Encode Secret `.data` values and NGC `auth` field | Always — Kubernetes Secret API requires base64-encoded `.data` values. |
| `json` (stdlib) | builtin | Serialize NGC dockerconfigjson payload | NGC docker secret only (D-12). |
| `datetime` (stdlib) | builtin | `lastSyncTime`, `lastTransitionTime` ISO-8601 timestamps | Matches existing pattern (`datetime.utcnow().isoformat() + 'Z'`) at main.py:727, 939, 972, 1109. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| kr8s Secret create/patch | Official `kubernetes` client (`CoreV1Api.create_namespaced_secret` / `patch_namespaced_secret`) | Heavier API, requires explicit `load_incluster_config()`, exception class is `kubernetes.client.ApiException` (status attribute) — diverges from project pattern. **Rejected** — D-01 locks kr8s. |
| `try create / except ServerError(409) → patch` | Pre-flight `exists()` check, then create or patch | Two API calls instead of one in the steady state; race window between check and create. **Rejected** — locked in D-02. |
| Adding finalizer to block delete handler | `optional=True` on `@kopf.on.delete` | Finalizer would force CRs to linger until operator processes delete (a problem if operator is down). **Resolved** — Section 4 recommends `optional=True`; the warning-only handler is best-effort by design (OPS-08). |

**Installation:** No new packages — existing operator dependencies cover everything.

**Version verification:**
- `kr8s 0.20.10` confirmed via `pip show kr8s` in the local environment [VERIFIED: pip output 2026-06-11].
- `kopf 1.38.0` confirmed via `pip show kopf` and `kopf.__version__` [VERIFIED: shell exec 2026-06-11].

## Architecture Patterns

### System Architecture Diagram

```
                                  ┌────────────────────────────────────────────┐
                                  │  Kubernetes API Server                     │
                                  │  ─ WarpCredential CR (warp.io/v1alpha1)    │
                                  │  ─ Secret <spec.secretRef.name>            │
                                  │  ─ Secret warp-<name>-apikey (derived)     │
                                  │  ─ Secret warp-<name>-docker (derived)     │
                                  │  ─ Secret warp-<name>-token  (derived)     │
                                  └────────┬─────────────────────┬─────────────┘
                                           │ watch                │ create/patch
                                           │ (kopf event)         │ (kr8s)
                                           ▼                      ▲
                ┌──────────────────────────────────────────────────┴──────────┐
                │  operator_module/main.py (kopf single-file module)          │
                │                                                              │
                │  ┌─────────────────────────────────────────────────────────┐ │
                │  │ @kopf.on.create / on.update / on.resume                 │ │
                │  │  reconcile_warpcredential(body, spec, name, ns, patch,  │ │
                │  │                          logger, **kwargs)               │ │
                │  │                                                          │ │
                │  │   1. Read kr8s.objects.Secret.get(spec.secretRef.name)  │ │
                │  │      └─NotFoundError → status[KeyMissing] +             │ │
                │  │         TemporaryError(delay=30)                        │ │
                │  │   2. Decode .data[spec.secretRef.key]                   │ │
                │  │      └─missing/empty → status[EmptyKey] + Permanent     │ │
                │  │   3. Dispatch on spec.type:                             │ │
                │  │       ┌─ nvidia-ngc → _derive_ngc_payloads(key)         │ │
                │  │       ├─ huggingface → _derive_hf_payload(key)          │ │
                │  │       └─ weka-storage → _derive_weka_payload(           │ │
                │  │                           username, token, endpoint)    │ │
                │  │   4. For each derived payload:                          │ │
                │  │       _apply_secret_idempotent(secret_obj)              │ │
                │  │         └─ try create() / except ServerError(409): patch│ │
                │  │   5. patch.status[conditions/derivedSecrets/lastSync]  │ │
                │  │      For weka-storage also patch.status['wekaEndpoint']│ │
                │  └─────────────────────────────────────────────────────────┘ │
                │                                                              │
                │  ┌─────────────────────────────────────────────────────────┐ │
                │  │ @kopf.on.delete(optional=True)                          │ │
                │  │  delete_warpcredential(name, namespace, logger, **_)    │ │
                │  │   logger.warning("...derived secrets NOT deleted...")   │ │
                │  │   return                                                │ │
                │  └─────────────────────────────────────────────────────────┘ │
                └──────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
operator_module/
├── main.py                           # ADD: ~150 LOC for new handlers + helpers
│   ├── _derive_ngc_payloads()        # NEW (near line 308, after _render_or_raise)
│   ├── _derive_hf_payload()          # NEW
│   ├── _derive_weka_payload()        # NEW
│   ├── _apply_secret_idempotent()    # NEW (kr8s create-or-patch wrapper)
│   ├── _build_condition()            # NEW (ISO 8601 lastTransitionTime helper)
│   ├── reconcile_warpcredential()    # NEW (after line 1246)
│   └── delete_warpcredential()       # NEW
└── tests/
    └── test_warp_credential.py       # NEW (~250 LOC, mirrors test_appstack.py style)
```

### Pattern 1: kr8s create-or-patch idempotency (OPS-09, D-02)
**What:** A single helper encapsulates "try create / on 409 → patch" so the handler body stays linear.
**When to use:** Any time the operator must converge to a desired-state Secret regardless of whether it pre-exists.
**Example:**
```python
# Recommended new helper (near main.py:308, mirrors _render_or_raise placement)
def _apply_secret_idempotent(secret_obj: kr8s.objects.Secret, *, ctx: str) -> None:
    """Create-or-patch a kr8s Secret. Locked by D-02 (no delete-and-recreate).

    Idempotent: on 409 Conflict (already exists), issues a merge-patch
    with the new .data and .type (kept identical for stable derivations).
    All other kr8s errors are surfaced via the Phase 18 dispatch matrix:
      NotFoundError / APITimeoutError / ServerError(>=500) -> TemporaryError(delay=30)
      ServerError(4xx, non-409)                            -> PermanentError
    """
    try:
        secret_obj.create()
    except kr8s.ServerError as e:
        status = e.response.status_code if getattr(e, 'response', None) is not None else None
        if status == 409:
            # Already exists -> merge-patch with new .data (keeps .type stable).
            secret_obj.patch({
                'data': secret_obj.raw['data'],
                'type': secret_obj.raw['type'],
            })
            return
        if status is not None and status >= 500:
            raise kopf.TemporaryError(
                f'{ctx}: API server error {status} writing Secret (will retry in 30s)',
                delay=30,
            ) from e
        raise kopf.PermanentError(f'{ctx}: API error writing Secret: {e}') from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(
            f'{ctx}: timeout writing Secret (will retry in 30s)', delay=30
        ) from e
```
[CITED: kr8s 0.20.10 source `kr8s/_api.py:186-201` — confirms 4xx and 5xx both raise `kr8s.ServerError` with `e.response.status_code` accessible; no dedicated `AlreadyExistsError` class exists. The error matrix mirrors `load_values_from_reference` at `operator_module/main.py:449-481`.]

### Pattern 2: Stacked kopf decorators on a single function (D-04)
**What:** Three decorators on one function — create/update/resume all converge to the same reconcile.
**When to use:** Idempotent reconcilers where create-time logic == update-time logic == resume-time logic (verified by re-checking remote state). All three paths must be exercised because:
- `on.create` — first time the CR is observed
- `on.update` — `spec` changes after the fact
- `on.resume` — fires per CR on operator pod restart; required for OPS-09 idempotency after operator downtime
**Example:**
```python
# Place after main.py:1246 (after delete_warrpappstore_function)
@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
    """Reconcile a WarpCredential CR — derive Secrets, update status.

    Locked decisions: D-04 (stacked decorators), D-07/D-08/D-09/D-10 (error class),
    D-11..D-13 (derivation helpers), D-14/D-15 (status writes), D-03 (no key logged).
    """
    ...
```
[CITED: kopf 1.38 docs — "It is a common pattern to declare both creation and resuming handlers pointing to the same function." Pattern confirmed in the project at main.py:951 (single decorator) and the existing pattern of `update_warrpappstore_function` at main.py:1159 calling the same `handle_appstack_deployment`.]

### Pattern 3: Delete handler without finalizer (D-05, OPS-08)
**What:** Use `optional=True` to register a delete handler without adding a finalizer.
**When to use:** When the delete handler does no destructive work that must be guaranteed to complete (here: only a warning log). Acceptable behavior is "best-effort log; if missed, the cluster state is unchanged anyway."
**Example:**
```python
@kopf.on.delete('warp.io', 'v1alpha1', 'warpcredentials', optional=True)
def delete_warpcredential(name, namespace, logger, **_):
    """OPS-08: log a warning; do NOT delete derived secrets.

    optional=True prevents kopf from adding a finalizer (which would block the
    CR's deletion until the operator runs). Without a finalizer, this handler
    may not always fire if Kubernetes wipes the resource before kopf processes
    the DELETED event — that's acceptable because the handler does no
    destructive work; the inaction (preserving derived secrets) is the
    contract. Cluster state is identical whether the warning was logged or not.
    """
    logger.warning(
        f"WarpCredential {namespace}/{name} deleted; derived secrets "
        f"warp-{name}-* are intentionally NOT removed (OPS-08). "
        f"Administrator must delete them manually if no longer needed."
    )
```
[CITED: kopf 1.38 `kopf.on.delete` signature (verified via `inspect.signature` on installed kopf 1.38.0): accepts `optional: Optional[bool] = None`.]
[CITED: github.com/nolar/kopf#701 maintainer (nolar) comment 2026-06: "with optional=True, there are no finalizers, and the execution of deletion handlers depends on luck... The deletion handler should still be called on a last 'goodbye' event — at least one, at least once; but no retries or second chances." The handler MAY be missed if K8s wipes the object before kopf processes it — acceptable here because the action is best-effort logging only.]

### Pattern 4: kr8s Secret object built from raw dict (D-13)
**What:** Instantiate `kr8s.objects.Secret` with a raw API dict whose `.data` values are already base64-encoded strings.
**When to use:** Always for derived Secrets in this phase.
**Example:**
```python
import base64
import kr8s

# In reconcile_warpcredential, after _derive_ngc_payloads(key) returns the dict:
apikey_secret = kr8s.objects.Secret({
    'apiVersion': 'v1',
    'kind': 'Secret',
    'metadata': {
        'name': f'warp-{name}-apikey',
        'namespace': namespace,
        # Owner reference recommended so kubectl delete WarpCredential <name>
        # would tidy up *if* OPS-08 ever changed — for now, OPS-08 says do NOT
        # use ownerReferences (otherwise K8s garbage-collects derived secrets).
    },
    'type': 'Opaque',
    'data': {
        'NGC_API_KEY': base64.b64encode(key.encode('utf-8')).decode('ascii'),
    },
})
_apply_secret_idempotent(apikey_secret, ctx=f"WarpCredential {namespace}/{name}: apikey")
```
[CITED: kr8s 0.20.10 source `kr8s/_objects.py:60-78` — Secret(resource: dict) stores the dict as `self.raw` and the create() method POSTs `json.dumps(self.raw_template)` to the K8s API.]

**IMPORTANT — Owner references intentionally OMITTED:** Adding `metadata.ownerReferences` pointing at the WarpCredential CR would cause Kubernetes garbage-collection to delete the derived secrets when the CR is deleted, which violates OPS-08. The planner MUST NOT add owner references on the derived Secrets.

### Pattern 5: Status patch shape (D-14, D-15) — see Section 2 below for exact dict structure.

### Anti-Patterns to Avoid

- **Don't use ownerReferences on derived Secrets.** It enables K8s garbage collection, violating OPS-08 (derived secrets must survive CR deletion).
- **Don't `delete()` then `create()` as the idempotency strategy.** D-02 forbids it — the window between delete and create leaves the cluster in an unreadable state and breaks workloads that have the Secret mounted.
- **Don't write raw key bytes to a tempfile.** D-03 — bytes must live only in process memory.
- **Don't `print()` anywhere in the operator paths.** CLAUDE.md project rule + API-08. Use the kopf `logger` kwarg (per-handler) or the module-level `logging.getLogger(__name__)`.
- **Don't use single quotes for kr8s/JSON dict keys.** Wait — operator module convention IS single quotes per CLAUDE.md ("operator uses single quotes for shell args"). Inspect main.py around line 939: existing code uses single-quoted dict keys (`'type': 'Ready'`). Match this.
- **Don't log raw key values in f-strings.** No `logger.info(f"got key {key}")` — even at DEBUG level (D-03).
- **Don't read the source Secret with the kubernetes-client library.** Already-established kr8s pattern at main.py:444 — divergence breaks the existing mock fixture pattern in test_appstack.py.
- **Don't pass `body`/`spec` into log records.** kopf may include the full body in logger context; avoid `extra={'body': body}` patterns.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Secret create/update idempotency | Custom `if exists then patch else create` with pre-flight `exists()` | `try create / except ServerError(409) → patch` | The pre-flight check has a TOCTOU race; the exception-based path is one round-trip in steady state. Locked by D-02. |
| HTTP error classification | Custom retry-aware HTTP client wrapper | Reuse existing `kr8s.ServerError` dispatch logic from `load_values_from_reference` at main.py:449–481 | The Phase 18 pattern is canonical. Mirror it line-for-line to maintain a single retry policy across the operator. |
| Base64 encoding | Custom byte ↔ string conversion | `base64.b64encode(s.encode('utf-8')).decode('ascii')` (stdlib) | Stdlib correctness; padding included automatically (D-12 requires standard padding). |
| Condition `lastTransitionTime` | Custom timestamp formatter | `datetime.utcnow().isoformat() + 'Z'` matching main.py:727, 939, 972, 1109 | Project pattern — diverging confuses status consumers (Phase 23 API). |
| Multiple-decorator dispatch | An `if event_type == "create" else update else resume` switch inside one decorator | Three `@kopf.on.*` decorators stacked on one function | kopf-native pattern; documented and supported. |
| Finalizer cleanup logic | Custom finalizer add/remove via kr8s patches | `@kopf.on.delete(..., optional=True)` opts out of finalizers entirely | OPS-08 says "log warning, do nothing" — finalizers add risk without benefit. |
| dockerconfigjson payload | YAML or string concat | `json.dumps({"auths": {"nvcr.io": {"username": "$oauthtoken", "password": key, "auth": <b64>}}})` (stdlib) | Stable JSON serialization; quoting handled. K8s rejects malformed dockerconfigjson at admission for type `kubernetes.io/dockerconfigjson`. |

**Key insight:** The Phase 18 work already built the "kr8s exception → kopf typed error" mapping. The Phase 22 plan must reuse that pattern rather than reinventing it. The same logic appears in `load_values_from_reference` lines 449–481 — extract it into a shared helper or copy line-for-line, but do NOT diverge.

## Runtime State Inventory

Phase 22 is a greenfield addition (new handlers, new derived Secrets, no rename/refactor). This phase **creates** runtime state rather than mutating existing state. Inventory below is informational only.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 22 introduces new Secret resources `warp-<name>-apikey`, `warp-<name>-docker`, `warp-<name>-token`; no pre-existing data to migrate. | None |
| Live service config | None — no external services beyond the K8s API. | None |
| OS-registered state | None. | None |
| Secrets/env vars | None — operator pod's existing service account (Phase 21 RBAC) already has the needed permissions. | None — verified via Phase 21 ROADMAP success criterion 5 (Helm Role+RoleBinding granting Secret CRUD scoped to App Store namespace). |
| Build artifacts | None — pure-Python additions to existing module; no rebuild of installed packages required. The operator container image already ships `kopf` and `kr8s`. | None |

**Nothing found in any category** — verified by the fact that the phase only adds new handler functions and helpers; it does not touch any existing reconcile path, file, or configuration.

## Common Pitfalls

### Pitfall 1: 409 Conflict not raising `AlreadyExistsError`
**What goes wrong:** Catching `kr8s.AlreadyExistsError` (which does not exist in kr8s 0.20.x) — code never compiles, or worse, a wildcard `except Exception` catches everything and silently swallows real errors.
**Why it happens:** Many Kubernetes client libraries have an `AlreadyExistsError`; kr8s does not — it raises `kr8s.ServerError` with `e.response.status_code == 409`.
**How to avoid:** Pattern `except kr8s.ServerError as e: if getattr(e, 'response', None) and e.response.status_code == 409: ...` (see Pattern 1 code snippet). Add a unit test that calls `_apply_secret_idempotent` with a mock that raises `kr8s.ServerError` with `.response.status_code = 409` and asserts patch was called.
**Warning signs:** ImportError on `from kr8s import AlreadyExistsError`. Silent NoOp on second `create()`. [VERIFIED: kr8s 0.20.10 source `kr8s/_exceptions.py` — only `APITimeoutError`, `ConnectionClosedError`, `ExecError`, `NotFoundError`, `ServerError` are defined; cross-checked via `dir(kr8s)`.]

### Pitfall 2: Logging raw key values via `body` kwarg
**What goes wrong:** A handler kwarg like `body` is passed through to `logger.info(body)` (or worse, `repr(body)` in a traceback), and the resulting log record contains the raw Secret data referenced by the CR.
**Why it happens:** kopf's `body` kwarg contains the full CR (which itself does NOT carry the key — D-03), but if the handler dumps the source Secret object's `.data` (post-decode) into a log line, the cleartext key is logged.
**How to avoid:** Tag all log records with metadata only (name, namespace, type, derived-secret-name). The variable holding the decoded key must never be passed to a logger directly or via f-string. The caplog-based test (Section 7) asserts that no log record across the entire reconcile contains the key string.
**Warning signs:** Reviewer sees `logger.info(f"... {key}")` or `logger.debug(f"derived data: {data}")` in any code change.

### Pitfall 3: `field='spec'` on `@kopf.on.update` (do NOT add it)
**What goes wrong:** Following the `WekaAppStore` pattern blindly (main.py:1159 uses `field='spec'`), the planner adds `field='spec'` to the WarpCredential update decorator. This means updates to `status` (e.g., the operator's own writes) won't re-trigger the handler — correct! — BUT it also means `metadata`-only updates (label edits) don't trigger a re-reconcile. For WarpCredential this is fine, but be deliberate.
**Why it happens:** Cargo-culting the existing pattern without re-evaluating for the new CR.
**How to avoid:** EXPLICITLY include `field='spec'` on `@kopf.on.update` to match the WekaAppStore pattern at main.py:1159 — this is the correct choice for WarpCredential (re-reconcile only on spec changes), and the planner should document the rationale in a code comment so future reviewers don't accidentally remove it.

### Pitfall 4: Missing `subresource='status'` confusion (NOT needed here)
**What goes wrong:** Engineer reads kr8s docs about `patch(..., subresource='status')` and concludes the handler must call `cr.patch(..., subresource='status')` directly to write status.
**Why it happens:** kopf hides the status-subresource patch — handlers use `patch.status[...] = ...` and kopf submits the patch (correctly targeting the status subresource because the CRD declares `subresources: { status: {} }` at crd.yaml line 287).
**How to avoid:** Use `patch.status['conditions'] = [...]` exactly as the existing handler does at main.py:721–724 and main.py:944–945. Do NOT call `kr8s.objects[...].patch(subresource='status')` from the handler — kopf handles it.

### Pitfall 5: `datetime.utcnow()` deprecation
**What goes wrong:** Python 3.12+ deprecation warnings on `datetime.utcnow()`.
**Why it happens:** `datetime.utcnow()` returns a naive datetime; Python 3.12 deprecates it in favor of `datetime.now(timezone.utc)`.
**How to avoid:** **Match the existing project pattern** — every existing call site (main.py:727, 939, 972, 1109) uses `datetime.utcnow().isoformat() + 'Z'`. Changing this requires a project-wide refactor; for Phase 22, mirror exactly. If lint blocks it, suppress for consistency with existing code (Phase 22 should not introduce a deprecation-fix scope creep).

### Pitfall 6: NGC `auth` field with URL-safe base64 or stripped padding
**What goes wrong:** Using `base64.urlsafe_b64encode` or `b64encode().rstrip(b'=')` makes the docker `auth` value technically smaller but breaks Docker's expected decode (Docker uses standard base64 with padding).
**Why it happens:** Engineer remembers JWT-style base64 conventions.
**How to avoid:** Use `base64.b64encode(f"$oauthtoken:{key}".encode('utf-8')).decode('ascii')` — D-12 explicit. Include a regression test: the `auth` field must equal `base64.b64encode(b'$oauthtoken:<known-test-key>').decode()` byte-for-byte.

### Pitfall 7: kr8s Secret `.type` must be a string, not omitted
**What goes wrong:** Omitting `type` in the raw dict causes K8s to default to `Opaque`, which is fine for token Secrets but WRONG for the NGC docker secret (must be `kubernetes.io/dockerconfigjson`).
**Why it happens:** Optimistic default-trust.
**How to avoid:** Always set `'type': 'Opaque'` or `'type': 'kubernetes.io/dockerconfigjson'` explicitly. The `_derive_*` helpers should return both the data dict AND the secret type so the caller has no ambiguity. Alternative (recommended): have helpers return `(name_suffix, secret_type, data_dict)` tuples.

### Pitfall 8: `kr8s.objects.Secret.create()` is synchronous but kr8s also has async paths
**What goes wrong:** Following async docs and adding `await` in the synchronous operator code path.
**Why it happens:** kr8s docs show both `async_create` and `create`; the latter dispatches sync/async based on the calling context.
**How to avoid:** kopf handlers (in this project) are synchronous functions — call `secret.create()` and `secret.patch(...)` directly. The kr8s library handles the sync wrapping internally. Verified: existing project code calls `pod.create()` synchronously at main.py:1136.

## Code Examples

Verified patterns from official sources and the existing project code:

### Example A: Status condition shape (CRD-locked)

Per CRD schema (`weka-app-store-operator-chart/templates/crd.yaml:330-358`), each condition entry has: `type` (required string), `status` (required enum "True"/"False"/"Unknown"), optional `reason`, `message`, `lastTransitionTime`. The `derivedSecrets` items (lines 359-370) have `name` (string) and `type` (string).

```python
# Source: project pattern at operator_module/main.py:720-728 and main.py:934-940
# Adapted to WarpCredential per D-14, D-15 and CRD schema lines 330-377.
from datetime import datetime

def _now_iso() -> str:
    """ISO 8601 UTC timestamp matching the project convention."""
    return datetime.utcnow().isoformat() + 'Z'

def _build_condition(type_: str, status: str, reason: str, message: str) -> dict:
    return {
        'type': type_,
        'status': status,
        'reason': reason,
        'message': message,
        'lastTransitionTime': _now_iso(),
    }

# Success path (in reconcile_warpcredential, after all derivations succeed):
conditions = [_build_condition('KeyReady', 'True', 'KeyPresent', 'All derived secrets reconciled')]
if cred_type == 'nvidia-ngc':
    conditions.append(_build_condition('DockerSecretReady', 'True', 'DockerSecretPresent',
                                       'NGC dockerconfigjson Secret reconciled'))

patch.status['conditions'] = conditions
patch.status['derivedSecrets'] = derived_secrets_list  # list of {'name': ..., 'type': ...}
patch.status['lastSyncTime'] = _now_iso()
if cred_type == 'weka-storage':
    patch.status['wekaEndpoint'] = spec.get('endpoint', '')

# Failure path (before raising kopf.TemporaryError or PermanentError):
patch.status['conditions'] = [
    _build_condition('KeyReady', 'False', 'KeyMissing',
                     f'Referenced Secret {namespace}/{secret_ref_name} not found'),
]
```
[CITED: crd.yaml:330-358 — condition schema; crd.yaml:359-370 — derivedSecrets schema; crd.yaml:371-377 — lastSyncTime + wekaEndpoint.]
[CITED: operator_module/main.py:720-728 — established `patch.status['conditions'] = [{...}]` pattern.]

**lastTransitionTime: always-set vs. only-on-change.** The Kubernetes API conventions doc recommends updating `lastTransitionTime` ONLY when the status field actually transitions. However, the existing project pattern (main.py:727, 939, 972, 1109) always sets `lastTransitionTime` to the current time on every reconcile. **Recommendation: match the project pattern (always set) for Phase 22.** Reasons: (1) consistency with existing handlers reduces cognitive load; (2) implementing transition detection requires reading the previous status from `kwargs['status']` and comparing — adds complexity for marginal value; (3) Phase 23 GUI does not rely on transition-time semantics.

### Example B: Source-secret read with Phase 18 error dispatch

```python
# Source: pattern from operator_module/main.py:443-468 (load_values_from_reference Secret branch)
# Adapted for the WarpCredential source-Secret read in reconcile_warpcredential.
import base64
import kr8s
import kopf

def _read_source_secret(name: str, namespace: str, *, ctx: str) -> dict[str, bytes]:
    """Read the WarpCredential's spec.secretRef Secret and return decoded bytes per key.

    Returns: dict[key_name] -> decoded bytes (callers select required keys per type).
    Error dispatch mirrors load_values_from_reference at main.py:449-481.
    """
    try:
        secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
    except kr8s.NotFoundError as e:
        raise kopf.TemporaryError(
            f'{ctx}: source Secret {namespace}/{name} not found (will retry in 30s)',
            delay=30,
        ) from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(
            f'{ctx}: timeout fetching source Secret {namespace}/{name} (will retry in 30s)',
            delay=30,
        ) from e
    except kr8s.ServerError as e:
        status = e.response.status_code if getattr(e, 'response', None) is not None else None
        if status is not None and status >= 500:
            raise kopf.TemporaryError(
                f'{ctx}: API server error {status} fetching Secret {namespace}/{name} (will retry in 30s)',
                delay=30,
            ) from e
        raise kopf.PermanentError(
            f'{ctx}: API error fetching Secret {namespace}/{name}: {e}'
        ) from e

    raw_data = secret.data or {}
    return {k: base64.b64decode(v) for k, v in raw_data.items()}
```
[CITED: operator_module/main.py:449-468 — line-for-line mirror of the Phase 18 dispatch matrix.]

### Example C: Derivation helpers (D-11, D-12, D-13)

```python
# Place near operator_module/main.py:308 (after _render_or_raise; same "pure helper" zone).
import base64
import json

def _b64(s: str) -> str:
    """Standard base64 with padding (D-12)."""
    return base64.b64encode(s.encode('utf-8')).decode('ascii')

def _derive_ngc_payloads(key: str) -> tuple[dict, dict]:
    """Return (apikey_data, docker_data) for nvidia-ngc credential type.

    Both values are dicts ready to drop into a kr8s Secret's 'data' field
    (already base64-encoded per D-13). Caller wraps each in a Secret object
    with the right metadata.name and metadata.type.

    NGC dockerconfigjson uses literal username '$oauthtoken' (nvcr.io convention).
    Locked: D-11 (helper signature), D-12 (base64 with padding, b64 of '$oauthtoken:<key>').
    """
    apikey_data = {'NGC_API_KEY': _b64(key)}
    docker_auth_b64 = _b64(f'$oauthtoken:{key}')
    docker_config = {
        'auths': {
            'nvcr.io': {
                'username': '$oauthtoken',
                'password': key,
                'auth': docker_auth_b64,
            }
        }
    }
    docker_data = {'.dockerconfigjson': _b64(json.dumps(docker_config))}
    return apikey_data, docker_data

def _derive_hf_payload(key: str) -> dict:
    """Return data dict for huggingface credential. D-11."""
    return {'HF_API_KEY': _b64(key)}

def _derive_weka_payload(username: str, token: str, endpoint: str) -> dict:
    """Return data dict for weka-storage credential. D-11.

    Three keys per OPS-06 and ROADMAP Phase 22 success criterion 3.
    """
    return {
        'WEKA_API_USERNAME': _b64(username),
        'WEKA_API_TOKEN': _b64(token),
        'WEKA_API_ENDPOINT': _b64(endpoint),
    }
```
[CITED: CONTEXT.md D-11, D-12, D-13; REQUIREMENTS.md OPS-04, OPS-05, OPS-06.]

### Example D: Reconcile handler skeleton

```python
# Place after operator_module/main.py:1246.
import kopf
import kr8s

_VALID_TYPES = {'nvidia-ngc', 'huggingface', 'weka-storage'}

@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials', field='spec')
@kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
    cred_type = spec.get('type')
    display_name = spec.get('displayName', name)
    ctx = f"WarpCredential {namespace}/{name}({display_name})"

    # OPS-01 (D-08): unknown type -> PermanentError
    if cred_type not in _VALID_TYPES:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'UnknownType', f'spec.type {cred_type!r} not recognized')]
        raise kopf.PermanentError(f'{ctx}: unknown spec.type {cred_type!r}')

    secret_ref = spec.get('secretRef', {})
    src_name = secret_ref.get('name')
    src_key = secret_ref.get('key')
    if not src_name or not src_key:
        # CRD admission should catch this; belt-and-suspenders (D-08 spirit).
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'InvalidSpec', 'spec.secretRef.name and .key required')]
        raise kopf.PermanentError(f'{ctx}: spec.secretRef.name and .key required')

    # OPS-02 (D-07): read source Secret with Phase 18 error dispatch
    try:
        src_data = _read_source_secret(src_name, namespace, ctx=ctx)
    except kopf.TemporaryError:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'KeyMissing',
            f'Source Secret {namespace}/{src_name} not found (retrying)')]
        raise

    if src_key not in src_data:
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'KeyMissing',
            f'Source Secret {namespace}/{src_name} has no key {src_key!r}')]
        raise kopf.PermanentError(f'{ctx}: source Secret {src_name} missing key {src_key!r}')

    key_bytes = src_data[src_key]
    key = key_bytes.decode('utf-8')

    # OPS-03 (D-09): empty/whitespace-only -> PermanentError
    if not key.strip():
        patch.status['conditions'] = [_build_condition(
            'KeyReady', 'False', 'EmptyKey',
            f'Source Secret {namespace}/{src_name}[{src_key}] is empty')]
        raise kopf.PermanentError(f'{ctx}: source Secret key {src_key!r} is empty')

    # Type dispatch
    derived_secrets_list = []

    if cred_type == 'nvidia-ngc':
        apikey_data, docker_data = _derive_ngc_payloads(key)
        apikey_secret = kr8s.objects.Secret({
            'apiVersion': 'v1', 'kind': 'Secret',
            'metadata': {'name': f'warp-{name}-apikey', 'namespace': namespace},
            'type': 'Opaque', 'data': apikey_data,
        })
        docker_secret = kr8s.objects.Secret({
            'apiVersion': 'v1', 'kind': 'Secret',
            'metadata': {'name': f'warp-{name}-docker', 'namespace': namespace},
            'type': 'kubernetes.io/dockerconfigjson', 'data': docker_data,
        })
        _apply_secret_idempotent(apikey_secret, ctx=f'{ctx}: apikey')
        _apply_secret_idempotent(docker_secret, ctx=f'{ctx}: docker')
        derived_secrets_list = [
            {'name': f'warp-{name}-apikey', 'type': 'Opaque'},
            {'name': f'warp-{name}-docker', 'type': 'kubernetes.io/dockerconfigjson'},
        ]

    elif cred_type == 'huggingface':
        hf_data = _derive_hf_payload(key)
        hf_secret = kr8s.objects.Secret({
            'apiVersion': 'v1', 'kind': 'Secret',
            'metadata': {'name': f'warp-{name}-token', 'namespace': namespace},
            'type': 'Opaque', 'data': hf_data,
        })
        _apply_secret_idempotent(hf_secret, ctx=f'{ctx}: token')
        derived_secrets_list = [{'name': f'warp-{name}-token', 'type': 'Opaque'}]

    elif cred_type == 'weka-storage':
        # See Section 5: source Secret contains USERNAME + TOKEN + ENDPOINT keys.
        username = src_data.get('WEKA_API_USERNAME', b'').decode('utf-8')
        token = src_data.get('WEKA_API_TOKEN', b'').decode('utf-8')
        endpoint_from_src = src_data.get('WEKA_API_ENDPOINT', b'').decode('utf-8')
        endpoint = spec.get('endpoint') or endpoint_from_src
        if not username.strip() or not token.strip() or not endpoint.strip():
            patch.status['conditions'] = [_build_condition(
                'KeyReady', 'False', 'EmptyKey',
                'weka-storage requires non-empty WEKA_API_USERNAME, WEKA_API_TOKEN, WEKA_API_ENDPOINT')]
            raise kopf.PermanentError(
                f'{ctx}: weka-storage source Secret missing required keys')
        weka_data = _derive_weka_payload(username, token, endpoint)
        weka_secret = kr8s.objects.Secret({
            'apiVersion': 'v1', 'kind': 'Secret',
            'metadata': {'name': f'warp-{name}-token', 'namespace': namespace},
            'type': 'Opaque', 'data': weka_data,
        })
        _apply_secret_idempotent(weka_secret, ctx=f'{ctx}: token')
        derived_secrets_list = [{'name': f'warp-{name}-token', 'type': 'Opaque'}]

    # Success: update status (D-14)
    conditions = [_build_condition('KeyReady', 'True', 'KeyPresent', 'Derived secrets reconciled')]
    if cred_type == 'nvidia-ngc':
        conditions.append(_build_condition(
            'DockerSecretReady', 'True', 'DockerSecretPresent', 'NGC docker Secret reconciled'))

    patch.status['conditions'] = conditions
    patch.status['derivedSecrets'] = derived_secrets_list
    patch.status['lastSyncTime'] = _now_iso()
    if cred_type == 'weka-storage':
        patch.status['wekaEndpoint'] = spec.get('endpoint', '')

    logger.info(f'{ctx}: reconciled {len(derived_secrets_list)} derived Secret(s)')
```
[CITED: All decisions D-04 through D-15 inline; mirror patterns from main.py:951 (create decorator), 1159 (update decorator with field='spec'), and 720-728 / 944-945 (patch.status writes).]

### Example E: Delete handler (OPS-08, D-05)

```python
@kopf.on.delete('warp.io', 'v1alpha1', 'warpcredentials', optional=True)
def delete_warpcredential(name, namespace, logger, **_):
    """OPS-08: log a warning; do NOT delete derived secrets.

    optional=True prevents kopf from adding a finalizer (per maintainer comment in
    nolar/kopf#701: with optional=True, no finalizer is added; the handler is
    best-effort). Acceptable here because the handler does no destructive work.
    """
    logger.warning(
        f'WarpCredential {namespace}/{name} deleted; derived secrets '
        f'warp-{name}-* are intentionally retained (OPS-08). '
        f'Administrator must delete them manually if no longer needed.'
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pre-flight `exists()` check then `create()` or `patch()` | `try create / except 409 → patch` | Established idiom (since k8s controllers existed) | Single round-trip in steady state; race-free |
| kopf finalizer always added when delete handler present | `optional=True` opts out | kopf 0.27+ | Lets warning-only delete handlers run without blocking deletion |
| `datetime.utcnow()` for ISO timestamps | `datetime.now(timezone.utc)` (py3.12+) | Python 3.12 deprecation | **Project sticks with `utcnow()` for consistency with existing main.py code** |
| `kubernetes.client.ApiException` for K8s API errors | `kr8s.ServerError` with `.response.status_code` | kr8s adoption (Phase 18) | Phase 22 mirrors Phase 18 pattern; no `kubernetes` client usage in reconcile paths |

**Deprecated/outdated:**
- `kubernetes.client.CoreV1Api.create_namespaced_secret` — NOT used; the operator standardized on kr8s. Do not introduce.

## Assumptions Log

> List all claims tagged `[ASSUMED]` in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| (none) | All claims verified or cited. | — | — |

Every factual claim in this RESEARCH.md is tagged `[VERIFIED: ...]` or `[CITED: ...]`. The one decision area that required reasoning (Section 5: weka-storage source Secret structure) is a **recommendation** built from the CRD text + CONTEXT.md + PRD reconciliation, explicitly called out as a recommendation rather than an assumption.

## Open Questions

1. **(RESOLVED — see Section 5)** Open question from CONTEXT.md about weka-storage source-Secret structure. **Recommendation locked.**

2. **None remaining.** The phase scope is fully covered by CONTEXT.md decisions + this research. The planner has enough to write concrete, file-and-line referenced tasks.

## Environment Availability

This phase has no external CLI/service dependencies beyond the existing operator runtime (kopf + kr8s installed in the operator container). No environment audit required.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Operator runtime | ✓ | 3.10 (CI uses 3.10+) | — |
| kopf | All reconcilers | ✓ | 1.38.0 [VERIFIED: `pip show kopf`] | — |
| kr8s | Secret CRUD | ✓ | 0.20.10 [VERIFIED: `pip show kr8s`] | — |

No missing dependencies. Plans for this phase do not need to install anything new.

## Security Domain

### Applicable ASVS Categories

The phase processes secret material (API keys, tokens) read from one Kubernetes Secret and written to another. The relevant ASVS L1 controls:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | The handler does not perform authentication — kr8s uses the operator's mounted ServiceAccount token automatically. |
| V3 Session Management | no | No HTTP sessions; the K8s API call is per-request bearer auth. |
| V4 Access Control | yes | Phase 21 RBAC (Role + RoleBinding) restricts Secret CRUD to the App Store namespace. Verify the operator's ServiceAccount is what's in use (no escalation). |
| V5 Input Validation | yes | `spec.type` enum validation (CRD admission + handler-level belt-and-suspenders D-08); empty-key check D-09; key-name presence check before read. |
| V6 Cryptography | yes | base64 is encoding, NOT crypto — explicitly. The derived Secret is plaintext at rest in etcd (subject to cluster's etcd encryption-at-rest config — out of scope). |
| V7 Error Handling | yes | Exceptions do NOT include the raw key value in error messages. All `PermanentError` / `TemporaryError` messages use metadata only (name, namespace, key-name — never the key value). |
| V8 Data Protection | yes | API-08 / D-03: raw key values never appear in log records at any level. Test enforces (Section 7). |
| V9 Communication | no | All API calls are intra-cluster; TLS is configured by the kr8s default (in-cluster CA) — not Phase 22's concern. |

### Known Threat Patterns for kopf + kr8s + Secret derivation

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted secretRef contents — a malicious admin populates the source Secret with `null` bytes or control characters | Tampering | D-09 (empty-key check) catches whitespace-only. Control chars are passed through to base64 — Kubernetes Secret API accepts arbitrary bytes; downstream consumers (NGC client, HF SDK) are responsible for further validation. |
| Log leakage of raw key (e.g., via traceback) | Information Disclosure | D-03 — never pass the key to logger; assert via caplog test (Section 7). Exception messages reference key NAME only, not VALUE. |
| status.derivedSecrets leakage | Information Disclosure | `derivedSecrets` contains only `name` and `type` (per CRD schema crd.yaml:362-370) — NOT the data. No leakage path. |
| RBAC escalation: operator account misconfigured to access cluster-wide Secrets | Elevation of Privilege | Phase 21 success criterion 5 verifies: Role + RoleBinding (NOT ClusterRole + ClusterRoleBinding) — scoped to App Store namespace only. Phase 22 must NOT introduce ClusterRole-requiring code. |
| Replay: stale CR re-applied with old key | Tampering | `try create / except 409 → patch` always converges to current `spec.secretRef` contents. No history retained. |
| Denial of service: rapid-fire CR creation triggers many reconciles | Denial of Service | kopf's per-resource queue serializes events; kr8s connections are pooled. Out-of-scope to add explicit rate limiting. |

### Phase-specific security verification items (for the planner's `<threat_model>`)

1. **T-22-01 (V8/API-08):** No `logger.{info,warning,debug,error}` call contains the decoded key value. **Test:** `caplog`-based assertion in `test_warp_credential.py` (see Section 7).
2. **T-22-02 (V5/OPS-03):** Empty / whitespace key raises `PermanentError` before any derived Secret write. **Test:** unit test passes empty string and "   " to the handler with a mock source Secret.
3. **T-22-03 (V4):** No call to `kr8s.objects.Secret.get` or `.create` with a namespace OTHER than the CR's `metadata.namespace`. **Test:** grep / code-review check; not unit-testable in isolation.
4. **T-22-04 (V7):** Exception messages reference only metadata (CR name, namespace, key NAME) — never the key VALUE. **Test:** unit test forces each error path and asserts the exception message does NOT contain the test key string.
5. **T-22-05 (V8/OPS-08):** Derived Secrets do NOT have `metadata.ownerReferences` pointing at the WarpCredential CR (would trigger K8s GC on CR deletion). **Test:** integration-level assertion or code-review.

---

## Section-Numbered Answers to `<research_focus>` Items

### 1. kr8s write API surface — create vs patch a Secret (in kr8s 0.20.10)

**Create call signature:** `secret_obj.create()` — no arguments. Implemented as a POST to the namespaced endpoint. Source: `kr8s/_objects.py:360-374`.

**Patch call signature:** `secret_obj.patch(patch_body, *, subresource=None, type=None)` — default merge-patch (Content-Type `application/merge-patch+json`); pass `type="json"` for JSON-Patch (RFC 6902). Source: `kr8s/_objects.py:450-481`.

**Exception on AlreadyExists:** kr8s 0.20.10 does NOT raise a dedicated `AlreadyExistsError`. The 409 Conflict is wrapped in `kr8s.ServerError` with `e.response.status_code == 409`. Source: `kr8s/_api.py:186-201` — verified path: `httpx.HTTPStatusError` with `status_code in [400, 500)` → `raise ServerError(error_message, status=error, response=e.response)`.

**Verified by code-reading installed kr8s 0.20.10 source at `/Users/christopherjenkins/pythonProjects/lib/python3.10/site-packages/kr8s/_api.py:186-201` and `_objects.py:360-481` on 2026-06-11.**

**`.data` structure:** The raw dict passed to `kr8s.objects.Secret({...})` must contain `data` (dict of key → base64-encoded string) per the Kubernetes Secret API. kr8s stores it on `self.raw` and POSTs verbatim. Source: `_objects.py:60-78`.

**`type` field:** Set in the raw dict at the top level: `{'apiVersion': 'v1', 'kind': 'Secret', 'type': 'Opaque'|'kubernetes.io/dockerconfigjson', ...}`.

**Minimal code shape (locked by D-02):**
```python
def _apply_secret_idempotent(secret_obj: kr8s.objects.Secret, *, ctx: str) -> None:
    try:
        secret_obj.create()
    except kr8s.ServerError as e:
        status_code = e.response.status_code if getattr(e, 'response', None) else None
        if status_code == 409:
            secret_obj.patch({'data': secret_obj.raw['data'], 'type': secret_obj.raw['type']})
            return
        if status_code is not None and status_code >= 500:
            raise kopf.TemporaryError(f'{ctx}: API server error {status_code}', delay=30) from e
        raise kopf.PermanentError(f'{ctx}: API error: {e}') from e
    except kr8s.APITimeoutError as e:
        raise kopf.TemporaryError(f'{ctx}: API timeout', delay=30) from e
```

**Pinned version:** `kr8s>=0.17.0` per `operator_module/requirements.txt:7`; locally installed `0.20.10`. Plans should NOT bump the floor version unless tests find an API change.

### 2. kopf status patch — exact shape for WarpCredential

The CRD declares `status.conditions[]`, `status.derivedSecrets[]`, `status.lastSyncTime`, `status.wekaEndpoint` at `weka-app-store-operator-chart/templates/crd.yaml:330-377`. The status subresource is declared at line 287 (`subresources: { status: {} }`), so kopf will automatically PATCH the status subresource when the handler exits.

**Patch shape (locked by CRD schema):**
```python
patch.status['conditions'] = [
    {
        'type': 'KeyReady',                    # required string (crd.yaml:339-341)
        'status': 'True',                       # required enum True|False|Unknown (crd.yaml:342-348)
        'reason': 'KeyPresent',                 # optional string (crd.yaml:349-351)
        'message': 'Derived secrets reconciled',# optional string (crd.yaml:352-354)
        'lastTransitionTime': '2026-06-11T15:00:00.000Z',  # optional date-time string (crd.yaml:355-358)
    },
    # For nvidia-ngc only, append a second condition with type='DockerSecretReady'.
]
patch.status['derivedSecrets'] = [
    {'name': 'warp-<name>-apikey', 'type': 'Opaque'},                # crd.yaml:359-370
    {'name': 'warp-<name>-docker', 'type': 'kubernetes.io/dockerconfigjson'},
]
patch.status['lastSyncTime'] = '2026-06-11T15:00:00.000Z'            # crd.yaml:371-374
patch.status['wekaEndpoint'] = 'https://weka-cluster:14000'          # crd.yaml:375-377 (weka-storage only)
```

**`lastTransitionTime` policy:** Always set to `_now_iso()` on every reconcile — match the project pattern at `main.py:727`, `939`, `972`, `1109`. Implementing "only on change" requires reading the previous `kwargs['status']` and comparing per-condition; out of scope for Phase 22 unless explicitly asked.

[CITED: crd.yaml lines 330-377 for the entire status sub-schema.]
[CITED: operator_module/main.py:720-728 for the existing `patch.status['conditions'] = [{...}]` write pattern.]

### 3. Phase 18 / existing operator error dispatch — exact pattern to mirror

**Helper context:** `load_values_from_reference` at `operator_module/main.py:411-483`. The error-dispatch try/except block is at **lines 439-468**. The pattern:

```python
# operator_module/main.py:449-468 (verbatim)
except kr8s.NotFoundError as e:
    raise kopf.TemporaryError(
        f"{ctx}: {kind} {namespace}/{name} not found (will retry in 30s)",
        delay=30,
    ) from e
except kr8s.APITimeoutError as e:
    raise kopf.TemporaryError(
        f"{ctx}: timeout fetching {kind} {namespace}/{name} (will retry in 30s)",
        delay=30,
    ) from e
except kr8s.ServerError as e:
    status = e.response.status_code if getattr(e, "response", None) is not None else None
    if status is not None and status >= 500:
        raise kopf.TemporaryError(
            f"{ctx}: API server error {status} fetching {kind} {namespace}/{name} (will retry in 30s)",
            delay=30,
        ) from e
    raise kopf.PermanentError(
        f"{ctx}: API error fetching {kind} {namespace}/{name}: {e}"
    ) from e
```

**Mapping for reuse:** Phase 22 plans should reference `operator_module/main.py:449-468` via `read_first` so executors mirror line-for-line. Add `kr8s.ConnectionClosedError` to the temporary set if seen in practice (currently not in the Phase 18 pattern; not strictly required).

### 4. kopf decorators for `warp.io/v1alpha1/warpcredentials`

**Decorator syntax (verified via `inspect.signature(kopf.on.delete)` on installed kopf 1.38.0):**

```python
@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials', field='spec')
@kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
    ...

@kopf.on.delete('warp.io', 'v1alpha1', 'warpcredentials', optional=True)
def delete_warpcredential(name, namespace, logger, **_):
    ...
```

**Positional arguments:** `(group, version, plural)` — order matches existing project usage at `main.py:951, 1159, 1175`.

**Multiple-decorator stacking:** kopf supports stacking. Each decorator registers an independent handler that resolves to the same function. Confirmed at `docs.kopf.dev/en/stable/handlers/`: "It is a common pattern to declare both creation and resuming handlers pointing to the same function."

**Finalizer behavior with `optional=True`:** Without `optional=True`, kopf adds a finalizer (default name `kopf.zalando.org/KopfFinalizerMarker`) to the CR, which blocks deletion until kopf removes it (i.e., requires the operator to be running for the CR to disappear). With `optional=True`, NO finalizer is added; the delete handler is best-effort (per kopf maintainer comment in nolar/kopf#701: "the execution of deletion handlers depends on luck — whatever is faster"). For Phase 22 this is correct: OPS-08 says "log warning, do not delete derived secrets," so missing the warning is acceptable.

**`field='spec'` on `@kopf.on.update`:** Matches the existing WekaAppStore pattern at `main.py:1159`. This ensures the handler does NOT re-fire when the operator's own status writes are observed — preventing infinite reconcile loops. **The planner MUST include `field='spec'` on the update decorator.**

[CITED: `inspect.signature(kopf.on.delete)` on installed kopf 1.38.0 shows `optional: Optional[bool] = None`.]
[CITED: operator_module/main.py:1159 — `@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')` is the established project pattern.]
[CITED: github.com/nolar/kopf#701 maintainer comment.]

### 5. weka-storage source-Secret structure — RECOMMENDATION (resolves CONTEXT.md open question)

**The question:** Does `spec.secretRef.key` (singular) provide ONLY the token, requiring `spec.endpoint` and a separate username field, OR does the referenced source Secret hold all three keys (`WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`)?

**Evidence:**

| Source | Claim | Line |
|--------|-------|------|
| CRD schema | `secretRef.required` = `[name, key]` (singular `key`) | crd.yaml:313-315 |
| REQUIREMENTS.md OPS-06 | Derived `warp-<name>-token` has THREE keys: `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT` | REQUIREMENTS.md:26 |
| ROADMAP success criterion 3 | Three keys in the derived Secret | ROADMAP.md:228 |
| PRD §2 (operator reconciler) | "Create or patch `warp-<name>-token` (Opaque, two keys: `WEKA_API_TOKEN` and `WEKA_API_ENDPOINT`)" — TWO keys, contradicts above | PRD-secret-management-overhaul.md:158 |
| PRD §4 (API request body) | POST /api/credentials body for weka-storage takes `username`, `key`, `endpoint` (THREE separate fields) | PRD-secret-management-overhaul.md:335-343 |
| PRD §5 (API-05 implementation) | "Read `WEKA_API_TOKEN`, `WEKA_API_USERNAME`, and `WEKA_API_ENDPOINT` from the credential's raw Secret" — THREE keys, contradicts §2 | PRD-secret-management-overhaul.md:627 |
| PRD §scenario 5 | "kubectl get secret warp-weka-cluster-primary-token ... exists with keys `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, and `WEKA_API_ENDPOINT`" | PRD-secret-management-overhaul.md:737 |

**Resolution:** REQUIREMENTS.md OPS-06, ROADMAP success criterion 3, and PRD §5 + PRD §scenario 5 all agree on **three keys**. PRD §2 (which says two keys) is **inconsistent with itself** (the same PRD says three keys later) and is the only place that says two. The contradiction inside the PRD is the source of the open question.

**RECOMMENDATION (lock this for the planner):**

The Phase 23 API backend creates the raw source Secret (`warp-cred-<slug>`) from the POST /api/credentials body. For weka-storage, the body provides `username`, `key` (the token), and `endpoint`. The backend stores **all three** in the raw source Secret with hard-coded key names:

- `WEKA_API_USERNAME` (from request body `username`)
- `WEKA_API_TOKEN` (from request body `key`)
- `WEKA_API_ENDPOINT` (from request body `endpoint`)

The CRD's `secretRef.key` field for weka-storage is **effectively unused** — it should be set to `WEKA_API_TOKEN` by convention (the primary credential field), and the operator reads all three keys directly using the hard-coded names. The CRD field exists to satisfy schema uniformity across all three types; for weka-storage the operator ignores `secretRef.key` and instead reads the well-known hard-coded keys.

**Concrete plan-level instructions:**

1. Operator reads the source Secret (one `kr8s.objects.Secret.get` call).
2. From the resulting `.data` dict, the operator extracts `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT` by literal key name. If any are missing or empty/whitespace-only, raise `kopf.PermanentError` with reason `EmptyKey` and a message naming the missing key.
3. `status.wekaEndpoint` is copied from `spec.endpoint` (which the backend mirrors from the body); fall back to the source Secret's `WEKA_API_ENDPOINT` only if `spec.endpoint` is absent (defense in depth).
4. The derived `warp-<name>-token` Secret contains exactly those three keys, base64-encoded.
5. The CRD's `spec.secretRef.key` value is read for symmetry-check purposes only — the operator does not use it to index the source Secret for weka-storage type. Document this in a code comment so reviewers don't try to use it.

**Why this resolution is correct:**

- It matches REQUIREMENTS.md (the source of truth for v1) OPS-06.
- It matches the ROADMAP Phase 22 success criterion 3 (the most recent acceptance bar).
- It matches the Phase 24 GUI form (three input fields: Name, Username, API Token, Endpoint per REQUIREMENTS.md GUI-05).
- It matches the API request body shape (PRD §4 lines 335-343).
- The PRD §2 "two keys" line is a stale draft inconsistent with the rest of the PRD; treating it as authoritative would force the GUI/API to omit username, breaking GUI-05 and API-03.

**For the planner's locked-truth block:** "weka-storage source Secrets contain three keys with literal names `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`. The CRD `spec.secretRef.key` is set to `WEKA_API_TOKEN` by convention but the operator reads all three by hard-coded name."

[CITED: REQUIREMENTS.md:26 (OPS-06); ROADMAP.md:228 (Phase 22 success criterion 3); PRD-secret-management-overhaul.md:335-343 (API body), :627 (API-05 read), :737 (scenario 5).]

### 6. Test isolation strategy

**Unit tests (helper functions, no kr8s):**

```python
# operator_module/tests/test_warp_credential.py — derivation tests
import base64
import json
from main import _derive_ngc_payloads, _derive_hf_payload, _derive_weka_payload

def test_ngc_apikey_data_is_b64_encoded():
    apikey, _docker = _derive_ngc_payloads('my-secret-key')
    assert apikey == {'NGC_API_KEY': base64.b64encode(b'my-secret-key').decode('ascii')}

def test_ngc_docker_auth_is_oauthtoken_b64():
    _apikey, docker = _derive_ngc_payloads('my-secret-key')
    docker_json = json.loads(base64.b64decode(docker['.dockerconfigjson']))
    assert docker_json['auths']['nvcr.io']['username'] == '$oauthtoken'
    assert docker_json['auths']['nvcr.io']['password'] == 'my-secret-key'
    assert docker_json['auths']['nvcr.io']['auth'] == base64.b64encode(b'$oauthtoken:my-secret-key').decode('ascii')

def test_hf_payload_has_only_hf_api_key():
    data = _derive_hf_payload('hf-token-xyz')
    assert set(data.keys()) == {'HF_API_KEY'}
    assert base64.b64decode(data['HF_API_KEY']).decode() == 'hf-token-xyz'

def test_weka_payload_three_keys():
    data = _derive_weka_payload('admin', 'tok-abc', 'https://weka-cluster:14000')
    assert set(data.keys()) == {'WEKA_API_USERNAME', 'WEKA_API_TOKEN', 'WEKA_API_ENDPOINT'}
```

**Handler-level tests (with minimal kr8s mocking):** Mirror `test_appstack.py` helpers. The smallest fixture set:

```python
# Mirror test_appstack.py:62-69
def _make_kr8s_secret(data_dict):
    """Return a MagicMock whose .data maps each key to base64-encoded value."""
    secret = MagicMock()
    secret.data = {
        k: base64.b64encode(v.encode('utf-8') if isinstance(v, str) else v).decode('utf-8')
        for k, v in data_dict.items()
    }
    return secret

def _make_kr8s_server_error(status_code):
    """Mirror test_appstack.py:72-80."""
    import kr8s
    err = kr8s.ServerError('server error')
    response = MagicMock()
    response.status_code = status_code
    err.response = response
    return err

def _make_patch_obj():
    """kopf patch object stand-in — supports patch.status['key'] = value."""
    patch = MagicMock()
    patch.status = {}
    return patch

# Example: handler-level test for KeyMissing
def test_reconcile_raises_temporary_error_when_secret_missing():
    import kr8s
    from main import reconcile_warpcredential

    patch_obj = _make_patch_obj()
    with pytest.raises(kopf.TemporaryError):
        with patch('main.kr8s.objects.Secret.get',
                   side_effect=kr8s.NotFoundError('not found')):
            reconcile_warpcredential(
                body={}, spec={'type': 'huggingface', 'displayName': 'HF Test',
                               'secretRef': {'name': 'src', 'key': 'token'}},
                name='hf-test', namespace='weka-app-store', patch=patch_obj,
                logger=logging.getLogger('test'),
            )
    # Status patched with KeyMissing condition BEFORE the raise
    assert patch_obj.status['conditions'][0]['reason'] == 'KeyMissing'

# Example: idempotency — second call patches instead of creating
def test_apply_secret_idempotent_patches_on_409():
    from main import _apply_secret_idempotent

    secret = MagicMock()
    secret.raw = {'data': {'k': 'v'}, 'type': 'Opaque'}
    secret.create.side_effect = _make_kr8s_server_error(409)
    secret.patch = MagicMock()

    _apply_secret_idempotent(secret, ctx='test')

    secret.create.assert_called_once()
    secret.patch.assert_called_once_with({'data': {'k': 'v'}, 'type': 'Opaque'})

# Example: 500 server error becomes TemporaryError
def test_apply_secret_idempotent_500_raises_temporary():
    from main import _apply_secret_idempotent
    secret = MagicMock()
    secret.create.side_effect = _make_kr8s_server_error(503)
    with pytest.raises(kopf.TemporaryError):
        _apply_secret_idempotent(secret, ctx='test')
```

**Minimum fixtures needed (one file, ~250 LOC total):**

1. `_make_kr8s_secret(data_dict)` — copied from `test_appstack.py:62-69`
2. `_make_kr8s_server_error(status_code)` — copied from `test_appstack.py:72-80`
3. `_make_patch_obj()` — new helper (kopf patch stand-in)
4. Patch target paths: `main.kr8s.objects.Secret.get` (read source) and `main.kr8s.objects.Secret` (constructor for derived Secret). For the constructor, monkey-patch with a factory that records the dict and returns a MagicMock supporting `.create()`, `.patch()`, `.raw`.

### 7. Logging safety — caplog-based assertion (API-08, D-03)

```python
# operator_module/tests/test_warp_credential.py — log safety
import logging
import re

def test_no_key_in_logs_anywhere(caplog):
    """API-08: raw key value MUST NOT appear in any log record at any level."""
    from main import reconcile_warpcredential

    test_key = 'super-secret-test-key-value-do-not-leak-42'
    src_secret = _make_kr8s_secret({'NGC_API_KEY': test_key})
    patch_obj = _make_patch_obj()

    with caplog.at_level(logging.DEBUG):  # capture everything
        with patch('main.kr8s.objects.Secret.get', return_value=src_secret), \
             patch('main.kr8s.objects.Secret', side_effect=_make_secret_class_mock()):
            reconcile_warpcredential(
                body={}, spec={'type': 'nvidia-ngc', 'displayName': 'NGC Test',
                               'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'}},
                name='ngc-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    # Across ALL captured log records (any level, any logger), the test key must not appear.
    for record in caplog.records:
        msg = record.getMessage()
        assert test_key not in msg, f'Key leaked in log: {record.levelname} {msg}'
        assert test_key not in str(record.args or ''), f'Key leaked in args: {record.args}'

def test_no_key_in_exception_message_on_empty_key(caplog):
    """When raising PermanentError for empty key, message must not contain the key."""
    from main import reconcile_warpcredential

    test_key = 'this-should-not-appear-in-error'  # but key value will be whitespace
    src_secret = _make_kr8s_secret({'NGC_API_KEY': '   '})  # whitespace -> empty
    patch_obj = _make_patch_obj()

    with pytest.raises(kopf.PermanentError) as exc_info:
        with patch('main.kr8s.objects.Secret.get', return_value=src_secret):
            reconcile_warpcredential(
                body={}, spec={'type': 'nvidia-ngc', 'displayName': 'NGC Test',
                               'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'}},
                name='ngc-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    # Exception message references KEY NAME, not VALUE.
    assert 'NGC_API_KEY' in str(exc_info.value)  # key name is OK to reference
    # No control character / whitespace key value leak
    assert '   ' not in str(exc_info.value).replace(' ', '_')  # crude check
```

**Existing project precedent:** No existing test in `operator_module/tests/` uses caplog. This will be the first. The pattern is canonical pytest + `caplog` fixture; verify with `pip show pytest` → ≥ 8.0 supports `caplog.records`, `caplog.at_level()`.

### 8. Threat model inputs — security-relevant data flows for the planner's `<threat_model>` block

| Threat | Data flow | Asset | Mitigation | ASVS L1 |
|--------|-----------|-------|------------|---------|
| T-22-01: Log leakage of raw key | Source Secret `.data[key]` → decoded → log message | Cleartext API key | D-03; caplog test; never f-string the decoded key | V8 |
| T-22-02: Empty/whitespace key passes through to derived Secret | Source Secret `.data[key]` → derived Secret without validation | Downstream consumer fails authentication; possible audit gap | D-09 / OPS-03 enforces non-empty; PermanentError raised | V5 |
| T-22-03: Operator escalates beyond namespace | RBAC misconfiguration via Helm | Cluster-wide Secret CRUD | Phase 21 Role+RoleBinding (NOT ClusterRole); verify in Helm chart unchanged | V4 |
| T-22-04: Exception messages leak key value | Exception traceback → kopf event log → cluster log | Cleartext key | All `kopf.*Error` messages reference key NAME only; test asserts | V7, V8 |
| T-22-05: Derived Secrets garbage-collected on CR delete | `metadata.ownerReferences` on derived Secret | Workload outage | Do NOT add ownerReferences; OPS-08 enforces survival | V8 |
| T-22-06: Status leaks credential | `status.derivedSecrets`, `status.wekaEndpoint`, `status.conditions[].message` | Cleartext credential or token | CRD schema constrains; status writes use only names/types/timestamps (D-14); message strings never include key value | V8 |
| T-22-07: Replay of stale spec | Old `spec.secretRef` resolved after rename | Derived Secret has stale key | Reconcile always reads current `spec.secretRef`; idempotent create-or-patch overwrites | V5 |
| T-22-08: Tampered source Secret content | Malicious admin sets non-UTF-8 bytes | base64 decode fails, downstream consumer rejects | `key_bytes.decode('utf-8')` raises UnicodeDecodeError — wrap in `try/except UnicodeDecodeError → PermanentError` (recommendation: add to the plan) | V5 |

**ASVS L1 categories covered:** V4 (Access Control), V5 (Input Validation), V7 (Error Handling), V8 (Data Protection). V6 (Cryptography) is not implicated (base64 is encoding, not crypto). V2/V3 (Authn/Session) not applicable.

## Project Constraints (from CLAUDE.md)

Extracted directives from `./CLAUDE.md` that the planner must reflect in PLAN.md:

1. **Single-file operator module by design.** `operator_module/main.py` is large by design — match this style, do NOT introduce new sub-packages for Phase 22.
2. **Quote style.** The operator module uses **single quotes** for shell args (project convention); for Python dict literals, the operator module currently uses single quotes (e.g., `'type': 'Ready'` at main.py:721-728). Match this style — do NOT switch to double quotes within `operator_module/main.py`.
3. **No `print()` in operator paths.** Use the `logger` kwarg (per-handler) or `logging.getLogger(__name__)`. Never `print(...)`.
4. **No linter/formatter enforced.** Follow PEP 8 and match the existing module's quote style.
5. **Test command:** `PYTHONPATH=operator_module pytest operator_module/tests/ -v` (CLAUDE.md Commands section).
6. **Helm chart packaging:** Not applicable to Phase 22 unless the planner decides a Chart.yaml bump is needed (no CRD or RBAC changes in Phase 22 — Phase 21 already shipped 0.1.64). **Recommendation: no Chart.yaml bump needed for Phase 22** (operator binary changes don't require Helm chart version bump; the operator image tag bump is a separate concern handled at deploy time).

## Sources

### Primary (HIGH confidence — VERIFIED)
- `kr8s` source code, installed 0.20.10 — `/Users/christopherjenkins/pythonProjects/lib/python3.10/site-packages/kr8s/_api.py:155-202` and `_objects.py:360-481` (create/patch/exception dispatch)
- `kopf` 1.38.0 — `inspect.signature(kopf.on.delete)` confirms `optional=True` parameter
- `weka-app-store-operator-chart/templates/crd.yaml:279-382` — CRD schema (locked by Phase 21)
- `operator_module/main.py:411-483` — Phase 18 kr8s exception dispatch pattern (canonical)
- `operator_module/main.py:951-1245` — existing kopf handlers (decorator syntax, patch.status pattern)
- `operator_module/tests/test_appstack.py:55-90` — kr8s mocking patterns
- `.planning/REQUIREMENTS.md` (OPS-01..OPS-09, API-08)
- `.planning/ROADMAP.md` Phase 22 success criteria
- `.planning/PRD-secret-management-overhaul.md` §2 + §4 + §5 + scenario 5 (weka-storage source Secret keys)
- `.planning/phases/22-operator-warpcredential-reconciler/22-CONTEXT.md` (locked decisions)

### Secondary (MEDIUM confidence — CITED)
- [Kopf documentation — Handlers](https://docs.kopf.dev/en/stable/handlers/) — stacked decorators pattern
- [Kopf documentation — Configuration](https://docs.kopf.dev/en/stable/configuration/) — finalizer name and persistence
- [kr8s documentation — Object API](https://docs.kr8s.org/en/stable/object.html) — create/patch usage
- [kr8s source — kr8s_org/kr8s GitHub](https://github.com/kr8s-org/kr8s/blob/main/kr8s/_exceptions.py) — exception class list

### Tertiary (LOW confidence — UNVERIFIED via this session, kept for context)
- [kopf issue #701 — on.delete with optional=True](https://github.com/nolar/kopf/issues/701) — maintainer comment quoted from GitHub API fetch; behavior verified by signature, but the specific edge case (whether handler fires when no finalizer + fast K8s delete) is a known kopf-internal behavior described by the maintainer

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — `kr8s` and `kopf` versions verified by direct introspection of installed packages; CRD schema verified by reading the file directly.
- Architecture: **HIGH** — All patterns mirror existing project code (line-cited).
- Error dispatch: **HIGH** — Mirrors Phase 18 line-for-line; pattern is unambiguous.
- Pitfalls: **MEDIUM-HIGH** — Pitfall list is comprehensive; the one nuance (optional=True + missed delete event) is documented from kopf maintainer.
- weka-storage source Secret structure: **MEDIUM** — recommendation reconciles a self-contradictory PRD; the recommendation matches three independent sources (REQUIREMENTS.md, ROADMAP, PRD §scenario 5) against one outlier (PRD §2).
- Tests: **HIGH** — Patterns directly mirror `test_appstack.py`; caplog usage is standard pytest.

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (30 days — stack is stable, no upcoming kr8s/kopf major releases that would invalidate findings).
