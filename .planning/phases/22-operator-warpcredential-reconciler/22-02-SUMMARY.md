---
phase: 22-operator-warpcredential-reconciler
plan: "02"
subsystem: operator
tags: [operator, kopf, kr8s, secrets, warpcredential, reconciler, status]
dependency_graph:
  requires:
    - _VALID_WARPCRED_TYPES (22-01)
    - _b64 (22-01)
    - _now_iso (22-01)
    - _build_condition (22-01)
    - _derive_ngc_payloads (22-01)
    - _derive_hf_payload (22-01)
    - _derive_weka_payload (22-01)
    - _read_source_secret (22-01)
    - _apply_secret_idempotent (22-01)
    - delete_warpcredential (22-01)
  provides:
    - reconcile_warpcredential (create/update/resume handler)
  affects:
    - operator_module/main.py
tech_stack:
  added: []
  patterns:
    - stacked kopf decorators (on.create / on.update(field='spec') / on.resume) on single function
    - status-patch-before-raise on all failure branches (S-4)
    - dispatch on spec.type to per-type _derive_* helpers (D-11)
    - weka-storage reads three hard-coded keys by literal name (RESEARCH §5)
key_files:
  created: []
  modified:
    - operator_module/main.py
decisions:
  - "D-04: three stacked decorators (create/update field='spec'/resume) on one reconcile function"
  - "D-07: TemporaryError for NotFoundError; status patched with KeyMissing before raise"
  - "D-08: PermanentError for unknown spec.type; status patched with UnknownType before raise"
  - "D-09: PermanentError for empty/whitespace key; status patched with EmptyKey before raise"
  - "D-14: success path writes conditions + derivedSecrets + lastSyncTime (+ wekaEndpoint for weka-storage)"
  - "D-15: all failure paths patch status.conditions before raising"
  - "T-22-01/API-08: no key value, token, username, or endpoint passed to any logger or exception message"
  - "T-22-05: no ownerReferences on any derived Secret"
  - "RESEARCH §5: weka-storage reads WEKA_API_USERNAME + WEKA_API_TOKEN + WEKA_API_ENDPOINT by literal name; spec.secretRef.key is for schema symmetry only"
metrics:
  duration: ~25min
  completed: "2026-06-11"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 1
---

# Phase 22 Plan 02: reconcile_warpcredential Handler Summary

Added the `reconcile_warpcredential` kopf handler to `operator_module/main.py`, wiring the Plan 01 helpers into a complete three-decorator reconcile loop that satisfies OPS-01..OPS-07, OPS-09, and API-08.

## What Was Built

### New code in operator_module/main.py

Lines 1485–1674 — `reconcile_warpcredential` with stacked decorators (pre-function comment block starting at 1485).

#### Decorator stack (lines 1489–1491)

```python
@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
@kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials', field='spec')
@kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
```

`field='spec'` on `@kopf.on.update` prevents infinite loops when the operator's own status writes are observed (Pitfall 3, D-04). A comment above the stacked decorators cites Pitfall 3 + D-04 so future maintainers do not remove it.

#### Failure branches with their `reason` strings (for Plan 03 assertions)

| Trigger | reason | kopf error type | Line range |
|---------|--------|-----------------|------------|
| `spec.type` not in `_VALID_WARPCRED_TYPES` | `'UnknownType'` | `PermanentError` | ~1528–1532 |
| `spec.secretRef.name` or `.key` absent | `'InvalidSpec'` | `PermanentError` | ~1538–1542 |
| Source Secret not found (`_read_source_secret` raises `TemporaryError`) | `'KeyMissing'` | re-raised `TemporaryError` | ~1547–1551 |
| API error reading source Secret (`_read_source_secret` raises `PermanentError`) | `'KeyReadError'` | re-raised `PermanentError` | ~1552–1556 |
| `src_key` not in decoded source Secret data (nvidia-ngc / huggingface) | `'KeyMissing'` | `PermanentError` | ~1563–1568 |
| Key value is empty or whitespace-only (nvidia-ngc / huggingface) | `'EmptyKey'` | `PermanentError` | ~1573–1578 |
| `WEKA_API_USERNAME` / `WEKA_API_TOKEN` / `WEKA_API_ENDPOINT` absent | `'KeyMissing'` | `PermanentError` | ~1631–1636 |
| Any weka-storage key value is empty or whitespace-only | `'EmptyKey'` | `PermanentError` | ~1637–1642 |

All failure paths call `patch.status['conditions'] = [_build_condition(...)]` BEFORE the `raise` statement (S-4, D-15).

#### Type dispatch summary

| `spec.type` | Derived Secrets | Helper called |
|-------------|----------------|---------------|
| `nvidia-ngc` | `warp-{name}-apikey` (Opaque, `NGC_API_KEY`) + `warp-{name}-docker` (`kubernetes.io/dockerconfigjson`) | `_derive_ngc_payloads(key)` |
| `huggingface` | `warp-{name}-token` (Opaque, `HF_API_KEY`) | `_derive_hf_payload(key)` |
| `weka-storage` | `warp-{name}-token` (Opaque, `WEKA_API_USERNAME` + `WEKA_API_TOKEN` + `WEKA_API_ENDPOINT`) | `_derive_weka_payload(username, token, endpoint)` |

Each derived Secret is applied via `_apply_secret_idempotent` (Plan 01) — no direct `kr8s.objects.Secret.create()` or `.patch()` calls in the handler body.

#### Success status write (OPS-07, D-14)

```python
patch.status['conditions'] = conditions          # includes DockerSecretReady for nvidia-ngc
patch.status['derivedSecrets'] = derived_secrets_list
patch.status['lastSyncTime'] = _now_iso()
# weka-storage only:
patch.status['wekaEndpoint'] = spec.get('endpoint', '')
```

#### weka-storage key extraction (RESEARCH §5)

Source Secret keys are read by literal name (`WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`). `spec.secretRef.key` is present for CRD schema symmetry only — the handler does not index the source Secret by its value for weka-storage type. A code comment documents this.

## Verification Results

All acceptance criteria passed:

```
@kopf.on.create decorator                   → 1
@kopf.on.update(field='spec') decorator     → 1
@kopf.on.resume decorator                   → 1
def reconcile_warpcredential                → 1
_derive_* helper call count                 → 9  (>= 4 required)
_apply_secret_idempotent call count         → 5  (>= 4 required; 1 def + 4 call sites)
'KeyMissing' count                          → 5  (>= 2 required)
'EmptyKey' count                            → 3  (>= 1 required)
'UnknownType' count                         → 2  (>= 1 required)
'KeyPresent' count                          → 1  (>= 1 required)
'DockerSecretReady' count                   → 1  (>= 1 required)
patch.status['conditions'] writes           → 10 (>= 5 new required)
patch.status['derivedSecrets']              → 1  (>= 1 required)
patch.status['lastSyncTime']                → 1  (>= 1 required)
patch.status['wekaEndpoint']                → 1  (>= 1 required)
ownerReferences occurrences                 → 0  (required = 0, T-22-05)
warp-{name}-apikey                          → 3  (>= 1 required)
warp-{name}-docker                          → 3  (>= 1 required)
warp-{name}-token                           → 5  (>= 1 required)
kubernetes.io/dockerconfigjson              → 2  (>= 1 required)
logger.info/warning in new handler          → 1  (success log line)
logging.*(.*key) module-level calls         → 0  (required = 0, API-08)
python -m py_compile operator_module/main.py → exit 0
```

## Security / Threat Model

All Plan 02 threat mitigations implemented:

- **T-22-01** — `logger.info` at handler exit uses only `ctx` (CR name+namespace+displayName) and `len(derived_secrets_list)`. The variables `key`, `token`, `username`, `endpoint` are never passed to any logger call.
- **T-22-02** — Empty/whitespace check on `key.strip()` (nvidia-ngc, huggingface) and per-key `.strip()` checks (weka-storage) BEFORE any `_derive_*` call; `PermanentError` raised with reason `EmptyKey`.
- **T-22-03** — Handler always passes `namespace=namespace` (CR metadata.namespace) to `_read_source_secret` and to every derived-Secret `metadata.namespace`. No alternative namespace string constructed.
- **T-22-04** — Exception messages reference `src_name`, `src_key` (key NAME), and `cred_type`. The resolved value of any key is never interpolated into an error message.
- **T-22-05** — Every `kr8s.objects.Secret({...})` dict has `metadata` with ONLY `name` and `namespace`. Zero occurrences of `ownerReferences` in the file (grep gate confirmed).
- **T-22-06** — `patch.status['conditions'][].message` strings contain only namespace/name metadata. `patch.status['derivedSecrets']` items have only `name` + `type`. `patch.status['wekaEndpoint']` mirrors `spec.endpoint` (already public CR data).

## Deviations from Plan

None — plan executed exactly as written.

The only implementation choice not explicitly dictated by the plan: the weka-storage validation loop iterates over the three keys in order (`WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`), raising `PermanentError` with an individual per-key message at the first missing/empty key. This matches RESEARCH §5 step 2 ("If any are missing or empty/whitespace-only, raise kopf.PermanentError... naming the missing key") and produces cleaner error messages than a combined check.

## Known Stubs

None. `reconcile_warpcredential` is fully implemented end-to-end. All three credential type dispatch branches are complete.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes introduced beyond what was planned. The handler operates entirely within the existing kopf + kr8s trust boundary.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| operator_module/main.py | FOUND |
| .planning/phases/22-operator-warpcredential-reconciler/22-02-SUMMARY.md | FOUND (this file) |
| Commit 8d808e9 (Task 2.1) | FOUND |
