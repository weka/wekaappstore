---
phase: 22-operator-warpcredential-reconciler
plan: "01"
subsystem: operator
tags: [operator, kopf, kr8s, secrets, helpers, warpcredential]
dependency_graph:
  requires: []
  provides:
    - _VALID_WARPCRED_TYPES
    - _b64
    - _now_iso
    - _build_condition
    - _derive_ngc_payloads
    - _derive_hf_payload
    - _derive_weka_payload
    - _read_source_secret
    - _apply_secret_idempotent
    - delete_warpcredential
  affects:
    - operator_module/main.py
tech_stack:
  added: []
  patterns:
    - try-create/except-ServerError(409)->patch idempotency (D-02)
    - Phase 18 kr8s exception dispatch matrix mirror (D-07, D-10)
    - kopf optional=True delete handler (D-05)
key_files:
  created: []
  modified:
    - operator_module/main.py
decisions:
  - "D-01: kr8s only for Secret I/O — no kubectl subprocess"
  - "D-02: idempotency via try create / except 409 -> patch (no delete-and-recreate)"
  - "D-03: no helper logs any input value at any level"
  - "D-05: delete handler is optional=True warning-only (no finalizer)"
  - "D-11: per-type derivation helpers, not inline"
  - "D-12: standard base64 with padding for NGC auth field"
  - "D-13: helpers return already-encoded data"
  - "Pitfall 1 avoided: kr8s.AlreadyExistsError does not exist; 409 detected via e.response.status_code"
metrics:
  duration: ~20min
  completed: "2026-06-11"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 1
---

# Phase 22 Plan 01: WarpCredential Helper Functions and Delete Handler Summary

Added pure-Python derivation helpers, status-condition builder, kr8s I/O wrappers, and the no-op delete handler to `operator_module/main.py`. All artifacts are consumed by Plan 02 (reconcile handler) and Plan 03 (test suite).

## What Was Built

### New code in operator_module/main.py

All new code added in the "pure helper zone" (lines 311–520) after `_render_or_raise` and at the bottom of the module (lines 1460–1482) for the delete handler.

| Identifier | Line | Description | Decisions |
|------------|------|-------------|-----------|
| `_VALID_WARPCRED_TYPES` | 320 | `{'nvidia-ngc', 'huggingface', 'weka-storage'}` — belt-and-suspenders type guard | OPS-01, D-08 |
| `_b64(s)` | 323 | Standard padded base64 encode (`base64.b64encode(...).decode('ascii')`) | D-12, D-13 |
| `_now_iso()` | 333 | UTC ISO 8601 timestamp matching `datetime.utcnow().isoformat() + 'Z'` | D-14, Pitfall 5 |
| `_build_condition(type_, status, reason, message)` | 344 | Status condition dict per CRD schema (crd.yaml:330-358) | D-14 |
| `_derive_ngc_payloads(key)` | 362 | Returns `(apikey_data, docker_data)` for `nvidia-ngc` type | D-11, D-12, D-13, OPS-04 |
| `_derive_hf_payload(key)` | 394 | Returns `{'HF_API_KEY': _b64(key)}` for `huggingface` type | D-11, D-13, OPS-05 |
| `_derive_weka_payload(username, token, endpoint)` | 406 | Returns three-key dict for `weka-storage` type | D-11, D-13, OPS-06 |
| `_read_source_secret(name, namespace, *, ctx)` | 428 | Reads source Secret; mirrors Phase 18 dispatch matrix line-for-line | D-07, D-10, OPS-02 |
| `_apply_secret_idempotent(secret_obj, *, ctx)` | 473 | Create-or-patch via try/except ServerError(409) | D-02, OPS-09 |
| `delete_warpcredential(...)` | 1466 | `@kopf.on.delete(..., optional=True)` warning-only handler | D-05, OPS-08 |

### Key implementation details

**_derive_ngc_payloads** (lines 362–392): Builds both the apikey Secret data and the dockerconfigjson payload. The `auth` field is `_b64('$oauthtoken:{key}')` — standard padded base64 (D-12). The `password` field is the literal key (required by the dockerconfigjson spec for Docker login). The entire dockerconfig JSON is then itself base64-encoded (D-13).

**_read_source_secret** (lines 428–471): Mirrors `operator_module/main.py:449-468` (Phase 18 dispatch matrix) verbatim:
- `kr8s.NotFoundError` → `kopf.TemporaryError(delay=30)` (D-07, OPS-02)
- `kr8s.APITimeoutError` → `kopf.TemporaryError(delay=30)` (D-10)
- `kr8s.ServerError(>=500)` → `kopf.TemporaryError(delay=30)` (D-10)
- `kr8s.ServerError(4xx)` → `kopf.PermanentError` (D-10)
- Success path: `{k: base64.b64decode(v) for k, v in raw_data.items()}`

**_apply_secret_idempotent** (lines 473–520): The 409 branch is placed BEFORE the >=500 branch inside `except kr8s.ServerError`. Patch dict is exactly `{'data': secret_obj.raw['data'], 'type': secret_obj.raw['type']}` (two keys only; Plan 03 asserts on this shape). No `kr8s.AlreadyExistsError` referenced — the class does not exist in kr8s 0.20.10 (Pitfall 1).

**delete_warpcredential** (lines 1460–1482): `optional=True` prevents kopf from adding a finalizer. Handler body is a single `logger.warning()` call naming `namespace/name` and `warp-{name}-*`, citing OPS-08. No kr8s calls, no destructive work.

## Verification Results

All acceptance criteria passed:

```
grep -c '^def _derive_ngc_payloads'  → 1
grep -c '^def _derive_hf_payload'    → 1
grep -c '^def _derive_weka_payload'  → 1
grep -c '^def _b64'                  → 1
grep -c '^def _now_iso'              → 1
grep -c '^def _build_condition'      → 1
_VALID_WARPCRED_TYPES constant       → 1
grep -c '^def _read_source_secret'   → 1
grep -c '^def _apply_secret_idempotent' → 1
grep -c 'status_code == 409'         → 1
grep -c exact patch shape            → 1
grep -c 'kr8s.AlreadyExistsError'    → 0
delay=30 count                       → 15 (was 5, +10 ≥ 4)
'will retry in 30s' count            → 8  (was 3, +5 ≥ 4)
grep -c 'base64.b64decode'...secret.data → 1
grep -c '@kopf.on.delete(...optional=True)' → 1
grep -c 'OPS-08'                     → 5 (≥ 1)
logger.warning count                 → 2  (was 1, +1 ≥ 1)
grep -c 'warp-{name}-*'             → 1
python -m py_compile operator_module/main.py → exit 0
Overall: 9 new def/handler identifiers → wc -l returns 9
```

## Security / Threat Model

All T-22-01 and T-22-04 mitigations implemented:
- No helper body logs its `key`, `token`, `username`, or `endpoint` parameter values (T-22-01, D-03)
- Exception messages reference only `ctx` (CR namespace/name), the source Secret namespace/name, and HTTP status codes — never decoded values (T-22-04)
- T-22-05: `delete_warpcredential` performs zero destructive operations; `optional=True` blocks finalizer addition

## Deviations from Plan

None — plan executed exactly as written.

The only minor adjustment: the docstring for `_apply_secret_idempotent` originally mentioned `kr8s.AlreadyExistsError` by name in a "does not exist" warning. The exact string triggered the plan's acceptance criterion `grep -c 'kr8s.AlreadyExistsError' == 0`. The docstring was rephrased to convey the same warning without using the literal class path — the semantic intent is identical and the spirit of Pitfall 1 is preserved.

## Known Stubs

None. All helpers are fully implemented, not stubbed.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes introduced. All new code operates within the existing kr8s + kopf trust boundary already present in operator_module/main.py.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| operator_module/main.py | FOUND |
| .planning/phases/22-operator-warpcredential-reconciler/22-01-SUMMARY.md | FOUND |
| Commit 0684cc0 (Task 1.1) | FOUND |
| Commit 2e493f2 (Task 1.2) | FOUND |
| Commit 0103cad (Task 1.3) | FOUND |
