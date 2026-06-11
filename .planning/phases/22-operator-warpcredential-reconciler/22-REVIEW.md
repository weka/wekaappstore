---
phase: 22-operator-warpcredential-reconciler
reviewed: 2026-06-11T06:43:08Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - operator_module/main.py
  - operator_module/tests/test_warp_credential.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-06-11T06:43:08Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the Phase 22 WarpCredential reconciler additions: six pure helper functions (`_b64`, `_now_iso`, `_build_condition`, `_derive_ngc_payloads`, `_derive_hf_payload`, `_derive_weka_payload`), two kr8s I/O helpers (`_read_source_secret`, `_apply_secret_idempotent`), the `delete_warpcredential` handler, and the `reconcile_warpcredential` handler. The test file covering 20 test cases was also reviewed.

The general structure is sound: error dispatch is idiomatic, the single-pass stacked-decorator pattern is correct, and most API-08 log-safety discipline holds. Two blockers were found. The first is a security defect: `_derive_ngc_payloads` embeds the raw API key as a plaintext `password` field in the `dockerconfigjson` JSON payload, directly violating API-08 and unnecessarily doubling key exposure. The second is a correctness defect: if `_apply_secret_idempotent` raises (TemporaryError or PermanentError) at any of its four call sites in `reconcile_warpcredential`, the CR status conditions are left in whatever state they were before (unset on first reconcile, or stale from a previous run) — violating OPS-02/OPS-03's requirement that status be patched before raising.

---

## Critical Issues

### CR-01: Raw NGC API key stored as plaintext `password` in `dockerconfigjson` payload

**File:** `operator_module/main.py:385`
**Issue:** `_derive_ngc_payloads` places the raw, un-obfuscated `key` value directly in the `password` field of the JSON structure that becomes `.dockerconfigjson`. While the whole blob is base64-encoded before storage, anyone who reads the Kubernetes Secret (e.g., via `kubectl get secret -o yaml` or RBAC permitting) and base64-decodes it receives the key in cleartext JSON. The `auth` field already encodes the credential as `base64("$oauthtoken:<key>")`, which is the only field `containerd` and Docker use. The `password` field is redundant and materially increases the exposure surface. This also violates API-08 at the data-at-rest level (the Secret's binary content contains the verbatim key as a JSON string value). The companion test at `test_warp_credential.py:140` asserts this behaviour, locking the defect in as "expected."

**Fix:** Remove the `password` field from the dockerconfigjson structure. The NGC convention only requires `username` and `auth`:

```python
docker_config = {
    'auths': {
        'nvcr.io': {
            'username': '$oauthtoken',
            'auth': docker_auth_b64,
        }
    }
}
```

Update `test_ngc_docker_auth_is_oauthtoken_b64` to remove the `password` assertion and add a negative assertion that `password` is not present:

```python
assert 'password' not in docker_json['auths']['nvcr.io']
assert docker_json['auths']['nvcr.io']['auth'] == base64.b64encode(
    b'$oauthtoken:my-secret-key'
).decode('ascii')
```

---

### CR-02: `_apply_secret_idempotent` failures leave CR status conditions unpatched (OPS-02/OPS-03 violation)

**File:** `operator_module/main.py:1597-1598, 1614, 1658`
**Issue:** The four call sites of `_apply_secret_idempotent` inside `reconcile_warpcredential` are bare (no surrounding try/except). If `_apply_secret_idempotent` raises `kopf.TemporaryError` (5xx, APITimeoutError) or `kopf.PermanentError` (non-409 4xx), the exception propagates directly to kopf with `patch.status['conditions']` either unset (first reconcile of a new CR) or holding the stale value from the prior reconcile cycle. The status conditions are only written at the success path (line 1667) or in the early-exit error paths (lines 1529, 1539, 1548, 1553). There is no failure status write for the Secret write phase.

This means an operator watching `kubectl get warpcredential <name> -o yaml` will see a blank or stale `conditions` array while the controller is in a retry backoff loop caused by a 5xx API server error. OPS-02 and OPS-03 explicitly require status conditions patched before raising.

**Fix:** Wrap all `_apply_secret_idempotent` call sites in a helper or add explicit try/except blocks that patch status before re-raising:

```python
# Pattern — apply around each _apply_secret_idempotent call site:
try:
    _apply_secret_idempotent(apikey_secret, ctx=f'{ctx}: apikey')
    _apply_secret_idempotent(docker_secret, ctx=f'{ctx}: docker')
except (kopf.TemporaryError, kopf.PermanentError):
    patch.status['conditions'] = [_build_condition(
        'KeyReady', 'False', 'SecretWriteError',
        'Failed to write derived Secret(s) to the API server (see operator logs)')]
    raise
```

Apply the same pattern around the `huggingface` call site (line 1614) and the `weka-storage` call site (line 1658).

---

## Warnings

### WR-01: `status.wekaEndpoint` written from `spec.endpoint` rather than the actual endpoint used in the derived Secret

**File:** `operator_module/main.py:1671`
**Issue:** The actual endpoint placed in the derived `WEKA_API_ENDPOINT` Secret key is `spec.get('endpoint') or endpoint_from_src` (line 1648) — it falls back to the source-Secret value when `spec.endpoint` is absent. However, the status write at line 1671 is `spec.get('endpoint', '')`, which returns `''` when `spec.endpoint` is absent. This means `status.wekaEndpoint` is empty string even though the derived Secret contains a non-empty endpoint from the source Secret. The status field misrepresents the live cluster state.

**Fix:** Write the resolved `endpoint` variable (already computed at line 1648) to status, not `spec.get('endpoint', '')`:

```python
# Replace line 1671:
patch.status['wekaEndpoint'] = endpoint
```

This requires ensuring the `endpoint` variable is in scope at that point (it is — it's defined at line 1648 inside the `else: # weka-storage` block, but the status write is outside that block). Move the `wekaEndpoint` write inside the `else` block or capture `endpoint` as a local before the branch:

```python
# Inside the weka-storage else block, before _apply_secret_idempotent:
patch.status['wekaEndpoint'] = endpoint  # resolved value actually in the Secret
```

---

### WR-02: `_apply_secret_idempotent` silently classifies `kr8s.ServerError` with no `.response` as `PermanentError`, preventing retry of potentially transient errors

**File:** `operator_module/main.py:500-512`
**Issue:** At lines 501 and 507, `status` is `None` when `e.response` is absent. The `== 409` branch (line 502) and `>= 500` branch (line 507) both skip. The code falls through to `raise kopf.PermanentError` (line 512). A `kr8s.ServerError` without a `.response` attribute can occur when the kr8s library wraps a socket-level or connection-level exception. Marking it permanent stops all future reconciliation of that CR, even though the underlying cause is transient. The same pattern exists in `_read_source_secret` (line 459-466), both are structural copies of `load_values_from_reference`.

**Fix:** Add an explicit guard: if `status is None`, treat as a transient/unknown error and raise `TemporaryError`:

```python
except kr8s.ServerError as e:
    status = e.response.status_code if getattr(e, 'response', None) is not None else None
    if status is None:
        raise kopf.TemporaryError(
            f'{ctx}: unclassified API error writing Secret (no response; will retry in 30s)',
            delay=30,
        ) from e
    if status == 409:
        secret_obj.patch({'data': secret_obj.raw['data'], 'type': secret_obj.raw['type']})
        return
    if status >= 500:
        raise kopf.TemporaryError(
            f'{ctx}: API server error {status} writing Secret (will retry in 30s)',
            delay=30,
        ) from e
    raise kopf.PermanentError(f'{ctx}: API error writing Secret: {e}') from e
```

Apply the same fix to `_read_source_secret`.

---

### WR-03: API-08 caplog test does not cover log records from `_read_source_secret` or `_apply_secret_idempotent` — only the single `logger.info` at line 1674 is captured

**File:** `operator_module/tests/test_warp_credential.py:550-587`
**Issue:** `test_no_key_in_logs_anywhere` uses `caplog.at_level(logging.DEBUG)` which captures all records that propagate through the Python logging hierarchy. The reconciler's `logger.info` call at line 1674 uses the kopf-injected `logger` argument (a `logging.getLogger('test')` in tests), which does propagate. However, `_read_source_secret` and `_apply_secret_idempotent` do not accept a logger — they have no log calls, so there is no leak path there. The test therefore does not validate the code paths it claims concern — its comment "across ALL captured records at ANY level" is accurate but the test can only see the one `logger.info` line in the success path. If a future change adds a debug log in a helper, this test will enforce the constraint correctly. The test is not wrong, but the assertion `assert len(caplog.records) >= 1` verifies only that one INFO record was captured, not that the test exercised all log paths. The "API-08 enforcement" claim in the comment is narrower than stated.

**Fix:** Add a `caplog.at_level(logging.DEBUG, logger='main')` scope to ensure the module-level `logging` calls (e.g., inside `HelmOperator` which this module also contains) are captured. Alternatively, add a targeted assertion that the test captured records from more than one source or explicitly inject a `logging.getLogger('main')` logger:

```python
# More precise: use caplog.at_level scoped to the module under test
with caplog.at_level(logging.DEBUG, logger='main'):
    ...
# Then assert both that records were captured AND the key is absent
assert any(r.name == 'main' or r.name == 'test' for r in caplog.records)
```

---

## Info

### IN-01: `test_ngc_docker_auth_is_oauthtoken_b64` asserts `password == 'my-secret-key'` — test locks in the CR-01 security defect

**File:** `operator_module/tests/test_warp_credential.py:140`
**Issue:** Line 140 explicitly checks `docker_json['auths']['nvcr.io']['password'] == 'my-secret-key'`. This assertion will need to be removed (and a negative assertion added) when CR-01 is fixed. Leaving it in place will cause the test to fail after the fix, which is the correct outcome — but the test currently validates a security defect as correct.

**Fix:** After resolving CR-01, change line 140 to:
```python
assert 'password' not in docker_json['auths']['nvcr.io']
```

---

### IN-02: `_now_iso()` uses deprecated `datetime.utcnow()` (Python 3.12+)

**File:** `operator_module/main.py:341`
**Issue:** `datetime.utcnow()` is deprecated since Python 3.12. The docstring acknowledges this and defers migration to a project-wide refactor, citing the existing pattern at main.py:727, 939, 972, 1109. This is consistent with the stated project convention and is low risk for the current runtime target, but should be tracked.

**Fix (deferred):** When the project-wide migration occurs, replace with:
```python
from datetime import timezone
return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
```

---

_Reviewed: 2026-06-11T06:43:08Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
