# Phase 22: Operator WarpCredential Reconciler - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 2 (1 modified, 1 created)
**Analogs found:** 2 / 2 (both exact in-repo matches — no external patterns needed)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `operator_module/main.py` (MODIFY: + ~150 LOC) | kopf handler module + pure helpers | event-driven (CRD reconcile) + request-response (kr8s API) | self-analog: existing handlers at `operator_module/main.py:951-1246` + helpers at `:291-308` + error dispatch at `:439-468` | exact |
| `operator_module/tests/test_warp_credential.py` (CREATE) | unit test module (helpers + handler-level) | n/a | `operator_module/tests/test_appstack.py` (kr8s mocking + kopf error assertion) + `operator_module/tests/test_render.py` (pure-helper testing) | exact |

---

## Pattern Assignments

### `operator_module/main.py` (handler module — modifications)

**Analog file:** `operator_module/main.py` itself (self-analog; mirror the existing internal patterns).

Place new helpers in the "pure helper" zone near `_render_or_raise` (after line 308). Place new handlers after `delete_warrpappstore_function` (after line 1246). Do NOT introduce new sub-packages — CLAUDE.md locks single-file design.

#### Excerpt A — Pure helper signature pattern to mirror (`_render_or_raise` at lines 291-308)

```python
def _render_or_raise(
    text: str,
    variables: Optional[Dict[str, str]],
    *,
    source_desc: str,
) -> str:
    """Render text with variables; convert KeyError/ValueError to kopf.PermanentError.

    Wraps Phase 16 render() so each substitution call site can pass a
    caller-specific source_desc (component name, valuesFiles index, kind,
    namespace/name, key) without duplicating try/except boilerplate.

    Per CONTEXT.md D-15 (Phase 18, locked).
    """
    try:
        return render(text, variables)
    except (KeyError, ValueError) as e:
        raise kopf.PermanentError(f"{source_desc}: {e}") from e
```

**Copy:** private `_underscore_prefix` naming, keyword-only `ctx`/`source_desc` argument, decision-citing docstring (e.g., "Per CONTEXT.md D-11, D-12"), `raise kopf.PermanentError(...) from e` (preserve `__cause__`).

**Add:** new helpers `_b64(s)`, `_now_iso()`, `_build_condition(type_, status, reason, message)`, `_derive_ngc_payloads(key)`, `_derive_hf_payload(key)`, `_derive_weka_payload(username, token, endpoint)`, `_apply_secret_idempotent(secret_obj, *, ctx)`, `_read_source_secret(name, namespace, *, ctx)` — all near line 308.

---

#### Excerpt B — kr8s Secret-read + error dispatch matrix (lines 439-468)

This block is **canonical for Phase 22**. The reconciler's source-Secret read and the create-or-patch `_apply_secret_idempotent` helper must mirror it line-for-line.

```python
    try:
        if kind == "ConfigMap":
            cm = kr8s.objects.ConfigMap.get(name=name, namespace=namespace)
            values_yaml = cm.data.get(key, "")
        elif kind == "Secret":
            secret = kr8s.objects.Secret.get(name=name, namespace=namespace)
            import base64
            values_yaml = base64.b64decode(secret.data.get(key, "")).decode('utf-8')
        else:
            raise kopf.PermanentError(f"{ctx}: unsupported valuesFiles kind: {kind}")
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

**Copy verbatim:** the four-branch except ladder (`NotFoundError` → Temp, `APITimeoutError` → Temp, `ServerError(>=500)` → Temp, `ServerError(4xx)` → Permanent); the `delay=30` literal; the message template `"<ctx>: ... (will retry in 30s)"`; the `from e` chaining; the `e.response.status_code if getattr(e, "response", None) is not None else None` defensive accessor.

**Adapt for write path:** add a 5th branch — `ServerError(409)` → call `secret_obj.patch({'data': ..., 'type': ...})` and return (D-02). The 409 branch goes BEFORE the `>=500` check.

**Note on quotes (Pitfall):** Lines 449-468 use **double quotes** for f-strings. The legacy status-patch block at lines 720-728 uses **single quotes** for dict keys/values. There is no project-wide convention — match the local context. CLAUDE.md says "operator uses single quotes for shell args" but does NOT mandate Python literal style. Phase 22 recommendation: use the surrounding-block's style for each new function. The status-condition builder (mirroring lines 720-728) should use single quotes; the error-dispatch block (mirroring lines 449-468) should use double quotes for f-strings.

---

#### Excerpt C — kopf handler signature + status patch (lines 720-728 and 951-975)

```python
# Line 720-728 (handle_appstack_deployment status write):
if 'patch' in kwargs:
    kwargs['patch'].status['appStackPhase'] = 'Installing'
    kwargs['patch'].status['conditions'] = [{
        'type': 'Ready',
        'status': 'False',
        'reason': 'DeploymentStarted',
        'message': f'Installing {len(ordered_components)} components',
        'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
    }]

# Line 951-975 (create handler — decorator + signature + PermanentError):
@kopf.on.create('warp.io', 'v1alpha1', 'wekaappstores')
def create_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    logging.info(f"*** WarrpAppStore Created: {name}")
    # ...
    else:
        error_msg = "Either appStack, helmChart, or image+binary must be specified"
        logging.error(error_msg)
        patch.status['conditions'] = [{
            'type': 'Ready',
            'status': 'False',
            'reason': 'InvalidSpec',
            'message': error_msg,
            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z'
        }]
        raise kopf.PermanentError(error_msg)

# Line 1159 (update decorator — field='spec' is REQUIRED to prevent status-write reentry):
@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')
def update_warrpappstore_function(body, spec, name, namespace, status, patch, **kwargs):
    ...

# Line 1175 (delete decorator):
@kopf.on.delete('warp.io', 'v1alpha1', 'wekaappstores')
def delete_warrpappstore_function(spec, name, namespace, **kwargs):
    ...
```

**Copy:**
- Decorator positional-arg order: `('warp.io', 'v1alpha1', '<plural>')` with single quotes.
- Handler kwarg list `(body, spec, name, namespace, status, patch, **kwargs)` — but Phase 22 reconciler can drop `status` if not consumed and accept `logger` (per `**kwargs` convention; the test examples in RESEARCH.md Section 6 use `logger` explicitly).
- `patch.status['conditions'] = [{...}]` shape (write BEFORE raising the error — see line 967-973).
- `datetime.utcnow().isoformat() + 'Z'` timestamp format (NOT `datetime.now(timezone.utc)` — match project, ignore the py3.12 deprecation per RESEARCH.md Pitfall 5).
- `from e` exception chaining is NOT used at line 974 (existing pattern raises bare PermanentError after logging). New code SHOULD use `from e` when re-raising kr8s errors (matches the more-canonical lines 449-468).

**Add (Phase 22-new):**
- Stacked decorators on a single function:
  ```python
  @kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')
  @kopf.on.update('warp.io', 'v1alpha1', 'warpcredentials', field='spec')
  @kopf.on.resume('warp.io', 'v1alpha1', 'warpcredentials')
  def reconcile_warpcredential(body, spec, name, namespace, patch, logger, **kwargs):
  ```
  `field='spec'` on update is REQUIRED (Pitfall 3 — prevents the operator's own status writes from re-triggering the handler).
- `@kopf.on.delete(..., optional=True)` — the `optional=True` is NEW for Phase 22 (existing `delete_warrpappstore_function` does NOT use it). Per RESEARCH.md Pattern 3, `optional=True` prevents kopf from adding a finalizer; the warning-only delete handler is best-effort by design (OPS-08).

---

#### Excerpt D — `logger` kwarg vs module-level `logging` (existing pattern at line 952-953)

Existing handlers use `logging.info(...)` at the module level (line 953: `logging.info(f"*** WarrpAppStore Created: {name}")`). Per-handler `logger` kwarg is NOT used by existing handlers but IS supported by kopf 1.38.

**Recommendation for Phase 22:** Accept `logger` in the handler signature (kopf passes a per-handler logger that auto-tags records with the resource identity), and use `logger.info(...)`, `logger.warning(...)`. This makes caplog assertions (Section 7 of RESEARCH.md) cleaner. **However, do NOT change existing handlers** — only the new WarpCredential handlers use `logger`.

**Conventions (locked):**
- **No `print()`** anywhere — CLAUDE.md project rule + API-08 enforcement.
- **No raw key in log records** at any level (D-03). Tag log records with metadata only: name, namespace, type, derived-secret-name. Never f-string the decoded key.
- **No `extra={'body': body}`** — kopf's `body` is the full CR which itself does not carry the key, but defensive practice is to not pass kwargs through to logger.

---

### `operator_module/tests/test_warp_credential.py` (test module — NEW)

**Analog file:** `operator_module/tests/test_appstack.py` (kr8s mocking + kopf error dispatch tests) + `operator_module/tests/test_render.py` (pure-helper unit tests).

#### Excerpt E — Test file header and sys.path setup (test_appstack.py:1-26 and conftest.py)

```python
"""Unit tests for operator_module.main WarpCredential reconciler (Phase 22).

Covers OPS-01..OPS-09 + API-08:
  - _derive_ngc_payloads / _derive_hf_payload / _derive_weka_payload (derivation correctness)
  - _apply_secret_idempotent (create-then-409->patch path; 5xx->TemporaryError; 4xx->PermanentError)
  - reconcile_warpcredential handler paths (missing secretRef, empty key, unknown type, success)
  - delete_warpcredential warning-only behavior
  - API-08: caplog-asserted no-key-leak across any log level
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))
```

**Copy verbatim:** the future-import (`from __future__ import annotations`), sys.path defense-in-depth block, the `from unittest.mock import MagicMock, patch` import line.

**Add:** module docstring tied to OPS-01..OPS-09 + API-08; imports for `base64`, `json`, `logging` needed by Phase 22 tests.

---

#### Excerpt F — kr8s mocking helpers to copy verbatim (test_appstack.py:62-80)

```python
def _make_kr8s_secret(data_dict):
    """Return a MagicMock whose .data maps each key to base64-encoded value (mimics kr8s Secret)."""
    secret = MagicMock()
    secret.data = {
        k: base64.b64encode(v.encode("utf-8") if isinstance(v, str) else v).decode("utf-8")
        for k, v in data_dict.items()
    }
    return secret


def _make_kr8s_server_error(status_code, message="server error"):
    """Construct a kr8s.ServerError with .response.status_code set."""
    import kr8s

    err = kr8s.ServerError(message)
    response = MagicMock()
    response.status_code = status_code
    err.response = response
    return err
```

**Copy verbatim** into `test_warp_credential.py`. (Do NOT import-share from `test_appstack.py` — operator tests prefer file-local copies of small helpers; see how `test_appstack.py` and `test_render.py` each redeclare sys.path setup.)

**Add (Phase 22-specific helpers):**
- `_make_patch_obj()` — returns a MagicMock with `.status = {}` (kopf patch stand-in).
- `_make_secret_class_mock()` — factory for patching the `kr8s.objects.Secret` *constructor* so derived-Secret instantiation can be intercepted; returns a MagicMock supporting `.create()`, `.patch()`, and `.raw` (the dict passed to the constructor).

---

#### Excerpt G — Error-dispatch assertion pattern (test_appstack.py:353-380)

```python
def test_missing_configmap_raises_temporary_error():
    """OP-11: kr8s.NotFoundError -> kopf.TemporaryError(delay=30)."""
    import kopf
    import kr8s
    from main import load_values_from_reference

    with patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", side_effect=kr8s.NotFoundError("not found")), \
         pytest.raises(kopf.TemporaryError) as exc_info:
        load_values_from_reference(
            kind="ConfigMap", name="missing-cm", key="values.yaml",
            namespace="staging", comp_name="vector-db", ref_index=0,
        )

    assert getattr(exc_info.value, "delay", None) == 30
    assert "missing-cm" in str(exc_info.value)
    assert "vector-db" in str(exc_info.value)
```

**Copy:**
- Import pattern: `import kopf` and `import kr8s` INSIDE the test function (mirrors line 361-362; avoids tight coupling at module-import time).
- `with patch("main.kr8s.objects.<Type>.<method>", side_effect=...), pytest.raises(kopf.<Error>) as exc_info:` compound `with` block.
- Assertion shape: `assert getattr(exc_info.value, "delay", None) == 30` and `assert <metadata> in str(exc_info.value)`.

**Adapt for Phase 22:**
- Patch target is `main.kr8s.objects.Secret.get` (existing line 444 surface).
- For idempotency tests, patch `main.kr8s.objects.Secret` (the constructor) to inject a MagicMock with controllable `.create()` / `.patch()` side effects.

---

#### Excerpt H — Pure-helper test pattern (test_render.py:38-42 and 95-101)

```python
def test_render_returns_unchanged_when_no_braces() -> None:
    """render() short-circuits on bare-shell-style strings via pre-scan guard."""
    from main import render

    assert render("$CRDS && $CRD", {}) == "$CRDS && $CRD"


def test_render_undefined_variable_raises_value_error() -> None:
    """render() with ${UNDEF} raises ValueError naming the variable (D-04, D-05)."""
    from main import render

    with pytest.raises(ValueError) as exc_info:
        render("value: ${UNDEF}", {"x": "y"})
    assert "UNDEF" in str(exc_info.value)
```

**Copy:**
- `from main import <name>` import-inside-test pattern (defers module import; lets sys.path setup land first).
- Decision-citing one-liner docstrings (`(D-04, D-05)`, `(API-08)`).
- `with pytest.raises(<Error>) as exc_info: ... assert "<keyword>" in str(exc_info.value)` shape.

**Add (Phase 22-specific derivation tests):** Verbatim from RESEARCH.md Section 6 — test_ngc_apikey_data_is_b64_encoded, test_ngc_docker_auth_is_oauthtoken_b64, test_hf_payload_has_only_hf_api_key, test_weka_payload_three_keys.

---

#### Excerpt I — caplog assertion pattern (no existing project precedent)

No existing operator test uses `caplog`. This test will be the first; pattern is canonical pytest:

```python
def test_no_key_in_logs_anywhere(caplog):
    """API-08: raw key value MUST NOT appear in any log record at any level."""
    from main import reconcile_warpcredential
    test_key = 'super-secret-test-key-value-do-not-leak-42'
    src_secret = _make_kr8s_secret({'NGC_API_KEY': test_key})
    patch_obj = _make_patch_obj()

    with caplog.at_level(logging.DEBUG):
        with patch('main.kr8s.objects.Secret.get', return_value=src_secret), \
             patch('main.kr8s.objects.Secret', side_effect=_make_secret_class_mock()):
            reconcile_warpcredential(
                body={}, spec={'type': 'nvidia-ngc', 'displayName': 'NGC Test',
                               'secretRef': {'name': 'src', 'key': 'NGC_API_KEY'}},
                name='ngc-test', namespace='weka-app-store',
                patch=patch_obj, logger=logging.getLogger('test'),
            )

    for record in caplog.records:
        msg = record.getMessage()
        assert test_key not in msg, f'Key leaked in log: {record.levelname} {msg}'
        assert test_key not in str(record.args or ''), f'Key leaked in args: {record.args}'
```

**Source:** This is a new pattern (no project precedent). Verbatim from RESEARCH.md Section 7. Verify `pytest >= 8.0` is available (it is — installed in the operator test env).

**Convention to add to test file:** Always use `caplog.at_level(logging.DEBUG)` — capture EVERYTHING, including DEBUG records, because API-08 says "no key value at any level."

---

## Shared Patterns

### Pattern S-1: ISO-8601 UTC timestamp
**Source:** `operator_module/main.py:727, 939, 972, 1109` (all four sites use the same expression)
**Apply to:** every `lastTransitionTime` and `lastSyncTime` write
```python
datetime.utcnow().isoformat() + 'Z'
```
Wrap once as `_now_iso()` (RESEARCH.md Section 1, Example A). Mirror exactly — do NOT switch to `datetime.now(timezone.utc)` despite py3.12 deprecation (Pitfall 5; project-wide convention takes precedence).

### Pattern S-2: kr8s exception → kopf typed error dispatch matrix
**Source:** `operator_module/main.py:449-468`
**Apply to:** `_read_source_secret` (read path) and `_apply_secret_idempotent` (write path)
```
kr8s.NotFoundError       -> kopf.TemporaryError(delay=30)
kr8s.APITimeoutError     -> kopf.TemporaryError(delay=30)
kr8s.ServerError(>=500)  -> kopf.TemporaryError(delay=30)
kr8s.ServerError(4xx)    -> kopf.PermanentError
kr8s.ServerError(409)    -> patch() and return     [WRITE PATH ONLY]
```
Always `raise ... from e` for `__cause__` preservation. Always include the resource locator (`ctx`) in the message. NEVER include the decoded key value.

### Pattern S-3: Decision-citing docstring
**Source:** `_render_or_raise` docstring lines 297-303 ("Per CONTEXT.md D-15 (Phase 18, locked).")
**Apply to:** every new helper and handler — cite the specific decision IDs (D-04, D-07, etc.) and the requirement IDs (OPS-01..OPS-09, API-08) in the docstring so future readers can trace intent without re-reading CONTEXT.md/RESEARCH.md.

### Pattern S-4: status-patch-before-raise
**Source:** `operator_module/main.py:967-974` (existing wekaappstores create handler) — `patch.status['conditions'] = [...]` IS executed BEFORE `raise kopf.PermanentError(...)`. kopf will submit the status patch on the way out even if the handler raises.
**Apply to:** every failure path in `reconcile_warpcredential`. D-15 requires status condition set to `False` with the appropriate reason (`KeyMissing`, `EmptyKey`, `UnknownType`, `DerivationFailed`, `InvalidSpec`) BEFORE the `raise`.

### Pattern S-5: No-ownerReferences on derived Secrets
**Source:** RESEARCH.md Pattern 4 + Anti-Patterns + T-22-05.
**Apply to:** every `kr8s.objects.Secret({...})` instantiation for derived secrets (`warp-<name>-apikey`, `warp-<name>-docker`, `warp-<name>-token`).
The `metadata` dict MUST NOT contain `ownerReferences`. Adding them would enable K8s garbage collection, violating OPS-08 (derived secrets must survive CR deletion).

### Pattern S-6: Mock import-pattern inside tests
**Source:** `test_appstack.py:361-363` — `import kopf` / `import kr8s` / `from main import <name>` are all INSIDE the test function body, not at module top.
**Apply to:** every new test in `test_warp_credential.py`. Reason: it lets sys.path setup run first (line 25-26 of test_appstack.py) before importing `main`.

---

## Conventions (Locked — Executor Must Respect)

| Convention | Source | Phase 22 Application |
|------------|--------|----------------------|
| Single quotes for kopf decorator positional args + dict literals in status writes | `main.py:721-728, 951, 1159, 1175` | `@kopf.on.create('warp.io', 'v1alpha1', 'warpcredentials')` — single quotes. `{'type': 'KeyReady', 'status': 'True', ...}` — single quotes. |
| Double quotes for f-strings in error-dispatch block | `main.py:450-467` | The four-branch except ladder uses double-quoted f-strings (`f"{ctx}: ... not found (will retry in 30s)"`). Mirror exactly. |
| No `print()` anywhere | CLAUDE.md + API-08 | Use `logger.info/warning/error` (kopf-passed per-handler) or `logging.getLogger(__name__)`. Never `print(...)`. |
| No raw key value in logs at ANY level | D-03 / API-08 / T-22-01 | Tag log records with metadata only (name, namespace, type, derived-secret-name). NEVER f-string the decoded `key` variable. Exception messages reference key NAME, not VALUE (T-22-04). |
| No `extra={'body': body}` patterns | RESEARCH.md Anti-Patterns | Don't pass `body`/`spec` kwargs through to logger; some kopf paths include the full CR in log records via `extra`. |
| `field='spec'` on `@kopf.on.update` | `main.py:1159` | REQUIRED on the update decorator — prevents the operator's own status writes from re-triggering the reconciler (Pitfall 3). Document the rationale in a code comment so a future cleanup doesn't remove it. |
| `optional=True` on `@kopf.on.delete` | NEW for Phase 22 (vs `main.py:1175` which omits it) | REQUIRED — prevents kopf from adding a finalizer. OPS-08 makes the delete handler best-effort logging only. |
| `datetime.utcnow().isoformat() + 'Z'` | `main.py:727, 939, 972, 1109` | Mirror exactly. Do NOT migrate to `datetime.now(timezone.utc)` despite py3.12 deprecation — project-wide convention. |
| `raise ... from e` exception chaining | `main.py:449-468` (canonical), `main.py:286, 308` | Always when re-raising kr8s errors as kopf errors. Existing line 974 lacks `from e` — new code should use it. |
| No `kubernetes` client (`k8s_client.CoreV1Api`) in reconcile paths | Phase 18 convention | All Secret read/write goes through `kr8s.objects.Secret.{get,create,patch}`. The `kubernetes` client at `main.py:15-22` is reserved for CRD-discovery helpers only. |
| `try create / except ServerError(409) → patch` (NOT delete-then-create, NOT pre-flight exists()) | D-02 / RESEARCH.md Pattern 1 / Pitfall 1 | The 409 branch is INSIDE the `except kr8s.ServerError` block, BEFORE the `>=500` check. Do NOT import `kr8s.AlreadyExistsError` — it does not exist in kr8s 0.20.10. |
| Test file naming: `test_<resource>.py` | `test_appstack.py`, `test_render.py` | `test_warp_credential.py` (snake_case, matches existing pattern). |
| `caplog.at_level(logging.DEBUG)` for log-safety tests | NEW for Phase 22 | Capture EVERYTHING — API-08 says "no key value at any level," so DEBUG records must be checked too. |

---

## No Analog Found

None — every Phase 22 file has a direct in-repo analog with strong line-cited excerpts. The only NEW patterns (no project precedent) are:

| New Pattern | Why No Precedent | Source |
|-------------|------------------|--------|
| Stacked `@kopf.on.create / on.update / on.resume` decorators on one function | Existing handlers split create/update/delete across separate functions (`main.py:951, 1159, 1175`). Phase 22 D-04 explicitly locks the stacked-decorator pattern. | kopf 1.38 docs ("It is a common pattern…") + RESEARCH.md Pattern 2 |
| `@kopf.on.delete(..., optional=True)` | Existing delete handler at `main.py:1175` does NOT use `optional=True`. Phase 22 OPS-08 requires the no-finalizer best-effort variant. | kopf 1.38 signature verified by `inspect.signature` + nolar/kopf#701 + RESEARCH.md Pattern 3 |
| `caplog`-based assertion of API-08 (no key leak) | No existing operator test uses caplog. | Canonical pytest pattern, verbatim from RESEARCH.md Section 7 |
| `_apply_secret_idempotent` create-or-patch helper | Existing operator only READS Secrets via kr8s (`main.py:444`); never WRITES. Phase 22 is the first write path. | RESEARCH.md Pattern 1 + Section 1 (kr8s 0.20.10 write API surface) |

All four are documented with full code excerpts in RESEARCH.md — the planner can cite RESEARCH.md sections directly in plan `read_first` blocks.

---

## Metadata

**Analog search scope:**
- `operator_module/main.py` (all 1246+ lines — single-file module by design)
- `operator_module/tests/conftest.py`, `test_appstack.py`, `test_render.py`
- `weka-app-store-operator-chart/templates/crd.yaml` (CRD schema reference)

**Files scanned:** 6
**Pattern extraction date:** 2026-06-11

## PATTERN MAPPING COMPLETE
