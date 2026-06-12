---
phase: 25-blueprint-credential-selector-sdk
plan: "01"
subsystem: webapp
tags: [credentials, helper, blueprint-sdk, refactor]
dependency_graph:
  requires: []
  provides: [_get_credentials_by_type, blueprint_detail_credentials_context]
  affects: [settings_page, blueprint_detail]
tech_stack:
  added: []
  patterns: [asyncio.to_thread, exception-fallback-empty-dict, ready-filter-at-helper-level]
key_files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/tests/test_credentials_api.py
decisions:
  - "Ready-filter applied at helper level (item.get('ready') guard) — single source of truth for SDK-02 contract; macros stay logic-free"
  - "Pre-existing weka_storage_credentials line-550 list comprehension preserved as no-op per CLAUDE.md §3"
  - "Exception set exactly (ApiException, ConnectionError, TimeoutError) — no broadening, no narrowing per D-02"
metrics:
  duration: "6 minutes"
  completed: "2026-06-12T04:40:38Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 25 Plan 01: Credential Helper Extraction and Blueprint Context Injection Summary

Module-level async helper `_get_credentials_by_type(ns)` extracted from `settings_page` with ready-filter applied at the helper level, injected into `blueprint_detail` template context with namespace resolution via `get_auth_status`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add failing tests for `_get_credentials_by_type` helper | d58b194 | `app-store-gui/tests/test_credentials_api.py` |
| 1 (GREEN) | Extract helper and refactor `settings_page` | 2e0b836 | `app-store-gui/webapp/main.py` |
| 2 (RED) | Add failing tests for `blueprint_detail` injection | 10aee7c | `app-store-gui/tests/test_credentials_api.py` |
| 2 (GREEN) | Inject `credentials_by_type` into `blueprint_detail` | 7a7eba6 | `app-store-gui/webapp/main.py`, `app-store-gui/tests/test_credentials_api.py` |

## Helper Shipped

**Signature:** `async def _get_credentials_by_type(ns: str) -> dict` at line 767 in `app-store-gui/webapp/main.py`

**Exact insertion site:** Immediately after `_build_credential_response_item` (which ends at the former line 783), before the `@app.get("/api/credentials")` route. This co-locates the helper with the dependency it consumes.

**Body:**
- Pre-seeds `credentials_by_type: dict = {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}`
- Defines nested `def _list()` → `client.CustomObjectsApi().list_namespaced_custom_object(..., namespace=ns)`
- `try:` calls `load_kube_config()` then `await asyncio.to_thread(_list)`
- Loops over items, builds each via `_build_credential_response_item(cr)`, appends `item` ONLY when `t in credentials_by_type and item.get("ready")` (the critical conjunction: unknown-type drop AND non-ready drop at the same guard)
- `except (ApiException, ConnectionError, TimeoutError): pass` — exception set locked by D-02
- Returns `credentials_by_type`

## Template Context Key Added to `blueprint_detail`

The `blueprint_detail` route now resolves namespace before the `return templates.TemplateResponse(...)` call:

```python
auth = await asyncio.to_thread(get_auth_status)
detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
ns = detected_ns or "default"
credentials_by_type = await _get_credentials_by_type(ns)
```

Key `"credentials_by_type": credentials_by_type` added to the context dict alongside all existing keys.

## Exception Set Behavior on K8s Failure

When the K8s API is unreachable (`ApiException`), the connection is refused (`ConnectionError`), or the call times out (`TimeoutError`), `_get_credentials_by_type` returns `{"nvidia-ngc": [], "huggingface": [], "weka-storage": []}` without raising. Blueprint pages render normally using Plan 02's macros in "hint mode" (showing "Add one in Settings." links instead of selects).

## Ready-Filter: Single Source of Truth for SDK-02

The ready-filter guard `if t in credentials_by_type and item.get("ready")` is applied inside `_get_credentials_by_type` — not in the macros. This means:
- Plan 02 macros can use simple `{% if credentials_by_type[type] %}` truthiness checks; no `selectattr('ready')` in Jinja2
- Consistent with `GET /api/credentials?type=<t>` endpoint (API-02, Phase 22) which also returns only `ready: true` items
- `_build_credential_response_item` already emits `ready` as a boolean, so `item.get("ready")` is safe

## No-op Fate of Pre-Existing Line-550 List Comprehension

The `settings_page` line `weka_storage_credentials = [c for c in credentials_by_type["weka-storage"] if c.get("ready")]` remains intact. After the helper's ready-filter takes effect, every entry in `credentials_by_type["weka-storage"]` is already ready, so this comprehension is a no-op that returns the same list unchanged. It is preserved per CLAUDE.md §3 ("Don't remove pre-existing dead code unless asked").

## Plan 02 and Plan 03 Dependency Note

Plan 02 (Jinja2 macro SDK, `_credential_macros.html`) and Plan 03 (credential macro tests, `test_credential_macros.py`) both depend on `_get_credentials_by_type` being callable as an async module-level function. This plan delivers that callable. The `credentials_by_type` variable is available in every blueprint template context, ready for macro consumption without any further backend changes.

## Deviations from Plan

None. Plan executed exactly as written. The one minor note: test stubs for `blueprint_detail` tests required using `async def` coroutines (not plain lambdas) for `_get_credentials_by_type` monkeypatching — this is correct implementation behavior (the helper is `async def`) and was captured in the test fix during the GREEN phase of Task 2.

## Deferred Items (Pre-Existing Test Failures, Out of Scope)

8 pre-existing test failures in `test_credentials_api.py` existed before this plan and remain unchanged:
- `test_make_credential_slug_normalizes_and_truncates` — slug truncation length mismatch (test expects 52 chars, code returns different length)
- `test_list_credentials_*` — apparent `type` keyword alias not being passed correctly
- `test_post_credential_*` — `create_credential()` signature mismatch (`type` keyword not found)

These are pre-existing failures unrelated to this plan's changes. Logged per deviation rule scope boundary.

## Known Stubs

None. This plan delivers production-ready helper + route injection logic. No placeholder values.

## Threat Flags

None. The `_get_credentials_by_type` helper exclusively uses `_build_credential_response_item` to build items — an existing function (Phase 23) that never reads raw Secret content. The trust boundary analysis in the plan's `<threat_model>` is satisfied:
- T-25-01 (Information Disclosure): mitigated by `_build_credential_response_item`'s design
- T-25-02 (DoS on K8s unreachable): mitigated by the exception fallback
- T-25-03 (Path traversal): pre-existing, unchanged
- T-25-04 (Log disclosure): no new logging added

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `app-store-gui/webapp/main.py` exists | FOUND |
| `app-store-gui/tests/test_credentials_api.py` exists | FOUND |
| `.planning/phases/25-blueprint-credential-selector-sdk/25-01-SUMMARY.md` exists | FOUND |
| Commit d58b194 (RED Task 1) exists | FOUND |
| Commit 2e0b836 (GREEN Task 1) exists | FOUND |
| Commit 10aee7c (RED Task 2) exists | FOUND |
| Commit 7a7eba6 (GREEN Task 2) exists | FOUND |
