---
plan: 25-03
phase: 25-blueprint-credential-selector-sdk
status: complete
self_check: PASSED
key-files:
  created:
    - app-store-gui/tests/test_credential_macros.py
---

## Summary

Created `app-store-gui/tests/test_credential_macros.py` — 8-test regression suite covering the Plan 01 backend wiring for Phase 25's credential SDK.

## Tests Shipped

**Test count:** 8 (all pass)

**Invocation:** `PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest app-store-gui/tests/test_credential_macros.py -v`

| Test | What it locks |
|------|--------------|
| `test_get_credentials_by_type_groups_by_type` | Groups ready CRs by type; pre-seeds all three known type keys even when list is empty (SDK-04 contract) |
| `test_get_credentials_by_type_returns_empty_lists_on_api_exception` | Empty dict-of-lists fallback on `ApiException` without re-raising (SDK-05) |
| `test_get_credentials_by_type_returns_empty_lists_on_connection_error` | Empty dict-of-lists fallback on `ConnectionError` without re-raising (SDK-05) |
| `test_get_credentials_by_type_returns_empty_lists_on_timeout_error` | Empty dict-of-lists fallback on `TimeoutError` — completes the D-02 exception triple (SDK-05) |
| `test_get_credentials_by_type_drops_unknown_type` | CRs with `spec.type = "unknown-type"` are silently dropped (never appear in any key) |
| `test_get_credentials_by_type_filters_non_ready_credentials` | Non-ready CRs filtered at the helper level (W-4 fix — SDK-02's "ready" contract locked at single source of truth) |
| `test_blueprint_detail_injects_credentials_by_type_into_context` | `blueprint_detail` injects `credentials_by_type` into `TemplateResponse` context (asserted via `response.context`, W-3); sentinel credential name roundtrips correctly |
| `test_blueprint_detail_falls_back_to_default_namespace` | When `get_auth_status` returns `{}` (no `details` key), helper is called with `"default"` |

## Patterns Reused

- Module preamble (`os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")` before `import webapp.main`) copied verbatim from `test_credentials_api.py:1-12`
- CR factory functions imported via relative import: `from .test_credentials_api import make_warpcred_cr_nvidia_ready, make_warpcred_cr_nvidia_not_ready, make_warpcred_cr_weka_ready`
- `_patch_list_credentials(monkeypatch, items, raises)` stub helper — adapted from `test_credentials_api.py:118-126`
- Async-handler invocation: `asyncio.run(main._get_credentials_by_type(...))` — same pattern as existing tests, no pytest-asyncio dependency

## What Is NOT Tested

No Jinja2 macro-rendering tests. The macro file (`_credential_macros.html`) is a static template with only `{% if %}` branching — its shape is validated by the grep acceptance criteria in Plan 02 (class strings, script presence, link targets). Adding macro-rendering tests would require a Jinja2 `Environment` setup duplicating the FastAPI runtime, which is out of scope for Phase 25.

No `/deploy-stream` or form-submit handler tests — D-10 excludes backend credential wiring from Phase 25.

## Full Suite Regression

`PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest app-store-gui/tests/ -v` — all 8 new tests pass; 25 pre-existing tests continue to pass; 8 pre-existing failures (unrelated to Phase 25, pre-dating this milestone) remain deferred.

## Self-Check

- [x] `test_credential_macros.py` exists with 8 test functions
- [x] All 8 tests pass via project's standard pytest invocation
- [x] `response.context` used (not `response.context_data`) — W-3 fix
- [x] Non-ready filter test covers W-4 behavior
- [x] All three exception types (ApiException, ConnectionError, TimeoutError) tested — D-02 triple
- [x] No TestClient, no conftest.py modification, no new pytest plugins
- [x] Factory functions imported via relative import (no duplication)
