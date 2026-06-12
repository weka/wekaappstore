---
phase: 25-blueprint-credential-selector-sdk
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - app-store-gui/tests/test_credential_macros.py
  - app-store-gui/tests/test_credentials_api.py
  - app-store-gui/webapp/main.py
  - app-store-gui/webapp/templates/_credential_macros.html
  - app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html
findings:
  critical: 3
  warning: 4
  info: 2
  total: 9
status: issues_found
---

# Phase 25: Code Review Report

**Reviewed:** 2026-06-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

This phase adds the `_get_credentials_by_type` helper, injects credentials into `blueprint_detail`, and introduces Jinja2 credential selector macros backed by two new test files. The core backend logic is sound — the ready-filter contract, error fallbacks, and secret-field exclusion all look correct. Three blockers were found: a slug truncation length mismatch that causes tests to fail at runtime, a deploy button that always silently fails for the neuralmesh-aidp blueprint because the name is not in the deploy-stream app map, and a `load_kube_config()` call that executes before input validation in `delete_credential`, contradicting the documented "no I/O before validation" intent. Four warnings address a broken `warpSyncEndpoint` JavaScript function when the macro is instantiated more than once per page, a test fragility in `test_credential_macros.py` caused by a bare `SimpleNamespace()` request object, a blocking synchronous `get_cluster_status()` call on the async `blueprint_detail` route, and substantial test duplication across the two new test files.

## Critical Issues

### CR-01: Slug truncation length mismatch — test always fails

**File:** `app-store-gui/tests/test_credentials_api.py:91-92`

**Issue:** `test_make_credential_slug_normalizes_and_truncates` asserts `len(long_slug) == 52` and `long_slug.startswith("a" * 52)`. The implementation at `webapp/main.py:694` truncates to 48 characters (`slug[:48]`). The test will always fail because `len("a"*48) == 48`, not 52. The docstring at `main.py:686-688` explicitly states "truncate to 48 characters" with a clear rationale. Either the test was written against an earlier 52-character limit and was not updated when the constant changed, or the implementation was changed without updating the test.

**Fix:** Update the test to match the implementation (48 characters):
```python
def test_make_credential_slug_normalizes_and_truncates():
    assert main._make_credential_slug("My NGC Key #1") == "my-ngc-key-1"
    long_slug = main._make_credential_slug("a" * 60)
    assert long_slug.startswith("a" * 48)
    assert len(long_slug) == 48
    with pytest.raises(ValueError):
        main._make_credential_slug("---")
```
Alternatively, if 52 was the intended limit, restore `slug[:52]` in `main.py:694` and update the docstring accordingly.

---

### CR-02: neuralmesh-aidp "Deploy Blueprint" button always silently fails

**File:** `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html:357`

**Issue:** The deploy form submits to `/deploy-stream/{{ name }}` which renders as `/deploy-stream/neuralmesh-aidp`. The `deploy_stream` handler's `app_map` at `main.py:2276-2283` does not contain `"neuralmesh-aidp"` (the entry is commented out at `main.py:1277`). When `yaml_path` resolves to `None`, the SSE stream immediately emits `{"type": "error", "message": "Unknown app"}`. The deploy button appears to work (the form submits, the progress UI appears), but the deployment always fails with an error message. This is a functional defect that will be user-visible.

**Fix:** Either add `neuralmesh-aidp` to the `deploy_stream` app map when the blueprint manifest is available, or disable/hide the deploy button in the template until the manifest is wired up:
```python
# In deploy_stream app_map (main.py ~line 2283):
"neuralmesh-aidp": os.path.join(BLUEPRINTS_DIR, "neuralmesh-aidp", "neuralmesh-aidp-stack.yaml"),
```
If the manifest doesn't exist yet, disable the button in the template:
```html
<button type="submit" disabled class="btn-purple w-full px-4 py-2 rounded-md text-sm font-medium opacity-50 cursor-not-allowed">Deploy Blueprint (coming soon)</button>
```

---

### CR-03: `delete_credential` calls `load_kube_config()` before name validation — contradicts "no I/O before validation" intent and misplaced error path

**File:** `app-store-gui/webapp/main.py:970-974`

**Issue:** `delete_credential` calls `load_kube_config()` on line 970 before the `_CREDENTIAL_NAME_RE` name validation check on line 973. `load_kube_config()` is a non-trivial network/filesystem operation (reads kubeconfig, may attempt in-cluster token discovery). A request with an invalid name incurs this cost unnecessarily. More critically, `get_weka_overview` validates the credential name *before* `load_kube_config()` (line 1032 vs line 1050) — the inconsistency shows the correct ordering was intended but not applied to `delete_credential`. The existing test `test_delete_credential_invalid_name_returns_400_without_io` only mocks out `load_kube_config`, so it passes despite this ordering problem; it does not verify that no I/O occurred at all.

**Fix:** Move the name validation before the `load_kube_config()` call, matching the pattern in `get_weka_overview`:
```python
@app.delete("/api/credentials/{name}")
async def delete_credential(name: str, namespace: str = Query("default")):
    try:
        # Validate first — no I/O on invalid input
        if not _CREDENTIAL_NAME_RE.match(name):
            return JSONResponse({"ok": False, "error": "invalid credential name"}, status_code=400)

        load_kube_config()
        ns = namespace.strip() or "default"
        ...
```

---

## Warnings

### WR-01: `warpSyncEndpoint` JS function redefined if macro is used more than once per page

**File:** `app-store-gui/webapp/templates/_credential_macros.html:60-65`

**Issue:** The `weka_storage_select` macro emits a `<script>` block defining the global `warpSyncEndpoint` function. If this macro is called more than once on the same page (e.g., two WEKA storage selectors), each invocation emits a `<script>` block that redefines `warpSyncEndpoint`. The second definition captures the `endpoint_field` from the second macro call (via the Jinja2-rendered literal `'{{ endpoint_field }}'`). This means the first selector's `onchange` handler would call the second selector's endpoint sync function, silently writing the wrong endpoint field. The `epId` computation `'ep-' + selectEl.name.replace(/^.*$/, '{{ endpoint_field }}')` is also unnecessarily complex — the regex `^.*$` always replaces the entire string, making the replace equivalent to `= 'ep-{{ endpoint_field }}'`.

**Fix:** Move the script to a single shared inline script that reads the target endpoint field ID from a `data-*` attribute on the select element, avoiding per-macro script emission:
```html
<select
  id="cred-{{ credential_field }}"
  name="{{ credential_field }}"
  required
  onchange="warpSyncEndpoint(this)"
  data-ep-target="ep-{{ endpoint_field }}"
  class="...">
```
```js
function warpSyncEndpoint(selectEl) {
  var epId = selectEl.dataset.epTarget;
  var epInput = document.getElementById(epId);
  var opt = selectEl.options[selectEl.selectedIndex];
  if (epInput && opt) { epInput.value = opt.dataset.endpoint || ''; }
}
```
Emit this script block once (e.g., in a separate `{% block scripts %}`) rather than inline per macro call.

---

### WR-02: `blueprint_detail` calls `get_cluster_status()` synchronously in an async route

**File:** `app-store-gui/webapp/main.py:1280`

**Issue:** `blueprint_detail` is an `async def` route. It calls `get_cluster_status()` directly (line 1280) as a synchronous call without `await asyncio.to_thread(...)`. `get_cluster_status()` calls `collect_cluster_inspection()` which performs Kubernetes API calls. This blocks the event loop for the duration of the K8s API call. The identical pattern in `index` and `settings_page` (lines 506 and 525) correctly wraps the call in `asyncio.to_thread`. The `blueprint_detail` handler was inconsistent — it later wraps `get_auth_status` in `asyncio.to_thread` (line 1318) but not `get_cluster_status`.

**Fix:** Wrap both blocking calls:
```python
# Replace line 1280:
status = await asyncio.to_thread(get_cluster_status)
```

---

### WR-03: Test fragility — bare `SimpleNamespace()` used as request object in `test_credential_macros.py`

**File:** `app-store-gui/tests/test_credential_macros.py:150, 167`

**Issue:** Two tests pass `SimpleNamespace()` (no attributes) as the `request` argument to `blueprint_detail`. Starlette's `TemplateResponse` constructor accesses `request.scope` to determine media type and other properties. With a bare `SimpleNamespace()`, this raises `AttributeError: 'types.SimpleNamespace' object has no attribute 'scope'` in some Starlette versions. The tests in `test_credentials_api.py` correctly use `SimpleNamespace(headers={}, cookies={}, query_params={}, url=SimpleNamespace(path="..."), scope={"type": "http"})`. The test at line 152 (`test_blueprint_detail_injects_credentials_by_type_into_context`) checks `response.context`, which requires the `TemplateResponse` to be constructed successfully.

The test at line 167 (`test_blueprint_detail_falls_back_to_default_namespace`) only asserts `captured_ns == ["default"]` and does not use the return value, but the coroutine will still raise if `TemplateResponse` fails before the function returns.

**Fix:** Use the same full request stub as `test_credentials_api.py`:
```python
request = SimpleNamespace(
    headers={}, cookies={}, query_params={},
    url=SimpleNamespace(path="/blueprint/neuralmesh-aidp"),
    scope={"type": "http"},
)
```

---

### WR-04: Substantial test duplication across `test_credential_macros.py` and `test_credentials_api.py`

**File:** `app-store-gui/tests/test_credential_macros.py:53-169`, `app-store-gui/tests/test_credentials_api.py:624-771`

**Issue:** Both files contain near-identical test coverage for `_get_credentials_by_type` and `blueprint_detail`. Seven `test_get_credentials_by_type_*` tests and two `test_blueprint_detail_*` tests appear in both files, testing the same code paths with the same fixture factories and monkeypatching helpers. `test_credential_macros.py` also re-imports `make_warpcred_cr_nvidia_ready`, `make_warpcred_cr_nvidia_not_ready`, and `make_warpcred_cr_weka_ready` from `test_credentials_api.py` rather than sharing them via a common fixture or conftest. This is not a test reliability failure by itself, but it creates a maintenance burden: changes to `_get_credentials_by_type` require updating assertions in two places, and the duplicate coverage makes test run time longer with no quality benefit.

**Fix:** Keep the canonical `_get_credentials_by_type` tests in `test_credentials_api.py` (the more complete set). Remove the duplicates from `test_credential_macros.py` and keep only the macro-specific rendering tests (which are the tests not duplicated: `test_get_credentials_by_type_groups_by_type` is a mild rename of `test_get_credentials_by_type_groups_ready_items`, `test_get_credentials_by_type_filters_non_ready_credentials` is duplicated by `test_get_credentials_by_type_filters_non_ready`). Move shared CR factories to `conftest.py`.

---

## Info

### IN-01: Unused `core` variable in `create_credential`

**File:** `app-store-gui/webapp/main.py:896`

**Issue:** `core = client.CoreV1Api()` is assigned on line 896 but never used directly in `create_credential`. The secret creation is delegated to `create_or_update_secret()` (called on line 897), which internally constructs its own `client.CoreV1Api()` instance. The `core` variable is only referenced later in the 409 rollback path (line 930: `core.delete_namespaced_secret`). This is not a bug — the rollback at line 929-937 does use `core` — but the variable is created before it's needed and in a different scope block from the code that uses it, which is slightly misleading.

**Fix:** No change required for correctness. As a minor clarity improvement, move `core = client.CoreV1Api()` to just before the CR creation block at line 917 where it is actually needed, or add a comment noting it is used in the 409 rollback.

---

### IN-02: Hardcoded "ok" CSS class used in template but not in CSS file under review

**File:** `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html:291`

**Issue:** A comment in the template (line 291-292) reads: `{# TEMPORARY (demo override): CPU and GPU badges hardcoded to green "ok". Revert by restoring the Jinja meets.cpu / meets.gpu conditionals from git history. #}`. The cluster compatibility section unconditionally shows green "ok" badges regardless of actual cluster state. This is a known temporary hack left in production code. If this template ships as-is, users will see misleading compatibility indicators even when their cluster does not meet requirements.

**Fix:** Restore the conditional badge rendering from git history before shipping to production:
```html
{% if meets.cpu is none %}
  <span class="badge" style="background: rgba(255,255,255,0.06);">unknown</span>
{% elif meets.cpu %}
  <span class="badge" style="background: #065f46; color: #d1fae5;">ok</span>
{% else %}
  <span class="badge" style="background: #7f1d1d; color: #fee2e2;">insufficient</span>
{% endif %}
```

---

_Reviewed: 2026-06-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
