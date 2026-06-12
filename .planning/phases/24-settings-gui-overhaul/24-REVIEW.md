---
phase: 24-settings-gui-overhaul
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - app-store-gui/webapp/main.py
  - app-store-gui/webapp/templates/settings.html
findings:
  critical: 3
  warning: 5
  info: 2
  total: 10
status: issues_found
---

# Phase 24: Code Review Report

**Reviewed:** 2026-06-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the two files added/modified in phase 24: the updated `main.py` backend (settings endpoint, credentials API, WEKA overview proxy) and the new `settings.html` template. The credentials API and WEKA proxy are new production surfaces, so they carry the highest risk.

Three critical issues were found: (1) the `POST /api/credentials` response shape does not match what the frontend's `submitAddForm` handler expects, causing every credential creation to silently land in the error branch; (2) the `DELETE /api/credentials/{name}` endpoint is missing a namespace query parameter in the frontend delete calls, so every deletion uses the wrong (default) namespace when the detected namespace differs; (3) the `_validate_weka_endpoint` SSRF guard has an incomplete private-IP blocklist that leaves RFC-1918 and private IPv6 ranges open.

Five warnings cover: the `delete_credential` namespace missing from `wireDeleteButton`; the `visibility change` listener for polling re-creates intervals but leaks the old interval IDs that were cleared without being removed from the map; the `delete_namespaced_secret` call in the 409 rollback path creates a new `CoreV1Api()` instance directly without going through the shared client; the missing `app_name` default in `deploy_stream` allowing an AttributeError; and an operator-precedence bug in a compound boolean condition.

---

## Critical Issues

### CR-01: `POST /api/credentials` response shape mismatch — every credential creation fails silently

**File:** `app-store-gui/webapp/templates/settings.html:810`

**Issue:** `submitAddForm` checks `data.ok && data.item && data.item.name` to decide whether the save succeeded (line 810). However `POST /api/credentials` in `main.py` (lines 934–940) returns `{"ok": true, "name": slug, "namespace": ns, "type": ..., "displayName": ...}` — there is no `item` wrapper key. `data.item` is always `undefined`, so the condition is always falsy, the form never closes, and every successful credential creation is instead displayed as an error. The newly created credential only appears after a full page reload.

**Fix:**
Either change the API response to wrap the created credential under `item`:
```python
# main.py ~line 934
return JSONResponse({
    "ok": True,
    "item": {
        "name": slug,
        "namespace": ns,
        "type": cred_type,
        "displayName": display_name.strip(),
        "ready": False,   # newly created, operator has not reconciled yet
    },
})
```
or change the JS check to match the flat shape:
```js
// settings.html ~line 810
if (data && data.ok && data.name) {
  // use data (not data.item) as the cred object
  renderCredentialRow(li, data);
  startCredentialPoll(data.name);
}
```

---

### CR-02: `wireDeleteButton` deletes against `default` namespace regardless of detected namespace

**File:** `app-store-gui/webapp/templates/settings.html:563`

**Issue:** `wireDeleteButton` calls `del('/api/credentials/${encodeURIComponent(name)}')` (line 563) without appending the `?namespace=` query parameter. `DELETE /api/credentials/{name}` defaults `namespace` to `"default"` (main.py line 953). When the pod's detected namespace is anything other than `default` (e.g. `weka-app-store`), the delete request hits the wrong namespace and the Kubernetes API returns 404, showing "Kubernetes API error: 404 Not Found" to the user even though the credential actually exists.

**Fix:**
```js
// settings.html ~line 563
const data = await del(
  `/api/credentials/${encodeURIComponent(name)}?namespace=${encodeURIComponent(DETECTED_NAMESPACE)}`
);
```

---

### CR-03: SSRF guard in `_validate_weka_endpoint` is incomplete — RFC-1918 and IPv6 private ranges bypass the check

**File:** `app-store-gui/webapp/main.py:1481`

**Issue:** `_validate_weka_endpoint` only blocks `127.`, `169.254.`, `0.`, and `::1` as forbidden host prefixes (line 1481). The full RFC-1918 private-IP space (`10.x.x.x`, `172.16–31.x.x`, `192.168.x.x`) and private IPv6 ranges (`fc00::/7`, `fe80::`) are **not** blocked. An attacker who can set a WEKA credential's endpoint can supply `http://10.0.0.1:6443/...` and use the WEKA proxy as a pivot to reach internal cluster services (e.g., the Kubernetes API server or etcd). Because endpoints come from Kubernetes Secrets, exploitation requires write access to Secrets in the namespace, but the risk still constitutes a server-side request forgery vector.

**Fix:**
```python
def _validate_weka_endpoint(endpoint: str) -> None:
    """Raise RuntimeError if endpoint is not a safe https:// or http:// URL."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("https", "http"):
        raise RuntimeError(f"WEKA endpoint must use https:// or http:// scheme, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    # Block loopback, link-local, APIPA, and unspecified
    forbidden_prefixes = ("127.", "169.254.", "0.", "::1", "fc", "fe80")
    # Block RFC-1918 private ranges
    import ipaddress
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise RuntimeError(f"WEKA endpoint resolves to a forbidden address: {host!r}")
    except ValueError:
        pass  # hostname (not a bare IP) — prefix check still applies
    if any(host.startswith(p) for p in forbidden_prefixes) or host in ("localhost",):
        raise RuntimeError(f"WEKA endpoint resolves to a forbidden host: {host!r}")
```

---

## Warnings

### WR-01: `visibilitychange` polling resume leaks old interval IDs

**File:** `app-store-gui/webapp/templates/settings.html:930-938`

**Issue:** When `document.hidden` becomes false (tab re-focused), the handler re-creates `setInterval` calls for every name in `pollIntervals` and overwrites the map value (line 935–936). But when the page was hidden, the handler only called `pollIntervals.forEach(clearInterval)` (line 932) — it cleared the interval timers but did **not** remove the entries from the `pollIntervals` map. As a result, on resume every polling name gets a new interval, and the old IDs that were stored in the map (now cancelled timers) are silently discarded but the count of active timers doubles with each hide/show cycle. Duplicate polling is harmless for short sessions but causes excessive API calls if the user switches tabs repeatedly.

**Fix:**
When hiding, clear intervals and also record them as needing resumption, or clear and empty the map, using a separate tracking set:
```js
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    pollIntervals.forEach((id, name) => clearInterval(id));
    // Mark all as paused (preserve names but clear IDs so resume creates fresh intervals)
    const names = Array.from(pollIntervals.keys());
    pollIntervals.clear();
    names.forEach(n => pollIntervals.set(n, null));  // null = paused
  } else {
    Array.from(pollIntervals.keys()).forEach(name => {
      const id = setInterval(() => pollCredentialOnce(name), POLL_MS);
      pollIntervals.set(name, id);
    });
  }
});
```
(Also update `stopCredentialPoll` and the `forEach(clearInterval)` in `beforeunload` to guard against null IDs.)

---

### WR-02: 409 rollback path creates an un-configured `CoreV1Api` instance

**File:** `app-store-gui/webapp/main.py:922-927`

**Issue:** Inside the 409 error handler for the WarpCredential CR creation (line 921), the rollback calls `client.CoreV1Api().delete_namespaced_secret(...)`. This constructs a brand-new `CoreV1Api` with a default `ApiClient`, which re-reads the global default configuration. If `_config_loaded` has been set but the underlying client configuration has been customized (e.g., in-cluster token injected after the first load), this may work, but it bypasses the intentional single-client pattern used everywhere else and is fragile. More concretely: `create_or_update_secret` already received the `core = client.CoreV1Api()` that was used for the create — but that object is out of scope here, requiring a re-construction.

**Fix:** Either hoist the `core` variable to be reused in the rollback, or pass it explicitly:
```python
# create the shared core client before the try block
core = client.CoreV1Api()
await asyncio.to_thread(create_or_update_secret, f"warp-cred-{slug}", ns, string_data)
# ... inside the 409 handler:
try:
    await asyncio.to_thread(
        core.delete_namespaced_secret,
        name=f"warp-cred-{slug}",
        namespace=ns,
    )
except Exception:
    pass
```

---

### WR-03: `deploy_stream` raises `AttributeError` when `app_name` is absent from query-style request

**File:** `app-store-gui/webapp/main.py:2228-2231`

**Issue:** The route is registered as both `@app.get("/deploy-stream/{app_name}")` and `@app.get("/deploy-stream")`. When a client hits `/deploy-stream` without the `{app_name}` path segment and without an `?app_name=` query parameter, FastAPI raises a `422 Unprocessable Entity` before the handler runs — which is fine — but the signature declares `app_name: str` without a default, so the bare path variant is actually unreachable without a value. More critically, `yaml_path = app_map.get(app_name)` at line 2259 is evaluated **before** the generator begins, so if `app_name` is somehow empty or not in the map, `yaml_path` is `None` and the SSE stream produces an error event correctly. However, the `app_name` Path parameter on the dual-route declaration has no default, meaning FastAPI will reject queries to `/deploy-stream?app_name=foo` with a 422 because it expects a path parameter. The correct pattern for an optional path segment is to declare a default.

**Fix:**
```python
@app.get("/deploy-stream/{app_name}")
@app.get("/deploy-stream")
async def deploy_stream(
    request: Request,
    app_name: str = Query(...),  # or accept via Path with Optional
    ...
):
```
Or, since both routes need to be handled, use a single route with an `Optional` path parameter via `app_name: Optional[str] = None` and validate inside.

---

### WR-04: Operator-precedence bug in compound boolean condition

**File:** `app-store-gui/webapp/main.py:1882`

**Issue:** The condition on line 1882 is:
```python
if cond.get("type") in ["Error", "Failed"] or cond.get("status") == "False" and cond.get("type") in ["Ready", "Initialized"]:
```
Due to Python operator precedence, `and` binds tighter than `or`, so this parses as:
```python
if (cond.get("type") in ["Error", "Failed"]) or (cond.get("status") == "False" and cond.get("type") in ["Ready", "Initialized"]):
```
This is likely the intended reading, but the absence of explicit parentheses is fragile and error-prone for future maintainers who may misread it as:
```python
if (cond.get("type") in ["Error", "Failed"] or cond.get("status") == "False") and ...:
```
This would incorrectly fire on any condition where `status == "False"` even when `type` is not in `["Ready", "Initialized"]`.

**Fix:** Add explicit parentheses to make the intent unambiguous:
```python
if (
    cond.get("type") in ["Error", "Failed"]
    or (cond.get("status") == "False" and cond.get("type") in ["Ready", "Initialized"])
):
```

---

### WR-05: `_fetch_credentials` in `settings_page` silently swallows all exceptions including programming errors

**File:** `app-store-gui/webapp/main.py:539-542`

**Issue:** The inner `_fetch_credentials` function catches bare `Exception` (line 542) and returns an empty list. This means a `TypeError` from a broken `_build_credential_response_item`, a `NameError`, or any other programming-error exception is silently eaten and the settings page renders with empty credential lists — giving no diagnostic signal. The bare `ApiException` catch at line 539 is appropriate, but the blanket `except Exception: return []` masks bugs.

**Fix:**
```python
except ApiException:
    return []
# Do not catch bare Exception here; let programming errors propagate so they surface in logs.
```
If silent degradation is intentional (k8s not reachable), narrow the catch to known transient failure types:
```python
except (ApiException, ConnectionError, TimeoutError):
    return []
```

---

## Info

### IN-01: Unused React/MUI CDN scripts loaded on every settings page

**File:** `app-store-gui/webapp/templates/settings.html:15-19`

**Issue:** Five CDN scripts (React 18, ReactDOM 18, Emotion React/Styled, MUI Material) are loaded in the `<head>` (lines 15–19) but the settings page uses no React or MUI components — all UI is plain HTML + vanilla JS. These scripts add ~600 KB of network transfer per page load for no benefit.

**Fix:** Remove the five CDN script tags from `settings.html`. They appear to have been copied from another template that does use React/MUI.

---

### IN-02: `detected_namespace` template variable used raw in a JS `const` without `| tojson` quoting guard on the fallback branch

**File:** `app-store-gui/webapp/templates/settings.html:496`

**Issue:** Line 496 renders:
```
const DETECTED_NAMESPACE = {{ (detected_namespace or 'default') | tojson }};
```
The `| tojson` filter is applied, which is correct. However, the Python side passes `detected_namespace` as a string (line 560 in main.py: `"detected_namespace": detected_ns or "default"`). The `or 'default'` inside the template expression is therefore redundant (it can never be falsy because main.py already guarantees a string value). This is not a bug today, but if someone removes the server-side fallback and `detected_namespace` becomes `None`, Jinja2 will render `const DETECTED_NAMESPACE = null;` (valid JSON), not an unquoted None — so the `tojson` filter protects correctly. The redundancy is just a clarity issue.

**Fix:** No code change required; the `| tojson` filter is the right defence. Consider removing the redundant `or 'default'` in the template to avoid confusion:
```
const DETECTED_NAMESPACE = {{ detected_namespace | tojson }};
```

---

_Reviewed: 2026-06-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
