---
phase: 23-backend-credentials-api-and-weka-overview-proxy
reviewed: 2026-06-11T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - app-store-gui/webapp/main.py
  - app-store-gui/webapp/templates/settings.html
  - app-store-gui/tests/test_credentials_api.py
findings:
  critical: 3
  warning: 5
  info: 3
  total: 11
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-06-11T00:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

This phase delivers three major additions: the WarpCredential CRUD API (`POST/GET/DELETE /api/credentials`), the WEKA REST API overview proxy (`GET /api/weka/overview`), and the Settings page refactor. The credential management logic, response sanitization, and WEKA proxy architecture are generally sound. The test suite provides good coverage of the happy paths and several edge cases.

However, three critical issues exist: a secret-before-CR ordering problem that leaves orphaned secrets on a naming collision, an unvalidated URL used directly in HTTP requests creating a SSRF vector, and a login-failure error message that may propagate raw WEKA error text (including embedded tokens or session IDs) to the caller. Five warnings cover logic gaps and robustness problems that will surface under realistic cluster conditions.

---

## Critical Issues

### CR-01: Orphaned raw Secret when WarpCredential CR creation races or conflicts

**File:** `app-store-gui/webapp/main.py:852-885`

**Issue:** `create_credential` creates the raw `warp-cred-<slug>` Secret (line 853) before attempting to create the WarpCredential CR (line 874). If the CR creation raises a 409 (line 883-884) or any other `ApiException`, the Secret has already been written but the CR was not. The handler returns a 409 to the caller, who will retry and generate a *new* slug via `_allocate_unique_credential_slug` — but the orphaned Secret (`warp-cred-<original-slug>`) is never cleaned up. Over time this accumulates unreferenced Secrets containing raw API keys, and the 409 path explicitly signals "retry", making the orphan nearly certain.

**Fix:** Reverse the creation order — create the CR first, then create (or update) the Secret. Alternatively, delete the Secret in the 409 catch before returning:

```python
        try:
            await asyncio.to_thread(
                co_api.create_namespaced_custom_object, ...
            )
        except ApiException as ae:
            if ae.status == 409:
                # Roll back the Secret we just created
                try:
                    await asyncio.to_thread(
                        client.CoreV1Api().delete_namespaced_secret,
                        name=f"warp-cred-{slug}",
                        namespace=ns,
                    )
                except Exception:
                    pass
                return JSONResponse({"ok": False, "error": f"slug {slug} already taken; retry"}, status_code=409)
            raise
```

The cleaner fix is to create the CR first (it carries no sensitive data) and only write the Secret after a successful CR create.

---

### CR-02: SSRF — WEKA endpoint URL from Secret is passed to HTTP client without validation

**File:** `app-store-gui/webapp/main.py:1501-1521` (`_resolve_weka_credential_secret`) and `app-store-gui/webapp/main.py:1015` (`get_weka_overview`)

**Issue:** `_resolve_weka_credential_secret` reads `WEKA_API_ENDPOINT` verbatim from a Kubernetes Secret (line 1519) and returns it. The caller in `get_weka_overview` passes it directly to `_weka_login` (line 1015) and then constructs three API URLs as `f"{base}/api/v2/..."` (lines 1025-1027). There is no validation that the URL is `https://` or that it points to a non-RFC-1918 host. An attacker with write access to the `warp-cred-*` Secret (or who can create a `weka-storage` credential) can redirect the proxy to `http://169.254.169.254/` (cloud metadata service), internal cluster endpoints, or `file://` URIs accepted by `urllib.request.urlopen`. This is a server-side request forgery (SSRF) vulnerability.

**Fix:** Add a URL validation step inside `_resolve_weka_credential_secret` before returning, or at the start of `get_weka_overview` after resolving:

```python
import urllib.parse

def _validate_weka_endpoint(endpoint: str) -> None:
    """Raise RuntimeError if endpoint is not a safe https:// URL."""
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in ("https", "http"):
        raise RuntimeError(f"WEKA endpoint must use https:// or http:// scheme, got: {parsed.scheme!r}")
    host = parsed.hostname or ""
    # Reject loopback, link-local, and cloud metadata IPs
    forbidden_prefixes = ("127.", "169.254.", "0.", "::1")
    if any(host.startswith(p) for p in forbidden_prefixes) or host in ("localhost",):
        raise RuntimeError(f"WEKA endpoint resolves to a forbidden host: {host!r}")
```

For a production service, requiring `https://` only and optionally restricting to a CIDR allowlist is strongly recommended.

---

### CR-03: WEKA login RuntimeError message may carry raw error text through to API response

**File:** `app-store-gui/webapp/main.py:1014-1017` and `app-store-gui/webapp/main.py:1451-1453`

**Issue:** `_weka_post_json` raises `RuntimeError(f"WEKA login failed: {exc.code}")` (line 1452), which is fine for HTTP status codes. However `_weka_login` is called through a try/except (lines 1014-1017) that catches `RuntimeError` and returns a fixed `"WEKA login failed"` string — good. But the outer try/except at lines 1043-1055 re-catches any `RuntimeError` with `err_str = str(e)` and, if it starts with `"WEKA "`, returns `err_str` directly to the caller (line 1054). The `_weka_get_json` helper (line 1431) raises `RuntimeError(f"WEKA API call failed: {url} -> {exc.reason}")` where `exc.reason` is the HTTPError reason string from the server. In some WEKA releases this reason string can include a URL that embeds a session token (as demonstrated in the test at line 546). If a WEKA data call (fileSystems/cluster/containers) fails, that URL — potentially including a token — reaches the caller's HTTP response body.

The test `test_weka_overview_login_failure_returns_502_without_leak` (line 536) validates the login path only, not the data-fetch paths.

**Fix:** Strip or replace the `url` portion in the RuntimeError messages raised by `_weka_get_json`:

```python
    except urllib.error.HTTPError as exc:
        # Never include the URL in the error; it may contain session tokens
        raise RuntimeError(f"WEKA API call failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"WEKA API call failed: connection error") from exc
```

---

## Warnings

### WR-01: `_config_loaded` singleton is never reset on config failure — subsequent calls silently skip re-loading

**File:** `app-store-gui/webapp/main.py:236-273`

**Issue:** `load_kube_config` uses a module-level `_config_loaded` flag (line 225). The flag is only set to `True` on success (line 261). On failure, the function raises `RuntimeError` before setting the flag, so the next call will retry — that is correct. However `_config_loaded` is **never reset** after a successful load. If credentials rotate (e.g. a new ServiceAccount token is mounted), the stale config is used for the process lifetime. More critically: if a test or caller patches kube config, the singleton means only the first load takes effect across the entire module lifetime. This is already causing a subtle test risk — `monkeypatch.setattr(main, "load_kube_config", lambda: None)` in tests works around it, but any test that calls `load_kube_config` directly instead of patching it will pick up the module-level flag and potentially skip loading entirely.

**Fix:** For a long-running server process this is an acceptable trade-off, but document it explicitly. For tests, add a fixture that resets `main._config_loaded = False` between test runs to avoid state leakage.

---

### WR-02: `_allocate_unique_credential_slug` truncation can silently produce an invalid slug after appending suffix

**File:** `app-store-gui/webapp/main.py:685-707`

**Issue:** `_make_credential_slug` truncates the slug to 52 characters (line 679). When `_allocate_unique_credential_slug` appends `-2` through `-99`, the resulting name can be up to 55 characters (`52 + len("-99") = 55`). The DNS-1123 subdomain limit is 253 characters, so this is not a hard constraint violation. However, the Kubernetes `metadata.name` field for most resources has a 63-character limit for label compatibility. At 55 characters, names are still valid, but the 52-character truncation was presumably chosen to leave headroom for a suffix. The issue is that no downstream validation confirms the final slug is within bounds, and the CRD spec limit is not checked. If the CRD enforces a shorter name limit, the create call will fail with a 422 after the secret was already written (exacerbating CR-01).

**Fix:** Reduce `_make_credential_slug` truncation limit from 52 to 48 characters, which guarantees the final candidate (`-99` suffix) stays within 51 characters — well below the 63-character common limit.

---

### WR-03: `get_weka_overview` uses `return_exceptions=False` in `asyncio.gather` — one WEKA endpoint failure aborts all three concurrent requests

**File:** `app-store-gui/webapp/main.py:1024-1029`

**Issue:** `asyncio.gather` with `return_exceptions=False` (the default, and what is explicitly set here) will raise on the first exception, cancelling the sibling tasks. If the `fileSystems` call succeeds but `containers` fails, the filesystem data is discarded and the caller sees a 502. The WEKA REST API is independently gateable per endpoint — a missing permission on `/containers` should not block capacity and filesystem data from being returned.

**Fix:** Use `return_exceptions=True`, then inspect each result:

```python
results = await asyncio.gather(
    asyncio.to_thread(_weka_get_json, f"{base}/api/v2/fileSystems", headers),
    asyncio.to_thread(_weka_get_json, f"{base}/api/v2/cluster", headers),
    asyncio.to_thread(_weka_get_json, f"{base}/api/v2/containers", headers),
    return_exceptions=True,
)
fs_resp = results[0] if not isinstance(results[0], Exception) else []
cluster_resp = results[1] if not isinstance(results[1], Exception) else {}
containers_resp = results[2] if not isinstance(results[2], Exception) else []
```

---

### WR-04: Inline `innerHTML` assignment in `settings.html` creates an XSS vector

**File:** `app-store-gui/webapp/templates/settings.html:325-330`

**Issue:** The blueprint table rows are constructed via `tr.innerHTML = \`...\`` (line 325), where `it.namespace` and `it.name` are interpolated directly. These values originate from Kubernetes `metadata.name` and `metadata.namespace`, which by API contract are DNS-1123 safe — but the template does not sanitize them. A malicious cluster operator who can create a CR with a crafted name (e.g. via a non-standard admission path or if the name validation has a gap) could inject HTML. Additionally, `it.creationTimestamp` goes through `new Date(...).toLocaleString()` which is fine, but `it.namespace` and `it.name` are used directly in HTML context.

The `data-ns` and `data-name` attributes on the Delete button (lines 329-330) are set via `innerHTML` template literals, which also leaves those attribute values unsanitized.

**Fix:** Use `textContent` assignment for each cell or a DOM creation approach instead of `innerHTML`:

```javascript
const tdNs = document.createElement('td');
tdNs.className = 'py-2 pr-4 font-mono';
tdNs.textContent = it.namespace || '-';
tr.appendChild(tdNs);
// ... similarly for name and timestamp
```

Alternatively, escape user-controlled strings before interpolation with a helper like `esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')`.

---

### WR-05: `POST /api/credentials` logs credential type and slug but `type` parameter is user-supplied

**File:** `app-store-gui/webapp/main.py:887`

**Issue:** `logger.info("Created WarpCredential: name=%s namespace=%s type=%s", slug, ns, type)` logs the `type` parameter directly. `type` is validated against `_VALID_CREDENTIAL_TYPES` before this point, so the value is safe in practice — but the parameter shadows Python's built-in `type` throughout the function (lines 798-900). This is a code quality issue that also means if the validation is ever refactored, a typo could produce unexpected log values. Additionally, `key` (the raw API key) is **not** logged, which is correct, but neither `username` nor `endpoint` are scrubbed from local variables — they linger in the stack frame. This is acceptable, but worth noting.

**Fix:** Rename the `type` parameter to `cred_type` throughout `create_credential` and `list_credentials` to avoid shadowing the built-in and to make the intent clearer:

```python
@app.post("/api/credentials")
async def create_credential(
    ...
    cred_type: str = Form(..., alias="type"),
    ...
):
```

---

## Info

### IN-01: Dead code — `_doc_id` function is defined but never called

**File:** `app-store-gui/webapp/main.py:453-455`

**Issue:** `_doc_id` is defined at the module level but is not referenced anywhere in `main.py`. It was likely used during development of `apply_blueprint_with_namespace`.

**Fix:** Remove the function or move it into `planning.py` if it is intended for use there.

---

### IN-02: `test_post_credential_weka_missing_username_returns_400` does not test missing `endpoint` case

**File:** `app-store-gui/tests/test_credentials_api.py:280-294`

**Issue:** There is a test for missing `username` but no corresponding test for missing `endpoint` when `type=weka-storage`. Both validations exist in the handler (lines 823-827), but only the username path is tested. The endpoint validation could be accidentally removed without a test catching it.

**Fix:** Add a companion test:

```python
def test_post_credential_weka_missing_endpoint_returns_400(monkeypatch):
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    response = asyncio.run(main.create_credential(
        display_name="X", type="weka-storage", namespace="default",
        key="tok", username="admin", endpoint=None,
    ))
    assert response.status_code == 400
    body = json.dumps(json.loads(response.body))
    assert "endpoint" in body.lower()
    assert "tok" not in body
```

---

### IN-03: `settings.html` loads React, React-DOM, Emotion, and MUI from CDN but never uses them

**File:** `app-store-gui/webapp/templates/settings.html:15-19`

**Issue:** Five CDN script tags load React 18, ReactDOM, Emotion, and MUI (roughly 1.5 MB of JavaScript) but no React component is mounted anywhere on the settings page. All interactivity is implemented in vanilla JS in the `<script>` block at the bottom. These scripts slow page load and expand the attack surface (CDN compromise).

**Fix:** Remove the five CDN `<script>` tags from `settings.html`. Keep only the Tailwind CDN tag which is actively used.

---

_Reviewed: 2026-06-11T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
