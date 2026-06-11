# Phase 23: Backend Credentials API and WEKA Overview Proxy - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 3 (1 modified with new routes + removals, 1 modified HTML/JS removal, 1 new test file)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `app-store-gui/webapp/main.py` (new routes + removals) | controller/service | CRUD + request-response | `app-store-gui/webapp/main.py` lines 573-638 | exact |
| `app-store-gui/webapp/templates/settings.html` (JS removal) | template | n/a | `app-store-gui/webapp/templates/settings.html` lines 52-112, 321-440 | exact (sections to delete) |
| `app-store-gui/tests/test_credentials_api.py` | test | request-response | `app-store-gui/tests/planning/test_apply_gateway.py` | role-match |

---

## Pattern Assignments

### `app-store-gui/webapp/main.py` — New `/api/credentials` Routes and `/api/weka/overview` Route

**Analog:** `app-store-gui/webapp/main.py` (existing routes at lines 535-638, 573-623, 1016-1066)

---

#### Imports pattern (lines 1-18)

All imports needed by new routes are already present. No new imports required:

```python
from fastapi import FastAPI, Request, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from typing import Optional, Dict, Any, List
import os
import base64
import json
import time
import asyncio
import re

from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
# urllib.request imported locally inside the git-sync downloader (line 1480); import at module
# level for the WEKA overview proxy — it is already available in stdlib, no requirements.txt change.
```

---

#### `create_or_update_secret()` pattern — reuse for `warp-cred-<slug>` Secret creation (lines 535-555)

```python
def create_or_update_secret(name: str, namespace: str, string_data: Dict[str, str]) -> Dict[str, Any]:
    """Create or update an Opaque secret with given string_data."""
    load_kube_config()
    ensure_namespace_exists(namespace)
    core = client.CoreV1Api()
    metadata = client.V1ObjectMeta(name=name, namespace=namespace)
    secret_body = client.V1Secret(metadata=metadata, type="Opaque", string_data=string_data)
    try:
        core.create_namespaced_secret(namespace=namespace, body=secret_body)
        return {"name": name, "namespace": namespace, "action": "created"}
    except ApiException as ae:
        if ae.status == 409:
            patched = client.V1Secret(metadata=metadata, type="Opaque", string_data=string_data)
            core.patch_namespaced_secret(name=name, namespace=namespace, body=patched)
            return {"name": name, "namespace": namespace, "action": "updated"}
        raise
```

**How to use for Phase 23:** Call `create_or_update_secret(f"warp-cred-{slug}", namespace, string_data)` where `string_data` keys depend on credential type per D-08.

---

#### `list_blueprints()` — CustomObjectsApi list pattern (lines 573-600)

```python
@app.get("/api/blueprints")
async def list_blueprints(namespace: str = Query("all", ...)):
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        items = []
        if (namespace or "").strip().lower() in ("all", "*"):
            resp = co_api.list_cluster_custom_object(group="warp.io", version="v1alpha1", plural="wekaappstores")
        else:
            resp = co_api.list_namespaced_custom_object(group="warp.io", version="v1alpha1", plural="wekaappstores", namespace=namespace.strip())
        for it in (resp or {}).get("items", []) or []:
            md = (it or {}).get("metadata", {}) or {}
            items.append({
                "name": md.get("name"),
                "namespace": md.get("namespace") or "default",
                "creationTimestamp": md.get("creationTimestamp"),
            })
        return JSONResponse({"ok": True, "items": items})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

**How to use for Phase 23 `GET /api/credentials`:** Replace `wekaappstores` with `warpcredentials`. Replace `plural="wekaappstores"` with `plural="warpcredentials"`. Build the response shape from the CR's `spec` and `status` fields (conditions, derivedSecrets, wekaEndpoint) — never from the raw Secret data. For `?type=<t>` filtering, filter items where `spec.type == t` AND the `KeyReady` condition has `status == "True"`.

---

#### `delete_blueprint()` — delete_namespaced_custom_object pattern (lines 603-623)

```python
@app.delete("/api/blueprints/{namespace}/{name}")
async def delete_blueprint(namespace: str, name: str):
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        body = client.V1DeleteOptions(propagation_policy="Foreground")
        co_api.delete_namespaced_custom_object(
            group="warp.io",
            version="v1alpha1",
            namespace=namespace,
            plural="wekaappstores",
            name=name,
            body=body,
        )
        return JSONResponse({"ok": True})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

**How to use for Phase 23 `DELETE /api/credentials/<name>`:** Delete the WarpCredential CR first (group=`warp.io`, plural=`warpcredentials`), then delete the `warp-cred-<name>` Secret via `core.delete_namespaced_secret(name=f"warp-cred-{name}", namespace=namespace)`. Do NOT delete any `warp-<name>-*` derived secrets. Per D-03, call `load_kube_config()` at the top of the handler.

---

#### `POST /api/secret/huggingface` — Form-based POST handler pattern to REMOVE + replicate shape (lines 558-570)

This handler is being **removed** per D-13. Its structure is the pattern to replicate for `POST /api/credentials`:

```python
@app.post("/api/secret/huggingface")
async def save_huggingface_key(api_key: str = Form(...), namespace: str = Form("default")):
    try:
        result = create_or_update_secret(
            name="hf-api-key",
            namespace=namespace.strip() or "default",
            string_data={"HF_API_KEY": api_key},
        )
        return JSONResponse({"ok": True, "secret_name": result["name"], "namespace": result["namespace"], "action": result["action"]})
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

**How to use for `POST /api/credentials`:** Use `Form(...)` fields — `display_name`, `type`, `namespace`, plus type-specific fields (`key`, `username`, `endpoint`). Follow the same `try / ApiException / Exception` structure. Return `{"ok": True, "name": slug, "namespace": ..., "type": ...}` — no credential values in response per D-16.

---

#### `asyncio.to_thread` pattern for sync calls inside async handlers (lines 493-496)

```python
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    status = await asyncio.to_thread(get_cluster_status)
    auth = await asyncio.to_thread(get_auth_status)
    return templates.TemplateResponse(...)
```

**How to use for Phase 23:** Wrap sync kubernetes-client calls and `urllib.request.urlopen` calls in `asyncio.to_thread(...)`. For the WEKA overview route, use `asyncio.gather` to run three parallel `asyncio.to_thread` coroutines (fileSystems, cluster, containers) after the login step per D-05.

---

#### Module-level cache dict pattern — `_last_ready_cache` (lines 1016-1066)

```python
_last_ready_cache: dict = {"ts": 0.0, "resp": {"ok": False, "ready": False}}

@app.get("/readyz")
async def readyz():
    try:
        ttl = float(os.getenv("READINESS_TTL_SECONDS", "5"))
        now = time.time()
        if ttl > 0 and (now - _last_ready_cache["ts"]) < ttl:
            return JSONResponse(_last_ready_cache["resp"], status_code=200 if _last_ready_cache["resp"].get("ready") else 503)

        # ... do real work ...

        # Cache result
        _last_ready_cache["ts"] = now
        _last_ready_cache["resp"] = resp

        return JSONResponse(resp, status_code=200 if ready else 503)
    except Exception as e:
        resp = {"ok": False, "ready": False, "error": str(e)}
        _last_ready_cache["ts"] = time.time()
        _last_ready_cache["resp"] = resp
        return JSONResponse(resp, status_code=503)
```

**How to use for `_weka_overview_cache` (D-06):** Place the dict near `_last_ready_cache` at module level:

```python
_weka_overview_cache: dict[str, dict] = {}
```

Each entry: `{"ts": float, "data": dict}`. TTL check: `time.time() - entry["ts"] < 60`. Cache key: the credential `metadata.name` string. Bypass when `?bust=1` is present (skip TTL check, re-fetch, update entry) per D-07.

---

#### Error handling convention (lines 568, 598-599, 620-623)

```python
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

For WEKA API auth failures on the overview proxy: return `JSONResponse({"ok": False, "error": "..."}, status_code=502)` per Claude's Discretion note in CONTEXT.md.

---

#### Slug generation (new logic, no direct analog — implement per D-11 and D-12)

```python
import re

def _make_credential_slug(display_name: str) -> str:
    slug = display_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:52]
```

Collision check per D-12: list existing `warpcredentials` via `CustomObjectsApi.list_namespaced_custom_object`, collect existing `metadata.name` values, then increment suffix (`-2`, `-3`, ...) until unique.

---

### `app-store-gui/webapp/templates/settings.html` — Sections to Remove

**Analog:** Same file (these are the exact blocks to delete)

#### HTML sections to remove (lines 52-112)

```html
<!-- Section 1: HuggingFace API Key -->
<section class="card rounded-lg p-5">
  ...  (lines 52-81)
</section>

<!-- Section 2: NVIDIA API Key -->
<section class="card rounded-lg p-5">
  ...  (lines 83-112)
</section>
```

Remove both `<section>` blocks in their entirety. The section immediately following (Section 3: Kubernetes Auth Status, line 114) stays.

#### JavaScript blocks to remove

**`getSettingsNamespace` / `setSettingsNamespace` / `renderSecrets` / `loadSecrets` block (lines 321-383):**
All four functions depend exclusively on HuggingFace and NVIDIA DOM elements (`hf-namespace`, `nvidia-namespace`, `hf-secrets-list`, `nvidia-secrets-list`). Remove entire block.

**`hfBtn` click handler block (lines 386-412):**
```javascript
// Save HF key
const hfBtn = document.getElementById('hf-save');
if (hfBtn) {
  hfBtn.addEventListener('click', async () => { ... });
}
```
Remove in full.

**`nvBtn` click handler block (lines 414-440):**
```javascript
// Save NVIDIA key
const nvBtn = document.getElementById('nvidia-save');
if (nvBtn) {
  nvBtn.addEventListener('click', async () => { ... });
}
```
Remove in full.

**`loadSecrets()` call at line 485:**
```javascript
loadSecrets();
```
Remove this standalone call (it is the only reference to `loadSecrets` outside its definition). The `refreshAuthStatus()` call on line 486 and `setInterval` on line 487 stay.

**Note:** Line 546 references `getSettingsNamespace()` in the blueprint uninstall section — check whether that reference survives. If the blueprint scope picker still uses `getSettingsNamespace`, that function must be kept or its body inlined. The CONTEXT.md D-14 specifies removing the function only where it "only supports the old secret list panels". Executor must inspect line 546 context carefully before removing.

---

### `app-store-gui/tests/test_credentials_api.py` — New Test File

**Analog:** `app-store-gui/tests/planning/test_apply_gateway.py`

#### Test file structure pattern

```python
from __future__ import annotations

import pytest
import unittest.mock

# Import the handler functions directly from webapp.main
import webapp.main as main
```

#### Stub injection pattern (from test_apply_gateway.py lines 34-55)

```python
class CustomObjectsApiStub:
    def create_namespaced_custom_object(self, **kwargs):
        operations.append(("create_namespaced_custom_object", kwargs))

    def list_namespaced_custom_object(self, **kwargs):
        # Return fixture data
        return {"items": [...]}

    def delete_namespaced_custom_object(self, **kwargs):
        operations.append(("delete_namespaced_custom_object", kwargs))
```

**For Phase 23 tests:** Inject via `unittest.mock.patch`:

```python
def test_get_credentials_shape(monkeypatch):
    class CustomObjectsApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            return {"items": [_make_credential_cr_fixture()]}

    class CoreV1ApiStub:
        def delete_namespaced_secret(self, **kwargs):
            operations.append(("delete_namespaced_secret", kwargs))

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CustomObjectsApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreV1ApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    # call handler directly or via TestClient
```

#### Coverage required (D-16)

- `GET /api/credentials` response shape — all status fields present, no credential values in response
- `POST /api/credentials` slug generation from `displayName`
- `POST /api/credentials` slug collision appends `-2`, `-3` suffix until unique
- `DELETE /api/credentials/<name>` deletes CR then raw Secret; does not touch derived secrets
- `GET /api/credentials?type=<t>` filters to `ready=True` items only
- `GET /api/weka/overview` cache behavior — second call within 60s returns cached data without re-fetching
- `GET /api/weka/overview?bust=1` bypasses cache and re-fetches

#### Test runner invocation pattern (from CLAUDE.md)

```bash
PYTHONPATH=mcp-server:app-store-gui pytest app-store-gui/tests/test_credentials_api.py -v
```

---

## Shared Patterns

### Error Handling
**Source:** `app-store-gui/webapp/main.py` lines 568, 598-599, 621-623
**Apply to:** All new `/api/credentials` and `/api/weka/overview` handlers

```python
    except ApiException as ae:
        return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

For WEKA REST API failures use `status_code=502` instead of 500 (gateway error convention).

### Success Response Shape
**Source:** `app-store-gui/webapp/main.py` lines 566, 595, 619
**Apply to:** All new handlers

```python
return JSONResponse({"ok": True, ...})
```

### `load_kube_config()` at top of each handler
**Source:** `app-store-gui/webapp/main.py` lines 541, 581, 607
**Apply to:** All new route handlers (D-03)

```python
@app.get("/api/credentials")
async def list_credentials(...):
    try:
        load_kube_config()
        co_api = client.CustomObjectsApi()
        ...
```

### `asyncio.to_thread` wrapping sync calls
**Source:** `app-store-gui/webapp/main.py` lines 495-496
**Apply to:** All new async route handlers that call sync kubernetes-client or urllib methods

```python
result = await asyncio.to_thread(sync_function, arg1, arg2)
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| slug generation logic in `POST /api/credentials` | utility | transform | No slug generation exists in this codebase; implement per D-11/D-12 using `re.sub` |
| WEKA REST API HTTP proxy in `GET /api/weka/overview` | service | request-response | No external HTTP proxy route exists in codebase; closest is the urllib.request usage at line 1480-1481 for git-sync binary download |

For the WEKA overview proxy, the git-sync download code at lines 1478-1491 shows the `urllib.request.urlopen` pattern used in the codebase:

```python
import urllib.request
with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
    shutil.copyfileobj(r, f)
```

Adapt for WEKA REST API calls: use `urllib.request.Request` with `Authorization: Bearer <token>` header, parse JSON response, wrap with `asyncio.to_thread` per D-04.

---

## WarpCredential CR Schema Reference (locked by Phase 21)

**Source:** `weka-app-store-operator-chart/templates/crd.yaml` lines 279-382

`spec` fields: `type` (enum: nvidia-ngc, huggingface, weka-storage), `displayName`, `secretRef.name`, `secretRef.key`, `endpoint` (weka-storage only).

`status` fields:
- `conditions[]` — array of `{type, status, reason, message, lastTransitionTime}`; condition type `KeyReady` drives `ready` flag; `DockerSecretReady` is nvidia-ngc only
- `derivedSecrets[]` — array of `{name, type}`
- `lastSyncTime` — ISO timestamp
- `wekaEndpoint` — string (weka-storage only; safe to expose, unlike raw Secret)

`GET /api/credentials` response shape per item:
```json
{
  "name": "my-ngc-cred",
  "namespace": "default",
  "type": "nvidia-ngc",
  "displayName": "My NGC Credential",
  "ready": true,
  "dockerSecretReady": true,
  "derivedSecrets": [{"name": "warp-ngc-my-ngc-cred-docker", "type": "kubernetes.io/dockerconfigjson"}],
  "lastSyncTime": "2026-06-11T00:00:00Z",
  "endpoint": null
}
```
Never include raw key values.

---

## Metadata

**Analog search scope:** `app-store-gui/webapp/main.py`, `app-store-gui/webapp/templates/settings.html`, `app-store-gui/tests/planning/test_apply_gateway.py`, `app-store-gui/tests/conftest.py`, `weka-app-store-operator-chart/templates/crd.yaml`
**Files scanned:** 6
**Pattern extraction date:** 2026-06-11
