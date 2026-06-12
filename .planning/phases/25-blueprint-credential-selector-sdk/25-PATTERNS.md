# Phase 25: Blueprint Credential Selector SDK — Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 4 (1 new template, 1 modified route module, 1 modified blueprint template, 1 new test)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `app-store-gui/webapp/templates/_credential_macros.html` (NEW) | Jinja2 macro template (SDK) | server-render, request-response | `app-store-gui/webapp/templates/settings.html` (Tailwind form/empty-state) | role-match (no existing macro file in repo) |
| `app-store-gui/webapp/main.py` (MODIFY) | FastAPI route module + helper extraction | request-response | `settings_page` at `main.py:522-564` (inline `_fetch_credentials` + dict build + context inject) | exact |
| `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html` (MODIFY) | Jinja2 blueprint detail template | server-render | `blueprint_openfold.html:212` (`{% include %}` proves Jinja2 directives work); macro call requires `{% from ... import %}` | role-match |
| `app-store-gui/tests/test_credential_macros.py` (NEW) | pytest unit test (stub injection, no TestClient) | n/a | `app-store-gui/tests/test_credentials_api.py` (monkeypatch `client.CustomObjectsApi` + `load_kube_config`, `asyncio.run` route handler) | exact |

---

## Pattern Assignments

### 1. `app-store-gui/webapp/main.py` — extract `_get_credentials_by_type(ns)` helper

**Analog (exact):** `settings_page` inline `_fetch_credentials` + `credentials_by_type` block at `main.py:530-549`.

**Block to extract verbatim** (`main.py:530-549`):

```python
async def _fetch_credentials() -> list:
    def _list():
        return client.CustomObjectsApi().list_namespaced_custom_object(
            group="warp.io", version="v1alpha1",
            plural="warpcredentials", namespace=ns,
        )
    try:
        load_kube_config()
        resp = await asyncio.to_thread(_list)
        return [_build_credential_response_item(cr) for cr in (resp or {}).get("items", []) or []]
    except (ApiException, ConnectionError, TimeoutError):
        return []

cred_items = await _fetch_credentials()

credentials_by_type: dict = {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}
for it in cred_items:
    t = it.get("type")
    if t in credentials_by_type:
        credentials_by_type[t].append(it)
```

**Refactor target — module-level helper signature (per D-01, D-02):**

```python
async def _get_credentials_by_type(ns: str) -> dict:
    """Return credentials grouped by type for a namespace.

    Returns {"nvidia-ngc": [...], "huggingface": [...], "weka-storage": [...]}.
    Falls back to empty dict-of-lists on ApiException | ConnectionError | TimeoutError.
    """
    credentials_by_type: dict = {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}

    def _list():
        return client.CustomObjectsApi().list_namespaced_custom_object(
            group="warp.io", version="v1alpha1",
            plural="warpcredentials", namespace=ns,
        )
    try:
        load_kube_config()
        resp = await asyncio.to_thread(_list)
        for cr in (resp or {}).get("items", []) or []:
            item = _build_credential_response_item(cr)
            t = item.get("type")
            if t in credentials_by_type:
                credentials_by_type[t].append(item)
    except (ApiException, ConnectionError, TimeoutError):
        pass
    return credentials_by_type
```

**`settings_page` rewrite (lines 522-564) — call helper, preserve existing behavior:**

```python
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = await asyncio.to_thread(get_auth_status)
    status = await asyncio.to_thread(get_cluster_status)
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    ns = detected_ns or "default"

    credentials_by_type = await _get_credentials_by_type(ns)
    weka_storage_credentials = [c for c in credentials_by_type["weka-storage"] if c.get("ready")]

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "auth": auth,
            "status": status,
            "detected_namespace": detected_ns or "default",
            "logo_b64": LOGO_B64,
            "credentials_by_type": credentials_by_type,
            "weka_storage_credentials": weka_storage_credentials,
        },
    )
```

**Placement:** Near other module-level helpers, e.g., immediately after `_build_credential_response_item` (`main.py:744-783`). This co-locates with the dependency it consumes.

**Critical constraints:**
- Exception set is exactly `(ApiException, ConnectionError, TimeoutError)` — must match existing inline block (no broadening, no narrowing).
- Helper calls `load_kube_config()` internally (same as inline version).
- Helper is `async def` because it uses `await asyncio.to_thread(_list)`.
- Three known types `nvidia-ngc`, `huggingface`, `weka-storage` are pre-seeded as empty lists so templates never KeyError.

---

### 2. `app-store-gui/webapp/main.py` — `blueprint_detail` context injection

**Analog (exact):** `settings_page` template context injection pattern at `main.py:524-564`.

**Insertion site in `blueprint_detail`** (`main.py:1257-1325`).

**Current shape of `blueprint_detail` template context** (`main.py:1308-1324`):

```python
return templates.TemplateResponse(
    request,
    use_template,
    {
        "request": request,
        "name": name,
        "yaml_path": yaml_path,
        "status": status,
        "requirements": reqs,
        "meets": meets,
        "oss_img_b64": oss_img_b64,
        "aidp_img_b64": aidp_img_b64,
        "logo_b64": LOGO_B64,
        "glocomp_logo_b64": GLOCOMP_LOGO_B64,
        "tokenvisor_logo_b64": TOKENVISOR_LOGO_B64,
        "tokenvisor_arch_b64": TOKENVISOR_ARCH_B64,
    },
)
```

**Add — `get_auth_status` namespace resolution + helper call before TemplateResponse (per D-03, D-04):**

Insert near the top of `blueprint_detail` (after line 1258, before `app_map` or before TemplateResponse — placement before TemplateResponse keeps the existing ordering intact):

```python
# Resolve namespace for credential lookup (same pattern as settings_page:524-528)
auth = await asyncio.to_thread(get_auth_status)
detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
ns = detected_ns or "default"
credentials_by_type = await _get_credentials_by_type(ns)
```

Then add `credentials_by_type` to the template-context dict:

```python
return templates.TemplateResponse(
    request,
    use_template,
    {
        "request": request,
        "name": name,
        # ... existing keys ...
        "tokenvisor_arch_b64": TOKENVISOR_ARCH_B64,
        "credentials_by_type": credentials_by_type,  # NEW
    },
)
```

**Critical constraints:**
- Must use `await asyncio.to_thread(get_auth_status)` — `get_auth_status` is synchronous (defined at `main.py:1134` as `def get_auth_status`). Wrapping pattern is established at `main.py:507` (`index` route) and `main.py:524-525` (`settings_page`).
- `detected_ns or "default"` fallback is verbatim from `settings_page:528`.
- Do NOT change the `blueprint_detail` route signature; route remains `async def blueprint_detail(request: Request, name: str)`.
- Existing template selection logic (`use_template = "blueprint.html"` fallback at lines 1299-1306) is untouched — the new key is injected for every blueprint template, not just AIDP. Templates that don't reference `credentials_by_type` will simply ignore it.

---

### 3. `app-store-gui/webapp/templates/_credential_macros.html` — NEW Jinja2 macro file

**Analog (role-match):** No existing Jinja2 macro file in this repo. Closest patterns:
- `settings.html:686-707` — exact Tailwind classes for `<label>` + `<input>`/`<select>` form controls (matches UI-SPEC Acceptance Criterion 1)
- `settings.html:136-140` — exact empty-state hint pattern (matches UI-SPEC Acceptance Criterion 2 and 8)
- `blueprint_openfold.html:212` — proves Jinja2 directives are honored by `Jinja2Templates(directory=TEMPLATES_DIR)` at `main.py:184`; macros use `{% from ... import %}`, not `{% include %}` (per D-05)

**Empty-state hint analog excerpt** (`settings.html:135-140`):

```html
<h3 class="font-semibold mb-2">WEKA Storage Overview</h3>
<p class="muted text-sm">
  No WEKA Storage credential configured.
  <a href="#weka-credentials" class="text-[var(--weka-purple)] underline hover:text-white">Add one above.</a>
</p>
```

The macro empty state diverges from this analog only in: (a) link target is `/settings#credentials` (cross-page) instead of `#weka-credentials`, and (b) link text is `Add one in Settings.` instead of `Add one above.`. UI-SPEC §Copywriting Contract locks these strings.

**Form control analog excerpt** (`settings.html:686-689`):

```html
<label for="${type}-name" class="block text-xs font-medium text-white/70 mb-1">Name</label>
<input id="${type}-name" name="display_name" type="text" required
  class="w-full px-3 py-2 rounded-md bg-gray-800/70 border border-white/10 focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)] text-sm"
  placeholder="e.g. weka-prod" />
```

This is the verbatim class string the macros must use on every `<input>` and `<select>` (UI-SPEC Acceptance Criterion 1 + 3).

**Exact macro output is locked by UI-SPEC §Component Contract** (`25-UI-SPEC.md:101-200`). The planner should reference UI-SPEC directly — do NOT paraphrase. Quick anchors:
- `credential_select` populated → `25-UI-SPEC.md:109-124`
- `credential_select` empty → `25-UI-SPEC.md:128-138`
- `weka_storage_select` populated (incl. inline `<script>`) → `25-UI-SPEC.md:150-188`
- `weka_storage_select` empty → `25-UI-SPEC.md:192-200`

**Critical constraints (from UI-SPEC §Verifiable Acceptance Criteria):**
1. Form-control class string verbatim: `w-full px-3 py-2 rounded-md bg-gray-800/70 border border-white/10 focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)] text-sm`
2. Empty-state copy verbatim: `No {type with hyphens replaced by spaces} credential configured.` + `<a href="/settings#credentials" class="text-[var(--weka-purple)] underline hover:text-white">Add one in Settings.</a>`
3. Every `<option>` in `weka_storage_select` populated branch carries `data-endpoint="{{ cred.endpoint or '' }}"`.
4. `<select>` in `weka_storage_select` populated branch carries `onchange="warpSyncEndpoint(this)"` declared inline.
5. Inline `<script>` defines `function warpSyncEndpoint(selectEl)` and writes `opt.dataset.endpoint || ''` to the endpoint `<input>`.
6. No `<style>` block, no new CSS custom property, no extra `<script>` tag beyond `warpSyncEndpoint`.
7. Empty-state link target is exactly `/settings#credentials` — not `/settings`, not `/settings/credentials`.

**Credential item shape** (consumed inside macros) — sourced from `_build_credential_response_item` at `main.py:744-783`:

```python
{
    "name": "...",            # used in <option value="...">
    "displayName": "...",     # used as <option> visible text
    "type": "nvidia-ngc" | "huggingface" | "weka-storage",
    "ready": True | False,
    "endpoint": "https://..." | None,  # only set for weka-storage
    "lastSyncTime": "...",
    # nvidia-ngc only: "dockerSecretReady": bool
    # if not ready: "error": "..."
}
```

The macros use only `cred.name`, `cred.displayName`, and (weka-storage) `cred.endpoint`.

---

### 4. `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html` — Configure card update

**Analog (exact):** Current Configure card at `blueprint_neuralmesh-aidp.html:308-322`.

**Current block** (lines 308-322):

```html
<div class="card rounded-lg p-6">
  <h2 class="text-lg font-semibold mb-3">Configure</h2>
  <form id="deploy-form" class="space-y-4">
    {# Namespace is resolved from the Home page's selected namespace (localStorage) via syncNamespaceField() on DOMContentLoaded. Kept as a hidden input so the submit handler keeps working without a visible field. #}
    <input type="hidden" id="namespace" name="namespace" value="default" />

    <button type="submit" class="btn-purple w-full px-4 py-2 rounded-md text-sm font-medium">Deploy Blueprint</button>
    <div id="deploy-result" class="text-xs mt-2"></div>
  </form>

  <div id="progress" class="mt-4 hidden">
    <h3 class="text-sm font-semibold mb-2">Installation Progress</h3>
    <ul id="progress-list" class="space-y-1 text-sm"></ul>
  </div>
</div>
```

**Target shape (per D-09, UI-SPEC §Reference Example):**

1. Add `{% from '_credential_macros.html' import credential_select, weka_storage_select %}` near the top of the file (top of body or right before the `<div class="card rounded-lg p-6">` containing the Configure form — Jinja2 macro imports are scoped to the template file, so placement above first use is sufficient).
2. Inside the `<form id="deploy-form" class="space-y-4">`, between the hidden `namespace` input (line 312) and the `<button type="submit">` (line 314), insert:

```jinja
{{ weka_storage_select(credential_field="weka_credential", endpoint_field="weka_endpoint", label="WEKA NeuralMesh") }}
{{ credential_select(type="nvidia-ngc", field_name="ngc_credential", label="NVIDIA NGC API Key") }}
```

**Critical constraints:**
- Do NOT modify the submit handler JS (`form.addEventListener('submit', ...)` at lines 342-380). Per D-10, the new credential field values are inert in the backend — Phase 25 is SDK delivery, not deploy-stream wiring.
- Do NOT modify the `<input type="hidden" id="namespace">` — it remains the namespace source.
- Do NOT modify the `<div id="deploy-result">` or `<div id="progress">` containers — host form owns submit-time error/progress UI (UI-SPEC §Form States).
- Do NOT touch other blueprint templates (`blueprint_nvidia-vss.html`, `blueprint_openfold.html`, `blueprint_glocomp-aurora.html`, `blueprint_tokenvisor-enterprise.html`) — per D-09 they are explicitly out of scope for this phase.

---

### 5. `app-store-gui/tests/test_credential_macros.py` — NEW pytest module

**Analog (exact):** `app-store-gui/tests/test_credentials_api.py` — uses `monkeypatch` for `client.CustomObjectsApi` and `load_kube_config`, calls async route handlers via `asyncio.run`, no FastAPI `TestClient`.

**Stub injection pattern** (`test_credentials_api.py:118-126`):

```python
def _patch_list_credentials(monkeypatch, items: list) -> None:
    """Helper: patch client and load_kube_config for list_credentials tests."""
    class CoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            return {"items": items}

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: SimpleNamespace())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
```

**Async-handler invocation pattern** (`test_credentials_api.py:134`):

```python
response = asyncio.run(main.list_credentials(namespace="default", type=None))
body = json.loads(response.body)
```

**CR fixture factories** (already in `test_credentials_api.py:19-81`) — re-use, do NOT redefine:
- `make_warpcred_cr_nvidia_ready(name, ns)`
- `make_warpcred_cr_nvidia_not_ready(name, ns)`
- `make_warpcred_cr_weka_ready(name, ns, endpoint)`

The new test module can `from webapp.main import _get_credentials_by_type` and import or copy these factories. Conftest at `app-store-gui/tests/conftest.py` is pytest-discoverable but currently has no credential-specific fixtures — the factories live in the sibling test file. Two safe options:
1. Import the factory functions: `from .test_credentials_api import make_warpcred_cr_nvidia_ready, make_warpcred_cr_weka_ready` (relative import; matches established no-conftest-sharing pattern).
2. Re-define minimal CR dicts inline in the new test file (terser but duplicative).

**Recommended test cases:**

1. `test_get_credentials_by_type_groups_by_type` — stub list returns one nvidia + one weka CR; assert returned dict has both populated correctly.
2. `test_get_credentials_by_type_returns_empty_lists_on_api_exception` — stub `list_namespaced_custom_object` raises `ApiException`; assert dict is `{"nvidia-ngc": [], "huggingface": [], "weka-storage": []}`.
3. `test_get_credentials_by_type_returns_empty_lists_on_connection_error` — same, with `ConnectionError`.
4. `test_get_credentials_by_type_unknown_type_dropped` — CR with `spec.type = "unknown"` not included in returned dict (matches the inline filter at `main.py:548`).
5. `test_blueprint_detail_injects_credentials_by_type_into_context` — patch `_get_credentials_by_type` to return a sentinel dict; patch `get_auth_status`, `get_cluster_status`, file IO as needed; call `asyncio.run(main.blueprint_detail(request_stub, name="neuralmesh-aidp"))`; assert sentinel is present in the rendered `TemplateResponse.context` (FastAPI exposes the context dict on `TemplateResponse`).
6. `test_blueprint_detail_falls_back_to_default_namespace` — patch `get_auth_status` to return `{}`; assert `_get_credentials_by_type` was called with `"default"`.

**Pytest invocation (per `CLAUDE.md`):**

```bash
PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest app-store-gui/tests/test_credential_macros.py -v
```

**Critical constraints:**
- No `fastapi.testclient.TestClient` — established pattern is direct async-handler invocation via `asyncio.run`.
- `os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")` MUST be set before `import webapp.main as main` (see `test_credentials_api.py:10-12`) — otherwise `main.py` module-level code fails.
- For test case 5, the `request` argument to `blueprint_detail` can be a `SimpleNamespace`-style stub; FastAPI's `TemplateResponse` will accept any object as `request` and stash it on the resulting response.

---

## Shared Patterns

### Pattern A — Sync-call wrapping inside async handlers
**Source:** `main.py:506-507`, `main.py:524-525`, `main.py:537-538`
**Apply to:** `blueprint_detail` (new `get_auth_status` call), `_get_credentials_by_type` (the `_list()` invocation)

```python
auth = await asyncio.to_thread(get_auth_status)
status = await asyncio.to_thread(get_cluster_status)
# ...
resp = await asyncio.to_thread(_list)
```

Never call synchronous K8s SDK or `get_auth_status`/`get_cluster_status` directly inside an `async def` route — always wrap in `asyncio.to_thread`.

### Pattern B — Namespace resolution
**Source:** `main.py:527-528` (`settings_page`)
**Apply to:** `blueprint_detail` (new code)

```python
detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
ns = detected_ns or "default"
```

The double-or guard against missing keys and the `isinstance(auth, dict)` defense are part of the established pattern — copy verbatim.

### Pattern C — Empty-state hint copy + link styling
**Source:** `settings.html:136-140`
**Apply to:** both macros' empty-state branches in `_credential_macros.html`

```html
<p class="muted text-sm">
  No {type-with-spaces} credential configured.
  <a href="/settings#credentials" class="text-[var(--weka-purple)] underline hover:text-white">Add one in Settings.</a>
</p>
```

Note: the macro uses `text-sm muted` (UI-SPEC) vs. analog `muted text-sm` — both compile identically; UI-SPEC ordering is authoritative.

### Pattern D — Form control Tailwind class string
**Source:** `settings.html:686-707` (and `settings.html:688` specifically)
**Apply to:** every `<input>` and `<select>` rendered by either macro

```
w-full px-3 py-2 rounded-md bg-gray-800/70 border border-white/10 focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)] text-sm
```

UI-SPEC Acceptance Criterion 1 requires this class string verbatim.

### Pattern E — Stub-injection test pattern (no TestClient)
**Source:** `test_credentials_api.py:118-126`, `test_credentials_api.py:134`
**Apply to:** new `test_credential_macros.py`

`monkeypatch.setattr(main.client, "CustomObjectsApi", ...)` + `monkeypatch.setattr(main, "load_kube_config", lambda: None)` + `asyncio.run(main.<handler>(...))` is the project's canonical pattern for testing async K8s-touching route handlers.

---

## No Analog Found

None. Every file has a strong analog within the codebase (either exact or role-match). The `_credential_macros.html` file has no prior macro-file analog but inherits its HTML/CSS patterns directly from `settings.html` and its Jinja2 import mechanism is project-default Jinja2 (no custom Environment configuration in `main.py:184`).

---

## Metadata

**Analog search scope:**
- `/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py` (route handlers, helpers)
- `/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/templates/*.html` (existing templates for include/import precedent, form patterns)
- `/Users/christopherjenkins/git/wekaappstore/app-store-gui/tests/test_credentials_api.py` (test patterns)
- `/Users/christopherjenkins/git/wekaappstore/app-store-gui/tests/planning/test_apply_gateway.py` (alternate stub-injection pattern)
- `/Users/christopherjenkins/git/wekaappstore/app-store-gui/tests/conftest.py` (fixture availability check)

**Files scanned:** 6 source files (main.py, settings.html, blueprint_neuralmesh-aidp.html, blueprint_openfold.html, test_credentials_api.py, test_apply_gateway.py)

**Pattern extraction date:** 2026-06-12
