# Phase 24: Settings GUI Overhaul - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 2 (1 template, 1 route handler in a shared file)
**Analogs found:** 2 / 2 (both in-file — strongest possible match quality)
**Scope note:** UI-SPEC §Performance Contract has decided to inline all new CSS/JS into `settings.html`. No separate static asset file is created. Stack constraint: vanilla HTML + Tailwind CDN + vanilla JS, no build step.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `app-store-gui/webapp/templates/settings.html` (HEAVY edit) | jinja2-template | server-rendered + client fetch-render | self — existing Cluster Status / Blueprint Uninstall sections (same file) | exact (same file, same chrome) |
| `app-store-gui/webapp/main.py` (MODIFY `/settings` route only) | fastapi-route-handler | request-response (Jinja2 context inject) | `app-store-gui/webapp/main.py` `index()` at line 503 (sister route) | exact (sister route, same `templates.TemplateResponse` call) |

No new files are created. No static asset files split out (UI-SPEC §Performance Contract).

---

## Pattern Assignments

### `app-store-gui/webapp/templates/settings.html` (jinja2-template, server-rendered + client fetch-render)

**Analog:** `app-store-gui/webapp/templates/settings.html` itself (the existing sections of the file are the analog for the new sections).

#### Pattern 1: `.card` wrapper for every top-level section

**Source:** `settings.html:53`, `settings.html:98`, `settings.html:189`, `settings.html:222` (used 4 times in the file already)
**Apply to:** Credential Management section, WEKA Storage Overview section, and each per-type sub-section (card-in-card).

```html
<section class="card rounded-lg p-5">
  <div class="flex items-center justify-between mb-4">
    <div>
      <h3 class="font-semibold">{Section Title}</h3>
      <p class="muted text-sm">{Helper copy}</p>
    </div>
    <!-- right-side controls (button, select) go here -->
  </div>
  <!-- section body -->
</section>
```

CSS for `.card` is defined once at `settings.html:24`:
```css
.card { background: rgba(31, 41, 55, 0.6); border: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(6px); }
```

Do NOT redefine — reuse via class.

---

#### Pattern 2: `text-2xl font-semibold` numeric counter (Cluster Status pattern)

**Source:** `settings.html:106-117` (CPU Worker Nodes card; identical pattern used 3 more times for GPU, k8s version, storage class)
**Apply to:** Capacity row cards (Total / Used / Available) per UI-SPEC §Component 5.

```html
<div class="card rounded-lg p-5">
  <div class="muted text-xs">CPU Worker Nodes</div>
  <div class="text-2xl font-semibold">{{ status.cpu_nodes if status.cpu_nodes is not none else '-' }}</div>
  <div class="text-xs muted mt-1">
    CPU cores:
    {% if status.cpu_cores_used is not none and status.cpu_cores_free is not none and status.cpu_cores_total is not none %}
      {{ status.cpu_cores_used }}/{{ status.cpu_cores_free }} ({{ status.cpu_cores_total }})
    {% else %}
      -
    {% endif %}
  </div>
</div>
```

Carry over EXACT classes: `card rounded-lg p-5` outer, `muted text-xs` sub-label above, `text-2xl font-semibold` numeric, `text-xs muted mt-1` sub-label below. Sentinel-dash on missing data (`'-'`).

---

#### Pattern 3: `.status-dot` traffic light (analog for credential state dots)

**Source:** `settings.html:28` (CSS rule) + `settings.html:60-62` (live use for Kubernetes Auth Status)
**Apply to:** Every credential row's leading status indicator (UI-SPEC §Component 3, GUI-06 / GUI-07 / GUI-08).

CSS (already defined at line 28 — DO NOT redefine):
```css
.status-dot { width: 10px; height: 10px; border-radius: 9999px; display: inline-block; margin-left: 8px; }
```

HTML usage (line 60-62):
```html
{% set auth_class = 'bg-green-500' if auth and auth.authenticated else 'bg-red-500' %}
<span id="auth-status-text" class="mr-2">{% if auth and auth.authenticated %}Connected{% else %}Not connected{% endif %}</span>
<span id="auth-status-dot" class="status-dot {{ auth_class }}" title="{{ auth.message if auth else '' }}"></span>
```

Three-state extension for Phase 24 (per UI-SPEC State Colour Contracts table, lines 113-116):
- Green: `<span class="status-dot bg-green-500" aria-hidden="true"></span>`
- Amber (pulsing): `<span class="status-dot bg-amber-500 animate-pulse" aria-hidden="true"></span>`
- Red: `<span class="status-dot bg-red-500" aria-hidden="true"></span>`

UI-SPEC line 50 anchors `w-2.5 h-2.5` (10 px) as the only non-multiple-of-4 dimension on the page — the `.status-dot` rule already enforces this; do NOT add Tailwind width/height to the `<span>`.

---

#### Pattern 4: Browser-native `confirm()` destructive confirmation

**Source:** `settings.html:336-341` (Blueprint Delete handler — exact analog called out in UI-SPEC line 295)
**Apply to:** Credential `[Delete]` button (GUI-09).

```javascript
tbody.querySelectorAll('button[data-ns][data-name]').forEach(btn => {
  btn.addEventListener('click', async (e) => {
    const ns = btn.getAttribute('data-ns');
    const name = btn.getAttribute('data-name');
    if (!ns || !name) return;
    if (!confirm(`Delete blueprint ${name} in namespace ${ns}? This cannot be undone.`)) return;
    const out = document.getElementById('bp-result');
    out.textContent = `Deleting ${name}...`;
    try {
      const data = await del(`/api/blueprints/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`);
      if (data && data.ok) {
        out.textContent = `Deleted ${name}. Refreshing...`;
        ...
```

For credentials, the copy is locked by UI-SPEC line 295:
```javascript
if (!confirm(`Delete credential "${displayName}"? Derived secrets in the cluster will remain — you can delete them manually with kubectl if needed.`)) return;
const data = await del(`/api/credentials/${encodeURIComponent(name)}`);
```

Note: `data-name`/`data-ns` button-attribute pattern (line 331) is the right way to plumb identifiers through the button — copy verbatim, switch attributes to `data-cred` and `data-ns`.

---

#### Pattern 5: Table render — sortable, escaped, server-class headers

**Source:** `settings.html:204-216` (HTML table shell) + `settings.html:306-334` (JS renderer)
**Apply to:** Filesystem table in WEKA Storage Overview (GUI-13).

HTML shell (lines 204-216):
```html
<div id="bp-list" class="overflow-x-auto">
  <div id="bp-empty" class="muted text-sm">Loading...</div>
  <table id="bp-table" class="min-w-full hidden">
    <thead>
      <tr class="text-left text-xs uppercase text-white/70">
        <th class="py-2 pr-4">Namespace</th>
        <th class="py-2 pr-4">Name</th>
        <th class="py-2 pr-4">Created</th>
        <th class="py-2 pr-4"></th>
      </tr>
    </thead>
    <tbody id="bp-tbody" class="text-sm"></tbody>
  </table>
</div>
```

JS render with HTML escape and sort (lines 317-334):
```javascript
tbody.innerHTML = '';
const sorted = items.slice().sort((a,b)=>{
  const nsA = (a.namespace||'').localeCompare(b.namespace||'');
  if (nsA !== 0) return nsA;
  return (a.name||'').localeCompare(b.name||'');
});
const esc = s => (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
for (const it of sorted) {
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="py-2 pr-4 font-mono">${esc(it.namespace)||'-'}</td>
    <td class="py-2 pr-4 font-mono">${esc(it.name)||'-'}</td>
    ...`;
  tbody.appendChild(tr);
}
```

Carry over EXACTLY:
- `min-w-full` table class
- `text-left text-xs uppercase text-white/70` header row class
- `py-2 pr-4` cell padding
- The `esc()` HTML-escape helper (mandatory per recent WR-04 XSS fix at commit c7ca314 — apply to `name` field even though backend already filters UUIDs)
- Sort direction for Phase 24: descending by `usedPercent` (GUI-13). Replace `localeCompare` with `(b.usedPercent ?? 0) - (a.usedPercent ?? 0)`.

For filesystem table: change `font-mono` to plain `text-sm` for the human `name` column (no UUIDs — see GUI-13). Add a fourth utilisation column with the inline progress-bar pattern from Pattern 9 below.

---

#### Pattern 6: Polling `setInterval` + cleanup (analog for amber-row poll)

**Source:** `settings.html:301-303` (auth-status poll)
**Apply to:** Per-row amber-state credential polling (GUI-07, UI-SPEC §Component 6).

```javascript
// Initial loads
refreshAuthStatus();
setInterval(refreshAuthStatus, 10000);
```

Existing pattern is page-global. Phase 24 needs a **per-row** variant:

```javascript
// UI-SPEC §Component 6: per-row polling map
const pollIntervals = new Map(); // name → intervalId
const pollStartedAt = new Map(); // name → ms epoch
const POLL_MS = 2000;
const POLL_TIMEOUT_MS = 30000;

function startCredentialPoll(name) {
  if (pollIntervals.has(name)) return;
  pollStartedAt.set(name, Date.now());
  const id = setInterval(() => pollCredentialOnce(name), POLL_MS);
  pollIntervals.set(name, id);
}

function stopCredentialPoll(name) {
  const id = pollIntervals.get(name);
  if (id) clearInterval(id);
  pollIntervals.delete(name);
  pollStartedAt.delete(name);
}

// UI-SPEC line 267: Page Visibility API pause/resume
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    pollIntervals.forEach(clearInterval);
  } else {
    Array.from(pollIntervals.keys()).forEach(name => {
      const id = setInterval(() => pollCredentialOnce(name), POLL_MS);
      pollIntervals.set(name, id);
    });
  }
});

// UI-SPEC line 346: clear on beforeunload
window.addEventListener('beforeunload', () => {
  pollIntervals.forEach(clearInterval);
});
```

The auth-status poll handler at lines 260-300 is the analog for the inner `pollCredentialOnce(name)` body — same `try { … } catch (e) { … }` shape, same DOM-update pattern with `setText` style helpers.

---

#### Pattern 7: Fetch helpers (`get`, `post`, `del`)

**Source:** `settings.html:246-257`
**Apply to:** All `/api/credentials*` and `/api/weka/overview` calls.

```javascript
async function post(url, data) {
  const res = await fetch(url, { method: 'POST', body: data });
  return res.json();
}
async function get(url) {
  const res = await fetch(url);
  return res.json();
}
async function del(url) {
  const res = await fetch(url, { method: 'DELETE' });
  return res.json();
}
```

These helpers are already in scope inside the existing `<script>` block — reuse. For POST credential creation use `new FormData(form)` matching the Phase 23 `Form(...)` server-side decision (D-13 / line 311 of UI-SPEC).

---

#### Pattern 8: Async load-and-render with empty + error states

**Source:** `settings.html:360-376` (`loadBlueprints` function — exact analog called out in UI-SPEC for the WEKA Overview fetch)
**Apply to:** WEKA Overview Refresh button handler + initial overview fetch (GUI-15).

```javascript
async function loadBlueprints() {
  const scopeSel = document.getElementById('bp-scope');
  const scope = scopeSel ? scopeSel.value : 'all';
  const namespace = (scope === 'current') ? (localStorage.getItem('selectedNamespace') || 'default') : 'all';
  const emptyEl = document.getElementById('bp-empty');
  if (emptyEl) emptyEl.textContent = 'Loading...';
  try {
    const data = await get(`/api/blueprints?namespace=${encodeURIComponent(namespace)}`);
    if (data && data.ok) {
      renderBlueprints(data.items || []);
    } else {
      if (emptyEl) emptyEl.textContent = data && data.error ? data.error : 'Failed to load blueprints';
    }
  } catch (e) {
    if (emptyEl) emptyEl.textContent = 'Failed to load blueprints: ' + e;
  }
}
```

State-machine mapping (UI-SPEC §Component 5):
- "Loading..." text → `loading` state (spinner per UI-SPEC line 186)
- `data.ok === false` → `error` state (red banner per UI-SPEC line 187)
- `renderBlueprints(items)` → `success` state (full panel)
- Initial `(empty) → loading → success/error` driven by `hidden` attribute toggles on three sibling `<div>` containers (UI-SPEC line 322).

---

#### Pattern 9: Form submit + result feedback

**Source:** `settings.html:336-356` (Blueprint delete handler with result display)
**Apply to:** Inline Add credential form submit (GUI-04, GUI-05).

```javascript
btn.addEventListener('click', async (e) => {
  ...
  const out = document.getElementById('bp-result');
  out.textContent = `Deleting ${name}...`;
  try {
    const data = await del(`/api/blueprints/...`);
    if (data && data.ok) {
      out.textContent = `Deleted ${name}. Refreshing...`;
      await loadBlueprints();
      out.textContent = 'Done.';
    } else {
      out.textContent = 'Error: ' + (data && data.error ? data.error : 'Unknown error');
    }
  } catch (err) {
    out.textContent = 'Request failed: ' + err;
  }
});
```

Phase 24 form save (UI-SPEC §Component 4): same try/catch shape; before-request set Save button to `disabled` with label `'Saving…'` (UI-SPEC line 298); on 200 with the returned credential `name`, immediately call `startCredentialPoll(data.item.name)` and re-render the row in amber state.

---

#### Pattern 10: Tailwind class palette for buttons and inputs (locked palette)

**Source (primary CTA `.btn-purple`):** `settings.html:25-26`
```css
.btn-purple { background: var(--weka-purple); color: white; }
.btn-purple:hover { background: var(--weka-purple-dark); }
```

**Source (secondary outline button):** `settings.html:200`
```html
<button id="bp-refresh" class="px-3 py-1.5 rounded-md text-sm font-medium border border-white/20 hover:bg-white/10">Refresh</button>
```

**Source (red-outline destructive button):** `settings.html:331`
```html
<button class="px-3 py-1 rounded-md text-sm border border-red-400/40 text-red-300 hover:bg-red-500/10" data-ns="..." data-name="...">Delete</button>
```

**Source (form select control):** `settings.html:196`
```html
<select id="bp-scope" class="px-2 py-1 rounded-md bg-gray-800/70 border border-white/10 text-sm">
```

**Apply to:**
- `[+ Add]` button per sub-section → reuse `.btn-purple` plus `px-3 py-1.5 rounded-md text-sm font-medium`
- `Save` button (primary inside form) → `.btn-purple` + `px-3 py-1.5 rounded-md text-sm font-medium`, when disabled add `opacity-50 cursor-not-allowed` (UI-SPEC line 312)
- `Cancel` button → outline secondary pattern from line 200
- `↺ Refresh` button → outline-purple (UI-SPEC line 104): `px-3 py-1.5 rounded-md text-sm font-medium border border-[var(--weka-purple)]/60 text-[var(--weka-purple)] hover:bg-[var(--weka-purple)]/10`
- `[Delete]` per credential row → copy line 331 verbatim
- Credential `<select>` (when multiple WEKA credentials) → copy line 196 + add focus ring per UI-SPEC line 106: `focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)]`
- Form `<input>` controls → UI-SPEC line 175 spec: `w-full px-3 py-2 rounded-md bg-gray-800/70 border border-white/10 focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)] text-sm`

---

#### Pattern 11: Jinja2 conditional render with sentinel fallback

**Source:** `settings.html:108`, `:120`, `:144`, `:148` (`{{ value if value else '-' }}`) and `:169-184` (multi-branch with empty / error states)
**Apply to:** WEKA Overview server-side initial render — `credentials_by_type` may be empty, `weka_storage_credentials` may be empty.

```jinja
{% if status.app_store_crd_installed is false %}
  <span class="text-red-400">CRD not installed</span>
{% else %}
  {% set crs = status.app_store_crs or [] %}
  {% if crs|length == 0 %}
    <span class="text-white/80">No CRs applied.</span>
  {% else %}
    <ul class="list-disc list-inside">
      {% for n in crs %}
        <li class="text-white/80">{{ n }}</li>
      {% endfor %}
    </ul>
  {% endif %}
{% endif %}
```

Apply to credential lists (GUI-02 empty state `(none stored)`), WEKA Overview hint (GUI-10 — render the hint vs the panel conditionally on `weka_storage_credentials | length`), and to the WEKA Overview success path when filesystems are pre-sorted server-side.

---

### `app-store-gui/webapp/main.py` (fastapi-route-handler, request-response)

**Analog:** `app-store-gui/webapp/main.py` lines 503-518 — the sister `index()` route. Already-present `settings_page()` at lines 521-537 is the actual target — extending its context dict.

#### Pattern 12: `templates.TemplateResponse` context injection

**Source:** `main.py:521-537` (the existing `settings_page()` — direct ancestor)

```python
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    auth = await asyncio.to_thread(get_auth_status)
    status = await asyncio.to_thread(get_cluster_status)
    # Use detected namespace if available, else default
    detected_ns = (auth.get("details", {}) or {}).get("namespace") if isinstance(auth, dict) else None
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "auth": auth,
            "status": status,
            "detected_namespace": detected_ns or "default",
            "logo_b64": LOGO_B64,
        },
    )
```

Phase 24 modification: extend the context dict with the credentials-by-type mapping required by UI-SPEC line 317 and GUI-10:

```python
# After existing asyncio.to_thread(get_auth_status) / get_cluster_status calls,
# fetch credentials in the detected namespace.
ns = detected_ns or "default"

async def _fetch_credentials() -> list[dict]:
    def _list():
        co = client.CustomObjectsApi()
        return co.list_namespaced_custom_object(
            group="warp.io", version="v1alpha1",
            plural="warpcredentials", namespace=ns,
        )
    try:
        load_kube_config()
        resp = await asyncio.to_thread(_list)
        return [_build_credential_response_item(cr) for cr in (resp or {}).get("items", []) or []]
    except ApiException:
        return []
    except Exception:
        return []

cred_items = await _fetch_credentials()

credentials_by_type: dict[str, list[dict]] = {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}
for it in cred_items:
    credentials_by_type.setdefault(it["type"], []).append(it)
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
        # New Phase 24 context
        "credentials_by_type": credentials_by_type,
        "weka_storage_credentials": weka_storage_credentials,
    },
)
```

**Reusable helpers already in this file:**
- `load_kube_config()` at `main.py:233` — call inside the worker
- `_build_credential_response_item(cr)` at `main.py:717` — already the response-shape builder; reuse to keep server-rendered rows and JS-rendered rows identical
- `asyncio.to_thread(sync_callable, ...)` at `main.py:505-506` — wrap any sync K8s call
- `client.CustomObjectsApi()` import is already in scope (used at `main.py:773`)

**Graceful degradation** matches SDK-05 (REQUIREMENTS.md line 66): if Kubernetes is unreachable, return empty lists so the template still renders.

#### Pattern 13: JSON error envelope on K8s failure (do NOT raise into template)

**Source:** `main.py:591-595`, `main.py:799-802` (every existing route)

```python
except ApiException as ae:
    return JSONResponse({"ok": False, "error": f"Kubernetes API error: {ae.status} {ae.reason}"}, status_code=500)
except Exception as e:
    return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

For the `/settings` route specifically, exceptions during `_fetch_credentials()` must NOT bubble — degrade to empty list (per SDK-05). The page still renders with "(none stored)" empty states.

---

## Shared Patterns

### Authentication / Authorization
**Source:** `app-store-gui/webapp/main.py` — `ClusterInitMiddleware` (called out in 23-02-SUMMARY threat flags)
**Apply to:** Nothing new in Phase 24. All `/api/credentials*` and `/api/weka/overview` routes already gated by the existing middleware. The `/settings` page itself is already gated. No additional auth code added.

### Error Handling (server)
**Source:** Whitelist pattern at `main.py:717-756` (`_build_credential_response_item`)
**Apply to:** Phase 24 reuses this verbatim — no new server-side error handler introduced. Any new server-side logic in `settings_page()` MUST swallow exceptions and pass empty lists (graceful degradation).

### Error Handling (client)
**Source:** `settings.html:291-299` (`refreshAuthStatus` catch block — degrade dot to red, set details to '-')
**Apply to:** Every new fetch in Phase 24. Concrete contract:
- Network failure during credential poll → keep amber, retry next tick, bubble into the 30 s budget (UI-SPEC line 261)
- Network failure on initial WEKA overview load → switch to `error` state banner (UI-SPEC line 187), no stale data shown
- Network failure on credential delete → render `<div class="text-xs text-red-400 mt-2">` near the sub-section list (UI-SPEC line 316)

### Validation
**Source:** `settings.html:60` (Jinja2 expression with sentinel) + browser-native `input[required]` + `endpointInput.checkValidity()` (UI-SPEC line 312)
**Apply to:** All Add forms. The Save button is `disabled` until all required inputs are non-empty AND the URL input (WEKA form only) passes `checkValidity()`. No JS validation library — use the HTML5 Constraint Validation API only.

### HTML escape on `innerHTML`
**Source:** `settings.html:323` (the `esc()` helper) — recent commit c7ca314 (WR-04) added this for XSS prevention.
**Apply to:** Every place that interpolates server-returned strings into `innerHTML` — credential rows, filesystem table rows, error reason strings, backend IP grid cells. Mandatory.

### Logging hygiene (server)
**Source:** API-08 + 23-02-SUMMARY self-check "logger credential-value grep gate → PASSED"
**Apply to:** New code in `settings_page()` must not log raw credential names through `print()` (always corrupts in MCP, never appropriate in GUI either). Use the existing `logger.info(...)` pattern from `main.py:973`.

### Accessibility focus ring
**Source:** UI-SPEC line 328 — `focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)]` already used elsewhere on the page (UI-SPEC line 106 / 175).
**Apply to:** Every new interactive element (`<button>`, `<input>`, `<select>`). Non-negotiable.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | All Phase 24 work is in two files that already have strong same-file analogs; nothing introduced without precedent. |

The combinations below have no existing analog in the codebase but are required by UI-SPEC — planner must use the UI-SPEC contract verbatim for these:

| Concern | Resolution |
|---------|-----------|
| `animate-pulse` on dots | Tailwind utility — no existing usage; UI-SPEC line 156 mandates it for amber state |
| `humanBytes(n)` formatter | No existing JS formatter — UI-SPEC line 220 specifies TiB / GiB / MiB rules |
| Relative-time formatter (`"2m ago"`) | No existing JS formatter — UI-SPEC line 286 specifies exact rules |
| Spinner element | UI-SPEC line 186 specifies an inline Tailwind spinner — no existing analog (auth uses dot only) |
| Card-in-card visual nesting | UI-SPEC line 135 — first time in codebase; `.card` class behaves identically when nested |
| Single-open-form invariant across sibling sub-sections | UI-SPEC line 177 (`closeAllAddForms()` helper) — new logic, no analog |

---

## Metadata

**Analog search scope:**
- `app-store-gui/webapp/templates/settings.html` (full file, 401 lines, read once)
- `app-store-gui/webapp/main.py` (read targeted ranges: 503-538, 545-619, 717-756, 759-803, 805-858, 922-978, 981-1081)
- `app-store-gui/webapp/templates/index.html` (grep only — confirmed no overlapping fetch patterns)

**Files scanned:** 3 (settings.html, main.py, index.html)
**Pattern extraction date:** 2026-06-12
**Stack constraint compliance:** All extracted patterns are vanilla HTML / Tailwind CDN utility classes / vanilla JS — no framework, no build step, no new CDN script tag, no new dependency. Matches UI-SPEC §Registry Safety (line 352).
