# Phase 25: Blueprint Credential Selector SDK - Context

**Gathered:** 2026-06-12 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the blueprint Jinja2 macro SDK (SDK-01..SDK-05) in `app-store-gui/webapp/templates/_credential_macros.html` and `app-store-gui/webapp/main.py`.

Deliverables:
- `_credential_macros.html`: Jinja2 macros `credential_select` and `weka_storage_select`
- Shared module-level helper `_get_credentials_by_type(ns)` extracted from `settings_page`
- Context injection: all `blueprint_detail` (and any other blueprint install) route handlers inject `credentials_by_type` dict
- The macros are an SDK for template authors — existing blueprint templates do not need macro calls added in this phase; the infrastructure (helper + injection) enables it

No new sub-packages, no new modules. All changes in `main.py` and `templates/`.
</domain>

<decisions>
## Implementation Decisions

### Shared Helper Extraction
- **D-01:** Extract the `_fetch_credentials` + `credentials_by_type` build logic from `settings_page` (currently inlined at `main.py:530-549`) into a module-level async helper: `async def _get_credentials_by_type(ns: str) -> dict`. Both `settings_page` and `blueprint_detail` call this helper — no duplication.
- **D-02:** Helper signature: `async def _get_credentials_by_type(ns: str) -> dict` — takes namespace, returns `{"nvidia-ngc": [...], "huggingface": [...], "weka-storage": [...]}`. Falls back to empty dict on `ApiException | ConnectionError | TimeoutError` (same exception set as existing inline block). Calls `load_kube_config()` internally.

### Namespace Resolution for Blueprint Route
- **D-03:** `blueprint_detail` route adds `await asyncio.to_thread(get_auth_status)` to resolve namespace — same pattern as `settings_page:524-528`. Falls back to `"default"`. Passes resolved namespace to `_get_credentials_by_type(ns)`. This ensures credentials from the correct namespace are shown.
- **D-04:** `credentials_by_type` is injected into the `blueprint_detail` template context alongside existing keys (`request`, `name`, `yaml_path`, `status`, `requirements`, `meets`, `logo_b64`, etc.).

### Macro File and Jinja2 Import Mechanism
- **D-05:** `_credential_macros.html` uses standard Jinja2 macro syntax. Templates that use the SDK import with `{% from '_credential_macros.html' import credential_select %}` or `{% from '_credential_macros.html' import weka_storage_select %}` — not `{% include %}`. This is the only correct mechanism for callable macros with arguments.
- **D-06:** `credential_select(type, field_name, label=None, required=True)` — renders a `<select name="{{ field_name }}">` populated from `credentials_by_type[type]`; each `<option value="{{ cred.name }}">{{ cred.displayName }}</option>`; when `credentials_by_type[type]` is empty or not populated, renders a hint `<p>` with an `<a href="/settings#credentials">` link instead of an empty select.
- **D-07:** `weka_storage_select(credential_field, endpoint_field, label)` — renders a credential `<select>` where each `<option>` carries `data-endpoint="{{ cred.endpoint }}"`. Renders an endpoint `<input type="url" name="{{ endpoint_field }}">` pre-populated from the first selected option's endpoint. Includes an inline `<script>` defining `warpSyncEndpoint(selectEl)` that updates the endpoint input when selection changes, and attaches it to the select's `onchange`.

### Scope of Template Macro Usage
- **D-08:** SDK-04 requires context injection in **all** blueprint install route handlers — `blueprint_detail` is the only such route (handles all `/blueprint/{name}` pages). After this phase, all blueprint pages have `credentials_by_type` available in their template context.
- **D-09:** Whether existing blueprint templates (`blueprint_nvidia-vss.html`, `blueprint_neuralmesh-aidp.html`, `blueprint_openfold.html`, etc.) call the macros is deferred — this phase delivers the SDK (macro library + context injection). The template `{% from ... import %}` calls are added by whoever next modifies a specific blueprint's install form.

### Claude's Discretion
- Exact placement of `_get_credentials_by_type` in `main.py` — near the existing `_fetch_credentials` usage site (above `settings_page`) or near other module-level helpers
- Whether `wrapSyncEndpoint` is an inline `<script>` in the macro or uses a `<script>` tag in the macro's output block — inline within the rendered output is fine given no build step
- HTML styling for the macro outputs — match existing `blueprint.html` and `settings.html` Tailwind/inline style patterns (dark theme, `muted` text class)
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §SDK-01 through SDK-05 — all deliverables for this phase

### Existing Patterns to Match
- `app-store-gui/webapp/main.py` lines 520-562 — `settings_page` route: `_fetch_credentials` inline helper + `credentials_by_type` dict build + context injection pattern to extract into `_get_credentials_by_type`
- `app-store-gui/webapp/main.py` lines 1257-1327 — `blueprint_detail` route: where to add `get_auth_status` call + `_get_credentials_by_type` call + context key
- `app-store-gui/webapp/main.py` lines 744-785 — `_build_credential_response_item()`: the credential shape injected into `credentials_by_type` lists (`name`, `displayName`, `type`, `ready`, `endpoint`, `lastSyncTime`)
- `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html` lines 212 — `{% include %}` usage showing templates understand Jinja2 directives (but `{% from ... import %}` is needed for macros)
- `app-store-gui/webapp/templates/settings.html` — existing dark-theme Tailwind HTML style used in credential forms; match visual language in macro outputs

### CRD Schema (locked)
- `weka-app-store-operator-chart/templates/crd.yaml` — `WarpCredential` spec and status fields (credential shape source of truth)

### Prior Phase Context
- `.planning/phases/23-backend-credentials-api-and-weka-overview-proxy/23-CONTEXT.md` — `GET /api/credentials?type=<t>` response shape; `credentials_by_type` dict structure; `ready: true` filtering

### Test Patterns
- `app-store-gui/tests/planning/test_apply_gateway.py` — stub injection pattern for GUI route tests
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `settings_page` inline `_fetch_credentials` async function (`main.py:530-549`): exact block to extract into `_get_credentials_by_type` — reads `CustomObjectsApi().list_namespaced_custom_object(group="warp.io", version="v1alpha1", plural="warpcredentials", namespace=ns)`, builds credential items via `_build_credential_response_item`, catches `ApiException|ConnectionError|TimeoutError`
- `_build_credential_response_item(cr)` at `main.py:744-785`: returns dict with `name`, `displayName`, `type`, `ready`, `endpoint`, `lastSyncTime`, `dockerSecretReady` (ngc only), `error` (if error state)
- `load_kube_config()` at `main.py:223`: always called before K8s API use
- `get_auth_status()` at existing location: returns dict with `details.namespace` key for namespace resolution

### Established Patterns
- `asyncio.to_thread` wrapper for sync K8s calls inside async route handlers (`main.py:495`)
- Fallback empty dict `{"nvidia-ngc": [], "huggingface": [], "weka-storage": []}` on K8s failure
- All templates in `app-store-gui/webapp/templates/` loaded by `Jinja2Templates(directory=TEMPLATES_DIR)` at `main.py:184`

### Integration Points
- `blueprint_detail` route (`main.py:1257`) is the entry point for all blueprint install pages — single route, template selection via `f"blueprint_{name}.html"` or `blueprint.html` fallback
- Templates that will eventually call macros: `blueprint_nvidia-vss.html` (NGC cred), `blueprint_neuralmesh-aidp.html` (HF cred), `blueprint_openfold.html` (WEKA storage cred)
- `settings_page` already injects `credentials_by_type` — the extracted helper must maintain identical behavior for that route
</code_context>

<specifics>
## Specific Ideas

- User's intent: the macro SDK allows blueprint template authors to add credential selection to any blueprint install page by adding `{% from '_credential_macros.html' import credential_select %}` and calling `{{ credential_select("nvidia-ngc", "ngc_credential") }}` in the form. Phase 25 builds the plumbing; template authors wire it up per blueprint.
- `warpSyncEndpoint` JavaScript: reads `selectedOption.dataset.endpoint` and writes to the endpoint input. Should be an inline script in the macro's HTML output (no build step in this repo).
- Fallback hint for `credential_select` when no credentials: match the "No WEKA credential configured" hint pattern already in `settings.html` WEKA Overview panel — small muted text with a link.
</specifics>

<deferred>
## Deferred Ideas

- Adding `{% from '_credential_macros.html' import ... %}` calls to existing blueprint templates (`blueprint_nvidia-vss.html`, `blueprint_neuralmesh-aidp.html`, `blueprint_openfold.html`) — this is future template authoring work, not a Phase 25 deliverable
- `credentials_by_type` injection for routes other than `blueprint_detail` — no other blueprint install routes exist currently

None — analysis stayed within phase scope.
</deferred>

---

*Phase: 25-blueprint-credential-selector-sdk*
*Context gathered: 2026-06-12*
