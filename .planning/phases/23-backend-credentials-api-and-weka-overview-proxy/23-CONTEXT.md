# Phase 23: Backend Credentials API and WEKA Overview Proxy - Context

**Gathered:** 2026-06-11 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `/api/credentials` CRUD endpoints and `/api/weka/overview` proxy to `app-store-gui/webapp/main.py`. Remove the old `/api/secret/nvidia` and `/api/secret/huggingface` endpoints and their corresponding settings page JavaScript.

New endpoints:
- `GET /api/credentials` — list all WarpCredential CRs with status shape
- `GET /api/credentials?type=<t>` — filtered, ready=true only (for blueprint SDK)
- `POST /api/credentials` — create raw `warp-cred-<slug>` Secret + WarpCredential CR
- `DELETE /api/credentials/<name>` — delete CR and raw Secret; leave derived secrets
- `GET /api/weka/overview?credential=<name>` — proxy to WEKA REST API with 60s cache

All new routes follow the existing single-file pattern in `app-store-gui/webapp/main.py`. No new sub-packages.
</domain>

<decisions>
## Implementation Decisions

### Kubernetes Client for Credentials API

- **D-01:** Use `CustomObjectsApi` (already imported, group `warp.io`, version `v1alpha1`, plural `warpcredentials`) for all WarpCredential CR operations: list, create, delete.
- **D-02:** Use `CoreV1Api` + `client.V1Secret` for `warp-cred-<slug>` Secret create and delete — same pattern as `create_or_update_secret()` at line 535.
- **D-03:** Call `load_kube_config()` at the top of each new route handler, matching all existing patterns. Do NOT add a separate kube client singleton.

### WEKA Overview HTTP Client and Caching Strategy

- **D-04:** Use `urllib.request` (sync) wrapped in `asyncio.to_thread` for all WEKA REST API calls — no new dependency (`httpx` is NOT in `requirements.txt`). Match the `asyncio.to_thread` pattern used at lines 495-496.
- **D-05:** `asyncio.gather` drives the three parallel calls (`fileSystems`, `cluster`, `containers`) after the login exchange — each is an `asyncio.to_thread` coroutine.
- **D-06:** Module-level cache: `_weka_overview_cache: dict[str, dict] = {}` where each entry is `{"ts": float, "data": dict}`. TTL = 60 seconds checked via `time.time() - entry["ts"] < 60`. Match the `_last_ready_cache` dict pattern at lines 1016-1065.
- **D-07:** `?bust=1` query param bypasses the cache unconditionally (skip TTL check, re-fetch, update cache entry).

### WEKA Credential Secret Structure

- **D-08:** `warp-cred-<slug>` Secret stores:
  - `nvidia-ngc` type: key `NGC_API_KEY` only
  - `huggingface` type: key `HF_API_KEY` only
  - `weka-storage` type: keys `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT` (all three in one Secret)
- **D-09:** For weka-storage, `WarpCredential.spec.secretRef.key` = `WEKA_API_TOKEN` (the operator reads this key; username and endpoint are additional keys the GUI reads directly by name).
- **D-10:** `POST /api/credentials` for weka-storage creates the raw Secret with all three keys populated from the request body fields: `username`, `key` (=token), `endpoint`.

### Slug Generation for metadata.name

- **D-11:** `displayName` → slug: lowercase, replace `[^a-z0-9]+` with `-`, strip leading/trailing hyphens, truncate to 52 characters. `metadata.name` = slug (used for both the `WarpCredential` CR and the `warp-cred-<slug>` Secret).
- **D-12:** Collision handling: before creation, list existing WarpCredential CRs via `CustomObjectsApi.list_namespaced_custom_object`; if slug already exists, append `-2`, `-3`, etc. until unique.

### Settings Page JavaScript Removal

- **D-13:** Remove `@app.post("/api/secret/huggingface")` handler (line 558) and `@app.post("/api/secret/nvidia")` handler (line 626) from `main.py`.
- **D-14:** In `app-store-gui/webapp/templates/settings.html`: remove the `hfBtn` click handler block (posts to `/api/secret/huggingface`), the `nvBtn` click handler block (posts to `/api/secret/nvidia`), and the `getSettingsNamespace`/`setSettingsNamespace`/`renderSecrets`/`loadSecrets` JS block that only supports the old secret list panels. Remove the corresponding HTML form sections for the old HuggingFace and NVIDIA NGC credential inputs.

### Test Approach

- **D-15:** Test file: `app-store-gui/tests/test_credentials_api.py`. Follow the `test_apply_gateway.py` stub pattern — no FastAPI `TestClient`, no live K8s; inject `CustomObjectsApiStub` and `CoreV1ApiStub` objects directly into the handler logic via module-level patching with `unittest.mock.patch`.
- **D-16:** Test coverage: GET credentials shape, POST slug generation and collision, DELETE CR + raw Secret without touching derived secrets, GET credentials?type filter (ready=true only), WEKA overview cache behavior (60s TTL, bust=1 bypass), no credential values in any response body.

### Claude's Discretion

- Exact placement of `_weka_overview_cache` module-level dict (near other module-level state, consistent with `_last_ready_cache`)
- Exact error message wording for credential-not-found (match existing `"Kubernetes API error: {ae.status} {ae.reason}"` convention)
- Whether `GET /api/weka/overview` returns `{"ok": false, "error": ...}` on WEKA API auth failure or raises a 502 — match the existing `JSONResponse({"ok": False, "error": ...}, status_code=502)` convention

### External Research Required

- **WEKA REST API field names** — exact response field names for `/api/v2/fileSystems`, `/api/v2/cluster`, `/api/v2/containers` cannot be inferred from this codebase. Researcher must consult WEKA public API docs or the PRD's Swagger UI reference (`https://<cluster>:14000/api/v2/docs`) before writing the response assembly logic.
- **WEKA auth token type** — whether `WEKA_API_TOKEN` is a long-lived static Bearer token (use directly) or a short-lived refresh token (requires `POST /api/v2/login/refresh`). Researcher must confirm from WEKA 4.x docs or PRD.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §API-01 through API-07 — all GUI API requirements for this phase
- `.planning/REQUIREMENTS.md` §API-08 — no raw credential values logged (applies to GUI backend)

### Existing GUI Patterns to Match
- `app-store-gui/webapp/main.py` lines 535-555 — `create_or_update_secret()` pattern for `CoreV1Api` + `V1Secret`
- `app-store-gui/webapp/main.py` lines 573-600 — `list_blueprints()` pattern for `CustomObjectsApi` CR listing
- `app-store-gui/webapp/main.py` lines 603-623 — `delete_blueprint()` pattern for `delete_namespaced_custom_object`
- `app-store-gui/webapp/main.py` lines 1016-1065 — existing `_last_ready_cache` dict TTL pattern to replicate for `_weka_overview_cache`
- `app-store-gui/webapp/main.py` lines 495-496 — `asyncio.to_thread` wrapping sync K8s calls in async route handler
- `app-store-gui/webapp/main.py` lines 558-570 — `/api/secret/huggingface` handler to REMOVE
- `app-store-gui/webapp/main.py` lines 626-638 — `/api/secret/nvidia` handler to REMOVE

### CRD Schema (locked by Phase 21)
- `weka-app-store-operator-chart/templates/crd.yaml` lines 279-382 — WarpCredential spec (`secretRef.name`, `secretRef.key`, `spec.type`, `spec.displayName`, `spec.endpoint`) and status (`conditions`, `derivedSecrets`, `lastSyncTime`, `wekaEndpoint`)

### Settings Page (JS to remove)
- `app-store-gui/webapp/templates/settings.html` — `hfBtn` handler (~line 387-412), `nvBtn` handler (~line 415-440), old JS support block (`getSettingsNamespace` etc. ~lines 322-383)

### PRD (WEKA API field names)
- `.planning/PRD-secret-management-overhaul.md` — WEKA REST API response shape details and Swagger UI reference for field name confirmation

### Prior Phase Context
- `.planning/phases/22-operator-warpcredential-reconciler/22-CONTEXT.md` — locked operator decisions; Phase 23 reads the `status.conditions`, `status.derivedSecrets`, `status.wekaEndpoint` fields the operator populates

### Test Patterns
- `app-store-gui/tests/planning/test_apply_gateway.py` — canonical stub injection pattern for GUI tests
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `create_or_update_secret(name, namespace, string_data)` at `main.py:535` — reusable for `warp-cred-<slug>` creation; only needs `string_data` key set adjusted per credential type
- `load_kube_config()` at `main.py:223` — call at top of each new route handler; handles both in-cluster and local kubeconfig
- `ensure_namespace_exists(namespace)` at `main.py:277` — call before Secret creation
- `asyncio.to_thread` at `main.py:495` — established pattern for running sync K8s/HTTP calls inside async FastAPI handlers
- Module-level cache dict pattern (`_last_ready_cache`) at `main.py:1016-1065` — replicate for `_weka_overview_cache`

### Established Patterns
- All new routes return `JSONResponse({"ok": True, ...})` on success, `JSONResponse({"ok": False, "error": ...}, status_code=5xx)` on error
- `ApiException` caught first, then `Exception` — both converted to `JSONResponse`, never propagated
- Existing imports already available: `re`, `time`, `asyncio`, `base64`, `json`, `urllib.request` (line 1480), `typing.Dict/Any/Optional`
- Single large `app-store-gui/webapp/main.py` — all new routes go here, no new modules

### Integration Points
- Phase 24 (Settings GUI) will call all four `/api/credentials` routes and `/api/weka/overview` — correct response shapes are critical
- Phase 25 (Blueprint SDK) will call `GET /api/credentials?type=<t>` — `ready: true` filtering must be reliable
- Operator Phase 22 populates `status.conditions[KeyReady]`, `status.derivedSecrets`, `status.lastSyncTime`, `status.wekaEndpoint` — `GET /api/credentials` reads these fields directly from WarpCredential CR status
</code_context>

<specifics>
## Specific Ideas

- WEKA overview cache key: credential `metadata.name` (string) — one cache entry per named credential
- `POST /api/credentials` body via FastAPI `Form(...)` fields (matching existing `/api/secret/*` style), not JSON body — keeps consistent with current API surface
- `DELETE /api/credentials/<name>`: first delete the `WarpCredential` CR (so operator stops reconciling), then delete the `warp-cred-<slug>` Secret (raw input); leave `warp-<name>-*` derived secrets untouched
- For the `GET /api/credentials` ready calculation: `ready = True` when the `KeyReady` condition in `status.conditions` has `status == "True"`
- For `nvidia-ngc`, also expose `dockerSecretReady` derived from `DockerSecretReady` condition
- For `weka-storage`, expose `endpoint` from `status.wekaEndpoint` (never from raw Secret)
</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within phase scope.
</deferred>

---

*Phase: 23-backend-credentials-api-and-weka-overview-proxy*
*Context gathered: 2026-06-11*
