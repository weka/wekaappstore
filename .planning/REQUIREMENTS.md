# Requirements: WEKA App Store v6.0 — Secret Management & WEKA Storage Integration

**Defined:** 2026-06-11
**Core Value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.
**Milestone Goal:** Give App Store administrators a first-class credential management system — named, multi-key storage for NGC/HuggingFace/WEKA credentials via a new `WarpCredential` CRD, automatic secret derivation by the operator, a blueprint Jinja2 macro SDK for credential selection, and live WEKA storage visibility on the Settings page.
**Source PRD:** `.planning/PRD-secret-management-overhaul.md`

## v1 Requirements

### CRD

- [x] **CRD-01**: `WarpCredential` CRD (`warp.io/v1alpha1`, kind `WarpCredential`) added to `weka-app-store-operator-chart/templates/crd.yaml`; multi-instance (one CR per named credential, multiple CRs of same type allowed)
- [x] **CRD-02**: CRD schema validates `spec.type` as an enum (`nvidia-ngc`, `huggingface`, `weka-storage`); invalid values rejected at admission
- [x] **CRD-03**: CRD schema requires `spec.displayName` (string), `spec.secretRef.name` (string), and `spec.secretRef.key` (string)
- [x] **CRD-04**: CRD schema accepts optional `spec.endpoint` field (string, URL) used only for `weka-storage` type; silently ignored for other types
- [x] **CRD-05**: CRD defines `status` subresource with `conditions` array (`KeyReady`, `DockerSecretReady` for nvidia-ngc), `derivedSecrets` list, `lastSyncTime`, and optional `wekaEndpoint` string
- [x] **CRD-06**: Helm chart RBAC adds a `Role` + `RoleBinding` granting the operator's service account create/patch/get/delete on Secrets, scoped to the App Store namespace only (not cluster-wide)

### OPS (Operator)

- [x] **OPS-01**: kopf `on.create` / `on.update` handler watches `WarpCredential` resources; raises `kopf.PermanentError` for unrecognised `spec.type`
- [x] **OPS-02**: If `spec.secretRef` Secret does not exist in the App Store namespace, raises `kopf.TemporaryError(delay=30)` (not a crash; retries until Secret appears)
- [x] **OPS-03**: Empty or whitespace-only key value read from the referenced Secret raises `kopf.PermanentError` naming the credential
- [x] **OPS-04**: For `nvidia-ngc`: creates/patches `warp-<name>-apikey` (Opaque, key `NGC_API_KEY`) and `warp-<name>-docker` (type `kubernetes.io/dockerconfigjson` for `nvcr.io`) in the App Store namespace; docker payload: `{"auths":{"nvcr.io":{"username":"$oauthtoken","password":"<key>","auth":"<base64>"}}}`
- [x] **OPS-05**: For `huggingface`: creates/patches `warp-<name>-token` (Opaque, key `HF_API_KEY`) in the App Store namespace
- [x] **OPS-06**: For `weka-storage`: creates/patches `warp-<name>-token` (Opaque, keys `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`) in the App Store namespace; copies `spec.endpoint` to `status.wekaEndpoint`
- [x] **OPS-07**: After each reconcile, updates `status.conditions` (type `KeyReady`, and `DockerSecretReady` for nvidia-ngc), `status.derivedSecrets` (list of created secret names with type), and `status.lastSyncTime`
- [x] **OPS-08**: On delete of a `WarpCredential`, derived secrets (`warp-<name>-*`) are NOT deleted; operator logs a warning; administrator must delete them manually
- [x] **OPS-09**: Derived secret creation is idempotent — if a derived secret is manually deleted, the next reconcile cycle restores it automatically within the kopf retry window

### GUI

- [ ] **GUI-01**: Settings page restructured so the Credential Management section appears first, above Kubernetes Auth Status, Cluster Status, Blueprint Management, and Debug sections
- [ ] **GUI-02**: Credential Management section shows three sub-sections: NVIDIA NGC API Keys, HuggingFace Tokens, WEKA Storage API Tokens — each with a list of stored credentials and a `[+ Add]` button
- [ ] **GUI-03**: `[+ Add]` expands an inline form at the bottom of that type's list; only one add-form is open at a time per credential type
- [ ] **GUI-04**: Add form for NGC and HuggingFace: required `Name` (free-text display name) and required `Key` (password input, `type="password"`)
- [ ] **GUI-05**: Add form for WEKA Storage: required `Name`, required `Username` (WEKA API user account), required `API Token` (`type="password"`), required `Endpoint` (URL input, client-side URL validation before Save is enabled)
- [ ] **GUI-06**: Green state (operator confirmed ready): credential row shows display name + `[●] Ready` + `[Delete]` button — no input fields visible
- [ ] **GUI-07**: Amber state (transitional): row shows `[◐] Verifying...` — polls every 2 seconds, times out after 30 seconds with inline error
- [ ] **GUI-08**: Red state (operator reported error): row shows display name + error reason + `[Delete]` button
- [ ] **GUI-09**: `[Delete]` button sends `DELETE /api/credentials/<name>`; row disappears from the list; derived secrets remain in cluster (not deleted)
- [ ] **GUI-10**: WEKA Storage Overview panel appears below Credential Management when at least one `weka-storage` credential is registered; hidden with a "No WEKA credential configured" hint otherwise
- [ ] **GUI-11**: Overview panel header: credential dropdown (if multiple WEKA credentials — replaced by static label if only one), `[↺ Refresh]` button, "Last updated" timestamp showing actual data age (not page-load time)
- [ ] **GUI-12**: Overview panel capacity row: Total / Used / Available in human-readable TiB/GiB, with a percentage utilisation bar
- [ ] **GUI-13**: Overview panel filesystem table: human-readable `name` only (no UUIDs), columns Total / Used / Utilisation, sorted descending by utilisation; filesystems ≥ 90% utilisation render bar in amber/orange; max 20 rows with "Show all" toggle if more exist
- [ ] **GUI-14**: Overview panel backend node grid: count header, primary management IP per node in a wrapped grid (no hostname resolution, no loopback/link-local addresses)
- [ ] **GUI-15**: Overview panel states: no credential → hint with link; loading → spinner; WEKA API error → error message only, no stale data; success → full panel

### API

- [ ] **API-01**: `GET /api/credentials` returns JSON array of all `WarpCredential` CRs; each entry: `name`, `displayName`, `type`, `ready` (bool), `lastSyncTime`; nginx-ngc adds `dockerSecretReady`; weka-storage adds `endpoint` (from `status.wekaEndpoint`, never from raw Secret); error state adds `error` string
- [ ] **API-02**: `GET /api/credentials?type=<t>` returns only credentials of the requested type with `ready: true` (used by blueprint SDK to populate dropdowns)
- [ ] **API-03**: `POST /api/credentials` body: `type`, `displayName`, `key`, optional `username` (weka-storage), optional `endpoint` (weka-storage); backend slugifies `displayName` to create `metadata.name`, creates raw `warp-cred-<slug>` Secret first, then creates `WarpCredential` CR pointing to it; slug collision appends short suffix
- [ ] **API-04**: `DELETE /api/credentials/<name>` deletes the `WarpCredential` CR and the raw `warp-cred-<slug>` Secret; derived `warp-<name>-*` secrets are left intact
- [ ] **API-05**: `GET /api/weka/overview?credential=<name>` proxies the WEKA REST API: resolves credential, reads raw Secret, exchanges for Bearer token via `POST /api/v2/login`, makes three parallel calls (`fileSystems`, `cluster`, `containers`), assembles and returns structured JSON; 60s server-side cache keyed by credential name; `?bust=1` query param bypasses cache (used by Refresh button)
- [ ] **API-06**: `/api/weka/overview` response schema: `capacity` object (totalBytes, usedBytes, availableBytes, usedPercent), `filesystems` array (name, totalBytes, usedBytes, usedPercent — no uid), `backendNodes` array (ip string only), `fetchedAt` ISO timestamp
- [ ] **API-07**: Remove `/api/secret/nvidia` and `/api/secret/huggingface` endpoints; update the old settings page JavaScript that called them
- [x] **API-08**: No raw key values or token values are logged at any log level by the GUI backend or the operator at any point in the credential lifecycle

### SDK

- [ ] **SDK-01**: New file `app-store-gui/webapp/templates/_credential_macros.html` containing Jinja2 macros `credential_select` and `weka_storage_select`
- [ ] **SDK-02**: `credential_select(type, field_name, label=None, required=True)` renders a `<select>` populated from `credentials_by_type[type]`; when no credentials of the requested type are ready, renders a hint paragraph with a link to `/settings#credentials` instead of an empty select
- [ ] **SDK-03**: `weka_storage_select(credential_field, endpoint_field, label)` renders a credential dropdown where each `<option>` carries a `data-endpoint` attribute; an endpoint `<input type="url">` is pre-populated from the selected credential's endpoint; an inline `warpSyncEndpoint` JavaScript function updates the endpoint input when the selection changes
- [ ] **SDK-04**: All blueprint install page route handlers in `app-store-gui/webapp/main.py` inject `credentials_by_type` (dict keyed by credential type, values are lists of ready credential objects) into the Jinja2 template context
- [ ] **SDK-05**: `credentials_by_type` data is fetched from live `WarpCredential` CRs at route-render time via a shared helper; empty dict is used as fallback if the Kubernetes API is unreachable (macros degrade gracefully to hint mode)

## v2 Requirements

### Keycloak Group Scoping

- **KCLO-01**: Uncomment `spec.groups` in `WarpCredential` CRD and enforce in admission
- **KCLO-02**: `GET /api/credentials?type=<t>` filters results by the authenticated user's Keycloak group memberships against `spec.groups`
- **KCLO-03**: Blueprint pages automatically show only credentials the current user is permitted to use (no changes to macros required)

### Credential Distribution

- **DIST-01**: At blueprint install time, selected credential names are used to copy derived secrets (`warp-<name>-*`) into the blueprint's target namespace
- **DIST-02**: Operator or install hook cleans up copied secrets when a blueprint is uninstalled

### Secret Rotation

- **ROT-01**: Administrator can update a credential's key value in place (without delete-and-recreate)
- **ROT-02**: Operator detects key change and re-derives dependent secrets within 60 seconds

## Out of Scope

| Feature | Reason |
|---------|--------|
| Distributing stored credentials into blueprint target namespaces | Follow-on work (DIST-01..02 in v2) |
| Keycloak group-scoped credential access enforcement | Follow-on; CRD seam is designed here but not wired (KCLO-01..03 in v2) |
| Managing database credentials (Neo4j, ArangoDB, MinIO) | Blueprint-internal, not platform-managed |
| Secret rotation or expiry notifications | ROT-01..02 in v2 |
| OpenAI API key management | Not needed by any shipped blueprint |
| External secrets managers (Vault, AWS Secrets Manager) | `spec.secretRef` abstraction leaves the door open without requiring it now |
| Real-time WEKA storage monitoring or alerting | Settings panel is on-demand, not a live dashboard |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CRD-01 | Phase 21 | Complete |
| CRD-02 | Phase 21 | Complete |
| CRD-03 | Phase 21 | Complete |
| CRD-04 | Phase 21 | Complete |
| CRD-05 | Phase 21 | Complete |
| CRD-06 | Phase 21 | Complete |
| OPS-01 | Phase 22 | Complete |
| OPS-02 | Phase 22 | Complete |
| OPS-03 | Phase 22 | Complete |
| OPS-04 | Phase 22 | Complete |
| OPS-05 | Phase 22 | Complete |
| OPS-06 | Phase 22 | Complete |
| OPS-07 | Phase 22 | Complete |
| OPS-08 | Phase 22 | Complete |
| OPS-09 | Phase 22 | Complete |
| GUI-01 | Phase 24 | Pending |
| GUI-02 | Phase 24 | Pending |
| GUI-03 | Phase 24 | Pending |
| GUI-04 | Phase 24 | Pending |
| GUI-05 | Phase 24 | Pending |
| GUI-06 | Phase 24 | Pending |
| GUI-07 | Phase 24 | Pending |
| GUI-08 | Phase 24 | Pending |
| GUI-09 | Phase 24 | Pending |
| GUI-10 | Phase 24 | Pending |
| GUI-11 | Phase 24 | Pending |
| GUI-12 | Phase 24 | Pending |
| GUI-13 | Phase 24 | Pending |
| GUI-14 | Phase 24 | Pending |
| GUI-15 | Phase 24 | Pending |
| API-01 | Phase 23 | Pending |
| API-02 | Phase 23 | Pending |
| API-03 | Phase 23 | Pending |
| API-04 | Phase 23 | Pending |
| API-05 | Phase 23 | Pending |
| API-06 | Phase 23 | Pending |
| API-07 | Phase 23 | Pending |
| API-08 | Phase 22 + 23 | Complete |
| SDK-01 | Phase 25 | Pending |
| SDK-02 | Phase 25 | Pending |
| SDK-03 | Phase 25 | Pending |
| SDK-04 | Phase 25 | Pending |
| SDK-05 | Phase 25 | Pending |

**Coverage:**
- v1 requirements: 38 total
- Mapped to phases: 38
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-11*
*Last updated: 2026-06-11 after v6.0 milestone start*
