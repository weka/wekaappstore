# PRD: Secret Management Overhaul for WEKA App Store

**Status:** Draft  
**Author:** Christopher Jenkins  
**Date:** 2026-06-11  
**Area:** Settings GUI · Operator CRD · Secret lifecycle · Blueprint integration · WEKA Storage visibility

---

## Problem

The App Store currently stores API keys as flat Opaque Kubernetes Secrets in whatever namespace the GUI happens to be connected to. This has several gaps:

1. **NVIDIA NGC requires two distinct secret forms.** Pulling container images from `nvcr.io` requires a `docker-registry`-type imagePullSecret. Application services (NIM-LLM, Nemo, VSS) separately require a plain Opaque secret. Today, blueprints like nvidia-vss bundle a one-off Kubernetes Job to perform this conversion at install time — fragile, duplicated per-blueprint, and invisible to the administrator.
2. **No authoritative central store.** Secrets land in an ad-hoc namespace with no single source of truth for the operator to rely on when distributing credentials to blueprint namespaces.
3. **One key per type.** Some customers need multiple NGC keys (e.g. production vs. research), multiple HuggingFace tokens for different projects, or different WEKA API tokens scoped to different clusters. The current flat model has no concept of named, selectable credentials.
4. **Settings UI is buried and cluttered.** Key entry forms are mixed in with cluster-status widgets and have no clear feedback mechanism.
5. **WEKA Storage API token has no first-class management.** Blueprints that integrate with the WEKA management plane — creating snapshots, provisioning filesystems, reading cluster telemetry — need a WEKA REST API token. Today this is a fully manual kubectl operation.
6. **Blueprint install pages have no standardised credential picker.** When installing a blueprint that requires an NGC key, there is no UI mechanism to select which stored credential to use.
7. **No storage visibility.** Administrators provisioning blueprints that create WEKA filesystems have no way to see current storage utilisation, existing filesystems, or backend node topology from within the App Store. They must context-switch to a separate WEKA management tool to check whether there is capacity to proceed.

---

## Goals

1. Give administrators a purpose-built interface to register any number of named credentials across three types: NVIDIA NGC, HuggingFace, and WEKA Storage API token.
2. Store credentials as individual `WarpCredential` CRs, making the operator the single authority for credential state and derived secret creation.
3. For NVIDIA NGC, automatically derive and maintain both required secret forms (docker pull secret + Opaque API key secret) per named credential immediately after registration.
4. Provide a blueprint page SDK — a Jinja2 macro library and backing REST endpoint — so blueprint authors can render a dropdown of available credentials for a given type with a single line of template code.
5. Design the Keycloak group-scoping field into the CRD now as a defined-but-empty extension point, without implementing it yet.
6. Keep the GUI surface clean: grouped lists per credential type, traffic-light status per entry, Name field on creation only.
7. Surface a WEKA Storage Overview panel in Settings, powered by the stored WEKA credential, showing total and used capacity, a human-readable filesystem inventory with per-filesystem utilisation, and backend node IP addresses.

---

## Non-Goals (this PRD)

- Distributing stored credentials into blueprint target namespaces at install time (follow-on work).
- Enforcing per-group credential access via Keycloak (follow-on; the CRD seam is designed here).
- Managing database credentials (Neo4j, ArangoDB, MinIO) — these are blueprint-internal.
- Secret rotation or expiry notifications.
- OpenAI API key management (not yet needed by any shipped blueprint).

---

## Background: What Blueprints Actually Need

Based on inspection of `/warp-blueprints/manifests/`:

| Credential type | Blueprints | Derived secret forms |
|---|---|---|
| NVIDIA NGC API Key | nvidia-vss, cluster_init | Opaque (`NGC_API_KEY`) + `kubernetes.io/dockerconfigjson` pull secret for `nvcr.io` |
| HuggingFace Token | future (no shipped blueprint yet) | Opaque (`HF_API_KEY`) |
| WEKA Storage API Token + Endpoint | blueprints using WEKA management REST API for snapshots, filesystem provisioning, cluster telemetry | Opaque (`WEKA_API_USERNAME` + `WEKA_API_TOKEN` + `WEKA_API_ENDPOINT`) |

The NVIDIA VSS blueprint currently embeds a Kubernetes Job that converts a flat `nvidia-api-key` secret into a docker pull secret at install time. This Job pattern should be retired — the operator will own that derivation.

---

## Solution

### 1. New CRD: `WarpCredential`

Replace the earlier singleton concept with a **multi-instance CRD** `WarpCredential` (group: `warp.io/v1alpha1`). Each instance represents one named credential. Multiple instances of the same `spec.type` are allowed and expected.

**Naming convention:** The `metadata.name` field is the stable identifier used internally for derived secret naming. The human-readable label is `spec.displayName`.

#### CRD Spec

```yaml
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: ngc-production           # kebab-case, unique within namespace
  namespace: weka-app-store      # always the App Store namespace
spec:
  type: nvidia-ngc               # nvidia-ngc | huggingface | weka-storage
  displayName: "NGC Production"  # shown in GUI and blueprint install dropdowns
  secretRef:
    name: warp-cred-ngc-production   # Opaque secret holding the raw key value
    key: NGC_API_KEY
  # endpoint is only used for weka-storage type credentials;
  # ignored for all other types
  endpoint: "https://weka-cluster:14000"   # WEKA management REST API URL
  # --- Future extension, not implemented in this phase ---
  # groups: []
  # When Keycloak integration is active, list Keycloak group names that
  # may select this credential on blueprint install pages.
  # Empty list = available to all authenticated administrators.
```

The operator does not store the key value in the CR. The raw value lives only in the referenced Kubernetes Secret. The CR declares that a credential exists, what type it is, what to call it, and what derivations to maintain.

#### CRD Status

```yaml
status:
  conditions:
    - type: KeyReady
      status: "True" | "False"
      reason: KeyPresent | KeyMissing | DerivationFailed
      message: "..."
      lastTransitionTime: "..."
    - type: DockerSecretReady    # nvidia-ngc only
      status: "True" | "False"
      reason: ...
      lastTransitionTime: "..."
  derivedSecrets:
    - name: warp-ngc-production-apikey    # created/maintained by operator
      type: Opaque
    - name: warp-ngc-production-docker    # created/maintained by operator
      type: kubernetes.io/dockerconfigjson
  lastSyncTime: "..."
```

The `derivedSecrets` list is populated by the operator after successful reconciliation. Blueprint install logic (follow-on work) uses these names when copying secrets to target namespaces.

For `weka-storage` credentials, the `endpoint` value is also written into `status` as a non-sensitive field so the API can return it directly without reading the underlying Secret:

```yaml
  wekaEndpoint: "https://weka-cluster:14000"   # surfaced for API / macro use
```

---

### 2. Operator: `WarpCredential` Reconciler

Add a new kopf handler in `operator_module/main.py` to watch `WarpCredential` resources.

#### Derived secret naming convention

All derived secrets are named `warp-<metadata.name>-<suffix>` so they are predictable, namespaced to the credential, and never collide:

| Credential type | Suffix | Secret type |
|---|---|---|
| `nvidia-ngc` | `apikey` | Opaque, key: `NGC_API_KEY` |
| `nvidia-ngc` | `docker` | `kubernetes.io/dockerconfigjson` for `nvcr.io` |
| `huggingface` | `token` | Opaque, key: `HF_API_KEY` |
| `weka-storage` | `token` | Opaque, key: `WEKA_API_TOKEN` |

Example: a `WarpCredential` named `ngc-production` produces `warp-ngc-production-apikey` and `warp-ngc-production-docker`.

#### Reconciliation logic

**On create/update of a `WarpCredential`:**

1. Validate `spec.type` is a known value; raise `kopf.PermanentError` if not.
2. Resolve `spec.secretRef` — if the referenced Secret does not exist, raise `kopf.TemporaryError` (retry 30s).
3. Read the raw key value from the referenced Secret.
4. **For `nvidia-ngc`:**
   - Create or patch `warp-<name>-apikey` (Opaque, key: `NGC_API_KEY`) in the App Store namespace.
   - Build the `.dockerconfigjson` payload: `{"auths": {"nvcr.io": {"username": "$oauthtoken", "password": "<key>", "auth": "<base64($oauthtoken:<key>)>"}}}`. Create or patch `warp-<name>-docker` (type: `kubernetes.io/dockerconfigjson`).
   - Set `KeyReady` and `DockerSecretReady` conditions.
5. **For `huggingface`:**
   - Create or patch `warp-<name>-token` (Opaque, key: `HF_API_KEY`).
   - Set `KeyReady` condition.
6. **For `weka-storage`:**
   - Create or patch `warp-<name>-token` (Opaque, two keys: `WEKA_API_TOKEN` and `WEKA_API_ENDPOINT`) in the App Store namespace. The token authenticates against the WEKA management REST API (typically port 14000); the endpoint is the base URL of that API. Both travel together because a token without its cluster endpoint is unusable. This credential is entirely separate from CSI driver credentials.
   - Copy `spec.endpoint` into `status.wekaEndpoint` (the endpoint is non-sensitive and making it available in status means the API never needs to read the raw Secret to serve it to blueprint pages).
   - Set `KeyReady` condition.
7. Update `status.derivedSecrets` with the names of all secrets created.
8. Update `status.lastSyncTime`.

**Error handling:**
- Missing `spec.secretRef` Secret → `kopf.TemporaryError` (retry 30s)
- Empty or whitespace-only key value → `kopf.PermanentError`
- Unknown `spec.type` → `kopf.PermanentError`

**On delete of `WarpCredential`:** Do NOT delete derived secrets (`warp-<name>-*`). They may be in active use by running workloads. Log a warning. Removal of derived secrets requires an explicit `kubectl delete` by an administrator.

#### Why reconcile rather than a one-shot Job?

A reconciler is idempotent and self-healing. If `warp-ngc-production-docker` is accidentally deleted, the next reconcile cycle restores it automatically. A Job fires once and cannot recover. The operator already has the RBAC, kopf machinery, and retry semantics — adding a handler here is minimal incremental work.

---

### 3. Settings GUI Overhaul

#### Layout

The `settings.html` page is restructured so **Credential Management** appears first, above all other sections.

```
┌──────────────────────────────────────────────────────────────┐
│  Credential Management                                        │
│                                                               │
│  NVIDIA NGC API Keys                              [+ Add]    │
│  ├── ● NGC Production       [●] Ready    [Delete]            │
│  └── ● NGC Research         [●] Ready    [Delete]            │
│                                                               │
│  HuggingFace Tokens                               [+ Add]    │
│  └── (none stored)                                           │
│                                                               │
│  WEKA Storage API Tokens                          [+ Add]    │
│  └── ● WEKA Cluster Primary  [●] Ready   [Delete]            │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  WEKA Storage Overview                                        │
│  [WEKA Cluster Primary ▼]                    [↺ Refresh]     │
│  ...                                                          │
└──────────────────────────────────────────────────────────────┘

  [Kubernetes Auth Status]
  [Cluster Status]
  [Blueprint Management]
  [Debug]
```

#### Adding a credential

Clicking **[+ Add]** on any type expands an inline form at the bottom of that type's list.

**For NVIDIA NGC and HuggingFace:**
```
  Name  [________________________________]   ← required, free text, e.g. "NGC Production"
  Key   [________________________________]   ← password input, type="password"
        [Save]  [Cancel]
```

**For WEKA Storage (three fields):**
```
  Name      [________________________________]   ← required, e.g. "WEKA Cluster Primary"
  Username  [________________________________]   ← WEKA API user account, e.g. "admin"
  API Token [________________________________]   ← password/token for that account (type="password")
  Endpoint  [________________________________]   ← URL input, e.g. https://weka-cluster:14000
            [Save]  [Cancel]
```

- **Name** is required and becomes `spec.displayName`. The backend derives `metadata.name` by slugifying it (lowercase, spaces to hyphens, max 52 chars). If a slug collision exists, the backend appends a short suffix.
- **Username** (WEKA only) is required. The WEKA REST API uses a login flow (`POST /api/v2/login`) where the token is submitted as the password for this username. Stored as `WEKA_API_USERNAME` in the credential Secret.
- **API Token** is required and non-empty before Save is enabled. Stored as `WEKA_API_TOKEN`. This is the password/token for the WEKA user account — ideally a long-lived org API token rather than a personal password.
- **Endpoint** (WEKA only) is required; validated as a URL. Stored in `spec.endpoint` on the CR and as `WEKA_API_ENDPOINT` in the credential Secret.
- Only one inline add-form is open at a time per credential type.

#### Credential row states

Each stored credential row renders in one of three states:

**Green — stored and operator-confirmed ready:**
```
● NGC Production    [●] Ready    [Delete]
```

**Amber — transitional (after Save, waiting for operator):**
```
◐ NGC Production    [◐] Verifying...
```
Poll interval: 2s, timeout: 30s. If operator does not confirm within 30s, row shows an inline error and the credential is removed from the list (the raw Secret was written; the operator will continue retrying in the background, and the status will update on next page load).

**Red — operator reports error:**
```
○ NGC Production    [○] Error: DerivationFailed — invalid key format    [Delete]
```

There is no "edit in place" mode. To replace a key value: click Delete, then re-add with the same name.

#### Behaviour on Delete

1. Frontend sends `DELETE /api/credentials/<name>`.
2. Backend deletes the raw credential Secret and deletes the `WarpCredential` CR.
3. Derived secrets (`warp-<name>-*`) are **not** deleted by the operator.
4. The credential row disappears from the list.

#### What is NOT shown

- Raw key values or raw secret names.
- The Kubernetes Secret names that back each entry.
- Namespace selector (all credentials are always in the App Store namespace).

---

### 4. Backend API Changes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/credentials` | List all `WarpCredential` CRs with status summary |
| `GET` | `/api/credentials?type=nvidia-ngc` | List by type (used by blueprint page SDK) |
| `POST` | `/api/credentials` | Create new named credential (body: `type`, `displayName`, `key`, optional `endpoint`) |
| `DELETE` | `/api/credentials/<name>` | Delete credential CR and raw Secret |
| `GET` | `/api/weka/overview?credential=<name>` | Proxy WEKA REST API; returns capacity, filesystems, backend nodes |

The existing `/api/secret/nvidia` and `/api/secret/huggingface` endpoints are removed. Any callers (the old settings page) are updated.

**`GET /api/credentials` response:**

```json
[
  {
    "name": "ngc-production",
    "displayName": "NGC Production",
    "type": "nvidia-ngc",
    "ready": true,
    "dockerSecretReady": true,
    "lastSyncTime": "2026-06-11T14:32:00Z"
  },
  {
    "name": "ngc-research",
    "displayName": "NGC Research",
    "type": "nvidia-ngc",
    "ready": true,
    "dockerSecretReady": true,
    "lastSyncTime": "2026-06-11T15:01:00Z"
  },
  {
    "name": "hf-nlp-team",
    "displayName": "HuggingFace NLP Team",
    "type": "huggingface",
    "ready": false,
    "error": "DerivationFailed: invalid token format"
  },
  {
    "name": "weka-cluster-primary",
    "displayName": "WEKA Cluster Primary",
    "type": "weka-storage",
    "ready": true,
    "endpoint": "https://weka-cluster:14000",
    "lastSyncTime": "2026-06-11T14:45:00Z"
  }
]
```

The `endpoint` field is included for `weka-storage` credentials only. It is read from `status.wekaEndpoint` — never from the underlying Secret — so no credential data is exposed in the listing.

**`POST /api/credentials` request body:**
```json
{
  "type": "nvidia-ngc",
  "displayName": "NGC Production",
  "key": "<raw key value>"
}
```

For `weka-storage`, the body includes `username` and `endpoint` in addition to `key`:
```json
{
  "type": "weka-storage",
  "displayName": "WEKA Cluster Primary",
  "username": "admin",
  "key": "<api token / password>",
  "endpoint": "https://weka-cluster:14000"
}
```

Backend creates the raw `warp-cred-<slug>` Secret, then creates the `WarpCredential` CR with the `secretRef` pointing to it. The raw key value is never stored in the CR.

---

### 5. Blueprint Credential Selector SDK

Blueprint install pages need a standardised way to let an administrator select which stored credential to use for a given type. This is solved with two components that blueprint authors use together.

#### 5a. REST endpoint for credential listing

```
GET /api/credentials?type=nvidia-ngc
```

Returns only credentials of the requested type that have `ready: true`. Used by both the Jinja2 macro (server-side) and any JavaScript on blueprint pages that needs dynamic refresh.

#### 5b. Jinja2 macro library: `_credential_macros.html`

A shared template file added to `app-store-gui/webapp/templates/_credential_macros.html`. Blueprint template authors import from it:

```jinja2
{% from "_credential_macros.html" import credential_select %}
```

**`credential_select` macro signature:**

```jinja2
{% macro credential_select(type, field_name, label=None, required=True) %}
```

**Rendered output (example — nvidia-ngc, two credentials available):**

```html
<div class="warp-credential-field">
  <label for="ngc_credential">NVIDIA NGC Credential</label>
  <select id="ngc_credential" name="ngc_credential" required
          data-credential-type="nvidia-ngc">
    <option value="">— Select NGC credential —</option>
    <option value="ngc-production">NGC Production</option>
    <option value="ngc-research">NGC Research</option>
  </select>
</div>
```

**Rendered output (no credentials stored):**

```html
<div class="warp-credential-field">
  <label for="ngc_credential">NVIDIA NGC Credential</label>
  <p class="warp-credential-hint">
    No nvidia-ngc credentials are stored.
    <a href="/settings#credentials">Add one in Settings ↗</a>
  </p>
</div>
```

When no credentials are ready, the macro renders a hint with a link to Settings rather than an empty select. This prevents silent misconfiguration.

#### 5c. Backend context injection

When the GUI renders any blueprint install page, a shared helper fetches all `WarpCredential` CRs grouped by type and injects `credentials_by_type` into the Jinja2 template context. Blueprint page authors never call the API themselves — they just use the macro and the data is already there.

```python
# In the blueprint page route handler (app-store-gui/webapp/main.py):
credentials_by_type = get_credentials_by_type()   # fetches WarpCredential CRs
return templates.TemplateResponse("blueprint_myapp.html", {
    "request": request,
    "credentials_by_type": credentials_by_type,   # injected automatically
    ...
})
```

#### 5c-ii. WEKA storage combined macro: `weka_storage_select`

Because the WEKA endpoint travels with the credential, a second macro is provided that renders the credential dropdown **and** the endpoint field as a linked pair:

```jinja2
{% macro weka_storage_select(credential_field="weka_credential",
                              endpoint_field="weka_endpoint",
                              label="WEKA Storage Credential") %}
```

**Rendered output (one WEKA credential stored):**

```html
<div class="warp-weka-field">
  <label>WEKA Storage Credential</label>
  <select name="weka_credential" required
          data-credential-type="weka-storage"
          onchange="warpSyncEndpoint(this, 'weka_endpoint')">
    <option value="">— Select WEKA credential —</option>
    <option value="weka-cluster-primary"
            data-endpoint="https://weka-cluster:14000">
      WEKA Cluster Primary
    </option>
  </select>

  <label>WEKA Cluster Endpoint</label>
  <input type="url" name="weka_endpoint"
         value="https://weka-cluster:14000"
         placeholder="https://weka-cluster:14000" />
  <p class="warp-hint">Pre-populated from the selected credential.
     Override only if this blueprint targets a different endpoint.</p>
</div>
```

The `data-endpoint` attribute on each option carries the stored endpoint for that credential. A small inline script (`warpSyncEndpoint`) updates the endpoint input when the selection changes. Blueprint pages that typically have one WEKA cluster will show the endpoint pre-filled and largely invisible to the user. Those that allow per-install overrides can leave the field editable.

If no WEKA credentials are stored, the same "Add one in Settings ↗" hint pattern applies.

#### 5d. What a blueprint page author writes

Full example for a blueprint page that needs an NGC key, a HuggingFace token, and WEKA storage access:

```jinja2
{% from "_credential_macros.html" import credential_select, weka_storage_select %}

<form method="POST" action="/api/blueprints/install/myapp">
  <!-- other install options -->

  {{ credential_select(
       type="nvidia-ngc",
       field_name="ngc_credential",
       label="NVIDIA NGC Credential"
  ) }}

  {{ credential_select(
       type="huggingface",
       field_name="hf_credential",
       label="HuggingFace Token",
       required=False
  ) }}

  {{ weka_storage_select(
       credential_field="weka_credential",
       endpoint_field="weka_endpoint",
       label="WEKA Storage Credential"
  ) }}

  <button type="submit">Install</button>
</form>
```

The selected credential `name` values are submitted with the form and used by the install API when creating the `WekaAppStore` CR. The follow-on distribution work will define exactly how selected credential names map to secrets copied into the blueprint's target namespace.

#### 5e. Keycloak extension seam

When Keycloak integration is implemented, the `GET /api/credentials?type=<t>` endpoint will filter results by the authenticated user's group memberships, comparing against `spec.groups` on each `WarpCredential`. The macro and blueprint page authors require no changes — the filtering happens at the API layer. Blueprint pages will automatically show only the credentials the current user is permitted to use.

---

### 7. WEKA Storage Overview Panel

#### Purpose

Once an administrator has registered a WEKA Storage credential, the App Store has everything it needs to query the WEKA management API on their behalf. This section defines a live storage overview panel on the Settings page that shows cluster capacity, filesystem inventory, and backend node topology — allowing administrators to make informed decisions when deploying blueprints that provision WEKA filesystems, without leaving the App Store.

#### UI layout

```
┌──────────────────────────────────────────────────────────────────┐
│  WEKA Storage Overview                                            │
│  [WEKA Cluster Primary ▼]                         [↺ Refresh]   │
│  Last updated: 2 minutes ago                                      │
│                                                                   │
│  Cluster Capacity                                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Total       400.0 TiB                                   │    │
│  │  Used        312.5 TiB  ████████████████░░░░░░  78%     │    │
│  │  Available    87.5 TiB                                   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Filesystems (6)                                                  │
│  ┌──────────────────┬──────────┬──────────┬───────────────────┐  │
│  │ Name             │ Total    │ Used     │ Utilisation       │  │
│  ├──────────────────┼──────────┼──────────┼───────────────────┤  │
│  │ default          │ 50.0 TiB │ 42.1 TiB │ ████████░░░  84% │  │
│  │ models           │ 100.0 TiB│ 31.2 TiB │ ███░░░░░░░░  31% │  │
│  │ checkpoints      │  80.0 TiB│ 78.8 TiB │ █████████░░  98% │  │
│  │ scratch          │  20.0 TiB│  4.0 TiB │ ██░░░░░░░░░  20% │  │
│  └──────────────────┴──────────┴──────────┴───────────────────┘  │
│                                                                   │
│  Backend Nodes (8)                                                │
│  10.0.1.1    10.0.1.2    10.0.1.3    10.0.1.4                   │
│  10.0.1.5    10.0.1.6    10.0.1.7    10.0.1.8                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

#### Credential selector

If multiple `WarpCredential` CRs of type `weka-storage` are registered, a dropdown at the top of the panel allows the administrator to switch between clusters. The panel data refreshes when the selection changes. If only one WEKA credential exists the dropdown is replaced by a static label.

#### Panel states

| State | Display |
|---|---|
| No WEKA credential registered | "No WEKA Storage credential configured. Add one in Credential Management ↑" — no panel content |
| Loading (initial fetch or Refresh clicked) | Spinner in place of data rows |
| WEKA API unreachable / auth failure | Error message with last-successful timestamp. Stale data is not shown — only the error and a Refresh button |
| Success | Full panel as above |

#### Refresh behaviour

- Data is fetched fresh on every page load of Settings.
- The **[↺ Refresh]** button forces an immediate re-fetch, bypassing the server-side response cache.
- The GUI backend caches the WEKA API response for 60 seconds to avoid hammering the cluster on rapid page reloads. Cache is keyed per credential name.
- The "Last updated" timestamp reflects when the data was fetched from WEKA, not when the page was loaded (so cached responses show the actual data age).

#### Filesystem display rules

- Show the filesystem `name` field only — never the internal UUID.
- Sort by utilisation descending so high-watermark filesystems appear first.
- Filesystems at ≥ 90% utilisation have their utilisation bar rendered in amber/orange rather than the default colour, as a visual warning to the administrator.
- No pagination for now; if a cluster has more than 20 filesystems, show a count and truncate with a "Show all" toggle.

#### Backend node display

- Show the primary management IP for each backend node — the first IP in the node's IP list that is not a loopback or link-local address.
- Display as a wrapped grid of IP addresses with a node count header.
- No hostname resolution; IPs only.

---

### 7a. Backend: `/api/weka/overview` endpoint

The GUI backend acts as a proxy to the WEKA management REST API. The WEKA token and any derived access tokens never reach the browser.

```
GET /api/weka/overview?credential=<name>
```

#### WEKA REST API authentication

The WEKA management API uses a two-step auth flow confirmed by the WEKA docs:

1. `POST <endpoint>/api/v2/login` — body: `{"username": "<user>", "password": "<token>"}` → returns:
   ```json
   { "access_token": "...", "token_type": "Bearer" }
   ```
   The `access_token` is **short-lived (5 minutes)**. A `refresh_token` (valid 1 year) is also returned and should be stored for renewal.

2. Subsequent API calls use `Authorization: Bearer <access_token>`.

3. When the access token expires, the backend calls `POST <endpoint>/api/v2/login/refresh` with the refresh token to obtain a new access token without re-entering credentials.

**Implication for what we store:** The `WEKA_API_TOKEN` stored in the credential Secret is a **refresh token** (or a long-lived org/API token if the WEKA cluster has been configured to issue one). The GUI backend treats it as a durable credential and exchanges it for a short-lived Bearer token before each API interaction. This exchange is transparent to the administrator — from their perspective the credential just works.

If WEKA 4.x supports dedicated API tokens (long-lived, not tied to a user session), those should be preferred. This is implementation-time detail to confirm against the target cluster version; the stored field name (`WEKA_API_TOKEN`) is the same either way.

The Settings form should also collect a **username** field for the WEKA API user account (stored as `WEKA_API_USERNAME` in the credential Secret alongside `WEKA_API_TOKEN`). The token is the password in the login call.

#### WEKA API calls made by the backend

Three calls, made in parallel after obtaining a valid access token:

| Call | WEKA endpoint | Purpose |
|---|---|---|
| 1 | `GET /api/v2/fileSystems` | Filesystem inventory — note camelCase `S` per WEKA API convention |
| 2 | `GET /api/v2/cluster` | Cluster-level health, performance, and capacity stats |
| 3 | `GET /api/v2/containers` | Container (process) list — filtered client-side for BACKEND role to get backend node IPs |

**Note on WEKA terminology:** In the WEKA API, what are commonly called "backend nodes" are represented as *containers* — WEKA processes running on physical hosts. The `GET /api/v2/containers` response includes a `roles` or `mode` field per container. BACKEND-role containers are the data-serving nodes whose IPs are operationally relevant. IP addresses for each container are retrieved via `GET /api/v2/containers/{uid}/netdevs` or may be included in the containers list response directly. The implementation should verify the exact field name against the live cluster's Swagger UI at `https://<cluster>:14000/api/v2/docs`.

**Field names to verify at implementation time** (based on WEKA docs — confirm against live Swagger):

| Endpoint | Field | Meaning |
|---|---|---|
| `GET /api/v2/fileSystems` | `name` | Human-readable filesystem name |
| `GET /api/v2/fileSystems` | `uid` | Internal UUID — display the `name`, never the `uid` |
| `GET /api/v2/fileSystems` | `total_budget` or `size` | Total capacity in bytes (verify field name) |
| `GET /api/v2/fileSystems` | `used_total` or `used_size` | Used bytes (verify field name) |
| `GET /api/v2/cluster` | (capacity subfields) | Total cluster capacity and used bytes (verify structure) |
| `GET /api/v2/containers` | `roles` or `mode` | Container role — filter for `BACKEND` |
| `GET /api/v2/containers/{uid}/netdevs` | `ip_address` or `ips` | IP address(es) per container network device |

The interactive Swagger UI at `https://<cluster>:14000/api/v2/docs` is the authoritative reference for exact field names during implementation.

#### Backend logic

1. Resolve the named `WarpCredential` CR. If `spec.type != weka-storage` or `status.KeyReady != True`, return 400.
2. Read `WEKA_API_TOKEN`, `WEKA_API_USERNAME`, and `WEKA_API_ENDPOINT` from the credential's raw Secret.
3. Exchange for a Bearer access token via `POST /api/v2/login`. On failure (invalid credentials, unreachable), return error response immediately.
4. Make the three API calls in parallel.
5. Assemble and return the structured response below.
6. Cache the assembled response for 60 seconds keyed by credential name. A `?bust=1` query param bypasses cache (used by the Refresh button).

#### App Store overview response schema

```json
{
  "credential": "weka-cluster-primary",
  "displayName": "WEKA Cluster Primary",
  "endpoint": "https://weka-cluster:14000",
  "fetchedAt": "2026-06-11T14:32:00Z",
  "capacity": {
    "totalBytes": 439804651110400,
    "usedBytes":  343397064744960,
    "availableBytes": 96407586365440,
    "usedPercent": 78.1
  },
  "filesystems": [
    {
      "name": "checkpoints",
      "totalBytes": 87960930222080,
      "usedBytes":  86558654611456,
      "usedPercent": 98.4
    },
    {
      "name": "default",
      "totalBytes": 54975581388800,
      "usedBytes":  46286956953600,
      "usedPercent": 84.2
    }
  ],
  "backendNodes": [
    { "ip": "10.0.1.1" },
    { "ip": "10.0.1.2" },
    { "ip": "10.0.1.3" }
  ]
}
```

Capacity values are always bytes in the App Store API response. The frontend formats them to TiB/GiB for display.

#### Error response

```json
{
  "error": "WEKAApiUnreachable",
  "message": "Connection refused at https://weka-cluster:14000",
  "credential": "weka-cluster-primary"
}
```

Possible `error` values: `WEKAApiUnreachable`, `WEKAAuthFailed`, `WEKACredentialNotReady`.

---

### 6. Helm Chart Changes

- Add `WarpCredential` CRD to `weka-app-store-operator-chart/templates/crd.yaml`.
- Add RBAC `Role` + `RoleBinding` in the App Store namespace granting the operator create/patch/get/delete on Secrets (scoped to App Store namespace only, not cluster-wide).
- No bootstrap CR required — the credential list starts empty and populates as administrators add entries through the GUI.

---

## Key Design Decisions

### Why `WarpCredential` (multi-instance) rather than a singleton registry?

Each credential reconciles independently. Adding a second NGC key does not risk re-triggering reconciliation of unrelated credentials. `kubectl get warpcredentials -n weka-app-store` gives a natural inventory. Deleting one entry is surgical. The singleton model would require list-diffing logic in the reconciler and makes RBAC scoping harder if groups are ever per-credential.

### Why Jinja2 macros rather than a JavaScript component or REST-only SDK?

The existing blueprint pages are Jinja2 templates rendered server-side. A macro fits the existing pattern exactly — one import line, one macro call, zero new dependencies. A JavaScript web component would work but adds a new dependency pattern and requires blueprint authors to handle async loading states. If a blueprint page eventually needs dynamic credential reload without a page refresh, it can call `GET /api/credentials?type=<t>` directly; the macro and the API endpoint are complementary, not competing.

### Why not use an external secrets manager (Vault, AWS Secrets Manager)?

Out of scope for the current self-hosted Kubernetes on WEKA deployment target. The `spec.secretRef` abstraction in `WarpCredential` already encapsulates where the raw value lives — a future implementation could point that reference at a Vault-synced Secret without changing the CRD spec or the reconciler logic above the point of secret resolution.

### NGC: two secrets from one key

The two NGC secret forms (Opaque API key + docker pull secret for `nvcr.io`) are both derived from the same key by the operator. Blueprint pages select a credential by name; the operator knows which derived secrets correspond to that name via the `status.derivedSecrets` list. Blueprint authors never construct the `dockerconfigjson` payload themselves.

### Keycloak scoping: designed-for but not implemented

The `spec.groups` field is documented in the CRD spec as a commented-out extension point. The API filtering layer is described in section 5e. No Keycloak connectivity, session management, or RBAC enforcement is built in this phase. Adding it later requires: (1) uncommenting `groups` in the CRD, (2) implementing auth middleware in the GUI, (3) adding the filter to `GET /api/credentials`. The blueprint pages and macros require no changes.

---

## Affected Files

| File | Change |
|---|---|
| `operator_module/main.py` | Add `WarpCredential` kopf handler |
| `app-store-gui/webapp/main.py` | Add `/api/credentials` endpoints; inject `credentials_by_type` into blueprint page routes; remove old `/api/secret/nvidia` + `/api/secret/huggingface` |
| `app-store-gui/webapp/templates/settings.html` | Restructure — Credential Management section first, multi-credential list UI per type |
| `app-store-gui/webapp/templates/_credential_macros.html` | New — `credential_select` macro (NGC, HF), `weka_storage_select` macro (credential + endpoint pair), `warpSyncEndpoint` inline script |
| `app-store-gui/webapp/templates/settings.html` (WEKA section) | Add WEKA Storage Overview panel with cluster selector, capacity bar, filesystem table, backend IP grid |
| `weka-app-store-operator-chart/templates/crd.yaml` | Add `WarpCredential` CRD |
| `weka-app-store-operator-chart/templates/rbac.yaml` | Add Secret CRUD Role + RoleBinding scoped to App Store namespace |

---

## Acceptance Criteria

1. Administrator opens Settings. Credential Management appears first. All three type sections are empty with **[+ Add]** buttons.
2. Administrator adds an NGC key with display name "NGC Production". An inline form appears with Name pre-filled, admin enters the key, clicks Save. The row enters amber state, then within 30 seconds turns green showing "NGC Production [●] Ready [Delete]".
3. Verifying in the cluster: `kubectl get warpcredential ngc-production -n weka-app-store` exists with `KeyReady=True` and `DockerSecretReady=True`. `warp-ngc-production-apikey` and `warp-ngc-production-docker` secrets exist in the App Store namespace.
4. Administrator adds a second NGC credential "NGC Research". Both entries appear under the NVIDIA NGC section. Both have separate derived secrets (`warp-ngc-research-apikey`, `warp-ngc-research-docker`).
5. Administrator adds a WEKA Storage credential named "WEKA Cluster Primary" with username, token, and endpoint. `kubectl get secret warp-weka-cluster-primary-token -n weka-app-store` exists with keys `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, and `WEKA_API_ENDPOINT`. The `WarpCredential` status exposes `wekaEndpoint: "https://weka-cluster:14000"`. The `WEKA_API_USERNAME` value is visible; `WEKA_API_TOKEN` is not logged anywhere.
6. If `warp-ngc-production-docker` is manually deleted, the operator restores it within 30 seconds.
7. Administrator clicks Delete on "NGC Research". The row disappears. Derived secrets `warp-ngc-research-*` remain in the cluster.
8. A blueprint install page using `{{ credential_select(type="nvidia-ngc", field_name="ngc_credential") }}` renders a `<select>` with both "NGC Production" and "NGC Research" as options.
9. A blueprint install page using `{{ weka_storage_select() }}` renders a credential dropdown with the endpoint field pre-populated to `https://weka-cluster:14000`. Changing the credential selection updates the endpoint field to match the newly selected credential's stored endpoint.
10. If no credentials of a given type are stored, the macro renders a hint with a link to Settings instead of an empty select or broken field.
11. The old `/api/secret/nvidia` and `/api/secret/huggingface` endpoints return 404.
12. No key values are logged by the GUI or operator at any log level.
13. With a registered WEKA Storage credential, the Settings page shows the WEKA Storage Overview panel below Credential Management. It displays total capacity, used capacity (with percentage bar), available capacity, a filesystem table sorted by utilisation descending (names only, no UUIDs), and a grid of backend node IPs.
14. Filesystems at ≥ 90% utilisation have their bar rendered in amber/orange.
15. Clicking **[↺ Refresh]** fetches fresh data from the WEKA API within 5 seconds.
16. With two WEKA credentials stored, a dropdown appears in the panel header. Switching the selection updates all panel data to reflect the chosen cluster.
17. If the WEKA API is unreachable, the panel shows an error message. No stale data is displayed.

---

## Open Questions

1. ~~**WEKA Storage token shape.**~~ Resolved: the WEKA Settings form collects `Username`, `API Token`, and `Endpoint URL`. All three are stored in the credential Secret (`WEKA_API_USERNAME` + `WEKA_API_TOKEN` + `WEKA_API_ENDPOINT`). The endpoint is additionally surfaced in `status.wekaEndpoint`. The `weka_storage_select` macro pre-populates the endpoint field.
2. **WEKA auth token type.** The WEKA REST API uses `POST /api/v2/login` with `{username, password}` issuing a 5-minute `access_token` and a 1-year `refresh_token`. The App Store stores the long-lived token and handles refresh transparently. If the target cluster supports dedicated org/API tokens usable as static Bearer credentials, those are preferred — implementation must verify at `https://<cluster>:14000/api/v2/docs`.
3. **RBAC for credential submission.** Should the `/api/credentials` POST/DELETE endpoints be gated by a Kubernetes RBAC check (e.g. requiring a `ClusterAdmin` bearer token in the request), or is access to the App Store GUI on the cluster network considered sufficient authorization for this phase?
4. **Slug collision handling.** If an admin adds two credentials with display names that slugify identically (e.g. "NGC Production" and "NGC  Production"), the backend appends a short suffix. Is a silent suffix acceptable or should the form reject duplicate display names outright?
