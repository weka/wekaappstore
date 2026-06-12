---
phase: 24-settings-gui-overhaul
plan: "01"
subsystem: app-store-gui
tags: [settings-gui, jinja2, credentials, page-order, warp-credentials]
dependency_graph:
  requires:
    - phase-23-plan-02: WarpCredential CRD and list_credentials() handler at main.py:759
    - phase-23: _build_credential_response_item() helper at main.py:717
  provides:
    - credentials_by_type Jinja2 context variable (nvidia-ngc, huggingface, weka-storage lists)
    - weka_storage_credentials Jinja2 context variable (ready weka-storage credentials only)
    - settings.html section shells for Credential Management and WEKA Storage Overview
    - data-* attribute contract for Plan 02 JS hydration
    - WEKA Overview placeholder container ids for Plan 03 fetch logic
  affects:
    - app-store-gui/webapp/main.py (settings_page route extended)
    - app-store-gui/webapp/templates/settings.html (page order restructured)
tech_stack:
  added: []
  patterns:
    - SDK-05 graceful-degradation (ApiException → empty list, /settings always HTTP 200)
    - Jinja2 auto-escape on credential displayName (T-24-01-01 mitigated, no |safe introduced)
    - asyncio.to_thread() for sync K8s API calls inside async route handler
    - Nested inner async function (_fetch_credentials) with ns closure (Pattern 12)
key_files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/webapp/templates/settings.html
decisions:
  - "Used locked-keys dict (not setdefault) for credentials_by_type — unknown credential types are discarded silently, not registered"
  - "WEKA Storage Overview section renders no-credential hint in else branch — avoids JS dependency for initial empty state"
  - "Plan 02 comment in credential row li is an intentional design annotation, not a data stub — rows render real displayName from server"
metrics:
  duration: "5m 15s"
  completed_date: "2026-06-12"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 2
---

# Phase 24 Plan 01: Settings Page Foundation — Credential Shells and Route Context

Extended the `/settings` route to fetch WarpCredential CRs and restructured the Settings page HTML order, placing Credential Management first and WEKA Storage Overview second, above all legacy sections.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend /settings route with credentials_by_type and weka_storage_credentials | f54eac9 | app-store-gui/webapp/main.py |
| 2 | Restructure settings.html section order — insert Credential Management and WEKA Storage Overview shells | 74d76f6 | app-store-gui/webapp/templates/settings.html |

## New Page Section Order

The locked top-to-bottom order in `/settings` HTML is now:

1. `<h2>Settings</h2>` (unchanged heading)
2. **Credential Management** (`id="credentials-section"`) — three sub-section cards
3. **WEKA Storage Overview** (`id="weka-overview-section"`) — shell with conditional hint or Plan 03 containers
4. Kubernetes Auth Status (moved from position 1 to 3)
5. Cluster Status (unchanged relative order)
6. Blueprint Uninstall (unchanged relative order)
7. Debug (unchanged relative order)

## New Jinja2 Context Keys

### `credentials_by_type: dict[str, list[dict]]`

Shape:
```python
{
    "nvidia-ngc": [...],    # list of _build_credential_response_item() dicts
    "huggingface": [...],   # list of _build_credential_response_item() dicts
    "weka-storage": [...],  # list of _build_credential_response_item() dicts
}
```

Keys are always present (locked — never grows via setdefault). Unknown credential types are silently discarded. Each item has the shape defined by `_build_credential_response_item()` at main.py:717: `name`, `namespace`, `type`, `displayName`, `ready`, `lastSyncTime`, `derivedSecrets`, `error` (optional), `dockerSecretReady` (ngc only), `endpoint` (weka-storage only).

### `weka_storage_credentials: list[dict]`

Filtered subset of `credentials_by_type["weka-storage"]` where `ready == True`. Used by:
- GUI-10 visibility test: if empty, show "No WEKA Storage credential configured. Add one above." hint
- GUI-11 dropdown (Plan 02): multiple vs single-label rendering for WEKA Overview credential selector

## data-* Attribute Contract (for Plans 02 and 03)

### Credential section wrappers

| Attribute | Element | Value | Plan consumer |
|-----------|---------|-------|---------------|
| `id="credentials-section"` | `<section>` | — | anchor / DOM query |
| `id="ngc-credentials"` | sub-section `<div>` | — | Plan 02 JS |
| `id="hf-credentials"` | sub-section `<div>` | — | Plan 02 JS |
| `id="weka-credentials"` | sub-section `<div>` | — | Plan 02 JS + GUI-10 link target |
| `data-add-form="<type>"` | `<button>` | nvidia-ngc / huggingface / weka-storage | Plan 02: open inline add form |
| `data-credential-list="<type>"` | `<ul>` | nvidia-ngc / huggingface / weka-storage | Plan 02: insert/remove rows |
| `data-cred-name="<name>"` | `<li>` | DNS-1123 slug (metadata.name) | Plan 02: detect existing rows for DELETE wiring |
| `data-cred-ready="true|false"` | `<li>` | boolean as lowercase string | Plan 02: hydrate to green/amber/red state |
| `data-empty-state="<type>"` | empty `<li>` | nvidia-ngc / huggingface / weka-storage | Plan 02: remove when first credential added |
| `data-add-form-container="<type>"` | `<div hidden>` | nvidia-ngc / huggingface / weka-storage | Plan 02: inject inline add form HTML |

### WEKA Overview containers (for Plan 03)

| Element id | Purpose |
|------------|---------|
| `weka-overview-controls` | Plan 03 injects credential dropdown / static label + Refresh button + Last updated timestamp |
| `weka-overview-loading` | Plan 03 shows during initial/refresh fetch (replaces "Loading WEKA cluster…" placeholder) |
| `weka-overview-error` | Plan 03 shows on fetch error (red banner) |
| `weka-overview-success` | Plan 03 shows on successful fetch (full capacity + filesystem + node grid) |

## Graceful Degradation (T-24-01-03 / SDK-05)

`_fetch_credentials()` catches both `ApiException` and `Exception` and returns `[]`. The `/settings` route always returns HTTP 200 regardless of Kubernetes API reachability. With zero credentials (K8s unreachable or empty namespace), the page renders three "(none stored)" empty states and the "No WEKA Storage credential configured." hint.

## Security Threat Mitigations

| Threat ID | Status |
|-----------|--------|
| T-24-01-01 (XSS via displayName) | Mitigated — Jinja2 auto-escape; `grep -c '|safe'` returns 0 |
| T-24-01-02 (info disclosure) | Mitigated — _build_credential_response_item() whitelist reused verbatim |
| T-24-01-03 (DoS via K8s outage) | Mitigated — _fetch_credentials() always returns list, never raises |
| T-24-01-04 (future JS XSS) | Accepted in this plan — Plan 02/03 own their esc() usage |
| T-24-01-05 (CSRF) | Accepted — GET-only route, no state changes |

## Deviations from Plan

None — plan executed exactly as written.

The PATTERNS.md Pattern 12 code block used `setdefault` in its example snippet, but the Task 1 action text explicitly prohibited `setdefault` (locked keys rule). The task action took precedence; the implementation uses the locked-dict pattern as specified in the task action.

## Known Stubs

None that block the plan's goal. The comment `{# Plan 02 renders rows server-side; placeholder for now #}` in settings.html is a design annotation — the `<li>` renders the real `cred.displayName` value from the server context. Row hydration with full state-row HTML (dot + badge + Delete button) is deferred to Plan 02 as specified.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary crossings beyond what the plan's threat model already covers.

## Self-Check

Files modified:
- `app-store-gui/webapp/main.py` — FOUND
- `app-store-gui/webapp/templates/settings.html` — FOUND

Commits:
- f54eac9 — FOUND
- 74d76f6 — FOUND

## Self-Check: PASSED
