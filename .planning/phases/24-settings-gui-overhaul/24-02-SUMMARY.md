---
phase: 24-settings-gui-overhaul
plan: "02"
subsystem: app-store-gui
tags: [settings-gui, credentials, traffic-light, polling, forms, xss-defense, vanilla-js]
dependency_graph:
  requires:
    - 24-01: data-* attribute contract (data-credential-list, data-add-form, data-add-form-container, data-cred-name, data-cred-ready, data-empty-state)
    - phase-23: POST /api/credentials, DELETE /api/credentials/<name>, GET /api/credentials?namespace=
  provides:
    - Credential Management inline JS block in settings.html <script> element
    - renderCredentialRow(li, cred) — DOM renderer for green/amber/red states
    - startCredentialPoll(name) / stopCredentialPoll(name) — per-row polling lifecycle
    - pollIntervals Map — active amber-row intervals (Plan 03 MUST NOT add a second visibilitychange listener)
    - wireDeleteButton(btn, name, displayName) — confirm + DELETE + fade-out
    - closeAllAddForms() / openAddForm(type) — single-open-form invariant
    - submitAddForm(event, form, type) — FormData POST with inline error display
    - hydrateInitialRows() — DOMContentLoaded reconcile of server-rendered placeholders
    - Module-level esc() helper (hoisted from renderBlueprints scope)
  affects:
    - app-store-gui/webapp/templates/settings.html (JS block appended; esc() hoisted)
tech_stack:
  added: []
  patterns:
    - Per-row polling Map (pollIntervals, pollStartedAt) per PATTERNS §Pattern 6
    - Page Visibility API pause/resume (UI-SPEC line 267)
    - beforeunload interval cleanup (UI-SPEC line 346)
    - HTML5 Constraint Validation API (checkValidity() for WEKA endpoint URL)
    - FormData POST matching Phase 23 Form(...) server-side signatures
    - CSS opacity transition for row delete (150ms)
    - XSS mitigation via module-level esc() on all cred.* DOM interpolations
key_files:
  created: []
  modified:
    - app-store-gui/webapp/templates/settings.html
decisions:
  - "Hoisted esc() to module level (was nested in renderBlueprints) — required for credential management XSS mitigation; renderBlueprints still works unchanged"
  - "Tasks 1 and 2 committed in a single commit — the DOMContentLoaded wire-up in Task 1 references polling helpers from Task 2 making them mutually dependent"
  - "pollCredentialOnce reads displayName for timeout error from the existing DOM text node — avoids re-fetching just to get displayName for a timeout message"
  - "CSS.escape() used for data-cred-name selector queries — defensive against names with special characters"
metrics:
  duration: "~20m"
  completed_date: "2026-06-12"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 1
---

# Phase 24 Plan 02: Credential Management Client-Side Behaviour

Added the full credential management JS inside the existing `<script>` element in `settings.html`: traffic-light state machine (green/amber/red), 2-second per-row polling with 30-second timeout, inline add forms with single-open-form invariant, browser-confirm delete with fade-out, and inline form-level error display. Server-rendered rows from Plan 01 are hydrated on DOMContentLoaded.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Hydrate server-rendered rows, inline add form scaffolding, traffic-light DOM renderer | 11f7bc3 | app-store-gui/webapp/templates/settings.html |
| 2 | Form validation, submit, polling state machine, visibility/beforeunload lifecycle | 11f7bc3 | app-store-gui/webapp/templates/settings.html |

(Tasks 1 and 2 are in the same commit — the polling helpers from Task 2 are referenced in the Task 1 DOMContentLoaded wire-up, making them inseparable at commit time.)

## New JS Module-Level Constants and Helpers

| Symbol | Type | Purpose | Plan 03 notes |
|--------|------|---------|---------------|
| `DETECTED_NAMESPACE` | `const string` | Jinja2-emitted namespace for all API calls | Reusable by Plan 03 for `/api/weka/overview` calls |
| `POLL_MS` | `const number` | 2000 — polling interval in ms | Plan 03 MUST NOT add credential polling (per-row only) |
| `POLL_TIMEOUT_MS` | `const number` | 30000 — amber timeout in ms | — |
| `CREDENTIAL_TYPES` | `const string[]` | `['nvidia-ngc','huggingface','weka-storage']` | Fixed render order |
| `TYPE_META` | `const object` | label + fields per type | — |
| `pollIntervals` | `Map<string,id>` | Active amber-row interval IDs | **Plan 03 MUST NOT register a second `visibilitychange` listener** — if Plan 03 ever needs polling, hook into the existing listener or extend this Map |
| `pollStartedAt` | `Map<string,ms>` | Per-row poll start epoch | — |
| `esc` | `function` | HTML-escape for innerHTML (WR-04) | Already at module level; reuse for WEKA overview rendering in Plan 03 |
| `truncate(s,n)` | `function` | Truncate string with ellipsis | — |
| `renderCredentialRow(li, cred)` | `function` | Renders green/amber/red DOM into a `<li>` | Plan 03 does not call this directly but the DOM shape it produces is documented below |
| `wireDeleteButton(btn, name, displayName)` | `function` | Attaches confirm + DELETE + fade-out | — |
| `hydrateInitialRows()` | `async function` | DOMContentLoaded reconcile | Called once; Plan 03 should not re-call |
| `closeAllAddForms()` | `function` | Collapses all open add forms | Plan 03 may call if it opens its own form-like UI |
| `openAddForm(type)` | `function` | Opens inline add form for given type | — |
| `updateSaveState(form)` | `function` | Enables/disables Save button | — |
| `submitAddForm(event, form, type)` | `async function` | Handles form submit | — |
| `startCredentialPoll(name)` | `function` | Starts per-row amber polling | — |
| `stopCredentialPoll(name)` | `function` | Stops and cleans up per-row polling | — |
| `pollCredentialOnce(name)` | `async function` | Single poll tick: green/red/timeout/amber | — |

## DOM Shape of Credential Row States

### Green state
```html
<li class="flex items-center justify-between py-2 border-b border-white/5 last:border-b-0"
    data-cred-name="{name}" aria-label="Credential {displayName}">
  <div class="flex items-center gap-2">
    <span class="status-dot bg-green-500" aria-hidden="true"></span>
    <span class="text-sm">{displayName}</span>
    <span class="text-xs px-2 py-0.5 rounded-full border border-green-500/40 bg-green-500/10 text-green-400" aria-live="polite">Ready</span>
  </div>
  <button class="px-3 py-1 rounded-md text-sm border border-red-400/40 text-red-300 hover:bg-red-500/10 ..."
          data-cred="{name}" aria-label="Delete credential {displayName}">Delete</button>
</li>
```
No `<input>` elements (GUI-06).

### Amber state
```html
<li class="flex items-center justify-between py-2 border-b border-white/5 last:border-b-0"
    data-cred-name="{name}" aria-label="Credential {displayName}">
  <div class="flex items-center gap-2">
    <span class="status-dot bg-amber-500 animate-pulse" aria-hidden="true"></span>
    <span class="text-sm">{displayName}</span>
    <span class="text-xs px-2 py-0.5 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-400" aria-live="polite">Verifying…</span>
  </div>
</li>
```
No Delete button (UI-SPEC line 156).

### Red state
```html
<li class="flex items-center justify-between py-2 border-b border-white/5 last:border-b-0"
    data-cred-name="{name}" aria-label="Credential {displayName}">
  <div class="flex items-center gap-2">
    <span class="status-dot bg-red-500" aria-hidden="true"></span>
    <span class="text-sm">{displayName}</span>
    <span class="text-red-400 text-xs ml-2" title="{full error}">{truncated error ≤120 chars}</span>
  </div>
  <button ...>Delete</button>
</li>
```

## Visibility / beforeunload Contract

- Single `visibilitychange` listener registered. When `document.hidden === true`, all intervals in `pollIntervals` are cleared (but NOT removed from the Map — the Map retains the names).
- On resume (`document.hidden === false`), new intervals are created for each name still in `pollIntervals` and stored back. The elapsed budget is NOT reset (elapsed = `Date.now() - pollStartedAt.get(name)`).
- **Plan 03 MUST NOT register a second `visibilitychange` listener.** Per UI-SPEC Performance Contract, the WEKA Overview is fetched only on initial render and on Refresh button click — no polling needed for Plan 03, so no listener conflict is expected.
- `beforeunload` clears all intervals via `pollIntervals.forEach(clearInterval)`. No Map cleanup needed (page is unloading).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Hoisted `esc()` to module level**
- **Found during:** Task 1 implementation
- **Issue:** The plan's `<interfaces>` block referenced `esc()` as already at module scope ("line 323"), but the actual file had `esc` declared inside the `renderBlueprints` function (line 414). The credential management code at module scope would have had no access to `esc()`, making the XSS mitigation impossible without redeclaring it.
- **Fix:** Moved the `const esc = s => ...` declaration to module level (before `renderBlueprints`) and removed the redundant nested declaration. `renderBlueprints` continues to work unchanged — it now references the module-level declaration via closure.
- **Files modified:** `app-store-gui/webapp/templates/settings.html`
- **Commit:** 11f7bc3

## Known Stubs

None. All credential management functionality is fully implemented. The Jinja2 comment `{# Plan 02 renders rows server-side; placeholder for now #}` at line 70 was a Plan 01 design annotation; it no longer reflects the actual code behavior (the JS now hydrates those rows with full DOM on DOMContentLoaded), but it is a comment, not a data stub.

## Threat Flags

No new trust boundary crossings beyond the plan's threat model.

| Threat ID | Status |
|-----------|--------|
| T-24-02-01 (XSS via DOM injection) | Mitigated — all cred.* interpolations use esc(); grep gate confirms 0 unescaped lines |
| T-24-02-02 (XSS via cred.error) | Mitigated — esc() on both innerHTML and title attribute |
| T-24-02-03 (Credential exposure via DOM) | Mitigated — password inputs cleared via closeAllAddForms() on Save success |
| T-24-02-04 (Credential exposure via URL) | Mitigated — POST via FormData; GET /api/credentials only carries namespace |
| T-24-02-05 (CSRF) | Accepted — same-origin, no CORS |
| T-24-02-06 (DoS via runaway polling) | Mitigated — per-row only, visibilitychange pause, 30s timeout |
| T-24-02-07 (race: delete during amber) | Mitigated — amber row has no Delete button |
| T-24-02-08 (open-redirect via endpoint) | Accepted — UX-only URL validation; backend applies SSRF guards |

## Self-Check

Files modified:
- `app-store-gui/webapp/templates/settings.html` — FOUND

Commits:
- 11f7bc3 — FOUND

## Self-Check: PASSED
