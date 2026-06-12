---
phase: 25-blueprint-credential-selector-sdk
plan: 02
subsystem: ui
tags: [jinja2, tailwind, html, macros, credentials, sdk]

# Dependency graph
requires:
  - phase: 25-blueprint-credential-selector-sdk-plan-01
    provides: _get_credentials_by_type helper and credentials_by_type template context injection in blueprint_detail
provides:
  - _credential_macros.html Jinja2 macro library with credential_select and weka_storage_select
  - blueprint_neuralmesh-aidp.html reference example demonstrating both macros in the Configure card
affects:
  - 25-blueprint-credential-selector-sdk-plan-03 (tests assert _get_credentials_by_type populates template context, NOT macro rendering)
  - future blueprint template authors (macro SDK consumers)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Jinja2 macro library file (_credential_macros.html) imported via {% from ... import %} — no {% include %}, no {% extends %}"
    - "Form-control class string verbatim from settings.html:688 reused across all inputs and selects in macros"
    - "Inline <script> for DOM interaction inside Jinja2 macro output (warpSyncEndpoint)"
    - "Defensive credentials_by_type guard in macros: if credentials_by_type and credentials_by_type[type] — works when context key absent"

key-files:
  created:
    - app-store-gui/webapp/templates/_credential_macros.html
  modified:
    - app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html

key-decisions:
  - "Macro file uses non-whitespace-stripping {% macro %} / {% endmacro %} form per UI-SPEC action contract"
  - "credential_select guard uses 'credentials_by_type and credentials_by_type[type]' for defensive rendering when context key absent (Plan 01 backend injection not yet in scope for this plan)"
  - "Option A placement for import directive: first line of file before <!DOCTYPE html> (no {% extends %} conflict)"
  - "Inert fields design: new weka_credential, weka_endpoint, ngc_credential fields submit with form but backend ignores them until follow-on work per D-10"
  - "Plan 03 tests do NOT need to render the macros — they only assert _get_credentials_by_type populates the template context correctly"

patterns-established:
  - "Macro SDK pattern: {% from '_credential_macros.html' import <macro_name> %} at top of blueprint template, then {{ <macro_name>(...) }} inside form"
  - "Empty-state copy pattern: No {type} credential configured. <a href=/settings#credentials>Add one in Settings.</a>"

requirements-completed:
  - SDK-01
  - SDK-02
  - SDK-03
  - SDK-04

# Metrics
duration: 8min
completed: 2026-06-12
---

# Phase 25 Plan 02: Blueprint Credential Selector SDK — Macros Summary

**Jinja2 macro SDK delivered: `_credential_macros.html` with `credential_select` and `weka_storage_select` macros, plus AIDP reference example using both inside the Configure card**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-12T04:27:00Z
- **Completed:** 2026-06-12T04:35:58Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

- Created `_credential_macros.html` (76 lines): two Jinja2 macros with locked HTML output per UI-SPEC §Component Contract
- `credential_select(type, field_name, label=None, required=True)`: populated `<select>` from `credentials_by_type[type]` or empty-state hint linking to `/settings#credentials`
- `weka_storage_select(credential_field, endpoint_field, label)`: credential dropdown with `data-endpoint` on every `<option>`, sibling `<input type="url">`, and inline `warpSyncEndpoint(selectEl)` script wired via `onchange="warpSyncEndpoint(this)"`
- Updated `blueprint_neuralmesh-aidp.html`: added `{% from '_credential_macros.html' import credential_select, weka_storage_select %}` at line 1, inserted both macro calls at lines 315-316 (between hidden namespace input and Deploy button)
- Submit handler at lines 343-367 untouched per D-10 — new fields are inert in backend until follow-on work

## Task Commits

1. **Task 1: Create `_credential_macros.html` with both macros** — `59e7fd3` (feat)
2. **Task 2: Update `blueprint_neuralmesh-aidp.html` Configure card** — `ca06b5b` (feat)

## Files Created/Modified

- `app-store-gui/webapp/templates/_credential_macros.html` — NEW (76 lines): Jinja2 macro library defining `credential_select` and `weka_storage_select` with locked HTML output, verbatim Tailwind class strings, inline `warpSyncEndpoint` script, no `<style>` block, no `| safe` filter
- `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html` — MODIFIED (+4 lines): macro import at line 1, two macro calls at lines 315-316 inside `<form id="deploy-form">`

## Macro Signatures (declared with non-whitespace-stripping form)

```
{% macro credential_select(type, field_name, label=None, required=True) %}
{% macro weka_storage_select(credential_field, endpoint_field, label) %}
```

## Verbatim Class String Reused from settings.html:688

All three form controls (`credential_select` `<select>`, `weka_storage_select` `<select>`, `weka_storage_select` `<input type="url">`) use:
```
w-full px-3 py-2 rounded-md bg-gray-800/70 border border-white/10 focus:outline-none focus:ring-2 focus:ring-[var(--weka-purple)] text-sm
```

## AIDP Insertion Site (post-edit line numbers)

- Line 1: `{% from '_credential_macros.html' import credential_select, weka_storage_select %}`
- Line 315: `{{ weka_storage_select(credential_field="weka_credential", endpoint_field="weka_endpoint", label="WEKA NeuralMesh") }}`
- Line 316: `{{ credential_select(type="nvidia-ngc", field_name="ngc_credential", label="NVIDIA NGC API Key") }}`

## Inert Fields Note (D-10)

`weka_credential`, `weka_endpoint`, and `ngc_credential` are submitted with the deploy form but the existing `/deploy-stream/{{ name }}?namespace=...` handler ignores them. Wiring credential values into the deploy-stream backend is explicitly out of scope for Phase 25 per D-10. Phase 25 is the SDK delivery; the AIDP deploy workflow picks up these fields in follow-on work.

## Macros Are Logic-Free (Plan 01 ready-filter)

The macros do NOT need any `selectattr('ready')` or `{% if cred.ready %}` guard. Plan 01's `_get_credentials_by_type` helper applies the ready-filter at the helper level (single source of truth) — every dict in each list is ready=True by the time the macro consumes it. The empty branch fires when the list is empty (no CRs, none ready, or K8s unreachable).

## Plan 03 Test Scope Note

Plan 03 tests do NOT need to render the macros; they only assert that `_get_credentials_by_type` populates the template context correctly (i.e., the helper groups ready credentials by type and falls back to empty lists on K8s errors).

## Decisions Made

- Non-whitespace-stripping `{% macro %}` form per UI-SPEC action contract (compliance, not preference)
- Defensive `credentials_by_type and credentials_by_type[type]` guard so macros render the empty branch instead of raising KeyError when Plan 01's context injection is absent
- Import directive placed at line 1 (Option A per plan) — no `{% extends %}` conflict in this template
- `data-endpoint="{{ cred.endpoint or '' }}"` uses `or ''` so the HTML attribute is always emitted even when a WEKA credential has no endpoint set

## Deviations from Plan

None — plan executed exactly as written. Both macro signatures, all class strings, empty-state copy, and AIDP insertion points match the locked UI-SPEC verbatim.

## Issues Encountered

None.

## Security Notes

- No `| safe` filter used anywhere in `_credential_macros.html`
- No `{% autoescape false %}` directive
- Jinja2 autoescape (enabled by default for `.html` via `Jinja2Templates(directory=TEMPLATES_DIR)`) neutralizes XSS via `cred.displayName`, `cred.name`, and `cred.endpoint` — threat T-25-05 and T-25-06 mitigated as specified in threat model

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. All surfaces (credential display in `<option>` elements) were pre-scoped in the plan's threat model.

## Next Phase Readiness

- Plan 03 (tests) can now test `_get_credentials_by_type` context injection independently of macro rendering
- Blueprint template authors can add credential selection to any blueprint by: `{% from '_credential_macros.html' import credential_select %}` and calling `{{ credential_select("nvidia-ngc", "ngc_credential") }}`
- The four remaining blueprint templates (nvidia-vss, openfold, glocomp-aurora, tokenvisor-enterprise) are future work per D-09

## Self-Check: PASSED

- `app-store-gui/webapp/templates/_credential_macros.html` exists: PASS
- `app-store-gui/webapp/templates/blueprint_neuralmesh-aidp.html` modified: PASS
- Task 1 commit `59e7fd3` exists: PASS
- Task 2 commit `ca06b5b` exists: PASS
- All acceptance criteria verified via automated checks: PASS
- `python -m py_compile app-store-gui/webapp/main.py` exits 0: PASS

---
*Phase: 25-blueprint-credential-selector-sdk*
*Completed: 2026-06-12*
