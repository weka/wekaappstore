---
phase: 30-wizard-stepper-live-progress
plan: "02"
subsystem: frontend-wizard
tags: [welcome-html, stepper, mui, wizard, validation, react]
dependency_graph:
  requires: [30-01]
  provides: [WIZ-01, WIZ-02, WIZ-03, WIZ-04, WIZ-05, WIZ-06, WIZ-07, WIZ-08, handleInstall-handoff-contract]
  affects: [app-store-gui/webapp/templates/welcome.html]
tech_stack:
  added: []
  patterns: [MUI-Stepper-5-step, client-side-validation-on-next-click, in-memory-secrets-only]
key_files:
  modified:
    - path: app-store-gui/webapp/templates/welcome.html
      change: "Replaced single-button prerequisite hard-block with 5-step MUI Stepper wizard (Node Prerequisites → Quay Credentials → WEKA Connection → WEKA Credentials → Review) with client-side validation, masked Review summary, namespace selector, buildVariables() + handleInstall entry point (30-03 handoff)"
decisions:
  - "D-01 applied: Stepper/Step/StepLabel/TextField/Checkbox/FormControlLabel destructured from existing @mui/material@5.15.14 UMD bundle — no new CDN script tags"
  - "D-02 applied: quay_password and weka_password live in React useState only; localStorage restricted to selectedNamespace"
  - "D-03 applied: 5-step order Node Prerequisites → Quay Credentials → WEKA Connection → WEKA Credentials → Review; progressActive gates stepper vs progress view"
  - "D-04 applied: validateStep() fires on Next/Submit click, blocks forward navigation with specific inline errors; regexes ^[^:]+:\\d+$ for endpoints, ^v?\\d+\\.\\d+(\\.\\d+)*$ for version tags"
  - "D-08 applied: buildVariables() returns exact x-variable keys from D-08 table; quay_dockerconfigjson/join_ip_ports_list/endpoints_csv excluded (server-derived)"
  - "D-09 applied: Review step masks quay_password and weka_password as bullets (••••••••); quay_dockerconfigjson never shown; namespace selector writes to localStorage"
  - "WIZ-08 applied: handleInitialize hard-block (wekaOperatorInstalled/wekaCsiInstalled early-return) removed"
metrics:
  duration: "~25 minutes"
  completed_date: "2026-06-25"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 1
---

# Phase 30 Plan 02: Wizard Stepper (5-Step Form + Validation + Masked Review) Summary

**One-liner:** Replaced single-button prerequisite hard-block in welcome.html with a 5-step MUI Stepper wizard collecting all WEKA install variables with client-side validation, masked Review step, and named handleInstall/progressActive handoff contract for 30-03.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add wizard state, MUI imports, buildVariables(), handleInstall + progressActive | 13f8373 | app-store-gui/webapp/templates/welcome.html |
| 2 | Render 5-step Stepper with steps 1-4 (prereqs, quay, connection, credentials) | 13f8373 | app-store-gui/webapp/templates/welcome.html |
| 3 | Review step + inline validation gate | 13f8373 | app-store-gui/webapp/templates/welcome.html |

Note: Tasks 1-3 were implemented as a coherent single-file rewrite committed atomically — all three logical layers (state/hooks, stepper render, validation/review) are interleaved in the same single-file React component.

## What Was Built

A complete 5-step MUI Stepper wizard inside `welcome.html`'s `WelcomeApp` component:

### Step 1 — Node Prerequisites (WIZ-02)
- Copy-paste `KubeletConfiguration` snippet (`cpuManagerPolicy: static`, `strict-cpu-reservation`, hugepage config) in a styled `<pre>` block
- MUI `Checkbox` via `FormControlLabel` labeled "I have applied node prerequisites on all worker nodes"
- Next button is `disabled={activeStep === 0 && !nodePrereqAck}` — blocked until checkbox checked

### Step 2 — Quay Credentials (WIZ-03)
- `TextField` for `quay_username`
- `TextField type="password"` for `quay_password`
- `TextField` for `operator_version` (default `v1.13.0`)
- Validation: username non-empty, password non-empty, operator_version matches `^v?\d+\.\d+(\.\d+)*$`

### Step 3 — WEKA Connection (WIZ-04)
- `TextField` for `join_ip_ports` (helper: comma-delimited `host:port` list)
- `TextField` for `weka_image_version`
- `FormControl`/`Select`/`MenuItem` dropdown for `weka_endpoint_scheme` (http|https, default `http`)
- Validation: each endpoint matches `^[^:]+:\d+$`, image version matches version regex

### Step 4 — WEKA Credentials (WIZ-05)
- `TextField` for `weka_org` (default `Root`)
- `TextField` for `weka_username`
- `TextField type="password"` for `weka_password`
- Validation: username non-empty, password non-empty

### Step 5 — Review (WIZ-06, D-09)
- Read-only summary table; `quay_password` and `weka_password` displayed as `••••••••`
- `quay_dockerconfigjson` not shown at all
- Namespace selector reusing `FormControl`/`Select`/`MenuItem` pattern writing to `localStorage`
- "Install" button runs `validateStep` for all steps 2-4, then calls `handleInstall` on clean

### Validation (WIZ-07, D-04)
- `validateStep(stepIndex)` fires on Next/Submit click only
- Blocks forward navigation; sets `fieldErrors` for inline `error`/`helperText` display on each TextField
- Specific error messages per field

### Hard-block removed (WIZ-08)
- The `if (wekaOperatorInstalled !== true || wekaCsiInstalled !== true)` early-return in `handleInitialize` has been removed
- Status dots remain as informational display

### Handoff contract for 30-03
- `buildVariables()` — returns full D-08 x-variable dict
- `handleInstall` — named function; calls `buildVariables()`, sets `showProgress=true`, `progressActive=true`
- `progressActive` — boolean state that gates the stepper region (`{!progressActive && <Box>...</Box>}`)
- `showProgress` — existing boolean state, reused as-is

## Deviations from Plan

### Implementation approach

**Observation:** All three tasks modify the same single-file React component (`welcome.html`). Splitting them into three separate git commits with git-add-patch would have produced confusingly partial intermediate states (e.g., commit 2 would have `Stepper` renders referencing `validateStep` that only existed after commit 3). The three task logical slices are architecturally inseparable in a single-file React component.

**Decision:** Implemented as one coherent file rewrite committed atomically at `13f8373`. All three task acceptance criteria are fully met in that commit. This matches CLAUDE.md "Surgical Changes" — the minimum change that satisfies the requirements.

## Security — Threat Register Mitigations

| Threat ID | Mitigation | Verified |
|-----------|-----------|---------|
| T-30-01 | quay_password/weka_password in React useState only; `localStorage.setItem` references only `selectedNamespace` | grep confirms |
| T-30-02 | Review step renders `••••••••` for both passwords; `quay_dockerconfigjson` absent from Review DOM | grep confirms |
| T-30-03 | `validateStep()` implements D-04 regexes as sole gate; client-side enforcement confirmed | source review |
| T-30-04 | No new CDN script tags; `<script src=` count unchanged at 7 | grep confirms |

## Known Stubs

The `progressActive` branch in the left column currently renders a placeholder message:
- **File:** `app-store-gui/webapp/templates/welcome.html`
- **Location:** `{progressActive && <Box>... "The install progress view will be wired in the next phase (30-03)" ...</Box>}`
- **Reason:** Intentional per plan spec — the install progress SSE view is the entire scope of plan 30-03. `handleInstall` body is a minimal entry point (flips flags); 30-03 replaces it with the EventSource wiring.

## Threat Flags

None. No new network endpoints, auth paths, or schema changes introduced. This is a pure frontend change.

## Self-Check: PASSED

- `app-store-gui/webapp/templates/welcome.html` committed at 13f8373
- `python -m py_compile app-store-gui/webapp/main.py operator_module/main.py` — passed
- `<script src=` count = 7 (unchanged)
- `{% raw %}` count = 46 (all sx={{ properly guarded)
- `buildVariables` present (2 occurrences: definition + call)
- `handleInstall` present (3 occurrences: definition + call + comment)
- `progressActive` state present (4 occurrences)
- Hard-block string `you need to install the WEKA Operator and WEKA CSI driver first` = 0 occurrences
- `••••••••` mask = 2 occurrences (quay_password, weka_password in Review)
- `quay_dockerconfigjson` not referenced in Review summary
- Regex `^[^:]+:\d+$` confirmed present
- Regex `^v?\d+\.\d+` confirmed present
