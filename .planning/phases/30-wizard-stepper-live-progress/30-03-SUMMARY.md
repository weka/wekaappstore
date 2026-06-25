---
phase: 30-wizard-stepper-live-progress
plan: "03"
subsystem: frontend-progress
tags: [welcome-html, sse, progress, cluster-init, redirect, retry, react]
dependency_graph:
  requires: [30-01, 30-02]
  provides: [PROG-01, PROG-03, D-10, D-11]
  affects: [app-store-gui/webapp/templates/welcome.html]
tech_stack:
  added: []
  patterns: [SSE-onmessage-React-state, unified-stage-list, cluster-init-auto-chain, Ready-gated-redirect]
key_files:
  modified:
    - path: app-store-gui/webapp/templates/welcome.html
      change: "Replaced handleInstall stub with full EventSource SSE consumer (app-store-install + cluster-init chain); added stages/installError state, stageColor() phase→color map, per-stage progress list, inline failure+retry, Ready-gated /cluster-status redirect"
decisions:
  - "D-05 applied: handleInstall opens EventSource to /deploy-stream?app_name=app-store-install with variables=JSON.stringify(buildVariables()) on the default message channel (es.onmessage); no addEventListener"
  - "D-06 applied: stageColor(phase) maps ready/healthy→green, failed/error→red, installing/upgrading→yellow, default→grey using MUI sx objects"
  - "D-07 applied: installError rendered inline in Alert with Retry Button; both complete.ok===false and error event paths covered; Retry clears installError and re-invokes handleInstall"
  - "D-10 applied: openClusterInitStream() called from inside the app-store-install complete handler when ok!==false; cluster-init stages appended to the same unified stages list"
  - "D-11 applied: cluster-init complete.ok===true triggers fetch('/cluster-status') then window.location.href = data.redirect_url || '/'; no poll-until-ready loop"
  - "D-12 confirmed: no ClusterInitMiddleware change needed — /deploy-stream and /welcome already exempt"
  - "Retry on cluster-init failure re-runs handleInstall (restarts both CRs): both app-store-install and app-store-cluster-init are idempotent operator upserts (Phase 27 D-09); acceptable for v8.0"
metrics:
  duration: "~12 minutes"
  completed_date: "2026-06-25"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 1
---

# Phase 30 Plan 03: SSE Progress View, Cluster-Init Chain, Failure+Retry Summary

**One-liner:** Wired handleInstall with a live per-stage SSE progress view (init/component/complete/error), automatic cluster-init chain after app-store-install Ready, inline failure+retry, and Ready-gated /cluster-status redirect — completing the two-phase install experience.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Stage-list progress view + app-store-install SSE consumer (PROG-01) | 1139287 | app-store-gui/webapp/templates/welcome.html |
| 2 | Failure + Retry (PROG-03) and cluster-init chain + Ready-gated redirect (D-10/D-11) | 1139287 | app-store-gui/webapp/templates/welcome.html |

Note: Tasks 1 and 2 were implemented as a single coherent commit — the stage list rendering (Task 1) and the chain/retry/redirect wiring (Task 2) are architecturally inseparable in the same single-file React component. Both task acceptance criteria are fully met in commit 1139287.

## What Was Built

### New state

- `stages` — `React.useState([])`: array of `{ name, phase, message }` objects; seeded from SSE `init` events and updated by `component` events for both app-store-install and cluster-init.
- `installError` — `React.useState(null)`: string|null; set on `complete.ok===false` or `error` events from either stream; cleared on Retry.

### stageColor(phase) — D-06 color map

Maps operator component phase values to MUI `sx` color objects (border/bgcolor/color):
- `ready`/`healthy` → green (`rgba(16,185,129,…)`)
- `failed`/`error` → red (`rgba(239,68,68,…)`)
- `installing`/`upgrading` → yellow (`rgba(251,191,36,…)`) — matches blueprint.html sectionClass yellow for in-progress
- default/`pending` → grey (`rgba(31,41,55,…)`)

### handleInstall — full SSE consumer (D-05, PROG-01)

Replaced the 30-02 stub body with:
1. Re-validation of all wizard steps (errs1/2/3 merged) — unchanged gate
2. Sets `showProgress=true`, `progressActive=true`, clears `installError` and `stages`
3. Opens `EventSource` to `/deploy-stream?app_name=app-store-install&variables=<JSON>` (default `message` channel, `es.onmessage`)
4. Handles all four event types:
   - `init` → seeds `stages` from `msg.items` (10 components for app-store-install)
   - `component` → updates matching stage phase/message; appends unknown names (mirrors blueprint.html:315-321)
   - `complete` → `ok!==false`: marks all stages ready, calls `openClusterInitStream()`; `ok===false`: sets `installError`
   - `error` → sets `installError`, closes stream

### openClusterInitStream() — D-10 cluster-init chain

Called automatically from the app-store-install `complete` success branch:
1. Opens `EventSource` to `/deploy-stream?app_name=cluster-init&namespace=${selectedNamespace}` (no variables param)
2. Handles same four event types; `init` appends stages to existing list (single unified list)
3. `component` events handled identically — tolerates both multi-component and single-stage cases
4. `complete.ok!==false` → marks all stages ready, fetches `/cluster-status?namespace=${selectedNamespace}`, navigates to `data.redirect_url || '/'` (D-11)
5. `complete.ok===false` or `error` → sets `installError`; Retry re-runs full `handleInstall` (idempotent)

### Progress view JSX (left column, replaces 30-02 stub)

When `progressActive` is true:
- Header shows "Installation Failed" if `installError` is set, else "Installing WEKA Storage Stack..."
- `stages.length > 0`: renders per-stage `Box` list with `stageColor(stage.phase)` as `sx` spread; `title={stage.message}` for tooltip; stage name (monospace) + phase (capitalized) on each row
- `stages.length === 0 && !installError`: indeterminate `LinearProgress` + "Connecting..." message while stream initializes
- `installError`: inline MUI `Alert severity="error"` (variant="outlined") with `AlertTitle` and Retry `Button` whose `onClick` clears `installError` and re-invokes `handleInstall`

## Deviations from Plan

### Atomic commit for Tasks 1+2

Tasks 1 and 2 both modify the same JSX render function and the same `handleInstall`/`openClusterInitStream` functions in a single-file React component. Splitting them would produce a partial intermediate state (Task 1 commit has stages rendering but no cluster-init chain or retry). Committed atomically at 1139287. All acceptance criteria for both tasks are met.

### Retry re-runs handleInstall on cluster-init failure

The plan states "Retry re-runs handleInstall — acceptable since both CRs are idempotent; document the choice in the SUMMARY." Implemented as specified. When cluster-init fails, the Retry Button clears `installError` and calls `handleInstall()`, which restarts both streams from the beginning. This is correct because: (a) the app-store-install CR upsert is idempotent (Phase 27 D-09), (b) the operator reconciles to the desired state regardless, (c) the risk of re-running an already-Ready app-store-install is just a quick re-poll that emits `complete.ok=true` again and re-chains cluster-init.

## Security — Threat Register Mitigations

| Threat ID | Mitigation | Verified |
|-----------|-----------|---------|
| T-30-05 | EventSource secrets in URL query string — accepted (documented); matches battle-tested blueprint.html pattern | confirmed |
| T-30-06 | installError displays msg.message verbatim because _redact_secrets already strips secrets server-side (30-01 confirmed) | grep confirms no client-side secret rendering |
| T-30-07 | Retry reads buildVariables() from in-memory React form state only; no localStorage write for secrets | grep confirms localStorage only references selectedNamespace |
| T-30-08 | Every terminal branch (complete, error, onerror) calls es.close() in both streams | source review |

## Known Stubs

None. The `progressActive` placeholder stub from 30-02 ("The install progress view will be wired in the next phase (30-03)") has been replaced with the full implementation. No stubs remain that prevent the plan's goal from being achieved.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes introduced. This is a pure frontend change to `welcome.html`.

## Self-Check: PASSED

- `app-store-gui/webapp/templates/welcome.html` committed at 1139287
- `python -m py_compile app-store-gui/webapp/main.py operator_module/main.py` — PASSED
- `grep -c EventSource welcome.html` = 4 (app-store-install open, cluster-init open in handleInstall; cluster-init open in openClusterInitStream; pre-existing init-logs EventSource at line 269)
- `grep app-store-install welcome.html` — 3 occurrences (comment, app_name literal, comment in openClusterInitStream)
- `grep app_name=cluster-init welcome.html` — 2 occurrences (openClusterInitStream + pre-existing handleInitialize)
- `grep /cluster-status welcome.html` — 3 occurrences (pre-existing poll, pre-existing fetch, new D-11 fetch)
- `grep addEventListener welcome.html` — 0 occurrences (only es.onmessage used)
- `grep 'alert(' welcome.html` — 0 occurrences
- `grep stageColor welcome.html` — 4 occurrences (definition + 1 call in render)
- `grep stages welcome.html` — all instances traced to new state + render
- `grep installError welcome.html` — 5 occurrences (state decl, set in complete/error handlers, JSX render)
- `grep localStorage welcome.html` — only selectedNamespace; no secrets
- `{% raw %}` count = 54 (matched `{% endraw %}` count = 54; all sx={{}} guarded)
- `grep setInterval welcome.html` — 1 occurrence (pre-existing handleInitialize poll, not new)
