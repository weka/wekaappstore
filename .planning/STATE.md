---
gsd_state_version: 1.0
milestone: v8.0
milestone_name: Guided Install Wizard — WEKA Operator, CSI & Storage Classes
status: planning
last_updated: "2026-06-24T13:05:00.000Z"
last_activity: 2026-06-24
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-24 — milestone v8.0 started)

**Core value:** A customer can stand up the full WEKA storage stack (operator, CSI driver, client, storage classes) from the App Store install wizard by answering a short form — no manual `kubectl`/`helm`/base64 work.
**Current focus:** Roadmap created (phases 27–31); ready to plan Phase 27.

## Current Position

Phase: 27 — Install Blueprint Authoring (not started)
Plan: —
Status: Roadmap complete, awaiting phase planning
Last activity: 2026-06-24 — v8.0 roadmap created, 28/28 requirements mapped

Progress: [----------] 0% (0/5 phases)

## Accumulated Context

### Key Architectural Decisions (carried into v8.0)

- **Decision C REVISED (load-bearing):** the operator pod runs `helm show crds` / `helm install oci://quay.io/...` from its OWN helm process, which authenticates via the helm registry config — NOT the Kubernetes `dockerconfigjson` pull secrets (those only cover kubelet image pulls). Operator-side `helm registry login` / `--registry-config` is real work, scoped into Phase 28 (no spike — planned upfront). Bundle with the `discover_chart_crds` empty-failure cache fix.
- **`[[var]]` Jinja2 delimiters for GUI substitution, `${VAR}` for operator substitution** — keep separate. The new blueprint is rendered once by the GUI with `[[ ]]`.
- **`stringData` over hand-`base64` for all wizard secrets** — eliminates the trailing-newline bug class. The ONE exception is the quay `dockerconfigjson`, which is built server-side in Python and injected as one `data` var (do not double-encode in `stringData`).
- **WekaClient 404 race** — gate the WekaClient manifest via `dependsOn: [weka-operator]` + `readinessCheck: {type: deployment}` on the operator (transitively guarantees CRDs served). Operator manifests have no readiness wait — never put a readinessCheck on the WekaClient manifest itself.
- **Secret-leak release gate (SEC-01 / Success Criterion 4):** allowlist excludes `*password*`/`*token*`/`*secret*`/`quay_dockerconfigjson` from the `warp.io/gui-variables` annotation; redact emitted SSE messages; never pass creds as helm `--set`.
- **SSE 900s deadline too short** — raise per-blueprint (45–60 min) and make timeout non-destructive (reconnect/resume from `componentStatus`), since a cold-cache WekaClient image pull + cluster join alone can exceed 15 min.
- **Node prereqs stay manual (Decision A1)** — copy-paste `KubeletConfiguration` + confirm checkbox; the App Store never restarts kubelet (auto-restart can brick every node).
- **Two chained CRs (Decision D/E)** — `app-store-install` runs to `Ready`, then the untouched `app-store-cluster-init` runs last; redirect on cluster-init `Ready` via the unchanged `ClusterInitMiddleware`.

### Open Blockers / Tracked Work (pre-existing, not v8.0)

- **v3.1 deferred** — E2E chat validation + four prerequisite fixes (see `.planning/v3.0-KNOWN-ISSUES.md`).
- **v5.0 Phases 19-20 unstarted** — Validator Soft-Warning and AIDP Migration Smoke Test.
- **DYN-07 external repo** — production blueprint `x-variables` migration in `warp-blueprints`.

### Pending Todos

None.

## Session Continuity

Last session: 2026-06-24
Stopped at: v8.0 roadmap created (phases 27–31, 28/28 requirements mapped)
Resume: `/gsd:plan-phase 27` to plan Install Blueprint Authoring
