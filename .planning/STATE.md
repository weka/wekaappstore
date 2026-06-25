---
gsd_state_version: 1.0
milestone: v8.0
milestone_name: Guided Install Wizard — WEKA Operator, CSI & Storage Classes
status: executing
stopped_at: Phase 30 context gathered (assumptions mode)
last_updated: "2026-06-25T00:24:39.165Z"
last_activity: 2026-06-25
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 10
  completed_plans: 8
  percent: 60
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-24 — milestone v8.0 started)

**Core value:** A customer can stand up the full WEKA storage stack (operator, CSI driver, client, storage classes) from the App Store install wizard by answering a short form — no manual `kubectl`/`helm`/base64 work.
**Current focus:** Phase 30 — wizard-stepper-live-progress

## Current Position

Phase: 30 (wizard-stepper-live-progress) — EXECUTING
Plan: 2 of 3
Status: Ready to execute
Last activity: 2026-06-25

Progress: [████████░░] 80%

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

### Phase 29 Decisions (2026-06-24)

- **NAMESPACE_PRESERVING_APPS set** — single source of truth for namespace-preserve logic; `{"cluster-init", "app-store-install"}` checked at all four sites (deploy(), deploy_stream validation, ns_for_apply, status-poll skip)
- **parse_deploy_timeout with DEFAULT_DEPLOY_TIMEOUT_SECONDS = 2100** — fallback within 1800–2400s band; blueprint declares 2700s for full operator+CSI+WekaClient install
- **find_blueprint cluster-init fixed-path lookup unchanged** — D-01: app-store-install found by generic os.walk
- **build_quay_dockerconfigjson uses base64.b64encode (not encodebytes)** — byte-exact, no trailing newline; load-bearing for helm --registry-config OCI pull
- **split_endpoints returns json.dumps list** — double-quoted JSON array for valid YAML flow-sequence in joinIpPorts
- **_is_secret_key single-predicate dual-site redaction** — matches *password*/*token*/*secret*/quay_dockerconfigjson; shared by annotation allowlist (_safe_gui_variables) and SSE value-redactor (_redact_secrets)

### Open Blockers / Tracked Work (pre-existing, not v8.0)

- **v3.1 deferred** — E2E chat validation + four prerequisite fixes (see `.planning/v3.0-KNOWN-ISSUES.md`).
- **v5.0 Phases 19-20 unstarted** — Validator Soft-Warning and AIDP Migration Smoke Test.
- **DYN-07 external repo** — production blueprint `x-variables` migration in `warp-blueprints`.

### Pending Todos

None.

## Session Continuity

Last session: 2026-06-25T00:24:39.159Z
Stopped at: Phase 30 context gathered (assumptions mode)
Resume: `/gsd:execute-phase 30` to execute Phase 30 (Wizard Stepper & Live Progress)
