---
phase: 14-end-to-end-validation
plan: "01"
subsystem: infra
tags: [kubernetes, gateway-api, httproute, bash, kubectl, e2e, openclaw, mcp]

requires:
  - phase: 13-kubernetes-manifests-and-sidecar-wiring
    provides: Phase 13 validation script, live openclaw-sandbox pod with MCP sidecar

provides:
  - validate-phase14-prereqs.sh — pre-E2E validation (Phase 13 delegate + OSS Rag catalog + pod health + MCP health 200 + port-forward instructions)
  - capture-e2e-evidence.sh — E2E evidence capture for all 4 requirements with --pre/--post/all modes
  - k8s/openclaw-gateway-svc.yaml — ClusterIP Service for openclaw-sandbox pod on port 18789
  - k8s/openclaw-ui-route.yaml — Gateway API HTTPRoute for openclaw.example.com

affects:
  - 14-02 — E2E chat session requires port-forward from prereq script and post-chat evidence from capture script

tech-stack:
  added: []
  patterns:
    - Bash validation script delegation pattern (Phase 14 delegates to Phase 13 --live, then adds Phase 14 checks)
    - Dynamic pod resolution via Sandbox CR status.selector (established in Phase 13, reused here)
    - Gateway API HTTPRoute following cluster_init/routes/ pattern (parentRef warp-edge-gateway, hostnames[], backendRefs[])
    - Evidence capture with --pre/--post/all modes for before/after chat session

key-files:
  created:
    - scripts/validate-phase14-prereqs.sh
    - scripts/capture-e2e-evidence.sh
    - k8s/openclaw-gateway-svc.yaml
    - k8s/openclaw-ui-route.yaml
  modified: []

key-decisions:
  - "HTTPRoute and Service manifests are authored but NOT applied — OpenClaw --bind=loopback prevents Service routing; E2E session uses kubectl port-forward instead"
  - "Service selector uses agents.x-k8s.io/sandbox-name-hash=62f96e10 (from TOPOLOGY.md) — prominently commented as potentially stale if Sandbox CR is modified"
  - "OSS Rag blueprint check is WARN-not-FAIL in prereq script — blueprint naming may differ from oss/rag literal strings"
  - "Evidence capture uses --pre/--post/all flags to separate before-chat and after-chat captures for clean E2E-01/02 vs E2E-03/04 evidence"

patterns-established:
  - "Phase validation delegation: Phase N+1 runs Phase N --live as Check 1, aborts if Phase N fails"
  - "Evidence capture script pattern: named --pre/--post modes, evidence/ directory, WARN-only on individual kubectl failures"

requirements-completed: []  # Plan 01 delivered E2E infrastructure prep only (scripts + unrouted HTTPRoute/Service manifests). E2E-01..04 were never completed — see .planning/v3.0-KNOWN-ISSUES.md. Frontmatter corrected 2026-04-21.

duration: 2min
completed: 2026-03-25
---

# Phase 14 Plan 01: E2E Infrastructure Prep Summary

**Phase 14 prereq validation script (delegates Phase 13), E2E evidence capture script with --pre/--post modes, and Gateway API Service + HTTPRoute manifests for openclaw.example.com — all committed and dry-run validated before the E2E chat session**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-24T22:40:47Z
- **Completed:** 2026-03-24T22:42:44Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `validate-phase14-prereqs.sh` — delegates Phase 13 live checks then adds 3 Phase 14 checks: OSS Rag blueprint in catalog (WARN-only), all containers Ready, MCP /health HTTP 200; prints port-forward command for E2E session
- Created `capture-e2e-evidence.sh` — captures kubectl outputs for E2E-01 through E2E-04 with `--pre` (before chat) and `--post` (after chat) separation modes
- Created `k8s/openclaw-gateway-svc.yaml` and `k8s/openclaw-ui-route.yaml` — complete Gateway API manifests matching cluster pattern; prominently document the --bind=loopback limitation and port-forward workaround; both pass `kubectl apply --dry-run=client`

## Task Commits

1. **Task 1: Create Phase 14 prerequisite validation script and evidence capture script** - `1bd2447` (feat)
2. **Task 2: Author OpenClaw Service and HTTPRoute manifests** - `c0d6869` (feat)

## Files Created/Modified

- `scripts/validate-phase14-prereqs.sh` — Pre-E2E validation: Phase 13 delegate + OSS Rag check + Ready containers + MCP health + port-forward instructions
- `scripts/capture-e2e-evidence.sh` — E2E evidence capture for all 4 requirements with --pre/--post/all modes
- `k8s/openclaw-gateway-svc.yaml` — ClusterIP Service for openclaw-sandbox pod via hash label selector
- `k8s/openclaw-ui-route.yaml` — Gateway API HTTPRoute routing openclaw.example.com to openclaw-gateway-svc

## Decisions Made

- HTTPRoute and Service manifests authored but NOT applied due to OpenClaw `--bind=loopback` — the service would route to eth0 which OpenClaw refuses; E2E session uses `kubectl port-forward` instead
- OSS Rag blueprint check is WARN-not-FAIL — blueprint directory naming may not literally contain "oss" or "rag"; operator can inspect catalog manually if WARN fires
- Evidence capture `--pre`/`--post` separation matches the E2E session flow: pre-captures confirm cluster state before the chat, post-captures confirm agent actions succeeded

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None — both scripts pass bash syntax check, both K8s manifests pass `kubectl apply --dry-run=client`.

## User Setup Required

None — no external service configuration required. All E2E session setup is handled by `validate-phase14-prereqs.sh` which prints the exact port-forward command to run.

## Next Phase Readiness

- Plan 14-02 (E2E chat session) can begin once:
  1. `bash scripts/validate-phase14-prereqs.sh wekaappstore` exits 0
  2. Port-forward is running: `kubectl port-forward pod/$POD 18789:18789 -n wekaappstore`
  3. OpenClaw Web UI is accessible at `http://localhost:18789`
- Evidence capture: run `bash scripts/capture-e2e-evidence.sh wekaappstore --pre` before starting the chat, then `--post` after the agent has applied the WekaAppStore CR

---
*Phase: 14-end-to-end-validation*
*Completed: 2026-03-25*
