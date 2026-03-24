---
phase: 12-nemoclaw-eks-topology
plan: "02"
subsystem: infra
tags: [eks, openclaw, nemoclaw, agent-sandbox, kubernetes, gpu, nvidia]

requires:
  - phase: 12-01
    provides: agent-sandbox operator installed (controller Running, Sandbox CRD registered), Sandbox CR manifest at k8s/agent-sandbox/openclaw-sandbox.yaml, validate-topology.sh smoke test script

provides:
  - OpenClaw pod Running on EKS GPU node (NVIDIA A10G, g5.4xlarge) in wekaappstore namespace
  - Sandbox CR deployed and validated via smoke test (all 5 checks PASS)
  - .planning/TOPOLOGY.md with live cluster values for Phase 13 manifest authoring
  - PROJECT.md Key Decision entry for agent-sandbox CRD topology approach
  - validate-topology.sh fixed to use actual operator label selector (agents.x-k8s.io/sandbox-name-hash)
  - openclaw-sandbox.yaml fixed to use --bind=loopback (required for containerized deployment)

affects:
  - 13-sidecar-deployment
  - phase 13 manifest authoring

tech-stack:
  added: []
  patterns:
    - "agent-sandbox CRD Sandbox CR as deployment unit for OpenClaw (not plain Deployment)"
    - "OpenClaw gateway --bind=loopback for sidecar pods (non-loopback requires controlUi config)"
    - "validate-topology.sh uses Sandbox CR status.selector for pod label resolution"
    - "loopback networking in shared pod network namespace confirmed as Phase 13 sidecar communication path"

key-files:
  created:
    - .planning/TOPOLOGY.md
  modified:
    - k8s/agent-sandbox/openclaw-sandbox.yaml
    - scripts/validate-topology.sh
    - .planning/PROJECT.md

key-decisions:
  - "OpenClaw gateway uses --bind=loopback (not --bind=lan) because non-loopback bind mode requires gateway.controlUi.allowedOrigins config; loopback is correct for sidecar deployment since all containers share pod network namespace"
  - "Phase 13 modifies Sandbox CR spec (spec.podTemplate.spec.containers[]), not a separate Deployment"
  - "MCP sidecar will connect to localhost:18789; gateway will reach MCP sidecar at localhost:8080"
  - "Loopback :8080 path confirmed open (NCLAW-03): connection refused = no listener, path not blocked"

patterns-established:
  - "Pattern 1: Sandbox CR label resolution — always query status.selector from Sandbox CR, never hardcode label"
  - "Pattern 2: OpenClaw gateway bind mode — use loopback for pod-internal access in EKS sidecar pattern"

requirements-completed: [NCLAW-01, NCLAW-03]

duration: "~15min (active); total wall-clock includes prior blocking for secrets"
completed: "2026-03-24"
---

# Phase 12 Plan 02: EKS Topology Deployment and Validation Summary

**OpenClaw deployed to EKS via agent-sandbox Sandbox CRD with NVIDIA A10G GPU, gateway running on loopback :18789, all 5 topology checks PASS, TOPOLOGY.md written for Phase 13**

## Performance

- **Duration:** ~15 min (active execution)
- **Started:** 2026-03-24T04:33:00Z (secrets confirmed)
- **Completed:** 2026-03-24T04:48:00Z
- **Tasks:** 3/3 (including auto-approved checkpoint)
- **Files modified:** 4

## Accomplishments

- OpenClaw pod Running on NVIDIA A10G (g5.4xlarge) in wekaappstore namespace — NCLAW-01 satisfied
- Loopback networking confirmed: port 8080 connection refused = path open, not blocked — NCLAW-03 satisfied
- All 5 topology smoke test checks PASS (pod Ready, GPU allocated, gateway :18789, GPU node, loopback :8080)
- TOPOLOGY.md written with live cluster values (image digest, GPU node, Phase 13 integration points)
- Two bugs auto-fixed: wrong label selector in validate-topology.sh; gateway startup failure from --bind=lan

## Task Commits

Each task was committed atomically:

1. **Task 1: Deploy Sandbox CR and run smoke test** - `4a3ee52` (feat)
2. **Task 2: Verify EKS deployment** - auto-approved (checkpoint:human-verify, auto_advance=true)
3. **Task 3: Write TOPOLOGY.md and update PROJECT.md** - `1c407eb` (feat)

## Files Created/Modified

- `.planning/TOPOLOGY.md` — Live cluster topology reference for Phase 13 manifest authoring (113 lines)
- `k8s/agent-sandbox/openclaw-sandbox.yaml` — Fixed --bind=loopback, removed invalid --dangerouslyAllowHostHeaderOriginFallback
- `scripts/validate-topology.sh` — Fixed pod label selector to use Sandbox CR status.selector dynamically
- `.planning/PROJECT.md` — Added Key Decision for agent-sandbox CRD topology with --bind=loopback rationale

## Decisions Made

- **Gateway bind mode:** `--bind=loopback` not `--bind=lan`. Non-loopback bind requires `gateway.controlUi.allowedOrigins` in config file (not a CLI flag). Loopback is correct for sidecar deployment — pod network namespace is shared.
- **Phase 13 target:** Sandbox CR spec modification (not a Deployment). Add MCP sidecar to `spec.podTemplate.spec.containers[]`.
- **MCP sidecar endpoint:** `localhost:18789` to reach gateway, `localhost:8080` for gateway to reach MCP sidecar.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] validate-topology.sh used wrong pod label selector**
- **Found during:** Task 1 smoke test
- **Issue:** Script used `sandbox.agents.x-k8s.io/name=openclaw-sandbox` but operator sets `agents.x-k8s.io/sandbox-name-hash=<hash>`; all kubectl wait/get calls failed silently
- **Fix:** Script now reads actual selector from `kubectl get sandbox ... -o jsonpath='{.status.selector}'` and uses it dynamically
- **Files modified:** scripts/validate-topology.sh
- **Verification:** Smoke test passes all 5 checks
- **Committed in:** 4a3ee52 (Task 1 commit)

**2. [Rule 1 - Bug] OpenClaw gateway fails to start with --bind=lan**
- **Found during:** Task 1, Step 5 (smoke test — pod in CrashLoopBackOff)
- **Issue:** `--bind=lan` (non-loopback) requires `gateway.controlUi.allowedOrigins` or `dangerouslyAllowHostHeaderOriginFallback=true` in config file; neither exists in containerized deployment; `--dangerouslyAllowHostHeaderOriginFallback` is not a valid CLI flag
- **Fix:** Changed to `--bind=loopback`; all pod-internal sidecar communication via localhost works regardless of bind interface
- **Files modified:** k8s/agent-sandbox/openclaw-sandbox.yaml
- **Verification:** Gateway starts successfully (`listening on ws://127.0.0.1:18789`); /healthz responds (smoke test step 3 PASS)
- **Committed in:** 4a3ee52 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both auto-fixes necessary for deployment to work. No scope creep. The --bind=loopback change is also more correct for Phase 13 sidecar pattern.

## Issues Encountered

- AWS SSO token expired before task start (handled in previous blocked run)
- K8s Secrets not yet created at start (handled in previous blocked run)
- Image pull time: ~60s from GHCR (image cached on second pod deletion/recreation)
- operator did not automatically reconcile pod on Sandbox spec update — required `kubectl delete pod openclaw-sandbox` to force recreation

## User Setup Required

None — no external service configuration required (secrets were created by user prior to continuation).

## Next Phase Readiness

- TOPOLOGY.md provides complete reference for Phase 13 manifest authoring
- Sandbox CR path confirmed: `k8s/agent-sandbox/openclaw-sandbox.yaml`
- MCP sidecar adds to `spec.podTemplate.spec.containers[]`
- Init container for openclaw.json writes to `/home/node/.openclaw` emptyDir volume
- Gateway at `localhost:18789`, MCP sidecar target at `localhost:8080`
- Loopback networking confirmed open (NCLAW-03) — no NetworkPolicy blocking sidecar communication
- Blocker "[Phase 12 gate]: NemoClaw EKS topology must be validated" is now resolved

---
*Phase: 12-nemoclaw-eks-topology*
*Completed: 2026-03-24*
