---
phase: 12-nemoclaw-eks-topology
plan: 01
subsystem: infra
tags: [kubernetes, agent-sandbox, openclaw, nvidia, gpu, eks, sandbox-crd]

# Dependency graph
requires:
  - phase: 11-streamable-http-transport
    provides: MCP server with HTTP transport ready for sidecar pattern (port 8080)

provides:
  - Sandbox CR manifest (agents.x-k8s.io/v1alpha1) for OpenClaw deployment via agent-sandbox CRD
  - Secret templates for openclaw-token and nvidia-api-key
  - Operator install script for agent-sandbox v0.2.1
  - Topology smoke test script with 5-step validation including loopback :8080 path probe

affects:
  - 13-mcp-sidecar-wiring
  - 14-agent-integration-testing

# Tech tracking
tech-stack:
  added: [agent-sandbox CRD v0.2.1, ghcr.io/openclaw/openclaw]
  patterns: [Sandbox CR wrapping standard podTemplate for GPU pod lifecycle, WARN-only loopback probe pattern for pre-sidecar topology validation]

key-files:
  created:
    - k8s/agent-sandbox/openclaw-sandbox.yaml
    - k8s/agent-sandbox/openclaw-secrets.yaml
    - scripts/install-agent-sandbox.sh
    - scripts/validate-topology.sh

key-decisions:
  - "Sandbox CR has no hardcoded namespace — applied with -n flag to match locked decision of same namespace as WEKA App Store components"
  - "Loopback :8080 probe is WARN-only in validate-topology.sh (exit 7 = PASS, exit 28 = WARN) since MCP sidecar does not exist until Phase 13"
  - "OpenClaw gateway health endpoint /healthz probe is also WARN-only per research open question #1 (exact path unconfirmed)"

patterns-established:
  - "Pattern: All K8s manifests use no hardcoded namespace — apply with kubectl -n <NAMESPACE>"
  - "Pattern: Smoke test WARN-only steps for future-phase features (loopback :8080 is NCLAW-03 evidence without requiring Phase 13)"

requirements-completed: [NCLAW-01, NCLAW-03]

# Metrics
duration: 2min
completed: 2026-03-24
---

# Phase 12 Plan 01: NemoClaw EKS Topology Manifests Summary

**Sandbox CR (agents.x-k8s.io/v1alpha1) and operator install/smoke test scripts for OpenClaw GPU deployment on EKS via agent-sandbox v0.2.1**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-24T03:16:46Z
- **Completed:** 2026-03-24T03:18:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Sandbox CR manifest for OpenClaw with GPU resources (`nvidia.com/gpu: 1`), security context (non-root, drop ALL caps), Secret refs, and PVC volume claim template
- Secret templates with placeholder values and `kubectl create secret` usage instructions at the top
- Operator install script (`install-agent-sandbox.sh`) that applies manifest.yaml + extensions.yaml, waits for controller Ready, and verifies the Sandbox CRD registration
- Topology smoke test (`validate-topology.sh`) with 5 validation steps: pod Ready, GPU limit set, gateway health (WARN-only), GPU node allocatable, and loopback :8080 path probe (WARN-only for NCLAW-03 evidence pre-Phase-13)

## Task Commits

Each task was committed atomically:

1. **Task 1: Sandbox CR manifest and Secret templates** - `467804f` (feat)
2. **Task 2: Operator install script and topology smoke test** - `a7619ec` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `k8s/agent-sandbox/openclaw-sandbox.yaml` - Sandbox CR with OpenClaw image, GPU resources, security context, env from Secrets, volume mounts for config (emptyDir) and workspace (PVC 2Gi)
- `k8s/agent-sandbox/openclaw-secrets.yaml` - Template Secrets for gateway token and NVIDIA API key with placeholder values and usage comments
- `scripts/install-agent-sandbox.sh` - Installs agent-sandbox v0.2.1 operator, waits for controller Ready, verifies Sandbox CRD
- `scripts/validate-topology.sh` - 5-step topology smoke test with structured PASS/FAIL/WARN output and loopback :8080 path probe

## Decisions Made
- Sandbox CR has no hardcoded namespace — applied with `kubectl -n <NAMESPACE>` to match the locked decision of deploying in the same namespace as existing WEKA App Store components
- Loopback :8080 probe in the smoke test is WARN-only: exit code 7 (connection refused) = PASS (path open, no listener expected pre-Phase-13), exit code 28 (timeout) = WARN (possible NetworkPolicy block)
- Gateway health /healthz probe is also WARN-only since the exact health endpoint path is an open question per research (research open question #1)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. 117 existing MCP tests pass with no regressions.

## User Setup Required
Before applying the manifests to EKS, secrets must be created. Replace placeholders in `k8s/agent-sandbox/openclaw-secrets.yaml` or use:
```bash
kubectl create secret generic openclaw-token --from-literal=token=YOUR_TOKEN -n <NAMESPACE>
kubectl create secret generic nvidia-api-key --from-literal=key=YOUR_NVIDIA_API_KEY -n <NAMESPACE>
```

## Next Phase Readiness
- Plan 02 can now apply these manifests to the live EKS cluster and run `scripts/validate-topology.sh` to confirm topology
- `scripts/install-agent-sandbox.sh` targets v0.2.1 — run it first if the agent-sandbox operator is not yet installed
- Smoke test loopback :8080 probe provides NCLAW-03 evidence without Phase 13 (connection refused = path open)
- Phase 13 (MCP sidecar wiring) will add the MCP sidecar as a second container in the same Sandbox podTemplate spec

---
*Phase: 12-nemoclaw-eks-topology*
*Completed: 2026-03-24*
