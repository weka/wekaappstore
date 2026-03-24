#!/usr/bin/env bash
# scripts/validate-topology.sh
# Validates Phase 12 topology: OpenClaw pod Running, GPU allocated, gateway reachable,
# GPU node healthy, and loopback path to port 8080 open (WARN-only, expects connection refused).
# Usage: bash scripts/validate-topology.sh [NAMESPACE]
#   NAMESPACE: target namespace (default: $NAMESPACE env var, fallback: default)
# Requires: kubectl configured with appropriate cluster access
set -euo pipefail

NAMESPACE="${1:-${NAMESPACE:-default}}"
SANDBOX_NAME="openclaw-sandbox"

PASS=0
FAIL=0
WARNS=()

echo "=== Phase 12 Topology Validation ==="
echo "Namespace: ${NAMESPACE}"
echo "Sandbox:   ${SANDBOX_NAME}"
echo ""

# ─── Resolve label selector from Sandbox CR status ───────────────────────────
# The agent-sandbox operator sets agents.x-k8s.io/sandbox-name-hash=<hash>
# (not sandbox.agents.x-k8s.io/name=<name>) as the pod selector.
# Retrieve the actual selector from the Sandbox CR status field.
SANDBOX_SELECTOR=$(kubectl get sandbox "${SANDBOX_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.selector}' 2>/dev/null || echo "")

if [[ -z "${SANDBOX_SELECTOR}" ]]; then
  echo "FAIL: Could not retrieve selector from Sandbox CR status. Is the operator running?"
  echo "      kubectl get sandbox ${SANDBOX_NAME} -n ${NAMESPACE}"
  exit 1
fi

# ─── Step 1: Pod Running ─────────────────────────────────────────────────────
echo "[1/5] Checking pod Ready status (timeout: 300s)..."
if kubectl wait --for=condition=Ready pod \
    -l "${SANDBOX_SELECTOR}" \
    -n "${NAMESPACE}" \
    --timeout=300s > /dev/null 2>&1; then
  echo "  PASS: Pod is Ready"
  PASS=$((PASS + 1))
else
  echo "  FAIL: Pod is not Ready after 300s"
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Topology Validation FAILED (1 failure) ==="
  exit 1
fi

# Retrieve pod name and node name for subsequent checks
POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
  -l "${SANDBOX_SELECTOR}" \
  -o jsonpath='{.items[0].metadata.name}')
NODE_NAME=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.spec.nodeName}')

# ─── Step 2: GPU Allocated ────────────────────────────────────────────────────
echo "[2/5] Checking GPU allocation..."
GPU_LIMIT=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}' 2>/dev/null || true)
if [[ "${GPU_LIMIT}" == "1" ]]; then
  echo "  PASS: GPU limit set (nvidia.com/gpu: 1)"
  PASS=$((PASS + 1))
else
  echo "  FAIL: GPU limit not set correctly (got: '${GPU_LIMIT}', expected: '1')"
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Topology Validation FAILED (1 failure) ==="
  exit 1
fi

# ─── Step 3: OpenClaw Gateway Health (WARN-only) ─────────────────────────────
echo "[3/5] Checking OpenClaw gateway responsiveness on port 18789 (WARN-only)..."
# NOTE: The /healthz path is unconfirmed per research open question #1.
# A connection refused or 404 may be expected depending on OpenClaw version.
GATEWAY_RC=0
kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -- \
  curl -sf --max-time 10 http://localhost:18789/healthz > /dev/null 2>&1 || GATEWAY_RC=$?
if [[ "${GATEWAY_RC}" -eq 0 ]]; then
  echo "  PASS: Gateway health endpoint responded on port 18789"
  PASS=$((PASS + 1))
else
  echo "  WARN: Gateway health endpoint /healthz not found (exit ${GATEWAY_RC})"
  echo "        Check pod logs: kubectl logs ${POD_NAME} -n ${NAMESPACE}"
  echo "        This may be expected if the gateway uses a different health path."
  WARNS+=("Step 3: Gateway health endpoint /healthz returned exit ${GATEWAY_RC} — may be expected")
fi

# ─── Step 4: GPU Node Healthy ────────────────────────────────────────────────
echo "[4/5] Checking GPU node allocatable resources..."
GPU_ALLOC=$(kubectl get node "${NODE_NAME}" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)
if [[ -n "${GPU_ALLOC}" ]] && [[ "${GPU_ALLOC}" -ge 1 ]]; then
  echo "  PASS: GPU node '${NODE_NAME}' has ${GPU_ALLOC} allocatable GPU(s)"
  PASS=$((PASS + 1))
else
  echo "  FAIL: GPU node '${NODE_NAME}' shows 0 or missing allocatable GPUs (got: '${GPU_ALLOC}')"
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Topology Validation FAILED (1 failure) ==="
  exit 1
fi

# ─── Step 5: Loopback Port 8080 Path Probe (WARN-only, NCLAW-03 evidence) ────
echo "[5/5] Probing loopback path to port 8080 (WARN-only — no MCP sidecar yet)..."
# No MCP sidecar exists in Phase 12 — port 8080 has no listener.
# Expected result: connection refused (curl exit 7) = loopback path OPEN.
# Timeout (exit 28) = loopback path may be blocked (NetworkPolicy concern).
# This step NEVER causes FAIL — it is informational only.
LOOPBACK_RC=0
LOOPBACK_STATUS=""
kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -- \
  curl -sf --max-time 5 http://localhost:8080/ > /dev/null 2>&1 || LOOPBACK_RC=$?

case "${LOOPBACK_RC}" in
  0)
    LOOPBACK_STATUS="PASS (unexpected success — something is listening on :8080)"
    echo "  PASS: Loopback path open and something is listening on port 8080 (unexpected but fine)"
    ;;
  7)
    LOOPBACK_STATUS="PASS (connection refused — loopback path open, no listener as expected)"
    echo "  PASS: Loopback path to :8080 is OPEN (connection refused = no listener, expected pre-Phase-13)"
    ;;
  28)
    LOOPBACK_STATUS="WARN: Timeout — loopback path to :8080 may be blocked by NetworkPolicy"
    echo "  WARN: Loopback path to :8080 timed out (exit 28) — possible NetworkPolicy block"
    echo "        This may prevent Phase 13 MCP sidecar communication. Investigate NetworkPolicies:"
    echo "        kubectl get networkpolicy -n ${NAMESPACE}"
    WARNS+=("Step 5: Loopback port 8080 timeout — possible NetworkPolicy block")
    ;;
  *)
    LOOPBACK_STATUS="WARN: Unexpected curl error (exit ${LOOPBACK_RC})"
    echo "  WARN: Loopback probe returned unexpected error (exit ${LOOPBACK_RC})"
    echo "        Details: $(kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -- curl -v --max-time 5 http://localhost:8080/ 2>&1 || true)"
    WARNS+=("Step 5: Loopback port 8080 probe exit ${LOOPBACK_RC} — unexpected error")
    ;;
esac

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Topology Validation Summary ==="
echo "Pod:    ${POD_NAME}"
echo "Node:   ${NODE_NAME}"
echo "GPU:    ${GPU_ALLOC}x nvidia.com/gpu allocatable"
echo "Port 8080 loopback: ${LOOPBACK_STATUS}"

if [[ ${#WARNS[@]} -gt 0 ]]; then
  echo ""
  echo "Warnings:"
  for warn in "${WARNS[@]}"; do
    echo "  - ${warn}"
  done
fi

if [[ "${FAIL}" -eq 0 ]]; then
  echo ""
  echo "=== Topology Validation PASSED (${PASS} checks passed, ${#WARNS[@]} warnings) ==="
  exit 0
else
  echo ""
  echo "=== Topology Validation FAILED (${FAIL} failures) ==="
  exit 1
fi
