#!/usr/bin/env bash
# scripts/validate-phase14-prereqs.sh
# Validates all Phase 14 preconditions before the E2E chat session.
# Delegates Phase 13 checks first, then adds Phase 14-specific prereqs.
#
# Usage: bash scripts/validate-phase14-prereqs.sh [NAMESPACE]
#   NAMESPACE: target namespace (default: wekaappstore)
#
# Exit codes:
#   0 — all checks PASS (WARNs do not cause failure)
#   1 — one or more FAIL
#
# Requires: kubectl configured with appropriate cluster access
set -euo pipefail

# ─── Argument parsing ─────────────────────────────────────────────────────────
NAMESPACE="${1:-wekaappstore}"
SANDBOX_NAME="openclaw-sandbox"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PASS=0
FAIL=0
WARNS=()

echo "=== Phase 14 Prerequisite Validation ==="
echo "Namespace: ${NAMESPACE}"
echo "Sandbox:   ${SANDBOX_NAME}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Phase 13 validation passes
# ─────────────────────────────────────────────────────────────────────────────
echo "[1] Running Phase 13 validation (delegate)..."
if bash "${REPO_ROOT}/scripts/validate-phase13.sh" "${NAMESPACE}" --live; then
  echo "  PASS: Phase 13 preconditions met"
  PASS=$((PASS + 1))
else
  echo "  FAIL: Phase 13 preconditions not met"
  echo "        Fix Phase 13 issues before running Phase 14 E2E session."
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Phase 14 Prerequisite Validation ABORTED ==="
  echo "Phase 13 preconditions not met — resolve Phase 13 failures first."
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Resolve pod dynamically
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Resolving openclaw-sandbox pod ---"

SANDBOX_SELECTOR=$(kubectl get sandbox "${SANDBOX_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.selector}' 2>/dev/null || echo "")

if [[ -z "${SANDBOX_SELECTOR}" ]]; then
  echo "  FAIL: Could not retrieve selector from Sandbox CR status."
  echo "        Is the agent-sandbox operator running?"
  echo "        kubectl get sandbox ${SANDBOX_NAME} -n ${NAMESPACE}"
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Phase 14 Prerequisite Validation FAILED (${FAIL} failures) ==="
  exit 1
fi

POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
  -l "${SANDBOX_SELECTOR}" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "${POD_NAME}" ]]; then
  echo "  FAIL: No pod found for selector '${SANDBOX_SELECTOR}' in namespace '${NAMESPACE}'"
  FAIL=$((FAIL + 1))
  echo ""
  echo "=== Phase 14 Prerequisite Validation FAILED (${FAIL} failures) ==="
  exit 1
else
  echo "  Resolved pod: ${POD_NAME}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: OSS Rag blueprint in catalog
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2] Checking OSS Rag blueprint in MCP sidecar catalog (BLUEPRINTS_DIR)..."

# Resolve the actual BLUEPRINTS_DIR from the running container env (handles mount path changes)
BLUEPRINTS_DIR=$(kubectl exec "${POD_NAME}" -c weka-mcp-sidecar -n "${NAMESPACE}" -- \
  sh -c 'echo $BLUEPRINTS_DIR' 2>/dev/null || echo "/app/git-sync-root/blueprints/manifests")

BLUEPRINTS_OUT=$(kubectl exec "${POD_NAME}" -c weka-mcp-sidecar -n "${NAMESPACE}" -- \
  ls "${BLUEPRINTS_DIR}" 2>&1 || echo "ERROR")

if [[ "${BLUEPRINTS_OUT}" == "ERROR" ]]; then
  echo "  FAIL: Could not list ${BLUEPRINTS_DIR} in weka-mcp-sidecar"
  echo "        kubectl exec ${POD_NAME} -c weka-mcp-sidecar -n ${NAMESPACE} -- ls ${BLUEPRINTS_DIR}"
  FAIL=$((FAIL + 1))
elif echo "${BLUEPRINTS_OUT}" | grep -qi "oss\|rag"; then
  echo "  PASS: OSS Rag blueprint found in catalog (${BLUEPRINTS_DIR})"
  echo "        Entries: ${BLUEPRINTS_OUT}"
  PASS=$((PASS + 1))
else
  echo "  WARN: No entry matching 'oss' or 'rag' (case-insensitive) found in ${BLUEPRINTS_DIR}"
  echo "        Blueprint naming may differ. Catalog contents:"
  echo "${BLUEPRINTS_OUT}" | sed 's/^/        /'
  WARNS+=("Check 2: No 'oss' or 'rag' blueprint found — verify catalog contents manually")
fi

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: All 3 containers are Ready
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3] Checking all 3 containers are Ready in pod ${POD_NAME}..."

READY_STATUSES=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.containerStatuses[*].ready}' 2>/dev/null || echo "")

# Count "true" values
READY_COUNT=$(echo "${READY_STATUSES}" | tr ' ' '\n' | grep -c "^true$" || true)
TOTAL_COUNT=$(echo "${READY_STATUSES}" | tr ' ' '\n' | wc -l | tr -d ' ')

if [[ "${READY_COUNT}" -ge 3 ]]; then
  echo "  PASS: All 3 containers are Ready (${READY_COUNT}/${TOTAL_COUNT} true)"
  PASS=$((PASS + 1))
else
  # Check if ALL are true by verifying no "false" exists
  if echo "${READY_STATUSES}" | grep -q "false"; then
    echo "  FAIL: Not all containers are Ready (${READY_COUNT}/${TOTAL_COUNT} ready)"
    echo "        kubectl get pod ${POD_NAME} -n ${NAMESPACE} -o wide"
    FAIL=$((FAIL + 1))
  else
    echo "  WARN: Could not determine all container Ready states (got: '${READY_STATUSES}')"
    echo "        kubectl describe pod ${POD_NAME} -n ${NAMESPACE}"
    WARNS+=("Check 3: Could not verify all containers Ready — check pod manually")
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: MCP sidecar health endpoint
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4] Checking MCP sidecar health endpoint (localhost:8080/health)..."

HEALTH_RC=0
HEALTH_OUT=$(kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -c weka-mcp-sidecar -- \
  python3 -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8080/health',timeout=10); print(r.status)" 2>&1) || HEALTH_RC=$?

if [[ "${HEALTH_RC}" -eq 0 ]] && echo "${HEALTH_OUT}" | grep -q "^200$"; then
  echo "  PASS: MCP sidecar /health endpoint responded (HTTP 200)"
  PASS=$((PASS + 1))
else
  echo "  FAIL: MCP sidecar /health endpoint did not respond (got: '${HEALTH_OUT}', exit ${HEALTH_RC})"
  echo "        kubectl logs ${POD_NAME} -c weka-mcp-sidecar -n ${NAMESPACE}"
  FAIL=$((FAIL + 1))
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 14 Prerequisites Summary ==="
echo "Checks passed: ${PASS}"
echo "Checks failed: ${FAIL}"
WARN_COUNT=${#WARNS[@]}

if [[ "${WARN_COUNT}" -gt 0 ]]; then
  echo "Warnings:      ${WARN_COUNT}"
  for warn in "${WARNS[@]}"; do
    echo "  - ${warn}"
  done
fi

echo ""
echo "Phase 14 Prerequisites: ${PASS}/4 PASS, ${WARN_COUNT} WARN"
echo ""
echo "To start the E2E chat session, run the port-forward:"
echo "  kubectl port-forward pod/${POD_NAME} 18789:18789 -n ${NAMESPACE}"
echo ""
echo "Then open OpenClaw Web UI in your browser:"
echo "  http://localhost:18789"
echo ""

if [[ "${FAIL}" -eq 0 ]]; then
  echo "=== Phase 14 Prerequisites PASSED (${PASS} checks passed, ${WARN_COUNT} warnings) ==="
  exit 0
else
  echo "=== Phase 14 Prerequisites FAILED (${FAIL} failures) ==="
  exit 1
fi
