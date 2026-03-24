#!/usr/bin/env bash
# scripts/capture-e2e-evidence.sh
# Captures kubectl evidence for all 4 E2E requirements.
#
# Usage: bash scripts/capture-e2e-evidence.sh [NAMESPACE] [--pre | --post]
#   NAMESPACE: target namespace (default: wekaappstore)
#   --pre:     capture only E2E-01 and E2E-02 (before chat session)
#   --post:    capture only E2E-03 and E2E-04 (after chat session + operator reconciliation)
#   (no flag): capture all 4 evidence sets
#
# Exit codes:
#   0 — evidence captured (individual kubectl failures are warnings, not fatal)
#   1 — critical setup error (cannot proceed)
#
# Requires: kubectl configured with appropriate cluster access
set -euo pipefail

# ─── Argument parsing ─────────────────────────────────────────────────────────
NAMESPACE="wekaappstore"
MODE="all"

for arg in "$@"; do
  case "${arg}" in
    --pre)
      MODE="pre"
      ;;
    --post)
      MODE="post"
      ;;
    --*)
      echo "Unknown flag: ${arg}"
      echo "Usage: $0 [NAMESPACE] [--pre | --post]"
      exit 1
      ;;
    *)
      NAMESPACE="${arg}"
      ;;
  esac
done

SANDBOX_NAME="openclaw-sandbox"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVIDENCE_DIR="${REPO_ROOT}/evidence"

echo "=== E2E Evidence Capture ==="
echo "Namespace: ${NAMESPACE}"
echo "Mode:      ${MODE}"
echo ""

# ─── Create evidence directory ────────────────────────────────────────────────
mkdir -p "${EVIDENCE_DIR}"
echo "Evidence directory: ${EVIDENCE_DIR}"
echo ""

# ─── Resolve pod dynamically (needed for E2E-02 and pod-specific checks) ──────
SANDBOX_SELECTOR=$(kubectl get sandbox "${SANDBOX_NAME}" -n "${NAMESPACE}" \
  -o jsonpath='{.status.selector}' 2>/dev/null || echo "")

POD_NAME=""
if [[ -n "${SANDBOX_SELECTOR}" ]]; then
  POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
    -l "${SANDBOX_SELECTOR}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
fi

if [[ -n "${POD_NAME}" ]]; then
  echo "  Resolved pod: ${POD_NAME}"
else
  echo "  WARN: Could not resolve pod — E2E-02 blueprint listing will be skipped"
fi
echo ""

# ─── Helper: run a command and write output, warn on failure ──────────────────
run_capture() {
  local desc="$1"
  local outfile="$2"
  local cmd="${@:3}"
  echo "  Capturing: ${desc}..."
  if eval "${cmd}" > "${outfile}" 2>&1; then
    echo "  -> ${outfile}"
  else
    echo "  WARN: Command failed, partial output may be in ${outfile}"
  fi
}

run_capture_append() {
  local desc="$1"
  local outfile="$2"
  local cmd="${@:3}"
  echo "  Capturing (append): ${desc}..."
  if eval "${cmd}" >> "${outfile}" 2>&1; then
    echo "  -> ${outfile} (appended)"
  else
    echo "  WARN: Command failed for ${desc}, partial output appended"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# E2E-01: Cluster resource data (node capacity, namespaces)
# Run before chat session (--pre or all)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "pre" ]] || [[ "${MODE}" == "all" ]]; then
  echo "--- E2E-01: Cluster resource data ---"

  run_capture \
    "kubectl get nodes -o wide" \
    "${EVIDENCE_DIR}/e2e-01-nodes.txt" \
    "kubectl get nodes -o wide"

  GPU_NODE="ip-172-3-1-203.us-west-2.compute.internal"
  run_capture_append \
    "node describe: Capacity + Allocatable" \
    "${EVIDENCE_DIR}/e2e-01-nodes.txt" \
    "kubectl describe node ${GPU_NODE} | grep -A10 'Capacity:\|Allocatable:'"

  run_capture_append \
    "kubectl get namespaces" \
    "${EVIDENCE_DIR}/e2e-01-nodes.txt" \
    "kubectl get namespaces"

  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# E2E-02: Blueprint catalog listing
# Run before chat session (--pre or all)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "pre" ]] || [[ "${MODE}" == "all" ]]; then
  echo "--- E2E-02: Blueprint catalog ---"

  if [[ -n "${POD_NAME}" ]]; then
    # Resolve BLUEPRINTS_DIR dynamically from the running container to handle mount path changes
    BLUEPRINTS_DIR_LIVE=$(kubectl exec "${POD_NAME}" -c weka-mcp-sidecar -n "${NAMESPACE}" -- \
      sh -c 'echo $BLUEPRINTS_DIR' 2>/dev/null || echo "/app/git-sync-root/blueprints/manifests")
    run_capture \
      "blueprint catalog listing (${BLUEPRINTS_DIR_LIVE})" \
      "${EVIDENCE_DIR}/e2e-02-blueprints.txt" \
      "kubectl exec ${POD_NAME} -c weka-mcp-sidecar -n ${NAMESPACE} -- ls -la ${BLUEPRINTS_DIR_LIVE}"
  else
    echo "  SKIP: No pod available — E2E-02 blueprint evidence not captured"
    echo "  SKIP: No pod available — run after pod is Ready" > "${EVIDENCE_DIR}/e2e-02-blueprints.txt"
  fi

  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# E2E-03: WekaAppStore CR created by agent
# Run AFTER chat session (--post or all)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "post" ]] || [[ "${MODE}" == "all" ]]; then
  echo "--- E2E-03: WekaAppStore CRs ---"

  run_capture \
    "kubectl get wekaappstores -o wide" \
    "${EVIDENCE_DIR}/e2e-03-wekaappstores.txt" \
    "kubectl get wekaappstores -n ${NAMESPACE} -o wide"

  run_capture_append \
    "kubectl get wekaappstore -o yaml (all CRs)" \
    "${EVIDENCE_DIR}/e2e-03-wekaappstores.txt" \
    "kubectl get wekaappstore -n ${NAMESPACE} -o yaml"

  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# E2E-04: Operator reconciliation status
# Run AFTER operator reconciliation (--post or all)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "post" ]] || [[ "${MODE}" == "all" ]]; then
  echo "--- E2E-04: Deployment status after reconciliation ---"

  run_capture \
    "kubectl get pods -o wide" \
    "${EVIDENCE_DIR}/e2e-04-status.txt" \
    "kubectl get pods -n ${NAMESPACE} -o wide"

  run_capture_append \
    "kubectl describe wekaappstore (all CRs)" \
    "${EVIDENCE_DIR}/e2e-04-status.txt" \
    "kubectl describe wekaappstore -n ${NAMESPACE}"

  echo ""
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "=== Evidence captured in ${EVIDENCE_DIR}/ ==="
echo ""
ls -la "${EVIDENCE_DIR}/" 2>/dev/null || echo "(directory is empty)"
echo ""
echo "Evidence files:"
case "${MODE}" in
  pre)
    echo "  - e2e-01-nodes.txt   (cluster nodes + capacity + namespaces)"
    echo "  - e2e-02-blueprints.txt  (blueprint catalog)"
    echo ""
    echo "Next: run the E2E chat session, then run with --post to capture post-chat evidence."
    ;;
  post)
    echo "  - e2e-03-wekaappstores.txt  (WekaAppStore CRs created by agent)"
    echo "  - e2e-04-status.txt         (pod status + operator reconciliation)"
    ;;
  all)
    echo "  - e2e-01-nodes.txt          (cluster nodes + capacity + namespaces)"
    echo "  - e2e-02-blueprints.txt     (blueprint catalog)"
    echo "  - e2e-03-wekaappstores.txt  (WekaAppStore CRs created by agent)"
    echo "  - e2e-04-status.txt         (pod status + operator reconciliation)"
    ;;
esac

exit 0
