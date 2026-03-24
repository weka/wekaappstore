#!/usr/bin/env bash
# scripts/validate-phase13.sh
# Validates Phase 13 manifests: Sandbox CR sidecar wiring, MCP sidecar health,
# openclaw.json generation, SKILL.md mount, and RBAC permissions.
#
# Usage: bash scripts/validate-phase13.sh [NAMESPACE] [--live]
#   NAMESPACE: target namespace (default: $NAMESPACE env var, fallback: wekaappstore)
#   --live:    run live cluster checks (requires kubectl cluster access)
#              dry-run structural checks always run first
#
# Exit codes:
#   0 — all checks PASS (WARNs do not cause failure)
#   1 — one or more FAIL
#
# Requires: kubectl configured with appropriate cluster access (for --live mode)
set -euo pipefail

# ─── Argument parsing ─────────────────────────────────────────────────────────
NAMESPACE="wekaappstore"
LIVE_MODE=false

for arg in "$@"; do
  case "${arg}" in
    --live)
      LIVE_MODE=true
      ;;
    --*)
      echo "Unknown flag: ${arg}"
      echo "Usage: $0 [NAMESPACE] [--live]"
      exit 1
      ;;
    *)
      NAMESPACE="${arg}"
      ;;
  esac
done

NAMESPACE="${NAMESPACE:-${NAMESPACE:-wekaappstore}}"
SANDBOX_NAME="openclaw-sandbox"

# ─── Resolve manifest paths relative to repo root ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RBAC_MANIFEST="${REPO_ROOT}/k8s/agent-sandbox/mcp-rbac.yaml"
CONFIGMAP_MANIFEST="${REPO_ROOT}/k8s/agent-sandbox/mcp-skill-configmap.yaml"
SANDBOX_MANIFEST="${REPO_ROOT}/k8s/agent-sandbox/openclaw-sandbox.yaml"

PASS=0
FAIL=0
WARNS=()

echo "=== Phase 13 Manifest Validation ==="
echo "Namespace: ${NAMESPACE}"
echo "Sandbox:   ${SANDBOX_NAME}"
echo "Mode:      $(${LIVE_MODE} && echo 'live' || echo 'dry-run')"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# DRY-RUN CHECKS (always run — no cluster needed)
# ─────────────────────────────────────────────────────────────────────────────

echo "--- Dry-run structural checks ---"

# ─── Check 1: RBAC YAML syntax ────────────────────────────────────────────────
echo "[1] kubectl dry-run: mcp-rbac.yaml..."
if [[ -f "${RBAC_MANIFEST}" ]]; then
  RBAC_OUT=$(kubectl apply --dry-run=client -f "${RBAC_MANIFEST}" -n "${NAMESPACE}" 2>&1 || true)
  if echo "${RBAC_OUT}" | grep -qE "created|configured|unchanged"; then
    echo "  PASS: mcp-rbac.yaml is valid YAML and accepted by API server (dry-run)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: mcp-rbac.yaml dry-run failed"
    echo "        ${RBAC_OUT}"
    FAIL=$((FAIL + 1))
  fi
else
  echo "  WARN: mcp-rbac.yaml not found at ${RBAC_MANIFEST} (Phase 13-01 may not be complete)"
  WARNS+=("Check 1: mcp-rbac.yaml not found — run Phase 13-01 first")
fi

# ─── Check 2: ConfigMap YAML syntax ───────────────────────────────────────────
echo "[2] kubectl dry-run: mcp-skill-configmap.yaml..."
if [[ -f "${CONFIGMAP_MANIFEST}" ]]; then
  CM_OUT=$(kubectl apply --dry-run=client -f "${CONFIGMAP_MANIFEST}" -n "${NAMESPACE}" 2>&1 || true)
  if echo "${CM_OUT}" | grep -qE "created|configured|unchanged"; then
    echo "  PASS: mcp-skill-configmap.yaml is valid YAML and accepted by API server (dry-run)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: mcp-skill-configmap.yaml dry-run failed"
    echo "        ${CM_OUT}"
    FAIL=$((FAIL + 1))
  fi
else
  echo "  WARN: mcp-skill-configmap.yaml not found at ${CONFIGMAP_MANIFEST} (Phase 13-01 may not be complete)"
  WARNS+=("Check 2: mcp-skill-configmap.yaml not found — run Phase 13-01 first")
fi

# ─── Check 3: Sandbox CR YAML syntax ──────────────────────────────────────────
echo "[3] kubectl dry-run: openclaw-sandbox.yaml..."
if [[ ! -f "${SANDBOX_MANIFEST}" ]]; then
  echo "  FAIL: openclaw-sandbox.yaml not found at ${SANDBOX_MANIFEST}"
  FAIL=$((FAIL + 1))
else
  SANDBOX_OUT=$(kubectl apply --dry-run=client -f "${SANDBOX_MANIFEST}" -n "${NAMESPACE}" 2>&1 || true)
  if echo "${SANDBOX_OUT}" | grep -qE "created|configured|unchanged"; then
    echo "  PASS: openclaw-sandbox.yaml is valid YAML and accepted by API server (dry-run)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: openclaw-sandbox.yaml dry-run failed"
    echo "        ${SANDBOX_OUT}"
    FAIL=$((FAIL + 1))
  fi
fi

# ─── Check 4: Structural grep checks on Sandbox CR ────────────────────────────
echo "[4] Structural content checks on openclaw-sandbox.yaml..."
STRUCT_FAIL=0

check_grep() {
  local pattern="$1"
  local desc="$2"
  if grep -q "${pattern}" "${SANDBOX_MANIFEST}" 2>/dev/null; then
    echo "  PASS: ${desc}"
  else
    echo "  FAIL: ${desc} — pattern not found: '${pattern}'"
    STRUCT_FAIL=$((STRUCT_FAIL + 1))
  fi
}

check_grep "serviceAccountName: weka-mcp-server-sa" "serviceAccountName: weka-mcp-server-sa present (K8S-02)"
check_grep "name: openclaw-json-generator"           "initContainer 'openclaw-json-generator' present (K8S-05)"
check_grep "name: weka-mcp-sidecar"                  "container 'weka-mcp-sidecar' present"
check_grep "name: git-sync"                          "container 'git-sync' present"
check_grep "readinessProbe"                          "readinessProbe defined on MCP sidecar (K8S-03)"
check_grep "BLUEPRINTS_DIR"                          "BLUEPRINTS_DIR env var set (K8S-04)"

if [[ "${STRUCT_FAIL}" -eq 0 ]]; then
  echo "  PASS: All 6 structural elements present"
  PASS=$((PASS + 1))
else
  echo "  FAIL: ${STRUCT_FAIL} structural element(s) missing from openclaw-sandbox.yaml"
  FAIL=$((FAIL + 1))
fi

# ─────────────────────────────────────────────────────────────────────────────
# LIVE CHECKS (only with --live flag)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${LIVE_MODE}" == "true" ]]; then
  echo ""
  echo "--- Live cluster checks ---"

  # ─── Resolve label selector from Sandbox CR status ────────────────────────
  SANDBOX_SELECTOR=$(kubectl get sandbox "${SANDBOX_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.selector}' 2>/dev/null || echo "")

  if [[ -z "${SANDBOX_SELECTOR}" ]]; then
    echo "FAIL: Could not retrieve selector from Sandbox CR status."
    echo "      Is the agent-sandbox operator running?"
    echo "      kubectl get sandbox ${SANDBOX_NAME} -n ${NAMESPACE}"
    exit 1
  fi

  POD_NAME=$(kubectl get pods -n "${NAMESPACE}" \
    -l "${SANDBOX_SELECTOR}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

  if [[ -z "${POD_NAME}" ]]; then
    echo "FAIL: No pod found for selector '${SANDBOX_SELECTOR}' in namespace '${NAMESPACE}'"
    FAIL=$((FAIL + 1))
  else
    echo "  Resolved pod: ${POD_NAME}"
  fi

  # ─── Check 5: RBAC — list nodes ─────────────────────────────────────────────
  echo "[5] RBAC: can-i list nodes as weka-mcp-server-sa..."
  RBAC_NODES=$(kubectl auth can-i list nodes \
    --as="system:serviceaccount:${NAMESPACE}:weka-mcp-server-sa" 2>&1 || true)
  if [[ "${RBAC_NODES}" == "yes" ]]; then
    echo "  PASS: weka-mcp-server-sa can list nodes"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: weka-mcp-server-sa cannot list nodes (got: '${RBAC_NODES}')"
    echo "        Ensure mcp-rbac.yaml was applied: kubectl apply -f k8s/agent-sandbox/mcp-rbac.yaml -n ${NAMESPACE}"
    FAIL=$((FAIL + 1))
  fi

  # ─── Check 6: RBAC — create WekaAppStore CRDs ─────────────────────────────
  echo "[6] RBAC: can-i create wekaappstores.warp.io as weka-mcp-server-sa..."
  RBAC_CRD=$(kubectl auth can-i create wekaappstores.warp.io \
    --as="system:serviceaccount:${NAMESPACE}:weka-mcp-server-sa" \
    --all-namespaces 2>&1 || true)
  if [[ "${RBAC_CRD}" == "yes" ]]; then
    echo "  PASS: weka-mcp-server-sa can create wekaappstores.warp.io"
    PASS=$((PASS + 1))
  else
    echo "  WARN: weka-mcp-server-sa cannot create wekaappstores.warp.io (got: '${RBAC_CRD}')"
    echo "        CRD may not be installed yet. Check: kubectl get crd wekaappstores.warp.io"
    WARNS+=("Check 6: weka-mcp-server-sa create wekaappstores.warp.io returned '${RBAC_CRD}' — CRD may not be installed")
  fi

  # ─── Remaining live checks require a pod ──────────────────────────────────
  if [[ -n "${POD_NAME}" ]]; then
    # ─── Check 7: MCP sidecar container is running ──────────────────────────
    echo "[7] Checking weka-mcp-sidecar container status in pod ${POD_NAME}..."
    SIDECAR_STATE=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="weka-mcp-sidecar")].state.running}' 2>/dev/null || echo "")
    SIDECAR_READY=$(kubectl get pod "${POD_NAME}" -n "${NAMESPACE}" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="weka-mcp-sidecar")].ready}' 2>/dev/null || echo "")
    if [[ "${SIDECAR_READY}" == "true" ]]; then
      echo "  PASS: weka-mcp-sidecar container is running and Ready"
      PASS=$((PASS + 1))
    elif [[ -n "${SIDECAR_STATE}" ]]; then
      echo "  WARN: weka-mcp-sidecar is running but not yet Ready (readinessProbe pending)"
      WARNS+=("Check 7: weka-mcp-sidecar running but not Ready — check logs: kubectl logs ${POD_NAME} -c weka-mcp-sidecar -n ${NAMESPACE}")
    else
      echo "  FAIL: weka-mcp-sidecar container not found or not running in pod ${POD_NAME}"
      echo "        kubectl describe pod ${POD_NAME} -n ${NAMESPACE}"
      FAIL=$((FAIL + 1))
    fi

    # ─── Check 8: MCP sidecar /health endpoint ──────────────────────────────
    echo "[8] Checking MCP sidecar health at localhost:8080/health..."
    HEALTH_RC=0
    kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -c weka-mcp-sidecar -- \
      curl -sf --max-time 10 http://localhost:8080/health > /dev/null 2>&1 || HEALTH_RC=$?
    if [[ "${HEALTH_RC}" -eq 0 ]]; then
      echo "  PASS: MCP sidecar /health endpoint responded (HTTP 200)"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: MCP sidecar /health endpoint did not respond (curl exit ${HEALTH_RC})"
      echo "        Check logs: kubectl logs ${POD_NAME} -c weka-mcp-sidecar -n ${NAMESPACE}"
      FAIL=$((FAIL + 1))
    fi

    # ─── Check 9: openclaw.json generated correctly ──────────────────────────
    echo "[9] Checking openclaw.json content in openclaw container..."
    OPENCLAW_JSON=$(kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -c openclaw -- \
      cat /home/node/.openclaw/openclaw.json 2>&1 || echo "ERROR")
    if echo "${OPENCLAW_JSON}" | grep -q '"transport":"streamable-http"' && \
       echo "${OPENCLAW_JSON}" | grep -q '"url":"http://localhost:8080/mcp"'; then
      echo "  PASS: openclaw.json contains correct transport and url"
      echo "        Content: ${OPENCLAW_JSON}"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: openclaw.json missing required fields or unreadable"
      echo "        Got: ${OPENCLAW_JSON}"
      FAIL=$((FAIL + 1))
    fi

    # ─── Check 10: SKILL.md mounted in openclaw container ────────────────────
    echo "[10] Checking SKILL.md mounted in openclaw container..."
    SKILL_HEAD=$(kubectl exec "${POD_NAME}" -n "${NAMESPACE}" -c openclaw -- \
      head -5 /home/node/.openclaw/SKILL.md 2>&1 || echo "ERROR")
    if [[ "${SKILL_HEAD}" != "ERROR" ]] && [[ -n "${SKILL_HEAD}" ]]; then
      echo "  PASS: SKILL.md is mounted and readable in openclaw container"
      echo "        First 5 lines:"
      echo "${SKILL_HEAD}" | sed 's/^/        /'
      PASS=$((PASS + 1))
    else
      echo "  FAIL: SKILL.md not found or unreadable at /home/node/.openclaw/SKILL.md"
      echo "        Ensure mcp-skill-configmap.yaml was applied and weka-mcp-skill-md ConfigMap exists"
      FAIL=$((FAIL + 1))
    fi
  else
    echo "  SKIP: Checks 7-10 skipped — no pod found"
    WARNS+=("Checks 7-10 skipped: no pod found for selector '${SANDBOX_SELECTOR}'")
  fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 13 Validation Summary ==="
echo "Checks passed: ${PASS}"
echo "Checks failed: ${FAIL}"

if [[ ${#WARNS[@]} -gt 0 ]]; then
  echo ""
  echo "Warnings:"
  for warn in "${WARNS[@]}"; do
    echo "  - ${warn}"
  done
fi

if [[ "${FAIL}" -eq 0 ]]; then
  echo ""
  echo "=== Phase 13 Validation PASSED (${PASS} checks passed, ${#WARNS[@]} warnings) ==="
  exit 0
else
  echo ""
  echo "=== Phase 13 Validation FAILED (${FAIL} failures) ==="
  exit 1
fi
