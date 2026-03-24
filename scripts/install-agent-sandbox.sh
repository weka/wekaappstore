#!/usr/bin/env bash
# scripts/install-agent-sandbox.sh
# Installs the agent-sandbox operator v0.2.1 on an EKS cluster and verifies readiness.
# Usage: bash scripts/install-agent-sandbox.sh
# Requires: kubectl configured with appropriate cluster access
set -euo pipefail

AGENT_SANDBOX_VERSION="v0.2.1"
GITHUB_BASE="https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}"

echo "=== Installing agent-sandbox operator ${AGENT_SANDBOX_VERSION} ==="

# Step 1: Apply operator manifests from GitHub releases
echo "[1/4] Applying manifest.yaml..."
kubectl apply -f "${GITHUB_BASE}/manifest.yaml"

echo "[1/4] Applying extensions.yaml..."
kubectl apply -f "${GITHUB_BASE}/extensions.yaml"

# Step 2: Wait for controller to be Ready
echo "[2/4] Waiting for agent-sandbox-system pods to be Ready (timeout: 120s)..."
kubectl wait --for=condition=Ready pods --all \
  -n agent-sandbox-system \
  --timeout=120s
echo "  PASS: All agent-sandbox-system pods are Ready"

# Step 3: Verify CRD is registered
echo "[3/4] Verifying Sandbox CRD is registered..."
if kubectl get crd sandboxes.agents.x-k8s.io > /dev/null 2>&1; then
  echo "  PASS: CRD sandboxes.agents.x-k8s.io is registered"
else
  echo "  FAIL: CRD sandboxes.agents.x-k8s.io not found — operator install may have failed"
  exit 1
fi

# Step 4: Print success message with controller pod status
echo "[4/4] Controller pod status:"
kubectl get pods -n agent-sandbox-system

echo ""
echo "=== agent-sandbox operator ${AGENT_SANDBOX_VERSION} installed successfully ==="
echo "You can now apply Sandbox CRs with: kubectl apply -f k8s/agent-sandbox/openclaw-sandbox.yaml -n <NAMESPACE>"
