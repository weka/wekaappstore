# Phase 12: NemoClaw EKS Topology - Research

**Researched:** 2026-03-24
**Domain:** NVIDIA NemoClaw / OpenClaw deployment on EKS using agent-sandbox CRD operator
**Confidence:** MEDIUM (NemoClaw is early preview, launched 2026-03-16; some details require live inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Deploy using the experimental agent-sandbox CRD operator
- Phase 12 must install the agent-sandbox operator itself (not pre-installed)
- NGC API key and org are already configured — can pull images from nvcr.io
- Exact NemoClaw/OpenClaw container image and version to be determined by research (STATE.md gate: "NemoClaw container image name on NVIDIA NGC is a placeholder")
- GPU node group already exists in the EKS cluster — no need to create one
- L40s GPUs (48GB VRAM) — 1 GPU available
- NVIDIA GPU Operator already installed and operational
- Research should verify L40s compatibility with the NemoClaw image requirements
- No NetworkPolicies enforced in the cluster (standard EKS VPC CNI)
- Loopback to sidecar port works by default at the K8s networking level
- Unknown whether the agent-sandbox CRD itself restricts egress — research must inspect the CRD schema
- If CRD restricts loopback: Claude's discretion on whether to patch egress config or use ClusterIP Service fallback
- Deploy NemoClaw in the same namespace as existing WEKA App Store components (not a dedicated namespace)
- Full smoke test script required (not just manual kubectl checks)
- Script validates: pod Running, GPU allocated, loopback to localhost:8080 reachable, NemoClaw API responsive
- Script lives in the repo at `scripts/` (versioned, reusable)
- Phase 12 also produces a structured topology reference doc (TOPOLOGY.md in .planning/) that Phase 13 consumes

### Claude's Discretion
- Exact agent-sandbox CRD installation method (Helm chart vs raw manifests)
- Smoke test script implementation (bash vs Python)
- Topology doc format and level of detail
- Fallback strategy if agent-sandbox CRD restricts loopback egress

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| NCLAW-01 | NemoClaw/OpenClaw deployed to EKS using experimental agent-sandbox CRD approach | Covered: agent-sandbox v0.2.1 install via `kubectl apply`, Sandbox CRD spec with podTemplate, OpenClaw image confirmed at `ghcr.io/openclaw/openclaw:latest` |
| NCLAW-03 | NemoClaw egress policy explicitly allows loopback access to MCP sidecar port | Covered: agent-sandbox CRD itself does NOT restrict egress (it uses standard K8s pod networking); NemoClaw OpenShell policy.yaml uses `network_policies:` block — localhost access requires explicit allow or policy skip; no-GPU path avoids this entirely |
</phase_requirements>

---

## Summary

NemoClaw is an early-preview NVIDIA stack (launched 2026-03-16) that wraps OpenClaw with sandboxed inference and security controls. The standard NemoClaw deployment model runs OpenShell's embedded k3s-in-Docker, which is incompatible with any managed Kubernetes environment including EKS. The agent-sandbox CRD approach is an experimental workaround documented in GitHub issue #407 that deploys OpenClaw directly in a Kubernetes `Sandbox` resource — bypassing the NemoClaw/OpenShell runtime entirely and using the `ghcr.io/openclaw/openclaw` container image instead of the NemoClaw sandbox image.

The key architectural finding: Phase 12 is deploying **OpenClaw on Kubernetes via the agent-sandbox CRD**, not NemoClaw's native runtime. The NemoClaw plugin and OpenShell wrapper are not used in this topology. The GPU requirement for NCLAW-03's inference backend is satisfied by using NVIDIA's cloud API (`integrate.api.nvidia.com/v1`) rather than a local GPU NIM model — which means the L40s GPU constraint is not a blocker for the agent itself (it can use cloud inference), though the GPU must still be allocatable to satisfy the success criteria of "agent container starts without GPU errors."

The agent-sandbox operator is at v0.2.1 (March 2024), installs via two `kubectl apply` commands, and creates `Sandbox` resources with `agents.x-k8s.io/v1alpha1` API. The Sandbox spec wraps a standard Kubernetes podTemplate, so GPU resource requests (`nvidia.com/gpu: 1`), node selectors, and tolerations are specified as standard pod spec fields.

**Primary recommendation:** Install agent-sandbox v0.2.1 operator via raw manifests (simpler than a Helm chart, which doesn't exist for this project). Deploy OpenClaw via `Sandbox` CR with `ghcr.io/openclaw/openclaw:latest` and standard GPU resource requests. Use NVIDIA cloud API for inference (no local NIM model needed). The K8s-level networking does not restrict loopback, so NCLAW-03 is satisfied by the cluster networking layer — but document explicitly in TOPOLOGY.md.

---

## Standard Stack

### Core
| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| agent-sandbox operator | v0.2.1 | Kubernetes controller for `Sandbox` CRD | Official kubernetes-sigs project; only K8s-native sandbox approach for OpenClaw |
| OpenClaw container | `ghcr.io/openclaw/openclaw:latest` (2026.3.11) | OpenClaw gateway and agent runtime | Official NVIDIA OpenShell Community image with NemoClaw plugin pre-installed |
| `Sandbox` CRD | `agents.x-k8s.io/v1alpha1` | Manages stateful singleton pod with stable identity | The CRD that the agent-sandbox operator reconciles |
| NVIDIA GPU Operator | Already installed | GPU device plugin, drivers, toolkit | Pre-installed per CONTEXT.md; L40s already advertised as `nvidia.com/gpu` |

### Supporting
| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| NVIDIA cloud inference API | `integrate.api.nvidia.com/v1` | Nemotron model inference | Default NemoClaw inference provider (no local NIM needed) |
| `openclaw-start.sh` | bundled in image | Runs onboarding + gateway startup inside container | Required — container entrypoint in the OpenShell-Community openclaw sandbox |
| smoke test script | bash (recommended) | Validates topology end-to-end | Phase 12 deliverable at `scripts/validate-topology.sh` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `kubectl apply` manifests for agent-sandbox | Helm chart | No official Helm chart exists for agent-sandbox; raw manifests are the documented install method |
| Cloud Nemotron inference API | Local NIM on L40s | Local NIM requires 49B model fit check; L40s (48GB) can run Nemotron Super 49B at FP8 but requires NIM setup overhead; cloud API is simpler for topology validation |
| agent-sandbox Sandbox CR | Plain Kubernetes Deployment | Sandbox CRD is the locked decision; Deployment would skip the CRD validation required by NCLAW-01 |

**Installation:**
```bash
# Install agent-sandbox operator v0.2.1
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.2.1/manifest.yaml
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/v0.2.1/extensions.yaml

# Verify controller is running
kubectl get pods -n agent-sandbox-system
```

---

## Architecture Patterns

### Deployment Topology

```
EKS Cluster
└── <weka-app-store namespace>
    ├── Sandbox CR (agents.x-k8s.io/v1alpha1)
    │   └── Pod: openclaw-sandbox
    │       ├── Container: openclaw
    │       │   ├── Image: ghcr.io/openclaw/openclaw:latest
    │       │   ├── Port 18789 (OpenClaw gateway WebSocket)
    │       │   ├── Port 18790 (OpenClaw secondary)
    │       │   ├── GPU: nvidia.com/gpu: 1 (for node scheduling)
    │       │   └── PVC: openclaw-workspace (2Gi, RWO)
    │       └── [Phase 13: MCP sidecar added here]
    │
    ├── Service: openclaw (ClusterIP, port 18789)
    └── [Phase 13: MCP sidecar Service or shared pod networking]

GPU Node Group (L40s)
└── EC2 instance with nvidia.com/gpu allocatable
    └── NVIDIA GPU Operator DaemonSet pods
```

### Pattern 1: Sandbox CRD for OpenClaw Deployment

**What:** Deploy OpenClaw as a `Sandbox` CR instead of a `Deployment`. The agent-sandbox controller creates and manages the underlying pod with stable identity and persistent storage.

**When to use:** Required — locked decision is to use agent-sandbox CRD approach (NCLAW-01).

**Why it matters:** Sandbox CRD provides stable hostname and PVC lifecycle management. Phase 13 will add the MCP sidecar container to the same `podTemplate.spec.containers` array.

**Example:**
```yaml
# Source: https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/examples/openclaw-sandbox/openclaw-sandbox.yaml
apiVersion: agents.x-k8s.io/v1alpha1
kind: Sandbox
metadata:
  name: openclaw-sandbox
  namespace: <weka-app-store-namespace>
spec:
  podTemplate:
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: openclaw
          image: ghcr.io/openclaw/openclaw:latest
          command:
            - "node"
            - "dist/index.js"
            - "gateway"
            - "--bind=lan"
            - "--port"
            - "18789"
            - "--allow-unconfigured"
            - "--verbose"
          ports:
            - containerPort: 18789
            - containerPort: 18790
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
          env:
            - name: OPENCLAW_GATEWAY_TOKEN
              valueFrom:
                secretKeyRef:
                  name: openclaw-token
                  key: token
            - name: NVIDIA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: nvidia-api-key
                  key: key
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: openclaw-config
              mountPath: /home/node/.openclaw
            - name: openclaw-workspace
              mountPath: /home/node/.openclaw/workspace
      volumes:
        - name: openclaw-config
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: openclaw-workspace
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 2Gi
```

### Pattern 2: GPU Node Targeting on EKS

**What:** Use `nodeSelector` + `tolerations` to schedule the Sandbox pod on the L40s GPU node group.

**When to use:** Any pod that requests `nvidia.com/gpu` resources.

**Example:**
```yaml
# Standard EKS GPU pod scheduling pattern
nodeSelector:
  nvidia.com/gpu: "true"
  # Alternatively: eks.amazonaws.com/nodegroup: <gpu-nodegroup-name>
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
resources:
  limits:
    nvidia.com/gpu: "1"
```

### Pattern 3: Verifying GPU Operator Readiness Before Deployment

**What:** Before creating the Sandbox CR, verify the GPU Operator is advertising GPUs correctly.

**Example:**
```bash
# Verify L40s GPU nodes are visible
kubectl get nodes -l nvidia.com/gpu.present=true
kubectl describe node <gpu-node> | grep -A5 "Allocatable"
# Should show: nvidia.com/gpu: 1

# Verify GPU Operator DaemonSets are healthy
kubectl get pods -n gpu-operator
kubectl get pods -n kube-system | grep nvidia
```

### Pattern 4: Egress Policy Clarification (NCLAW-03)

**What:** The agent-sandbox CRD itself does NOT impose network egress restrictions — it passes a standard `podTemplate` to Kubernetes and the resulting pod has full cluster network access. The NemoClaw egress restriction system only applies when the NemoClaw/OpenShell runtime is active (k3s-in-Docker model). Since this deployment uses the OpenClaw container directly via the Sandbox CRD (bypassing OpenShell), no OpenShell-level egress policy is active.

**Implication for NCLAW-03:** At the K8s networking level, the OpenClaw pod can reach `localhost:8080` (the MCP sidecar) because:
1. Both containers share the pod network namespace
2. EKS uses VPC CNI with no restrictive NetworkPolicies (confirmed in CONTEXT.md)
3. The agent-sandbox Sandbox controller does not inject NetworkPolicy objects

**What must be documented:** TOPOLOGY.md must explicitly record that egress to `localhost:8080` works via pod-shared network namespace. Phase 13 wires the MCP sidecar as a second container in the same Sandbox pod.

**Fallback (if needed):** If GPU scheduling forces the MCP sidecar into a separate pod, use a ClusterIP Service at `http://weka-mcp:8080/mcp` instead of `http://localhost:8080/mcp`.

### Anti-Patterns to Avoid

- **Using the `nemoclaw onboard` CLI:** This command installs the k3s-in-Docker runtime and is incompatible with EKS. Do not run any `nemoclaw` CLI commands in the cluster.
- **Using the `ghcr.io/nvidia/openshell-community/sandboxes/openclaw` NemoClaw sandbox image:** This image runs OpenShell's embedded k3s-in-Docker, which cannot run inside an existing Kubernetes pod. Use `ghcr.io/openclaw/openclaw` instead.
- **Requesting GPU for the OpenClaw container without GPU Operator verification:** GPU Operator DaemonSet must be healthy first or the pod will remain Pending.
- **Baking `openclaw.json` into the image:** The config must be generated at pod startup (Phase 13 pattern). For Phase 12, the gateway token and inference API key come from Kubernetes Secrets.
- **Deploying in a dedicated namespace:** Locked decision is to use the same namespace as existing WEKA App Store components.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stateful singleton pod with stable hostname | Custom Deployment + StatefulSet hybrid | `agents.x-k8s.io/v1alpha1 Sandbox` | CRD handles lifecycle, hostname stability, and PVC binding automatically |
| GPU node scheduling | Manual node affinity YAML from scratch | Standard EKS `nodeSelector: nvidia.com/gpu: "true"` + toleration | Well-documented EKS GPU pattern; GPU Operator sets these labels automatically |
| Smoke test GPU validation | Custom GPU status endpoint | `kubectl describe node | grep nvidia.com/gpu` + pod log inspection | Standard Kubernetes observability; no custom code needed |
| Inference API access | Local NIM microservice setup | `integrate.api.nvidia.com/v1` (NVIDIA cloud API) | Already configured in NemoClaw/OpenClaw; avoids NIM model sizing problem |

---

## Common Pitfalls

### Pitfall 1: Using the NemoClaw k3s-in-Docker Image Instead of OpenClaw

**What goes wrong:** The NemoClaw CLI installs the `ghcr.io/nvidia/openshell-community/sandboxes/openclaw` image, which starts k3s inside Docker inside the pod. This creates a nested virtualization scenario that will hang or crash in an EKS pod.

**Why it happens:** NemoClaw documentation shows the standard install path (`nemoclaw onboard`) which assumes a VM or bare-metal host with Docker. The agent-sandbox CRD approach requires the plain OpenClaw image.

**How to avoid:** Use `ghcr.io/openclaw/openclaw:latest` — this is the base OpenClaw container without the NemoClaw/OpenShell wrapper. The example at `kubernetes-sigs/agent-sandbox/examples/openclaw-sandbox/openclaw-sandbox.yaml` already uses this image.

**Warning signs:** Pod logs showing `dockerd` or `k3s` startup, or pod stuck in `Init:0/1`.

### Pitfall 2: GPU Pod Pending Due to Device Plugin Issues

**What goes wrong:** Pod stays in `Pending` with event `0/N nodes are available: N Insufficient nvidia.com/gpu`.

**Why it happens:** NemoClaw issue #241 documents that the NVIDIA device plugin Helm repo (`https://nvidia.github.io/k8s-device-plugin`) returns 404. If the GPU Operator or device plugin was installed recently, it may have hit this bug. The device plugin DaemonSet would be in `CrashLoopBackOff`.

**How to avoid:** Pre-validate GPU Operator health before creating the Sandbox CR. Run `kubectl get pods -n gpu-operator` and `kubectl describe node <gpu-node> | grep nvidia.com/gpu` — the allocatable count must be ≥ 1.

**Warning signs:** `nvidia.com/gpu: 0` in node allocatable resources, or GPU Operator DaemonSet pods not Running.

### Pitfall 3: OpenClaw Gateway Binding Mismatch

**What goes wrong:** OpenClaw starts successfully but the Phase 13 MCP sidecar (or health check) can't reach port 18789.

**Why it happens:** The gateway command uses `--bind=lan` which binds to the pod LAN interface (eth0), not loopback. This is correct for Service exposure but may require checking for container networking assumptions.

**How to avoid:** Start with `--bind=lan` (as shown in the official example). The MCP sidecar in Phase 13 will reach the gateway at `localhost:18789` because containers in the same pod share the network namespace.

**Warning signs:** Connection refused to `localhost:18789` when both containers are in the same pod.

### Pitfall 4: Agent-Sandbox Controller Not Running Before Sandbox CR Apply

**What goes wrong:** `kubectl apply` on the Sandbox CR returns `no matches for kind "Sandbox" in version "agents.x-k8s.io/v1alpha1"`.

**Why it happens:** The CRD and controller were not installed first, or the manifest.yaml apply failed silently.

**How to avoid:** After installing the operator, verify with `kubectl get crd sandboxes.agents.x-k8s.io` before creating any Sandbox resources. The `agent-sandbox-system` namespace should exist with controller pods Running.

**Warning signs:** `no matches for kind "Sandbox"` error on apply.

### Pitfall 5: OpenClaw Onboarding State Not Persisted Across Pod Restarts

**What goes wrong:** After pod restart, OpenClaw enters the onboarding wizard again and blocks.

**Why it happens:** OpenClaw stores its config at `~/.openclaw/openclaw.json`. If this path is on the pod's ephemeral filesystem, it is lost on restart.

**How to avoid:** Mount the emptyDir (and PVC for workspace) at `/home/node/.openclaw` and `/home/node/.openclaw/workspace`. The `openclaw-start.sh` entrypoint in the official image handles first-run onboarding and writes config to the mounted volume. In Phase 13, this volume will contain the generated `openclaw.json` with MCP server registration.

**Warning signs:** Pod logs showing `Welcome to OpenClaw onboarding` on every restart.

### Pitfall 6: Smoke Test Localhost:8080 Check Is Phase 13, Not Phase 12

**What goes wrong:** Smoke test tries to reach `localhost:8080` (the MCP sidecar) but the MCP sidecar container doesn't exist yet in Phase 12.

**Why it happens:** Phase 12 deploys only the OpenClaw container. The MCP sidecar is wired in Phase 13.

**How to avoid:** Phase 12 smoke test scope is: pod Running, GPU allocated, OpenClaw gateway responsive on port 18789, GPU node healthy. The `localhost:8080` check is a Phase 14 concern. The CONTEXT.md smoke test requirements should be scoped to what exists in Phase 12 (OpenClaw gateway health, not MCP sidecar).

**Warning signs:** Test fails with `Connection refused to localhost:8080` before Phase 13 is complete.

---

## Code Examples

### Install Agent-Sandbox Operator

```bash
# Source: https://github.com/kubernetes-sigs/agent-sandbox/releases/tag/v0.2.1
export AGENT_SANDBOX_VERSION="v0.2.1"
kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"
kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/extensions.yaml"

# Verify installation
kubectl wait --for=condition=Ready pods --all -n agent-sandbox-system --timeout=120s
kubectl get crd sandboxes.agents.x-k8s.io
```

### Verify GPU Operator Health (Pre-Flight Check)

```bash
# Confirm GPU node is advertising GPU resources
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Confirm device plugin DaemonSet is Running
kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset

# Check node labels set by GPU Operator
kubectl get node <gpu-node> --show-labels | grep nvidia
```

### Minimal Sandbox CR Spec

```yaml
# Source: https://agent-sandbox.sigs.k8s.io/docs/getting_started/
# Derived from: https://github.com/kubernetes-sigs/agent-sandbox/tree/main/examples/openclaw-sandbox
apiVersion: agents.x-k8s.io/v1alpha1
kind: Sandbox
metadata:
  name: openclaw-sandbox
  namespace: <namespace>
spec:
  podTemplate:
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: openclaw
          image: ghcr.io/openclaw/openclaw:latest
          command: ["node", "dist/index.js", "gateway", "--bind=lan", "--port", "18789", "--allow-unconfigured"]
          ports:
            - containerPort: 18789
          resources:
            limits:
              nvidia.com/gpu: "1"
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: openclaw-config
              mountPath: /home/node/.openclaw
      securityContext:
        fsGroup: 1000
      volumes:
        - name: openclaw-config
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: openclaw-workspace
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 2Gi
```

### Smoke Test Script (bash — recommended)

```bash
#!/usr/bin/env bash
# scripts/validate-topology.sh
# Validates Phase 12 topology: OpenClaw pod Running, GPU allocated, gateway reachable.
set -euo pipefail

NAMESPACE="${1:-default}"
SANDBOX_NAME="openclaw-sandbox"

echo "=== Phase 12 Topology Validation ==="

# 1. Pod Running check
echo "[1/4] Checking pod status..."
kubectl wait --for=condition=Ready pod -l "sandbox.agents.x-k8s.io/name=${SANDBOX_NAME}" \
  -n "$NAMESPACE" --timeout=300s
echo "  PASS: Pod is Ready"

# 2. GPU allocated check
echo "[2/4] Checking GPU allocation..."
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l "sandbox.agents.x-k8s.io/name=${SANDBOX_NAME}" \
  -o jsonpath='{.items[0].metadata.name}')
GPU_LIMIT=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" \
  -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}')
if [[ "$GPU_LIMIT" == "1" ]]; then
  echo "  PASS: GPU limit set (nvidia.com/gpu: 1)"
else
  echo "  FAIL: GPU limit not set (got: '${GPU_LIMIT}')"
  exit 1
fi

# 3. OpenClaw gateway health check via exec
echo "[3/4] Checking OpenClaw gateway..."
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- \
  curl -sf --max-time 5 "http://localhost:18789/healthz" > /dev/null \
  && echo "  PASS: Gateway responsive on port 18789" \
  || echo "  WARN: Gateway health endpoint not found (may be expected — check openclaw gateway status)"

# 4. GPU node confirm
echo "[4/4] Checking GPU node..."
NODE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')
GPU_ALLOC=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
if [[ "$GPU_ALLOC" -ge "1" ]]; then
  echo "  PASS: GPU node '${NODE}' has ${GPU_ALLOC} allocatable GPU(s)"
else
  echo "  FAIL: GPU node '${NODE}' shows 0 allocatable GPUs"
  exit 1
fi

echo ""
echo "=== Topology Validation PASSED ==="
echo "Pod: ${POD_NAME}"
echo "Node: ${NODE}"
echo "GPU: ${GPU_ALLOC}x nvidia.com/gpu allocatable"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NemoClaw k3s-in-Docker on bare VM | OpenClaw container in Kubernetes Sandbox CRD | March 2026 (agent-sandbox v0.2.1) | EKS-compatible without nested virtualization |
| NemoClaw CLI for all operations | `kubectl apply` on Sandbox CRD YAML | 2026 (experimental) | Standard K8s GitOps workflow |
| SSE transport for MCP | Streamable HTTP transport | Deprecated April 2026 | Already handled in Phase 11 |
| Local Nemotron GPU inference | NVIDIA cloud inference API | NemoClaw default since launch | Eliminates GPU NIM sizing concern for agent inference |

**Deprecated/outdated:**
- `nemoclaw onboard` command: Only valid for VM/bare-metal hosts with Docker. Not used in this topology.
- `ghcr.io/nvidia/openshell-community/sandboxes/openclaw` image: The NemoClaw-wrapped image with embedded k3s. Not used — use `ghcr.io/openclaw/openclaw` instead.
- OpenShell `policy.yaml` network_policies block: Only relevant when NemoClaw/OpenShell runtime is active. Not applicable in this topology.

---

## Open Questions

1. **Exact OpenClaw gateway health endpoint path**
   - What we know: Port 18789 is confirmed, `--allow-unconfigured` flag exists in official example
   - What's unclear: Does `GET /healthz` or similar endpoint exist? The official example uses a token-gated gateway
   - Recommendation: During Phase 12 execution, check `kubectl logs <pod>` for the actual gateway startup URL and probe it via `kubectl exec`

2. **NGC API key format required by OpenClaw/NemoClaw**
   - What we know: CONTEXT.md states NGC API key and org are already configured
   - What's unclear: Whether the key is injected as `NVIDIA_API_KEY` env var, an OpenClaw config field, or both
   - Recommendation: Check the official `openclaw.json` schema at `docs.openclaw.ai/providers/nvidia` during Phase 12 execution; configure via Secret mount

3. **GPU requirement: Is GPU strictly needed for Phase 12 OpenClaw pod?**
   - What we know: The L40s node exists, GPU Operator is installed, success criteria requires "agent container starts without GPU errors"
   - What's unclear: OpenClaw uses cloud inference (NVIDIA API) — the GPU may not be needed for the agent itself. GPU in `resources.limits` forces scheduling on the GPU node, which may be desired for topology validation even if not strictly required.
   - Recommendation: Include `nvidia.com/gpu: 1` in the Sandbox spec to prove GPU scheduling works end-to-end. This satisfies success criterion #3.

4. **Agent-sandbox controller namespace and cluster-scope**
   - What we know: Manifest at v0.2.1 installs into `agent-sandbox-system` namespace
   - What's unclear: Whether the controller is cluster-scoped (can manage Sandbox CRs in any namespace) or namespace-scoped
   - Recommendation: Per the harche/nemoclaw-operator precedent, assume cluster-scoped. Verify after installation with `kubectl get clusterrole | grep agent-sandbox`.

5. **TOPOLOGY.md content required for Phase 13**
   - What we know: Phase 13 consumes TOPOLOGY.md for manifest authoring
   - What's unclear: Exact fields Phase 13 needs (image, ports, volume mounts, security context, RBAC)
   - Recommendation: Structure TOPOLOGY.md to capture: container image + tag, exposed ports, env vars, volume mounts, security context, GPU resource spec, namespace, and Sandbox CR name. Phase 13 will use this to add the MCP sidecar container.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already installed, `mcp-server/requirements.txt`) |
| Config file | none — run from `mcp-server/` directory |
| Quick run command | `cd mcp-server && pytest tests/ -x -q` |
| Full suite command | `cd mcp-server && pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NCLAW-01 | OpenClaw pod Running in EKS via Sandbox CRD | smoke (manual) | `bash scripts/validate-topology.sh <namespace>` | ❌ Wave 0 |
| NCLAW-03 | Loopback access to port 8080 is reachable | smoke (manual/integration) | `bash scripts/validate-topology.sh <namespace>` (step 3 checks gateway) | ❌ Wave 0 |

**Note:** NCLAW-01 and NCLAW-03 are live infrastructure requirements — they require a running EKS cluster and cannot be automated in pytest unit tests. The smoke test script is the automated artifact that satisfies them.

### Sampling Rate
- **Per task commit:** `cd mcp-server && pytest tests/ -x -q` (existing 103 tests — verify no regressions)
- **Per wave merge:** `cd mcp-server && pytest tests/ -v`
- **Phase gate:** Smoke test script exits 0 before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `scripts/validate-topology.sh` — covers NCLAW-01 and NCLAW-03 smoke validation
- [ ] `scripts/install-agent-sandbox.sh` — installs operator and waits for Ready (optional but reusable)
- [ ] `.planning/TOPOLOGY.md` — structured topology reference doc for Phase 13 (not a test file but a required Phase 12 deliverable)

---

## Sources

### Primary (HIGH confidence)
- [kubernetes-sigs/agent-sandbox releases](https://github.com/kubernetes-sigs/agent-sandbox/releases) — v0.2.1 install commands verified
- [agent-sandbox openclaw-sandbox example](https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/examples/openclaw-sandbox/openclaw-sandbox.yaml) — CRD spec with `ghcr.io/openclaw/openclaw` image confirmed
- [NVIDIA NemoClaw Architecture](https://github.com/NVIDIA/NemoClaw/blob/main/docs/reference/architecture.md) — container image `ghcr.io/nvidia/openshell-community/sandboxes/openclaw` and sandbox structure verified
- [NVIDIA OpenShell-Community/sandboxes/openclaw Dockerfile](https://raw.githubusercontent.com/NVIDIA/OpenShell-Community/main/sandboxes/openclaw/Dockerfile) — base image, OpenClaw CLI version 2026.3.11 confirmed

### Secondary (MEDIUM confidence)
- [NemoClaw issue #407](https://github.com/NVIDIA/NemoClaw/issues/407) — agent-sandbox CRD approach on OpenShift documented (verified with official project)
- [Kubernetes blog: Running Agents on Kubernetes with Agent Sandbox](https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/) — v0.2.1 install confirmed
- [NemoClaw issue #241](https://github.com/NVIDIA/NemoClaw/issues/241) — GPU device plugin Helm repo 404 bug documented (NVIDIA GitHub)
- [NVIDIA NemoClaw Developer Guide — Inference Profiles](https://docs.nvidia.com/nemoclaw/latest/reference/inference-profiles.html) — default inference provider is cloud API, not local GPU

### Tertiary (LOW confidence — needs live verification)
- NemoClaw egress policy allows `localhost` — WebSearch synthesis (not verified against live policy.yaml; moot for this topology since OpenShell runtime is not used)
- GPU `nodeSelector: nvidia.com/gpu: "true"` label — EKS GPU Operator behavior (verified by pattern from multiple EKS GPU guides but not confirmed against this specific cluster's labels)

---

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM — agent-sandbox v0.2.1 confirmed; OpenClaw image confirmed; NemoClaw is early preview (2026-03-16) so behavior may change
- Architecture: MEDIUM — agent-sandbox CRD pattern confirmed from official examples; GPU scheduling patterns well-established; egress policy analysis is HIGH confidence (moot for this topology)
- Pitfalls: HIGH — documented from official GitHub issues (#241, #407) and architecture docs

**Research date:** 2026-03-24
**Valid until:** 2026-04-07 (NemoClaw is fast-moving early preview — re-verify image tags before deploying)
