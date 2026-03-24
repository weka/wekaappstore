# EKS Topology Reference

**Generated:** 2026-03-24
**Cluster:** warp-cluster-udp (arn:aws:eks:us-west-2:130745022161:cluster/warp-cluster-udp)
**Namespace:** wekaappstore
**Status:** Validated via scripts/validate-topology.sh — all 5 checks PASS

## Sandbox Resource

| Field | Value |
|-------|-------|
| API Version | agents.x-k8s.io/v1alpha1 |
| Kind | Sandbox |
| Name | openclaw-sandbox |
| Namespace | wekaappstore |
| Manifest | k8s/agent-sandbox/openclaw-sandbox.yaml |

## OpenClaw Container

| Field | Value |
|-------|-------|
| Image | ghcr.io/openclaw/openclaw:latest |
| Image Digest | ghcr.io/openclaw/openclaw@sha256:dda9f4b94761e87864c901cf34ee858daf89fa1deed9dcb671e8845a6b24062e |
| Pod Name | openclaw-sandbox |
| Ports | 18789 (gateway WebSocket), 18790 (secondary) |
| GPU | nvidia.com/gpu: 1 (limits and requests) |
| Bind Mode | --bind=loopback (gateway listens on 127.0.0.1:18789) |
| Security Context | runAsNonRoot, uid 1000, drop ALL |
| Volume Mounts | /home/node/.openclaw (emptyDir), /home/node/.openclaw/workspace (PVC 2Gi) |
| Restart Count | 0 (stable) |

### Gateway Command

```
node dist/index.js gateway --bind=loopback --port 18789 --allow-unconfigured --verbose
```

**Note:** `--bind=loopback` used instead of `--bind=lan` because non-loopback bind requires `gateway.controlUi.allowedOrigins` in config. Loopback bind is correct for sidecar deployment — all containers in the same pod share a network namespace and can reach `localhost:18789`.

## GPU Node

| Field | Value |
|-------|-------|
| Node Name | ip-172-3-1-203.us-west-2.compute.internal |
| Instance Type | g5.4xlarge |
| GPU Product | NVIDIA A10G |
| GPU Family | ampere |
| GPU Count | 1 allocatable |
| GPU Memory | 23028 MiB |
| GPU Compute | compute major: 8, minor: 6 |
| Node Selector | nvidia.com/gpu: "true" |
| Tolerations | nvidia.com/gpu Exists NoSchedule |
| Allocated (node) | nvidia.com/gpu: 1 request / 1 limit |

## Networking

| Field | Value |
|-------|-------|
| Pod Network | Shared namespace (all containers see localhost) |
| NetworkPolicies | None enforced (standard EKS VPC CNI) |
| Agent-Sandbox Egress | No restrictions (CRD passes through standard pod networking) |
| Loopback Access | Confirmed working — NCLAW-03 validated |
| Loopback :8080 Path | PASS (connection refused — path open, no listener pre-Phase-13) |
| Gateway Bind | --bind=loopback (127.0.0.1:18789) |
| Gateway WebSocket URL | ws://127.0.0.1:18789 |

## Phase 13 Integration Points

- MCP sidecar goes in: `spec.podTemplate.spec.containers[]` (second container)
- MCP sidecar reaches gateway at: `localhost:18789`
- Gateway reaches MCP sidecar at: `localhost:8080` (Phase 11 MCP server port)
- openclaw.json generation: init container writes to shared emptyDir volume at `/home/node/.openclaw`
- Blueprint volume: add to `spec.podTemplate.spec.volumes[]` and mount in sidecar
- Sandbox CR pattern: Phase 13 modifies `k8s/agent-sandbox/openclaw-sandbox.yaml` spec, NOT a Deployment spec

### MCP Sidecar Connection Config

```json
{
  "mcpServers": {
    "weka-app-store": {
      "transport": "streamable-http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

This config goes in `/home/node/.openclaw/openclaw.json` written by init container at pod startup.

## Secrets

| Secret Name | Key | Purpose |
|-------------|-----|---------|
| openclaw-token | token | Gateway authentication (OPENCLAW_GATEWAY_TOKEN env var) |
| nvidia-api-key | key | NVIDIA cloud inference API (NVIDIA_API_KEY env var) |

## Operator Labels

The agent-sandbox operator sets a hash-based label on the pod (not the Sandbox name directly):

| Label | Value |
|-------|-------|
| agents.x-k8s.io/sandbox-name-hash | 62f96e10 |

Use `kubectl get sandbox openclaw-sandbox -n wekaappstore -o jsonpath='{.status.selector}'` to get the current selector dynamically.

## Key Decision

Topology uses agent-sandbox CRD (not plain Deployment) per user decision. This means Phase 13 modifies the Sandbox CR spec, not a Deployment spec.

**Validated Phase 12:** OpenClaw pod Running with GPU on EKS in wekaappstore namespace. Loopback networking confirmed working (NCLAW-03). All 5 smoke test checks PASS.
