# Stack Research

**Domain:** MCP server in Python — WEKA App Store OpenClaw tool integration
**Researched:** 2026-03-23
**Confidence:** MEDIUM-HIGH — MCP SDK Streamable HTTP verified via PyPI and gofastmcp.com docs (HIGH); OpenClaw K8s operator sidecar pattern verified via official docs (HIGH); openclaw.json HTTP mcpServers format verified via multiple community sources (MEDIUM); NemoClaw-specific EKS details still LOW due to alpha status

---

## Context: Milestone Scope — v3.0 Additions Only

This document **extends** the v2.0 STACK.md. Everything in the previous STACK.md remains valid and is not repeated here. The v2.0 validated stack is:

- `mcp[cli]>=1.26.0` — FastMCP with stdio transport (production)
- `kubernetes>=27.0.0` — K8s API client (reused by tools)
- `PyYAML>=6.0.1` — YAML parsing (reused by validate/apply)
- `pytest>=8.0.0`, `pytest-asyncio` — test runner

**v3.0 adds:** Streamable HTTP transport on the MCP server, OpenClaw/NemoClaw deployment to EKS via the `openclaw-rocks` Kubernetes operator, and sidecar wiring to register the MCP server over HTTP.

---

## Recommended Stack — New Additions for v3.0

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `mcp[cli]` (Streamable HTTP transport) | `>=1.26.0` (already pinned) | Add HTTP transport mode to existing FastMCP server | No new package needed — Streamable HTTP is built into `mcp>=1.9`. Call `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)` and the `/mcp` endpoint is live. The existing `mcp[cli]>=1.26.0` pin already covers this. |
| OpenClaw Kubernetes Operator | `oci://ghcr.io/openclaw-rocks/charts/openclaw-operator` (latest) | Deploy and manage OpenClaw agent instances on EKS via the `OpenClawInstance` CRD | Official community operator from `openclaw-rocks/k8s-operator`. Manages the full pod (StatefulSet, Service, RBAC, NetworkPolicy, PVC), handles sidecar injection via `spec.sidecars`, and accepts arbitrary custom containers running alongside the main agent container. Verified via official openclaw.rocks docs. |
| OpenClaw container image | `alpine/openclaw:2026.3.11` (or latest `alpine/openclaw`) | Agent runtime in the OpenClawInstance pod | Verified via dev.to deployment guide (March 2026). Exposes gateway on port 18789. The operator's nginx gateway-proxy sidecar handles external traffic forwarding to the loopback-bound gateway. |

### Supporting Libraries

No new Python libraries are required. All Streamable HTTP transport support is already included in `mcp>=1.9`. The sidecar pattern only requires Kubernetes manifests/YAML changes, not Python code changes.

| What | Where | Notes |
|------|-------|-------|
| Streamable HTTP server | `mcp.run()` call in `server.py` | Switch transport arg from default stdio to `"streamable-http"` for the HTTP variant. See integration section below. |
| Environment variable `MCP_PORT` | `server.py` and `config.py` | Expose port as env var so the Kubernetes manifest can configure it without rebuilding the image. Default: `8080`. |
| Kubernetes `ConfigMap` for openclaw config | `OpenClawInstance` spec | MCP server registration goes into `spec.config.raw.agents.defaults.mcpServers` as a `streamable-http` entry pointing to `http://localhost:8080/mcp`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `helm` v3.x | Install the openclaw-rocks operator on EKS | `helm install openclaw-operator oci://ghcr.io/openclaw-rocks/charts/openclaw-operator --namespace openclaw-operator-system --create-namespace` |
| `kubectl port-forward` | Local access to OpenClaw gateway during dev | `kubectl port-forward svc/my-agent 18789:18789 -n openclaw` — exposes the agent chat gateway locally |
| `mcp dev server.py` | Interactive MCP Inspector for HTTP transport smoke-test | Already available via `mcp[cli]`; run with `--transport streamable-http` flag to test HTTP mode |

---

## Installation

No new Python packages. All changes are in existing code and Kubernetes manifests.

```bash
# Verify mcp[cli]>=1.26.0 is already installed (it should be from v2.0)
pip show mcp

# Install OpenClaw operator on EKS (one-time cluster setup)
helm install openclaw-operator \
  oci://ghcr.io/openclaw-rocks/charts/openclaw-operator \
  --namespace openclaw-operator-system \
  --create-namespace
```

---

## Integration Points

### 1. Adding HTTP Transport to server.py

The MCP server currently runs stdio only. Add HTTP support by reading a transport env var:

```python
# mcp-server/server.py (additions only)
import os

if __name__ == "__main__":
    from config import validate_required
    validate_required()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        port = int(os.environ.get("MCP_PORT", "8080"))
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run()  # stdio default — unchanged for local dev
```

**Key facts (HIGH confidence — verified via PyPI and gofastmcp.com):**
- Transport string is `"streamable-http"` (hyphen, not underscore)
- MCP endpoint is always at `/mcp` path — `http://localhost:8080/mcp`
- `host="0.0.0.0"` is required in a container so the sidecar is reachable from within the pod
- No new dependencies needed — Streamable HTTP is included in `mcp>=1.9`
- For a stateless server (no per-session state), initialize with `FastMCP("name", stateless_http=True)` for horizontal scaling compatibility

**Also update config.py** to add `MCP_TRANSPORT` and `MCP_PORT` as optional env vars so they appear in documentation and validation output.

### 2. Dockerfile Update

The existing `CMD ["python", "-m", "server"]` runs stdio. For the sidecar container, the deployment YAML overrides this via `args`:

```yaml
# In the OpenClawInstance sidecar spec — no Dockerfile change needed
containers:
- name: weka-mcp-server
  image: wekachrisjen/weka-app-store-mcp:latest
  args: ["python", "-m", "server"]
  env:
  - name: MCP_TRANSPORT
    value: "streamable-http"
  - name: MCP_PORT
    value: "8080"
  - name: BLUEPRINTS_DIR
    value: "/blueprints"
  - name: KUBERNETES_AUTH_MODE
    value: "in-cluster"
```

The existing Dockerfile CMD stays as-is. The container behaves as stdio by default and as HTTP when `MCP_TRANSPORT=streamable-http` is injected.

### 3. OpenClawInstance CRD — Sidecar Registration

The operator manages OpenClaw agent pods via the `OpenClawInstance` CRD. To add the MCP server sidecar and register it over HTTP:

```yaml
apiVersion: openclaw.rocks/v1alpha1
kind: OpenClawInstance
metadata:
  name: weka-agent
  namespace: openclaw
spec:
  envFrom:
    - secretRef:
        name: openclaw-api-keys        # ANTHROPIC_API_KEY or OPENAI_API_KEY
  storage:
    persistence:
      enabled: true
      size: 10Gi
  # MCP server sidecar — communicates with the agent over localhost
  sidecars:
    - name: weka-mcp-server            # must not collide with reserved names
      image: wekachrisjen/weka-app-store-mcp:latest
      ports:
        - containerPort: 8080
          name: mcp-http
      env:
        - name: MCP_TRANSPORT
          value: "streamable-http"
        - name: MCP_PORT
          value: "8080"
        - name: BLUEPRINTS_DIR
          value: "/blueprints"
        - name: KUBERNETES_AUTH_MODE
          value: "in-cluster"
      resources:
        requests:
          cpu: 200m
          memory: 256Mi
        limits:
          cpu: 500m
          memory: 512Mi
      # readOnlyRootFilesystem enforced by operator for custom sidecars
  # Register the sidecar MCP server in OpenClaw's config
  config:
    raw:
      agents:
        defaults:
          mcpServers:
            weka-app-store:
              transport: "streamable-http"
              url: "http://localhost:8080/mcp"
```

**Sidecar naming constraints (HIGH confidence — operator docs):**
Reserved container names rejected by the operator webhook: `openclaw`, `gateway-proxy`, `chromium`, `tailscale`, `ollama`, `web-terminal`. Use `weka-mcp-server`.

**Port choice:** Port `8080` avoids conflict with OpenClaw's gateway on `18789` and the bridge on `18790`.

**localhost routing:** Both containers share a network namespace in the pod. The MCP server on `localhost:8080` is directly reachable from the OpenClaw agent container — no Service or ingress needed.

### 4. openclaw.json Update (from stdio to HTTP)

The existing `openclaw.json` in the repo documents the stdio startup contract. For the EKS sidecar deployment, this changes to an HTTP endpoint:

```json
{
  "name": "weka-app-store-mcp",
  "description": "MCP server for the WEKA App Store.",
  "transport": "streamable-http",
  "url": "http://localhost:8080/mcp",
  "env": {
    "required": ["BLUEPRINTS_DIR"],
    "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "MCP_PORT"]
  },
  "container": "wekachrisjen/weka-app-store-mcp:latest",
  "skill": "mcp-server/SKILL.md"
}
```

The `startup` block (command/args for stdio) is removed for the HTTP deployment variant. Keep the stdio block for local dev documentation.

### 5. RBAC — ServiceAccount for In-Cluster K8s Access

The MCP server sidecar uses `KUBERNETES_AUTH_MODE=in-cluster` to call the K8s API. The `OpenClawInstance` operator creates a ServiceAccount, but it needs to be annotated with permission to list/get pods, nodes, namespaces, storageclasses, and custom resources:

```yaml
# ClusterRole for MCP server tools (read-only + WekaAppStore apply)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: weka-mcp-server-role
rules:
  - apiGroups: [""]
    resources: ["nodes", "namespaces", "pods", "persistentvolumes"]
    verbs: ["get", "list"]
  - apiGroups: ["storage.k8s.io"]
    resources: ["storageclasses"]
    verbs: ["get", "list"]
  - apiGroups: ["warp.io"]
    resources: ["wekaappstores"]
    verbs: ["get", "list", "create"]
  - apiGroups: ["apiextensions.k8s.io"]
    resources: ["customresourcedefinitions"]
    verbs: ["get", "list"]
```

Bind this to the ServiceAccount the operator creates for the `OpenClawInstance` pod.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Streamable HTTP on port 8080 inside pod | stdio subprocess spawned by OpenClaw | Use stdio for local dev and single-user desktop scenarios — it's simpler and is the current production config. Use HTTP for Kubernetes sidecar because OpenClaw can't spawn a subprocess from inside a co-located container |
| `spec.sidecars` custom container in `OpenClawInstance` | Separate Kubernetes Deployment with ClusterIP Service | Use separate Deployment only if the MCP server needs to be shared across multiple OpenClaw agent instances. For a single-agent pod, sidecar is simpler, shares network namespace, and requires no inter-pod DNS or service |
| `openclaw-rocks/k8s-operator` Helm chart | Raw manifests (Deployment, Service, PVC, RBAC by hand) | Use raw manifests if the operator adds unacceptable overhead or if the cluster policy blocks CRD installation. The operator Helm chart is the official documented path and handles all the pod assembly complexity |
| Port 8080 for MCP HTTP | Port 3721 (openclaw-mcp-server community default) | Port choice is arbitrary inside the pod. Use 8080 because it's the de facto HTTP alt-port convention and avoids confusion with OpenClaw's own ports (18789, 18790) |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| SSE transport (`"sse"`) | SSE transport is deprecated per the MCP spec as of March 2025 and superseded by Streamable HTTP. Some clients may still support it but it is being removed | `transport="streamable-http"` |
| `stateless_http=True` on this server | The WEKA MCP tools are already stateless by design (no session state). Setting `stateless_http=True` disables the MCP session-id mechanism entirely, which may break clients that expect session tracking. Leave it at default (stateful session management is handled by the SDK, not by the tools) | Default `FastMCP("name")` without `stateless_http` |
| `asyncio_mode = "auto"` in pytest config | Auto mode causes pytest-asyncio to claim all async tests globally, conflicting with anyio (pulled in by `mcp`). Already flagged in v2.0 STACK.md | `asyncio_mode = "strict"` with explicit `@pytest.mark.asyncio` |
| Exposing the MCP HTTP port as a Kubernetes Service | The MCP server is an agent-private tool endpoint — no other workload should call it directly. Exposing it as a Service opens a security surface and is not needed when the sidecar and agent are co-located | Keep MCP on localhost within the pod; no Service, no Ingress |
| NIM/NemoClaw as the agent runtime for this milestone | NemoClaw is alpha (announced March 16 2026) with no published Kubernetes operator or stable config schema. OpenClaw is the stable deployment target for v3.0. NemoClaw is a future milestone item | `alpine/openclaw` + `openclaw-rocks` operator |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `mcp[cli]>=1.26.0` | Streamable HTTP | Confirmed available in `mcp>=1.9`. `1.26.0` is the latest stable (PyPI, Jan 24 2026). No separate package needed for HTTP transport. |
| `mcp[cli]>=1.26.0` | Python >=3.10 | Existing container uses `python:3.10-slim` — compatible |
| OpenClaw operator (latest) | Kubernetes 1.28+ | EKS current default AMIs (1.29, 1.30, 1.31) are all compatible |
| `alpine/openclaw:2026.3.11` | OpenClaw operator latest | Verified via dev.to guide (March 2026) |

---

## Sources

- PyPI `mcp` package — v1.26.0 confirmed latest stable (Jan 24 2026): https://pypi.org/project/mcp/
- gofastmcp.com deployment docs — `mcp.run(transport="streamable-http", host, port)` signature: https://gofastmcp.com/deployment/running-server
- MCPcat Streamable HTTP guide — `/mcp` endpoint path, production config: https://mcpcat.io/guides/building-streamablehttp-mcp-server/
- OpenClaw K8s operator official install docs: https://docs.openclaw.ai/install/kubernetes
- openclaw-rocks/k8s-operator GitHub — `spec.sidecars`, reserved container names, sidecar architecture: https://github.com/openclaw-rocks/k8s-operator
- DeepWiki openclaw-rocks/k8s-operator sidecar docs — three sidecar categories, gateway-proxy constraint: https://deepwiki.com/openclaw-rocks/k8s-operator/5.1-sidecar-containers
- openclaw.rocks deploy guide — `OpenClawInstance` CRD YAML with `spec.sidecars`: https://openclaw.rocks/blog/deploy-openclaw-kubernetes
- Community `openclaw-mcp-server` reference — `streamable-http` URL format `http://HOST:PORT/mcp`: https://github.com/rodgco/openclaw-mcp-server
- masteryodaa/openclaw-sdk DeepWiki — `HttpMcpServer` `transport: "streamable-http"`, `url` field: https://deepwiki.com/masteryodaa/openclaw-sdk/2.15-mcp-server-integration
- dev.to OpenClaw on Kubernetes — `alpine/openclaw:2026.3.11` image, port 18789: https://dev.to/thenjdevopsguy/running-openclaw-on-kubernetes-57ki
- MCP spec Streamable HTTP (March 26 2025) — official transport introduction: https://blog.cloudflare.com/streamable-http-mcp-servers-python/

---

*Stack research for: Streamable HTTP transport + NemoClaw/OpenClaw EKS sidecar deployment (WEKA App Store v3.0 milestone)*
*Researched: 2026-03-23*
*Extends: previous STACK.md (v2.0 stdio MCP server — still valid, not replaced)*
