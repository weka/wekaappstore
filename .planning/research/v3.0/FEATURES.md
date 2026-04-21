# Feature Research

**Domain:** Live EKS Deployment — Streamable HTTP MCP Transport + NemoClaw/OpenClaw Integration
**Researched:** 2026-03-23
**Confidence:** HIGH for transport mechanics and FastMCP API; MEDIUM for NemoClaw/OpenClaw-specific config (early preview, evolving); LOW for NemoClaw EKS-specific Kubernetes YAML (insufficient official detail yet)

---

## Context: What Is Already Built (Do Not Rebuild)

This milestone is v3.0. The following exist and must NOT be touched unless a specific task requires it:

| Already Shipped | Location |
|-----------------|----------|
| 8-tool MCP server (stdio transport) | `mcp-server/server.py` |
| 103 tests covering all tools | `mcp-server/tests/` |
| SKILL.md agent workflow | `mcp-server/SKILL.md` |
| Mock agent harness | `mcp-server/harness/` |
| Dockerfile + CI/CD to Docker Hub | `mcp-server/Dockerfile`, `.github/workflows/` |
| openclaw.json (stdio config) | `mcp-server/openclaw.json` |
| FastMCP scaffold via `mcp.server.fastmcp.FastMCP` | `mcp-server/server.py` |

**What this milestone adds:** Streamable HTTP transport mode, NemoClaw deployment to EKS, sidecar wiring, and end-to-end happy-path validation with a real agent.

---

## Feature Landscape

### Table Stakes (Required for Milestone to Succeed)

These features must exist before the live agent chat experience is possible. Missing any one of them breaks the deployment or integration loop.

| Feature | Why Required | Complexity | Dependencies on Existing |
|---------|--------------|------------|--------------------------|
| Streamable HTTP transport on MCP server | OpenClaw registers MCP tools via HTTP URL in sidecar pattern, not stdio subprocess in K8s. Without HTTP transport the agent cannot reach the server. | LOW | `FastMCP.run(transport="streamable-http")` or `mcp.http_app()`. No tool code changes. Existing `server.py` adds one run-mode branch. |
| HTTP server runs on configurable port (default 8000) | Port must be consistent with the OpenClaw registration config. Configurable so it can be changed without rebuild. | LOW | Add `MCP_PORT` env var to `config.py`. Pass to `mcp.run()` call. |
| Single `/mcp` endpoint path (POST + GET) | MCP Streamable HTTP spec requires one endpoint supporting both POST (client requests) and GET (SSE subscription). This is the canonical MCP endpoint. | LOW | FastMCP `path` parameter defaults to `/mcp/`. No additional routing needed. |
| Session ID header (`Mcp-Session-Id`) management | Spec requires servers to assign a session ID at init and clients to include it on subsequent requests. FastMCP handles this automatically. | LOW | Built into FastMCP. No custom code. |
| Origin header validation (security) | MCP spec security requirement: servers MUST validate `Origin` header to prevent DNS rebinding attacks. Required on non-localhost HTTP deployments. | LOW-MEDIUM | FastMCP includes this by default on Starlette-backed servers when binding to non-loopback addresses. Verify in deployment config. |
| Container exposes port 8000 | Dockerfile EXPOSE and Kubernetes containerPort must match the MCP endpoint port so the OpenClaw container can reach the sidecar on `localhost:8000/mcp`. | LOW | Add `EXPOSE 8000` to Dockerfile and update entrypoint to use HTTP transport. |
| `openclaw.json` updated for HTTP transport | Registration config currently specifies `"transport": "stdio"` with a command/args startup block. For sidecar deployment, it must use `"transport": "streamable-http"` with a `"url": "http://localhost:8000/mcp"` field. | LOW | Edit `mcp-server/openclaw.json`. No code changes. Confirmed format from community docs and OpenClaw mcpServers pattern. |
| NemoClaw deployed to EKS cluster | The agent must be running somewhere. NemoClaw is the deployment target for this milestone. | MEDIUM | Requires NVIDIA NemoClaw (early preview as of 2026-03-16). Uses OpenShell sandbox runtime. Installer script or Helm chart. EKS GPU node required for NIM inference. |
| MCP server container deployed as sidecar in NemoClaw pod | OpenClaw/NemoClaw communicates with MCP servers on `localhost` inside the pod. The MCP server must be a sidecar container, not a separate Service. Both containers must be in the same pod spec. | MEDIUM | `spec.sidecars` in OpenClaw operator CRD, or equivalent NemoClaw pod spec. Container name must not conflict with reserved names (`openclaw`, `gateway-proxy`, etc.). |
| Environment variables injected into MCP sidecar | The MCP server needs `BLUEPRINTS_DIR`, `KUBERNETES_AUTH_MODE`, and optionally `KUBECONFIG` or in-cluster service account. These must be passed via pod env or ConfigMap/Secret. | LOW-MEDIUM | Same env var model as current Dockerfile. In-cluster auth (`KUBERNETES_AUTH_MODE=incluster`) preferred on EKS. |
| Agent can complete happy-path blueprint deployment end-to-end | This is the milestone acceptance criterion. The agent must call inspect_cluster, inspect_weka, list_blueprints, get_blueprint, get_crd_schema, validate_yaml, apply, and status against a real cluster and real WEKA. | HIGH | Requires all of the above plus: live EKS cluster, live WEKA, NemoClaw running, tools registered. |
| SKILL.md updated to reference HTTP transport context | SKILL.md may contain hints about stdio startup or local invocation. Any such references must be updated or removed for the HTTP deployment model. | LOW | Review `mcp-server/SKILL.md` for transport-specific language. Minor edit if needed. |

### Differentiators (Competitive Advantage)

These features are not strictly required for the milestone acceptance criterion but meaningfully improve the experience or reduce operational risk.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dual transport support (HTTP + stdio) | Preserve stdio mode for local development and mock harness while adding HTTP for EKS. Developers can still run `python -m server` locally. No CI/CD breakage. | LOW-MEDIUM | FastMCP transport is selected at run time via `if __name__ == "__main__"` branch or env var. Existing stdio tests and harness continue to work unchanged. |
| Health endpoint at `/healthz` | Kubernetes liveness and readiness probes need an HTTP endpoint. Without it, the pod has no signal for restart logic. Standard Kubernetes expectation for any HTTP sidecar. | LOW | Add a small FastAPI/Starlette route outside the MCP path, or use a separate health server. FastMCP's Starlette app can have additional routes mounted. |
| In-cluster RBAC: minimal service account | MCP server needs read access to Kubernetes resources (pods, nodes, namespaces, storage classes) and write access to WekaAppStore CRs. A purpose-built ServiceAccount with minimal Role/RoleBinding is cleaner than using a broad cluster-admin credential. | MEDIUM | Create `ServiceAccount`, `Role`, `RoleBinding` in the deployment manifests. Verbs: `get`, `list`, `watch` on nodes/namespaces/storageclasses; `create`, `get`, `list`, `watch` on WekaAppStore CRs. |
| `docker-compose` for local HTTP mode | Lets developers test the HTTP transport locally before EKS deployment. Simulates the sidecar network topology with containers on the same Docker network. | LOW | Add a `docker-compose.yml` with OpenClaw and MCP server containers on shared network. One command for local E2E. |
| E2E validation script against real cluster | A script or pytest scenario that exercises the full agent tool chain against the live EKS cluster and WEKA, verifiable without manual OpenClaw chat. Confirms tool wiring is correct before running the full agent experience. | MEDIUM | Extend the existing harness or add a separate integration test mode. Run against real K8s + real WEKA via env var flag. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Exposing MCP server as a separate Kubernetes Service with NodePort or LoadBalancer | Seems natural — makes the server accessible without sidecar complexity. | Breaks the localhost communication model. Adds network hops, TLS requirements, and authentication complexity. OpenClaw's MCP integration is designed around localhost/sidecar for in-cluster use. Networking surface increases security risk. | Keep MCP server as sidecar. OpenClaw reaches it on `localhost:8000/mcp`. |
| Running both stdio and HTTP simultaneously in one process | Appealing as a "flexible" deployment. | MCP stdio assumes stdin/stdout are the protocol stream. Running HTTP in the same process that is also listening on stdin for MCP messages creates a conflict. The two transports are architecturally separate modes. | Select transport via env var (`MCP_TRANSPORT=http` or `MCP_TRANSPORT=stdio`) at startup. Single process, single mode. |
| Persisting SSE streams as long-lived connections in K8s | Seems necessary for real-time tool progress. | Long-lived SSE in K8s sidecars requires careful resource limits, liveness probe tuning, and reconnect logic. The apply tool completes in one request (CR submission is fast). Operator does async work independently. | Return HTTP 200 synchronously for tool calls. Use status polling (existing `status` tool) for post-apply progress. FastMCP supports both SSE and sync response — let the SDK decide. |
| Adding agent planning logic or YAML generation to the MCP server | Developers want to reduce agent load. | Reintroduces the v1.0 backend-brain anti-pattern that was explicitly removed. The entire architectural pivot was from backend-brain to OpenClaw-native reasoning. Adding planning logic back means the agent's YAML output and the server's YAML generation diverge. | Keep all reasoning in OpenClaw. Server provides tools and CRD schema. Agent generates YAML. Server validates it. |
| Storing NemoClaw API keys in MCP server environment | Seems convenient for a single-pod deployment. | The MCP server never needs the model API key. Only NemoClaw needs it. Spreading credentials across containers increases blast radius. | Store model API keys only in the OpenClaw/NemoClaw container env or Kubernetes Secret mounted to that container. MCP server needs only its own K8s credentials. |
| Removing the existing stdio tests to simplify the test suite | Seems like cleanup once HTTP transport is added. | The mock harness and 103 stdio-mode tests are the only regression safety net. Removing them leaves the tool logic unprotected. HTTP integration tests are a complement, not a replacement. | Keep all existing stdio tests. Add HTTP transport integration tests as an additional suite. |

---

## Feature Dependencies

```
[Streamable HTTP transport in server.py]
    └──requires──> FastMCP http_app() or run(transport="streamable-http") API
    └──requires──> port config in config.py
    └──enables──> [HTTP sidecar registration]

[Dockerfile HTTP entrypoint]
    └──requires──> Streamable HTTP transport mode in server.py
    └──enables──> [NemoClaw sidecar pod spec]

[openclaw.json HTTP registration config]
    └──requires──> known MCP endpoint URL (localhost:8000/mcp)
    └──enables──> [NemoClaw agent tool registration]

[NemoClaw deployed to EKS]
    └──requires──> EKS cluster with GPU node (NIM inference)
    └──requires──> NVIDIA NemoClaw installer or Helm chart
    └──requires──> OpenClaw operator or NemoClaw pod spec

[MCP sidecar in NemoClaw pod]
    └──requires──> Dockerfile HTTP entrypoint
    └──requires──> MCP container image published to Docker Hub
    └──requires──> pod spec: spec.sidecars or NemoClaw equivalent
    └──requires──> env vars: BLUEPRINTS_DIR, KUBERNETES_AUTH_MODE=incluster
    └──requires──> ServiceAccount with K8s RBAC

[openclaw.json HTTP registration config]
    └──must be loaded by NemoClaw at startup
    └──references──> http://localhost:8000/mcp URL

[Live agent chat experience]
    └──requires──> NemoClaw deployed to EKS
    └──requires──> MCP sidecar running in pod
    └──requires──> openclaw.json HTTP config loaded
    └──requires──> live EKS cluster (real K8s resources)
    └──requires──> live WEKA storage (real WEKA API)
    └──enables──> happy-path E2E validation

[SKILL.md + openclaw.json]
    └──already ship -- update transport references only
    └──no tool code changes required
```

### Dependency Notes

- **HTTP transport is a prerequisite for everything:** NemoClaw sidecar pattern requires HTTP. Without it the agent cannot call the tools in a pod-based deployment. This is Phase 1 of the milestone.
- **NemoClaw EKS deployment has external dependency on NVIDIA early preview:** NemoClaw was announced March 16, 2026 as early preview. APIs and deployment mechanisms may change without notice. Build deployment manifests defensively and expect at least one revision.
- **Sidecar pod spec format is partially unclear:** OpenClaw operator uses `spec.sidecars` with standard Kubernetes `Container` spec. NemoClaw may have a different pod spec model. This needs hands-on verification during deployment.
- **In-cluster RBAC is independent:** ServiceAccount and RBAC can be built and tested before NemoClaw is running. MCP server just needs `KUBERNETES_AUTH_MODE=incluster` env var.
- **Existing 103 tests have zero dependencies on new transport:** All existing tests use FastMCP in-process transport. They continue to pass unchanged. HTTP transport adds new tests, not changes to old ones.

---

## MVP Definition

### Launch With (v3.0 — this milestone)

Minimum needed to validate the live agent experience:

- [ ] `server.py` HTTP transport mode — `MCP_TRANSPORT=http` env var selects `mcp.run(transport="streamable-http", port=8000)` at startup
- [ ] `Dockerfile` updated — entrypoint defaults to HTTP mode; EXPOSE 8000
- [ ] `config.py` updated — `MCP_PORT`, `MCP_TRANSPORT` env vars
- [ ] `openclaw.json` HTTP variant — `"transport": "streamable-http"`, `"url": "http://localhost:8000/mcp"`
- [ ] Kubernetes deployment manifests — pod spec with NemoClaw + MCP sidecar containers, ServiceAccount, RBAC Role/RoleBinding, ConfigMap for env vars
- [ ] NemoClaw running on EKS — installer script executed, OpenClaw agent accessible
- [ ] MCP server registered with NemoClaw via openclaw.json HTTP config
- [ ] Happy-path E2E validation — agent completes: inspect_cluster → inspect_weka → list_blueprints → get_blueprint → get_crd_schema → validate_yaml → apply → status against real cluster

### Add After Validation (v3.x)

- [ ] Health endpoint at `/healthz` — needed before production readiness; low effort, high value
- [ ] `docker-compose` for local HTTP mode testing — useful for onboarding new developers
- [ ] RBAC tuning — start with broad permissions, narrow after observing what the server actually calls

### Future Consideration (v4+)

- [ ] TLS on MCP endpoint — only if MCP server is ever exposed outside the pod
- [ ] Multi-NemoClaw instance deployment — requires centralized tool registration, out of scope for v3
- [ ] Webhook-based status streaming — only if operator adds push notifications

---

## Feature Prioritization Matrix

| Feature | Value | Cost | Priority |
|---------|-------|------|----------|
| HTTP transport mode in server.py | HIGH | LOW | P1 |
| Dockerfile HTTP entrypoint + EXPOSE 8000 | HIGH | LOW | P1 |
| config.py MCP_PORT/MCP_TRANSPORT vars | HIGH | LOW | P1 |
| openclaw.json HTTP transport config | HIGH | LOW | P1 |
| Kubernetes deployment manifests | HIGH | MEDIUM | P1 |
| NemoClaw EKS install | HIGH | MEDIUM | P1 |
| ServiceAccount + RBAC | MEDIUM | LOW-MEDIUM | P1 |
| MCP sidecar registered with NemoClaw | HIGH | MEDIUM | P1 |
| Happy-path E2E validation | HIGH | HIGH | P1 |
| Health endpoint | MEDIUM | LOW | P2 |
| docker-compose local HTTP testing | MEDIUM | LOW | P2 |
| RBAC narrowing | LOW | LOW | P3 |
| TLS on MCP endpoint | LOW | HIGH | P3 |

**Priority key:**
- P1: Required for milestone acceptance
- P2: Should add once core works
- P3: Defer to future milestone

---

## How Streamable HTTP Transport Works (Verified)

Source: [MCP Transports Spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)

The server exposes a single endpoint (e.g., `http://localhost:8000/mcp`) supporting both HTTP POST and HTTP GET.

- **POST**: Client sends JSON-RPC requests. Server responds with either `application/json` (single response) or `text/event-stream` (SSE stream for multi-message responses). Client must send `Accept: application/json, text/event-stream`.
- **GET**: Client opens an SSE stream for server-initiated messages. Server either returns `text/event-stream` or `405 Method Not Allowed`.
- **Session IDs**: Server assigns `Mcp-Session-Id` on init. Client includes it on all subsequent requests. Clients send `DELETE` to terminate sessions.
- **Resumability**: Server may attach event IDs to SSE events. Client uses `Last-Event-ID` header to resume after disconnect.

**FastMCP implementation (HIGH confidence):**

```python
# Option 1: direct run (simplest)
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp")

# Option 2: ASGI app for Uvicorn (production)
app = mcp.http_app(path="/mcp")
# then: uvicorn server:app --host 0.0.0.0 --port 8000
```

Both options produce an endpoint at `http://HOST:8000/mcp`. FastMCP handles session management, SSE, and Origin validation automatically through its Starlette-backed ASGI app.

**Transport selection in server.py (recommended pattern):**

```python
import os
transport = os.getenv("MCP_TRANSPORT", "streamable-http")
port = int(os.getenv("MCP_PORT", "8000"))

if __name__ == "__main__":
    from config import validate_required
    validate_required()
    if transport == "streamable-http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
    else:
        mcp.run()  # stdio (default, for local dev and mock harness)
```

---

## How openclaw.json HTTP Registration Works (MEDIUM confidence)

Based on confirmed community patterns and OpenClaw mcpServers format, the HTTP sidecar registration config follows this structure:

```json
{
  "mcpServers": {
    "weka-app-store-mcp": {
      "transport": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Compared to the current `openclaw.json` stdio config:
- Remove `"startup"` block (no subprocess needed — server is already running as sidecar)
- Replace `"transport": "stdio"` with `"transport": "streamable-http"`
- Add `"url"` pointing to localhost MCP endpoint
- Keep `"name"`, `"description"`, `"tools"` array, `"skill"` reference

The `openclaw.json` file or equivalent `~/.openclaw/openclaw.json` / gateway config must be loaded by the OpenClaw/NemoClaw container at startup. For Kubernetes, this is typically injected via ConfigMap mounted at the expected config path. Config path and loading mechanism must be verified against NemoClaw's current docs (early preview — may differ from base OpenClaw).

---

## NemoClaw EKS Deployment Model (LOW-MEDIUM confidence)

NemoClaw (announced March 16, 2026, early preview) is an NVIDIA open-source reference stack that runs OpenClaw within an NVIDIA OpenShell sandboxed environment. Key architecture facts:

- **OpenShell runtime**: Kubernetes-compatible; uses Landlock + seccomp + netns isolation
- **Deployment method**: Installer script (`curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash`) or direct Helm chart
- **GPU requirement**: NVIDIA GPU node required on EKS for NIM/Nemotron inference; NVIDIA GPU Operator must be installed
- **MCP integration**: Inherits OpenClaw's MCP tool registration mechanism; tools registered via `openclaw.json` or gateway config
- **Sidecar support**: OpenClaw operator supports custom sidecars via `spec.sidecars` with standard `Container` spec; NemoClaw may use same or similar pattern

**EKS prerequisites (HIGH confidence from NVIDIA docs):**
- EKS cluster with GPU node group (instance type: p3.xlarge / p3.2xlarge or newer)
- NVIDIA GPU Operator installed via Helm
- NVIDIA device plugin enabled (`k8s.io/gpu` resource)
- Service account with RBAC for MCP server K8s access

**Sidecar pod spec pattern (MEDIUM confidence — inferred from OpenClaw operator docs):**

```yaml
# OpenClaw operator OpenClawInstance CR or NemoClaw pod spec
spec:
  sidecars:
    - name: weka-mcp-server
      image: wekachrisjen/weka-app-store-mcp:latest
      ports:
        - containerPort: 8000
      env:
        - name: MCP_TRANSPORT
          value: "streamable-http"
        - name: MCP_PORT
          value: "8000"
        - name: BLUEPRINTS_DIR
          value: "/blueprints"
        - name: KUBERNETES_AUTH_MODE
          value: "incluster"
      volumeMounts:
        - name: blueprints
          mountPath: /blueprints
```

The MCP server communicates with NemoClaw over `localhost:8000/mcp` — no network hop, no Service resource needed, no TLS.

**CAUTION:** NemoClaw is early preview software. Pod spec format, config file paths, and MCP registration mechanism details are not fully documented as of 2026-03-23. Plan for 1-2 iterations during deployment based on real behavior.

---

## Dependencies on Existing Code

| New Work | Existing Dependency | Change Needed |
|----------|---------------------|---------------|
| HTTP transport mode | `mcp-server/server.py` | Add transport branch in `__main__` block |
| Port/transport config | `mcp-server/config.py` | Add `MCP_PORT`, `MCP_TRANSPORT` env vars |
| Dockerfile HTTP mode | `mcp-server/Dockerfile` | Update entrypoint, add `EXPOSE 8000` |
| openclaw.json HTTP | `mcp-server/openclaw.json` | Replace startup block with url/transport fields |
| K8s manifests | None (new files) | Create in `k8s/` or `mcp-server/k8s/` directory |
| SKILL.md | `mcp-server/SKILL.md` | Remove any stdio-specific startup language |
| CI/CD | `.github/workflows/` | Verify image push still works; no transport-specific changes |

**Nothing changes in tools/:** All 8 tools are transport-agnostic. The tool registration code, response shapes, approval gate, and validation logic are unaffected by switching from stdio to HTTP.

---

## Sources

- [MCP Transports Specification 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — Streamable HTTP protocol details, session management, SSE behavior (HIGH confidence — official spec)
- [FastMCP HTTP Deployment Guide](https://gofastmcp.com/deployment/http) — `run(transport="streamable-http")`, `http_app()`, Uvicorn integration (HIGH confidence — official FastMCP docs)
- [OpenClaw Kubernetes Operator — Sidecar Containers](https://deepwiki.com/openclaw-rocks/k8s-operator/5.1-sidecar-containers) — `spec.sidecars` field, reserved container names, localhost communication pattern (MEDIUM confidence — third-party DeepWiki, consistent with GitHub repo)
- [OpenClaw MCP Configuration — community](https://openclawvps.io/blog/add-mcp-openclaw) — `mcpServers` block with `streamable-http` transport and `url` field (MEDIUM confidence — community source, consistent with other references)
- [MCP Server Transports — Roo Code Docs](https://docs.roocode.com/features/mcp/server-transports) — `streamable-http` config format: `{"type": "streamable-http", "url": "http://localhost:8080/mcp"}` (MEDIUM confidence — third-party, consistent with MCP spec)
- [NVIDIA NemoClaw Overview](https://docs.nvidia.com/nemoclaw/latest/about/overview.html) — architecture, OpenShell sandbox, early preview status (MEDIUM confidence — official NVIDIA docs, early preview caveats apply)
- [NVIDIA NemoClaw Quickstart](https://docs.nvidia.com/nemoclaw/latest/get-started/quickstart.html) — installer script, sandbox connect, TUI/CLI modes (MEDIUM confidence — official NVIDIA docs)
- [NVIDIA GPU Operator for EKS](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) — Helm installation, EKS compatibility (HIGH confidence — official NVIDIA docs)
- [MCP Python SDK Issue #1367 — Mounting Streamable HTTP on existing FastAPI app](https://github.com/modelcontextprotocol/python-sdk/issues/1367) — known complexity of mounting MCP endpoint on existing FastAPI app (MEDIUM confidence — official SDK issue tracker)
- [OpenClaw Kubernetes Operator — Deploy Guide](https://openclaw.rocks/blog/deploy-openclaw-kubernetes) — `OpenClawInstance` CRD, Helm chart install, sidecar pattern (MEDIUM confidence — official OpenClaw blog)

---

*Feature research for: Live EKS Deployment — Streamable HTTP MCP Transport + NemoClaw/OpenClaw Agent Chat*
*Researched: 2026-03-23*
