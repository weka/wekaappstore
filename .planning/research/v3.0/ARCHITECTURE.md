# Architecture Research: HTTP Transport and NemoClaw Sidecar (v3.0)

**Domain:** MCP tool server with Streamable HTTP transport for sidecar deployment alongside NemoClaw/OpenClaw on EKS
**Researched:** 2026-03-23
**Confidence:** MEDIUM — FastMCP HTTP API is HIGH confidence (official docs). OpenClaw sidecar HTTP MCP registration is MEDIUM confidence (third-party sources, no official MCP-in-sidecar example found). NemoClaw-specific config is LOW confidence (NemoClaw GitHub docs are local-deployment focused, no Kubernetes examples found).

---

## Context: What Changes in v3.0

v2.0 shipped a complete 8-tool MCP server using **stdio transport**. OpenClaw spawned it as a child process. This works for desktop OpenClaw but not for Kubernetes deployment, where OpenClaw and the MCP server run as separate containers in the same pod.

v3.0 adds **Streamable HTTP transport** to the same MCP server, enables deployment as a **sidecar container** in the NemoClaw/OpenClaw pod, and registers the tools via a URL endpoint instead of a process command.

**Nothing in the existing tool implementations changes.** The tools, response schemas, approval gating, and business logic imports are identical. Only the transport layer and deployment topology change.

---

## Standard Architecture

### System Overview: v3.0 (EKS Sidecar)

```
EKS Cluster
┌──────────────────────────────────────────────────────────────────────┐
│  Namespace: nemoclaw (or openclaw)                                   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │  Pod: nemoclaw-agent                                     │        │
│  │                                                          │        │
│  │  ┌────────────────────────┐  ┌───────────────────────┐  │        │
│  │  │  NemoClaw / OpenClaw   │  │  weka-app-store-mcp   │  │        │
│  │  │  container             │  │  sidecar container    │  │        │
│  │  │                        │  │                       │  │        │
│  │  │  Agent runtime         │  │  FastMCP              │  │        │
│  │  │  Conversation state    │  │  transport=http       │  │        │
│  │  │  OpenClaw gateway      │  │  host=0.0.0.0         │  │        │
│  │  │                        │  │  port=8080            │  │        │
│  │  │  mcpServers:           │  │  path=/mcp            │  │        │
│  │  │    url: localhost:8080 │◄─┤                       │  │        │
│  │  │                        │  │  8 tools (unchanged)  │  │        │
│  │  └────────────────────────┘  └──────────┬────────────┘  │        │
│  │                                         │               │        │
│  └─────────────────────────────────────────┼───────────────┘        │
│                                            │ pod-local              │
│                                            │ localhost:8080         │
└────────────────────────────────────────────┼─────────────────────── ┘
                                             │
              ┌──────────────────────────────┼─────────────────────┐
              │  Kubernetes API (in-cluster) │  WEKA API           │
              │  ServiceAccount token        │  WEKA_ENDPOINT env  │
              │  read+write CRs              │  read-only          │
              └──────────────────────────────┴─────────────────────┘
```

### Comparison: v2.0 (stdio) vs v3.0 (HTTP sidecar)

| Concern | v2.0 (stdio) | v3.0 (HTTP sidecar) |
|---------|-------------|---------------------|
| Transport | stdio — child process | Streamable HTTP — network |
| Deployment | OpenClaw spawns process | Pod sidecar container |
| Registration | `command` + `args` in openclaw.json | `url` in openclaw.json (or openclaw.yaml ConfigMap) |
| Tools | 8 tools — unchanged | 8 tools — unchanged |
| Port | None | 8080 (pod-local, not exposed externally) |
| Kubernetes auth | kubeconfig or in-cluster | in-cluster ServiceAccount |
| Health check | N/A | `/health` custom route on MCP server |
| Blueprint data | Filesystem path env var | Filesystem path env var (same) |
| Config change | `mcp.run()` without args | `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)` |

---

## New Components (v3.0 Additions)

### Component Responsibilities

| Component | Status | Responsibility |
|-----------|--------|---------------|
| `server.py` transport arg | **MODIFY** | Add `transport="streamable-http"` with host/port; keep stdio as default for local dev |
| Health check route | **NEW** | `@mcp.custom_route("/health")` for Kubernetes liveness/readiness probes |
| Kubernetes Deployment manifest | **NEW** | Pod spec with NemoClaw container + MCP sidecar container |
| ConfigMap (openclaw config) | **NEW** | OpenClaw's openclaw.json with `mcpServers.url` pointing to `localhost:8080/mcp` |
| ServiceAccount + RBAC | **NEW** | In-cluster identity for MCP server to read/write CRs and inspect nodes |
| Blueprint ConfigMap or PVC | **NEW** | How blueprint YAML files are mounted into the sidecar container |

### Components That Do NOT Change

| Component | Why Unchanged |
|-----------|--------------|
| All 8 tool modules (`tools/*.py`) | Transport is invisible to tool implementations |
| Business logic imports (`webapp/inspection/`, `webapp/planning/`) | Same Python imports regardless of transport |
| `openclaw.json` tool descriptions | Tool contract is transport-independent |
| `SKILL.md` agent workflow | Workflow is transport-independent |
| Dockerfile | Same image runs for both stdio and HTTP; entrypoint changes via env or CMD override |
| Test suite (103 tests) | Tests call tools directly, not through transport |

---

## Architectural Patterns

### Pattern 1: Dual-Mode Transport (stdio + HTTP from same server.py)

**What:** `server.py` reads a `MCP_TRANSPORT` environment variable. If set to `http`, it calls `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)`. If unset or `stdio`, it calls `mcp.run()` (default stdio behavior). No other code changes.

**When to use:** Always — enables the same container image to run in both local dev (stdio) and EKS sidecar (HTTP) modes.

**Trade-offs:**
- Pro: Single image, no branching Dockerfiles.
- Pro: Local development workflow unchanged — existing `mcp dev` and mock harness still work.
- Pro: CI can run the full test suite without any transport involvement.
- Con: The server binding (`0.0.0.0`) in HTTP mode is permissive. Acceptable inside a pod because the pod's NetworkPolicy restricts external access. If exposed via a Service, add authentication.

**Example:**
```python
# server.py — transport selection at startup
import os

transport = os.environ.get("MCP_TRANSPORT", "stdio")

if __name__ == "__main__":
    from config import validate_required
    validate_required()
    if transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
    else:
        mcp.run()
```

**Confidence:** HIGH — FastMCP `mcp.run(transport="streamable-http", host=..., port=...)` is documented at [gofastmcp.com/deployment/running-server](https://gofastmcp.com/deployment/running-server).

---

### Pattern 2: Health Check Route for Kubernetes Probes

**What:** Add a `GET /health` route to the MCP server using FastMCP's `custom_route` decorator. Returns `{"status": "ok"}`. Kubernetes liveness and readiness probes call this endpoint.

**When to use:** Required for sidecar deployment. Without a health check, Kubernetes cannot determine if the sidecar is ready to receive tool calls.

**Trade-offs:**
- Pro: Standard Kubernetes pattern. Prevents tool calls from reaching a partially-initialized server.
- Pro: FastMCP custom routes run on the same port and process as the MCP endpoint.
- Con: None significant.

**Example:**
```python
# server.py
@mcp.custom_route("/health", methods=["GET"])
async def health():
    return {"status": "ok"}
```

**Kubernetes probe config:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

**Confidence:** HIGH — FastMCP custom routes documented at gofastmcp.com; Kubernetes probe pattern is standard.

---

### Pattern 3: Sidecar Container with Shared Pod Network

**What:** The MCP server runs as a second container (`weka-app-store-mcp`) in the same Kubernetes pod as NemoClaw/OpenClaw. Both containers share the pod's network namespace. NemoClaw connects to the MCP server via `http://localhost:8080/mcp` — no Service, no DNS lookup, no external network hop.

**When to use:** This is the target v3.0 deployment topology.

**Trade-offs:**
- Pro: Pod-local communication. MCP endpoint is not reachable outside the pod without an explicit Service.
- Pro: NemoClaw discovers the MCP server via environment variable injection or static config — no service discovery needed.
- Pro: Same lifecycle — pod restart brings both containers down and back up together.
- Con: Sidecar container startup race: if NemoClaw starts before the MCP server is ready, initial tool discovery fails. Mitigated by the `/health` readiness probe — NemoClaw's startup should wait for the sidecar's readiness gate.
- Con: Container resource limits must account for both NemoClaw and the MCP server. The MCP server is lightweight (Python process, <256Mi RAM typical).

**OpenClaw mcpServers registration (HTTP transport):**
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

**Confidence:** MEDIUM — The `mcpServers.url` with `transport: streamable-http` pattern is confirmed by multiple third-party sources ([openclawvps.io](https://openclawvps.io/blog/add-mcp-openclaw), community examples). The exact OpenClaw/NemoClaw Kubernetes ConfigMap key names need validation against live deployment because official Kubernetes docs for OpenClaw focus on Chromium/Ollama sidecars, not custom MCP server sidecars.

---

### Pattern 4: Kubernetes Sidecar Container Spec

**What:** The NemoClaw/OpenClaw Kubernetes Deployment or StatefulSet spec adds the MCP server as a second container entry. The MCP server container gets the same pod-level ServiceAccount but may need additional RBAC (node read, namespace list, CR read/write).

**Example Deployment excerpt:**
```yaml
spec:
  containers:
  - name: nemoclaw
    image: nvcr.io/nvidia/nemoclaw:latest  # placeholder — validate actual image name
    # ... NemoClaw config ...

  - name: weka-app-store-mcp
    image: wekachrisjen/weka-app-store-mcp:latest
    env:
    - name: MCP_TRANSPORT
      value: "http"
    - name: KUBERNETES_AUTH_MODE
      value: "in-cluster"
    - name: BLUEPRINTS_DIR
      value: "/blueprints"
    - name: LOG_LEVEL
      value: "INFO"
    ports:
    - containerPort: 8080
      name: mcp-http
    volumeMounts:
    - name: blueprints
      mountPath: /blueprints
      readOnly: true
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 30
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "512Mi"
        cpu: "500m"

  volumes:
  - name: blueprints
    configMap:
      name: weka-blueprints   # or use a PVC / git-sync sidecar
```

**Confidence:** MEDIUM (standard Kubernetes sidecar pattern). NemoClaw container image name and exact deployment structure require validation once NemoClaw EKS docs are available.

---

### Pattern 5: Blueprint Data in the Sidecar Container

**What:** Blueprint YAML files must be accessible to the MCP server sidecar at the path configured in `BLUEPRINTS_DIR`. Three mounting options exist:

| Option | Mechanism | Trade-offs |
|--------|-----------|------------|
| ConfigMap | `kubectl create configmap weka-blueprints --from-file=blueprints/` | Simple, limited to <1MB per key, no live update without pod restart |
| PersistentVolumeClaim | Mount existing PVC containing blueprints | Supports larger catalogs; requires PVC lifecycle management |
| Init container (git-sync) | Init container clones repo to emptyDir; sidecar mounts emptyDir | Latest blueprints on pod start; no PVC; slightly complex |

**Recommended:** ConfigMap for initial deployment. If the blueprint catalog exceeds ConfigMap size limits or needs live refresh without pod restart, move to git-sync init container.

**Confidence:** HIGH — standard Kubernetes volume mounting patterns.

---

## Data Flow

### Tool Call Flow (HTTP transport)

```
User chat → NemoClaw reasoning → tool call decision
    ↓
OpenClaw reads mcpServers config: url=http://localhost:8080/mcp
    ↓
HTTP POST to localhost:8080/mcp  (JSON-RPC: tools/call)
    ↓
FastMCP receives request on Streamable HTTP handler
    ↓
Routes to tool module (e.g., tools/inspect_cluster.py)
    ↓
Tool calls webapp/inspection/cluster.py (direct Python import)
    ↓
cluster.py reads Kubernetes API via in-cluster ServiceAccount
    ↓
Result flows back: Python dict → FastMCP serializes → HTTP response
    ↓
NemoClaw receives JSON tool result
    ↓
Agent reasons, continues workflow
```

### Startup Sequence (Pod initialization)

```
1. Pod scheduled on EKS node
2. weka-app-store-mcp container starts
3. FastMCP binds port 8080, /health returns 200
4. Readiness probe passes → container marked Ready
5. NemoClaw container starts (or checks sidecar readiness)
6. NemoClaw reads openclaw.json / ConfigMap
7. OpenClaw connects to http://localhost:8080/mcp
8. tools/list call → discovers 8 tools
9. Agent ready for user chat
```

**Risk at step 5:** OpenClaw/NemoClaw startup behavior when sidecar is not yet ready is not confirmed. If NemoClaw starts before step 4, tool discovery may fail silently. Mitigation: add `initContainers` or a startup script that waits for localhost:8080/health before launching the main NemoClaw process.

---

## Integration Points

### New Integration Points (v3.0)

| Boundary | Communication | Notes |
|----------|---------------|-------|
| NemoClaw → MCP sidecar | HTTP POST to `localhost:8080/mcp` | Pod-local; no Service or DNS needed |
| Kubernetes liveness probe → MCP sidecar | HTTP GET `localhost:8080/health` | Requires custom_route in server.py |
| MCP sidecar → Kubernetes API | `kubernetes` Python client, in-cluster auth | Existing; no change except auth mode must be `in-cluster` |
| MCP sidecar → blueprint files | Volume mount at `BLUEPRINTS_DIR` | ConfigMap or PVC; path injected via env var |
| OpenClaw ConfigMap → NemoClaw config | Kubernetes ConfigMap or `openclaw.yaml` | Registration moves from `command` to `url` field |

### Unchanged Integration Points

| Boundary | Communication | Notes |
|----------|---------------|-------|
| MCP tool modules → webapp/ business logic | Direct Python import | PYTHONPATH unchanged; same Dockerfile structure |
| MCP sidecar → WEKA API | HTTP via `inspection/weka.py` | Same; `WEKA_ENDPOINT` env var injected at pod level |
| apply tool → Kubernetes API (write) | `apply_gateway.py` via kubernetes client | Same; in-cluster auth handles credentials |

---

## New vs. Modified vs. Unchanged (v3.0 scope)

### New (must be built for v3.0)

| Component | Location | Description |
|-----------|----------|-------------|
| Transport env var check | `mcp-server/server.py` | Read `MCP_TRANSPORT`; branch on `http` vs `stdio` |
| Health check route | `mcp-server/server.py` | `@mcp.custom_route("/health")` for Kubernetes probes |
| Kubernetes Deployment manifest | `k8s/deployment.yaml` (new dir) | Pod spec with NemoClaw + MCP sidecar |
| ServiceAccount + RBAC manifests | `k8s/rbac.yaml` | In-cluster identity with needed permissions |
| ConfigMap: openclaw config | `k8s/openclaw-config.yaml` | Contains openclaw.json with `mcpServers.url` |
| ConfigMap: blueprints | `k8s/blueprints-configmap.yaml` | Blueprint YAML files mounted as volume |
| `MCP_TRANSPORT` env var in Dockerfile CMD | `mcp-server/Dockerfile` | Optional; can be set at deploy time instead |

### Modified (existing, small change)

| Component | Change Required |
|-----------|----------------|
| `mcp-server/server.py` | Add transport branch and health route; ~10 lines |
| `mcp-server/openclaw.json` | Update `transport` from `stdio` to `streamable-http`; add `url` field; remove `startup.command` |
| `mcp-server/requirements.txt` | Verify `mcp[cli]>=1.26.0` includes Streamable HTTP (it does — confirmed) |

### Unchanged

| Component | Reason |
|-----------|--------|
| All `mcp-server/tools/*.py` | Transport-agnostic |
| All `app-store-gui/webapp/` imports | Not affected by transport layer |
| `mcp-server/Dockerfile` | Same image; transport selected at runtime via env |
| `mcp-server/SKILL.md` | Agent workflow independent of transport |
| 103 existing tests | Test tool logic, not transport |
| GitHub Actions CI/CD | Same build process; image tag strategy unchanged |

---

## Build Order for v3.0 Phases

Dependencies flow left to right. Build in this order to enable testing at each gate.

```
Phase 1: server.py transport change + health check
    → Gate: mcp.run(transport="streamable-http") starts, /health returns 200
    → Enables: local HTTP testing with curl / MCP inspector

Phase 2: openclaw.json updated for HTTP transport
    → Gate: openclaw.json has url field, transport=streamable-http
    → Enables: testing OpenClaw registration with HTTP (if OpenClaw available locally)

Phase 3: Kubernetes RBAC + ServiceAccount
    → Gate: ServiceAccount exists in EKS cluster with correct permissions
    → Enables: in-cluster MCP server to read nodes, namespaces, CRs

Phase 4: Blueprint data strategy (ConfigMap or PVC)
    → Gate: BLUEPRINTS_DIR resolves to files inside pod
    → Enables: list_blueprints and get_blueprint work in-cluster

Phase 5: Kubernetes Deployment manifest (NemoClaw + sidecar)
    → Gate: pod starts, both containers Running, /health ready
    → Enables: NemoClaw connects to MCP server via localhost:8080/mcp

Phase 6: OpenClaw config wired in Kubernetes ConfigMap
    → Gate: OpenClaw reads mcpServers.url, tools/list returns 8 tools
    → Enables: full agent conversation

Phase 7: E2E happy-path validation
    → Gate: inspect → validate → apply completes against real EKS cluster
```

**Critical dependency:** Phases 3-7 require EKS cluster access. Phase 1-2 can be done and validated locally before any EKS work.

---

## Anti-Patterns

### Anti-Pattern 1: Exposing the MCP HTTP Port via a Kubernetes Service

**What people do:** Add a `Service` pointing to the sidecar's port 8080, making the MCP endpoint reachable from outside the pod.

**Why it's wrong:** The MCP server has no authentication middleware. Any pod in the cluster (or any network path to the Service) could call the `apply` tool and create WekaAppStore CRs. The approval gate is OpenClaw-side; it does not protect against direct HTTP callers.

**Do this instead:** Keep port 8080 pod-local only. No Kubernetes Service for the MCP port. If external access is ever needed, add authentication middleware to the FastMCP server first (Bearer token via `@mcp.custom_route` guard or reverse proxy with mTLS).

### Anti-Pattern 2: Different Images for stdio vs HTTP

**What people do:** Create a separate Dockerfile or image tag for the HTTP transport version to avoid "contaminating" the stdio image.

**Why it's wrong:** Creates image drift. The stdio and HTTP images will diverge over time. Tests that pass on the stdio image may not catch issues in the HTTP image.

**Do this instead:** One image. `MCP_TRANSPORT=http` environment variable selects the transport at runtime. CI builds one image and tests both modes.

### Anti-Pattern 3: Hard-Coding localhost:8080 in the MCP Server

**What people do:** Set `host="127.0.0.1"` in `mcp.run()` assuming the server only needs to talk to localhost.

**Why it's wrong:** `127.0.0.1` (loopback) is unreachable from other containers in the same pod if the container's network interface is not `lo`. NemoClaw connects from a different container process. Use `host="0.0.0.0"` so the server binds to all pod interfaces.

**Do this instead:** `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)`. The pod's NetworkPolicy handles external access restriction — not the bind address.

### Anti-Pattern 4: Missing Readiness Probe Causes Race Condition

**What people do:** Deploy both containers with no readiness probe on the MCP sidecar. NemoClaw starts, calls tools/list, gets a connection refused because the MCP server is still initializing.

**Why it's wrong:** OpenClaw's behavior when tool discovery fails at startup is unclear. It may proceed without tools, requiring a pod restart to recover. Silent failure is hard to debug.

**Do this instead:** Add `readinessProbe` on the MCP sidecar pointing to `/health`. Either delay NemoClaw startup (via init container or startup script checking `/health`) or confirm that NemoClaw retries tool discovery after the sidecar becomes ready.

### Anti-Pattern 5: Stdout Logging in HTTP Mode

**What people do:** Add `print()` debug statements after switching to HTTP mode, assuming stdout is safe because it is no longer the MCP channel.

**Why it's actually fine — but stay disciplined:** In HTTP mode, stdout is not the MCP channel (unlike stdio mode), so print() does not corrupt the protocol. However, if the server is ever switched back to stdio, this breaks immediately. Use `logging` to stderr in all cases. The existing `server.py` already enforces this via `logging.basicConfig(stream=sys.stderr, ...)`.

---

## Scaling Considerations

This project is a single-cluster, single-operator deployment. Scaling is not a primary concern for v3.0.

| Scale | Architecture Notes |
|-------|--------------------|
| Single operator, EKS cluster | HTTP sidecar at port 8080 pod-local. One pod. |
| Multiple NemoClaw pods (team use) | Each pod gets its own MCP sidecar — stateless, scales horizontally. Blueprint ConfigMap is read-only and shared. |
| Multiple clusters | Separate pod per cluster. MCP server uses in-cluster auth scoped to that cluster. |

---

## Open Questions (Require Validation in Live EKS)

1. **NemoClaw ConfigMap key names:** The exact field path in the OpenClaw ConfigMap for `mcpServers.url` needs to be verified against the running NemoClaw version. Community sources use `mcpServers.<name>.url` with `transport: streamable-http` but official NemoClaw Kubernetes docs do not confirm this schema.

2. **NemoClaw startup behavior:** Does NemoClaw wait for sidecars with readiness probes, or does it start immediately? If immediate, an init container or startup script may be needed to avoid a tool discovery race.

3. **NemoClaw image name:** `nvcr.io/nvidia/nemoclaw:latest` is a placeholder. The actual NVIDIA NGC image name and tag need to be confirmed.

4. **RBAC minimum permissions:** The ServiceAccount needs `nodes/list`, `namespaces/list`, `storageclasses/list`, and `wekastores.warp.io` CRD read/write. Full permission list should be derived from what each tool calls.

5. **Blueprint data size:** If the blueprint catalog exceeds ConfigMap's 1MB limit, a PVC or git-sync init container is needed instead.

---

## Sources

- [FastMCP — Running Server (Official)](https://gofastmcp.com/deployment/running-server) — HIGH confidence
- [MCPcat — Streamable HTTP Server Guide](https://mcpcat.io/guides/building-streamablehttp-mcp-server/) — MEDIUM confidence
- [OpenClaw Kubernetes Operator (Official)](https://github.com/openclaw-rocks/k8s-operator) — MEDIUM confidence
- [OpenClaw k8s-operator Sidecar Containers (DeepWiki)](https://deepwiki.com/openclaw-rocks/k8s-operator/5.1-sidecar-containers) — MEDIUM confidence
- [OpenClaw mcpServers HTTP config (openclawvps.io)](https://openclawvps.io/blog/add-mcp-openclaw) — MEDIUM confidence (third-party, consistent with MCP spec)
- [OpenClaw deploy on Kubernetes (openclaw.rocks)](https://openclaw.rocks/blog/deploy-openclaw-kubernetes) — MEDIUM confidence
- [NemoClaw GitHub (NVIDIA)](https://github.com/NVIDIA/NemoClaw) — LOW confidence (docs cover local deployment only; Kubernetes config not documented)
- [Roo Code MCP Server Transports](https://docs.roocode.com/features/mcp/server-transports) — MEDIUM confidence (confirms streamable-http URL registration pattern)

---

*Architecture research for: WEKA App Store MCP tool server — v3.0 HTTP transport and EKS sidecar deployment*
*Researched: 2026-03-23*
*Replaces and extends: ARCHITECTURE.md (v2.0 stdio architecture, researched 2026-03-20)*
