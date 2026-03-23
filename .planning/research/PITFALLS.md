# Pitfalls Research

**Domain:** Adding Streamable HTTP MCP transport and NemoClaw/OpenClaw EKS sidecar deployment to an existing MCP server
**Researched:** 2026-03-23
**Confidence:** HIGH for MCP transport and Kubernetes patterns (official spec + SDK issues verified); MEDIUM for NemoClaw-specific behavior (early preview software, March 2026 release, interfaces may change)

---

> **Note on scope:** This document covers pitfalls specific to v3.0 additions: Streamable HTTP transport, EKS sidecar deployment, NemoClaw sandbox behavior, and OpenClaw tool registration via HTTP. Pitfalls for the existing MCP server tool contract (response shape, approval gate, YAML validation, deprecated code) are documented in the v2.0 pitfalls file and remain valid.

---

## Critical Pitfalls

### Pitfall 1: Mounting Streamable HTTP on Existing FastAPI App Breaks the Session Manager

**What goes wrong:**
The current `server.py` runs as a standalone stdio process. When adding Streamable HTTP transport, the natural instinct is to mount `mcp.streamable_http_app()` onto the existing FastAPI app using `app.mount("/mcp", mcp_app)`. This fails with `RuntimeError: Task group is not initialized. Make sure to use run()`. The MCP app's Starlette lifespan (which initializes the session manager) is not invoked when the app is mounted as a sub-application inside FastAPI — only the outer FastAPI lifespan runs.

A secondary failure mode: mounting at `/mcp` produces 307 redirects and 404s because the MCP endpoint path is hardcoded inside the returned Starlette app, making FastAPI's mount prefix stack incorrectly.

**Why it happens:**
The FastMCP python SDK returns a pre-configured Starlette ASGI application. Starlette/FastAPI nested lifespan composition is not automatic — each mounted sub-app's lifespan must be explicitly wrapped and invoked by the parent app's lifespan context manager. Most developers do not know this because normal FastAPI sub-apps (APIRouter, StaticFiles) do not use lifespan.

**How to avoid:**
Run the MCP server as a standalone Uvicorn/ASGI process separate from the existing FastAPI backend. Both processes run in the same pod but on different ports (e.g., FastAPI on 8080, MCP on 8001). They communicate only via shared volume mounts and environment variables — not through FastAPI app mounting. This approach avoids the lifespan problem entirely and keeps the transport boundary clean.

If colocation in a single process is required, use a combined lifespan: create an `asynccontextmanager` that calls both the FastAPI app lifespan and `mcp.streamable_http_app()` lifespan in sequence, pass it as the `lifespan=` argument to the outer FastAPI app, then add MCP routes manually to the outer app rather than using `app.mount()`.

**Warning signs:**
- `RuntimeError: Task group is not initialized` on the first HTTP MCP request after startup.
- 307 redirects to `/mcp/mcp` (double prefix) when accessing the MCP endpoint.
- The MCP server works fine in stdio mode but fails immediately when `transport="streamable-http"` is added.

**Phase to address:**
Streamable HTTP transport implementation phase — design the process boundary (separate process vs. single process) before writing any transport code.

---

### Pitfall 2: NemoClaw's Deny-All Egress Policy Blocks the Sidecar MCP Server

**What goes wrong:**
NemoClaw's default sandbox policy (`openclaw-sandbox.yaml`) blocks all outbound network egress except a hardcoded allowlist: Anthropic APIs, NVIDIA inference endpoints, GitHub, npm registry, Telegram, and OpenClaw services. When OpenClaw (inside the NemoClaw sandbox) tries to register and call the sidecar MCP server at `http://localhost:8001/mcp`, the OpenShell runtime blocks the connection and logs the attempt. The agent either fails to initialize tools at all or receives connection errors on every tool call.

This is not a configuration bug — it is the policy working correctly. The mistake is deploying without explicitly whitelisting `localhost:8001` (or the equivalent loopback address) in the egress policy before the agent starts.

**Why it happens:**
NemoClaw's documentation focuses on remote endpoint allowlisting. Developers assume loopback traffic is inherently permitted (it is in standard Linux networking), but OpenShell's network policy enforcement operates at the application layer and applies to all outbound connections including localhost. The sidecar MCP server is an "unknown" endpoint until explicitly added.

**How to avoid:**
Before deploying, add the sidecar MCP server's loopback address to the NemoClaw network policy. Edit `nemoclaw-blueprint/policies/openclaw-sandbox.yaml` to include `127.0.0.1:8001` (or the configured port) in the egress allowlist. Run `nemoclaw onboard` after the policy change. Verify the policy is applied before starting the agent by inspecting the TUI's policy view.

If NemoClaw is deployed to EKS using the `agent-sandbox` CRD pattern (rather than its default k3s-in-Docker mode), verify that the CRD's network policy fields support the same loopback allowlisting — this is an open research item since NemoClaw's EKS support is community-driven and not officially documented.

**Warning signs:**
- OpenShell TUI shows a blocked connection attempt to `127.0.0.1` when the agent starts up.
- Tool calls time out immediately rather than returning errors (connection refused vs. timeout distinguish different failure modes).
- NemoClaw TUI prompts the operator to approve or deny a connection to `localhost` — this is the policy working as designed.

**Phase to address:**
NemoClaw deployment configuration phase — policy file must be prepared and verified before the agent pod is started.

---

### Pitfall 3: NemoClaw Requires k3s-in-Docker, Not Native EKS Pod Scheduling

**What goes wrong:**
NemoClaw's primary installation model is Docker + k3s embedded in Docker, running on an Ubuntu VM. It installs OpenShell, creates an isolated k3s cluster inside Docker, and manages the agent within that nested cluster. This model does not map onto a standard EKS pod. Running `nemoclaw install` on an EKS node does not produce a working deployment — it attempts to create a Docker-based k3s cluster inside a container, which requires privileged mode, nested container runtimes, and Docker-in-Docker that EKS worker nodes typically do not permit.

The community workaround is to use the `agent-sandbox` CRD from `kubernetes-sigs/agent-sandbox` as the sandbox runtime instead of OpenShell's embedded k3s. This requires deploying the `agent-sandbox` controller to the EKS cluster and configuring NemoClaw to use it as the runtime backend. As of March 2026 this is community-driven (GitHub Issue #407 in NVIDIA/NemoClaw) and not officially supported.

**Why it happens:**
NemoClaw is presented as "run OpenClaw more securely in Kubernetes" but the Kubernetes in question is the embedded k3s cluster it creates inside Docker, not the user's existing cluster. The marketing implies standard Kubernetes compatibility that does not currently exist for managed services like EKS.

**How to avoid:**
Do not attempt to run `nemoclaw install` directly on EKS worker nodes. Two viable approaches:

1. **Separate VM approach:** Deploy NemoClaw on a dedicated EC2 instance alongside the EKS cluster. The VM runs NemoClaw's k3s-in-Docker stack. The sidecar MCP server runs in EKS as a standard pod. NemoClaw calls the MCP server over the cluster's internal DNS or a Service endpoint rather than localhost. This gives up the low-latency sidecar benefit but avoids the EKS compatibility gap.

2. **agent-sandbox CRD approach (experimental):** Deploy the `agent-sandbox` controller to EKS, configure NemoClaw to use it as the runtime backend. This is documented in GitHub Issue #407 but is not officially supported. Treat it as experimental for this milestone.

**Warning signs:**
- `nemoclaw install` requires Docker daemon access from within the pod (likely a privileged container request).
- Pod fails to start with `permission denied` on `/var/run/docker.sock` or similar.
- The node reports nested container creation failures in `kubelet` logs.

**Phase to address:**
Infrastructure and deployment planning phase — select the deployment topology before any Kubernetes manifests are written.

---

### Pitfall 4: OpenClaw HTTP MCP Registration Expects a Stable URL, Not a Pod IP

**What goes wrong:**
In `openclaw.json`, an MCP server registered via HTTP uses a URL field: `"url": "http://<address>/mcp"`. If that address is a Pod IP, it changes every time the pod restarts. After a pod restart, OpenClaw's registration still points to the old IP. The agent starts, fails to connect to the MCP server, and reports tool unavailability — or worse, connects to a different workload that happens to reuse the old IP.

**Why it happens:**
Local testing uses `http://localhost:8001/mcp` which is stable. When moving to EKS, developers copy the pod IP from `kubectl get pods -o wide` and hardcode it in the config. Pod IPs are ephemeral — they are not stable across restarts, node migrations, or rolling updates.

**How to avoid:**
Register the MCP server using a Kubernetes Service ClusterIP DNS name: `http://weka-mcp-server.<namespace>.svc.cluster.local:8001/mcp`. Services have stable DNS names and ClusterIPs regardless of which pod backs them. Create a headless Service pointing to the MCP server pod before configuring OpenClaw. If NemoClaw runs on a separate VM (approach 1 from Pitfall 3), use a LoadBalancer or NodePort Service to expose the MCP server and register the stable endpoint.

For the sidecar pattern specifically (NemoClaw and MCP server in the same pod), `localhost` is the correct address since both containers share the same network namespace. This is stable across pod IP changes.

**Warning signs:**
- `openclaw.json` contains a dotted-decimal IP address.
- After deploying a new pod version, agents report tools unavailable without any code change.
- `openclaw.json` is generated at deployment time from `kubectl get pod` output rather than from a Service DNS name.

**Phase to address:**
OpenClaw registration and deployment configuration phase — finalize the MCP server address scheme before writing `openclaw.json` or `generate_openclaw_config.py` output.

---

### Pitfall 5: Streamable HTTP Session ID Is Not Forwarded by All MCP Clients

**What goes wrong:**
When the MCP server issues an `Mcp-Session-Id` header in its `InitializeResult` response, the protocol requires clients to include that header in all subsequent requests. Several MCP clients (including some versions of OpenClaw's internal MCP client) use `fetch()` internally and do not persist or forward the `Mcp-Session-Id` header across requests. The server then treats each subsequent request as a new session, re-initializes state, and either returns 400 Bad Request or creates a new session on every tool call.

The FastMCP Python SDK has a documented issue (Issue #808) where the server does not recognize the `X-Session-ID` header even when the client sends it correctly. Both the server and client sides have active known issues as of early 2026.

**Why it happens:**
Streamable HTTP transport is relatively new (TypeScript SDK 1.10.0, April 2025; Python SDK support followed later). Session management was an afterthought in many client implementations that assumed stateless operation. When both server and client implementations are immature simultaneously, session header forwarding is the first thing to break.

**How to avoid:**
Design the MCP server to operate in stateless mode for this milestone. Do not rely on `Mcp-Session-Id` for any required functionality. The MCP spec allows servers to omit the session ID entirely, in which case clients do not need to forward anything. For a sidecar deployment with a single OpenClaw client, stateless mode is sufficient — there is no horizontal scaling concern and no need for session affinity.

If stateful sessions are needed in future, test session header forwarding explicitly with the version of OpenClaw/NemoClaw deployed before enabling it.

**Warning signs:**
- Each tool call triggers a new `initialize` handshake in the server logs (stateless symptom).
- Server logs show 400 responses to requests after the first `initialize`.
- OpenClaw reports tools available after `initialize` but unavailable during the planning session.

**Phase to address:**
Streamable HTTP transport implementation phase — make stateless vs. stateful decision explicit in the server implementation before writing session management code.

---

### Pitfall 6: MCP Server Container Starts After OpenClaw Tries to Register Tools

**What goes wrong:**
In the sidecar pod, OpenClaw starts and immediately attempts to register tools from `openclaw.json`. If the MCP server container has not yet completed startup (Python imports, Kubernetes client initialization, tool registration), the registration request arrives at a port that is not yet listening. OpenClaw logs a connection error and marks the tools as unavailable. Depending on the client, it may not retry registration after startup.

Kubernetes starts both containers concurrently unless startup ordering is explicitly configured. There is no built-in guarantee that the MCP server sidecar is ready before the main OpenClaw container starts tool registration.

**Why it happens:**
Kubernetes init containers run before the main containers, but sidecar-pattern containers (defined as init containers with `restartPolicy: Always` in Kubernetes 1.29+) start before the main container only if defined that way. Most sidecar configurations use regular containers in `spec.containers`, which start concurrently. Developers test locally where startup times are short and the race does not manifest.

**How to avoid:**
Define the MCP server as a native sidecar (init container with `restartPolicy: Always` in Kubernetes 1.29+ / EKS 1.29+). This guarantees the MCP server starts and passes its `startupProbe` before the main OpenClaw container starts. Add a startup probe to the MCP server container:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: 8001
  failureThreshold: 30
  periodSeconds: 2
```

Add a `/health` endpoint to the MCP server that returns 200 only after all tools are registered and the server is accepting connections.

**Warning signs:**
- OpenClaw logs show tool registration errors only on pod cold starts, not during operation.
- The problem disappears if the pod is given extra time before first use (masking the race).
- No `startupProbe` is defined on the MCP server container.

**Phase to address:**
Kubernetes manifests and deployment phase — container startup ordering must be specified in the pod spec, not left to chance.

---

### Pitfall 7: Kubernetes RBAC Gives the Sidecar Pod the Operator's Permissions

**What goes wrong:**
If the NemoClaw/MCP sidecar pod uses the same service account as the WEKA operator, it inherits full cluster-write permissions including the ability to create, update, and delete `WekaAppStore` CRDs, modify RBAC bindings, and access Secrets. The MCP server's apply tool needs only `create` on `wekaappstores` resources. The inspection tools need only `get`/`list` on nodes, namespaces, storage classes, and pods. Running with operator-level permissions violates least-privilege and creates a blast radius if the MCP server is compromised through prompt injection or a dependency vulnerability.

**Why it happens:**
The existing WEKA App Store already has an operator service account with broad permissions. It is expedient to reuse it rather than creating a new account. The default EKS pod identity (node instance role) may also grant broad IAM permissions that are not needed by the MCP server.

**How to avoid:**
Create a dedicated service account `weka-mcp-server-sa` with a ClusterRole that grants:
- `get`, `list` on `nodes`, `namespaces`, `storageclasses`, `pods`
- `get`, `list` on `persistentvolumes`, `persistentvolumeclaims`
- `create` on `wekaappstores` (in the operator's namespace)
- `get`, `list` on `wekaappstores` (for status tool)

Block access to `secrets`, `configmaps`, `serviceaccounts`, RBAC resources, and all non-inspection resources. Annotate the service account with IRSA to scope AWS permissions to nothing (or read-only CloudWatch if logging is required).

**Warning signs:**
- The pod spec references `serviceAccountName: weka-operator-sa`.
- No custom ClusterRole or Role exists for the MCP server.
- `kubectl auth can-i delete wekaappstores --as=system:serviceaccount:<ns>:weka-mcp-server-sa` returns "yes".

**Phase to address:**
Kubernetes manifests phase — service account and RBAC must be defined before deployment, not added as a follow-up hardening task.

---

### Pitfall 8: The MCP Server's Origin Validation Blocks Intra-Pod Requests

**What goes wrong:**
The MCP specification (section: Streamable HTTP, Security Warning) requires servers to validate the `Origin` header on all incoming connections to prevent DNS rebinding attacks. When OpenClaw calls the MCP server from within the same pod via `http://localhost:8001/mcp`, the request may have no `Origin` header, or the header may be set to `null` (a common browser security behavior for same-origin requests). If the FastMCP server's Origin validation rejects requests with missing or `null` Origin headers, every intra-pod tool call fails with 403 Forbidden.

**Why it happens:**
Origin validation is designed for browser-facing servers. MCP client implementations that do not originate from a browser (like OpenClaw's programmatic fetch client) may not send an Origin header at all. The server-side default validation behavior may be overly strict for a pod-internal transport scenario.

**How to avoid:**
Explicitly configure the FastMCP server's allowed origins to include the intra-pod case. For a sidecar deployment, the simplest safe configuration is: allow requests with no Origin header (loopback-only binding already prevents external access) while still binding the server to `127.0.0.1` rather than `0.0.0.0`. Do not disable Origin validation for publicly accessible endpoints. Verify the FastMCP server is bound to `127.0.0.1` in the pod network namespace so external network actors cannot reach it regardless of Origin policy.

**Warning signs:**
- `403 Forbidden` responses to tool calls in the server log with an Origin-related message.
- Tool calls succeed from a curl test that includes `-H "Origin: http://localhost"` but fail from the OpenClaw client.
- The server is bound to `0.0.0.0` in the Kubernetes pod (exposes to cluster network) instead of `127.0.0.1`.

**Phase to address:**
Streamable HTTP transport implementation phase — Origin policy and binding address must be set explicitly when configuring the HTTP transport.

---

### Pitfall 9: openclaw.json Generated at Build Time Does Not Survive Pod Restarts

**What goes wrong:**
`generate_openclaw_config.py` produces `openclaw.json` with the MCP server URL embedded. If this file is baked into the container image or generated once at build time, it becomes stale when:
- The MCP server port changes.
- The deployment namespace changes.
- The transport changes from stdio to HTTP (requiring `"url"` instead of `"command"`/`"args"` fields).

The agent starts with an outdated registration and cannot reach the tools. Because `openclaw.json` is inside the image, fixing it requires a new image build and redeploy.

**Why it happens:**
During v2.0 development, the generate script targets local stdio registration where the command is a fixed path (`python server.py`). HTTP registration requires a URL, which is environment-specific. The build-time generation pattern works for static local registrations but breaks for environment-specific deployments.

**How to avoid:**
Generate `openclaw.json` at pod startup using an init container or a startup script that injects the correct URL from environment variables. Store the generated file in an `emptyDir` volume shared between containers. The generation command becomes:

```bash
MCP_SERVER_URL=http://localhost:8001/mcp python generate_openclaw_config.py > /shared/openclaw.json
```

The `generate_openclaw_config.py` script must be updated to support HTTP transport output (producing `"url": "$MCP_SERVER_URL"`) in addition to the existing stdio output.

**Warning signs:**
- `openclaw.json` contains a hardcoded localhost URL baked into the container image.
- The file is not in a ConfigMap, mounted volume, or generated at startup.
- Changing the MCP server port requires a new image build.

**Phase to address:**
OpenClaw registration and deployment configuration phase — config generation must be runtime, not build-time, for EKS deployment.

---

### Pitfall 10: The SSE Transport Is Being Deprecated — Do Not Implement It

**What goes wrong:**
Developers researching MCP HTTP transport find documentation and examples for the older `HTTP+SSE` transport (protocol version 2024-11-05). Implementing SSE transport instead of Streamable HTTP results in a server that requires two endpoints (`/sse` for the SSE stream, `/messages` for POST), maintains persistent connections, limits horizontal scaling, and will lose client support as the deprecation completes (effective April 1, 2026, per the SSE Transport Deprecation announcement).

**Why it happens:**
The SSE transport has significantly more community examples, blog posts, and StackOverflow answers than the newer Streamable HTTP transport. Searching for "MCP HTTP Python" returns SSE-first results. The older FastMCP tutorials use `transport="sse"`. It is easy to implement the wrong transport.

**How to avoid:**
Use `transport="streamable-http"` explicitly in the FastMCP server. Do not implement the SSE endpoints. Verify the MCP spec version in use is 2025-03-26 or later. Add a comment to the server startup code identifying the transport version.

If backward compatibility with older OpenClaw versions is needed, implement the Streamable HTTP transport only and check the OpenClaw version being deployed against the MCP protocol version it supports. If OpenClaw only supports SSE, that is an argument to upgrade OpenClaw, not to implement a deprecated transport.

**Warning signs:**
- `server.py` imports or references `SseServerTransport`.
- The server exposes a `/sse` endpoint.
- Online tutorials used as references are dated before April 2025.

**Phase to address:**
Streamable HTTP transport implementation phase — confirm the transport name explicitly in code review before merging.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Mount MCP app on existing FastAPI with `app.mount()` | One process, one port | Lifespan initialization failure; 307 redirect loops; hard to debug | Never — run as separate process or use combined lifespan |
| Hardcode MCP URL in `openclaw.json` image | Simple build | Breaks on port/namespace change; requires image rebuild for config change | Never for EKS deployment |
| Use same service account as operator | No new RBAC work | Blast radius if MCP server is compromised; violates least-privilege | Never — RBAC is a one-time setup |
| Implement SSE transport for faster start | More examples available | Deprecated April 2026; OpenClaw may drop support | Never — Streamable HTTP is a 2025 standard |
| Skip startup probe on MCP sidecar | Simpler manifests | Race condition on pod cold start; intermittent tool registration failures | Never for production; acceptable for one-off local testing only |
| Bind MCP server to `0.0.0.0` in pod | Accessible for debugging | Exposes MCP to entire cluster network; Origin bypass possible | Never in production; acceptable only behind a NetworkPolicy that blocks all external access |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| FastMCP + FastAPI mount | `app.mount("/mcp", mcp.streamable_http_app())` — lifespan not invoked | Run as separate process on separate port; or compose lifespans manually |
| NemoClaw sandbox + sidecar | Assume loopback traffic is always permitted | Add `127.0.0.1:<port>` to `openclaw-sandbox.yaml` egress allowlist before deploying |
| openclaw.json HTTP registration | Use `"command"/"args"` stdio fields for HTTP server | Use `"url"` field pointing to `http://localhost:<port>/mcp`; generate at pod startup |
| EKS pod identity for MCP server | Reuse operator service account | Create `weka-mcp-server-sa` with read-only + scoped apply ClusterRole |
| Sidecar startup ordering | Both containers start concurrently, OpenClaw loses the race | Define MCP server as native sidecar init container with `startupProbe` |
| MCP session management | Implement stateful sessions assuming client will forward `Mcp-Session-Id` | Default to stateless mode; do not require session ID for single-client sidecar pattern |
| NemoClaw on EKS | Run `nemoclaw install` on EKS worker node | Use separate EC2 VM with Docker, or community `agent-sandbox` CRD (experimental) |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Stateful MCP sessions with Kubernetes HPA | Session affinity required; sticky sessions break with pod scale-out | Use stateless mode for sidecar deployment; sticky sessions only if scaling is needed | When pod count > 1 |
| Python MCP server cold start delay | First tool call after pod restart takes 5-10 seconds while Python imports initialize | Add `startupProbe` with sufficient `failureThreshold`; pre-warm via health endpoint | Every pod restart in EKS (rolling updates, node replacements) |
| Large `openclaw.json` with many MCP servers | OpenClaw loads all tool schemas at session start; context budget consumed | Keep tool descriptions concise (under 200 tokens each); 8 tools already defined — do not add more without removing others | With 8+ verbose tools registered simultaneously |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| MCP server bound to `0.0.0.0` in pod | Any pod in the cluster can call the apply tool directly, bypassing OpenClaw's approval flow | Bind to `127.0.0.1`; for sidecar pattern this is the only address needed |
| NemoClaw sandbox egress unrestricted (policy disabled for convenience) | Agent can exfiltrate cluster data or call external endpoints | Maintain deny-by-default policy; add only `127.0.0.1:<mcp-port>` plus required inference endpoints |
| Production WEKA API credentials shared with NemoClaw sandbox | Credentials exposed to agent context; sandbox policy drift could allow exfiltration | Pass only a read-only WEKA API token to the MCP server; never give the agent direct WEKA API access |
| Operator service account token accessible from sidecar | Sidecar can read the token from `/var/run/secrets/kubernetes.io/serviceaccount/token` and use it for unrestricted cluster operations | Use a dedicated service account for the sidecar pod; mount only the MCP-specific service account |

---

## "Looks Done But Isn't" Checklist

- [ ] **Streamable HTTP transport:** Often the SSE transport is implemented instead — verify `server.py` uses `transport="streamable-http"` and does not expose `/sse` endpoint.
- [ ] **NemoClaw egress policy:** Often loopback address is omitted — verify `openclaw-sandbox.yaml` explicitly allows `127.0.0.1:<mcp-port>` before agent start.
- [ ] **openclaw.json HTTP registration:** Often still uses `"command"/"args"` stdio format — verify it uses `"url": "http://localhost:<port>/mcp"` for the HTTP transport.
- [ ] **Sidecar startup ordering:** Often both containers start concurrently — verify MCP server is defined as native sidecar init container with `restartPolicy: Always` and a `startupProbe`.
- [ ] **Service account RBAC:** Often the operator service account is reused — verify `weka-mcp-server-sa` exists with its own ClusterRole and is referenced in the pod spec.
- [ ] **MCP server binding:** Often bound to `0.0.0.0` — verify the Uvicorn/FastMCP server starts with `host="127.0.0.1"` in the pod.
- [ ] **generate_openclaw_config.py:** Often generates stdio config only — verify it supports `--transport http` output with `"url"` field and is called at pod startup, not build time.
- [ ] **NemoClaw EKS compatibility:** Often assumed to be a standard pod — verify the deployment topology decision (separate VM vs. agent-sandbox CRD) is documented and tested before wiring the MCP server endpoint.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| FastMCP mounted on FastAPI (lifespan failure) | MEDIUM | Move to separate process on separate port; update pod spec and service; update `openclaw.json` URL |
| NemoClaw blocks sidecar (egress policy) | LOW | Edit policy YAML, run `nemoclaw onboard`; no code changes needed |
| NemoClaw incompatible with EKS | HIGH | Switch to separate VM deployment; update MCP server Service to expose externally; update OpenClaw registration URL; retests required |
| Pod IP hardcoded in `openclaw.json` | LOW | Move config generation to pod startup script; parameterize URL from env var |
| MCP server starts after OpenClaw (race condition) | LOW | Add native sidecar definition and `startupProbe` to pod spec |
| Wrong service account used | LOW | Create new service account and ClusterRole; update pod spec; no data migration |
| SSE transport implemented instead of Streamable HTTP | MEDIUM | Rewrite server transport; test new transport end-to-end; update `openclaw.json` registration format |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| FastMCP + FastAPI lifespan failure | Transport implementation | MCP server runs as standalone process; curl to `/health` returns 200 before any agent test |
| NemoClaw egress blocks sidecar | Deployment config | NemoClaw TUI shows no blocked connection to `127.0.0.1` on agent start |
| NemoClaw EKS incompatibility | Architecture/topology decision | Deployment topology documented and smoke-tested before writing any K8s manifests |
| Pod IP in openclaw.json | Registration config | `openclaw.json` is generated at pod startup from env vars; no dotted-decimal IP in the file |
| Session ID forwarding failure | Transport implementation | Server runs in stateless mode; no `Mcp-Session-Id` required; tool calls succeed without session header |
| Sidecar startup race | K8s manifests | MCP server defined as native sidecar; pod start logs show MCP server ready before OpenClaw init |
| Operator service account reuse | K8s manifests | `kubectl auth can-i create wekaappstores --as=system:serviceaccount:<ns>:weka-mcp-server-sa` returns "yes"; `kubectl auth can-i delete nodes` returns "no" |
| Origin header rejection | Transport implementation | Intra-pod curl with no Origin header returns 200; server bound to `127.0.0.1` confirmed in startup logs |
| Build-time openclaw.json | Registration config | Changing `MCP_SERVER_PORT` env var produces correct `openclaw.json` at startup without image rebuild |
| SSE transport implemented instead of Streamable HTTP | Transport implementation | Code review confirms `transport="streamable-http"`; no `/sse` endpoint exists |

---

## Sources

- [MCP Specification 2025-03-26: Transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — Streamable HTTP session management, Origin validation requirements, backwards compatibility (HIGH confidence — official spec)
- [MCP Python SDK Issue #1367: Mounting Streamable HTTP on FastAPI fails](https://github.com/modelcontextprotocol/python-sdk/issues/1367) — lifespan initialization failure, redirect loop root cause (HIGH confidence — active SDK issue)
- [MCP Python SDK Issue #808: FastMCP does not recognize X-Session-ID header](https://github.com/modelcontextprotocol/python-sdk/issues/808) — session header recognition bug (HIGH confidence — verified SDK issue)
- [MCP Python SDK Issue #737: RuntimeError: Received request before initialization was complete](https://github.com/modelcontextprotocol/python-sdk/issues/737) — premature session manager shutdown when embedded in FastAPI (HIGH confidence — verified SDK issue)
- [NVIDIA NemoClaw GitHub: Support OpenShift deployment via agent-sandbox CRD — Issue #407](https://github.com/NVIDIA/NemoClaw/issues/407) — EKS/OpenShift deployment community workaround (MEDIUM confidence — community issue, not official)
- [NVIDIA NemoClaw Documentation: Network Policies](https://docs.nvidia.com/nemoclaw/latest/reference/network-policies.html) — deny-all egress default, allowlist configuration (HIGH confidence — official NVIDIA docs)
- [Scaling HTTP Streamable MCP Servers on Kubernetes: Handling Sticky Sessions](https://zhimin-wen.medium.com/scaling-http-streamable-mcp-servers-on-kubernetes-handling-sticky-sessions-24212857c8ca) — session affinity requirements, stateless mode recommendation (MEDIUM confidence — community article)
- [Why MCP Deprecated SSE and Went with Streamable HTTP — fka.dev](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/) — SSE deprecation rationale and timeline (HIGH confidence — verified against spec)
- [SSE Transport Deprecation: Migration to Streamable HTTP — Keboola](https://changelog.keboola.com/sse-transport-deprecation-migration-to-streamable-http/) — deprecation effective date April 2026 (MEDIUM confidence — third-party migration announcement)
- [Kubernetes Sidecar Containers: Start Sidecar First](https://scalefactory.com/blog/2025/06/19/start-sidecar-first-in-kubernetes/) — native sidecar init container pattern for startup ordering (HIGH confidence — Kubernetes official KEP + community guide)
- [EKS Security Best Practices — Wiz](https://www.wiz.io/academy/container-security/eks-security-best-practices) — IRSA least-privilege, instance role inheritance pitfall (MEDIUM confidence — vendor guide aligned with AWS official docs)
- [Getting Started With NemoClaw: Avoid the Obvious Mistakes — Stormap](https://stormap.ai/post/getting-started-with-nemoclaw-install-onboard-and-avoid-the-obvious-mistakes) — infrastructure stack misunderstanding, policy management mistakes (MEDIUM confidence — community guide)
- Codebase review: `mcp-server/server.py`, `mcp-server/generate_openclaw_config.py` — current stdio transport pattern, config generation baseline (HIGH confidence — direct inspection)

---
*Pitfalls research for: Streamable HTTP MCP transport and NemoClaw/OpenClaw EKS sidecar deployment*
*Researched: 2026-03-23*
