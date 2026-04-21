# Project Research Summary

**Project:** WEKA App Store — v3.0 NemoClaw/OpenClaw EKS Integration
**Domain:** MCP tool server Streamable HTTP transport + EKS sidecar deployment alongside NemoClaw/OpenClaw
**Researched:** 2026-03-23
**Confidence:** MEDIUM-HIGH — transport mechanics are HIGH; NemoClaw EKS topology is a known open question

## Executive Summary

The v3.0 milestone adds a single critical capability to an already-complete 8-tool MCP server: Streamable HTTP transport so the server can run as a pod sidecar alongside a NemoClaw/OpenClaw agent on EKS. The existing tools, tests, Dockerfile, and business logic are all transport-agnostic and require no changes. The only code changes are approximately 10 lines in `server.py` (transport branch + health route) and a config update to `openclaw.json`. Everything else is new Kubernetes manifests — deployment spec, RBAC, and ConfigMaps.

The recommended architecture is a dual-mode server: stdio remains the default for local dev and CI, while `MCP_TRANSPORT=streamable-http` activates HTTP mode for EKS. The MCP server runs as a native Kubernetes sidecar (init container with `restartPolicy: Always`) to guarantee startup ordering before OpenClaw attempts tool registration, binds to `0.0.0.0:8080` inside the pod (required for same-pod networking), and registers with OpenClaw via `http://localhost:8080/mcp`. FastMCP handles session management, SSE, and Origin validation automatically. No new Python dependencies are required — Streamable HTTP is included in `mcp>=1.9`.

The primary deployment risk is NemoClaw's EKS compatibility. NemoClaw is early-preview software (announced March 16, 2026) that runs OpenClaw inside an embedded k3s-in-Docker stack, not as a native EKS pod. Direct `nemoclaw install` on EKS worker nodes does not work without privileged mode and nested container runtimes. Two mitigations exist: deploy NemoClaw on a dedicated EC2 VM alongside the EKS cluster (stable but loses the localhost sidecar topology), or use the community `agent-sandbox` CRD (experimental, GitHub Issue #407). This topology decision must be made and validated before any Kubernetes manifests are written — wrong topology means rewriting all of Phase 3.

## Key Findings

### Recommended Stack

The v3.0 stack requires no new Python packages. `mcp[cli]>=1.26.0` (already pinned from v2.0) includes Streamable HTTP transport since `mcp>=1.9`. The OpenClaw operator is installed via Helm from `oci://ghcr.io/openclaw-rocks/charts/openclaw-operator` and manages pod assembly via the `OpenClawInstance` CRD. The agent container image is `alpine/openclaw:2026.3.11`. All existing v2.0 dependencies — `kubernetes>=27.0.0`, `PyYAML>=6.0.1`, `pytest>=8.0.0`, `pytest-asyncio` — remain unchanged.

**Core technologies:**
- `mcp[cli]>=1.26.0`: Streamable HTTP transport — already installed; `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)` is the complete API; no new package needed
- `openclaw-rocks/k8s-operator` (Helm): OpenClawInstance CRD management — handles StatefulSet, Service, RBAC, NetworkPolicy, PVC, and sidecar injection via `spec.sidecars`
- `alpine/openclaw:2026.3.11`: Agent runtime container — exposes OpenClaw gateway on port 18789; MCP tools registered via `mcpServers.url` in the OpenClaw config
- `helm` v3.x: Operator install tool — one-time cluster setup

**Critical version notes:**
- Transport string is `"streamable-http"` with a hyphen, not underscore
- MCP endpoint is always at `/mcp` path: `http://localhost:8080/mcp`
- SSE transport is deprecated effective April 2026 — do not implement it; `transport="sse"` is the wrong string

### Expected Features

The feature set for this milestone is narrow and well-defined. Everything centers on enabling the 8 existing tools to be called from an EKS-deployed agent over HTTP.

**Must have (table stakes — required for milestone acceptance):**
- Streamable HTTP transport in `server.py` — without it the agent cannot reach the tools in a pod-based deployment; this is the prerequisite for everything else
- `MCP_TRANSPORT` and `MCP_PORT` env vars in `config.py` — transport selected at runtime without image rebuild
- Dockerfile `EXPOSE 8080` and env var entrypoint override — enables the existing image to run in HTTP mode without a new Dockerfile
- `openclaw.json` updated to `"transport": "streamable-http"` with `"url": "http://localhost:8080/mcp"` — replaces the stdio `startup` command/args block
- Kubernetes manifests: pod spec with NemoClaw + MCP sidecar (native sidecar init container pattern), ServiceAccount, RBAC ClusterRole/Binding, ConfigMap for OpenClaw config, ConfigMap for blueprints
- NemoClaw running on EKS (or dedicated VM) with the MCP sidecar registered and reachable
- Happy-path E2E validation: `inspect_cluster` → `inspect_weka` → `list_blueprints` → `get_blueprint` → `get_crd_schema` → `validate_yaml` → `apply` → `status` against a real cluster and real WEKA

**Should have (add after core works — v3.x):**
- Health endpoint at `/healthz` — effectively mandatory for Kubernetes liveness/readiness probes; low effort and required before production stability
- `docker-compose` for local HTTP mode testing — enables local E2E simulation before EKS deployment
- RBAC narrowing — start permissive, narrow after observing actual API calls

**Defer (v4+):**
- TLS on MCP endpoint — only if MCP server is ever exposed outside the pod
- Multi-NemoClaw instance deployment — requires centralized tool registration
- Webhook-based status streaming — only if the WEKA operator adds push notifications

**Anti-features to avoid:**
- Exposing MCP server as a Kubernetes Service with NodePort/LoadBalancer — breaks the localhost model and opens a security surface; the apply tool has no auth middleware
- Running both stdio and HTTP simultaneously in one process — protocol conflict; select transport via env var at startup
- Adding YAML generation or planning logic to the MCP server — reintroduces the v1.0 backend-brain anti-pattern that was explicitly removed
- Removing the existing 103 stdio tests — they are the only tool-logic regression safety net; HTTP transport tests are additive, not replacements

### Architecture Approach

The v3.0 architecture is a sidecar pod pattern: the MCP server container and the NemoClaw/OpenClaw container share a pod's network namespace. The MCP server binds to `0.0.0.0:8080` inside the pod (required — both containers share the pod's loopback interface, so the server must bind to all interfaces to be reachable from the OpenClaw container). OpenClaw registers the tools via `http://localhost:8080/mcp` configured in a Kubernetes ConfigMap injected as `openclaw.json`. Blueprint YAML files are mounted via a Kubernetes ConfigMap (with PVC or git-sync init container as fallback if the catalog exceeds 1MB). All 8 tool modules, all webapp business logic imports, the Dockerfile, and all 103 tests are unchanged.

**Major components:**
1. `server.py` transport branch + `/health` route — ~10 lines; the entire code change in the MCP server
2. `k8s/deployment.yaml` — pod spec with NemoClaw container + MCP sidecar defined as native sidecar init container with startup/readiness/liveness probes
3. `k8s/rbac.yaml` — dedicated `weka-mcp-server-sa` ServiceAccount with scoped ClusterRole (get/list nodes, namespaces, storageclasses, pods; create/list WekaAppStore CRs)
4. `k8s/openclaw-config.yaml` ConfigMap — contains the HTTP `mcpServers.url` registration replacing the stdio startup block; generated at pod startup from env vars
5. `k8s/blueprints-configmap.yaml` ConfigMap — blueprint YAML files mounted as volume into the sidecar at `BLUEPRINTS_DIR`

**Build order (dependency-driven):**
Phase 1 (server code changes) can be completed and fully validated locally — no cluster access needed. Phases 2+ require EKS. Do not proceed to Phase 3 (manifests) until the NemoClaw deployment topology is confirmed from Phase 2.

### Critical Pitfalls

1. **NemoClaw does not run natively on EKS** — NemoClaw's installer creates a k3s-in-Docker stack requiring privileged containers and nested container runtimes not available on EKS worker nodes. Validate the topology decision (dedicated EC2 VM vs. experimental `agent-sandbox` CRD) before writing any Kubernetes manifests. Recovery cost is HIGH if topology is wrong — it requires rewriting all of Phase 3.

2. **FastMCP lifespan failure when mounted on existing FastAPI** — `app.mount("/mcp", mcp.streamable_http_app())` fails with `RuntimeError: Task group is not initialized` because the MCP Starlette app's lifespan is never invoked by the outer FastAPI app. Run the MCP server as a standalone process on its own port; never mount it onto an existing FastAPI app.

3. **Sidecar startup race condition** — Kubernetes starts all `spec.containers` concurrently by default. If OpenClaw attempts tool registration before the MCP server is ready, it marks tools unavailable with no automatic retry. Define the MCP server as a native sidecar init container (`restartPolicy: Always`, Kubernetes 1.29+) with a `startupProbe` on `/health`. This guarantees ordering.

4. **NemoClaw deny-all egress blocks the sidecar** — NemoClaw's OpenShell sandbox enforces egress policy at the application layer, including loopback. Without explicitly adding `127.0.0.1:8080` to `openclaw-sandbox.yaml`, every tool call is blocked. The policy file must be prepared and `nemoclaw onboard` run before the agent pod starts.

5. **RBAC over-permission via operator service account reuse** — Reusing the WEKA operator's service account gives the MCP sidecar full cluster-write permissions. Create `weka-mcp-server-sa` with a scoped ClusterRole. This is a one-time setup task, not a "harden later" item — treat it as a Phase 3 hard gate.

6. **Session ID not forwarded by all MCP clients** — Some OpenClaw/NemoClaw client implementations do not forward the `Mcp-Session-Id` header (confirmed SDK Issue #808), causing the server to treat each tool call as a new session. Design the server for stateless operation from the start; do not rely on session continuity for any tool.

7. **`openclaw.json` baked into the image at build time** — The OpenClaw registration config must be generated at pod startup from environment variables (via init container or startup script writing to an `emptyDir` volume), not baked into the container image. A hardcoded URL breaks on port or namespace changes and requires an image rebuild to fix.

---

## Implications for Roadmap

Based on research, the work falls into four phases with a hard dependency gate between Phase 1 (fully local, no EKS required) and Phase 2+ (requires EKS cluster access and topology validation).

### Phase 1: Streamable HTTP Transport

**Rationale:** HTTP transport is the prerequisite for every other milestone deliverable. It can be built and fully validated locally before any cluster access is available. All pitfalls in this phase are code-level and avoidable with explicit implementation choices. FastMCP documentation is HIGH confidence.

**Delivers:** Working MCP server running in both stdio (default) and Streamable HTTP mode, selected by `MCP_TRANSPORT` env var. `/health` endpoint returning 200. `openclaw.json` updated for HTTP registration with `"url"` field replacing the stdio `startup` block. All 103 existing tests still pass unchanged. Smoke test: `curl localhost:8080/health` and `mcp dev server.py --transport streamable-http` both work.

**Addresses:** HTTP transport mode in server.py (P1), Dockerfile EXPOSE and env var entrypoint (P1), config.py MCP_PORT/MCP_TRANSPORT vars (P1), openclaw.json HTTP transport config (P1)

**Avoids:**
- FastMCP lifespan failure — MCP server runs as standalone process, not mounted on FastAPI
- SSE transport trap — explicitly use `transport="streamable-http"` and verify in code review
- Session ID forwarding failure — initialize server in stateless mode; do not rely on `Mcp-Session-Id`
- Origin header rejection — configure binding address to `0.0.0.0` with explicit pod NetworkPolicy restriction

**Research flag:** Standard patterns, well-documented FastMCP API — skip research-phase. The FastMCP docs at gofastmcp.com are HIGH confidence. No unknowns.

### Phase 2: NemoClaw Topology Decision and EKS Deployment

**Rationale:** NemoClaw's EKS incompatibility (Pitfall 1) is the highest-risk item in the milestone. The topology decision determines the entire Kubernetes manifest structure. Writing manifests before this is resolved wastes effort and requires a full rewrite if the wrong approach is chosen.

**Delivers:** A running NemoClaw/OpenClaw instance accessible on EKS (or an alongside EC2 VM). A documented, tested topology decision. GPU node group and NVIDIA GPU Operator confirmed operational. The NemoClaw container image name confirmed from NVIDIA NGC. NemoClaw startup behavior with sidecar readiness probes confirmed (does it wait, or does it need an init container guard?).

**Addresses:** NemoClaw deployed to EKS (P1 prerequisite), EKS GPU node group setup

**Avoids:**
- NemoClaw EKS incompatibility — validate deployment approach before manifests are written
- Wasted manifest work from wrong topology assumption — Phase 3 cannot start until this gate passes

**Research flag:** NEEDS research-phase. The `agent-sandbox` CRD approach is documented only in GitHub Issue #407 and is not officially supported. The dedicated VM approach is stable but changes the sidecar networking model. Must validate against the actual NemoClaw version available.

### Phase 3: Kubernetes Manifests and Sidecar Wiring

**Rationale:** With transport working (Phase 1) and topology confirmed (Phase 2), the Kubernetes artifacts can be built correctly once. All manifest decisions depend on topology. RBAC, startup ordering, ConfigMap structure, and the NemoClaw egress policy must all be addressed here — not deferred to post-deployment hardening.

**Delivers:** Complete Kubernetes manifest set: Deployment/StatefulSet or OpenClawInstance CRD YAML with native sidecar init container, `weka-mcp-server-sa` ServiceAccount, ClusterRole with scoped permissions, ClusterRoleBinding, OpenClaw config ConfigMap (generated at pod startup), blueprints ConfigMap. NemoClaw egress policy updated to allow `127.0.0.1:8080`. Both containers running, `/health` returning 200, `kubectl logs` clean.

**Addresses:** Kubernetes deployment manifests (P1), ServiceAccount + RBAC (P1), blueprint data strategy, sidecar startup ordering, openclaw.json runtime generation

**Avoids:**
- Sidecar startup race — native sidecar init container with `startupProbe` on `/health`
- RBAC over-permission — dedicated `weka-mcp-server-sa` with scoped ClusterRole
- NemoClaw egress blocking sidecar — `openclaw-sandbox.yaml` updated before pod start
- Build-time `openclaw.json` — generated at pod startup from env vars via init container writing to emptyDir
- Pod IP hardcoded in config — URL is env-var-generated at startup, no dotted-decimal IPs in any config file

**Uses:** OpenClaw operator `spec.sidecars` or equivalent NemoClaw pod spec; `MCP_TRANSPORT=streamable-http`; `KUBERNETES_AUTH_MODE=in-cluster`

**Research flag:** NEEDS research-phase for the NemoClaw ConfigMap field path for `mcpServers.url` (official NemoClaw Kubernetes docs do not confirm the schema) and NemoClaw startup behavior with sidecar readiness probes. Standard Kubernetes RBAC and ConfigMap patterns are well-documented and do not need research.

### Phase 4: End-to-End Validation

**Rationale:** With all components deployed, validate the full happy path against a real cluster and real WEKA. This is the milestone acceptance criterion. Validation is functional testing — no research or design uncertainty remains at this point.

**Delivers:** Agent completes the full tool chain (`inspect_cluster` → `inspect_weka` → `list_blueprints` → `get_blueprint` → `get_crd_schema` → `validate_yaml` → `apply` → `status`) against a live EKS cluster and live WEKA storage. SKILL.md reviewed and any stdio-specific transport language removed.

**Addresses:** Happy-path E2E validation (P1), SKILL.md transport reference cleanup

**Avoids:**
- Tool registration failures from startup race — pod start logs confirm MCP server ready before OpenClaw init
- Stale `openclaw.json` — confirm URL was generated at pod startup, not from image

**Research flag:** Standard execution — no research-phase needed. Validation is functional testing against live infrastructure.

### Phase Ordering Rationale

- **Phase 1 is fully local and independent:** Transport code changes are ~10 lines with HIGH-confidence documentation. Completing this first creates a validated artifact before any cluster cost is incurred.
- **Phase 2 before Phase 3:** The NemoClaw EKS compatibility issue (recovery cost HIGH) makes topology discovery non-negotiable before manifest authoring begins. A wrong topology assumption means rewriting all of Phase 3.
- **Phase 3 is where the unknown complexity lives:** RBAC, startup ordering, egress policy, and config generation are all Kubernetes problems with known solutions — but the correct solutions depend on topology from Phase 2.
- **Phase 4 is validation, not development:** No design decisions remain. It confirms the deployment works end-to-end.
- **Phases 2 and 3 carry the schedule risk:** Phase 1 (code) and Phase 4 (validation) are predictable. Phase 2 (NemoClaw topology) and Phase 3 (manifests) are where unexpected iterations are most likely given NemoClaw's early-preview status.

### Research Flags

Phases needing `/gsd:research-phase` during planning:
- **Phase 2:** NemoClaw EKS deployment topology — `agent-sandbox` CRD (experimental, Issue #407 only) vs. dedicated EC2 VM; neither approach is officially documented for production EKS use
- **Phase 3:** NemoClaw ConfigMap field path for `mcpServers.url`; NemoClaw startup behavior with sidecar readiness probes; exact NemoClaw container image name on NVIDIA NGC (placeholder only in current research)

Phases with standard patterns (skip research-phase):
- **Phase 1:** FastMCP Streamable HTTP API is fully documented; dual-mode transport pattern is explicit; no unknowns
- **Phase 4:** Functional E2E testing; no design uncertainty

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | `mcp[cli]>=1.26.0` Streamable HTTP verified via PyPI and official FastMCP docs; OpenClaw operator verified via official docs and GitHub; no new packages needed |
| Features | HIGH | Feature set is narrow and well-bounded; transport mechanics confirmed against official MCP spec; NemoClaw EKS-specific features are MEDIUM due to early-preview status |
| Architecture | MEDIUM | FastMCP HTTP API is HIGH confidence; sidecar `mcpServers.url` registration format is MEDIUM (multiple community sources, no official NemoClaw Kubernetes example found); NemoClaw EKS pod spec is LOW-MEDIUM |
| Pitfalls | HIGH | Critical pitfalls sourced from official SDK issue tracker (Issues #1367, #808, #737), official NemoClaw network policy docs, official Kubernetes native sidecar KEP, and EKS security best practices |

**Overall confidence:** MEDIUM-HIGH — The code changes (Phase 1) are HIGH confidence with no implementation uncertainty. The deployment topology (Phase 2) has one known unresolved question that is a known unknown, not a surprise discovery. The manifest work (Phase 3) is HIGH confidence once topology is resolved.

### Gaps to Address

- **NemoClaw EKS deployment topology (block on Phase 3):** Must select between VM+Service approach and experimental `agent-sandbox` CRD before Phase 3 manifests are written. Confirm via hands-on test with the actual NemoClaw version available. This is the single highest-risk item in the milestone.
- **NemoClaw container image name (block on Phase 3):** `nvcr.io/nvidia/nemoclaw:latest` is a placeholder. Confirm the actual NGC image name and tag before writing the Deployment manifest.
- **NemoClaw ConfigMap schema (block on Phase 3):** The exact YAML key path for `mcpServers.url` in NemoClaw's runtime config must be verified against the deployed version. Community sources agree on `mcpServers.<name>.url` with `transport: streamable-http` but official NemoClaw Kubernetes docs do not confirm this schema.
- **NemoClaw sidecar startup behavior (block on Phase 3):** Whether NemoClaw/OpenClaw waits for sidecar readiness probes or starts immediately is not confirmed. If it does not retry, a startup guard is needed. Validate during Phase 2 deployment.
- **Blueprint catalog size (resolve before Phase 3):** If the blueprint YAML catalog exceeds ConfigMap's 1MB limit, a PVC or git-sync init container is needed instead of a ConfigMap. Measure actual blueprint directory size before committing to the ConfigMap strategy.

---

## Sources

### Primary (HIGH confidence)
- PyPI `mcp` package — v1.26.0 confirmed latest stable: https://pypi.org/project/mcp/
- FastMCP Running Server docs — `mcp.run(transport="streamable-http", host, port)` signature: https://gofastmcp.com/deployment/running-server
- MCP Transports Specification 2025-03-26 — Streamable HTTP session management, Origin validation: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
- MCP Python SDK Issue #1367 — FastAPI mount lifespan failure root cause: https://github.com/modelcontextprotocol/python-sdk/issues/1367
- MCP Python SDK Issue #808 — Session ID header recognition bug: https://github.com/modelcontextprotocol/python-sdk/issues/808
- MCP Python SDK Issue #737 — RuntimeError on premature session manager shutdown: https://github.com/modelcontextprotocol/python-sdk/issues/737
- NVIDIA NemoClaw Network Policies docs — deny-all egress default, allowlist config: https://docs.nvidia.com/nemoclaw/latest/reference/network-policies.html
- Kubernetes native sidecar KEP + community guide — startup ordering guarantee: https://scalefactory.com/blog/2025/06/19/start-sidecar-first-in-kubernetes/
- OpenClaw K8s operator official docs — `spec.sidecars`, reserved container names: https://docs.openclaw.ai/install/kubernetes
- openclaw-rocks/k8s-operator GitHub — sidecar architecture, reserved names: https://github.com/openclaw-rocks/k8s-operator
- SSE Transport Deprecation rationale and timeline: https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/

### Secondary (MEDIUM confidence)
- OpenClaw k8s-operator sidecar DeepWiki — sidecar categories, naming constraints: https://deepwiki.com/openclaw-rocks/k8s-operator/5.1-sidecar-containers
- openclaw.rocks deploy guide — `OpenClawInstance` CRD YAML with `spec.sidecars`: https://openclaw.rocks/blog/deploy-openclaw-kubernetes
- community openclaw-mcp-server — `streamable-http` URL format `http://HOST:PORT/mcp`: https://github.com/rodgco/openclaw-mcp-server
- masteryodaa/openclaw-sdk DeepWiki — `HttpMcpServer` `transport: "streamable-http"`, `url` field: https://deepwiki.com/masteryodaa/openclaw-sdk/2.15-mcp-server-integration
- NVIDIA NemoClaw Quickstart — installer script, early preview caveats: https://docs.nvidia.com/nemoclaw/latest/get-started/quickstart.html
- NVIDIA GPU Operator for EKS — Helm installation, EKS compatibility: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
- dev.to OpenClaw on Kubernetes — `alpine/openclaw:2026.3.11` image, port 18789: https://dev.to/thenjdevopsguy/running-openclaw-on-kubernetes-57ki
- MCPcat Streamable HTTP guide — endpoint path, production config: https://mcpcat.io/guides/building-streamablehttp-mcp-server/
- openclawvps.io — `mcpServers` block with `streamable-http` transport and `url` field: https://openclawvps.io/blog/add-mcp-openclaw
- Scaling HTTP Streamable MCP Servers — stateless mode recommendation: https://zhimin-wen.medium.com/scaling-http-streamable-mcp-servers-on-kubernetes-handling-sticky-sessions-24212857c8ca
- EKS Security Best Practices — IRSA least-privilege, instance role inheritance pitfall: https://www.wiz.io/academy/container-security/eks-security-best-practices

### Tertiary (LOW confidence — needs validation during implementation)
- NVIDIA/NemoClaw GitHub Issue #407 — `agent-sandbox` CRD community workaround for EKS (experimental, not officially supported): https://github.com/NVIDIA/NemoClaw/issues/407
- NemoClaw GitHub — Kubernetes deployment docs absent; local-deploy focus only; EKS config schema not published: https://github.com/NVIDIA/NemoClaw

---
*Research completed: 2026-03-23*
*Extends: .planning/research/SUMMARY.md v2.0 (2026-03-20) — v3.0 HTTP transport and EKS sidecar additions*
*Ready for roadmap: yes*
