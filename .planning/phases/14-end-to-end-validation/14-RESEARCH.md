# Phase 14: End-to-End Validation - Research

**Researched:** 2026-03-25
**Domain:** Kubernetes Gateway API (Envoy), OpenClaw E2E chat testing, kubectl evidence capture, WekaAppStore operator reconciliation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Target Blueprint**
- OSS Rag blueprint — the specific blueprint for the happy-path E2E test
- Requires WEKA storage (wekafs storage class) — WEKA CSI is live and configured
- Requires GPU resources — cluster has GPU nodes available
- Enough cluster capacity for a full deploy (not a dry-run)
- Leave the deployed blueprint running after test — no cleanup

**Agent Access**
- Chat via OpenClaw Web UI in browser
- Set up an Envoy Gateway API HTTPRoute to expose the OpenClaw UI
- Hostname: `openclaw.example.com`
- Uses Kubernetes Gateway API with HTTPRoute resources (not traditional Ingress)
- OpenClaw gateway is at localhost:18789 in the pod — HTTPRoute backends to the pod/service

**WEKA Storage**
- WEKA CSI is live and working in the EKS cluster
- wekafs storage class exists, PVCs can be provisioned
- inspect_weka will return real data (free capacity, filesystems, mounts)
- Enough free WEKA capacity to actually deploy OSS Rag

**Evidence and Pass/Fail Criteria**
- Evidence format: chat transcripts paired with kubectl outputs
- E2E-01: Agent returns real cluster resource data (GPU, CPU, RAM, namespaces) through chat — kubectl output confirms matching values
- E2E-02: Agent lists and describes blueprints through chat — OSS Rag appears in the catalog
- E2E-03: Agent generates, validates, and applies a WekaAppStore CR through full SKILL.md workflow — `kubectl get wekaappstores` shows the CR
- E2E-04: Agent reports FULL deployment success (not just CR creation) — operator fully reconciles all components, agent confirms everything is Running/Ready
- Chat transcripts saved as evidence alongside kubectl verification commands

### Claude's Discretion
- HTTPRoute manifest structure and TLS configuration for openclaw.example.com
- Whether to create an Envoy Gateway or reuse an existing one in the cluster
- How to expose the OpenClaw WebSocket gateway through Envoy (WebSocket upgrade handling)
- Timeout settings for waiting on operator reconciliation
- Evidence file format and storage location

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| E2E-01 | Agent can inspect cluster resources and WEKA storage through chat | SKILL.md steps 1-2 (inspect_cluster, inspect_weka); pod running after Phase 13 confirms tools accessible |
| E2E-02 | Agent can list and describe blueprints through chat | SKILL.md steps 3-4 (list_blueprints, get_blueprint); git-sync keeps catalog current |
| E2E-03 | Agent can generate, validate, and apply a WekaAppStore CR through the full SKILL.md workflow | SKILL.md steps 5-11 (schema → YAML → validate → apply); RBAC grants create on wekaappstores.warp.io |
| E2E-04 | Agent reports deployment status after apply | SKILL.md step 12 (status); operator reconciliation visible via kubectl get wekaappstores |
</phase_requirements>

---

## Summary

Phase 14 is an integration/validation phase, not a code-writing phase. The primary work is infrastructure setup (HTTPRoute manifest to expose OpenClaw UI) and then conducting a live chat session with the agent that exercises the full 12-step SKILL.md workflow against OSS Rag. Evidence is captured as chat transcript files paired with kubectl verification outputs.

The critical new infrastructure gap is that the `openclaw-sandbox` pod has no Kubernetes Service, and the OpenClaw gateway uses `--bind=loopback` (127.0.0.1:18789), meaning Envoy cannot route directly to the container port from outside the pod. This requires creating a Service of type ClusterIP that exposes port 18789 and targets the pod, then an HTTPRoute pointing to that Service. The existing `warp-edge-gateway` (Envoy, port 80) already supports `allowedRoutes.namespaces.from: All`, so it can accept an HTTPRoute from the `wekaappstore` namespace. The pattern is already established by `grafana-route.yaml`, `appstore-route.yaml`, etc.

WebSocket upgrade is the key technical concern: OpenClaw Web UI communicates via WebSocket on port 18789. Envoy Gateway (v1.5.6) handles WebSocket upgrade automatically when the protocol field in the backend is left as HTTP (no special annotation needed for standard Gateway API v1). The HTTPRoute will carry both HTTP and WebSocket traffic to port 18789.

**Primary recommendation:** Create a ClusterIP Service for the openclaw-sandbox pod port 18789, write an HTTPRoute to `warp-edge-gateway` with hostname `openclaw.example.com`, then conduct the chat session following SKILL.md steps 1-12, capturing evidence at each stage.

---

## Standard Stack

### Core (all already deployed in the cluster)

| Component | Version/Config | Purpose | Why Standard |
|-----------|---------------|---------|--------------|
| Envoy Gateway | v1.5.6 (Helm) | Routes external HTTP/WS traffic to cluster services | Already installed as `warp-edge-gateway` in `envoy-gateway-system` |
| Kubernetes Gateway API | v1.2.0 | HTTPRoute / Gateway resources | Already installed; all cluster routes use this |
| warp-edge-gateway | Gateway in `envoy-gateway-system` | The single edge gateway, port 80, `from: All` namespaces | Established pattern; all existing routes use it |
| agent-sandbox Sandbox CR | agents.x-k8s.io/v1alpha1 | Hosts OpenClaw + MCP sidecar pod | Phase 12-13 deployed; pod is Running |

### Supporting

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| kubectl exec | Health checks, evidence capture | Verifying pod internals during E2E |
| kubectl get wekaappstores | Confirms CR creation (E2E-03) | After agent apply step |
| kubectl describe wekaappstore | Operator reconciliation status (E2E-04) | Monitoring operator progress |

### Alternatives Considered

| Standard Choice | Alternative | Tradeoff |
|----------------|-------------|----------|
| Reuse `warp-edge-gateway` | Create new Gateway for openclaw | Reuse is simpler — no new LB provisioned, no new IP to update in /etc/hosts |
| ClusterIP Service + HTTPRoute | NodePort direct | HTTPRoute is the project pattern; NodePort bypasses Envoy and is inconsistent |
| No TLS (HTTP on port 80) | TLS on port 443 | CONTEXT.md defers TLS to Claude's discretion; HTTP is consistent with all existing routes (grafana, appstore, molstar, argo all use HTTP/port 80) |

**Installation:** No new installs required. All components already running.

---

## Architecture Patterns

### OpenClaw Service + HTTPRoute Pattern

The openclaw-sandbox pod is managed by the agent-sandbox operator. The operator creates the pod but does NOT create a Service automatically. A manual Service is needed.

**Problem:** `--bind=loopback` means OpenClaw gateway only listens on 127.0.0.1:18789 within the pod. Envoy cannot route to a loopback address — it routes to a Service ClusterIP which hits the pod's external-facing interface.

**Resolution options:**
1. Create a ClusterIP Service selecting the openclaw-sandbox pod on port 18789 — this is the cleanest pattern matching all existing routes in the cluster.
2. The agent-sandbox operator sets a hash label (`agents.x-k8s.io/sandbox-name-hash: 62f96e10`) as the pod selector. The Service selector must use this label OR a stable label. However, relying on a hash label is fragile. Better: add a stable label to the Sandbox CR's pod template spec and use that in the Service selector.

**Recommended approach:** Add `labels: app: openclaw-sandbox` under `spec.podTemplate.spec.` (or use the existing hash label if the pod already has it and it's stable). The Service selects `agents.x-k8s.io/sandbox-name-hash: 62f96e10` (confirmed from TOPOLOGY.md).

**CRITICAL PITFALL:** The loopback bind (`--bind=loopback`) means 127.0.0.1:18789. A Kubernetes Service proxies traffic to the pod's eth0 interface IP, not loopback. This means traffic routed via Service to port 18789 will be refused if OpenClaw only listens on 127.0.0.1. **Resolution:** The Sandbox CR must be updated to use `--bind=lan` instead of `--bind=loopback`, OR the HTTPRoute must target the pod IP directly (not a Service). Per TOPOLOGY.md note: `--bind=lan` requires `controlUi.allowedOrigins` config. The fix is to pass a `gateway.controlUi.allowedOrigins` config or set the env var that enables LAN bind without config restriction.

**Alternative resolution to loopback problem:** Use `kubectl port-forward` as a bridge (not production-worthy, but acceptable for E2E test). This runs on the developer's machine: `kubectl port-forward pod/<pod> 18789:18789 -n wekaappstore`. The browser connects to localhost:18789 without Envoy. This avoids the `--bind=loopback` problem entirely and does not require a Service or HTTPRoute change. The "openclaw.example.com" hostname becomes localhost:18789 accessed locally.

**Decision left to planner:** The CONTEXT.md marks HTTPRoute structure as Claude's discretion. Given the loopback complication, the research recommends the planner evaluate port-forward vs. Service+HTTPRoute. Both achieve E2E chat access; port-forward is simpler for a one-time validation, Service+HTTPRoute is the production-worthy form.

### HTTPRoute Manifest Pattern (if Service+HTTPRoute approach chosen)

```yaml
# Source: cluster_init/routes/ — all existing routes follow this pattern
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: openclaw-ui
  namespace: wekaappstore
spec:
  parentRefs:
    - name: warp-edge-gateway
      namespace: envoy-gateway-system
  hostnames:
    - openclaw.example.com
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: openclaw-gateway-svc
          port: 18789
```

```yaml
# Service must target the openclaw pod using the hash label from TOPOLOGY.md
apiVersion: v1
kind: Service
metadata:
  name: openclaw-gateway-svc
  namespace: wekaappstore
spec:
  selector:
    agents.x-k8s.io/sandbox-name-hash: 62f96e10
  ports:
    - port: 18789
      targetPort: 18789
      protocol: TCP
  type: ClusterIP
```

### WebSocket Handling

Envoy Gateway v1.5.6 with standard Gateway API HTTPRoute handles WebSocket upgrade automatically. No `Upgrade` annotation or BackendLBPolicy is required for basic WebSocket proxying. The HTTPRoute protocol HTTP carries both HTTP/1.1 and WebSocket upgrade headers transparently.

**Confidence:** MEDIUM — based on Envoy Gateway documentation pattern and general Gateway API knowledge. Verify by testing WebSocket connection after deploy.

### SKILL.md Workflow as Test Script

The 12-step SKILL.md is the authoritative test script for E2E-01 through E2E-04:

| Step | SKILL.md Action | Requirement Covered |
|------|----------------|---------------------|
| 1 | `inspect_cluster` | E2E-01 (cluster resources) |
| 2 | `inspect_weka` | E2E-01 (WEKA storage data) |
| 3 | `list_blueprints` | E2E-02 (catalog listing) |
| 4 | `get_blueprint` (OSS Rag) | E2E-02 (blueprint description) |
| 5 | `get_crd_schema` | — (internal) |
| 6 | Generate YAML | — (internal) |
| 7-8 | `validate_yaml` + retry loop | E2E-03 (validation gate) |
| 9 | Re-run `inspect_cluster` | — (safety) |
| 10 | Present plan, await approval | E2E-03 (explicit confirmation) |
| 11 | `apply` (confirmed=true) | E2E-03 (CR created) |
| 12 | `status` (repeat until Ready) | E2E-04 (operator reconciliation) |

### Evidence Capture Pattern

Each requirement needs paired evidence:

```bash
# E2E-01 evidence: capture kubectl node/resource data to compare against chat transcript
kubectl get nodes -o wide -n wekaappstore > evidence/e2e-01-nodes.txt
kubectl describe node ip-172-3-1-203.us-west-2.compute.internal >> evidence/e2e-01-nodes.txt

# E2E-02 evidence: confirm OSS Rag appears in catalog
# (from chat transcript — no kubectl equivalent; blueprint is in git-sync'd catalog)

# E2E-03 evidence: confirm CR exists after apply
kubectl get wekaappstores -n wekaappstore -o wide > evidence/e2e-03-cr-created.txt
kubectl get wekaappstore oss-rag -n wekaappstore -o yaml >> evidence/e2e-03-cr-created.txt

# E2E-04 evidence: operator reconciliation status
kubectl get wekaappstore oss-rag -n wekaappstore -o jsonpath='{.status}' > evidence/e2e-04-status.txt
kubectl describe wekaappstore oss-rag -n wekaappstore >> evidence/e2e-04-status.txt
kubectl get pods -n wekaappstore >> evidence/e2e-04-status.txt
```

Chat transcripts saved as Markdown files: `evidence/chat-transcript-e2e.md` with timestamps.

### Anti-Patterns to Avoid

- **Starting the E2E session before Phase 13 preconditions are met:** Run `scripts/validate-phase13.sh --live` first. If any FAIL, the MCP tools will not work.
- **Assuming `--bind=loopback` is transparent to Envoy:** It is not. Envoy routes via Service ClusterIP which hits eth0, not loopback. Must use port-forward or change bind mode.
- **Applying the blueprint before agent gives explicit confirmation prompt:** The SKILL.md workflow requires `confirmed=true` be set only after user says "proceed" in chat — do not bypass this even in testing.
- **Capturing evidence only from kubectl:** E2E-01 requires that the chat transcript shows values matching kubectl output. Screenshot the chat or save it as text; kubectl output alone is insufficient evidence.
- **Using a different blueprint than OSS Rag:** Locked decision. OSS Rag is the target.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP routing to OpenClaw | Custom Nginx sidecar | Envoy Gateway HTTPRoute | Envoy already deployed; HTTPRoute is the project standard |
| WebSocket proxying | Manual TCP tunnel | Envoy handles it automatically | Standard WebSocket upgrade is transparent to Gateway API |
| Blueprint sync | Manual copy into pod | git-sync container (already running) | Already in Phase 13 Sandbox CR |
| Cluster resource verification | Custom script | kubectl get nodes / describe | Existing tools; no new code needed |
| Operator status polling | Custom watch script | kubectl get/describe wekaappstore | Operator updates .status field; kubectl describe shows conditions |

---

## Common Pitfalls

### Pitfall 1: loopback bind blocks external routing

**What goes wrong:** Envoy routes to the Service ClusterIP, which delivers packets to the pod's eth0 interface. OpenClaw is listening only on 127.0.0.1:18789. The connection is refused.
**Why it happens:** `--bind=loopback` was chosen in Phase 12 because `--bind=lan` requires `gateway.controlUi.allowedOrigins` config unavailable in Sandbox CRD mode.
**How to avoid:** Use `kubectl port-forward pod/<pod-name> 18789:18789 -n wekaappstore` on the dev machine. The browser connects to http://localhost:18789 directly. Alternatively, find the OpenClaw config that enables LAN bind and update the Sandbox CR.
**Warning signs:** Envoy returns 503 Bad Gateway or connection refused for openclaw.example.com.

### Pitfall 2: Pod selector changes after Sandbox CR update

**What goes wrong:** If the Sandbox CR is modified (e.g., to change `--bind` mode), the agent-sandbox operator may create a new pod with a different hash label. The Service selector `agents.x-k8s.io/sandbox-name-hash: 62f96e10` becomes stale.
**Why it happens:** The hash is derived from the Sandbox CR spec — any change generates a new hash.
**How to avoid:** Re-query the selector after any Sandbox CR change: `kubectl get sandbox openclaw-sandbox -n wekaappstore -o jsonpath='{.status.selector}'`
**Warning signs:** Service shows 0 endpoints; `kubectl describe svc openclaw-gateway-svc` shows no ready addresses.

### Pitfall 3: OSS Rag blueprint not yet in warp-blueprints git repo

**What goes wrong:** Agent calls `list_blueprints` and OSS Rag is not in the catalog. git-sync has synced the repo but the blueprint was not committed.
**Why it happens:** The warp-blueprints repo may not yet contain the OSS Rag blueprint definition.
**How to avoid:** Before the E2E session, exec into the MCP sidecar and verify: `kubectl exec <pod> -c weka-mcp-sidecar -- ls /app/blueprints/` and confirm OSS Rag is present.
**Warning signs:** Agent lists blueprints but OSS Rag is absent; `list_blueprints` returns empty or partial catalog.

### Pitfall 4: Operator reconciliation takes longer than expected

**What goes wrong:** E2E-04 requires FULL deployment success — all components Running/Ready. The operator may take several minutes to pull images, create PVCs (WEKA), and start all pods.
**Why it happens:** OSS Rag has GPU + WEKA storage components that require image pulls, CSI provisioning, and GPU scheduling.
**How to avoid:** Wait up to 10-15 minutes between agent `apply` and calling `status`. SKILL.md step 12 already instructs to retry status polling "until Ready or Failed." Document expected duration in the evidence file.
**Warning signs:** `appStackPhase` stuck in `Pending` or `Progressing` beyond 5 minutes — check individual pod events with `kubectl describe pod`.

### Pitfall 5: MCP health endpoint not ready before E2E chat session

**What goes wrong:** Agent starts the SKILL.md workflow but tool calls return errors because the MCP sidecar restarted (imagePullPolicy: Always may pull a new image on pod restart).
**Why it happens:** If the Sandbox CR was modified for the HTTPRoute work, the operator recreates the pod and the MCP sidecar goes through a readiness cycle.
**How to avoid:** Run `scripts/validate-phase13.sh --live` immediately before beginning the E2E chat session. Confirm check 7 (sidecar Ready) and check 8 (health 200) both PASS.

### Pitfall 6: Evidence not captured in real-time

**What goes wrong:** Chat transcript is not saved; browser closes; evidence is lost.
**Why it happens:** OpenClaw Web UI chat is ephemeral.
**How to avoid:** Copy the full chat exchange to a file during or immediately after the session. Use browser developer tools or manual copy-paste into `evidence/chat-transcript-e2e.md`.

---

## Code Examples

### Verify Phase 13 preconditions before E2E

```bash
# Source: scripts/validate-phase13.sh (existing script)
bash scripts/validate-phase13.sh wekaappstore --live
```

Expected: "Phase 13 Validation PASSED (10 checks passed, 0 warnings)"

### Get dynamic pod selector (handles hash label changes)

```bash
# Source: scripts/validate-phase13.sh and validate-topology.sh — established pattern
SELECTOR=$(kubectl get sandbox openclaw-sandbox -n wekaappstore \
  -o jsonpath='{.status.selector}')
POD=$(kubectl get pods -n wekaappstore -l "${SELECTOR}" \
  -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD   Selector: $SELECTOR"
```

### Port-forward approach (recommended for loopback bypass)

```bash
# Source: standard kubectl; no project-specific modification needed
kubectl port-forward pod/$POD 18789:18789 -n wekaappstore
# Browser: http://localhost:18789
```

### HTTPRoute manifest for openclaw.example.com

```yaml
# Source: cluster_init/routes/ pattern — consistent with all existing routes
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: openclaw-ui
  namespace: wekaappstore
spec:
  parentRefs:
    - name: warp-edge-gateway
      namespace: envoy-gateway-system
  hostnames:
    - openclaw.example.com
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /
      backendRefs:
        - name: openclaw-gateway-svc
          port: 18789
```

### Service for openclaw pod (required before HTTPRoute)

```yaml
# Source: required new manifest — no existing openclaw Service exists
apiVersion: v1
kind: Service
metadata:
  name: openclaw-gateway-svc
  namespace: wekaappstore
spec:
  selector:
    agents.x-k8s.io/sandbox-name-hash: "62f96e10"
  ports:
    - name: gateway
      port: 18789
      targetPort: 18789
      protocol: TCP
  type: ClusterIP
```

Note: hash `62f96e10` is from TOPOLOGY.md and must be confirmed live. Re-query if Sandbox CR changes.

### E2E evidence capture commands

```bash
# After E2E chat session — save to evidence/
mkdir -p evidence

# E2E-01: cluster resources
kubectl get nodes -o wide > evidence/e2e-01-nodes.txt
kubectl describe node ip-172-3-1-203.us-west-2.compute.internal | \
  grep -A5 "Capacity:\|Allocatable:" >> evidence/e2e-01-nodes.txt

# E2E-03: CR created
kubectl get wekaappstores -A -o wide > evidence/e2e-03-wekaappstores.txt

# E2E-04: operator reconciliation outcome
kubectl describe wekaappstore <cr-name> -n wekaappstore > evidence/e2e-04-reconcile.txt
kubectl get pods -n wekaappstore >> evidence/e2e-04-reconcile.txt
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SSE transport for MCP | Streamable HTTP (stateless) | April 2025 MCP spec; Phase 11 | OpenClaw uses streamable-http; SSE deprecated |
| Ingress (nginx) | Gateway API HTTPRoute | Gateway API v1 stable (2024) | All cluster routes use HTTPRoute; Ingress is not used |
| Dedicated EC2 VM for NemoClaw | agent-sandbox CRD (Kubernetes-native) | Phase 12 user decision | Pod shares network namespace; loopback access to MCP sidecar |

**Deprecated/outdated:**
- Traditional Kubernetes Ingress: Not used in this cluster. All routing via Gateway API HTTPRoute.
- SSE MCP transport: Deprecated April 2026. Phase 11 removed it from the MCP server.

---

## Open Questions

1. **Will `--bind=loopback` prevent Envoy from reaching OpenClaw on port 18789?**
   - What we know: `--bind=loopback` makes the gateway listen only on 127.0.0.1. Envoy routes via Service ClusterIP to eth0. These are different interfaces.
   - What's unclear: Whether OpenClaw has an environment variable or config option to allow LAN bind without `controlUi.allowedOrigins`.
   - Recommendation: Plan Wave 1 as port-forward approach (zero risk, immediate access). Plan Wave 2 as Service+HTTPRoute if LAN bind is achievable. Planner should create a diagnostic task first.

2. **Is OSS Rag blueprint present in the warp-blueprints GitHub repo?**
   - What we know: git-sync syncs `https://github.com/weka/warp-blueprints.git` every 60 seconds.
   - What's unclear: Whether the OSS Rag blueprint definition has been committed to that repo.
   - Recommendation: Wave 0 task should exec into MCP sidecar and verify blueprint exists in `/app/blueprints/` before starting E2E chat session.

3. **How long does OSS Rag operator reconciliation take?**
   - What we know: OSS Rag requires WEKA storage PVCs + GPU workloads. WEKA CSI provisioning is live and functional.
   - What's unclear: Number of components in OSS Rag, total pod count, image sizes.
   - Recommendation: Build in 15-minute poll window in the E2E plan. E2E-04 is only PASS when `appStackPhase == Ready` (not just when CR is created).

4. **Does the agent-sandbox operator hash label change if the Sandbox CR is modified?**
   - What we know: TOPOLOGY.md shows hash `62f96e10` for the current Sandbox CR.
   - What's unclear: Whether any modifications in Phase 14 (e.g., changing `--bind=lan`) will regenerate the hash.
   - Recommendation: Always resolve the selector dynamically using `kubectl get sandbox ... -o jsonpath='{.status.selector}'` in scripts rather than hardcoding the hash.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | bash scripts (existing pattern); kubectl-based assertion |
| Config file | none — scripts are standalone |
| Quick run command | `bash scripts/validate-phase13.sh wekaappstore --live` (Phase 13 pre-check) |
| Full suite command | `bash scripts/validate-phase13.sh wekaappstore --live` + `kubectl get wekaappstores -n wekaappstore` + manual chat transcript review |

### Phase Requirements — Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| E2E-01 | Agent returns real GPU/CPU/RAM/namespace data matching cluster | manual (chat) + automated kubectl assertion | `kubectl describe node ip-172-3-1-203.us-west-2.compute.internal \| grep -E "nvidia.com/gpu\|Allocatable"` | ❌ Wave 0 |
| E2E-02 | Agent lists blueprints; OSS Rag appears | manual (chat) + automated catalog check | `kubectl exec $POD -c weka-mcp-sidecar -- ls /app/blueprints/` | ❌ Wave 0 |
| E2E-03 | WekaAppStore CR exists after apply | manual (chat) + automated kubectl assertion | `kubectl get wekaappstores -n wekaappstore -o name \| grep oss-rag` | ❌ Wave 0 |
| E2E-04 | Operator reconciles all components; agent reports Ready | manual (chat) + automated status check | `kubectl get wekaappstore <name> -n wekaappstore -o jsonpath='{.status.appStackPhase}'` | ❌ Wave 0 |

**Note:** E2E requirements are predominantly manual validation (chat session is human-driven). Automated commands confirm the kubectl-side of the evidence; chat transcript is the primary evidence artifact.

### Sampling Rate

- **Per task commit:** `bash scripts/validate-phase13.sh wekaappstore --live` (confirm sidecar still healthy)
- **Per wave merge:** full evidence set: nodes output + wekaappstores output + status output + chat transcript
- **Phase gate:** all 4 evidence files present + chat transcript shows all 12 SKILL.md steps completed + `appStackPhase == Ready`

### Wave 0 Gaps

- [ ] `scripts/validate-phase14-prereqs.sh` — confirms Phase 13 PASS + OSS Rag blueprint present + pod selector current
- [ ] `evidence/` directory — storage location for chat transcript and kubectl outputs
- [ ] Resolve loopback bind question: determine whether port-forward or Service+HTTPRoute approach before writing Wave 1 tasks

---

## Sources

### Primary (HIGH confidence)

- `mcp-server/SKILL.md` — definitive 12-step agent workflow; tool reference table
- `k8s/agent-sandbox/openclaw-sandbox.yaml` — current Sandbox CR with all sidecar wiring
- `cluster_init/routes/*.yaml` — all four existing HTTPRoute examples; established pattern
- `cluster_init/ai-edge-gateway.yaml` — warp-edge-gateway definition (port 80, `from: All`)
- `cluster_init/app-store-cluster-init.yaml` — Envoy Gateway + GatewayClass install pattern
- `.planning/TOPOLOGY.md` — pod selector hash, container ports, GPU node details
- `scripts/validate-phase13.sh` — Phase 13 precondition checks; reusable as E2E pre-check
- `.planning/REQUIREMENTS.md` — E2E-01 through E2E-04 definitions
- `.planning/phases/14-end-to-end-validation/14-CONTEXT.md` — user decisions

### Secondary (MEDIUM confidence)

- Envoy Gateway v1.5.6 WebSocket handling: standard Gateway API HTTP protocol carries WebSocket upgrade transparently; no special annotation required for basic cases. Source: Envoy Gateway docs pattern (not directly verified via web search for this specific version).

### Tertiary (LOW confidence)

- OpenClaw LAN bind config option: unclear whether `controlUi.allowedOrigins` can be satisfied via environment variable rather than config file. Research did not find OpenClaw-specific documentation confirming this.

---

## Metadata

**Confidence breakdown:**
- Standard Stack: HIGH — all components confirmed deployed from TOPOLOGY.md and existing manifests
- Architecture: HIGH for HTTPRoute pattern (4 existing examples); MEDIUM for loopback resolution (open question)
- Pitfalls: HIGH — derived directly from Phase 12-13 decisions and TOPOLOGY.md known constraints
- E2E workflow: HIGH — SKILL.md is the authoritative source

**Research date:** 2026-03-25
**Valid until:** 2026-04-24 (stable stack; SKILL.md is the primary dependency)
