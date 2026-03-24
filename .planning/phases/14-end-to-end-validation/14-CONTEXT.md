# Phase 14: End-to-End Validation - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate the full SKILL.md tool chain through live agent chat against a real EKS cluster with live WEKA storage. The agent must inspect cluster resources, browse blueprints, generate/validate/apply a WekaAppStore CR for the OSS Rag blueprint, and report full deployment success. Evidence is captured as chat transcripts + kubectl outputs. This phase also sets up Envoy Gateway API access to the OpenClaw Web UI.

</domain>

<decisions>
## Implementation Decisions

### Target Blueprint
- OSS Rag blueprint — the specific blueprint for the happy-path E2E test
- Requires WEKA storage (wekafs storage class) — WEKA CSI is live and configured
- Requires GPU resources — cluster has GPU nodes available
- Enough cluster capacity for a full deploy (not a dry-run)
- Leave the deployed blueprint running after test — no cleanup

### Agent Access
- Chat via OpenClaw Web UI in browser
- Set up an Envoy Gateway API HTTPRoute to expose the OpenClaw UI
- Hostname: `openclaw.example.com`
- Uses Kubernetes Gateway API with HTTPRoute resources (not traditional Ingress)
- OpenClaw gateway is at localhost:18789 in the pod — HTTPRoute backends to the pod/service

### WEKA Storage
- WEKA CSI is live and working in the EKS cluster
- wekafs storage class exists, PVCs can be provisioned
- inspect_weka will return real data (free capacity, filesystems, mounts)
- Enough free WEKA capacity to actually deploy OSS Rag

### Evidence and Pass/Fail Criteria
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

</decisions>

<specifics>
## Specific Ideas

- The SKILL.md 12-step workflow defines the exact sequence the agent should follow: inspect_cluster → inspect_weka → list_blueprints → get_blueprint → get_crd_schema → generate YAML → validate_yaml → (retry loop) → confirm with user → apply → deployment_status
- The apply tool requires explicit user confirmation in the chat — this is part of the E2E test
- OpenClaw gateway uses --bind=loopback at localhost:18789 — Envoy HTTPRoute must target the correct backend
- The agent-sandbox Sandbox CR currently has no Service — may need to create one for Envoy to route to, or use pod IP directly

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/SKILL.md`: Defines the 12-step agent workflow — the test script for the E2E conversation
- `scripts/validate-topology.sh`: Phase 12 smoke test — can verify pod health before E2E
- `scripts/validate-phase13.sh`: Phase 13 live validation — confirms sidecar is ready before E2E
- `cluster_init/`: Contains existing Gateway API route examples that can be referenced for the OpenClaw HTTPRoute
- `k8s/agent-sandbox/openclaw-sandbox.yaml`: Current Sandbox CR — may need a Service for Envoy routing

### Established Patterns
- Gateway API HTTPRoute used elsewhere in the cluster (per cluster_init/ manifests)
- kubectl-based evidence capture (used in Phase 12 and 13 validation scripts)
- Chat evidence format TBD by research

### Integration Points
- Envoy Gateway must route to OpenClaw gateway port 18789 (WebSocket + HTTP)
- OpenClaw Web UI served by the gateway process
- MCP tools accessible via localhost:8080 from within the pod (Phase 13 confirmed)
- WekaAppStore CRs created in target namespace — operator watches and reconciles

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-end-to-end-validation*
*Context gathered: 2026-03-25*
