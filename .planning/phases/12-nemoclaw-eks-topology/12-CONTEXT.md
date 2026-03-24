# Phase 12: NemoClaw EKS Topology - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Deploy NemoClaw/OpenClaw to EKS using the experimental agent-sandbox CRD approach. Validate the topology (pod running, GPU allocated, loopback reachable) and document it as a structured reference for Phase 13 manifest authoring. No application code changes — this is infrastructure setup and validation.

</domain>

<decisions>
## Implementation Decisions

### NemoClaw Deployment Method
- Deploy using the experimental agent-sandbox CRD operator
- Phase 12 must install the agent-sandbox operator itself (not pre-installed)
- NGC API key and org are already configured — can pull images from nvcr.io
- Exact NemoClaw/OpenClaw container image and version to be determined by research (STATE.md gate: "NemoClaw container image name on NVIDIA NGC is a placeholder")

### GPU Node Group
- GPU node group already exists in the EKS cluster — no need to create one
- L40s GPUs (48GB VRAM) — 1 GPU available
- NVIDIA GPU Operator already installed and operational
- Research should verify L40s compatibility with the NemoClaw image requirements

### Egress and Network Policy
- No NetworkPolicies enforced in the cluster (standard EKS VPC CNI)
- Loopback to sidecar port works by default at the K8s networking level
- Unknown whether the agent-sandbox CRD itself restricts egress — research must inspect the CRD schema
- If CRD restricts loopback: Claude's discretion on whether to patch egress config or use ClusterIP Service fallback

### Namespace
- Deploy NemoClaw in the same namespace as the existing WEKA App Store components (not a dedicated namespace)

### Topology Validation
- Full smoke test script required (not just manual kubectl checks)
- Script validates: pod Running, GPU allocated, loopback to localhost:8080 reachable, NemoClaw API responsive
- Script lives in the repo at `scripts/` (versioned, reusable)
- Phase 12 also produces a structured topology reference doc (TOPOLOGY.md in .planning/) that Phase 13 consumes for manifest authoring

### Claude's Discretion
- Exact agent-sandbox CRD installation method (Helm chart vs raw manifests)
- Smoke test script implementation (bash vs Python)
- Topology doc format and level of detail
- Fallback strategy if agent-sandbox CRD restricts loopback egress

</decisions>

<specifics>
## Specific Ideas

- STATE.md gates: "NemoClaw EKS topology must be validated before Phase 13 manifests are written — wrong topology = full Phase 13 rewrite"
- STATE.md gates: "NemoClaw container image name on NVIDIA NGC is a placeholder; confirm before writing Deployment manifest"
- STATE.md gates: "NemoClaw ConfigMap schema for mcpServers.url must be verified against the deployed version"
- L40s with 48GB VRAM — research should check if this is sufficient for the chosen NIM model profile
- Single GPU constraint means research should find a model profile that fits in 1x L40s

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/server.py`: Already supports HTTP transport on port 8080 — sidecar pattern ready (Phase 11)
- `mcp-server/openclaw.json`: Already configured for `transport: "streamable-http"` and `url: "http://localhost:8080/mcp"` (Phase 11)
- `mcp-server/Dockerfile`: Already has `EXPOSE 8080` (Phase 11)
- `weka-app-store-operator-chart/templates/clusterrole.yaml`: Existing RBAC patterns for reference

### Established Patterns
- Environment-variable-driven configuration (MCP_TRANSPORT, MCP_PORT, BLUEPRINTS_DIR)
- Helm chart packaging in `weka-app-store-operator-chart/`
- Container images published to `wekachrisjen/` on Docker Hub

### Integration Points
- Phase 13 will consume the topology reference doc to write Deployment, ServiceAccount, RBAC, and volume manifests
- The smoke test script validates the same localhost:8080 endpoint that Phase 11's health check uses
- Agent-sandbox CRD namespace must match where the MCP sidecar will be deployed

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-nemoclaw-eks-topology*
*Context gathered: 2026-03-24*
