# Phase 13: Kubernetes Manifests and Sidecar Wiring - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Author the complete Kubernetes manifest set to wire the MCP server as a sidecar into the existing OpenClaw Sandbox CR pod. This includes RBAC (ServiceAccount, ClusterRole, ClusterRoleBinding), blueprint delivery via git-sync, runtime openclaw.json generation via init container, SKILL.md delivery via ConfigMap, and readiness probe wiring. Phase 12's TOPOLOGY.md is the authoritative reference for all integration points.

</domain>

<decisions>
## Implementation Decisions

### Blueprint Catalog Delivery
- Catalog is medium-sized (1-10MB) — too large for ConfigMap (1MB limit)
- Use git-sync as a continuous background sidecar container (not one-shot init)
- Pull from `main` branch of warp-blueprints repo
- MCP sidecar mounts the synced volume at `BLUEPRINTS_DIR`
- Matches the existing pattern used by the GUI deployment in the Helm chart

### openclaw.json Generation
- Init container with shell script writes openclaw.json from env vars at pod startup
- Uses minimal alpine/busybox image (not MCP server image) — fast pull, small footprint
- Writes to `/home/node/.openclaw/openclaw.json` on the shared emptyDir volume (per TOPOLOGY.md)
- openclaw.json includes both MCP server config (transport, url) AND SKILL.md path reference
- Never baked into the image (v3.0 locked decision)

### RBAC Scope and Boundaries
- New dedicated ServiceAccount: `weka-mcp-server-sa` (ROADMAP success criterion #5 — locked)
- ClusterRole (not namespaced Role) — required because inspect_cluster and inspect_weka read cluster-wide resources
- Cluster-wide write access for WekaAppStore CRs — matches how the operator watches all namespaces
- ClusterRoleBinding binds the ClusterRole to the ServiceAccount in the wekaappstore namespace

### SKILL.md and Tool Registration
- SKILL.md delivered via ConfigMap mounted as a file into the OpenClaw container
- Research should verify how OpenClaw discovers SKILL.md (explicit path in openclaw.json vs convention directory)
- Tool registration is automatic — OpenClaw reads openclaw.json at gateway start and auto-registers MCP servers
- MCP sidecar starts FIRST, passes /health readiness probe, then OpenClaw reads openclaw.json and discovers ready tools

### Startup Ordering
- Init containers run first: (1) openclaw.json generator writes config
- Then containers start: (1) MCP sidecar starts and becomes ready (/health returns 200), (2) git-sync starts pulling blueprints, (3) OpenClaw gateway starts and reads openclaw.json
- K8S-03: Readiness probe on MCP sidecar's /health endpoint prevents tool discovery before server is ready

### Claude's Discretion
- Exact ClusterRole rules (research should map each of the 8 tools to required API groups/resources/verbs)
- git-sync container image version and configuration env vars
- Init container shell script implementation details
- Whether to use K8s native sidecar containers (restartPolicy: Always on init containers) or regular containers for startup ordering

</decisions>

<specifics>
## Specific Ideas

- TOPOLOGY.md is the authoritative source for all integration points — Phase 13 modifies `k8s/agent-sandbox/openclaw-sandbox.yaml` Sandbox CR spec, NOT a Deployment spec
- MCP sidecar image: `wekachrisjen/weka-app-store-mcp` (Docker Hub, same CI pipeline)
- MCP sidecar env vars: `MCP_TRANSPORT=http`, `MCP_PORT=8080`, `BLUEPRINTS_DIR=<git-sync mount path>`
- OpenClaw gateway at localhost:18789 (--bind=loopback), MCP sidecar at localhost:8080
- STATE.md gate: "NemoClaw ConfigMap schema for mcpServers.url must be verified against the deployed version"
- Existing Secrets: `openclaw-token` and `nvidia-api-key` already in `wekaappstore` namespace (Phase 12)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `k8s/agent-sandbox/openclaw-sandbox.yaml`: Existing Sandbox CR — Phase 13 adds sidecar containers to `spec.podTemplate.spec`
- `k8s/agent-sandbox/openclaw-secrets.yaml`: Existing Secret templates — already applied
- `mcp-server/Dockerfile`: MCP server image with EXPOSE 8080
- `mcp-server/openclaw.json`: Reference for the JSON structure init container must generate
- `mcp-server/SKILL.md`: Content for the ConfigMap
- `weka-app-store-operator-chart/templates/clusterrole.yaml`: Existing RBAC pattern for reference
- `scripts/validate-topology.sh`: Smoke test — can be extended for Phase 13 sidecar validation

### Established Patterns
- Sandbox CR pattern: `agents.x-k8s.io/v1alpha1` with `spec.podTemplate.spec` for pod configuration
- Environment-variable-driven config: MCP_TRANSPORT, MCP_PORT, BLUEPRINTS_DIR
- Docker Hub image publishing: `wekachrisjen/weka-app-store-mcp`
- git-sync pattern in existing Helm chart (GUI deployment uses git-sync for blueprints)

### Integration Points
- Sandbox CR `spec.podTemplate.spec.containers[]` — add MCP sidecar and git-sync containers
- Sandbox CR `spec.podTemplate.spec.initContainers[]` — add openclaw.json generator
- Sandbox CR `spec.podTemplate.spec.volumes[]` — add blueprint volume, config emptyDir
- Sandbox CR `spec.podTemplate.spec.serviceAccountName` — set to `weka-mcp-server-sa`
- TOPOLOGY.md Phase 13 Integration Points section — authoritative reference for all wiring

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-kubernetes-manifests-and-sidecar-wiring*
*Context gathered: 2026-03-24*
