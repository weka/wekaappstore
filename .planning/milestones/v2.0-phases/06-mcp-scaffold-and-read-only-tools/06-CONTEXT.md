# Phase 6: MCP Scaffold and Read-Only Tools - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a runnable MCP server using the official `mcp[cli]` SDK with FastMCP that exposes 5 read-only tools (`inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`) with flat, agent-facing response schemas. All tools are read-only and testable with mocked K8s/WEKA backends.

</domain>

<decisions>
## Implementation Decisions

### Tool Response Shape
- Single flat JSON object per tool response — no nested domain/status/freshness wrappers from v1.0
- Top-level `captured_at` timestamp on every response
- Aggregated cluster summary for GPU data (gpu_total, gpu_models, gpu_memory_total_gib) — not per-node breakdown
- Separate `warnings` array at top level for blocker/problem information (e.g. "GPU operator not detected")
- Data fields carry facts (gpu_operator_installed: false); warnings array carries things that need agent attention
- `inspect_cluster` and `inspect_weka` are separate tools — agent calls what it needs

### Blueprint Catalog Source
- Blueprints are sourced from the `weka/warp-blueprints` GitHub repo, synced into the container at `/app/manifests/manifest` at runtime
- MCP server scans this directory to discover blueprints — not a hardcoded list
- `BLUEPRINTS_DIR` env var configurable, defaults to `/app/manifests/manifest` (same pattern as the FastAPI app)
- `list_blueprints` returns full metadata per blueprint: name, description, category, Helm chart ref, default values, resource requirements (GPU/CPU/RAM/storage minimums), prerequisites, supported configurations
- `get_blueprint` returns complete blueprint detail including Helm values schema and defaults
- Supplemental metadata (resource requirements, descriptions) can be carried in a YAML registry file if not extractable from the manifests themselves

### MCP Server Location
- Separate container — own Dockerfile, own image, deployed as sidecar or standalone
- Own `requirements.txt` adding `mcp[cli]` and referencing shared deps independently
- Claude's discretion on whether code lives at `mcp-server/` root or inside `app-store-gui/`
- Imports reusable code from `inspection/cluster.py`, `inspection/weka.py`, `planning/apply_gateway.py`, `planning/validator.py` via shared PYTHONPATH

### CRD Schema Delivery
- `get_crd_schema` reads the live CRD from the running cluster via apiextensions API — always current, no Helm template noise
- Response includes 1-2 example `WekaAppStore` YAML documents picked from the blueprints directory (`/app/manifests/manifest`) — real manifests, not hand-written
- Examples help the agent pattern-match valid YAML structure alongside the schema

### Claude's Discretion
- Exact MCP server directory placement (repo root vs inside app-store-gui/)
- Internal module structure of the MCP server
- How to extract blueprint metadata from manifest files (parsing strategy)
- Tool description wording (must include when/why/sequencing guidance per success criteria)

</decisions>

<specifics>
## Specific Ideas

- Blueprints repo: https://github.com/weka/warp-blueprints/tree/main/manifests — this is the authoritative source
- Runtime path: `/app/manifests/manifest` — where git-sync puts them in the container
- The `BLUEPRINTS_DIR` env var is already used by the FastAPI app for the same purpose
- The MCP server should feel like a toolbox — each tool does one thing, returns a clean flat response

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `inspection/cluster.py`: `collect_cluster_inspection()` — injectable K8s client seams, returns per-domain dicts. Wrap and flatten for MCP tool.
- `inspection/weka.py`: `collect_weka_inspection()` — reads WekaCluster CRs, returns filesystem inventory and capacity. Wrap and flatten.
- `planning/apply_gateway.py`: Apply path with namespace handling. Reused in Phase 7.
- `planning/validator.py`: CRD validation rules. Component-level helpers are the safe reuse target (not `validate_structured_plan()`).

### Established Patterns
- K8s client injection: `collect_cluster_inspection(core_api=..., storage_api=...)` — no import-time side effects, safe to call from MCP server process
- Env var config: `BLUEPRINTS_DIR`, `KUBERNETES_AUTH_MODE` — existing patterns for runtime configuration
- The FastAPI app already scans the blueprints directory at runtime for UI rendering

### Integration Points
- The MCP server shares Python code with `app-store-gui/webapp/` via PYTHONPATH or package install
- Blueprint directory is a shared volume between webapp container and MCP server container
- K8s API access via service account (same cluster)
- CRD read via apiextensions API (already used in `cluster.py` for checking CRD installation)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-mcp-scaffold-and-read-only-tools*
*Context gathered: 2026-03-20*
