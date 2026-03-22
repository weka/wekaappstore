# Phase 9: Deployment and Registration - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

The MCP server ships as a container image and OpenClaw/NemoClaw operators can register and invoke it using documented, repeatable configuration steps. Includes Dockerfile, environment variable configuration, registration documentation, CI workflow, and image distribution to wekachrisjen Docker Hub.

</domain>

<decisions>
## Implementation Decisions

### Dockerfile Strategy
- Base image: `python:3.10-slim`
- Copy both `mcp-server/` and `app-store-gui/webapp/` into the image, set PYTHONPATH — self-contained, no shared volumes needed for code
- Blueprints are NOT baked into the image — volume mount at runtime from host/git-sync sidecar at `/app/manifests/manifest`
- Import check entrypoint: run `python -c "import server"` as sanity check before starting, catches missing dependencies at container start
- CMD: `python -m server` for stdio transport

### Environment Config Surface
- K8s auth: in-cluster service account by default, `KUBECONFIG` env var override for local dev/testing outside cluster
- `BLUEPRINTS_DIR` required — validated at startup with clear error if missing
- `LOG_LEVEL` optional with INFO default — all logging to stderr (stdio transport requirement)
- `KUBERNETES_AUTH_MODE` optional — existing pattern from webapp
- `WEKA_ENDPOINT` optional — override for direct WEKA API access as alternative to K8s CRD discovery (future-proofing for non-K8s WEKA)
- Required env vars validated at startup (fail fast); optional vars checked lazily when tools are called

### Registration Documentation
- Single `README.md` in `mcp-server/` — sections: quick start, Dockerfile build, env vars reference, OpenClaw registration, NemoClaw registration, troubleshooting
- Full copy-paste command blocks for every step (build, tag, push, register) — operators follow end-to-end without guessing
- NemoClaw registration section: placeholder with known patterns (stdio transport, env vars, SKILL.md path) plus clear `TODO: Update when NemoClaw alpha schema is published` marker
- "Verify it works" section with echo pipe smoke test: pipe a `tools/list` JSON-RPC request to container stdin, verify 8 tools come back

### Image Distribution
- Docker Hub registry: `wekachrisjen/weka-app-store-mcp` (WEKA corporate account)
- Tagging: semver + latest (`wekachrisjen/weka-app-store-mcp:v2.0.0` and `:latest`)
- Full CI workflow with tests: `.github/workflows/mcp-server.yml`
- CI triggers: run tests on every PR touching `mcp-server/`; build and push image only on version tag push (`v*`)

### Claude's Discretion
- Exact Dockerfile layer ordering for cache optimization
- README section ordering and prose style
- GitHub Actions workflow details (runner OS, caching strategy)
- Whether to add a .dockerignore file
- Smoke test exact JSON-RPC request format

</decisions>

<specifics>
## Specific Ideas

- Docker Hub account is `wekachrisjen` inside a WEKA corporate account — use for all images
- openclaw.json already exists at mcp-server/openclaw.json with tool descriptions, startup command, env vars, container reference, and SKILL.md path (created in Phase 8)
- The existing webapp uses `BLUEPRINTS_DIR` env var pointing to `/app/manifests/manifest` — same pattern for MCP server container
- NemoClaw alpha config schema not yet published as of 2026-03-20 (STATE.md blocker)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/openclaw.json`: Already contains tool list, startup command, env vars, container field — registration docs can reference directly
- `mcp-server/generate_openclaw_config.py`: Auto-generates openclaw.json from server.py registrations — drift detection already tested
- `mcp-server/requirements.txt`: `mcp[cli]>=1.26.0`, `kubernetes>=27.0.0`, `PyYAML>=6.0.1`, `pytest>=8.0.0`
- `mcp-server/server.py`: Entry point, registers all 8 tools via `register_*(mcp)` pattern

### Established Patterns
- `PYTHONPATH=.:../app-store-gui` used in all test commands — Dockerfile needs equivalent
- `BLUEPRINTS_DIR` env var pattern from FastAPI webapp
- `KUBERNETES_AUTH_MODE` env var pattern from existing codebase
- All logging to stderr via Python logging (MCPS-11 requirement already implemented)

### Integration Points
- Container needs K8s service account for cluster API access (inspect tools, apply tool, CRD schema)
- Blueprint directory mounted as volume from git-sync sidecar or host path
- OpenClaw registers MCP server via openclaw.json config pointing to the container
- Stdio transport: OpenClaw spawns container process, communicates via stdin/stdout

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-deployment-and-registration*
*Context gathered: 2026-03-23*
