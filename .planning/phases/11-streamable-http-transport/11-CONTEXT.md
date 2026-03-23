# Phase 11: Streamable HTTP Transport - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Add Streamable HTTP transport to the existing MCP server so it can run as a K8s sidecar. The server becomes dual-mode: stdio (default, for CI/local dev) and HTTP (for sidecar deployment). All 8 tools are transport-agnostic — no tool code changes. This is purely a server startup and configuration phase.

</domain>

<decisions>
## Implementation Decisions

### Port and Binding
- Default port: 8080, configurable via `MCP_PORT` env var
- Bind address: `0.0.0.0` (all interfaces — required for pod-local networking)
- MCP path: always `/mcp` (MCP spec convention, not configurable)
- Transport selected via `MCP_TRANSPORT` env var: `stdio` (default) or `http`

### Health Endpoint
- `/health` returns HTTP 200 when FastMCP is initialized and tools are registered
- Does NOT check K8s connectivity or blueprint directory — server-ready only
- Response body: `{"status": "ok", "tools": 8, "transport": "http"}`
- HTTP-only — no health endpoint in stdio mode

### Dockerfile and Image
- Single image supports both transports — `MCP_TRANSPORT` env var selects at runtime
- Add `EXPOSE 8080` to Dockerfile (documents intent, no runtime effect)
- Same CI pipeline, same tags, same `wekachrisjen/weka-app-store-mcp` image

### openclaw.json Update
- Single openclaw.json updated to HTTP config: `"transport": "streamable-http"`, `"url": "http://localhost:8080/mcp"`
- Stdio startup block removed (stdio is for local dev, not OpenClaw registration)
- `generate_openclaw_config.py` updated to produce HTTP variant — drift detection test stays valid

### Claude's Discretion
- Exact FastMCP `mcp.run()` arguments for HTTP transport
- How to structure the transport branch in server.py `__main__`
- Test strategy for HTTP transport (integration test with actual HTTP client vs mocking)
- Whether to add `MCP_TRANSPORT` and `MCP_PORT` to config.py or read directly in server.py

</decisions>

<specifics>
## Specific Ideas

- Research confirmed: `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)` is the FastMCP API
- Research warned: FastMCP cannot be mounted on existing FastAPI as sub-app (Starlette lifespan issue) — must run as separate process
- Research warned: design for stateless mode — MCP Python SDK has active issues with session ID forwarding
- The `mcp[cli]>=1.26.0` already in requirements.txt includes Streamable HTTP support — no new deps

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/server.py`: Current entry point with `mcp.run()` call — add transport branch here
- `mcp-server/config.py`: Env var pattern established — add `MCP_TRANSPORT` and `MCP_PORT` here
- `mcp-server/generate_openclaw_config.py`: Generates openclaw.json from tool registrations — update transport section
- `mcp-server/tests/test_openclaw_config.py`: Drift detection test — will catch if openclaw.json and generator diverge

### Established Patterns
- `config.py` reads env vars at module import time, `validate_required()` called from `__main__` only
- `logging.basicConfig()` called before FastMCP import (critical ordering)
- All tools registered via `register_*(mcp)` pattern — transport-agnostic

### Integration Points
- `server.py __main__` block is the only place that calls `mcp.run()` — transport change is localized here
- Dockerfile CMD `["python", "-m", "server"]` — unchanged, transport selected by env var
- GitHub Actions CI runs tests with default stdio transport — unaffected

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-streamable-http-transport*
*Context gathered: 2026-03-23*
