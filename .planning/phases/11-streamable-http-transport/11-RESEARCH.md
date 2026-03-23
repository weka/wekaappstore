# Phase 11: Streamable HTTP Transport - Research

**Researched:** 2026-03-23
**Domain:** MCP Python SDK (FastMCP) Streamable HTTP transport, uvicorn, Starlette
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Port and Binding:** Default port 8080, configurable via `MCP_PORT` env var. Bind address `0.0.0.0`. MCP path always `/mcp`. Transport selected via `MCP_TRANSPORT` env var: `stdio` (default) or `http`.
- **Health Endpoint:** `/health` returns HTTP 200 when FastMCP is initialized and tools are registered. Does NOT check K8s connectivity or blueprint directory. Response body: `{"status": "ok", "tools": 8, "transport": "http"}`. HTTP-only — no health endpoint in stdio mode.
- **Dockerfile:** Single image supports both transports — `MCP_TRANSPORT` env var selects at runtime. Add `EXPOSE 8080`. Same CI pipeline, same tags, same `wekachrisjen/weka-app-store-mcp` image.
- **openclaw.json:** Single openclaw.json updated to HTTP config: `"transport": "streamable-http"`, `"url": "http://localhost:8080/mcp"`. Stdio startup block removed. `generate_openclaw_config.py` updated to produce HTTP variant — drift detection test stays valid.

### Claude's Discretion

- Exact FastMCP `mcp.run()` arguments for HTTP transport
- How to structure the transport branch in server.py `__main__`
- Test strategy for HTTP transport (integration test with actual HTTP client vs mocking)
- Whether to add `MCP_TRANSPORT` and `MCP_PORT` to config.py or read directly in server.py

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| XPORT-01 | MCP server supports Streamable HTTP transport on a configurable port alongside existing stdio | FastMCP 1.26.0 `run(transport="streamable-http")` confirmed; host/port set at FastMCP constructor time |
| XPORT-02 | `MCP_TRANSPORT` env var selects transport mode (`stdio` default, `http` for sidecar deployment) | Transport branch in `__main__` block reads `MCP_TRANSPORT`; `mcp` object constructed with conditional args |
| XPORT-03 | Health endpoint (`/health`) returns 200 when server is ready for tool calls | `FastMCP.custom_route("/health", methods=["GET"])` decorator confirmed working; route appears in Starlette app routes |
| XPORT-04 | HTTP transport operates in stateless mode (no session ID dependency) | `stateless_http=True` constructor parameter confirmed; set in `StreamableHTTPSessionManager` |
</phase_requirements>

---

## Summary

Phase 11 adds Streamable HTTP transport to the MCP server without touching any tool code. The MCP Python SDK version `1.26.0` (already installed) provides full Streamable HTTP support via `FastMCP.run(transport="streamable-http")`. Transport mode, host, port, and stateless behavior are all configured at `FastMCP` **constructor time** — not at `run()` call time — which drives the key architectural decision: the `mcp` object in `server.py` must be constructed conditionally based on `MCP_TRANSPORT`.

The health endpoint uses `FastMCP.custom_route("/health", methods=["GET"])` — a built-in decorator that adds arbitrary Starlette routes to the HTTP app without auth requirements. This was verified: `/health` and `/mcp` both appear in the generated Starlette routes. The SDK uses `uvicorn` for HTTP serving; no additional dependencies are needed.

Existing tests (103 passing, confirmed baseline) are unaffected by default. One test (`test_openclaw_json_has_stdio_transport`) explicitly asserts `transport == "stdio"` and must be updated to assert `"streamable-http"`. The drift detection test (`test_openclaw_json_matches_generation`) stays valid because it compares tool names and descriptions only, not transport metadata.

**Primary recommendation:** Construct `mcp = FastMCP(name, host=..., port=..., stateless_http=True)` conditionally in `server.py` based on `MCP_TRANSPORT`, register all tools on the resulting `mcp` object (unchanged), then call `mcp.run(transport=transport)` in `__main__`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[cli]` | 1.26.0 (installed) | FastMCP server, Streamable HTTP transport, uvicorn runner | Already in requirements.txt; 1.26.0 adds stateless HTTP support |
| `starlette` | 0.52.1 (transitive) | ASGI framework backing FastMCP HTTP app, custom routes | Installed via mcp SDK; `Request`/`JSONResponse` used for health endpoint |
| `uvicorn` | 0.38.0 (transitive) | ASGI server for HTTP mode | Used internally by `FastMCP.run_streamable_http_async()` |

### No New Dependencies
All required packages arrive transitively with `mcp[cli]>=1.26.0`. The requirements.txt line is unchanged.

---

## Architecture Patterns

### Recommended Project Structure (changes only)

```
mcp-server/
├── server.py               # Add transport branch in __main__; conditionally build mcp
├── config.py               # Add MCP_TRANSPORT and MCP_PORT constants
├── generate_openclaw_config.py  # Update to emit streamable-http transport block
├── openclaw.json           # Regenerated: transport=streamable-http, url=http://localhost:8080/mcp
├── Dockerfile              # Add EXPOSE 8080
└── tests/
    ├── test_server.py      # Unchanged (tests mcp object and tool registration — not transport)
    ├── test_config.py      # Add MCP_TRANSPORT and MCP_PORT config tests
    ├── test_openclaw_config.py  # Update test_openclaw_json_has_stdio_transport assertion
    └── test_http_transport.py  # New: health endpoint and HTTP mode config tests
```

### Pattern 1: Conditional mcp Object Construction

**What:** FastMCP host, port, and stateless_http are constructor-time settings; they cannot be changed after construction. The `mcp` object must be built with the correct settings before tool registration.

**When to use:** Always — this is the only correct approach for dual-mode.

**Key insight:** `mcp.run(transport="streamable-http")` uses `mcp.settings.host` and `mcp.settings.port` set at construction. Calling `mcp.run(transport="streamable-http")` on an `mcp` built with defaults will bind to `127.0.0.1:8000`, not `0.0.0.0:8080`.

**Verified FastMCP constructor signature (from source):**
```python
# Source: inspected from mcp 1.26.0 FastMCP.__init__
FastMCP(
    name: str,
    host: str = "127.0.0.1",   # MUST override for sidecar
    port: int = 8000,           # MUST override for sidecar
    stateless_http: bool = False,  # MUST set True for XPORT-04
)
```

**Example pattern for server.py:**
```python
# Source: verified against FastMCP 1.26.0 constructor and run() source
import config

_transport = config.MCP_TRANSPORT  # "stdio" or "http"
_port = config.MCP_PORT            # int, default 8080

if _transport == "http":
    mcp = FastMCP(
        "weka-app-store-mcp",
        host="0.0.0.0",
        port=_port,
        stateless_http=True,
    )
else:
    mcp = FastMCP("weka-app-store-mcp")

# Tool registration is transport-agnostic — unchanged
register_inspect_cluster(mcp)
register_inspect_weka(mcp)
# ... etc.

if __name__ == "__main__":
    from config import validate_required
    validate_required()
    mcp.run(transport=_transport if _transport == "stdio" else "streamable-http")
```

### Pattern 2: Health Endpoint via custom_route

**What:** `FastMCP.custom_route` is a built-in decorator that appends a Starlette `Route` to `_custom_starlette_routes`. These routes are included in the Starlette app created by `streamable_http_app()` and served alongside `/mcp`.

**Verified:** After calling `@mcp.custom_route("/health", methods=["GET"])`, the route appears in `mcp.streamable_http_app().routes` as a real Starlette `Route`. Confirmed in testing with mcp 1.26.0.

**Important:** `custom_route` must be called on the `mcp` instance BEFORE `mcp.run()` is called (i.e., at module or `__main__` time). Since the health endpoint is HTTP-only per the locked decisions, register it inside the `if _transport == "http":` branch.

```python
# Source: FastMCP.custom_route docstring + verified with mcp 1.26.0
from starlette.requests import Request
from starlette.responses import JSONResponse

if _transport == "http":
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "tools": 8, "transport": "http"})
```

### Pattern 3: config.py Additions

**What:** `MCP_TRANSPORT` and `MCP_PORT` follow the established `config.py` pattern (read at import time, no classes).

```python
# Consistent with existing config.py pattern
MCP_TRANSPORT: str = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))
```

**Note:** `MCP_PORT` must be cast to `int` because `os.environ.get()` returns `str`. The `FastMCP` constructor expects `port: int`.

### Pattern 4: openclaw.json HTTP Transport Block

**What:** The `transport` field changes from `"stdio"` to `"streamable-http"`. The `startup` block (subprocess spawn) is replaced with `url`. The `env` section stays; env vars are runtime config, not spawn config.

**Target structure:**
```json
{
  "_comment": "Best-effort format -- may need revision when NemoClaw alpha schema is published.",
  "name": "weka-app-store-mcp",
  "transport": "streamable-http",
  "url": "http://localhost:8080/mcp",
  "env": {
    "required": ["BLUEPRINTS_DIR"],
    "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "KUBECONFIG", "MCP_TRANSPORT", "MCP_PORT"]
  },
  "container": "weka-app-store-mcp:latest",
  "skill": "mcp-server/SKILL.md",
  "tools": [...]
}
```

The `generate_openclaw_config.py` `build_openclaw_config()` function must be updated to emit this shape. The drift detection test (`test_openclaw_json_matches_generation`) compares tool names and descriptions only — it will remain valid after this change.

### Anti-Patterns to Avoid

- **Creating mcp with default host/port then running HTTP transport:** `FastMCP("name")` defaults to `host="127.0.0.1"`, `port=8000`. Running `mcp.run(transport="streamable-http")` on this will NOT bind to `0.0.0.0:8080`. Always construct with explicit `host`/`port` for HTTP mode.
- **Registering health route unconditionally:** The health endpoint only applies in HTTP mode. Registering it in stdio mode would silently succeed but never be reachable and creates confusion.
- **Calling validate_required() before mcp construction:** Current server.py calls `validate_required()` in `__main__` only, which is correct. Do not move it to module level — tests import the module without `BLUEPRINTS_DIR` set.
- **Mounting FastMCP on an existing FastAPI app:** CONTEXT.md confirmed this fails due to Starlette lifespan conflicts. Do not attempt. FastMCP must run standalone via `mcp.run()`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP/ASGI server | Custom uvicorn setup, custom Starlette app | `FastMCP.run(transport="streamable-http")` | FastMCP manages uvicorn Config + Server + serve() lifecycle internally |
| Health route | Separate Flask/FastAPI process, manual socket listener | `FastMCP.custom_route("/health", methods=["GET"])` | Built-in; adds route to same Starlette app; no extra process, no port conflicts |
| Stateless session handling | Custom session ID forwarding logic | `stateless_http=True` in FastMCP constructor | SDK handles session management; active issues with session ID forwarding make stateless the correct choice |
| Transport selection logic | argparse, click, complex branching | Single `if MCP_TRANSPORT == "http":` branch in server.py | Phase constraint: all 8 tools are transport-agnostic; only startup differs |

---

## Common Pitfalls

### Pitfall 1: FastMCP Defaults to Localhost Binding

**What goes wrong:** `FastMCP("name")` defaults to `host="127.0.0.1"`. Running in HTTP mode with this default means the server is only reachable from within the same container, not from the OpenClaw process on the pod network.

**Why it happens:** FastMCP applies DNS rebinding protection automatically when `host` is a localhost variant. When `host="0.0.0.0"`, this protection is not applied (confirmed: `transport_security` is `None`), which is correct for sidecar deployment.

**How to avoid:** Always construct `FastMCP(..., host="0.0.0.0", port=8080)` in the `http` branch. Verified: `mcp.settings.host == "0.0.0.0"` and `mcp.settings.transport_security is None` after construction.

**Warning signs:** `curl localhost:8080/health` returns connection refused when `MCP_TRANSPORT=http`.

### Pitfall 2: test_openclaw_json_has_stdio_transport Will Fail After openclaw.json Update

**What goes wrong:** `test_openclaw_config.py` line 89-94 asserts `transport == "stdio"`. After updating openclaw.json to `"streamable-http"`, this test will fail.

**Why it happens:** The test was written to validate the original stdio transport config. It now needs to validate the new HTTP transport config.

**How to avoid:** Update the test assertion to `assert transport == "streamable-http"` and add a companion assertion that `"url"` is present (or that `"startup"` block is absent).

**Warning signs:** Test suite red after running `python generate_openclaw_config.py`.

### Pitfall 3: MCP_PORT Must Be Cast to int

**What goes wrong:** `os.environ.get("MCP_PORT", "8080")` returns a string. Passing a string to `FastMCP(..., port="8080")` raises a Pydantic validation error because `port: int` in FastMCP's Settings model.

**How to avoid:** `MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))` in config.py.

**Warning signs:** `pydantic.ValidationError: port: value is not a valid integer` at startup.

### Pitfall 4: stateless_http=False (Default) May Cause Issues With OpenClaw

**What goes wrong:** Without `stateless_http=True`, the MCP SDK creates sessions keyed by session ID. If the OpenClaw client doesn't forward `Mcp-Session-Id` headers correctly (documented as an active SDK issue in CONTEXT.md), tool calls will fail or create ghost sessions.

**How to avoid:** Always set `stateless_http=True` for the HTTP branch. XPORT-04 requires this.

**Warning signs:** Tool calls return 4xx errors or session-not-found errors in HTTP mode.

### Pitfall 5: logging.basicConfig Must Be Called Before FastMCP Construction

**What goes wrong:** The existing `server.py` calls `logging.basicConfig()` before `from mcp.server.fastmcp import FastMCP`. This ordering must be preserved in the refactored version — if `mcp = FastMCP(...)` is moved to a location before `logging.basicConfig()`, the SDK hijacks log handlers.

**Why it happens:** FastMCP imports configure the MCP SDK logging at import time (GitHub issue python-sdk#1656).

**How to avoid:** The import order in `server.py` is already correct. When restructuring the file, ensure `mcp = FastMCP(...)` stays after `logging.basicConfig()` — it currently does.

---

## Code Examples

### Full server.py __main__ Pattern

```python
# Source: FastMCP 1.26.0 verified constructor + run() signatures
if __name__ == "__main__":
    from config import validate_required, MCP_TRANSPORT, MCP_PORT
    validate_required()

    _transport_arg = "streamable-http" if MCP_TRANSPORT == "http" else "stdio"
    mcp.run(transport=_transport_arg)
```

Note: `mcp` is constructed at module level (before `__main__`) because tests import `server` to access `server.mcp`. The conditional construction must happen at module level.

### Revised Module-Level mcp Construction

```python
# Source: verified against FastMCP 1.26.0
from starlette.requests import Request
from starlette.responses import JSONResponse

_transport = config.MCP_TRANSPORT  # read from config after logging setup
_port = config.MCP_PORT

if _transport == "http":
    mcp = FastMCP("weka-app-store-mcp", host="0.0.0.0", port=_port, stateless_http=True)

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:  # noqa: RUF029
        return JSONResponse({"status": "ok", "tools": 8, "transport": "http"})
else:
    mcp = FastMCP("weka-app-store-mcp")

# Tool registration — unchanged and transport-agnostic
register_inspect_cluster(mcp)
# ... rest of registrations unchanged
```

### config.py Additions

```python
# Transport mode — 'stdio' (default, CI-safe) or 'http' (sidecar deployment)
MCP_TRANSPORT: str = os.environ.get("MCP_TRANSPORT", "stdio")

# HTTP listening port — only relevant when MCP_TRANSPORT=http
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))
```

### Updated generate_openclaw_config.py build_openclaw_config()

```python
# Transport block differs between stdio and http
# Phase 11 target: always emit streamable-http (stdio is local-dev only)
return {
    "_comment": "Best-effort format -- may need revision when NemoClaw alpha schema is published.",
    "name": "weka-app-store-mcp",
    "description": "...",
    "transport": "streamable-http",
    "url": "http://localhost:8080/mcp",
    "env": {
        "required": ["BLUEPRINTS_DIR"],
        "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "KUBECONFIG", "MCP_TRANSPORT", "MCP_PORT"],
    },
    "container": "weka-app-store-mcp:latest",
    "skill": "mcp-server/SKILL.md",
    "tools": tools,
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SSE transport (`/sse` + `/messages/`) | Streamable HTTP (`/mcp`) | MCP spec 2025-03-26 | SSE deprecated; Streamable HTTP is the standard for sidecar deployment |
| `transport: "stdio"` with subprocess spawn | `transport: "streamable-http"` with `url:` | Phase 11 | OpenClaw connects via HTTP, not subprocess; no spawn overhead |
| `stateless_http=False` (default) | `stateless_http=True` | Required for XPORT-04 | Avoids session ID forwarding issues documented in MCP Python SDK |

**Deprecated/outdated:**
- SSE transport: The `run(transport="sse")` path still exists in FastMCP but is deprecated per MCP spec. REQUIREMENTS.md explicitly calls it out of scope. Do not use.
- `transport: "stdio"` in openclaw.json: Replaced by `transport: "streamable-http"` + `url` in this phase.

---

## Open Questions

1. **How will tests import `server.mcp` when `MCP_TRANSPORT` is unset in CI?**
   - What we know: Tests import `server` without setting `MCP_TRANSPORT`. With the proposed changes, unset `MCP_TRANSPORT` defaults to `"stdio"` (via `config.MCP_TRANSPORT` default), so `mcp = FastMCP("weka-app-store-mcp")` — same as today.
   - What's unclear: Whether any test will need `MCP_TRANSPORT=http` to exercise the HTTP branch.
   - Recommendation: Tests for HTTP mode should use `monkeypatch.setenv("MCP_TRANSPORT", "http")` with a `_reload_config()` pattern (matching test_config.py) to exercise the HTTP branch without affecting the module-level `mcp` object tests. New `test_http_transport.py` tests can directly construct a FastMCP instance in HTTP mode for unit testing, independent of server module state.

2. **Should `test_http_transport.py` use a live HTTP server or mock?**
   - What we know: `FastMCP.run_streamable_http_async()` spawns a real uvicorn server. Standing up a live server in pytest requires `pytest-anyio` or `anyio.from_thread.run_sync`. The health endpoint returns a simple JSONResponse that doesn't require server state.
   - Recommendation: For health endpoint tests, construct the Starlette app directly via `mcp.streamable_http_app()` and use Starlette's `TestClient`. This avoids real network binding in CI. For `MCP_TRANSPORT` config tests, use `monkeypatch` with `_reload_config()`. This keeps tests under 30 seconds and avoids port conflicts.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0.0+ |
| Config file | none (pytest discovers from tests/) |
| Quick run command | `cd mcp-server && PYTHONPATH=.:../app-store-gui BLUEPRINTS_DIR=/tmp python -m pytest tests/ -q --tb=short` |
| Full suite command | `cd mcp-server && PYTHONPATH=.:../app-store-gui BLUEPRINTS_DIR=/tmp python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| XPORT-01 | FastMCP constructed in HTTP mode binds `0.0.0.0:8080` | unit | `pytest tests/test_http_transport.py::test_http_mcp_host_and_port -x` | ❌ Wave 0 |
| XPORT-01 | `mcp.run(transport="streamable-http")` path is reached in HTTP mode | unit | `pytest tests/test_http_transport.py::test_transport_arg_selection -x` | ❌ Wave 0 |
| XPORT-02 | `MCP_TRANSPORT=stdio` starts server as before; all 103 existing tests pass | regression | `pytest tests/ -q` | ✅ existing |
| XPORT-02 | `MCP_TRANSPORT=http` config value is read from env var | unit | `pytest tests/test_config.py::test_mcp_transport_default -x` | ❌ Wave 0 |
| XPORT-02 | `MCP_PORT` config value is read as int from env var | unit | `pytest tests/test_config.py::test_mcp_port_default -x` | ❌ Wave 0 |
| XPORT-03 | `/health` returns 200 with `{"status": "ok", "tools": 8, "transport": "http"}` | unit | `pytest tests/test_http_transport.py::test_health_endpoint_returns_200 -x` | ❌ Wave 0 |
| XPORT-03 | `/health` only present in HTTP mode (not stdio) | unit | `pytest tests/test_http_transport.py::test_health_only_in_http_mode -x` | ❌ Wave 0 |
| XPORT-04 | FastMCP constructed with `stateless_http=True` in HTTP mode | unit | `pytest tests/test_http_transport.py::test_stateless_mode_enabled -x` | ❌ Wave 0 |

**Updated existing test:**
| Test | Change Required | File |
|------|-----------------|------|
| `test_openclaw_json_has_stdio_transport` | Update assertion: `transport == "streamable-http"` | `test_openclaw_config.py` |
| `test_startup_env_includes_pythonpath` | Remove or update — startup block is being removed from openclaw.json | `test_openclaw_config.py` |
| `test_openclaw_json_matches_generation` | No change — compares tool names/descriptions only, not transport metadata | `test_openclaw_config.py` |

### Sampling Rate
- **Per task commit:** `cd mcp-server && PYTHONPATH=.:../app-store-gui BLUEPRINTS_DIR=/tmp python -m pytest tests/ -q --tb=short`
- **Per wave merge:** `cd mcp-server && PYTHONPATH=.:../app-store-gui BLUEPRINTS_DIR=/tmp python -m pytest tests/ -v`
- **Phase gate:** Full suite green (all tests including new HTTP tests) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_http_transport.py` — covers XPORT-01, XPORT-03, XPORT-04; use Starlette `TestClient` against `mcp.streamable_http_app()`
- [ ] `tests/test_config.py` additions — `test_mcp_transport_default`, `test_mcp_port_default`, `test_mcp_port_int_cast` — covers XPORT-02 config layer

**Starlette TestClient import (available via transitive mcp dependency):**
```python
from starlette.testclient import TestClient
```

---

## Sources

### Primary (HIGH confidence)
- FastMCP 1.26.0 source (inspected locally) — `FastMCP.__init__`, `FastMCP.run`, `FastMCP.run_streamable_http_async`, `FastMCP.streamable_http_app`, `FastMCP.custom_route`
- Live test: `FastMCP("weka-app-store-mcp", host="0.0.0.0", port=8080, stateless_http=True)` verified constructor params
- Live test: `@mcp.custom_route("/health", methods=["GET"])` verified routes in `streamable_http_app().routes`
- Live test: `mcp.run signature: (self, transport: Literal['stdio', 'sse', 'streamable-http'] = 'stdio', mount_path: str | None = None)`
- `pytest` baseline: 103 tests passing confirmed (`cd mcp-server && PYTHONPATH=.:../app-store-gui BLUEPRINTS_DIR=/tmp python -m pytest tests/ --tb=no -q`)

### Secondary (MEDIUM confidence)
- MCP REQUIREMENTS.md: `SSE transport` explicitly out of scope; `stateless mode` required per XPORT-04
- CONTEXT.md: Research confirmed `mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)` — NOTE: `host`/`port` are constructor args, not `run()` args; CONTEXT.md wording is slightly imprecise but intent is correct

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — SDK installed locally, signatures inspected from source
- Architecture patterns: HIGH — constructor params and custom_route verified by running code
- Pitfalls: HIGH — localhost binding default and MCP_PORT int cast verified programmatically; test breakage verified by reading test assertions
- openclaw.json changes: HIGH — current file and generator both read and understood

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (mcp SDK stable; no fast-moving changes expected in 1.26.x)
