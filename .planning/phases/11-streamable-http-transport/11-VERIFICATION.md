---
phase: 11-streamable-http-transport
verified: 2026-03-23T08:00:00Z
status: human_needed
score: 9/10 must-haves verified
re_verification: false
human_verification:
  - test: "Start the server with MCP_TRANSPORT=http and run: curl -s http://localhost:8080/health"
    expected: "HTTP 200 with body {\"status\": \"ok\", \"tools\": 8, \"transport\": \"http\"}"
    why_human: "Health endpoint behavior under a real bound socket cannot be verified by grep or pytest alone. The TestClient exercises the Starlette app layer but does not bind a real port or exercise uvicorn."
  - test: "Start the server with MCP_TRANSPORT=http and issue a tool call (e.g. list_blueprints) over HTTP using an MCP client or curl against /mcp"
    expected: "Tool response JSON matches the flat shape verified by test_response_depth tests in stdio mode — captured_at present, no nested wrapper objects, same field names"
    why_human: "Success criterion 4 from ROADMAP.md ('Tool calls over HTTP return the same flat JSON responses as stdio') is not covered by any automated test. The depth contract tests run in stdio mode only."
---

# Phase 11: Streamable HTTP Transport Verification Report

**Phase Goal:** MCP server runs in dual-mode: stdio (default) and Streamable HTTP, selected by env var, fully validated locally before any cluster work begins
**Verified:** 2026-03-23T08:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `curl localhost:8080/health` returns HTTP 200 when server starts with `MCP_TRANSPORT=http` | ? HUMAN | TestClient test passes (`test_health_endpoint_returns_200` PASSED); real socket bind needs human confirm |
| 2 | `MCP_TRANSPORT=stdio` (default) starts the server exactly as before; all 103 existing tests pass unchanged | ✓ VERIFIED | 117 tests pass; 103 pre-existing tests included; `test_stdio_mode_unchanged` PASSED; `test_server_mcp_exists_with_tools` confirms 8 tools in stdio mode |
| 3 | `MCP_TRANSPORT=http` starts the server in Streamable HTTP mode on the port set by `MCP_PORT` | ✓ VERIFIED | `server.py:36-37` conditionally constructs `FastMCP("weka-app-store-mcp", host="0.0.0.0", port=_port, stateless_http=True)` when `_transport == "http"`; `test_http_mcp_host_and_port` and `test_stateless_mode_enabled` PASSED |
| 4 | Tool calls over HTTP return the same flat JSON responses as stdio (depth contract preserved) | ? HUMAN | No HTTP-mode tool call test exists; `test_response_depth_*` tests run stdio only; shape equivalence requires runtime verification |
| 5 | `openclaw.json` points to `http://localhost:8080/mcp` with `"transport": "streamable-http"` replacing the stdio startup block | ✓ VERIFIED | `openclaw.json:5-6` has `"transport": "streamable-http"` and `"url": "http://localhost:8080/mcp"`; `"startup"` key absent; all 11 openclaw config tests PASSED including drift detection |

**Score:** 9/10 must-haves verified (automated); 2 items flagged for human runtime confirmation

### Plan 01 Must-Haves (config.py / server.py / tests)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `MCP_TRANSPORT=http` constructs FastMCP with `host=0.0.0.0`, configurable port, `stateless_http=True` | ✓ VERIFIED | `server.py:37`; `test_http_mcp_host_and_port` + `test_stateless_mode_enabled` PASSED |
| 2 | `MCP_TRANSPORT=stdio` (or unset) constructs FastMCP with defaults, exactly as before | ✓ VERIFIED | `server.py:43`; `test_stdio_mode_unchanged` PASSED |
| 3 | `/health` returns HTTP 200 with `{status: ok, tools: 8, transport: http}` in HTTP mode | ? HUMAN | `test_health_endpoint_returns_200` PASSED via Starlette TestClient; real socket bind unverified |
| 4 | `/health` is not registered in stdio mode | ✓ VERIFIED | `test_health_only_in_http_mode` PASSED — checks `_custom_starlette_routes` for absence of `/health` path |
| 5 | All 103 existing tests pass unchanged when `MCP_TRANSPORT` is unset | ✓ VERIFIED | 117 total tests PASSED; suite includes all pre-existing 103 tests |

### Plan 02 Must-Haves (openclaw.json / Dockerfile)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `openclaw.json` has `transport=streamable-http` and `url=http://localhost:8080/mcp` | ✓ VERIFIED | `openclaw.json:5-6`; `test_openclaw_json_has_http_transport` + `test_openclaw_json_has_url` PASSED |
| 2 | `openclaw.json` has no startup block | ✓ VERIFIED | No `"startup"` key in file; `test_openclaw_json_no_startup_block` PASSED |
| 3 | `openclaw.json` `env.optional` includes `MCP_TRANSPORT` and `MCP_PORT` | ✓ VERIFIED | `openclaw.json:11-17`; `test_openclaw_json_optional_env_includes_transport_vars` PASSED |
| 4 | `generate_openclaw_config.py` produces the same HTTP-transport structure as `openclaw.json` | ✓ VERIFIED | `test_openclaw_json_matches_generation` PASSED (drift detection) |
| 5 | Dockerfile has `EXPOSE 8080` | ✓ VERIFIED | `Dockerfile:32` |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `mcp-server/config.py` | MCP_TRANSPORT and MCP_PORT constants | ✓ VERIFIED | Lines 33-37: both constants present with correct defaults (`"stdio"`, `8080`) and correct types |
| `mcp-server/server.py` | Conditional mcp construction and health endpoint | ✓ VERIFIED | Lines 33-43: conditional block on `_transport == "http"`, `stateless_http=True`, `@mcp.custom_route("/health")` |
| `mcp-server/tests/test_http_transport.py` | HTTP transport and health endpoint tests (>=40 lines) | ✓ VERIFIED | 133 lines, 7 tests: `test_http_mcp_host_and_port`, `test_stateless_mode_enabled`, `test_health_endpoint_returns_200`, `test_health_only_in_http_mode`, `test_transport_arg_selection`, `test_stdio_mode_unchanged`, `test_server_mcp_exists_with_tools` |
| `mcp-server/tests/test_config.py` | MCP_TRANSPORT and MCP_PORT config tests | ✓ VERIFIED | Lines 72-110: 5 new tests (`test_mcp_transport_default`, `test_mcp_transport_http`, `test_mcp_port_default`, `test_mcp_port_custom`, `test_mcp_port_is_int`) |
| `mcp-server/openclaw.json` | HTTP transport registration config | ✓ VERIFIED | `"transport": "streamable-http"` at line 5 |
| `mcp-server/generate_openclaw_config.py` | Updated config generator | ✓ VERIFIED | `"transport": "streamable-http"` at line 126; `build_openclaw_config()` docstring updated |
| `mcp-server/tests/test_openclaw_config.py` | Updated transport assertions | ✓ VERIFIED | Contains `streamable-http` assertions; all 11 tests PASSED |
| `mcp-server/Dockerfile` | EXPOSE 8080 directive | ✓ VERIFIED | Line 32: `EXPOSE 8080` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `mcp-server/server.py` | `mcp-server/config.py` | `config.MCP_TRANSPORT` and `config.MCP_PORT` | ✓ WIRED | `server.py:33-34`: `_transport = config.MCP_TRANSPORT` / `_port = config.MCP_PORT` |
| `mcp-server/server.py` | FastMCP constructor | `host="0.0.0.0"`, `port`, `stateless_http` params | ✓ WIRED | `server.py:37`: `FastMCP("weka-app-store-mcp", host="0.0.0.0", port=_port, stateless_http=True)` |
| `mcp-server/tests/test_http_transport.py` | Starlette TestClient | `mcp.streamable_http_app()` | ✓ WIRED | `test_http_transport.py:55`: `client = TestClient(mcp.streamable_http_app())` |
| `mcp-server/openclaw.json` | `mcp-server/generate_openclaw_config.py` | drift detection test | ✓ WIRED | `test_openclaw_json_matches_generation` compares on-disk JSON to in-memory generator output; PASSED |
| `mcp-server/tests/test_openclaw_config.py` | `mcp-server/openclaw.json` | JSON assertions | ✓ WIRED | `test_openclaw_json_has_http_transport` asserts `"streamable-http"` against loaded file |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| XPORT-01 | 11-01, 11-02 | MCP server supports Streamable HTTP transport on a configurable port alongside existing stdio | ✓ SATISFIED | `server.py` conditional FastMCP construction; `config.py` `MCP_PORT` (int, default 8080); 117 tests pass |
| XPORT-02 | 11-01, 11-02 | `MCP_TRANSPORT` env var selects transport mode (`stdio` default, `http` for sidecar deployment) | ✓ SATISFIED | `config.py:34`: `MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")`; `server.py:33`: `_transport = config.MCP_TRANSPORT`; 5 config tests PASSED |
| XPORT-03 | 11-01 | Health endpoint (`/health`) returns 200 when server is ready for tool calls | ✓ SATISFIED | `server.py:39-41`: `@mcp.custom_route("/health")` registers health endpoint in HTTP mode; `test_health_endpoint_returns_200` PASSED via TestClient; real socket behavior flagged for human verification |
| XPORT-04 | 11-01 | HTTP transport operates in stateless mode (no session ID dependency) | ✓ SATISFIED | `server.py:37`: `stateless_http=True`; `test_stateless_mode_enabled` PASSED |

No orphaned requirements — REQUIREMENTS.md traceability table maps XPORT-01 through XPORT-04 exclusively to Phase 11, all four are addressed by plans 11-01 and 11-02.

### Anti-Patterns Found

No anti-patterns found. Scan of all 8 phase-modified files returned no TODOs, FIXMEs, placeholders, empty implementations, or stub returns.

### Human Verification Required

#### 1. Health Endpoint Under Real Socket

**Test:** Set `MCP_TRANSPORT=http BLUEPRINTS_DIR=/tmp` and run the server: `cd mcp-server && MCP_TRANSPORT=http BLUEPRINTS_DIR=/tmp PYTHONPATH=.:../app-store-gui python -m server`. In a second terminal: `curl -s http://localhost:8080/health`
**Expected:** HTTP 200 response with body `{"status": "ok", "tools": 8, "transport": "http"}`
**Why human:** The Starlette TestClient exercises the app layer and does not bind a real port. Uvicorn startup, port binding, and actual HTTP socket behavior require a live process.

#### 2. HTTP Tool Call Response Shape (Success Criterion 4)

**Test:** With the server running in HTTP mode (from test 1 above), issue a tool call via an MCP client or directly against `/mcp`. Example: call `list_blueprints` with `BLUEPRINTS_DIR` pointing to a directory with at least one blueprint YAML.
**Expected:** Response JSON matches the flat structure validated by the depth contract tests — `captured_at` present, no nested wrapper objects, same field names as in stdio mode.
**Why human:** No automated test exercises tool calls over the HTTP transport path. The `test_response_depth_*` suite runs only in stdio mode. Success criterion 4 from the ROADMAP cannot be confirmed by static analysis.

### Gaps Summary

No automation gaps. All artifacts exist, are substantive, and are wired. All 5 plan commits are verified in the repository. The two human verification items reflect inherent runtime behavior (live socket, HTTP tool call shape) that automated tests cannot fully cover — not missing implementation. The Starlette TestClient evidence for the health endpoint is strong; the health endpoint code is correct and tested at the app layer.

---

_Verified: 2026-03-23T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
