# Stack Research

**Domain:** MCP server in Python — WEKA App Store OpenClaw tool integration
**Researched:** 2026-03-20
**Confidence:** MEDIUM-HIGH — MCP SDK verified via PyPI/GitHub (HIGH); OpenClaw stdio config format verified via community docs (MEDIUM); NemoClaw is alpha as of March 16 2026, official config schema not yet published (LOW on NemoClaw-specific details)

---

## Context: This Is A Brownfield Addition

The existing validated stack is **not changing**. Everything below is what needs to be added for the MCP server milestone only.

**Existing stack (already in `requirements.txt`, do not re-research):**
- `fastapi>=0.111.0`, `uvicorn[standard]>=0.30.0`, `Jinja2>=3.1.4` — web UI (keep as-is)
- `kubernetes>=27.0.0` — K8s API client (reused by MCP tool implementations)
- `PyYAML>=6.0.1` — YAML parsing (reused by validate and apply tools)
- `pytest>=8.0.0` — test runner (extended with async support)

**Existing code to reuse as tool implementations (do not duplicate):**
- `webapp/inspection/cluster.py` — `collect_cluster_inspection()` → behind `weka_appstore_inspect_cluster` tool
- `webapp/inspection/weka.py` — WEKA inspection → behind `weka_appstore_inspect_weka` tool
- `webapp/planning/apply_gateway.py` — `ApplyGateway` → behind `weka_appstore_apply` tool
- `webapp/planning/validator.py` — validation logic → behind `weka_appstore_validate_yaml` tool

---

## Recommended Stack

### Core Technologies (New Additions Only)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `mcp` (official Python SDK) | `>=1.9,<2` | MCP server framework with `FastMCP` high-level API | Official Anthropic SDK; FastMCP is bundled inside the `mcp` package (not a separate install); provides `@mcp.tool()` decorator that auto-generates JSON schema from Python type hints; stdio transport is the default and what OpenClaw uses to spawn the server process |
| `mcp[cli]` extra | same as `mcp` | MCP Inspector dev tool for manual tool testing | The `cli` extra installs the `mcp dev` and `mcp run` commands; `mcp dev server.py` opens a browser-based inspector to call tools interactively without writing any test code first — essential for iterating on tool shapes before writing automated tests |

**Version pin rationale:** Pin `<2` because v2 is pre-alpha on the `main` branch as of March 2026. v1.x is the stable release. Latest confirmed stable on PyPI: `1.9.4` (June 2025). The `mcp` package requires Python >=3.10.

### Supporting Libraries (New Additions Only)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest-asyncio` | `>=1.3.0` | Async test execution for MCP tool tests | Required because FastMCP's in-memory test client calls are async coroutines; the existing test suite uses only synchronous pytest; latest stable: 1.3.0 (November 2025) |

**Note on event loop management:** The `mcp` package pulls in `anyio` as a transitive dependency. Do not install anyio explicitly unless pinning a version. Configure `pytest-asyncio` in strict mode (explicit `@pytest.mark.asyncio` markers) to avoid the known conflict where pytest-asyncio auto mode and anyio's pytest plugin both attempt to own the event loop.

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `mcp dev <server.py>` | Opens MCP Inspector in the browser — interactive tool call UI | Installed via `mcp[cli]` extra; use before writing any automated test to verify tool schemas, argument shapes, and return values |
| `mcp run <server.py>` | Launches the server via stdio in the terminal | Use to smoke-test the server entrypoint as OpenClaw would invoke it (spawns process, reads JSON-RPC from stdin) |
| `unittest.mock.patch` / `pytest monkeypatch` | Mock K8s API calls in unit tests | Already available in stdlib + pytest; no new package needed; the existing codebase uses this pattern extensively (see `tests/planning/`) |

---

## Installation

Add to `app-store-gui/requirements.txt` (or a separate `mcp-server/requirements.txt` if the MCP server becomes a standalone deployment unit):

```
# MCP server
mcp[cli]>=1.9,<2

# Async test support for MCP tool tests
pytest-asyncio>=1.3.0
```

```bash
# Install
pip install "mcp[cli]>=1.9,<2" "pytest-asyncio>=1.3.0"
```

---

## Tool Registration Pattern

The FastMCP `@mcp.tool()` decorator is the registration mechanism. Type hints drive the JSON schema that OpenClaw sees; the docstring becomes the tool description the agent reads.

```python
# mcp_server/server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weka-app-store")

@mcp.tool()
def weka_appstore_inspect_cluster() -> dict:
    """Return a bounded snapshot of Kubernetes cluster state.

    Returns GPU inventory (model, count, memory per node), CPU allocatable vs
    requested, RAM allocatable vs requested, namespaces, and storage classes.
    Read-only. No cluster mutation.
    """
    from webapp.inspection.cluster import collect_cluster_inspection
    return collect_cluster_inspection()

@mcp.tool()
def weka_appstore_validate_yaml(yaml_content: str) -> dict:
    """Validate a WekaAppStore YAML document against the CRD and operator contract.

    Returns pass/fail with actionable errors. Call this before weka_appstore_apply.
    """
    from webapp.planning.validator import validate_plan_yaml
    return validate_plan_yaml(yaml_content)

if __name__ == "__main__":
    mcp.run()  # stdio transport by default — what OpenClaw uses
```

**Critical rules:**
- Decorator is `@mcp.tool()` with parentheses — `@mcp.tool` (no parentheses) raises `TypeError`
- Type hints on parameters generate the JSON schema OpenClaw uses to construct arguments
- Docstrings are agent-facing, not human-facing — write them to guide the agent's reasoning
- `mcp.run()` with no arguments defaults to stdio transport; this is correct for OpenClaw subprocess model
- Return `dict` — FastMCP serializes it to JSON automatically

---

## OpenClaw Registration Format

OpenClaw registers MCP servers in YAML configuration. The server runs as a child process via stdio — no HTTP server, no port, no TLS:

```yaml
# ~/.openclaw/openclaw.yaml (or openclaw.json — format depends on OpenClaw version)
agents:
  - id: main
    model: anthropic/claude-sonnet-4-5
    mcp_servers:
      - name: weka-app-store
        command: python3
        args:
          - /path/to/app-store-gui/mcp_server/server.py
        env:
          KUBECONFIG: /path/to/.kube/config
          WEKA_API_ENDPOINT: https://weka-cluster.example.com
```

OpenClaw spawns the process at agent start and communicates over stdin/stdout using JSON-RPC 2.0. The agent discovers available tools through the MCP protocol handshake.

**Confidence: MEDIUM** — format verified via community documentation (clawctl.com MCP setup guide, March 2026). NemoClaw alpha as of March 16 2026 does not yet publish an official config schema. The stdio subprocess model itself is confirmed by the MCP specification and multiple sources.

---

## Mock Agent Harness Pattern

FastMCP provides an in-process test client — the entire tool chain can be exercised without spawning a subprocess or running a live OpenClaw instance:

```python
# tests/mcp/conftest.py
import pytest
from mcp import Client
from mcp_server.server import mcp

@pytest.fixture
async def mcp_client():
    async with Client(transport=mcp) as client:
        yield client
```

```python
# tests/mcp/test_tool_cluster.py
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_inspect_cluster_returns_domains(mcp_client):
    mock_result = {"domains": {"cpu": {"status": "complete", "observed": {"free_cores": 48.0}}}}
    with patch("webapp.inspection.cluster.collect_cluster_inspection", return_value=mock_result):
        result = await mcp_client.call_tool("weka_appstore_inspect_cluster", {})
    assert result.data["domains"]["cpu"]["status"] == "complete"
```

The mock agent harness (`test_mock_harness.py`) calls tools in the full deployment sequence to prove end-to-end flow without a live agent:

```python
@pytest.mark.asyncio
async def test_full_deploy_flow(mcp_client, mock_k8s, mock_weka):
    # 1. Inspect cluster resources
    cluster = await mcp_client.call_tool("weka_appstore_inspect_cluster", {})
    # 2. Inspect WEKA storage
    weka = await mcp_client.call_tool("weka_appstore_inspect_weka", {})
    # 3. List blueprints
    catalog = await mcp_client.call_tool("weka_appstore_list_blueprints", {})
    # 4. Validate generated YAML
    validation = await mcp_client.call_tool(
        "weka_appstore_validate_yaml",
        {"yaml_content": SAMPLE_WEKAAPPSTORE_YAML}
    )
    assert validation.data["valid"] is True
    # 5. Apply (mocked K8s write)
    apply_result = await mcp_client.call_tool(
        "weka_appstore_apply",
        {"yaml_content": SAMPLE_WEKAAPPSTORE_YAML, "namespace": "ai-platform"}
    )
    assert apply_result.data["applied"] == ["WekaAppStore"]
```

---

## Recommended File Layout

The MCP server is a new top-level module alongside `webapp/`, not wired into the FastAPI app:

```
app-store-gui/
  mcp_server/
    __init__.py
    server.py          # FastMCP instance; all @mcp.tool() registrations
    tools/
      __init__.py
      cluster.py       # thin wrapper over webapp/inspection/cluster.py
      weka.py          # thin wrapper over webapp/inspection/weka.py
      blueprints.py    # blueprint catalog read (list + get)
      validate.py      # thin wrapper over webapp/planning/validator.py
      apply.py         # thin wrapper over webapp/planning/apply_gateway.py
      status.py        # WekaAppStore CR status query via K8s custom objects API
      schema.py        # CRD schema retrieval for agent YAML generation context
  tests/
    mcp/
      conftest.py           # async fixtures: in-memory FastMCP client, mock K8s/WEKA
      test_tool_cluster.py
      test_tool_weka.py
      test_tool_blueprints.py
      test_tool_validate.py
      test_tool_apply.py
      test_tool_status.py
      test_tool_schema.py
      test_mock_harness.py  # end-to-end tool chain without live OpenClaw
```

---

## pytest Configuration Update

Add `asyncio_mode = strict` to avoid event loop conflicts between pytest-asyncio and anyio:

```ini
# pytest.ini
[pytest]
asyncio_mode = strict
```

This keeps pytest-asyncio in strict mode (tests must be explicitly marked `@pytest.mark.asyncio`) and prevents anyio's pytest plugin from interfering.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `mcp` official SDK with bundled `FastMCP` | `fastmcp` PyPI package (PrefectHQ / jlowin) | The community `fastmcp` v3.0 (January 2026) adds granular auth, OpenTelemetry, and OpenAPI provider support — switch to it if the project needs multi-server composition, fine-grained authorization, or built-in tracing |
| stdio transport (`mcp.run()` default) | Streamable HTTP transport | Use HTTP if the MCP server must serve multiple concurrent OpenClaw instances, if browser-based MCP clients need direct access, or if a shared remote MCP server deployment is needed |
| MCP server as a module in `app-store-gui/` | Separate Python package/repo | Use a separate package only if the MCP server needs independent versioning, a separate container image, or an independent deployment pipeline |
| `pytest-asyncio` strict mode | anyio `@pytest.mark.anyio` | Use anyio's marker if the whole test suite migrates to anyio backends — for now, strict asyncio_mode avoids retrofitting the existing synchronous test files |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pip install fastmcp` (PyPI `fastmcp` package) | This is `PrefectHQ/fastmcp` — a community fork with its own release cycle and diverged APIs. It is NOT the official FastMCP. The official FastMCP lives inside the `mcp` package at `from mcp.server.fastmcp import FastMCP` | `pip install "mcp[cli]>=1.9,<2"` |
| `mcp>=2` | v2 is pre-alpha on the `main` branch with breaking API changes as of March 2026 | `mcp>=1.9,<2` |
| HTTP/SSE transport for the initial implementation | Requires a running server with an open port, firewall rules, and optional TLS; OpenClaw needs a URL instead of a command; adds operational complexity for no benefit when only one agent instance connects | `mcp.run()` stdio (default) |
| Custom chat UI routes in FastAPI | OpenClaw provides the conversation interface natively — building a parallel UI creates two authoritative interfaces | OpenClaw's native WebSocket Gateway / chat channels |
| Session or conversation state in MCP tool responses | OpenClaw has built-in memory and session management — adding state to tools creates conflicts with the agent's own tracking | Stateless tools; let OpenClaw own state |
| Planning/compiler/family-matcher logic in the MCP server | OpenClaw reasons about YAML structure from the CRD schema context the tools provide — reimplementing this in Python duplicates the agent's core reasoning capability | `weka_appstore_get_crd_schema` + `weka_appstore_validate_yaml` give the agent what it needs; remove the v1.0 compiler and family-matcher |
| `asyncio_mode = "auto"` in pytest-asyncio | Auto mode causes pytest-asyncio to claim all async tests globally, which conflicts with anyio's plugin that the `mcp` package pulls in | `asyncio_mode = "strict"` with explicit `@pytest.mark.asyncio` markers |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `mcp>=1.9,<2` | Python >=3.10 | Confirm Python version is >=3.10 in the project; the rest of the existing stack (FastAPI, kubernetes-client) supports 3.10+ |
| `mcp>=1.9,<2` | `anyio` (transitive) | `mcp` pulls in anyio automatically; no explicit anyio pin needed unless a specific version conflict appears |
| `pytest-asyncio>=1.3.0` | `pytest>=8.0.0` | Existing repo already has `pytest>=8.0.0` — compatible |
| `pytest-asyncio>=1.3.0` | anyio pytest plugin | Keep `asyncio_mode = strict` to prevent both plugins competing for the event loop |
| `kubernetes>=27.0.0` | MCP server tools | The existing kubernetes client is imported by tool implementations; no version change needed |
| `PyYAML>=6.0.1` | MCP validate tool | Existing PyYAML is reused by `weka_appstore_validate_yaml`; no version change needed |

---

## Sources

- PyPI `mcp` package (official Anthropic SDK) — v1.9.4 confirmed latest stable, v2 pre-alpha: https://pypi.org/project/mcp/
- GitHub `modelcontextprotocol/python-sdk` — v1.x is current stable release: https://github.com/modelcontextprotocol/python-sdk
- MCP Python SDK official API docs: https://py.sdk.modelcontextprotocol.io/
- FastMCP in-process testing guide (official MCP docs): https://gofastmcp.com/servers/testing
- OpenClaw MCP server YAML config format (community, MEDIUM confidence): https://www.clawctl.com/blog/mcp-server-setup-guide
- MCP tool registration decorator pattern (official MCP docs): https://modelcontextprotocol.io/docs/develop/build-server
- NemoClaw alpha announcement — NVIDIA, March 16 2026: https://nvidianews.nvidia.com/news/nvidia-announces-nemoclaw
- pytest-asyncio PyPI — v1.3.0 released November 2025: https://pypi.org/project/pytest-asyncio/
- MCPcat unit testing guide (mock patterns, in-process client): https://mcpcat.io/guides/writing-unit-tests-mcp-servers/
- Stop Vibe-Testing Your MCP Server (jlowin.dev) — pytest fixture patterns for FastMCP: https://www.jlowin.dev/blog/stop-vibe-testing-mcp-servers

---

*Stack research for: MCP server + OpenClaw tool integration (WEKA App Store v2.0 milestone)*
*Researched: 2026-03-20*
*Replaces: previous STACK.md (NemoClaw planning layer, v1.0 architecture — now superseded by OpenClaw pivot)*
