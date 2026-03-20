# Phase 6: MCP Scaffold and Read-Only Tools - Research

**Researched:** 2026-03-20
**Domain:** MCP Python SDK (FastMCP), inspection layer flattening, blueprint catalog parsing, CRD schema delivery
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Tool Response Shape**
- Single flat JSON object per tool response — no nested domain/status/freshness wrappers from v1.0
- Top-level `captured_at` timestamp on every response
- Aggregated cluster summary for GPU data (gpu_total, gpu_models, gpu_memory_total_gib) — not per-node breakdown
- Separate `warnings` array at top level for blocker/problem information (e.g. "GPU operator not detected")
- Data fields carry facts (gpu_operator_installed: false); warnings array carries things that need agent attention
- `inspect_cluster` and `inspect_weka` are separate tools — agent calls what it needs

**Blueprint Catalog Source**
- Blueprints are sourced from the `weka/warp-blueprints` GitHub repo, synced into the container at `/app/manifests/manifest` at runtime
- MCP server scans this directory to discover blueprints — not a hardcoded list
- `BLUEPRINTS_DIR` env var configurable, defaults to `/app/manifests/manifest` (same pattern as the FastAPI app)
- `list_blueprints` returns full metadata per blueprint: name, description, category, Helm chart ref, default values, resource requirements (GPU/CPU/RAM/storage minimums), prerequisites, supported configurations
- `get_blueprint` returns complete blueprint detail including Helm values schema and defaults
- Supplemental metadata (resource requirements, descriptions) can be carried in a YAML registry file if not extractable from the manifests themselves

**MCP Server Location**
- Separate container — own Dockerfile, own image, deployed as sidecar or standalone
- Own `requirements.txt` adding `mcp[cli]` and referencing shared deps independently
- Claude's discretion on whether code lives at `mcp-server/` root or inside `app-store-gui/`
- Imports reusable code from `inspection/cluster.py`, `inspection/weka.py`, `planning/apply_gateway.py`, `planning/validator.py` via shared PYTHONPATH

**CRD Schema Delivery**
- `get_crd_schema` reads the live CRD from the running cluster via apiextensions API — always current, no Helm template noise
- Response includes 1-2 example `WekaAppStore` YAML documents picked from the blueprints directory (`/app/manifests/manifest`) — real manifests, not hand-written
- Examples help the agent pattern-match valid YAML structure alongside the schema

### Claude's Discretion
- Exact MCP server directory placement (repo root vs inside app-store-gui/)
- Internal module structure of the MCP server
- How to extract blueprint metadata from manifest files (parsing strategy)
- Tool description wording (must include when/why/sequencing guidance per success criteria)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCPS-01 | MCP server scaffold using official `mcp[cli]` SDK with FastMCP, stdio transport | FastMCP 1.26.0, `mcp.run()` defaults to stdio — confirmed |
| MCPS-02 | `inspect_cluster` tool returns flat GPU, CPU, RAM, namespace, and storage class data | `collect_cluster_inspection()` + `flatten_cluster_status()` already exist; need MCP wrapper + re-flatten to drop `inspection_snapshot` key |
| MCPS-03 | `inspect_weka` tool returns WEKA capacity, filesystems, and mount data | `collect_weka_inspection()` exists; domains wrapper must be stripped for flat output |
| MCPS-04 | `list_blueprints` tool returns blueprint catalog with names, descriptions, and resource requirements | Scan `BLUEPRINTS_DIR` for YAML manifests; parse WekaAppStore YAML; supplemental metadata registry pattern identified |
| MCPS-05 | `get_blueprint` tool returns full blueprint detail including Helm values schema and defaults | Single-blueprint detail view from same YAML parse; Helm values YAML already present in `weka-csi-config/blueprint-default-values.yaml` example |
| MCPS-06 | `get_crd_schema` tool returns the `WekaAppStore` CRD spec for agent YAML generation | `apiextensions_api.read_custom_resource_definition("wekaappstores.warp.io")` — pattern already in `cluster.py` |
| MCPS-10 | All tool responses use flat agent-friendly JSON, not nested v1.0 planning models | `flatten_cluster_status()` is the template; write parallel flatten functions for weka and blueprint tools |
| MCPS-11 | All logging goes to stderr, never stdout (stdio transport requirement) | `logging.basicConfig(stream=sys.stderr)` before FastMCP init; confirmed MCP protocol requirement |
</phase_requirements>

---

## Summary

Phase 6 builds a FastMCP server (official `mcp[cli]` SDK, version 1.26.0 as of early 2026) that exposes 5 read-only tools. The critical constraint is that all output must be flat: the existing `collect_cluster_inspection()` and `collect_weka_inspection()` functions return domain-wrapped dicts with nested `observed`, `freshness`, `blockers` structures. `flatten_cluster_status()` already exists as a template — this phase creates parallel flatten functions for every tool and never exposes the inner domain model to agents.

The MCP transport is stdio (default for `mcp.run()`). This means logging must go exclusively to stderr. The FastMCP `@mcp.tool()` decorator extracts function name, docstring, and type hints automatically into the MCP tool schema. Returning a plain `dict` from a tool produces both a `TextContent` JSON serialization (backward-compatible) and `structuredContent` in the MCP response — the agent sees flat JSON either way.

**Primary recommendation:** Create `mcp-server/` at repo root with `server.py` (FastMCP entry point), `tools/` (one file per tool domain), `blueprints.py` (scanner/parser), and `flatten.py` (all flatten functions). Configure `logging.basicConfig(stream=sys.stderr, level=logging.INFO)` before instantiating `FastMCP`. The blueprint metadata parsing strategy: read WekaAppStore YAML manifests, extract `spec` fields; for fields not in manifest (resource minimums, category), maintain a `metadata_registry.yaml` sidecar.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mcp[cli]` | 1.26.0 | MCP server framework (FastMCP + dev CLI) | Official SDK from Anthropic/MCP project; `mcp dev server.py` works only with FastMCP |
| `kubernetes` | >=27.0.0 | K8s API calls for cluster/weka/CRD inspection | Already used in inspection layer |
| `PyYAML` | >=6.0.1 | Parse blueprint YAML manifests | Already used in webapp |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `fastapi` | — | NOT needed in MCP server | The MCP server is stdio-only, no HTTP needed |
| `uvicorn` | — | NOT needed in MCP server | Same — stdio transport only |
| `anyio` | pulled by `mcp` | Async runtime | Already a transitive dep of `mcp[cli]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `mcp[cli]` (official) | `fastmcp` (third-party PyPI package by @jlowin) | Official SDK is preferred; `mcp dev` works only with official SDK; third-party package has different API |
| `stdio` transport | `streamable-http` transport | HTTP only needed for multi-client; adds complexity; OpenClaw/NemoClaw uses stdio |

**Installation (MCP server `requirements.txt`):**
```bash
mcp[cli]>=1.26.0
kubernetes>=27.0.0
PyYAML>=6.0.1
```

---

## Architecture Patterns

### Recommended Project Structure
```
mcp-server/
├── server.py           # FastMCP entry point, mcp.run()
├── tools/
│   ├── __init__.py
│   ├── inspect_cluster.py   # inspect_cluster tool + flatten
│   ├── inspect_weka.py      # inspect_weka tool + flatten
│   ├── blueprints.py        # list_blueprints + get_blueprint + scanner
│   └── crd_schema.py        # get_crd_schema tool
├── flatten.py          # Shared flatten helpers (if needed cross-tool)
├── config.py           # Env var config (BLUEPRINTS_DIR, KUBERNETES_AUTH_MODE)
└── requirements.txt    # mcp[cli], kubernetes, PyYAML
```

The MCP server shares Python code with `app-store-gui/webapp/` by having `app-store-gui/` on `PYTHONPATH` (set in Dockerfile). This avoids duplicating `inspection/cluster.py`, `inspection/weka.py`, etc.

### Pattern 1: FastMCP Tool Registration

**What:** Decorate a function with `@mcp.tool()`. FastMCP extracts name, docstring (tool description), and type hints (input schema) automatically.
**When to use:** Every tool in this phase uses this pattern.

```python
# Source: Official MCP SDK docs (modelcontextprotocol.io/docs/develop/build-server)
import logging
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging BEFORE FastMCP init to prevent SDK from hijacking handlers
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

mcp = FastMCP("weka-app-store")

@mcp.tool()
def inspect_cluster() -> dict:
    """Call this tool FIRST to understand cluster resources before blueprint selection.

    Returns a flat snapshot of available CPU cores, memory, GPU devices,
    namespaces, and storage classes. Call before list_blueprints to know
    which blueprints fit the cluster. Call again after time passes to
    refresh — results are not cached.

    Returns: flat JSON with captured_at, warnings[], and all resource fields.
    """
    snapshot = collect_cluster_inspection(...)
    return _flatten_for_mcp(snapshot)


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport
```

### Pattern 2: Flat Response Shape

**What:** Strip domain wrappers from v1 inspection results. Return a single dict with `captured_at` at top level, resource fields at top level, and a `warnings` array.
**When to use:** Every tool response in this phase.

The existing `flatten_cluster_status()` in `cluster.py` is the right template, but it retains `inspection_snapshot` (the full nested original). The MCP tool flatten must drop that and restructure GPU inventory.

```python
# Source: Pattern derived from existing flatten_cluster_status() in cluster.py
def flatten_inspect_cluster_for_mcp(snapshot: dict) -> dict:
    """Flatten collect_cluster_inspection() output to flat MCP tool response."""
    cpu = snapshot.get("domains", {}).get("cpu", {}).get("observed", {})
    memory = snapshot.get("domains", {}).get("memory", {}).get("observed", {})
    gpu_domain = snapshot.get("domains", {}).get("gpu", {})
    gpu = gpu_domain.get("observed", {})
    namespaces = snapshot.get("domains", {}).get("namespaces", {}).get("observed", {})
    storage = snapshot.get("domains", {}).get("storage_classes", {}).get("observed", {})

    # Collect warnings from all blockers
    warnings = []
    for domain in snapshot.get("domains", {}).values():
        for blocker in domain.get("blockers", []):
            warnings.append(blocker.get("message", ""))
    if not snapshot.get("gpu_operator_installed"):
        warnings.append("GPU operator not detected — GPU workloads may not schedule")

    # Aggregate GPU inventory
    inventory = gpu.get("inventory", [])
    gpu_models = [item.get("model") for item in inventory if item.get("model")]
    gpu_total = sum(item.get("count", 0) for item in inventory)
    gpu_memory_total_gib = sum(
        (item.get("count", 0) * (item.get("memory_gib") or 0))
        for item in inventory
    )

    return {
        "captured_at": snapshot.get("captured_at"),
        "k8s_version": snapshot.get("k8s_version"),
        "cpu_nodes": cpu.get("cpu_nodes"),
        "gpu_nodes": gpu.get("gpu_nodes"),
        "cpu_cores_total": cpu.get("allocatable_cores"),
        "cpu_cores_free": cpu.get("free_cores"),
        "memory_gib_total": memory.get("allocatable_gib"),
        "memory_gib_free": memory.get("free_gib"),
        "gpu_total": gpu_total,
        "gpu_models": gpu_models,
        "gpu_memory_total_gib": round(gpu_memory_total_gib, 2),
        "gpu_operator_installed": snapshot.get("gpu_operator_installed"),
        "visible_namespaces": namespaces.get("names", []),
        "storage_classes": storage.get("names", []),
        "default_storage_class": snapshot.get("default_storage_class"),
        "app_store_crd_installed": snapshot.get("app_store_crd_installed"),
        "app_store_cluster_init_present": snapshot.get("app_store_cluster_init_present"),
        "app_store_crs": snapshot.get("app_store_crs", []),
        "warnings": [w for w in warnings if w],
    }
```

### Pattern 3: Blueprint Catalog Scanning

**What:** Scan `BLUEPRINTS_DIR` for YAML files. Parse WekaAppStore manifests to extract blueprint metadata.
**When to use:** `list_blueprints` and `get_blueprint` tools.

Blueprints in `weka/warp-blueprints` are WekaAppStore YAML documents with `spec.appStack.components[]` containing Helm chart references. The example `blueprint-default-values.yaml` in `weka-csi-config/` shows that blueprint metadata (names, categories, resource minimums) is NOT embedded in the manifest — it comes from supplemental Helm values files. The metadata registry approach (a `registry.yaml` sidecar in `BLUEPRINTS_DIR`) is the right call for fields like `category`, `gpu_minimum`, `description` that aren't in the WekaAppStore spec.

```python
# Source: Pattern derived from existing FastAPI app blueprint scanning
import os
import yaml
from pathlib import Path

BLUEPRINTS_DIR = os.environ.get("BLUEPRINTS_DIR", "/app/manifests/manifest")

def scan_blueprints(blueprints_dir: str) -> list[dict]:
    """Scan BLUEPRINTS_DIR for WekaAppStore YAML manifests."""
    results = []
    path = Path(blueprints_dir)
    if not path.exists():
        return results
    for yaml_file in sorted(path.glob("**/*.yaml")):
        try:
            with open(yaml_file) as f:
                docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if (
                    isinstance(doc, dict)
                    and doc.get("apiVersion", "").startswith("warp.io")
                    and doc.get("kind") == "WekaAppStore"
                ):
                    results.append({"source_file": str(yaml_file), "manifest": doc})
        except Exception:
            pass
    return results
```

### Pattern 4: CRD Schema via Live Cluster Read

**What:** Read `wekaappstores.warp.io` CRD via `apiextensions_api.read_custom_resource_definition()`.
**When to use:** `get_crd_schema` tool.

This pattern is already present in `cluster.py` — the apiextensions call at line 431. Extend it to return the full `openAPIV3Schema` from the CRD spec, plus 1-2 example documents from the blueprints directory.

```python
# Source: Pattern from app-store-gui/webapp/inspection/cluster.py line 431
from kubernetes import client

def get_crd_schema_raw(apiextensions_api=None) -> dict:
    apiextensions_api = apiextensions_api or client.ApiextensionsV1Api()
    crd = apiextensions_api.read_custom_resource_definition("wekaappstores.warp.io")
    versions = crd.spec.versions or []
    schema = {}
    for version in versions:
        if version.storage:
            schema = (version.schema.open_apiv3_schema or {})
            break
    return {
        "group": crd.spec.group,
        "version": next((v.name for v in versions if v.storage), None),
        "kind": crd.spec.names.kind if crd.spec.names else None,
        "schema": schema,
    }
```

### Anti-Patterns to Avoid

- **Exposing `inspection_snapshot` in MCP responses:** The v1 planning code embeds a full snapshot inside `fit_findings.inspection_snapshot`. This nested structure must never appear in MCP tool output — flatten everything.
- **Using `print()` anywhere in server code:** Writes to stdout, corrupts JSON-RPC stream. Use `logging.info()` with stderr handler.
- **Calling `config.load_kube_config()` at import time:** `cluster.py` already avoids this with injectable API clients. Keep that pattern — import-time K8s calls will fail in test context.
- **Raising unhandled exceptions from tools:** FastMCP converts unhandled exceptions to `isError: true` tool results, but the message may leak internal details. Catch K8s `ApiException` and return structured error in the flat response with a `warnings` entry instead.
- **Hardcoding blueprint names:** Scanner must be directory-driven, not hardcoded.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MCP protocol framing | Custom JSON-RPC implementation | `mcp[cli]` FastMCP | Protocol compliance is complex; `mcp dev` only works with official SDK |
| Tool schema generation | Manual `inputSchema` JSON | FastMCP type hint extraction | Docstring + type hints auto-generates correct JSON Schema |
| Stdio/stderr isolation | Manual pipe management | `mcp.run()` default + `logging.basicConfig(stream=sys.stderr)` | SDK handles framing; only need to configure logging destination |
| K8s API client management | Custom K8s auth code | Existing injectable pattern from `cluster.py` | Already handles kubeconfig and in-cluster auth via `KUBERNETES_AUTH_MODE` env var |

**Key insight:** The inspection layer is already built and injectable. The phase work is (a) thin MCP tool functions that call existing collectors, (b) flatten functions that strip domain wrappers, and (c) a blueprint scanner. There is no new K8s or WEKA client code needed.

---

## Common Pitfalls

### Pitfall 1: FastMCP Configures Logging on `__init__`
**What goes wrong:** If `logging.basicConfig(stream=sys.stderr)` is called AFTER `FastMCP(...)` instantiation, the SDK may have already added handlers that write to stderr with a different format, or may have called `basicConfig` itself, which is a no-op on subsequent calls.
**Why it happens:** FastMCP v1.x configures the logging ecosystem on `__init__()`. This is a documented issue (GitHub issue #1656 in python-sdk).
**How to avoid:** Configure `logging.basicConfig(stream=sys.stderr, ...)` as the FIRST statement in `server.py`, before any imports that trigger FastMCP init.
**Warning signs:** Log messages appearing on stdout during `mcp dev` testing.

### Pitfall 2: `flatten_cluster_status()` Retains Nested Snapshot
**What goes wrong:** The existing `flatten_cluster_status()` in `cluster.py` returns `inspection_snapshot` as a top-level key containing the full nested v1 snapshot. If reused directly as the MCP response, it violates the "2 key traversals" constraint.
**Why it happens:** `flatten_cluster_status()` was designed for the FastAPI planning layer, not for agent-facing output.
**How to avoid:** Write a separate `flatten_inspect_cluster_for_mcp()` function that does NOT include `inspection_snapshot` and aggregates GPU inventory into `gpu_total`/`gpu_models`/`gpu_memory_total_gib`.
**Warning signs:** Agent response contains `inspection_snapshot` key with nested `domains` object.

### Pitfall 3: Blueprint Directory Missing at Startup
**What goes wrong:** MCP server starts before git-sync has populated `/app/manifests/manifest`. `list_blueprints` returns empty list, or tools error.
**Why it happens:** Container startup race condition between git-sync sidecar and MCP server process.
**How to avoid:** Blueprint tools should return empty catalog (not error) if `BLUEPRINTS_DIR` is absent or empty. Add a `warnings` entry: `"blueprints_dir_empty: BLUEPRINTS_DIR not found or empty"`.
**Warning signs:** `list_blueprints` returns `{"blueprints": [], "warnings": ["..."]}` in healthy output.

### Pitfall 4: `mcp dev server.py` Only Works with FastMCP
**What goes wrong:** Using the low-level `Server` API (not `FastMCP`) means `mcp dev` and `mcp run` CLI tools are not available.
**Why it happens:** MCP CLI tool only understands FastMCP-based servers.
**How to avoid:** Always use `from mcp.server.fastmcp import FastMCP` and `mcp.run()`. Never use the low-level `mcp.server.Server` class for this project.
**Warning signs:** `mcp dev server.py` fails with import or attribute errors.

### Pitfall 5: Tool Descriptions Missing Sequencing Guidance
**What goes wrong:** Agents call tools in wrong order (e.g., `get_blueprint` before knowing which blueprints exist, or `apply` before `validate_yaml`).
**Why it happens:** Tool descriptions that only say "what" the tool does, not "when" or "before/after what".
**How to avoid:** Every tool docstring MUST open with a "when and why" sentence. Example: "Call this tool FIRST before selecting a blueprint. Returns..." and "Call this after `list_blueprints` to get full detail for a specific blueprint name."
**Warning signs:** Tool description does not contain the word "before", "after", or "first" anywhere.

---

## Code Examples

### Server Entry Point
```python
# Source: Official MCP Python SDK docs + FastMCP README pattern
import logging
import sys

# Must configure logging BEFORE FastMCP import/init
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from mcp.server.fastmcp import FastMCP
from tools.inspect_cluster import register_inspect_cluster
from tools.inspect_weka import register_inspect_weka
from tools.blueprints import register_blueprint_tools
from tools.crd_schema import register_crd_schema

mcp = FastMCP("weka-app-store-mcp")

register_inspect_cluster(mcp)
register_inspect_weka(mcp)
register_blueprint_tools(mcp)
register_crd_schema(mcp)

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
```

### Tool with Error Handling (non-raising pattern)
```python
# Source: Pattern from gofastmcp.com/servers/tools + project inspection layer
from kubernetes.client.rest import ApiException

@mcp.tool()
def inspect_cluster() -> dict:
    """Call this tool FIRST when you need to understand what cluster resources are
    available. Returns a flat snapshot of CPU, memory, GPU devices, namespaces, and
    storage classes. Use before calling list_blueprints to determine which blueprints
    fit available resources. Sequencing: inspect_cluster -> list_blueprints ->
    get_blueprint -> (Phase 7) validate_yaml -> apply.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        from webapp.inspection.cluster import collect_cluster_inspection
        snapshot = collect_cluster_inspection()
        return flatten_inspect_cluster_for_mcp(snapshot)
    except ApiException as exc:
        logger.warning("K8s API error in inspect_cluster: %s", exc)
        return {
            "captured_at": _utc_now(),
            "warnings": [f"K8s API unavailable: {exc.status} {exc.reason}"],
            "k8s_version": None,
            "cpu_nodes": None,
            "gpu_nodes": None,
            # ... all other fields None
        }
```

### Running `mcp dev` for Testing
```bash
# From mcp-server/ directory
mcp dev server.py

# This starts the MCP Inspector UI at localhost:6274
# Lists all tools and allows manual invocation with mocked responses
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Low-level `Server` class + manual handler registration | `FastMCP` decorator API | Early 2024 (SDK v0.9+) | `mcp dev` CLI only works with FastMCP |
| SSE transport as default | Stdio as default for local/single-operator | MCP spec 2024 | No HTTP server needed for OpenClaw integration |
| Unstructured text-only tool responses | `structuredContent` alongside text (MCP spec 2025-06-18) | June 2025 spec update | Agents can receive typed JSON directly, not parse text |

**Deprecated/outdated:**
- `mcp.server.Server` direct instantiation: still works but loses `mcp dev` and `mcp run` CLI tooling
- `json_response=True` FastMCP init parameter: older pattern; returning `dict` from tools now automatically produces `structuredContent` in SDK 1.x

---

## Open Questions

1. **Blueprint manifest format in `weka/warp-blueprints`**
   - What we know: The repo URL is `https://github.com/weka/warp-blueprints/tree/main/manifests`. The operator chart CRD defines the `WekaAppStore` spec. The `blueprint-default-values.yaml` in `weka-csi-config/` is a large Helm values file, not a WekaAppStore manifest.
   - What's unclear: What the actual WekaAppStore YAML files in `warp-blueprints/manifests` look like — are they raw WekaAppStore CRs, or Helm charts, or something else? This determines the parser strategy.
   - Recommendation: The planner should include a Wave 0 task to `git clone` or `curl` one manifest from the real repo to confirm format before writing the blueprint scanner. If manifests are raw WekaAppStore CRs, parse directly. If they're Helm charts, a supplemental metadata registry is definitely required.

2. **PYTHONPATH sharing between containers**
   - What we know: CONTEXT.md says "Imports reusable code via shared PYTHONPATH". The inspection modules live in `app-store-gui/webapp/`.
   - What's unclear: Whether the Dockerfile for the MCP server will mount `app-store-gui/` as a volume, install it as a package, or use a `sys.path` hack.
   - Recommendation: Use `COPY app-store-gui/ /app/webapp/` in the MCP server Dockerfile and set `ENV PYTHONPATH=/app`. This keeps the pattern clean and testable.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | None detected at repo root — tests run from `app-store-gui/` |
| Quick run command | `cd app-store-gui && python -m pytest tests/ -x -q` |
| Full suite command | `cd app-store-gui && python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCPS-01 | FastMCP server instantiates and lists 5 tools | unit | `pytest mcp-server/tests/test_server.py::test_tool_list -x` | Wave 0 |
| MCPS-02 | `inspect_cluster` returns flat JSON, all fields <= 2 traversals, `captured_at` present | unit | `pytest mcp-server/tests/test_inspect_cluster.py::test_flat_response -x` | Wave 0 |
| MCPS-03 | `inspect_weka` returns flat WEKA JSON, `captured_at` present | unit | `pytest mcp-server/tests/test_inspect_weka.py::test_flat_response -x` | Wave 0 |
| MCPS-04 | `list_blueprints` returns list with metadata per blueprint | unit | `pytest mcp-server/tests/test_blueprints.py::test_list_blueprints -x` | Wave 0 |
| MCPS-05 | `get_blueprint` returns full detail for known name, error shape for unknown | unit | `pytest mcp-server/tests/test_blueprints.py::test_get_blueprint -x` | Wave 0 |
| MCPS-06 | `get_crd_schema` returns schema dict with `kind`, `group`, `schema` keys | unit | `pytest mcp-server/tests/test_crd_schema.py::test_get_crd_schema -x` | Wave 0 |
| MCPS-10 | No response contains keys requiring >2 traversals | unit (loop over all tool outputs) | `pytest mcp-server/tests/test_response_depth.py -x` | Wave 0 |
| MCPS-11 | No tool writes to stdout; all logging goes to stderr | unit (capture stdout/stderr) | `pytest mcp-server/tests/test_logging.py::test_no_stdout -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cd app-store-gui && python -m pytest tests/ -x -q`
- **Per wave merge:** `cd app-store-gui && python -m pytest tests/ -v` (plus `mcp dev server.py` smoke check)
- **Phase gate:** All existing tests green + all new MCP server tests green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `mcp-server/tests/__init__.py` — test package init
- [ ] `mcp-server/tests/conftest.py` — shared mocked K8s API fixtures (mirrors `app-store-gui/tests/conftest.py` topology mocks)
- [ ] `mcp-server/tests/test_server.py` — tool list smoke test
- [ ] `mcp-server/tests/test_inspect_cluster.py` — flat output contract tests
- [ ] `mcp-server/tests/test_inspect_weka.py` — flat output contract tests
- [ ] `mcp-server/tests/test_blueprints.py` — scanner + list + get tests
- [ ] `mcp-server/tests/test_crd_schema.py` — schema shape tests
- [ ] `mcp-server/tests/test_response_depth.py` — depth-2 contract enforcer
- [ ] `mcp-server/tests/test_logging.py` — stderr-only contract test
- [ ] `mcp-server/requirements.txt` — `mcp[cli]>=1.26.0`, `kubernetes>=27.0.0`, `PyYAML>=6.0.1`, `pytest>=8.0.0`

---

## Sources

### Primary (HIGH confidence)
- Official MCP Python SDK docs (modelcontextprotocol.io/docs/develop/build-server) — FastMCP tool registration, stdio logging requirements, `mcp.run()` pattern
- PyPI `mcp` package page (pypi.org/project/mcp/) — version 1.26.0 confirmed, `mcp[cli]` install syntax
- gofastmcp.com/servers/tools — FastMCP `@mcp.tool` decorator, dict return type, error handling
- Official MCP concepts/tools spec (modelcontextprotocol.io/docs/concepts/tools) — tool schema format, `structuredContent`, error handling protocol
- Project source code: `app-store-gui/webapp/inspection/cluster.py` — `collect_cluster_inspection()`, `flatten_cluster_status()` confirmed injectable
- Project source code: `app-store-gui/webapp/inspection/weka.py` — `collect_weka_inspection()` confirmed injectable
- Project source code: `app-store-gui/webapp/planning/apply_gateway.py` — K8s apply pattern
- Project source code: `weka-app-store-operator-chart/templates/crd.yaml` — full `WekaAppStore` openAPIV3Schema

### Secondary (MEDIUM confidence)
- GitHub issue modelcontextprotocol/python-sdk#1656 — FastMCP logging init side effect, confirmed real pitfall
- `weka-csi-config/blueprint-default-values.yaml` — example blueprint Helm values structure (confirms supplemental metadata registry needed)

### Tertiary (LOW confidence)
- Inferred blueprint manifest format from CRD schema — actual `warp-blueprints` repo manifests not directly inspected (flagged as Open Question 1)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — mcp 1.26.0 version confirmed from PyPI, FastMCP API confirmed from official docs
- Architecture: HIGH — existing inspection code structure confirmed by direct read, flatten pattern confirmed
- Blueprint parsing: MEDIUM — CRD schema confirmed, actual manifest files in `warp-blueprints` repo not inspected directly
- Pitfalls: HIGH — logging pitfall confirmed from GitHub issue; others confirmed from docs and code inspection

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (mcp SDK evolves quickly; re-verify if >30 days pass before planning completes)
