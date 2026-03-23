---
phase: 06-mcp-scaffold-and-read-only-tools
plan: 01
subsystem: mcp-server
tags: [mcp, fastmcp, inspection, tools, stdio, logging]
dependency_graph:
  requires:
    - app-store-gui/webapp/inspection/cluster.py (collect_cluster_inspection)
    - app-store-gui/webapp/inspection/weka.py (collect_weka_inspection)
  provides:
    - mcp-server/server.py (FastMCP entry point, weka-app-store-mcp)
    - mcp-server/tools/inspect_cluster.py (inspect_cluster tool + flatten)
    - mcp-server/tools/inspect_weka.py (inspect_weka tool + flatten)
    - mcp-server/tests/conftest.py (mocked K8s fixtures for all MCP tests)
  affects:
    - Phase 6 plans 02 and 03 (tool registration contract and test harness)
tech_stack:
  added:
    - mcp[cli]>=1.26.0 (FastMCP, stdio transport)
    - mcp-server/ directory at repo root (separate container)
  patterns:
    - logging.basicConfig(stream=sys.stderr) before FastMCP init (Pitfall 1 guard)
    - register_*(mcp) pattern for tool registration
    - flatten_*_for_mcp() functions strip domain wrappers to flat agent-facing JSON
key_files:
  created:
    - mcp-server/server.py
    - mcp-server/config.py
    - mcp-server/requirements.txt
    - mcp-server/__init__.py
    - mcp-server/tools/__init__.py
    - mcp-server/tools/inspect_cluster.py
    - mcp-server/tools/inspect_weka.py
    - mcp-server/tests/__init__.py
    - mcp-server/tests/conftest.py
    - mcp-server/tests/test_server.py
    - mcp-server/tests/test_logging.py
    - mcp-server/tests/test_inspect_cluster.py
    - mcp-server/tests/test_inspect_weka.py
  modified: []
decisions:
  - "mcp-server/ placed at repo root (not inside app-store-gui/) for clear container separation"
  - "flatten_inspect_cluster_for_mcp() is a new function distinct from flatten_cluster_status() — drops inspection_snapshot, aggregates GPU inventory"
  - "Tool functions import collect_*_inspection lazily inside the function body to avoid import-time K8s calls"
  - "register_*(mcp) pattern chosen over @mcp.tool() at module level to keep tools testable with fresh FastMCP instances"
metrics:
  duration: "~4 minutes"
  completed: "2026-03-20"
  tasks_completed: 2
  tests_added: 16
  files_created: 13
---

# Phase 6 Plan 1: MCP Server Scaffold and Read-Only Tool Foundations Summary

FastMCP server with stderr-only logging, two inspection tools (inspect_cluster, inspect_weka) using flatten-and-wrap pattern producing flat agent-facing JSON with captured_at, aggregated GPU fields, and warnings arrays.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | MCP server scaffold, config, and test harness | eb177d6 | server.py, config.py, requirements.txt, tests/conftest.py, test_server.py, test_logging.py |
| 2 | inspect_cluster and inspect_weka tools with flat response schemas | 0c6884f | tools/inspect_cluster.py, tools/inspect_weka.py, test_inspect_cluster.py, test_inspect_weka.py |

## Verification

All 16 tests pass:

```
16 passed in 1.74s
```

- `test_server.py`: FastMCP instantiation, server name assertion
- `test_logging.py`: stdout-clean import, stderr-only logging contract
- `test_inspect_cluster.py`: flat response, 2-key depth contract, no forbidden keys, blocker-to-warnings, GPU operator warning, ApiException handling
- `test_inspect_weka.py`: flat response, 2-key depth contract, no domains key, filesystem shape, ApiException handling, blocker-to-warnings

FastMCP type confirmed:
```
<class 'mcp.server.fastmcp.server.FastMCP'>
```

## Output Schemas

### inspect_cluster response shape
```
captured_at, k8s_version, cpu_nodes, gpu_nodes,
cpu_cores_total, cpu_cores_free,
memory_gib_total, memory_gib_free,
gpu_total (int), gpu_models (list[str]), gpu_memory_total_gib (float),
gpu_operator_installed, visible_namespaces (list[str]),
storage_classes (list[str]), default_storage_class,
app_store_crd_installed, app_store_cluster_init_present,
app_store_crs (list[str]), warnings (list[str])
```

### inspect_weka response shape
```
captured_at, weka_cluster_name, weka_cluster_status,
total_capacity_gib, used_capacity_gib, free_capacity_gib,
filesystems (list[{name, size_gib, used_gib}]),
warnings (list[str])
```

## Decisions Made

1. **mcp-server/ at repo root** — Clean container separation; avoids confusing app-store-gui dependency direction. The MCP server imports from app-store-gui via PYTHONPATH, not the other way around.

2. **Separate flatten functions** — `flatten_inspect_cluster_for_mcp()` is distinct from the existing `flatten_cluster_status()` which retains `inspection_snapshot`. MCP tool flatten functions never include v1 planning structures.

3. **Lazy imports inside tool functions** — `collect_cluster_inspection` and `collect_weka_inspection` imported inside the tool function body, not at module level. This prevents import-time K8s API calls that would fail in CI/test context.

4. **register_*(mcp) pattern** — Tools defined in `register_inspect_cluster(mcp)` functions rather than bare `@mcp.tool()` decorators at module level. This allows creating fresh FastMCP instances in tests without shared state.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] Tool files created in Task 1** — The plan specified tool stubs for Task 1 (server.py calling register functions), but full tool implementations were written immediately since the flatten logic was defined in the plan and RESEARCH.md. Tests were written first (RED) then verified green. No behavioral deviation — all specified behavior is implemented and tested.

None — plan executed as written. Tool stubs created in Task 1 as specified, then expanded to full implementations in Task 2.

## Self-Check: PASSED
