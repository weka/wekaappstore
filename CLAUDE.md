# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Syntax validation
```bash
python -m py_compile app-store-gui/webapp/main.py operator_module/main.py
```

### Helm chart
```bash
helm lint weka-app-store-operator-chart
helm template weka-app-store ./weka-app-store-operator-chart > /tmp/rendered.yaml
```

### MCP server tests
```bash
PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest mcp-server/tests/ -v
```

### Operator tests
```bash
PYTHONPATH=operator_module pytest operator_module/tests/ -v
```

### Run a single test
```bash
PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest mcp-server/tests/test_blueprints.py::test_name -v
```

### MCP server smoke test (requires Docker)
```bash
docker build -f mcp-server/Dockerfile . -t weka-app-store-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  docker run --rm -i -e BLUEPRINTS_DIR=/tmp weka-app-store-mcp
```

### Publish new chart version
```bash
# 1. Bump version in weka-app-store-operator-chart/Chart.yaml
helm package weka-app-store-operator-chart -d docs/
helm repo index docs --url https://weka.github.io/wekaappstore
# Commit and push docs/
```

## Architecture

This repo contains three distinct runtimes packaged together into a single Helm chart (`weka-app-store-operator-chart`).

### Three runtimes

**1. Kubernetes Operator** (`operator_module/main.py`)
- Kopf-based controller watching `WekaAppStore` custom resources (`warp.io/v1alpha1`).
- Reconciles by calling `helm` and `kubectl` via subprocess — there is no Helm SDK.
- Main class is `HelmOperator`; entry points are `create_warrpappstore_function`, `update_warrpappstore_function`, `delete_warrpappstore_function`.
- Raises `kopf.PermanentError` for bad specs (no retry) and `kopf.TemporaryError` for transient failures (auto-retried).

**2. Web GUI** (`app-store-gui/webapp/main.py`)
- FastAPI + Jinja2 UI that lets users browse blueprints and submit deployments.
- Applying a blueprint creates/updates a `WekaAppStore` CR, which the operator then reconciles.
- Blueprint YAML files are NOT stored in this repo — they come from an external `warp-blueprints` git repo synced at runtime via `git-sync` into `BLUEPRINTS_DIR`.
- `ClusterInitMiddleware` gates all routes until the cluster is initialized.

**3. MCP Server** (`mcp-server/`)
- `fastmcp`-based stdio/HTTP MCP server exposing 8 tools to AI agents (OpenClaw/NemoClaw): `inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`, `validate_yaml`, `apply`, `status`.
- Each tool is registered via `register_*()` functions in `mcp-server/tools/`. Add tools there, not in `server.py`.
- **Critical:** `logging.basicConfig` must be called before `FastMCP` is imported (fastmcp SDK issue #1656). Never use `print()` in any MCP tool — it corrupts the stdio protocol stream.
- Transport is selected by `MCP_TRANSPORT` env var: `stdio` (default) or `http`.

### Custom Resource: `WekaAppStore`
Defined in `weka-app-store-operator-chart/templates/crd.yaml`. Three deployment modes:
1. `appStack.components[]` — multi-component ordered deployment with readiness checks (primary path)
2. Single `helmChart` — one Helm release
3. Legacy pod mode

Variable substitution: `${VAR}` tokens in `kubernetesManifest` strings and `valuesFiles` content are substituted from `spec.appStack.variables` at reconcile time. Single-pass only; `${namespace}` auto-defaults to `metadata.namespace`. Undefined vars raise `PermanentError`.

### Key design patterns
- Both `operator_module/main.py` and `app-store-gui/webapp/main.py` are large single-file modules by design. Match this style — don't introduce new sub-packages unless the task explicitly requires refactoring.
- `PYTHONPATH` must include both `mcp-server/` and `app-store-gui/` when running MCP tests, because the MCP tools import inspection helpers from the GUI module.
- No linter or formatter is enforced. Follow PEP 8 and match the existing module's quote style (GUI uses double quotes, operator uses single quotes for shell args).
- All Docker images push to `wekachrisjen/` on Docker Hub. CI (`.github/workflows/mcp-server.yml`) builds and pushes on `v*` tags.

### Helm chart publishing
Packaged chart archives live in `docs/` alongside `docs/index.yaml`. GitHub Pages serves this as the Helm repo at `https://weka.github.io/wekaappstore`. After packaging a new version, always re-run `helm repo index docs/` before committing.
