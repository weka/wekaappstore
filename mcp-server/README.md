# WEKA App Store MCP Server

MCP server that provides OpenClaw/NemoClaw agents with tools to inspect Kubernetes clusters, browse WEKA App Store blueprint catalogs, validate manifests, apply blueprints, and monitor deployment status.

---

## Quick Start

Build the image and test locally with a tools/list probe:

```bash
docker build -f mcp-server/Dockerfile . -t weka-app-store-mcp
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  docker run --rm -i -e BLUEPRINTS_DIR=/app/manifests/manifest weka-app-store-mcp
```

Expected: a JSON response listing all 8 tool names (`inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`, `validate_yaml`, `apply`, `status`).

---

## Building the Image

Build, tag, and push a versioned release:

```bash
# Build
docker build -f mcp-server/Dockerfile . -t wekachrisjen/weka-app-store-mcp:v2.0.0

# Tag latest
docker tag wekachrisjen/weka-app-store-mcp:v2.0.0 wekachrisjen/weka-app-store-mcp:latest

# Push both tags
docker push wekachrisjen/weka-app-store-mcp:v2.0.0
docker push wekachrisjen/weka-app-store-mcp:latest
```

> The CI/CD pipeline (`.github/workflows/mcp-server.yml`) performs these steps automatically when you push a `v*` tag. Manual pushes are only needed for local testing.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `BLUEPRINTS_DIR` | **Yes** | — | Absolute path to directory containing blueprint YAML files. Server exits with FATAL error if unset. |
| `KUBECONFIG` | No | `None` | Path to kubeconfig file. Override for local development outside the cluster. |
| `KUBERNETES_AUTH_MODE` | No | `in-cluster` | Kubernetes auth mode. Use `kubeconfig` for local dev, `in-cluster` for pod-deployed server. |
| `LOG_LEVEL` | No | `INFO` | Python logging level. Set to `DEBUG` for verbose output to stderr. |
| `WEKA_ENDPOINT` | No | `None` | Reserved for future non-Kubernetes WEKA endpoint. Not used in current release. |

All variables are read at startup. `BLUEPRINTS_DIR` is validated immediately — the server will not start without it.

---

## OpenClaw Registration

The `mcp-server/openclaw.json` file contains the complete registration configuration for OpenClaw. Its contents:

```json
{
  "name": "weka-app-store-mcp",
  "description": "MCP server for the WEKA App Store...",
  "transport": "stdio",
  "startup": {
    "command": "python",
    "args": ["-m", "server"],
    "cwd": "mcp-server/"
  },
  "env": {
    "required": ["BLUEPRINTS_DIR"],
    "optional": ["KUBERNETES_AUTH_MODE", "LOG_LEVEL", "KUBECONFIG"]
  },
  "container": "weka-app-store-mcp:latest",
  "skill": "mcp-server/SKILL.md"
}
```

**To register:**

1. Copy `mcp-server/openclaw.json` to OpenClaw's MCP server config directory (typically `~/.openclaw/servers/` or the path specified in your OpenClaw config).
2. Update the `container` field to reference the exact image tag you deployed:
   ```json
   "container": "wekachrisjen/weka-app-store-mcp:v2.0.0"
   ```
3. The `skill` field points to `mcp-server/SKILL.md`. This file defines the 12-step agent workflow including mandatory validate-before-apply and re-inspect-before-apply rules. OpenClaw loads it automatically before calling any tool.
4. Reload OpenClaw's server registry. The server will appear as `weka-app-store-mcp`.

**Environment setup in OpenClaw:** Set `BLUEPRINTS_DIR` to the path where your blueprint YAML files are stored (e.g., `/opt/weka/blueprints`). All other variables are optional.

---

## NemoClaw Registration

NemoClaw uses the same stdio transport pattern as OpenClaw. The known configuration fields are:

```json
{
  "transport": "stdio",
  "command": "python",
  "args": ["-m", "server"],
  "cwd": "mcp-server/",
  "env": {
    "BLUEPRINTS_DIR": "<path-to-blueprint-yaml-files>"
  },
  "skill": "mcp-server/SKILL.md"
}
```

> **TODO:** Update this section when the NemoClaw alpha config schema is published. The fields above are best-effort — the exact registration format, config file location, and reload procedure may differ from OpenClaw.

---

## Verify It Works

After starting the server (via OpenClaw, NemoClaw, or directly), pipe a `tools/list` request to verify all 8 tools are registered:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  docker run --rm -i \
    -e BLUEPRINTS_DIR=/tmp \
    wekachrisjen/weka-app-store-mcp:latest
```

Expected response contains all 8 tools: `inspect_cluster`, `inspect_weka`, `list_blueprints`, `get_blueprint`, `get_crd_schema`, `validate_yaml`, `apply`, `status`.

The server starts without error if `BLUEPRINTS_DIR` is set (even `/tmp` is valid for the smoke test — it will return an empty blueprint list rather than failing).

---

## CI/CD

The `.github/workflows/mcp-server.yml` pipeline runs automatically:

| Trigger | Jobs run |
|---|---|
| Pull request touching `mcp-server/**` | `test` only |
| Push to `mcp-server/**` | `test` only |
| Push a `v*` tag (e.g., `v2.0.1`) | `test`, then `build-push` |

The `build-push` job only runs after `test` passes. It pushes to [`wekachrisjen/weka-app-store-mcp`](https://hub.docker.com/r/wekachrisjen/weka-app-store-mcp) with both a versioned tag and `latest`.

**One-time GitHub Secrets setup** (repo Settings > Secrets and variables > Actions):

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (generate at hub.docker.com > Security > Access Tokens) |

---

## Troubleshooting

**`FATAL: BLUEPRINTS_DIR environment variable is required`**

The server exited at startup. Set `BLUEPRINTS_DIR` to an absolute path containing your blueprint YAML files:
```bash
-e BLUEPRINTS_DIR=/path/to/blueprints
```

**`ModuleNotFoundError` or `ImportError` at startup**

Check that `PYTHONPATH` includes both `mcp-server` and `app-store-gui` directories. In the Docker image this is set at build time. For local dev:
```bash
PYTHONPATH=mcp-server:app-store-gui python -m server
```

If running from the image, the Dockerfile sets `PYTHONPATH` correctly — a failed import likely means the image was not built from the repo root:
```bash
# Always build from repo root (not from mcp-server/)
docker build -f mcp-server/Dockerfile .
```

**No tools in `tools/list` response / stdout corruption breaking stdio**

The MCP stdio transport uses stdout exclusively for JSON-RPC messages. Any `print()` statement in server code writes to stdout and corrupts the protocol stream. The server uses `logging` (stderr) for all diagnostics. If you add custom code, use `logging.getLogger(__name__).info(...)` — never `print()`.

**`list_blueprints` returns empty catalog**

`BLUEPRINTS_DIR` is set but the server found no YAML files. Verify:
```bash
ls $BLUEPRINTS_DIR/*.yaml $BLUEPRINTS_DIR/*.yml 2>/dev/null
```
If running in a container, confirm the volume mount path matches `BLUEPRINTS_DIR`:
```bash
-v /host/blueprints:/app/blueprints -e BLUEPRINTS_DIR=/app/blueprints
```
