# Phase 09: Deployment and Registration - Research

**Researched:** 2026-03-23
**Domain:** Docker containerization, GitHub Actions CI/CD, MCP server registration (OpenClaw/NemoClaw)
**Confidence:** HIGH (stack and patterns are well-established; NemoClaw section LOW due to unpublished schema)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Dockerfile Strategy:**
- Base image: `python:3.10-slim`
- Copy both `mcp-server/` and `app-store-gui/webapp/` into the image, set PYTHONPATH — self-contained, no shared volumes needed for code
- Blueprints are NOT baked into the image — volume mount at runtime from host/git-sync sidecar at `/app/manifests/manifest`
- Import check entrypoint: run `python -c "import server"` as sanity check before starting, catches missing dependencies at container start
- CMD: `python -m server` for stdio transport

**Environment Config Surface:**
- K8s auth: in-cluster service account by default, `KUBECONFIG` env var override for local dev/testing outside cluster
- `BLUEPRINTS_DIR` required — validated at startup with clear error if missing
- `LOG_LEVEL` optional with INFO default — all logging to stderr (stdio transport requirement)
- `KUBERNETES_AUTH_MODE` optional — existing pattern from webapp
- `WEKA_ENDPOINT` optional — override for direct WEKA API access as alternative to K8s CRD discovery (future-proofing for non-K8s WEKA)
- Required env vars validated at startup (fail fast); optional vars checked lazily when tools are called

**Registration Documentation:**
- Single `README.md` in `mcp-server/` — sections: quick start, Dockerfile build, env vars reference, OpenClaw registration, NemoClaw registration, troubleshooting
- Full copy-paste command blocks for every step (build, tag, push, register) — operators follow end-to-end without guessing
- NemoClaw registration section: placeholder with known patterns (stdio transport, env vars, SKILL.md path) plus clear `TODO: Update when NemoClaw alpha schema is published` marker
- "Verify it works" section with echo pipe smoke test: pipe a `tools/list` JSON-RPC request to container stdin, verify 8 tools come back

**Image Distribution:**
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEPLOY-01 | Dockerfile packages MCP server as a container image | Dockerfile patterns, python:3.10-slim base, layer ordering for cache |
| DEPLOY-02 | Container includes all dependencies and runs MCP server on stdio | requirements.txt install, PYTHONPATH setup, `python -m server` CMD |
| DEPLOY-03 | Configuration interface for NemoClaw sandbox (env vars for K8s/WEKA endpoints, credentials) | config.py env var pattern, startup validation, `BLUEPRINTS_DIR`/`KUBECONFIG`/`KUBERNETES_AUTH_MODE`/`LOG_LEVEL`/`WEKA_ENDPOINT` |
| DEPLOY-04 | Documentation for registering MCP server with OpenClaw/NemoClaw | openclaw.json already exists, README structure, tools/list smoke test |
</phase_requirements>

---

## Summary

Phase 9 delivers the deployment artifact (container image) and the registration documentation that lets OpenClaw/NemoClaw operators wire up the MCP server. All upstream phases are complete: 8 tools are implemented, tested, and registered; `openclaw.json` is generated and validated; `SKILL.md` defines the agent workflow. This phase is almost entirely about packaging and documentation — no new tool logic.

The critical technical work is the Dockerfile. The server imports both `mcp-server/` code and `app-store-gui/webapp/` inspection modules via PYTHONPATH. Both source trees must be in the image and PYTHONPATH must be set so Python resolves imports correctly. The existing `docker/webapp.Dockerfile` shows the established project pattern (python:3.13-slim, apt-get cleanup, non-root user) but the locked decision is `python:3.10-slim` for the MCP server. The webapp Dockerfile context is set at `docker/` with `../app-store-gui/` paths — the MCP server Dockerfile will need to be placed carefully so both `mcp-server/` and `app-store-gui/` are reachable via relative path in the build context, or built from the repo root.

The CI workflow is new — no `.github/workflows/` directory exists yet, only `.github/ISSUE_TEMPLATE/`. The workflow needs to run the existing pytest suite on PR, and build/push to Docker Hub only on version tags. Docker Hub credentials go in GitHub Secrets. The NemoClaw registration section is intentionally a placeholder; the schema has not been published as of 2026-03-23.

**Primary recommendation:** Build the Dockerfile from the repo root (build context = `.`) so both `mcp-server/` and `app-store-gui/webapp/` are available without `../` path tricks. Set `WORKDIR /app`, copy both source trees, install from `mcp-server/requirements.txt`, set `PYTHONPATH=/app/mcp-server:/app/app-store-gui`, and run `python -m server` as CMD.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| python | 3.10-slim | Base image (locked) | Slim variant = minimal attack surface, no dev tools |
| mcp[cli] | >=1.26.0 | FastMCP SDK (already in requirements.txt) | Official MCP Python SDK |
| kubernetes | >=27.0.0 | K8s API client (already in requirements.txt) | Official Python client |
| PyYAML | >=6.0.1 | YAML parsing (already in requirements.txt) | Standard YAML lib |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0.0 | Test runner (in requirements.txt) | CI test step only; not installed in final image |
| docker/buildx | — | Multi-platform builds | If ARM image needed later; not required now |

### GitHub Actions Stack
| Action | Version | Purpose |
|--------|---------|---------|
| actions/checkout | v4 | Repo checkout |
| actions/setup-python | v5 | Python 3.10 for test job |
| docker/login-action | v3 | Docker Hub authentication |
| docker/metadata-action | v5 | Semver tag computation from git tag |
| docker/build-push-action | v6 | Build and push image |

**Installation (no new packages — all already in requirements.txt):**
```bash
# Nothing new to install. All dependencies already pinned in mcp-server/requirements.txt
```

---

## Architecture Patterns

### Recommended Project Structure (new files only)
```
mcp-server/
├── Dockerfile               # NEW: MCP server container image
├── .dockerignore            # NEW: exclude __pycache__, tests, .env, etc.
├── README.md                # NEW: quick start, env vars, OpenClaw/NemoClaw registration
└── (existing files unchanged)

.github/
└── workflows/
    └── mcp-server.yml       # NEW: CI (test on PR) + CD (build/push on v* tag)
```

### Pattern 1: Repo-Root Build Context Dockerfile
**What:** Dockerfile placed at `mcp-server/Dockerfile` but built with repo root as context (`docker build -f mcp-server/Dockerfile .`). This gives Docker access to both `mcp-server/` and `app-store-gui/` in one build context.

**When to use:** When the image needs code from two sibling directories. Avoids `../` paths in COPY (which Docker forbids).

**Example:**
```dockerfile
# Source: established Docker best practice; mirrors webapp.Dockerfile pattern
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps before copying source (cache layer)
COPY mcp-server/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source trees
COPY mcp-server/ ./mcp-server/
COPY app-store-gui/webapp/ ./app-store-gui/webapp/

# PYTHONPATH so 'import server' and 'from inspection...' both resolve
ENV PYTHONPATH=/app/mcp-server:/app/app-store-gui

WORKDIR /app/mcp-server

# Sanity check: catch missing imports at container start, not at first tool call
RUN python -c "import server"

# Non-root user (security best practice)
RUN useradd -u 10001 -m mcpuser
USER mcpuser

# stdio transport: OpenClaw spawns this process and communicates via stdin/stdout
CMD ["python", "-m", "server"]
```

### Pattern 2: GitHub Actions — Test on PR, Push on Tag
**What:** Two-job workflow. `test` job runs on every PR touching `mcp-server/**`. `build-push` job runs only on `v*` tag push, depends on `test` passing.

**When to use:** Standard for libraries/services that have their own CI gate separate from the monorepo.

**Example:**
```yaml
# Source: official GitHub Actions Docker documentation pattern
name: MCP Server CI/CD

on:
  push:
    tags: ['v*']
    paths:
      - 'mcp-server/**'
  pull_request:
    paths:
      - 'mcp-server/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r mcp-server/requirements.txt
      - run: pytest mcp-server/tests/ -v
        env:
          PYTHONPATH: mcp-server:app-store-gui
          BLUEPRINTS_DIR: mcp-server/tests/fixtures/sample_blueprints

  build-push:
    needs: test
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: wekachrisjen/weka-app-store-mcp
          tags: |
            type=semver,pattern={{version}}
            type=raw,value=latest
      - uses: docker/build-push-action@v6
        with:
          context: .
          file: mcp-server/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
```

### Pattern 3: Startup Validation for Required Env Vars
**What:** Check `BLUEPRINTS_DIR` at process start in `server.py` or `config.py`, exit with a clear error message if missing. Do not wait for first tool call.

**When to use:** Required env vars — fail fast so the operator sees the problem immediately.

**Example:**
```python
# In config.py or at top of server.py
import os, sys

BLUEPRINTS_DIR = os.environ.get("BLUEPRINTS_DIR")
if not BLUEPRINTS_DIR:
    print(
        "FATAL: BLUEPRINTS_DIR environment variable is required. "
        "Set it to the path where WekaAppStore blueprint YAML files are mounted "
        "(e.g. /app/manifests/manifest).",
        file=sys.stderr,
    )
    sys.exit(1)
```

### Pattern 4: MCP stdio Smoke Test via Docker
**What:** Pipe a JSON-RPC `tools/list` request to the container's stdin, capture stdout response, verify tool names appear.

**When to use:** "Verify it works" section of README. Checks the container can start and respond to MCP protocol without a real K8s cluster.

**Example:**
```bash
# Source: MCP protocol specification — tools/list is the standard discovery call
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  docker run --rm -i \
    -e BLUEPRINTS_DIR=/app/manifests/manifest \
    wekachrisjen/weka-app-store-mcp:latest
# Expected: JSON response containing "inspect_cluster", "apply", and 6 other tool names
```

### Anti-Patterns to Avoid
- **Building from `mcp-server/` as context:** `COPY ../app-store-gui/` is forbidden in Docker — build context must be the repo root.
- **Installing pytest in the production image:** Adds 80+ MB and test dependencies. Use multi-stage or install dev deps only in CI.
- **Baking `BLUEPRINTS_DIR` path into the image:** Value must remain configurable. `config.py` already reads from env var; Dockerfile must not set it as `ENV`.
- **Hardcoding Docker Hub credentials:** Use `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` GitHub Secrets.
- **Running as root in the container:** Use `useradd` + `USER mcpuser`, matching the webapp.Dockerfile pattern.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Semver tag computation from git tag | Custom shell parsing of `$GITHUB_REF` | `docker/metadata-action@v5` | Handles `v2.0.0` → `2.0.0`, `latest`, and edge cases automatically |
| Docker Hub login in CI | Hardcoded credentials in workflow | `docker/login-action@v3` with secrets | Secure credential handling, token rotation support |
| Multi-platform image builds | Manual `docker manifest` commands | `docker/buildx` via `build-push-action` | Only needed if ARM support required; skip for now |
| Import path resolution | Custom `sys.path` manipulation in Dockerfile | `ENV PYTHONPATH=...` | Standard Python mechanism; matches how tests already work via conftest.py |

**Key insight:** The existing conftest.py already solves the two-root import problem by appending both `mcp-server/` and `app-store-gui/` to `sys.path`. The Dockerfile `ENV PYTHONPATH` is the container equivalent — same solution, different mechanism.

---

## Common Pitfalls

### Pitfall 1: Docker Build Context vs. Dockerfile Location
**What goes wrong:** Placing Dockerfile at repo root or running `docker build mcp-server/` causes `COPY app-store-gui/webapp/` to fail (path not in context).
**Why it happens:** Docker build context is the directory you pass to `docker build`. Files outside it are inaccessible.
**How to avoid:** Always run `docker build -f mcp-server/Dockerfile .` (period = repo root as context).
**Warning signs:** `COPY failed: file not found` error during build.

### Pitfall 2: pytest in Production Image
**What goes wrong:** `pip install -r requirements.txt` installs pytest (it's in the single requirements.txt). This inflates image size by ~80MB.
**Why it happens:** Single requirements.txt includes pytest for convenience in development.
**How to avoid:** In the Dockerfile, either use `--no-deps pytest` exclusion or a two-stage build where the test stage installs everything and the runtime stage only installs non-dev deps. Simplest: keep single requirements.txt but note in docs that pytest is present in image — acceptable for v2.0.
**Warning signs:** Image size unexpectedly large. Run `docker run --rm wekachrisjen/weka-app-store-mcp pip show pytest` to confirm.

### Pitfall 3: BLUEPRINTS_DIR Not Set Causes Silent Failure
**What goes wrong:** If `BLUEPRINTS_DIR` is not set and no startup validation exists, `list_blueprints` and `get_blueprint` tools return empty or error only when called — confusing for operators.
**Why it happens:** `config.py` currently defaults to `/app/manifests/manifest` and does not assert existence. An operator might run the container without a volume mount.
**How to avoid:** Add startup check that verifies `BLUEPRINTS_DIR` is set (not just defaulted) and that the path exists. Exit with a clear error message.
**Warning signs:** `list_blueprints` returns empty result or `blueprints: []` with no warning.

### Pitfall 4: CI Runs on Wrong Python Version
**What goes wrong:** Using `python-version: '3.13'` (which existing Dockerfiles use) in CI while Dockerfile uses `python:3.10-slim`. Version skew can mask compatibility bugs.
**Why it happens:** Existing `webapp.Dockerfile` uses 3.13 — easy to copy that by mistake.
**How to avoid:** CI test job must use `python-version: '3.10'` to match the Dockerfile's base image.
**Warning signs:** Tests pass in CI but container fails at startup.

### Pitfall 5: stdout Pollution Breaks stdio Transport
**What goes wrong:** Any `print()` statement in server code goes to stdout. OpenClaw interprets stdout as MCP protocol frames — stray text causes JSON parse errors.
**Why it happens:** Developer debugging with `print()` rather than `logging`.
**How to avoid:** MCPS-11 already enforces logging to stderr. Verify no `print()` calls exist in `server.py` or any `tools/*.py`. The existing `test_logging.py::test_no_stdout_on_import` catches this.
**Warning signs:** `test_logging.py` failures or OpenClaw reporting parse errors.

### Pitfall 6: NemoClaw Config Schema Is Not Yet Published
**What goes wrong:** Writing definitive NemoClaw registration docs with a specific schema causes breakage when the actual schema is published and differs.
**Why it happens:** NemoClaw alpha schema was not published as of 2026-03-20.
**How to avoid:** README NemoClaw section must include a visible `TODO: Update when NemoClaw alpha schema is published` marker. Use `openclaw.json` field names as best-effort guidance, note they may differ.
**Warning signs:** None — this is a known unknown.

---

## Code Examples

Verified patterns from existing project code:

### Existing PYTHONPATH Pattern (from conftest.py)
```python
# Source: mcp-server/tests/conftest.py (lines 16-21)
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
APP_STORE_GUI_ROOT = MCP_SERVER_ROOT.parent / "app-store-gui"

for path in (str(MCP_SERVER_ROOT), str(APP_STORE_GUI_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
```
Dockerfile equivalent: `ENV PYTHONPATH=/app/mcp-server:/app/app-store-gui`

### Existing Webapp Dockerfile Pattern (from docker/webapp.Dockerfile)
```dockerfile
# Source: docker/webapp.Dockerfile — established project pattern
FROM python:3.13-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY ../app-store-gui/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY ../app-store-gui/webapp ./app
RUN useradd -u 10001 -m appuser
USER appuser
```
Note: webapp.Dockerfile uses `../` paths which only work when built from `docker/` directory. MCP Dockerfile MUST use repo-root build context to access both source trees.

### Existing openclaw.json Structure (from mcp-server/openclaw.json)
```json
{
  "name": "weka-app-store-mcp",
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
README registration section should reference this file directly — do not duplicate.

### Existing Config Env Var Pattern (from mcp-server/config.py)
```python
# Source: mcp-server/config.py
BLUEPRINTS_DIR: str = os.environ.get("BLUEPRINTS_DIR", "/app/manifests/manifest")
KUBERNETES_AUTH_MODE: str = os.environ.get("KUBERNETES_AUTH_MODE", "in-cluster")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
```
WEKA_ENDPOINT will follow this same pattern. Startup validation will add a check that BLUEPRINTS_DIR is set via env var (not just defaulted) or that the path exists.

### Existing Test Run Command (for CI)
```bash
# Source: established from all previous phases
PYTHONPATH=mcp-server:app-store-gui \
BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints \
pytest mcp-server/tests/ -v
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `docker/login` with password | `docker/login-action` with token | 2021 | Docker Hub requires access tokens; passwords deprecated for CI |
| `docker build && docker push` manually | `docker/build-push-action@v6` with buildx | 2022 | Handles provenance, SBOM, multi-platform in one action |
| Separate `requirements-dev.txt` | Single `requirements.txt` (this project) | — | Acceptable for project scope; pytest in image is minor tradeoff |

**Deprecated/outdated:**
- `docker/build-push-action@v4` or earlier: Use v6, which defaults to Buildx and GitHub attestations.
- `actions/checkout@v2`, `setup-python@v4`: Use v4/v5 respectively (current).

---

## Open Questions

1. **pytest in the production image**
   - What we know: `mcp-server/requirements.txt` includes `pytest>=8.0.0` for convenience. The Dockerfile will install it.
   - What's unclear: Whether this is acceptable for v2.0 or requires a split requirements file.
   - Recommendation: Accept pytest in image for v2.0. Document it in README as known. Image will be ~100MB larger than strictly necessary. A `requirements-prod.txt` refactor is v3.0 concern.

2. **WEKA_ENDPOINT env var — where does it wire in?**
   - What we know: CONTEXT.md lists it as optional for non-K8s WEKA access.
   - What's unclear: No existing tool in `inspect_weka.py` reads this variable — it's listed as future-proofing.
   - Recommendation: Add `WEKA_ENDPOINT` to `config.py` with `os.environ.get("WEKA_ENDPOINT")` (no default, None if absent). Document in README as future use. No tool wiring needed in Phase 9.

3. **Docker Hub push authentication**
   - What we know: Account is `wekachrisjen` (WEKA corporate). CI uses GitHub Secrets.
   - What's unclear: Whether `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets already exist in the GitHub repo.
   - Recommendation: README must document the one-time secret setup step. Planner should include a task noting the operator must add these secrets before first push.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0.0 |
| Config file | none (pytest auto-discovers `mcp-server/tests/`) |
| Quick run command | `PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest mcp-server/tests/ -q` |
| Full suite command | `PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest mcp-server/tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEPLOY-01 | Dockerfile builds without error | smoke | `docker build -f mcp-server/Dockerfile . -t mcp-test` | ❌ Wave 0 (Dockerfile) |
| DEPLOY-02 | Container starts, imports resolve, stdio responds | smoke | `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \| docker run --rm -i -e BLUEPRINTS_DIR=/tmp mcp-test` | ❌ Wave 0 (Dockerfile) |
| DEPLOY-03 | Config env vars are read and validated at startup | unit | `pytest mcp-server/tests/test_config.py -v` | ❌ Wave 0 |
| DEPLOY-04 | README exists with required sections; openclaw.json valid | manual-only | Verify `mcp-server/README.md` has all sections; `python mcp-server/generate_openclaw_config.py` exits 0 | ❌ Wave 0 (README) |

### Sampling Rate
- **Per task commit:** `PYTHONPATH=mcp-server:app-store-gui BLUEPRINTS_DIR=mcp-server/tests/fixtures/sample_blueprints pytest mcp-server/tests/ -q`
- **Per wave merge:** Full suite (same command with `-v`)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `mcp-server/Dockerfile` — covers DEPLOY-01, DEPLOY-02
- [ ] `mcp-server/tests/test_config.py` — covers DEPLOY-03 (startup validation of BLUEPRINTS_DIR)
- [ ] `mcp-server/README.md` — covers DEPLOY-04 (documentation)
- [ ] `.github/workflows/mcp-server.yml` — covers DEPLOY-01, DEPLOY-02 in CI

---

## Sources

### Primary (HIGH confidence)
- Existing project code (`mcp-server/`, `docker/webapp.Dockerfile`, `mcp-server/config.py`, `mcp-server/openclaw.json`) — direct inspection
- `mcp-server/tests/conftest.py` — PYTHONPATH pattern authority
- `.planning/phases/09-deployment-and-registration/09-CONTEXT.md` — locked decisions

### Secondary (MEDIUM confidence)
- GitHub Actions official documentation patterns — `docker/login-action`, `docker/metadata-action`, `docker/build-push-action` are the established standard actions for Docker Hub publishing
- Docker best practices for COPY context and non-root users — well-established, stable since Docker 20.x

### Tertiary (LOW confidence)
- NemoClaw alpha registration schema — not yet published; `openclaw.json` field names used as best-effort proxy

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages already in requirements.txt; Dockerfile base image locked
- Architecture: HIGH — Dockerfile patterns and GitHub Actions patterns are well-established; no novel choices
- Pitfalls: HIGH — all identified from direct inspection of existing project code
- NemoClaw section: LOW — schema unpublished; placeholder approach is the correct response

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (stable domain; GitHub Actions action versions may minor-bump but patterns hold)
