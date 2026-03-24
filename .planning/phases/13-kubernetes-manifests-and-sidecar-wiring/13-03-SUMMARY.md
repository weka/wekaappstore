---
phase: 13-kubernetes-manifests-and-sidecar-wiring
plan: "03"
subsystem: infra
tags: [kubernetes, mcp, sidecar, openclaw, eks, docker, rbac, live-deployment, openclaw-json, multi-arch]

# Dependency graph
requires:
  - phase: 13-kubernetes-manifests-and-sidecar-wiring plan 01
    provides: mcp-rbac.yaml, mcp-skill-configmap.yaml
  - phase: 13-kubernetes-manifests-and-sidecar-wiring plan 02
    provides: openclaw-sandbox.yaml with complete sidecar wiring, validate-phase13.sh
  - phase: 11-streamable-http-transport
    provides: MCP server source code with /health endpoint on port 8080
provides:
  - Live EKS deployment: all Phase 13 manifests applied to wekaappstore namespace
  - wekachrisjen/weka-app-store-mcp:latest multi-arch Docker image (linux/amd64 + linux/arm64)
  - validate-phase13.sh passing 10/10 checks against live cluster
  - Corrected openclaw.json schema: mcp.servers.<name>.{url,transport,skill} format
affects:
  - phase-14 (integration testing can run against live sidecar)

# Tech tracking
tech-stack:
  added:
    - docker buildx multi-platform (linux/amd64 + linux/arm64) for weka-app-store-mcp image
  patterns:
    - kubectl auth can-i output for cluster-scoped resources contains warning prefix before yes/no — grep for ^yes$ not string equality
    - Python containers lack curl; use python3 urllib for HTTP health checks in scripts
    - webapp/__init__.py lazy import pattern: use __getattr__ to avoid pulling FastAPI into non-GUI import contexts
    - openclaw.json config uses mcp.servers.<name>.{url,transport,skill} nested structure (not top-level flat schema)
    - Agent-sandbox operator does not auto-recreate pods on CR spec change; must delete pod to force recreation with new spec

key-files:
  created: []
  modified:
    - app-store-gui/webapp/__init__.py
    - k8s/agent-sandbox/openclaw-sandbox.yaml
    - scripts/validate-phase13.sh

key-decisions:
  - "openclaw.json schema is mcp.servers.<name>.{url,transport,skill} — discovered by live cluster test; old flat schema rejected by OpenClaw 2026.3.23"
  - "Multi-arch image required for EKS (linux/amd64); docker buildx with --platform linux/amd64,linux/arm64 used"
  - "webapp/__init__.py uses __getattr__ lazy import to avoid FastAPI dependency in Docker build context (MCP tools only use webapp.planning and webapp.inspection)"
  - "validate-phase13.sh Check 5 fixed: grep ^yes$ not == yes string comparison (kubectl emits warning prefix for cluster-scoped resources)"
  - "validate-phase13.sh Check 8 fixed: python3 urllib.request instead of curl (weka-mcp-sidecar has python3 not curl)"

patterns-established:
  - "OpenClaw config validation: always run openclaw config validate before deploying openclaw.json config to confirm schema acceptance"
  - "Multi-platform Docker: use docker buildx build --platform linux/amd64,linux/arm64 --push for EKS deployments from Apple Silicon"

requirements-completed: [K8S-01, K8S-02, K8S-03, K8S-04, K8S-05, NCLAW-02, NCLAW-04]

# Metrics
duration: 18min
completed: 2026-03-24
---

# Phase 13 Plan 03: Kubernetes Manifests and Sidecar Wiring — Live Deployment Summary

**All Phase 13 manifests deployed to EKS wekaappstore namespace: pod running 3/3, MCP sidecar healthy, correct openclaw.json schema discovered and fixed, multi-arch image built and pushed, 10/10 validation checks pass.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-03-24T05:59:23Z
- **Completed:** 2026-03-24T06:17:52Z
- **Tasks:** 2 (1 auto + 1 checkpoint auto-approved)
- **Files modified:** 3

## Accomplishments

- Applied mcp-rbac.yaml, mcp-skill-configmap.yaml, and openclaw-sandbox.yaml to wekaappstore namespace in correct order; pod immediately showed 3/3 containers after pod recreation
- Discovered and corrected openclaw.json config schema through live cluster testing — OpenClaw 2026.3.23 uses nested `mcp.servers.<name>` format, not the flat top-level schema from the planning doc
- Built and pushed `wekachrisjen/weka-app-store-mcp:latest` multi-arch image (linux/amd64 + linux/arm64) via docker buildx — first live deployment of the MCP server image
- All 10 validation checks pass: dry-run structural checks (4), RBAC permissions (2), sidecar ready (1), /health 200 (1), openclaw.json content (1), SKILL.md mount (1)

## Task Commits

Each task was committed atomically:

1. **Task 1: Apply manifests to EKS cluster and wait for pod readiness** - `e3666cc` (feat)
2. **Task 2: Human verification of live MCP sidecar deployment** - auto-approved (AUTO_CFG=true)

**Plan metadata:** (docs commit follows this SUMMARY)

## Files Created/Modified

- `k8s/agent-sandbox/openclaw-sandbox.yaml` - Init container printf updated to generate correct mcp.servers nested JSON schema
- `scripts/validate-phase13.sh` - Two bug fixes: Check 5 uses grep for ^yes$; Check 8 uses python3 urllib instead of curl
- `app-store-gui/webapp/__init__.py` - Lazy import via __getattr__ to avoid FastAPI dependency pull in MCP server Docker build

## Decisions Made

- **openclaw.json schema corrected**: Research and Phase 13-02 used the `mcp-server/openclaw.json` planning doc as the schema reference, which had a "Best-effort" comment. Live cluster testing revealed OpenClaw 2026.3.23 rejects `name`, `description`, `transport`, `url`, `skill` as top-level keys. The correct schema is `mcp.servers.<name>.{url, transport, skill}`. Discovery method: probed openclaw config set for each key systematically in a test pod.
- **Multi-arch image**: Docker build on Apple Silicon (arm64) produces arm64 image; EKS nodes are amd64. docker buildx with `--platform linux/amd64,linux/arm64` produces the correct multi-arch manifest list that EKS can pull.
- **webapp/__init__.py lazy import**: The MCP server Dockerfile imports `webapp.planning.apply_gateway` and `webapp.inspection.*`, which trigger `webapp/__init__.py`. The original `from .main import app` imports FastAPI, which is not in `mcp-server/requirements.txt`. Changed to `__getattr__` so `webapp.planning` and `webapp.inspection` imports succeed without pulling FastAPI.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] openclaw.json schema mismatch — top-level flat format rejected by OpenClaw 2026.3.23**
- **Found during:** Task 1 (Apply manifests)
- **Issue:** The init container generated `{"name":"...","transport":"streamable-http","url":"...","skill":"..."}` based on the planning doc's "best-effort" schema. OpenClaw's config validator rejected all keys with "Unrecognized keys: name, description, transport, url, skill". OpenClaw container was in CrashLoopBackOff.
- **Fix:** Probed OpenClaw's config set command to discover the real schema. Updated init container printf arg in openclaw-sandbox.yaml to generate `{"mcp":{"servers":{"weka-app-store-mcp":{"url":"http://localhost:8080/mcp","transport":"streamable-http","skill":"..."}}}}`.
- **Files modified:** `k8s/agent-sandbox/openclaw-sandbox.yaml`
- **Verification:** openclaw config validate returns "Config valid" with the new schema; Check 9 passes in validate-phase13.sh
- **Committed in:** e3666cc (Task 1 commit)

**2. [Rule 3 - Blocking] MCP sidecar image not built/pushed — ErrImagePull on first apply**
- **Found during:** Task 1 (Apply manifests)
- **Issue:** `wekachrisjen/weka-app-store-mcp:latest` had never been pushed to Docker Hub. Pod entered ImagePullBackOff immediately.
- **Fix:** Built image with `docker buildx build --platform linux/amd64,linux/arm64` and pushed. First attempt failed (arm64-only manifest); second attempt with --platform linux/amd64,linux/arm64 produced correct multi-arch manifest.
- **Files modified:** None (image push, no source changes)
- **Verification:** Check 7 (sidecar ready) passes; pod running 3/3
- **Committed in:** e3666cc (as part of Task 1 — build+push is deployment work)

**3. [Rule 1 - Bug] webapp/__init__.py eagerly imports FastAPI — Docker build failed**
- **Found during:** Task 1 (building MCP sidecar image)
- **Issue:** `docker build` sanity check `python -c "import server"` failed with `ModuleNotFoundError: No module named 'fastapi'`. The MCP server Dockerfile only installs `mcp-server/requirements.txt` (no fastapi), but `webapp/__init__.py` has `from .main import app` which pulls fastapi eagerly.
- **Fix:** Replaced eager import with `__getattr__` lazy import. Only actual `from webapp.main import app` calls load FastAPI; library imports of `webapp.planning` and `webapp.inspection` skip main.py entirely.
- **Files modified:** `app-store-gui/webapp/__init__.py`
- **Verification:** Docker build succeeds (`import server` passes); GUI still works via uvicorn webapp.main:app
- **Committed in:** e3666cc (Task 1 commit)

**4. [Rule 1 - Bug] validate-phase13.sh Check 5 false FAIL — string comparison misses kubectl warning prefix**
- **Found during:** Task 1 (running validation script)
- **Issue:** `kubectl auth can-i list nodes` outputs `Warning: resource 'nodes' is not namespace scoped\n\nyes` for cluster-scoped resources. Script compared `[[ "${RBAC_NODES}" == "yes" ]]` which failed even though RBAC was correctly granted.
- **Fix:** Changed to `echo "${RBAC_NODES}" | grep -q "^yes$"` to match the yes/no line regardless of warning prefix.
- **Files modified:** `scripts/validate-phase13.sh`
- **Verification:** Check 5 now passes in validate-phase13.sh
- **Committed in:** e3666cc (Task 1 commit)

**5. [Rule 1 - Bug] validate-phase13.sh Check 8 false FAIL — curl not available in Python container**
- **Found during:** Task 1 (running validation script)
- **Issue:** Script used `kubectl exec ... -- curl -sf http://localhost:8080/health` but the MCP sidecar is a Python container (python:3.10-slim) without curl. Check failed with exit code 1 even though /health was returning 200 (confirmed by readiness probe logs).
- **Fix:** Replaced curl with `python3 -c "import urllib.request; r=urllib.request.urlopen(...); print(r.status)"` and checked output for "200".
- **Files modified:** `scripts/validate-phase13.sh`
- **Verification:** Check 8 now passes in validate-phase13.sh
- **Committed in:** e3666cc (Task 1 commit)

---

**Total deviations:** 5 auto-fixed (2 bugs in manifests, 1 blocking build issue, 2 bugs in validation script)
**Impact on plan:** All fixes necessary for correct live deployment. No scope creep. The core plan (apply manifests, validate live cluster) executed as designed; fixes addressed gaps between planning assumptions and production reality.

## Issues Encountered

- **Agent-sandbox operator pod recreation**: The Sandbox CR was "configured" by kubectl apply but the operator does not automatically delete/recreate the pod on spec changes. Required `kubectl delete pod openclaw-sandbox` after each manifest update to trigger pod recreation with the new spec. This is expected operator behavior (immutable pod spec in Kubernetes).
- **openclaw.json schema**: The planning research referenced `mcp-server/openclaw.json` which had a "Best-effort format" comment. The actual OpenClaw 2026.3.23 schema differs significantly. Schema discovery required systematic probing of the openclaw CLI config set command.

## User Setup Required

None — all cluster resources applied automatically by this plan. The live cluster is now running the full MCP sidecar stack.

## Next Phase Readiness

- Pod running 3/3 (openclaw, weka-mcp-sidecar, git-sync) in wekaappstore namespace
- `bash scripts/validate-phase13.sh --live wekaappstore` passes 10/10 checks
- MCP sidecar /health endpoint returning 200; OpenClaw reads `mcp.servers.weka-app-store-mcp` from openclaw.json
- SKILL.md mounted and readable in OpenClaw container at /home/node/.openclaw/SKILL.md
- Blueprint directory populated by git-sync from warp-blueprints repo
- Phase 14 (integration testing) can immediately send tool calls to the MCP sidecar through OpenClaw

---
*Phase: 13-kubernetes-manifests-and-sidecar-wiring*
*Completed: 2026-03-24*
