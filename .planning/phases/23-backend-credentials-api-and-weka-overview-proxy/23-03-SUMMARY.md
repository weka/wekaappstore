---
phase: 23
plan: "03"
subsystem: app-store-gui
tags:
  - weka-overview
  - credentials
  - caching
  - http-proxy
dependency_graph:
  requires:
    - "23-02"
  provides:
    - GET /api/weka/overview
    - _weka_overview_cache
    - _weka_login
    - _weka_get_json
    - _weka_post_json
    - _weka_ssl_context
    - _resolve_weka_credential_secret
    - _assemble_weka_overview
  affects:
    - app-store-gui/webapp/main.py
tech_stack:
  added:
    - ssl (stdlib — TLS context for WEKA REST API)
    - datetime (stdlib — ISO 8601 fetchedAt timestamp)
    - urllib.request (stdlib — sync HTTP promoted to top-level import)
    - urllib.error (stdlib — HTTP/URL error handling)
  patterns:
    - asyncio.to_thread wrapping sync urllib calls in async FastAPI handler
    - Module-level TTL cache dict (matching _last_ready_cache pattern)
    - Tolerant field-name extraction with ordered fallbacks
key_files:
  modified:
    - app-store-gui/webapp/main.py
decisions:
  - "Tolerant access token extraction tries resp['access_token'], resp['data']['access_token'], resp['token'] in order — matching PRD 'verify against Swagger' note for deployed WEKA version compatibility"
  - "Namespace-scoped cache key (ns/credential) prevents cross-tenant cache reads (T-23-03-04)"
  - "Error message for WEKA login failures collapses to fixed string 'WEKA login failed' to prevent URL/response-body echo to browser (T-23-03-08)"
  - "capacity_source: fallback-sum tag surfaces field-name drift as data instead of 500 error (T-23-03-07)"
metrics:
  duration: "~25 minutes"
  completed_date: "2026-06-11"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 1
---

# Phase 23 Plan 03: WEKA Overview Proxy Summary

GET /api/weka/overview with parallel WEKA REST API calls, 60s namespace-scoped cache, and credential no-leak posture.

## What Was Built

Added `GET /api/weka/overview?credential=<name>` to `app-store-gui/webapp/main.py`. The handler resolves a named `weka-storage` WarpCredential CR, reads its raw Secret, exchanges username+token for a Bearer token at WEKA's `/api/v2/login`, calls `/api/v2/fileSystems`, `/api/v2/cluster`, and `/api/v2/containers` in parallel, assembles the API-06 response shape, and caches the result per credential for 60 seconds.

## Line Ranges

| Symbol | Lines (worktree main.py) |
|--------|--------------------------|
| Top-level imports (ssl, datetime, urllib.*) | 20-23 |
| `_weka_overview_cache` dict + `_WEKA_CACHE_TTL_SECONDS` | 1403-1404 |
| `_weka_ssl_context()` | 1407-1419 |
| `_weka_get_json()` | 1421-1435 |
| `_weka_post_json()` | 1436-1456 |
| `_weka_login()` | 1457-1486 |
| `_resolve_weka_credential_secret()` | 1487-1523 |
| `_assemble_weka_overview()` | 1524-1622 |
| `@app.get("/api/weka/overview")` route handler | 965-1057 |

## Access Token JSON Path

`_weka_login()` implements a tolerant extractor that tries in order:
1. `resp["access_token"]` — most common in WEKA 4.x
2. `resp["data"]["access_token"]` — used in some WEKA versions with envelope responses
3. `resp["token"]` — older WEKA 3.x fallback

The exact path used in production must be confirmed against the live Swagger UI at `https://<cluster>:14000/api/v2/docs`. If none of the three are present, a `RuntimeError` with the Swagger URL is raised and surfaces as HTTP 502.

## Cluster Capacity Fallback

`_assemble_weka_overview()` checks for a `capacity` subdict in the cluster response first. If absent (field-name drift scenario per T-23-03-07), it sums filesystem totals as a fallback and tags the output with `"capacity_source": "fallback-sum"`. Whether this fallback was needed against a live WEKA cluster is TBD — no live cluster was available during implementation.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | WEKA helpers, cache dict, assembler | 7bb221e | app-store-gui/webapp/main.py (+253 lines) |
| 2 | GET /api/weka/overview route | 18fb264 | app-store-gui/webapp/main.py (+93 lines) |

## Deviations from Plan

None - plan executed exactly as written.

## Threat Model Coverage

All T-23-03-* mitigations implemented:

| Threat ID | Status | Implementation |
|-----------|--------|----------------|
| T-23-03-01 | Mitigated | Bearer token used only in local `headers` dict; `_assemble_weka_overview` whitelists output fields; acceptance criteria grep gate confirms no `Bearer ` in responses |
| T-23-03-02 | Mitigated | `endpoint` comes from K8s Secret (admin-controlled); handler only issues calls to fixed WEKA paths; `credential` param is regex-validated before any K8s lookup |
| T-23-03-03 | Mitigated | DNS-1123 regex rejects `/`, `..`, and non-`[a-z0-9-]` chars; invalid name → 400 without I/O |
| T-23-03-04 | Mitigated | Cache key is `f"{ns}/{credential}"` — namespace-scoped |
| T-23-03-05 | Mitigated | 60s cache; each WEKA call has 15s timeout |
| T-23-03-06 | Mitigated | Default verifying TLS; `WEKA_OVERVIEW_INSECURE_TLS=true` opt-out explicit |
| T-23-03-07 | Mitigated | Tolerant field-name fallbacks; fallback-sum tag on capacity mismatch |
| T-23-03-08 | Mitigated | Error strings include only URL/status code; login errors collapse to fixed string |
| T-23-03-09 | Accepted | No audit log; K8s audit log is system-of-record |

## Known Stubs

None. All fields in the API-06 response shape are populated from live WEKA API data.

## Self-Check: PASSED

- app-store-gui/webapp/main.py: FOUND
- 23-03-SUMMARY.md: FOUND
- Commit 7bb221e (Task 1): FOUND
- Commit 18fb264 (Task 2): FOUND
