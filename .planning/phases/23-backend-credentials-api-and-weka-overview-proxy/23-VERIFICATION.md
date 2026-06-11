---
phase: 23-backend-credentials-api-and-weka-overview-proxy
verified: 2026-06-11T00:00:00Z
status: human_needed
score: 8/8 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Visit /settings in a browser after deploying. Confirm no JavaScript console errors appear and the four sections (Kubernetes Auth Status, Cluster Status, Blueprint Uninstall, Debug) render in order."
    expected: "Page loads cleanly, no ReferenceError, blueprint list still loads, auth status polls every 10 seconds."
    why_human: "Jinja2 parse check passes but runtime browser JS execution cannot be verified programmatically."
  - test: "With a real WEKA cluster and a weka-storage WarpCredential CR, call GET /api/weka/overview?credential=<name>. Then call it again within 60 seconds."
    expected: "First response has cached:false and a populated capacity/filesystems/backendNodes payload. Second response has cached:true and the same fetchedAt timestamp."
    why_human: "No live WEKA cluster is available in the test environment; the 60s cache and real API field names cannot be validated without one."
  - test: "With a real WEKA cluster, call GET /api/weka/overview?credential=<name>&bust=1 after a cached response exists."
    expected: "fetchedAt advances; cached:false; three new WEKA API calls are made."
    why_human: "Requires live WEKA endpoint."
---

# Phase 23: Backend Credentials API and WEKA Overview Proxy — Verification Report

**Phase Goal:** Add backend credentials API and WEKA overview proxy — remove deprecated /api/secret/* endpoints, introduce /api/credentials CRUD backed by WarpCredential CRs, and add /api/weka/overview proxy with 60s cache.
**Verified:** 2026-06-11T00:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GET /api/secret/huggingface returns 404 (route removed) | VERIFIED | `grep -nE "^@app\.post\(\"/api/secret/(huggingface\|nvidia)\"\)"` returns no matches in main.py |
| 2 | GET /api/secret/nvidia returns 404 (route removed) | VERIFIED | Same grep — no `save_huggingface_key` or `save_nvidia_key` function definitions found |
| 3 | GET /api/credentials returns JSON shape with name, displayName, type, ready, lastSyncTime per WarpCredential CR — no raw credential values | VERIFIED | `@app.get("/api/credentials")` at line 752; `_build_credential_response_item` at line 710 uses explicit whitelist; test `test_list_credentials_returns_shape_without_secret_values` PASSED |
| 4 | GET /api/credentials?type=<t> returns only credentials of that type with ready=true | VERIFIED | Type filter logic in handler; test `test_list_credentials_type_filter_returns_only_ready` PASSED |
| 5 | POST /api/credentials creates warp-cred-<slug> Secret AND WarpCredential CR with slug collision handling | VERIFIED | `create_or_update_secret(f"warp-cred-{slug}", ...)` at line 853; `create_namespaced_custom_object` at line 879; tests `test_post_credential_nvidia_creates_secret_and_cr` and `test_post_credential_slug_collision_appends_suffix` PASSED |
| 6 | DELETE /api/credentials/<name> deletes CR first then warp-cred-<name> Secret; derived secrets NOT touched | VERIFIED | `delete_namespaced_custom_object` then `delete_namespaced_secret(name=f"warp-cred-{name}")` at lines 935/948; test `test_delete_credential_deletes_cr_then_raw_secret_preserves_derived` PASSED |
| 7 | GET /api/weka/overview makes login + three parallel WEKA calls; 60s cache; ?bust=1 bypass | VERIFIED | `@app.get("/api/weka/overview")` at line 965; `asyncio.gather` at line 1024 with `/api/v2/fileSystems`, `/api/v2/cluster`, `/api/v2/containers`; cache at `_weka_overview_cache[cache_key]` line 1036; tests `test_weka_overview_cache_hit_avoids_refetch` and `test_weka_overview_bust_query_bypasses_cache` PASSED |
| 8 | No raw credential values in any response body or log line | VERIFIED | Logger leak grep gate (`logger.*api_key\|token=\|WEKA_API_TOKEN\|...`) returned zero matches; `_build_credential_response_item` whitelist confirmed; tests `test_build_credential_response_item_omits_secret_fields`, `test_weka_overview_login_failure_returns_502_without_leak` PASSED |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app-store-gui/webapp/main.py` | /api/credentials CRUD + /api/weka/overview + helpers | VERIFIED | Routes at lines 752, 798, 906, 965; helpers at 651–1622; compiles clean |
| `app-store-gui/webapp/templates/settings.html` | HF/NVIDIA sections removed; localStorage fallback wired | VERIFIED | No `Section 1: HuggingFace` or `Section 2: NVIDIA` in file; `localStorage.getItem('selectedNamespace') \|\| 'default'` at line 362 |
| `app-store-gui/tests/test_credentials_api.py` | 23 tests covering all must-haves | VERIFIED | 23 tests collected and passed in 0.63s |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `@app.get("/api/credentials")` | `CustomObjectsApi.list_namespaced_custom_object` | `plural="warpcredentials"` | VERIFIED | Lines 772, 777 — both namespaced and cluster-wide branches present |
| `@app.post("/api/credentials")` | `create_or_update_secret` then `create_namespaced_custom_object` | `warp-cred-{slug}` Secret first | VERIFIED | Line 853 (Secret), line 879 (CR); `_CREDENTIAL_TYPE_KEYS[type]["secret_ref_key"]` at line 864 |
| `@app.delete("/api/credentials/{name}")` | `delete_namespaced_custom_object` then `delete_namespaced_secret` | CR first, raw Secret second | VERIFIED | Lines 935 (CR), 948 (Secret); no derived `warp-{name}-*` deletion |
| `@app.get("/api/weka/overview")` | WEKA REST API via `_weka_login` + `asyncio.gather` | `_resolve_weka_credential_secret` → `read_namespaced_secret(warp-cred-...)` | VERIFIED | Lines 1003, 1008, 1024–1028 |
| `@app.get("/api/weka/overview")` | `_weka_overview_cache` 60s TTL | `bust=1` bypasses | VERIFIED | Cache check lines 979–985; bust bypass line 977; update line 1036 |
| `settings.html loadBlueprints()` | `localStorage.getItem('selectedNamespace')` | Inline fallback | VERIFIED | Line 362; `getSettingsNamespace` function absent from file |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `list_credentials` handler | `items` list | `CustomObjectsApi.list_namespaced_custom_object(plural="warpcredentials")` | Yes — queries K8s API for live CRs | FLOWING |
| `get_weka_overview` handler | `data` dict | `_weka_login` + three `_weka_get_json` calls + `_assemble_weka_overview` | Yes — reads K8s Secret, calls live WEKA REST API | FLOWING (requires live WEKA for production; mocked in tests) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| main.py compiles | `python -m py_compile app-store-gui/webapp/main.py` | exit 0 | PASS |
| Deprecated routes absent | `grep -nE "^@app\.post\(\"/api/secret/(huggingface\|nvidia)\"\)"` | no output | PASS |
| All 3 credentials routes present | grep for GET/POST/DELETE /api/credentials | lines 752, 798, 906 | PASS |
| WEKA overview route present | grep for /api/weka/overview | line 965 | PASS |
| asyncio.gather with 3 WEKA paths | grep for /api/v2/fileSystems, /api/v2/cluster, /api/v2/containers | all present at lines 1025–1027 | PASS |
| No credential leak in logs | logger grep gate for token/key literals | zero matches | PASS |
| Full test suite | `pytest app-store-gui/tests/test_credentials_api.py -v` | 23 passed in 0.63s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| API-01 | 23-02 | GET /api/credentials returns safe JSON shape | SATISFIED | Route at line 752; `_build_credential_response_item` whitelist; test PASSED |
| API-02 | 23-02 | GET /api/credentials?type=<t> returns ready-only | SATISFIED | Type filter in list handler; test PASSED |
| API-03 | 23-02 | POST /api/credentials creates Secret + CR with slug collision | SATISFIED | Lines 853, 879; collision tests PASSED |
| API-04 | 23-02 | DELETE /api/credentials/<name> removes CR + raw Secret, preserves derived | SATISFIED | Lines 935, 948; delete order test PASSED |
| API-05 | 23-03 | GET /api/weka/overview with 60s cache and ?bust=1 | SATISFIED | Route at line 965; cache logic; tests PASSED |
| API-06 | 23-03 | /api/weka/overview response schema (capacity, filesystems without uid, backendNodes, fetchedAt) | SATISFIED | `_assemble_weka_overview` at line 1524; shape test PASSED |
| API-07 | 23-01 | Remove /api/secret/nvidia and /api/secret/huggingface | SATISFIED | Routes absent from main.py; settings.html cleaned |
| API-08 | 23-02, 23-03, 23-04 | No raw credential values in logs or responses | SATISFIED | Logger grep gate passes; response whitelist; 4 no-leak tests PASSED |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | — |

No TBD/FIXME/XXX markers found in modified files. No placeholder returns or empty implementations detected in the new route handlers or helpers.

### Human Verification Required

#### 1. Settings Page Browser Smoke Test

**Test:** Deploy the application and navigate to `/settings`. Open browser developer tools (Console tab).
**Expected:** Page renders with four sections in order (Kubernetes Auth Status, Cluster Status, Blueprint Uninstall, Debug). No JavaScript `ReferenceError` or console errors appear. The blueprint list loads and auth status polling fires every 10 seconds.
**Why human:** Jinja2 template parses correctly (verified), but runtime JavaScript execution in a real browser cannot be verified statically. The `loadBlueprints()` namespace fallback to `localStorage` is the only change that could produce a silent behavioral regression.

#### 2. WEKA Overview Cache Behavior (Live Cluster)

**Test:** With a live WEKA cluster credential registered (`weka-storage` type), call `GET /api/weka/overview?credential=<name>` twice within 60 seconds. Then call with `?bust=1`.
**Expected:** First call returns `cached:false` with populated `capacity`, `filesystems`, `backendNodes`. Second call returns `cached:true` with identical `fetchedAt`. Third call (`bust=1`) returns `cached:false` with an advanced `fetchedAt`.
**Why human:** No live WEKA cluster available in the test environment. The cache logic is fully tested with mocks (23 tests pass), but the actual WEKA REST API field names (`access_token` vs `data.access_token` vs `token`, `total_budget` vs `size`, etc.) can only be confirmed against a real cluster. The `_assemble_weka_overview` tolerant fallbacks cover known variants but field-name drift remains a production risk.

### Gaps Summary

No gaps. All 8 must-have truths are verified against the actual codebase. The two human verification items are operational confirmation checks (browser rendering, live WEKA API compatibility) — not implementation gaps.

---

_Verified: 2026-06-11T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
