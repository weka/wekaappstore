---
phase: 23-backend-credentials-api-and-weka-overview-proxy
plan: "02"
subsystem: api
tags: [fastapi, kubernetes, warpcredential, crd, credentials-api, slug-generation]

# Dependency graph
requires:
  - phase: 23-01
    provides: Old /api/secret/nvidia and /api/secret/huggingface handlers removed; WarpCredential CRD shipped by Phase 21; operator reconciler shipped by Phase 22
provides:
  - "GET /api/credentials handler: lists WarpCredential CRs with safe status shape (API-01)"
  - "GET /api/credentials?type=<t>: filters to ready=true items of requested type (API-02)"
  - "POST /api/credentials: creates warp-cred-<slug> Secret + WarpCredential CR with collision handling (API-03)"
  - "DELETE /api/credentials/{name}: deletes CR + raw Secret; preserves derived secrets (API-04)"
  - "_make_credential_slug helper: displayName → DNS-1123 slug (D-11)"
  - "_allocate_unique_credential_slug helper: -2/-3 suffix collision resolution (D-12)"
  - "_build_credential_response_item helper: CR dict → safe response shape (no credential values)"
  - "_CREDENTIAL_TYPE_KEYS constant: per-type secret key layout"
affects:
  - "23-03 (WEKA overview proxy - same file)"
  - "23-04 (tests for these handlers)"
  - "24 (Settings GUI consumes all four /api/credentials routes)"
  - "25 (Blueprint SDK consumes GET /api/credentials?type=<t>)"

# Tech tracking
tech-stack:
  added: ["re (stdlib, added import)"]
  patterns:
    - "asyncio.to_thread wrapping sync CustomObjectsApi calls inside async FastAPI handlers"
    - "_CREDENTIAL_TYPE_KEYS constant mapping type string to per-type secret key layout"
    - "_build_credential_response_item whitelist pattern: fields explicitly allowed, no wildcard dict spread"

key-files:
  created: []
  modified:
    - "app-store-gui/webapp/main.py"

key-decisions:
  - "Added import re (D-11 requires re.sub for slug sanitization; was missing from existing imports)"
  - "Inserted re import alongside other stdlib imports per PEP 8 alphabetical ordering"
  - "_CREDENTIAL_NAME_RE compiled at module level rather than inside handler for performance"
  - "type query param uses Optional[str] = Query(None) — shadows builtin type but matches plan spec exactly"
  - "asyncio.to_thread used for both K8s list and create/delete calls per D-04 pattern"
  - "404 on CR or Secret delete treated as idempotent success (defensive; operator may have already cleaned up)"

patterns-established:
  - "Credential type-key constant pattern: _CREDENTIAL_TYPE_KEYS[type]['secret_ref_key'] drives both Secret layout and CR spec.secretRef.key"
  - "Slug allocation pattern: list-then-check before create avoids 409s except in narrow race window"
  - "Response whitelist pattern: _build_credential_response_item builds an explicit dict — no ** spread from CR that could accidentally include credential fields"

requirements-completed: [API-01, API-02, API-03, API-04, API-08]

# Metrics
duration: 25min
completed: 2026-06-11
---

# Phase 23 Plan 02: Backend Credentials API Summary

**Four /api/credentials route handlers (GET list+filter, POST create, DELETE) with slug helpers, type-key constant, and no-credential-value response whitelisting for the WarpCredential CRD**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-11T08:10:00Z
- **Completed:** 2026-06-11T08:35:00Z
- **Tasks:** 2 of 2
- **Files modified:** 1 (app-store-gui/webapp/main.py)

## Accomplishments

- Added `import re` (was missing; required by `_make_credential_slug`)
- Added `_CREDENTIAL_TYPE_KEYS` constant, `_VALID_CREDENTIAL_TYPES` tuple, and three private helpers (`_make_credential_slug`, `_allocate_unique_credential_slug`, `_build_credential_response_item`)
- Added `GET /api/credentials` handler covering API-01 (list) and API-02 (type+ready filter)
- Added `POST /api/credentials` handler covering API-03 (slug generation, collision handling, Secret + CR creation)
- Added `DELETE /api/credentials/{name}` handler covering API-04 (CR deleted first, raw Secret second, derived secrets untouched)
- All handlers pass the no-credential-value grep gate (API-08)

## Inserted Line Ranges

| Artifact | Lines | Description |
|---|---|---|
| `import re` | 9 | New import added alongside other stdlib imports |
| `_CREDENTIAL_TYPE_KEYS` | 647–660 | Per-type secret key layout constant |
| `_VALID_CREDENTIAL_TYPES` | 662 | Convenience tuple for POST validation |
| `def _make_credential_slug` | 665–679 | D-11 slug generation |
| `def _allocate_unique_credential_slug` | 681–703 | D-12 collision resolution |
| `def _build_credential_response_item` | 706–746 | Safe CR → API response item transform |
| `@app.get("/api/credentials")` | 748–792 | List + type-filter handler |
| `@app.post("/api/credentials")` | 794–896 | Create Secret + CR handler |
| `_CREDENTIAL_NAME_RE` | 899 | Module-level compiled DNS-1123 regex |
| `@app.delete("/api/credentials/{name}")` | 902–959 | Delete CR + raw Secret handler |

## Task Commits

1. **Task 1: Slug helpers, type-key constant, GET /api/credentials** — `f809fb5` (feat)
2. **Task 2: POST /api/credentials and DELETE /api/credentials/{name}** — `f027217` (feat)

## Files Created/Modified

- `app-store-gui/webapp/main.py` — Added `import re`, four route handlers, three private helpers, and two constants (319 new lines inserted between the `/api/secrets` and `/api/namespaces` handlers)

## Decisions Made

- **`import re` added:** The plan's `<interfaces>` block noted `re` is already imported — it was NOT. Added as a Rule 1 auto-fix (would have been a NameError at runtime). Alphabetical position after `import os`.
- **`type` parameter name kept:** Shadows builtin `type` but the plan spec explicitly uses it. Acceptable for FastAPI query params where the name appears in OpenAPI docs.
- **`_CREDENTIAL_NAME_RE` compiled at module level:** One-time compile avoids repeating `re.compile` on every DELETE request.
- **`asyncio.to_thread` used for all sync K8s calls:** Matches D-04 and existing patterns at lines 495–496.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added missing `import re`**
- **Found during:** Task 1 (implementing `_make_credential_slug`)
- **Issue:** The plan's `<interfaces>` section stated "Confirm `re` is already imported at the top of main.py (line 14 area — verify)" — but `re` was not in the import list. `_make_credential_slug` uses `re.sub`, so this would be a `NameError` at first call.
- **Fix:** Added `import re` between `import os` and `import yaml` (PEP 8 alphabetical order within stdlib block)
- **Files modified:** `app-store-gui/webapp/main.py`
- **Verification:** `grep -nE "^import re$" app-store-gui/webapp/main.py` returns line 9; `python -m py_compile` exits 0
- **Committed in:** f809fb5 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — missing import that would cause NameError)
**Impact on plan:** Required for correctness; zero scope change.

## Issues Encountered

None beyond the missing `re` import documented above.

## Known Stubs

None — all handlers are fully wired to real Kubernetes API calls. No hardcoded empty values or placeholder responses.

## Threat Flags

No new security-relevant surface beyond what is documented in the plan's `<threat_model>`. The four new routes are gated by the existing `ClusterInitMiddleware` (T-23-02-06 accepted residual).

## Self-Check

- `python -m py_compile app-store-gui/webapp/main.py` → exits 0
- `grep -n "_CREDENTIAL_TYPE_KEYS"` → line 647
- `grep -n "_VALID_CREDENTIAL_TYPES"` → line 662
- `grep -n "def _make_credential_slug"` → line 665 (one definition)
- `grep -n "def _allocate_unique_credential_slug"` → line 681 (one definition)
- `grep -n "def _build_credential_response_item"` → line 706 (one definition)
- `grep -nE '@app\.get\("/api/credentials"\)'` → line 748 (one match)
- `grep -nE '@app\.post\("/api/credentials"\)'` → line 794 (one match)
- `grep -nE '@app\.delete\("/api/credentials/\{name\}\"\)'` → line 902 (one match)
- `plural="warpcredentials"` → lines 690, 768, 773 (three matches — allocate_unique + list_cluster + list_namespaced)
- logger credential-value grep gate → PASSED (no credential values in log calls)
- derived secrets grep gate → PASSED (no warp-<name>-* patterns referenced)
- Task 1 commit f809fb5 → exists in git log
- Task 2 commit f027217 → exists in git log

## Self-Check: PASSED

## Next Phase Readiness

- Plan 23-03 can now add `GET /api/weka/overview` in the same `main.py`; all existing patterns established
- Plan 23-04 can add `app-store-gui/tests/test_credentials_api.py` with stub injection against these handlers
- Phase 24 (Settings GUI) has a complete read/write credential API surface to build against

---
*Phase: 23-backend-credentials-api-and-weka-overview-proxy*
*Completed: 2026-06-11*
