---
phase: 29-backend-wiring-secret-safety
plan: 02
subsystem: api
tags: [base64, jinja2, yaml, quay, kubernetes-secrets, weka-csi, python]

# Dependency graph
requires:
  - phase: 29-backend-wiring-secret-safety/29-01
    provides: NAMESPACE_PRESERVING_APPS set and parse_deploy_timeout; prerequisite for this plan's deploy_stream wiring
  - phase: 27-install-blueprint-authoring
    provides: app-store-install.yaml blueprint with [[ quay_dockerconfigjson ]], [[ join_ip_ports_list ]], [[ endpoints_csv ]] token sites
  - phase: 28-operator-helm-auth-crd-discovery
    provides: D-02 quay_dockerconfigjson shape consumed by operator helm --registry-config

provides:
  - build_quay_dockerconfigjson(user, password) pure helper — byte-exact base64 auth with no trailing newline
  - split_endpoints(join_ip_ports) pure helper — returns join_ip_ports_list (json.dumps string) and endpoints_csv
  - deploy_stream derives and merges quay_dockerconfigjson + endpoint forms into user_vars before template.render
  - 6 unit tests proving byte-exactness, whitespace trimming, empty-entry dropping, and valid-YAML flow-sequence render

affects:
  - 29-03 (secret safety — quay_dockerconfigjson now in user_vars, plan 29-03 must redact it from annotation + SSE)
  - 30-install-wizard-frontend (wizard posts quay_username + quay_password; server derives quay_dockerconfigjson)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "server-side variable derivation: pure helpers merge into user_vars before Jinja2 template.render"
    - "json.dumps(entries) for YAML flow-sequence: double-quoted array avoids single-quote repr ambiguity"
    - "base64.b64encode(...).decode('ascii') — never encodebytes — for no-trailing-newline byte-exact encoding"

key-files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/tests/test_dynamic_blueprint.py

key-decisions:
  - "build_quay_dockerconfigjson uses base64.b64encode (not encodebytes) — byte-exact encoding, no trailing newline, D-04"
  - "split_endpoints returns join_ip_ports_list as json.dumps string (double-quoted JSON array) so Jinja2 [[ ]] render produces valid YAML flow-sequence — resolved D-05 open item"
  - "derivation guarded by key presence: blueprints without quay_username/quay_password or join_ip_ports are unaffected"
  - "quay_dockerconfigjson key is assigned directly (not via update) to match blueprint x-variables entry name"

requirements-completed: [PROG-02]

# Metrics
duration: 3min
completed: 2026-06-24
---

# Phase 29 Plan 02: Backend Variable Derivation Summary

**Server-side `build_quay_dockerconfigjson` and `split_endpoints` helpers wired into `deploy_stream` before Jinja2 render, with byte-exact encoding verified by 6 unit tests including YAML flow-sequence parse check**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-24T12:09:30Z
- **Completed:** 2026-06-24T12:13:16Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- `build_quay_dockerconfigjson(user, password)` added near `parse_x_variables` — compact JSON string, `base64.b64encode` only, auth decodes to exactly `user:pass` with no trailing bytes (D-04, T-29-04)
- `split_endpoints(join_ip_ports)` added — trims whitespace, drops empty entries, returns `join_ip_ports_list` as `json.dumps` string and `endpoints_csv` as comma-joined string (D-05, T-29-05)
- Both merged into `user_vars` in `deploy_stream` immediately before `template.render(**user_vars)`, guarded so blueprints without these keys are unaffected
- 6 new unit tests in `test_dynamic_blueprint.py`: byte-exact decode (no trailing `\n`), structure, single/multiple/whitespace/empty endpoints, and YAML flow-sequence render via Jinja2 `[[]]` env — all 38 tests pass

## Task Commits

1. **Task 1: Add build_quay_dockerconfigjson and split_endpoints helpers** - `1c0f1ea` (feat)
2. **Task 2: Merge derived vars into user_vars before Jinja2 render** - `e9451cc` (feat)
3. **Task 3: Unit tests for derivation correctness and render validity** - `4c15f5a` (feat)

## Files Created/Modified

- `app-store-gui/webapp/main.py` - Added `build_quay_dockerconfigjson` and `split_endpoints` module-level helpers; added derivation wiring in `deploy_stream` before `template.render`
- `app-store-gui/tests/test_dynamic_blueprint.py` - Added 6 unit tests (tests 22–27) for the new helpers

## Decisions Made

- Used `json.dumps(entries)` for `join_ip_ports_list` (double-quoted JSON array) rather than Python list repr (single quotes) — resolved D-05 open item: `joinIpPorts: [[ join_ip_ports_list ]]` renders and `yaml.safe_load`s as a proper list
- Used direct dict assignment for `quay_dockerconfigjson` (not `user_vars.update`) to match the existing x-variables key name, while using `user_vars.update(split_endpoints(...))` for the two endpoint keys
- Derivation is guarded (`if user_vars.get("quay_username") or user_vars.get("quay_password")`) so the cluster-init blueprint and any blueprint without quay creds is unaffected

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Docstring mentioned forbidden pattern string**
- **Found during:** Task 1 (acceptance criteria check)
- **Issue:** The docstring for `build_quay_dockerconfigjson` contained the word "encodebytes" in "NOT encodebytes" — the acceptance criterion `grep -c 'encodebytes\|encodestring' app-store-gui/webapp/main.py` returns 0 requires zero matches
- **Fix:** Rewrote docstring to describe the invariant without naming the forbidden function
- **Files modified:** app-store-gui/webapp/main.py
- **Verification:** `grep -c 'encodebytes\|encodestring'` returns 0
- **Committed in:** 1c0f1ea (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in acceptance criteria compliance)
**Impact on plan:** Trivial docstring wording change. No behavioral impact.

## Issues Encountered

Pre-existing failures in `app-store-gui/tests/test_credentials_api.py` (7 tests fail with `TypeError: create_credential() got an unexpected keyword argument 'type'`) — confirmed pre-existing before this plan, unrelated to deliverables. All `test_dynamic_blueprint.py` tests pass (38/38).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SC2 complete: server derives `quay_dockerconfigjson` (byte-exact) and both endpoint forms, merged before render
- `quay_dockerconfigjson` is now present in `user_vars` at annotation-stamp time — plan 29-03 (secret safety) must exclude it from `warp.io/gui-variables` annotation and redact from SSE stream (T-29-06 transfer)
- Ready for plan 29-03: annotation allowlist + SSE redaction

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: information_disclosure | app-store-gui/webapp/main.py | quay_dockerconfigjson now lives in user_vars and would be written to the annotation and emitted in SSE — mitigation owned by plan 29-03 (T-29-06 transfer) |

---
*Phase: 29-backend-wiring-secret-safety*
*Completed: 2026-06-24*

## Self-Check: PASSED

- `app-store-gui/webapp/main.py` exists and contains `def build_quay_dockerconfigjson` and `def split_endpoints`
- `app-store-gui/tests/test_dynamic_blueprint.py` exists with tests 22–27
- Commits 1c0f1ea, e9451cc, 4c15f5a all exist in git log
- `python -m py_compile app-store-gui/webapp/main.py` passes
- 38 tests pass in test_dynamic_blueprint.py
