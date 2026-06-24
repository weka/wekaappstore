---
phase: 29-backend-wiring-secret-safety
plan: 03
subsystem: api
tags: [security, redaction, kubernetes-secrets, sse, python, allowlist]

# Dependency graph
requires:
  - phase: 29-backend-wiring-secret-safety/29-02
    provides: build_quay_dockerconfigjson and split_endpoints in user_vars at annotation time (quay_dockerconfigjson must be redacted)
  - phase: 29-backend-wiring-secret-safety/29-01
    provides: NAMESPACE_PRESERVING_APPS + parse_deploy_timeout; deploy_stream structure this plan wires into

provides:
  - _is_secret_key(name) — single source-of-truth predicate for both redaction sites
  - _safe_gui_variables(user_vars) — annotation allowlist (drops *password*/*token*/*secret*/quay_dockerconfigjson keys)
  - _redact_secrets(message, user_vars) — SSE message redactor (replaces secret VALUES with ***)
  - Both wired at annotation stamp (~3067) and SSE component emit (~3116, ~3126)
  - 14 unit tests proving no secret leak at either point

affects:
  - 30-install-wizard-frontend (wizard posts quay_password/weka_password — these are now safe from annotation/SSE leak)
  - 31-end-to-end-verification (SEC-01 secret-leak gate is now implemented — verify with real install)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "single-predicate dual-site redaction: one _is_secret_key predicate shared by both annotation allowlist and SSE value-set"
    - "SSE redaction by value (not key): replaces actual secret strings in operator messages, not just key-based filtering"
    - "annotation allowlist via key-drop: _safe_gui_variables returns a shallow copy with secret keys removed"

key-files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/tests/test_dynamic_blueprint.py

key-decisions:
  - "_is_secret_key is the SINGLE source of truth — both _safe_gui_variables and _redact_secrets import this predicate, preventing divergence"
  - "SSE redactor builds secret-value-set from user_vars at call time (not at startup) — works correctly even when user_vars changes between calls"
  - "Empty secret values are excluded from the SSE redaction set to avoid spurious *** replacements on unset fields"
  - "Annotation uses _safe_gui_variables (key-drop); SSE uses _redact_secrets (value-replace) — different mechanisms for different threat models"

requirements-completed: [SEC-01]

# Metrics
duration: 8min
completed: 2026-06-24
---

# Phase 29 Plan 03: Secret Safety Summary

**Single `_is_secret_key` predicate gates both the CR annotation allowlist and the SSE message redactor, closing the secret-leak vector at both etcd-persistence and browser/proxy-log emission sites**

## Performance

- **Duration:** ~8 min (including connection interruption recovery)
- **Completed:** 2026-06-24
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- `_is_secret_key(name)` module-level predicate added at main.py:1820 — matches `*password*`/`*token*`/`*secret*` (case-insensitive) or exact `quay_dockerconfigjson`; this is the SINGLE source of truth (D-09, LOCKED)
- `_safe_gui_variables(user_vars)` added at main.py:1834 — returns shallow copy with all secret keys removed; wired at the annotation stamp (main.py:3067)
- `_redact_secrets(message, user_vars)` added at main.py:1845 — builds secret-value-set from same predicate, replaces values with `***`; wired at both SSE component emit sites (main.py:3116 and 3126)
- 14 new unit tests (tests 39–52) in `test_dynamic_blueprint.py`: predicate coverage, allowlist key-drop, mutation safety, SSE value-replace, multi-secret, empty-value guard, and full annotation/SSE integration checks — all 52 tests pass

## Task Commits

1. **Task 1: Add _is_secret_key, _safe_gui_variables, _redact_secrets helpers** - `85485a1` (feat)
2. **Task 2: Wire annotation allowlist and SSE redactor** - `35a18e9` (feat)
3. **Task 3: Unit tests** - `6c34a22` (test)

## Files Created/Modified

- `app-store-gui/webapp/main.py` — Added three helpers at ~1812–1857; wired `_safe_gui_variables` at annotation stamp and `_redact_secrets` at two SSE emit sites
- `app-store-gui/tests/test_dynamic_blueprint.py` — Added 14 unit tests (tests 39–52)

## Decisions Made

- Used `str(v)` when building the SSE secret-value-set so non-string values (ints, etc.) are correctly redacted
- Excluded empty/falsy values from the redaction set to avoid replacing empty string `""` throughout messages
- The annotation allowlist drops keys (not values) — appropriate since etcd stores the full dict and dropping a key is sufficient
- The SSE redactor replaces values (not keys) — necessary because operator messages contain rendered content, not Python dicts

## Phase 29 Completion

All three plans complete. Phase 29 success criteria met:
- SC1: `NAMESPACE_PRESERVING_APPS` extends namespace-preserve to `app-store-install` ✓
- SC2: `build_quay_dockerconfigjson` (byte-exact) + `split_endpoints` (dual form) ✓
- SC3: `parse_deploy_timeout` + `x-deploy-timeout: 2700` in blueprint ✓
- SC4: `_safe_gui_variables` + `_redact_secrets` — zero secret leak at annotation and SSE ✓

## Next Phase Readiness

- Ready for Phase 30: wizard stepper can post quay_password/weka_password safely (SEC-01 now enforced)
- Phase 31 E2E verification should include secret-leak gate check: `kubectl get wekaappstore -o yaml` must not contain password/token/secret values

---
*Phase: 29-backend-wiring-secret-safety*
*Completed: 2026-06-24*

## Self-Check: PASSED

- `app-store-gui/webapp/main.py` contains `_is_secret_key`, `_safe_gui_variables`, `_redact_secrets`
- Annotation stamp at ~3067 uses `_safe_gui_variables(user_vars)`
- SSE emit sites at ~3116 and ~3126 use `_redact_secrets`
- `python -m py_compile app-store-gui/webapp/main.py` passes
- 52 tests pass in test_dynamic_blueprint.py
