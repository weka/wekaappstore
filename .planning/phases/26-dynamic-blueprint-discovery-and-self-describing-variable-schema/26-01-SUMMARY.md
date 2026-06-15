---
phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema
plan: 01
subsystem: ui
tags: [fastapi, jinja2, yaml, blueprint-discovery, dynamic-forms]

# Dependency graph
requires:
  - phase: 25-blueprint-credential-selector-sdk
    provides: _get_credentials_by_type helper and credentials_by_type context key in blueprint_detail
provides:
  - parse_x_variables helper that extracts x-variables schema from raw blueprint YAML text
  - find_blueprint helper that walks BLUEPRINTS_DIR to locate a blueprint YAML by app name
  - blueprint_detail route updated to use find_blueprint and pass variable_schema + available_creds
  - blueprint.html generic Configure card that renders fields dynamically from variable_schema
  - variables JSON convention for /deploy-stream endpoint (Plan 26-02 implements receiver)
affects:
  - 26-02 (depends on find_blueprint and the variables JSON param convention)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - x-variables YAML schema block: blueprint authors declare variable metadata in YAML; GUI renders form dynamically
    - parse_x_variables + find_blueprint composition: two pure helpers compose into blueprint_detail route
    - Jinja2 ns_creds_missing list pattern: uses set/append to collect missing required credentials during template rendering

key-files:
  created:
    - app-store-gui/tests/test_dynamic_blueprint.py
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/webapp/templates/blueprint.html

key-decisions:
  - "parse_x_variables returns {} on any failure (including YAML parse errors) — never raises; callers treat empty dict as no schema"
  - "find_blueprint special-cases cluster-init to its fixed path; all other names use os.walk scan matching stem or parent dir name"
  - "blueprint_detail app_map removed; deploy and deploy_stream still use their own app_map until Plan 26-02 migrates them"
  - "blueprint.html uses available_creds (alias for credentials_by_type) to keep template variable names self-documenting"
  - "ns_creds_missing Jinja2 list built via set _ = ns_creds_missing.append() pattern; tojson filter serializes it to JS"

patterns-established:
  - "x-variables block: top-level YAML key maps variable names to {type, required, description, placeholder, credential_type} metadata"
  - "find_blueprint(app_name, blueprints_dir=None) signature: explicit blueprints_dir param for test isolation (no monkeypatch needed)"
  - "Dynamic form rendering: for var_name, var_meta in variable_schema.items() — string -> input, credential -> select from available_creds"

requirements-completed: [DYN-01, DYN-02, DYN-03, DYN-04, DYN-08]

# Metrics
duration: 7min
completed: 2026-06-15
---

# Phase 26 Plan 01: Dynamic Blueprint Discovery and Self-Describing Variable Schema Summary

**parse_x_variables + find_blueprint helpers enabling blueprint.html Configure card to render dynamically from x-variables YAML schema with automatic credential selects and pre-flight disabled state**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-06-15T04:56:52Z
- **Completed:** 2026-06-15T05:03:13Z
- **Tasks:** 2 (Task 1: TDD helpers, Task 2: route + template update)
- **Files modified:** 3

## Accomplishments

- Added `parse_x_variables(yaml_text: str) -> dict` to `main.py` — extracts `x-variables` block from raw YAML, returns `{}` on any parse failure
- Added `find_blueprint(app_name: str, blueprints_dir: str = None) -> Optional[str]` to `main.py` — walks BLUEPRINTS_DIR, matches by file stem or parent dir name, special-cases cluster-init
- `blueprint_detail` now resolves `yaml_path = find_blueprint(name)` and passes `variable_schema` + `available_creds` to the template context
- `blueprint.html` Configure card replaced with dynamic Jinja2 form loop — renders `<input>` for string vars and `<select>` from `available_creds` for credential vars; Deploy button disabled when schema is empty or required credentials are missing
- JS submit handler generalized to serialize all form values into a `variables` JSON dict for the new `/deploy-stream?variables=...` API

## Function Signatures Shipped

```python
def parse_x_variables(yaml_text: str) -> dict:
    """Extract the x-variables schema block from raw blueprint YAML text. Returns {} on any parse failure."""

def find_blueprint(app_name: str, blueprints_dir: str = None) -> Optional[str]:
    """Scan BLUEPRINTS_DIR for a blueprint YAML file with an x-variables block matching app_name. Returns absolute path or None."""
```

**Insertion points in main.py:** `parse_x_variables` at line 1265, `find_blueprint` at line 1281, both before `@app.get("/blueprint/{name}")` decorator at line 1313.

## New Template Context Keys Added to blueprint_detail

| Key | Value | Purpose |
|-----|-------|---------|
| `variable_schema` | `parse_x_variables(file_content)` | Dict of var_name -> metadata dict from x-variables YAML block |
| `available_creds` | alias for `credentials_by_type` | Self-documenting alias for the generic template's credential selects |

## Jinja2 Pattern for Dynamic Form Rendering in blueprint.html

```jinja2
{% set ns_creds_missing = [] %}
{% for var_name, var_meta in variable_schema.items() %}
  {% if var_name == "namespace" %}
    {# always rendered as fixed read-only field above #}
  {% elif var_meta.get("type") == "credential" %}
    {# renders <select> from available_creds[credential_type] or missing notice #}
  {% else %}
    {# renders <input> with placeholder from var_meta #}
  {% endif %}
{% endfor %}
```

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for parse_x_variables and find_blueprint** - `798d4f4` (test)
2. **Task 1 GREEN: parse_x_variables and find_blueprint helpers** - `2731ea9` (feat)
3. **Task 2: blueprint_detail route update and blueprint.html rewrite** - `42070da` (feat)

**Plan metadata:** see final commit below

## Files Created/Modified

- `app-store-gui/webapp/main.py` - Added `parse_x_variables` and `find_blueprint` helpers before `blueprint_detail`; updated `blueprint_detail` to use `find_blueprint`, extract `variable_schema`, and pass both new context keys
- `app-store-gui/webapp/templates/blueprint.html` - Replaced hardcoded vLLM fields with dynamic `variable_schema` loop; new generic JS submit handler using `variables: JSON.stringify(variables)`
- `app-store-gui/tests/test_dynamic_blueprint.py` - 10 unit tests covering both helpers (empty YAML, missing key, string vars, credential vars, parse errors, path matching, cluster-init special case)

## Decisions Made

- `parse_x_variables` uses `yaml.safe_load` (already imported) — safe against arbitrary Python object construction from blueprint-authored YAML
- `find_blueprint` uses `os.walk` bounded to BLUEPRINTS_DIR; the `name` parameter is only used as a comparison string (never as a path component), preventing path traversal
- `blueprint.html` renders inline (no `_credential_macros.html` import) to keep the generic template self-contained and avoid tight coupling to the macro library
- Namespace field always rendered as read-only first; `namespace` key in `variable_schema` is skipped during the loop
- Deploy button is `disabled` at template render time when `not variable_schema`; additionally disabled at DOMContentLoaded when `missingCreds.length > 0`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 7 pre-existing test failures in `test_credentials_api.py` (`list_credentials` and `create_credential` function signature mismatches) and 5 in `planning/` integration tests — confirmed pre-existing before any changes, out of scope per CLAUDE.md guidance.

## Note for Plan 26-02

Plan 26-02 depends on:
1. `find_blueprint(app_name)` — already available in `main.py` for the deploy route to use
2. The `variables` JSON string param convention: `GET /deploy-stream?app_name=X&variables={"key":"value"}` — established in `blueprint.html` JS submit handler; Plan 26-02 implements the `deploy_stream` receiving end

## Threat Surface

No new surface beyond what's documented in the plan's threat model. Jinja2 auto-escaping is active in FastAPI's Jinja2Templates, protecting against XSS from blueprint YAML content. `os.walk` is bounded to `BLUEPRINTS_DIR`.

## Self-Check: PASSED

- `parse_x_variables` exists in `main.py`: confirmed at line 1265
- `find_blueprint` exists in `main.py`: confirmed at line 1281
- Both before `@app.get("/blueprint/{name}")` at line 1313: confirmed
- `variable_schema` in template context: confirmed at line 1384
- `available_creds` in template context: confirmed at line 1385
- `for var_name, var_meta in variable_schema.items()` in blueprint.html: count=1
- `variables: JSON.stringify(variables)` in blueprint.html: count=1
- All 10 new tests pass
- `python -m py_compile app-store-gui/webapp/main.py` exits 0
- Commits: 798d4f4, 2731ea9, 42070da all present in git log

---
*Phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema*
*Completed: 2026-06-15*
