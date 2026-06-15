---
phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema
plan: 02
subsystem: ui
tags: [fastapi, jinja2, yaml, deploy-stream, schema-validation, blueprint-variables]

# Dependency graph
requires:
  - phase: 26-01
    provides: parse_x_variables and find_blueprint helpers in main.py
provides:
  - deploy_stream with generic variables: str = "{}" param (replaces 7 hardcoded positional params)
  - Required-field validation in deploy_stream before apply (schema-driven, cluster-init exempt)
  - POST /deploy route using find_blueprint instead of local app_map dict
  - x-variables blocks in ai-research.yaml and data-pipeline.yaml sample fixtures
  - Integration tests 11-15 covering full schema-validation flow without live cluster
affects:
  - blueprint.html JS submit handler (already sending variables JSON — now matched by receiver)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Generic variables JSON dict pattern: deploy_stream accepts any key-value pairs from form; no main.py change needed for new blueprints
    - Schema-driven required-field validation: parse_x_variables schema gates apply; cluster-init exempt
    - template.render(**user_vars): full dict unpacked into Jinja2 render call

key-files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml
    - mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml
    - app-store-gui/tests/test_dynamic_blueprint.py

key-decisions:
  - "deploy_stream extracts namespace from user_vars dict; falls back to 'default' if absent or empty"
  - "Required-field validation reads schema via parse_x_variables on the blueprint file (same file used for render); cluster-init is hard-coded exempt"
  - "template.render(**user_vars) passes all keys from the JSON dict; Jinja2 ignores undefined vars silently"
  - "POST /deploy kept minimal (legacy path): signature unchanged, only app_map replaced with find_blueprint"
  - "Sample fixtures use namespace (required) + storage_class (optional) as canonical x-variables examples"

requirements-completed: [DYN-05, DYN-06, DYN-07]

# Metrics
duration: ~18min
completed: 2026-06-15
---

# Phase 26 Plan 02: Deploy Stream Refactor and Sample Fixture Migration Summary

**Generic variables JSON dict replaces 7 hardcoded positional params in deploy_stream; schema-based required-field validation runs before apply; x-variables blocks added to both sample blueprint fixtures as canonical format documentation**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-15T05:05:00Z
- **Completed:** 2026-06-15T05:23:00Z
- **Tasks:** 2 (Task 1: deploy_stream refactor, Task 2: fixture migration + integration tests)
- **Files modified:** 4
- **Tests added:** 14 (tests 1-8 Task 1, tests 11-15 Task 2; all pass)

## Accomplishments

### Task 1: deploy_stream refactor

- Removed 7 hardcoded positional params from `deploy_stream`: `namespace`, `storage_class`, `vllm_chat_model`, `vllm_embed_model`, `vllm_model`, `weka_cluster_filesystem`, `openfold_storage_capacity`, `deployment_name`
- New signature: `async def deploy_stream(request, app_name=None, variables="{}")`
- `json.loads(variables)` parses the JSON dict; on failure emits `{"type": "error", "message": "Invalid variables JSON"}`
- `find_blueprint(app_name)` replaces local `app_map` dict — returns `None` for unknown apps (SSE error)
- `namespace = str(user_vars.get("namespace", "default") or "default").strip() or "default"` extracted from dict
- Required-field validation loop: `for var_name, meta in schema.items(): if meta.get("required") and not str(user_vars.get(var_name, "") or "").strip()` — yields SSE error and returns before apply
- cluster-init is hard-coded exempt from required-field validation
- `template.render(**user_vars)` — full dict unpacked; replaces 10-line hardcoded `template.render(namespace=..., storage_class=..., ...)` call
- `_norm()` helper and all 7 normalized variables removed (no longer needed)
- Removed `app_map` local dict from `POST /deploy` route; replaced with `yaml_path = find_blueprint(app_name)`

**All 3 `app_map` dicts removed from main.py:** blueprint_detail (Plan 26-01), deploy_stream (this plan), POST /deploy (this plan).

### Task 2: Fixture migration + integration tests

- `ai-research.yaml`: `x-variables` block added as first document key (line 1); namespace (required) + storage_class (optional)
- `data-pipeline.yaml`: `x-variables` block added as first document key (line 1); namespace (required) + storage_class (optional)
- `apiVersion` preserved immediately after the x-variables block in both files
- Integration tests 11-15 added to `test_dynamic_blueprint.py`:
  - Test 11: Required variable with empty value causes SSE error; apply not called
  - Test 12: Optional variable absent does NOT cause error
  - Test 13: cluster-init bypasses required-field validation
  - Test 14: `parse_x_variables` on ai-research.yaml returns non-empty dict
  - Test 15: `parse_x_variables` on data-pipeline.yaml returns non-empty dict

## New deploy_stream Signature

```python
@app.get("/deploy-stream/{app_name}")
@app.get("/deploy-stream")
async def deploy_stream(
    request: Request,
    app_name: Optional[str] = None,
    variables: str = "{}",
):
```

## Required-Field Validation Logic

```python
schema = parse_x_variables(raw_schema_text)
for var_name, meta in schema.items():
    if meta.get("required") and not str(user_vars.get(var_name, "") or "").strip():
        yield sse_event({"type": "error", "message": f"Required variable missing: {var_name}"})
        return
```

Skipped entirely when `app_name == "cluster-init"`.

## Sample Fixture x-variables Format

```yaml
x-variables:
  namespace:
    type: string
    required: true
    description: "Target Kubernetes namespace for the <app> stack"
  storage_class:
    type: string
    required: false
    description: "Kubernetes StorageClass for <purpose> volumes"
    placeholder: "e.g. weka-storageclass"

apiVersion: warp.io/v1alpha1
...
```

**Runtime blueprint authors must add `x-variables` blocks** to their YAML files in the `warp-blueprints` repo following this exact format. The `x-variables` key must appear before `apiVersion` (first top-level key in the YAML document). The sample fixtures `ai-research.yaml` and `data-pipeline.yaml` serve as canonical reference implementations.

## Task Commits

Each task committed atomically following TDD (RED then GREEN):

1. **Task 1 RED:** `b180ebb` — failing tests for deploy_stream refactor
2. **Task 1 GREEN:** `b22c5cc` — refactor deploy_stream, remove all app_map dicts
3. **Task 2 RED:** `48350d1` — integration tests for schema-validation flow and fixture x-variables
4. **Task 2 GREEN:** `554a302` — x-variables blocks in sample fixtures

## Test Summary

**Total tests in test_dynamic_blueprint.py:** 24 (10 from Plan 26-01, 14 added in Plan 26-02)

All 24 tests pass. All 117 MCP server tests pass (fixture update did not break MCP blueprint tools).

## Verification Results

```
python -m py_compile app-store-gui/webapp/main.py -> PASS
grep -cE "storage_class: Optional|..." -> 0 (all per-variable params removed)
grep -c "app_map" app-store-gui/webapp/main.py -> 0 (all three app_map dicts removed)
grep -c "x-variables:" ai-research.yaml -> 1
grep -c "x-variables:" data-pipeline.yaml -> 1
grep -n "^x-variables:" ai-research.yaml -> line 1 (first key)
grep -n 'variables: str = "{}"' main.py -> 1 match (deploy_stream)
grep -n "template.render(**user_vars)" main.py -> 1 match
pytest app-store-gui/tests/test_dynamic_blueprint.py -v -> 24 passed
pytest mcp-server/tests/ -v -> 117 passed
```

## Deviations from Plan

None — plan executed exactly as written.

The only notable observation: Plan 26-01 was executed on a different worktree branch. This worktree was initialized from `c05d3a7` (before 26-01 merged). A `git merge main` was performed at the start of execution to pull in the 26-01 work (`aaa4ba1`) before implementing 26-02.

## Known Stubs

None. All variables flow end-to-end from the form to the template render. The sample fixtures use real Helm chart references. No hardcoded placeholder values in the deploy path.

## Threat Surface

No new surface beyond what is documented in the plan's threat_model. Key mitigations applied:
- T-26-05 (key injection): `template.render(**user_vars)` — Jinja2 `[[var]]` delimiters only substitute declared tokens; unknown keys silently ignored
- T-26-08 (DoS via large JSON): FastAPI/uvicorn HTTP request size limits apply before handler runs

## Note for Blueprint Authors

To make a blueprint work with the Phase 26 dynamic deploy flow, add an `x-variables` block as the **first top-level key** in the YAML file (before `apiVersion`). The `find_blueprint` helper requires this block to locate the file by `app_name`. Blueprints without an `x-variables` block are invisible to `find_blueprint` and will return `"Unknown app"` from `deploy_stream`.

## Self-Check: PASSED

- `parse_x_variables` exists in main.py: confirmed at line 1265
- `find_blueprint` exists in main.py: confirmed at line 1281
- `variables: str = "{}"` in deploy_stream: confirmed at line 2294
- `json.loads(variables)` in deploy_stream: confirmed at line 2310
- `template.render(**user_vars)` in deploy_stream: confirmed at line 2387
- `app_map` count in main.py: 0
- `x-variables:` at line 1 in ai-research.yaml: confirmed
- `x-variables:` at line 1 in data-pipeline.yaml: confirmed
- All 24 tests in test_dynamic_blueprint.py pass
- All 117 MCP server tests pass
- `python -m py_compile app-store-gui/webapp/main.py` exits 0
- Commits b180ebb, b22c5cc, 48350d1, 554a302 present in git log

---
*Phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema*
*Completed: 2026-06-15*
