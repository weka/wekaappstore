---
phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema
verified: 2026-06-15T06:00:00Z
status: gaps_found
score: 4/6 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Generic blueprint.html renders dynamically and a blueprint page without x-variables (or with empty variable_schema) renders without a 500 error"
    status: failed
    reason: "ns_creds_missing is set only inside {% if variable_schema %} (line 158-212) but referenced unconditionally at line 261 as {{ ns_creds_missing | tojson }}. When variable_schema is empty or falsy, Jinja2 resolves it as Undefined, which is not JSON-serializable. This causes a TypeError 500 on any blueprint page that has no x-variables block — confirmed by direct Python interpreter test."
    artifacts:
      - path: "app-store-gui/webapp/templates/blueprint.html"
        issue: "Line 261: `const missingCreds = {{ ns_creds_missing | tojson }};` is outside the {% if variable_schema %} block (which ends at line 212). ns_creds_missing is undefined here when variable_schema is empty."
    missing:
      - "Move `{% set ns_creds_missing = [] %}` to BEFORE the `{% if variable_schema %}` block (e.g. line 157), or guard the JS line as `const missingCreds = {{ ns_creds_missing | default([]) | tojson }};`"

  - truth: "Existing production blueprints (oss-rag, openfold, nvidia-vss) are migrated to x-variables format (ROADMAP SC-5 / DYN-07)"
    status: failed
    reason: "ROADMAP success criterion 5 states: 'Existing blueprints (oss-rag, openfold, nvidia-vss) are migrated to x-variables format and continue to deploy correctly.' DYN-07 states: 'All existing blueprint YAML files that use [[var]] placeholders (oss-rag-stack.yaml, openfold-stack.yaml, any nvidia blueprint) are migrated.' Neither oss-rag-stack.yaml nor openfold-stack.yaml nor any nvidia blueprint file was found in this repo, and the plans only migrated the two test fixture files (ai-research.yaml, data-pipeline.yaml). The production blueprint files live in the external warp-blueprints repo and were not migrated. REQUIREMENTS.md still shows DYN-07 as 'Not started'."
    artifacts:
      - path: "mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml"
        issue: "Has x-variables block but contains no [[var]] substitution tokens — user-supplied variables are silently discarded at render time"
      - path: "mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml"
        issue: "Same issue — has x-variables block but no [[var]] tokens in the spec body"
    missing:
      - "Migrate oss-rag-stack.yaml, openfold-stack.yaml, and any nvidia blueprint YAML in the warp-blueprints repo to include x-variables blocks"
      - "Add [[namespace]], [[storage_class]], etc. tokens to the sample fixture spec bodies so render substitution is exercised end-to-end"
      - "Update REQUIREMENTS.md DYN-07 traceability row to 'Complete' after migration"
---

# Phase 26: Dynamic Blueprint Discovery and Self-Describing Variable Schema — Verification Report

**Phase Goal:** Blueprint YAML files with an `x-variables` block are discovered automatically from `BLUEPRINTS_DIR`; a single generic `blueprint.html` renders the install form dynamically from the schema; `type: credential` variables render WarpCredential dropdowns; the deploy route accepts all variables as a generic dict; no `main.py` or template change is required when a new blueprint is added
**Verified:** 2026-06-15T06:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| SC-1 | Adding a YAML with x-variables to BLUEPRINTS_DIR causes it to appear with correct form fields — zero main.py/template changes required | VERIFIED | `find_blueprint` walks BLUEPRINTS_DIR via `os.walk`; `blueprint_detail` calls `find_blueprint(name)` and passes `variable_schema` to template; `blueprint.html` loops `variable_schema.items()` — confirmed in code |
| SC-2 | A `type: credential` variable renders a `<select>` from `available_creds`; no ready credentials shows `/settings` link | VERIFIED | `blueprint.html` lines 163-189 implement this logic; tested by 24 passing tests including credential-type metadata parsing |
| SC-3 | `/deploy-stream` accepts `variables` JSON string, validates required fields before applying, substitutes via `[[var]]` Jinja2 | VERIFIED | `deploy_stream` signature confirmed `variables: str = "{}"` at line 2294; validation loop at lines 2339-2343; `template.render(**user_vars)` at line 2387; 8 deploy_stream tests all pass |
| SC-4 | Hardcoded `app_map` and all 7 per-variable positional params removed from `main.py` | VERIFIED | `grep -c app_map main.py` = 0; `grep -cE "storage_class: Optional\|vllm_chat_model..."` = 0; confirmed by test_deploy_stream_no_app_map_in_source passing |
| SC-5 | Existing blueprints (oss-rag, openfold, nvidia-vss) migrated to x-variables format and deploy correctly | FAILED | No production blueprint YAML files (oss-rag-stack.yaml, openfold-stack.yaml, nvidia blueprints) were migrated. Only two test fixture files were updated. REQUIREMENTS.md shows DYN-07 still "Not started". |
| SC-6 | blueprint.html renders without error for blueprints that have no x-variables block (empty variable_schema) | FAILED | `ns_creds_missing` is set only inside `{% if variable_schema %}` (line 159) but referenced unconditionally at JS line 261. When `variable_schema` is empty, Jinja2 raises `TypeError: Object of type Undefined is not JSON serializable` — confirmed by direct test. |

**Score: 4/6 truths verified**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app-store-gui/webapp/main.py` | `def parse_x_variables` helper | VERIFIED | Line 1265 — returns `{}` on all failure paths; uses `yaml.safe_load` |
| `app-store-gui/webapp/main.py` | `def find_blueprint` helper | VERIFIED | Line 1281 — walks BLUEPRINTS_DIR; special-cases cluster-init; returns `None` when no match |
| `app-store-gui/webapp/main.py` | `blueprint_detail` uses `find_blueprint`, passes `variable_schema` + `available_creds` | VERIFIED | Line 1315: `yaml_path = find_blueprint(name)`; lines 1384-1385: both keys in context dict |
| `app-store-gui/webapp/main.py` | `deploy_stream` with `variables: str = "{}"`, validation, `template.render(**user_vars)` | VERIFIED | Lines 2294, 2339-2343, 2387 — all confirmed |
| `app-store-gui/webapp/templates/blueprint.html` | Dynamic form loop over `variable_schema.items()` | VERIFIED | Line 160: `{% for var_name, var_meta in variable_schema.items() %}`; no hardcoded `storage_class` or `vllmChatModel` fields |
| `app-store-gui/webapp/templates/blueprint.html` | `ns_creds_missing` properly scoped for JS | STUB | Defined in `{% if variable_schema %}` block (line 159) but referenced unconditionally at JS line 261 — causes 500 when `variable_schema` is empty |
| `app-store-gui/tests/test_dynamic_blueprint.py` | 24 tests covering both helpers + deploy_stream integration | VERIFIED | 24 tests collected and passing |
| `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` | x-variables block as first key | VERIFIED | Line 1: `x-variables:` confirmed; `namespace` (required) + `storage_class` (optional) declared |
| `mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml` | x-variables block as first key | VERIFIED | Line 1: `x-variables:` confirmed; same schema as ai-research |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `blueprint_detail` | `find_blueprint` | `yaml_path = find_blueprint(name)` | WIRED | Line 1315 — confirmed |
| `blueprint_detail` | `parse_x_variables` | `variable_schema = parse_x_variables(_f.read())` | WIRED | Lines 1361-1364 — confirmed |
| `blueprint.html` form | `variable_schema` | `{% for var_name, var_meta in variable_schema.items() %}` | WIRED | Line 160 — confirmed |
| `blueprint.html` JS | `ns_creds_missing` | `{{ ns_creds_missing \| tojson }}` at line 261 | NOT_WIRED | Variable undefined when `variable_schema` is empty — causes 500 |
| `deploy_stream` | `find_blueprint` | `yaml_path = find_blueprint(app_name)` | WIRED | Line 2316 — confirmed |
| `deploy_stream` | `parse_x_variables` | `schema = parse_x_variables(raw_schema_text)` | WIRED | Line 2339 — confirmed |
| `POST /deploy` | `find_blueprint` | `yaml_path = find_blueprint(app_name)` | WIRED | Line 1392 — confirmed |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `blueprint.html` | `variable_schema` | `parse_x_variables(file_content)` in `blueprint_detail` | Yes — reads from disk via `find_blueprint` | FLOWING (when blueprint has x-variables) |
| `blueprint.html` | `available_creds` | `_get_credentials_by_type(ns)` async call | Yes — queries live Kubernetes CRs | FLOWING |
| `deploy_stream` render | `user_vars` | `json.loads(variables)` from HTTP param | Yes — from browser form submission | FLOWING |
| `deploy_stream` render | `rendered` | `template.render(**user_vars)` on blueprint YAML | Hollow for sample fixtures — no `[[var]]` tokens | HOLLOW — fixture YAMLs have x-variables metadata but no `[[var]]` substitution tokens; all user variables are silently discarded |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `parse_x_variables` returns empty for no x-variables | `pytest test_parse_x_variables_empty_string` | PASS | PASS |
| `find_blueprint` returns path when matching file exists | `pytest test_find_blueprint_finds_matching_yaml` | PASS | PASS |
| `deploy_stream` rejects missing required var | `pytest test_deploy_stream_validates_required_variables` | PASS | PASS |
| `deploy_stream` signature has no positional params | `pytest test_deploy_stream_signature_uses_variables_param` | PASS | PASS |
| `ns_creds_missing` undefined outside if-block | `python3 -c "from jinja2 import Environment; env=Environment(); t=env.from_string('{% if variable_schema %}{% set ns_creds_missing = [] %}{% endif %}const x={{ ns_creds_missing|tojson}};'); t.render(variable_schema={})"` | TypeError: Object of type Undefined is not JSON serializable | FAIL |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| DYN-01 | 26-01 | Any YAML with x-variables block in BLUEPRINTS_DIR is discoverable — no restart/code change | SATISFIED | `find_blueprint` uses `os.walk` scan; no cache or registry required |
| DYN-02 | 26-01 | x-variables block supports `type`, `required`, `description`, `placeholder` fields | SATISFIED | `parse_x_variables` returns the raw metadata dict; template reads all four fields |
| DYN-03 | 26-01 | Single generic `blueprint.html` renders install form by looping x-variables schema | SATISFIED | Dynamic form loop confirmed at line 160 |
| DYN-04 | 26-01 | `type: credential` variable renders `<select>` from ready WarpCredentials; no credentials shows `/settings` link | SATISFIED | Lines 162-189 in blueprint.html implement both cases |
| DYN-05 | 26-02 | `/deploy-stream` accepts `variables` JSON string, validates required fields before applying | SATISFIED | Lines 2308-2343 in main.py |
| DYN-06 | 26-02 | Hardcoded `app_map` and all 7 per-variable positional params removed from `main.py` | SATISFIED | `grep -c app_map main.py` = 0; per-variable Optional params = 0 |
| DYN-07 | 26-02 | Existing blueprint YAMLs (oss-rag-stack.yaml, openfold-stack.yaml, nvidia blueprints) migrated to include x-variables blocks | BLOCKED | Production blueprint YAMLs are in the external warp-blueprints repo (not this repo) and were NOT migrated. Only test fixture files (ai-research.yaml, data-pipeline.yaml) were updated. REQUIREMENTS.md shows "Not started". |
| DYN-08 | 26-01 | Blueprint install pages surface pre-flight credential check; Install button disabled when required credentials absent | SATISFIED | Deploy button disabled at template render time (`{% if not variable_schema %}disabled`) and at DOMContentLoaded via `missingCreds.length > 0` check — however, the `ns_creds_missing` 500 bug (CR-01) means this only works when `variable_schema` is non-empty |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `app-store-gui/webapp/templates/blueprint.html` | 261 | `{{ ns_creds_missing \| tojson }}` references Jinja2 variable set only inside `{% if variable_schema %}` block (lines 158-212) | BLOCKER | TypeError 500 on any blueprint page where `variable_schema` is empty — including all blueprints that predate Phase 26 and have no x-variables block |
| `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` | all | Has `x-variables` schema but no `[[var]]` substitution tokens in the spec body | WARNING | All user-supplied variables are silently ignored at render time; the deploy flow is not end-to-end exercised by these fixtures |
| `mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml` | all | Same as ai-research.yaml — x-variables declared but no `[[var]]` tokens | WARNING | Same silent-discard issue |

---

### Human Verification Required

None needed beyond the identified code gaps. The two blockers are programmatically verifiable and confirmed above.

---

### Gaps Summary

**Two blockers prevent the phase goal from being fully achieved:**

**Blocker 1: `ns_creds_missing` 500 error (blueprint.html line 261)**

The `ns_creds_missing` Jinja2 variable is initialized inside `{% if variable_schema %}` at line 159 but referenced unconditionally in the JavaScript block at line 261. Any blueprint whose YAML is not found by `find_blueprint` (app not in BLUEPRINTS_DIR, or YAML has no x-variables block) will cause `blueprint_detail` to pass `variable_schema={}` to the template, leaving `ns_creds_missing` as Jinja2's `Undefined`. The `| tojson` filter on `Undefined` raises `TypeError: Object of type Undefined is not JSON serializable`, rendering the page as a 500.

Fix: Add `{% set ns_creds_missing = [] %}` before line 158 (outside the `{% if variable_schema %}` block), or change line 261 to `{{ ns_creds_missing | default([]) | tojson }}`.

**Blocker 2: DYN-07 / SC-5 — Production blueprint migration not done**

ROADMAP SC-5 and DYN-07 both require that `oss-rag-stack.yaml`, `openfold-stack.yaml`, and nvidia blueprint YAMLs be migrated to include correct `x-variables` blocks. These production blueprint files live in the external `warp-blueprints` git repo (not this repo) and were not modified. The plans only migrated the two test fixture files (`ai-research.yaml`, `data-pipeline.yaml`). REQUIREMENTS.md still shows DYN-07 as "Not started."

Note: The sample fixture migration is also incomplete — the newly-added `x-variables` blocks in both fixture files declare variables (`namespace`, `storage_class`) but the YAML spec body contains no `[[namespace]]` or `[[storage_class]]` substitution tokens, meaning the variables are silently discarded at render time and the deploy path is not exercised end-to-end by these fixtures.

---

_Verified: 2026-06-15T06:00:00Z_
_Verifier: Claude (gsd-verifier)_
