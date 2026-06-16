---
phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema
verified: 2026-06-17T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/6
  gaps_closed:
    - "Generic blueprint.html renders without 500 error when variable_schema is empty — ns_creds_missing now defined unconditionally at line 157, before {% if variable_schema %}"
    - "Fixture YAMLs ai-research.yaml and data-pipeline.yaml contain [[namespace]] tokens (3 and 2 occurrences respectively); deploy substitution path exercised end-to-end"
  gaps_remaining: []
  regressions: []
---

# Phase 26: Dynamic Blueprint Discovery and Self-Describing Variable Schema — Verification Report

**Phase Goal:** Blueprint YAML files with an `x-variables` block are discovered automatically from `BLUEPRINTS_DIR`; a single generic `blueprint.html` renders the install form dynamically from the schema; `type: credential` variables render WarpCredential dropdowns; the deploy route accepts all variables as a generic dict; no `main.py` or template change is required when a new blueprint is added.
**Verified:** 2026-06-17T00:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (Plan 03)

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| SC-1 | Adding a YAML with x-variables to BLUEPRINTS_DIR causes it to appear with correct form fields — zero main.py/template changes required | VERIFIED | `find_blueprint` walks BLUEPRINTS_DIR via `os.walk`; `blueprint_detail` passes `variable_schema` to template; `blueprint.html` loops `variable_schema.items()` |
| SC-2 | A `type: credential` variable renders a `<select>` from `available_creds`; no ready credentials shows `/settings` link | VERIFIED | `blueprint.html` lines 164-190 implement credential select and fallback link |
| SC-3 | `/deploy-stream` accepts `variables` JSON string, validates required fields before applying, substitutes via `[[var]]` Jinja2 | VERIFIED | `deploy_stream` signature has `variables: str = "{}"`, validation loop at lines 2324-2343, `template.render(**user_vars)` at line 2390 |
| SC-4 | Hardcoded `app_map` and all 7 per-variable positional params removed from `main.py` | VERIFIED | `grep -c app_map main.py` = 0; per-variable Optional params = 0 |
| SC-5 | Fixture YAMLs exercised end-to-end with [[namespace]] substitution tokens | VERIFIED | ai-research.yaml: 3 `[[` occurrences; data-pipeline.yaml: 2 `[[` occurrences; hardcoded `ai-platform`/`data-platform` strings fully replaced. DYN-07 production blueprint migration (external warp-blueprints repo) documented as out-of-scope in REQUIREMENTS.md. |
| SC-6 | blueprint.html renders without error for blueprints that have no x-variables block (empty variable_schema) | VERIFIED | `{% set ns_creds_missing = [] %}` moved to line 157 — before the `{% if variable_schema %}` block at line 159. JS reference at line 262 now always has a defined value. All 24 dynamic blueprint tests pass. |

**Score: 6/6 truths verified**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app-store-gui/webapp/main.py` | `def parse_x_variables` helper | VERIFIED | Line 1265 — returns `{}` on all failure paths |
| `app-store-gui/webapp/main.py` | `def find_blueprint` helper with `os.path.abspath` for cluster-init | VERIFIED | Line 1287: `return os.path.abspath(os.path.join(blueprints_dir, "cluster_init", ...))` — consistent with all other return paths |
| `app-store-gui/webapp/main.py` | `deploy_stream` dead code guard removed | VERIFIED | `grep "if app_name == .cluster-init. and not namespace"` returns empty — dead code removed |
| `app-store-gui/webapp/templates/blueprint.html` | `{% set ns_creds_missing = [] %}` before `{% if variable_schema %}` | VERIFIED | Line 157 is the unconditional `{% set %}`, line 159 is `{% if variable_schema %}` — correct order confirmed |
| `app-store-gui/webapp/templates/blueprint.html` | Labels use `var_name` title-cased, not description | VERIFIED | Lines 172 and 193: `{{ var_name \| replace("_", " ") \| title }}` — no `var_meta.get("description")` in label elements |
| `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` | x-variables block + `[[namespace]]` tokens in spec body | VERIFIED | 3 `[[` occurrences: `metadata.namespace`, `target_namespace` (vector-db), `target_namespace` (research-api) |
| `mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml` | x-variables block + `[[namespace]]` tokens in spec body | VERIFIED | 2 `[[` occurrences: `metadata.namespace`, `target_namespace` (spark-operator) |
| `app-store-gui/tests/test_dynamic_blueprint.py` | 24 tests all passing | VERIFIED | `pytest app-store-gui/tests/test_dynamic_blueprint.py -q` → 24 passed |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `blueprint_detail` | `find_blueprint` | `yaml_path = find_blueprint(name)` | WIRED | Line 1315 — confirmed |
| `blueprint_detail` | `parse_x_variables` | `variable_schema = parse_x_variables(_f.read())` | WIRED | Lines 1361-1364 — confirmed |
| `blueprint.html` form | `variable_schema` | `{% for var_name, var_meta in variable_schema.items() %}` | WIRED | Line 161 — confirmed |
| `blueprint.html` JS | `ns_creds_missing` | `{{ ns_creds_missing \| tojson }}` at line 262 | WIRED | Variable unconditionally set at line 157 — no longer causes 500 |
| `deploy_stream` | `find_blueprint` | `yaml_path = find_blueprint(app_name)` | WIRED | Line 2316 — confirmed |
| `deploy_stream` | `parse_x_variables` | `schema = parse_x_variables(raw_schema_text)` | WIRED | Line 2339 — confirmed |
| `POST /deploy` | `find_blueprint` | `yaml_path = find_blueprint(app_name)` | WIRED | Line 1392 — confirmed |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 24 dynamic blueprint tests pass | `pytest app-store-gui/tests/test_dynamic_blueprint.py -q` | 24 passed in 0.74s | PASS |
| `ns_creds_missing` defined unconditionally | `grep -n "ns_creds_missing" blueprint.html \| head -1` | Line 157 — before `{% if variable_schema %}` at line 159 | PASS |
| Dead code guard absent | `grep "if app_name == .cluster-init. and not namespace" main.py` | No output | PASS |
| cluster-init path uses abspath | `grep "os.path.abspath.*cluster_init" main.py` | Line 1287 confirmed | PASS |
| ai-research.yaml has `[[` tokens | `grep -c '\[\[' ai-research.yaml` | 3 | PASS |
| data-pipeline.yaml has `[[` tokens | `grep -c '\[\[' data-pipeline.yaml` | 2 | PASS |
| No hardcoded `ai-platform` in ai-research.yaml | `grep 'ai-platform' ai-research.yaml` | No output | PASS |
| No hardcoded `data-platform` in data-pipeline.yaml | `grep 'data-platform' data-pipeline.yaml` | No output | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| DYN-01 | 26-01 | Any YAML with x-variables block in BLUEPRINTS_DIR is discoverable — no restart/code change | SATISFIED | `find_blueprint` uses `os.walk` scan |
| DYN-02 | 26-01 | x-variables block supports `type`, `required`, `description`, `placeholder` fields | SATISFIED | `parse_x_variables` returns raw metadata dict; template reads all four fields |
| DYN-03 | 26-01 | Single generic `blueprint.html` renders install form by looping x-variables schema | SATISFIED | Dynamic form loop at line 161 |
| DYN-04 | 26-01 | `type: credential` variable renders `<select>` from ready WarpCredentials; no credentials shows `/settings` link | SATISFIED | Lines 164-190 in blueprint.html |
| DYN-05 | 26-02 | `/deploy-stream` accepts `variables` JSON string, validates required fields before applying | SATISFIED | Lines 2308-2343 in main.py |
| DYN-06 | 26-02 | Hardcoded `app_map` and all 7 per-variable positional params removed from `main.py` | SATISFIED | `grep -c app_map main.py` = 0 |
| DYN-07 | 26-02/03 | Fixture YAMLs migrated to [[namespace]] tokens; production blueprints (oss-rag, openfold, nvidia) are in external warp-blueprints repo | PARTIAL (external repo) | ai-research.yaml and data-pipeline.yaml migrated; REQUIREMENTS.md updated with footnote explaining external-repo constraint |
| DYN-08 | 26-01/03 | Blueprint install pages surface pre-flight credential check; Install button disabled when required credentials absent | SATISFIED | `ns_creds_missing` 500 bug (SC-6) fixed; credential check JS at line 262 now always has a defined list |

---

### Human Verification Required

None. All gaps were programmatically verifiable and are now confirmed closed.

---

### Gaps Summary

All prior blockers are closed. No gaps remain.

- **Blocker 1 (SC-6):** `ns_creds_missing` moved to line 157 (before `{% if variable_schema %}`). The JS reference at line 262 always has a defined value. 24 tests pass.
- **Blocker 2 (SC-5 / DYN-07):** Fixture YAMLs now contain `[[namespace]]` tokens (ai-research: 3, data-pipeline: 2). Hardcoded namespace strings fully replaced. Production blueprint migration (external `warp-blueprints` repo) is documented as an out-of-scope external concern in REQUIREMENTS.md — not a gap in this repo.
- **WR-01 (label duplication):** Labels use `var_name | replace("_", " ") | title` only; `var_meta.get("description")` removed from label elements.
- **WR-02 (cluster-init abspath):** `find_blueprint` cluster-init path now wrapped in `os.path.abspath()`.
- **WR-03 (dead code):** Unreachable `if app_name == "cluster-init" and not namespace:` guard removed from `deploy_stream`.

---

_Verified: 2026-06-17T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
