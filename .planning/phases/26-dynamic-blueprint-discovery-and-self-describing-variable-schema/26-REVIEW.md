---
phase: 26-dynamic-blueprint-discovery-and-self-describing-variable-schema
reviewed: 2026-06-15T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - app-store-gui/webapp/main.py
  - app-store-gui/webapp/templates/blueprint.html
  - app-store-gui/tests/test_dynamic_blueprint.py
  - mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml
  - mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-06-15T00:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 26 introduces `parse_x_variables` and `find_blueprint` helpers in `main.py`, refactors `blueprint_detail` and `deploy_stream` to use dynamic blueprint discovery, adds a dynamic form in `blueprint.html` driven by the x-variables schema, and adds two fixture blueprints with x-variables blocks.

The core logic in `parse_x_variables` and `find_blueprint` is sound. The `deploy_stream` refactor correctly removes the hardcoded `app_map` and validates required variables before applying. The template form rendering logic is mostly correct, including the Jinja2 list-mutation workaround for the `ns_creds_missing` accumulator.

Two blockers were found: the `ns_creds_missing` variable is conditionally defined inside a Jinja2 block but referenced unconditionally in JavaScript below it (causes a 500 on any blueprint page without x-variables); and both new fixture YAML files declare `x-variables` but contain no `[[ ]]` substitution tokens, meaning user-supplied variables are silently discarded at render time.

---

## Critical Issues

### CR-01: `ns_creds_missing` Undefined When `variable_schema` Is Empty — 500 Error on Blueprint Pages

**File:** `app-store-gui/webapp/templates/blueprint.html:261`

**Issue:** `ns_creds_missing` is defined at line 159 only inside a `{% if variable_schema %}` block. If `variable_schema` is an empty dict (which happens whenever `find_blueprint` returns `None` — i.e., for any blueprint that has no YAML or no `x-variables` block), the `{% set ns_creds_missing = [] %}` at line 159 is never executed. The JavaScript at line 261 then references it unconditionally:

```js
const missingCreds = {{ ns_creds_missing | tojson }};
```

Jinja2's default `Undefined` object is not JSON-serializable. `json.dumps(Undefined())` raises `TypeError: Object of type Undefined is not JSON serializable`, which propagates as an unhandled 500 to the browser. Confirmed via direct interpreter test.

All existing blueprint pages that do not yet have `x-variables` blocks (e.g., `oss-rag`, `nvidia-rag`, `neuralmesh-aidp`, `glocomp-aurora`, `tokenvisor-enterprise`) will 500 on load after this change is deployed.

**Fix:** Move the `{% set ns_creds_missing = [] %}` declaration to before the `{% if variable_schema %}` block, or guard the JavaScript reference:

```jinja2
{# Before the if block, outside any conditional: #}
{% set ns_creds_missing = [] %}
{% if variable_schema %}
  ...
  {% for var_name, var_meta in variable_schema.items() %}
    ...
    {% set _ = ns_creds_missing.append(ctype) %}
    ...
  {% endfor %}
  ...
{% endif %}
```

---

### CR-02: Fixture Blueprint YAMLs Declare `x-variables` but Contain No `[[ ]]` Substitution Tokens — Variables Silently Discarded

**File:** `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml:1-57`
**File:** `mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml:1-40`

**Issue:** Both fixture files have a correct `x-variables` block declaring `namespace` (required) and `storage_class` (optional). However, the YAML body uses hardcoded namespace values (`ai-platform`, `data-platform`) throughout, with no `[[ namespace ]]` or `[[ storage_class ]]` template tokens anywhere. When `deploy_stream` renders the blueprint via:

```python
env = Environment(variable_start_string="[[", variable_end_string="]]")
template = env.from_string(raw_tpl)
rendered = template.render(**user_vars)
```

the user-supplied `namespace` value is silently ignored. The blueprint always deploys into the hardcoded namespace regardless of what the user selects in the UI. Confirmed by `grep -n '\[\[' ai-research.yaml` and `grep -n '\[\[' data-pipeline.yaml` returning zero results.

This makes the fixtures misleading as reference implementations, and tests 14 and 15 pass while validating only schema presence, not schema utility.

**Fix:** Replace hardcoded namespace strings in both fixtures with `[[ namespace ]]` tokens:

```yaml
# ai-research.yaml — apply in metadata and each target_namespace:
metadata:
  name: ai-research
  namespace: [[ namespace ]]
spec:
  appStack:
    components:
      - name: vector-db
        target_namespace: [[ namespace ]]
        ...
```

Apply the same change to `data-pipeline.yaml`. Optionally add `[[ storage_class ]]` references where a StorageClass would be configured.

---

## Warnings

### WR-01: Field Description Rendered Twice — Label Duplicates Sub-Label Hint

**File:** `app-store-gui/webapp/templates/blueprint.html:171` and `186-188`

**Issue:** For any variable that has a `description` field, the label text at line 171 is set to the description (`var_meta.get("description") or ...`), and then the same description is also rendered as a muted sub-label at lines 186-188:

```jinja2
<label ...>{{ var_meta.get("description") or (var_name | replace...) }}</label>
...
{% if var_meta.get("description") %}
  <div class="muted text-xs mt-1">{{ var_meta.get("description") }}</div>
{% endif %}
```

The description text appears twice for every field that has one. The intent is clearly to show the variable name or description as the label, and the description as a help hint. The label should use a human-readable form of `var_name`, and the description should appear only in the sub-label.

**Fix:**
```jinja2
<label class="block text-sm mb-1" for="field-{{ var_name }}">
  {{ var_name | replace("_", " ") | title }}
</label>
{% if var_meta.get("description") %}
  <div class="muted text-xs mt-1">{{ var_meta.get("description") }}</div>
{% endif %}
```

---

### WR-02: `find_blueprint` Cluster-Init Path Not Normalized with `os.path.abspath`

**File:** `app-store-gui/webapp/main.py:1287`

**Issue:** The cluster-init special case returns the path directly from `os.path.join` without wrapping in `os.path.abspath`:

```python
if app_name == "cluster-init":
    return os.path.join(blueprints_dir, "cluster_init", "app-store-cluster-init.yaml")
```

All other paths returned by `find_blueprint` go through `os.path.abspath(filepath)` at line 1307. If `blueprints_dir` is a relative path (which can happen in tests using `blueprints_dir="some/relative"` or when BLUEPRINTS_DIR resolution falls through to the relative fallback at line 156), the cluster-init path will be relative while all other returned paths are absolute. The downstream check at `deploy_stream:2330` (`if not os.path.isabs(yaml_path)`) handles this, but the inconsistency is a latent defect.

**Fix:**
```python
if app_name == "cluster-init":
    return os.path.abspath(os.path.join(blueprints_dir, "cluster_init", "app-store-cluster-init.yaml"))
```

---

### WR-03: Dead Code — Cluster-Init Namespace Fallback Is Unreachable

**File:** `app-store-gui/webapp/main.py:2325-2326`

**Issue:** Line 2322 normalizes `namespace` to always be a non-empty string, defaulting to `"default"` three ways:

```python
namespace = str(user_vars.get("namespace", "default") or "default").strip() or "default"
```

After this line, `namespace` can never be falsy. The guard two lines later is therefore unreachable dead code:

```python
if app_name == "cluster-init" and not namespace:   # ← `not namespace` is always False
    namespace = "default"
```

**Fix:** Remove lines 2325-2326.

---

## Info

### IN-01: Extra User Variables Not in `x-variables` Schema Are Silently Passed to Template

**File:** `app-store-gui/webapp/main.py:2387`

**Issue:** `deploy_stream` validates that all required variables in the schema are present, but does not restrict which variables are accepted. Any key in the user-supplied `variables` JSON is passed to the Jinja2 template via `**user_vars`:

```python
rendered = template.render(**user_vars)
```

A user who knows the template's internal variable names can inject values for tokens not declared in the `x-variables` schema, bypassing the schema entirely. With operator-controlled templates this is low risk, but it means the schema is a documentation contract, not an enforcement boundary.

**Fix:** Consider filtering `user_vars` to only keys declared in `schema` before rendering:

```python
allowed_keys = set(schema.keys()) | {"namespace"}
filtered_vars = {k: v for k, v in user_vars.items() if k in allowed_keys}
rendered = template.render(**filtered_vars)
```

---

### IN-02: `os.environ.setdefault` in Test Module Is Order-Dependent

**File:** `app-store-gui/tests/test_dynamic_blueprint.py:11`

**Issue:**

```python
os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")
import webapp.main as main
```

`setdefault` only sets the variable if it is not already set. If `webapp.main` has already been imported by another test module earlier in the pytest session (Python caches modules), the `BLUEPRINTS_DIR` that `main` resolved at import time will be whatever the first importer set (or found on disk). This can cause tests in this module to run against a different `BLUEPRINTS_DIR` than intended.

**Fix:** Use `monkeypatch.setenv` within individual tests that care about `BLUEPRINTS_DIR`, or apply it at module scope via a session-scoped fixture, so the value is predictable regardless of import order.

---

_Reviewed: 2026-06-15T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
