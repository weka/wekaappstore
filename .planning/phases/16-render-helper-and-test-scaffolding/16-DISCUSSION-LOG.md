# Phase 16: render() Helper and Test Scaffolding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `16-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 16-render-helper-and-test-scaffolding
**Areas discussed:** Pre-scan guard pattern, render() error type, Test setup wiring, JSON-safety test fixture

---

## Pre-scan Guard Pattern

### Q1: What pattern should trigger render-vs-skip?

| Option | Description | Selected |
|--------|-------------|----------|
| `${` substring only | Skip render when `'${' not in text`. Simplest, fastest. Bare `$VAR` shell-style passes through untouched. Matches v5.0 syntax. | ✓ |
| Custom Template subclass + `${` check | Subclass `string.Template` to ONLY match `${VAR}` and `$$`, ignoring bare `$identifier`. Solves mixing-case cleanly; adds ~5 lines of regex override. | |
| Regex match on full Template grammar | Use `re.search(r'\$\{[^}]+\}\|\$\$\|\$[A-Za-z_]', text)`. More precise gate, but bare `$VAR` still fails at substitute time. | |

**User's choice:** `${` substring only.
**Notes:** Mixing-case landmine acknowledged and pushed to Deferred. Custom subclass is reconsidered if AIDP migration hits the landmine in production.

---

### Q2: When should pre-scan run relative to the empty-variables check (OP-05)?

| Option | Description | Selected |
|--------|-------------|----------|
| Empty-vars first, then pre-scan | `if not variables: return text` first, then `if '${' not in text: return text`. Cheapest path for the no-vars case (most common during reconcile of pre-v5.0 CRs). | ✓ |
| Pre-scan first, then empty-vars | Check `${` first — saves work even when variables is non-empty but text has no placeholders. | |
| Combined single check | `if not variables or '${' not in text: return text` — one branch, smallest code surface. | |

**User's choice:** Empty-vars first, then pre-scan.

---

### Q3: Backward-compat regression test fixture?

| Option | Description | Selected |
|--------|-------------|----------|
| Both — literal + real cluster_init excerpt | Test 1: `render('$CRDS && $CRD', {})` and `render('$CRDS && $CRD', {'namespace': 'foo'})` both unchanged. Test 2: load actual shell-script from `cluster_init/app-store-cluster-init.yaml` and assert byte-identical render. | ✓ |
| Literal only | Just `render('$CRDS && $CRD', {})` unchanged. Faster but doesn't catch real-world drift. | |
| Literal + synthetic shell script with all 4 vars | Hand-crafted multi-line script containing `$CRDS`, `$CRD`, `$MISSING`, `$GATEWAY_API_URL`. No fixture file load; covers all 4 named risks. | |

**User's choice:** Both — literal + real cluster_init excerpt.

---

## render() Error Type

### Q1: What exception does render() raise on undefined/malformed?

| Option | Description | Selected |
|--------|-------------|----------|
| `ValueError` with descriptive message | Catches `KeyError`/`ValueError` from `Template.substitute()`, raises `ValueError(...)`. Stdlib idiomatic; no kopf in this layer. Phase 18 wraps with component context. | ✓ |
| Custom RenderError class | `class RenderError(Exception)` with `.var_name` attribute. Phase 18 introspects rather than parses messages. Adds one class. | |
| Re-raise `KeyError` / `ValueError` as-is | `raise type(e)(descriptive_msg) from e`. Phase 18 must catch both types separately. | |

**User's choice:** `ValueError` with descriptive message.

---

### Q2: Error message format?

| Option | Description | Selected |
|--------|-------------|----------|
| Variable name + cause | `f"Undefined variable: ${{{name}}}"` for undefined; `f"Malformed placeholder in template: {original_error}"` for malformed. | ✓ |
| Variable name only, terse | `raise ValueError(name)`. Phase 18 must add ALL context. | |
| Full structured prefix — 'render error: ...' | All messages start with `render error:`. Pattern-matchable. | |

**User's choice:** Variable name + cause.

---

### Q3: Exception chaining?

| Option | Description | Selected |
|--------|-------------|----------|
| `raise ValueError(msg) from e` | Preserves underlying `KeyError`/`ValueError` traceback. Best for debugging when Phase 18's PermanentError surfaces. | ✓ |
| `raise ValueError(msg) from None` | Hide original exception. Cleaner traceback in production logs. | |
| Plain `raise ValueError(msg)` | Default Python behavior — implicit context. | |

**User's choice:** `raise ValueError(msg) from e`.

---

## Test Setup Wiring

### Q1: Where does pytest get installed?

| Option | Description | Selected |
|--------|-------------|----------|
| New `operator_module/requirements-dev.txt` | Production deps stay minimal. Dev/test deps split. Image build doesn't pull pytest. | ✓ |
| Add pytest to `operator_module/requirements.txt` | Single deps file, mirrors `mcp-server/requirements.txt` (which has pytest alongside production). Operator container ships pytest. | |
| Reuse `mcp-server/requirements.txt` install | No new file; cross-component coupling. | |

**User's choice:** New `operator_module/requirements-dev.txt`.

---

### Q2: How does pytest discover `operator_module/main.py`?

| Option | Description | Selected |
|--------|-------------|----------|
| `operator_module/tests/conftest.py` adds parent to sys.path | Mirrors `mcp-server/tests/conftest.py`. Tests do `from main import render`. Self-contained. Run: `pytest operator_module/tests/`. | ✓ |
| Top-level `pyproject.toml` with src layout | Project-wide change; affects Docker builds and CI. | |
| Tests use `from operator_module.main import render` | Tests run from repo root only. Couples invocation to a single working directory. | |

**User's choice:** `operator_module/tests/conftest.py` adds parent to sys.path.

---

### Q3: Add a project-root `pytest.ini`?

| Option | Description | Selected |
|--------|-------------|----------|
| Stay invocation-only | No project-root config. Run `pytest operator_module/tests/test_render.py` explicitly. Mirrors how `mcp-server/tests/` runs today. | ✓ |
| Add minimal project-root `pytest.ini` | `testpaths = operator_module/tests mcp-server/tests`. Creeps into mcp-server/. | |
| Add `operator_module/pytest.ini` (per-package) | Self-contained; redundant with conftest.py. | |

**User's choice:** Stay invocation-only.

---

### Q4: Add CI wiring?

| Option | Description | Selected |
|--------|-------------|----------|
| No CI in this phase | Repo has no CI today. Adding GHA workflow expands scope and impacts merge gating. Defer to a future infra phase. | ✓ |
| Add minimal GitHub Actions workflow | `.github/workflows/operator-tests.yml` runs pytest on push. New file outside operator_module. | |
| Add a `make test` target | Repo-root Makefile target; documentation-first. | |

**User's choice:** No CI in this phase.

---

## JSON-Safety Test Fixture

### Q1: What does the JSON-safety test cover?

| Option | Description | Selected |
|--------|-------------|----------|
| Two cases — plain JSON + JSON with `${VAR}` | Test 1: `render('{"auths": {"nvcr.io": {...}}}', {'namespace': 'foo'})` byte-identical (no `${` → pre-scan skips). Test 2: a JSON literal with one `${namespace}` correctly substitutes. | ✓ |
| Three cases — plain + sub + landmine | Adds: assert that JSON containing both `${namespace}` AND a literal `$oauthtoken` raises `ValueError` naming `oauthtoken`. Locks the known limitation. | |
| One case — plain JSON only | Just byte-identical no-substitution case. Mixing landmine deferred to Phase 18. | |

**User's choice:** Two cases — plain JSON + JSON with `${VAR}`.

---

### Q2: What JSON content?

| Option | Description | Selected |
|--------|-------------|----------|
| AIDP `dockerconfigjson` excerpt | Real `{"auths": {"nvcr.io": {...}}}` from `aidp/appstack/weka-aidp-appstack.yaml`. Inline as Python string constant. | ✓ |
| Smaller fabricated JSON literal | Hand-rolled `{"key": "value", "nested": {"a": 1}}`. Less anchored to reality. | |
| Both — small + AIDP | Cheap test for braces plus AIDP regression backstop. | |

**User's choice:** AIDP dockerconfigjson excerpt.

---

### Q3: How to record the bare-`$identifier` landmine?

| Option | Description | Selected |
|--------|-------------|----------|
| Capture as deferred concern in CONTEXT.md | Note in CONTEXT.md: manifests with `${VAR}` + bare `$identifier` (e.g. `$oauthtoken`) fail. AIDP migration must escape `$$oauthtoken` or pre-resolve. | ✓ |
| Add docstring warning + a test asserting failure mode | render() docstring warns; `test_bare_dollar_with_substitute()` asserts `ValueError`. Locks behavior in code. | |
| Both (CONTEXT + docstring + test) | Full coverage in three places. | |

**User's choice:** Capture as deferred concern in CONTEXT.md.
**Notes:** Locking the failure mode in a test was rejected because it would create churn if a future Template subclass solves the landmine.

---

## Claude's Discretion

- Exact placement of `render()` within `operator_module/main.py` (top-level helper area near `_deep_merge` / `merge_values`).
- Whether `render()` has a one-line or multi-line docstring (style only).
- Exact form of the conftest sys.path insertion (`Path(__file__).resolve().parents[1]` recommended; match mcp-server style).
- Pinning of pytest in `requirements-dev.txt` (`pytest>=8.0.0` recommended; match mcp-server).

## Deferred Ideas

- **Bare `$identifier` mixing landmine** (e.g., `$oauthtoken` inside JSON when `${VAR}` is also present in same string) — captured under "Known Limitations" in CONTEXT.md. Phase 18 must surface a clear error; Phase 20 (AIDP migration) must work around with `$$oauthtoken` escape or pre-resolution.
- **Project-root `pytest.ini` / `pyproject.toml`** — future test-infra phase that consolidates `operator_module/tests/` and `mcp-server/tests/`.
- **CI wiring (GitHub Actions / `make test`)** — future test-infra phase.
- **Custom `Template` subclass** that ignores bare `$identifier` — reconsider if AIDP migration hits the landmine in production and the workarounds prove insufficient.
