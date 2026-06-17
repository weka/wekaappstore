# Phase 16: render() Helper and Test Scaffolding - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

> ## ⚠️ SUPERSEDED (2026-06-17) — `render()` is now ALLOWLIST substitution, NOT strict `string.Template`
>
> The strict `string.Template.substitute()` contract below (**D-01..D-06, OP-02, OP-04**, and the
> dependent **Phase 18 OP-07 / DOC-04** "undefined `${VAR}` → `PermanentError`") **caused a production
> outage** and has been reversed. **Do NOT restore strict substitution, the `$$` escape, or the
> undefined/malformed → raise behavior.**
>
> **What broke:** operator chart `0.1.65` / image `v0.12` shipped this strict `render()` to a live
> cluster. `stack_vars` always contains at least `{'namespace': ...}` (main.py:923), so every
> component manifest containing a `${` was pushed through `string.Template.substitute()`, which raised
> `Invalid placeholder` on the **first shell `$`** — `$(cmd)`, `$VAR`, `${SHELL_VAR}`, a `${` inside a
> bash comment, or `$$` (the shell PID). The AIDP appstack alone has **46** such tokens across
> ngc-secrets, keycloak-secret-sync, envoy-endpoint-discovery, embedding-gateway, … so **every
> Job-script component failed at deploy** (`Malformed placeholder in template: Invalid placeholder in
> string: line 99, col 18`).
>
> **Current contract (`operator_module/main.py` `render()`):** substitute **only** `${name}` where
> `name` is an explicitly provided variable (allowlist regex); leave **all** other `$`-content
> byte-for-byte (`$(`, `$VAR`, `${unknown}`, `${}`, `$$`); **never raise** on foreign/malformed
> placeholders. The delimiter was never the problem — over-broad substitution was. Undefined-variable
> detection (if ever desired) belongs at the variable-resolution layer, because in manifest text an
> "undefined" `${X}` is indistinguishable from a legitimate shell `${X}`.
>
> Regression locked by `test_render_shell_manifest_only_substitutes_known_vars` (test_render.py). The
> decisions below are retained for historical context only.

<domain>
## Phase Boundary

Build a pure, tested `render(text, variables) -> str` helper inside `operator_module/main.py` and initialize `operator_module/tests/` with `__init__.py`, `conftest.py`, and `test_render.py`. The helper performs single-pass `${VAR}` substitution via Python `string.Template`, with a pre-scan guard that protects existing shell-script content (e.g. `cluster_init/app-store-cluster-init.yaml` containing bare `$CRDS`/`$CRD`/`$MISSING`/`$GATEWAY_API_URL`).

**No live operator paths are touched in this phase.** No call site invokes `render()` yet — wiring into `handle_appstack_deployment` and `load_values_from_reference` is Phase 18's responsibility. Phase 16 only delivers (a) the helper and (b) the test scaffolding it ships in.

Requirements covered: OP-01, OP-02, OP-03, OP-04, OP-05, TST-01.

</domain>

<decisions>
## Implementation Decisions

### Pre-scan Guard
- **D-01:** Pre-scan check is a literal substring test: `if '${' not in text: return text`. Bare shell-style `$VAR` (no braces) passes through untouched.
- **D-02:** Check ordering: empty/None variables first, then pre-scan substring check, then `string.Template(text).substitute(variables)`. Cheapest path runs first for the no-vars case (most common during reconcile of pre-v5.0 CRs).
- **D-03:** No custom `Template` subclass. Stdlib `string.Template` is used as-is. Mixing-case landmine (bare `$identifier` co-existing with `${VAR}` in same string) is captured under Deferred / Known Limitations rather than solved here.

### render() Error Type
- **D-04:** `render()` raises `ValueError` on undefined or malformed placeholders. No `kopf` dependency in this layer. Phase 18 catches `ValueError` and wraps in `kopf.PermanentError(...)` with component context.
- **D-05:** Error message format: `f"Undefined variable: ${{{name}}}"` for undefined references; `f"Malformed placeholder in template: {original_error}"` for malformed cases (`${}`, `${123}`, etc.). Phase 18 prefixes with `... in component {comp_name}.kubernetesManifest`.
- **D-06:** Use chained exceptions: `raise ValueError(msg) from e`. Preserves the underlying `KeyError`/`ValueError` traceback for debuggability.

### Test Scaffolding
- **D-07:** Pytest is added to a new `operator_module/requirements-dev.txt` (NOT `requirements.txt`). Production image stays minimal; dev/test deps split out. Run `pip install -r operator_module/requirements-dev.txt` for testing.
- **D-08:** `operator_module/tests/conftest.py` injects the parent directory onto `sys.path` (mirrors `mcp-server/tests/conftest.py`). Tests import as `from main import render`. Self-contained; no project-root config needed.
- **D-09:** No project-root `pytest.ini` or `pyproject.toml [tool.pytest]`. Test invocation is explicit: `pytest operator_module/tests/test_render.py`. Matches how `mcp-server/tests/` runs today.
- **D-10:** No CI wiring in this phase. The repo has no GitHub Actions today (per `.planning/codebase/TESTING.md`); adding one creeps scope outside `operator_module/`. Defer to a future test-infra phase.

### Test Coverage (TST-01)
- **D-11:** Backward-compat regression test loads the actual shell-script excerpt from `cluster_init/app-store-cluster-init.yaml` (lines ~131–158, the `$CRDS`/`$CRD`/`$MISSING`/`$GATEWAY_API_URL` Job command block) and asserts byte-identical output for both `render(content, {})` and `render(content, {'namespace': 'default'})`. Pairs with the literal short test `render('$CRDS && $CRD', {})` from the success criteria.
- **D-12:** JSON-safety test uses the real AIDP `dockerconfigjson` payload (the `{"auths": {"nvcr.io": {...}}}` block from `aidp/appstack/weka-aidp-appstack.yaml`). Asserts byte-identical output when no `${VAR}` is present, AND a separate substitution case where a smaller JSON-bearing string with `${namespace}` correctly resolves.
- **D-13:** Required test cases per TST-01: pre-scan guard (literal + cluster_init excerpt), `$$` escape, JSON-safety (plain + substitution), undefined-variable error (asserts `ValueError` with variable name in message), malformed-placeholder error (asserts `ValueError` for `${}` and `${123}`), no-op when variables is `None` and `{}`.

### Claude's Discretion
- Exact placement of `render()` within `operator_module/main.py` (top-level helper area; Claude picks a sensible neighbor — likely just after `_load_kube_config_once` or with the other small helpers).
- Whether `render()` has a one-line docstring or a multi-line block (style only).
- Whether the conftest uses `Path(__file__).resolve().parents[1]` or a string-based equivalent — match the mcp-server pattern.
- Pinning strategy for pytest version in `requirements-dev.txt` (`>=8.0.0` mirrors mcp-server, recommended).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Specs
- `.planning/PRD-appstack-variable-substitution.md` — Source PRD; `Proposed Solution → Operator Change` defines the render() shape; `Acceptance Criteria` items 6 and 7 cover JSON-safety and `$$` escape.
- `.planning/REQUIREMENTS.md` — Phase 16 requirements OP-01..05 and TST-01.
- `.planning/ROADMAP.md` §"Phase 16: render() Helper and Test Scaffolding" — phase goal and success criteria.
- `.planning/STATE.md` §"Key Architectural Decisions (v5.0)" — non-negotiable invariants (pre-scan guard, single-pass, both `KeyError` AND `ValueError` caught, `handle_helm_deployment` non-wiring).

### Code to Touch / Reference
- `operator_module/main.py` — render() lives here. 1102 lines today; helper goes near other small top-level utilities.
- `operator_module/requirements.txt` — production deps; do NOT add pytest here.
- `cluster_init/app-store-cluster-init.yaml` lines ~131–158 — the shell-script Job manifest used as the backward-compat regression fixture (contains `$CRDS`/`$CRD`/`$MISSING`/`$GATEWAY_API_URL`).
- `/Users/christopherjenkins/git/aidp/appstack/weka-aidp-appstack.yaml` (separate repo) — source of the AIDP `dockerconfigjson` literal used as the JSON-safety fixture content.

### Patterns to Mirror
- `mcp-server/tests/conftest.py` — sys.path injection pattern. Copy the structure, adapt parent path to `operator_module/`.
- `mcp-server/requirements.txt` — pytest version pin (`pytest>=8.0.0`).

### Codebase Maps
- `.planning/codebase/TESTING.md` — current test posture (no project-level pytest, no CI). Confirms why this phase doesn't add CI.
- `.planning/codebase/STRUCTURE.md` — directory layout; confirms `operator_module/` has no existing `tests/` directory.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mcp-server/tests/conftest.py` — direct template for `operator_module/tests/conftest.py`. The sys.path injection pattern works as-is; only the parent path differs.
- `mcp-server/tests/` test file structure (e.g., `test_blueprints.py`, `test_validate_yaml.py`) — naming and module-level pytest patterns to mirror in `test_render.py`.

### Established Patterns
- Per-component `requirements.txt` (operator_module/, mcp-server/, app-store-gui/) — split deps live next to the code they support; we follow this with `operator_module/requirements-dev.txt`.
- pytest 8.x is the pinned baseline (`mcp-server/requirements.txt`: `pytest>=8.0.0`).
- No project-root pytest config — tests run from explicit subpaths today (`pytest mcp-server/tests/`). Phase 16 mirrors this for `operator_module/tests/`.

### Integration Points
- `operator_module/main.py` already exposes top-level helpers (`_deep_merge`, `merge_values`, `discover_chart_crds`, `load_values_from_reference`) — `render()` is a peer to these.
- `handle_appstack_deployment` (line 551) and `load_values_from_reference` (line 352) are the two future wiring sites — Phase 16 must NOT modify them.
- `cluster_init/app-store-cluster-init.yaml` is the canonical backward-compat fixture; the shell-script Job at lines ~131–158 is the specific risk content.

### Codebase Constraints
- Operator's production `requirements.txt` is intentionally minimal (3 deps). Adding pytest to it inflates the operator container image and runs counter to the existing split.
- No project-level pytest config exists; introducing one creeps scope outside `operator_module/`.

</code_context>

<specifics>
## Specific Ideas

- Helper signature: `def render(text: str, variables: Optional[Dict[str, str]]) -> str:` — consistent with the type-hint style elsewhere in `main.py` (`Dict[str, Any]`, `Optional[...]`). Variables map is `str → str` (non-string values are blocked at the CRD admission layer per Phase 17's CRD-03; defensive coding here can stay minimal).
- Error message exemplars (locked in the test):
  - `render("value: ${UNDEF}", {"x": "y"})` → `ValueError("Undefined variable: ${UNDEF}")`
  - `render("bad: ${}", {"x": "y"})` → `ValueError("Malformed placeholder in template: ...")` (note: variables dict is non-empty to bypass the D-02 empty-vars short-circuit; in production the dict always contains auto-default `namespace`)
  - `render("bad: ${123}", {"x": "y"})` → `ValueError("Malformed placeholder in template: ...")`
- The `$$` test: `render("price is $$5", {"x": "y"})` → `"price is $5"`.
- The cluster_init regression test loads from disk (using `Path(__file__).parents[2] / 'cluster_init' / 'app-store-cluster-init.yaml'`) rather than copy-pasting the shell script — this catches drift if the bootstrap manifest is later edited.

</specifics>

<deferred>
## Deferred Ideas / Known Limitations

### Bare `$identifier` mixing landmine — for Phase 18 / Phase 20

Manifests that mix `${VAR}` substitution with bare `$identifier` shell-style references (e.g. AIDP's `dockerconfigjson` containing `"username": "$oauthtoken"`) will trigger `string.Template.substitute()` and fail strict-mode against the bare reference. Concretely:

```
render('value: ${ns} and key: $oauthtoken', {'ns': 'foo'})
→ ValueError: Undefined variable: $oauthtoken (Template treats $oauthtoken as a substitution)
```

This is a real risk for Phase 20 (AIDP migration). When AIDP introduces `${namespace}` into a manifest that already contains `$oauthtoken`, the manifest will fail to render.

**Workarounds for Phase 20:**
1. Escape the bare reference: `$oauthtoken` → `$$oauthtoken` (Template emits `$oauthtoken` literally).
2. Pre-resolve the value: replace `$oauthtoken` with the literal token string in the manifest before applying.

**Phase 18 responsibility:** When the wrapping `kopf.PermanentError` is raised on this case, the message must include the offending variable name AND the component name so AIDP authors can locate it quickly. Phase 16's `ValueError` already names the variable; Phase 18 adds component context.

**This is not a Phase 16 deliverable.** No code mitigation here; just documented behavior. A test asserting this failure mode was considered (option 3 of the JSON-safety question) and explicitly NOT chosen — locking the failure here would create churn if a future Template subclass is introduced.

### Other deferred

- Project-root `pytest.ini` or `pyproject.toml [tool.pytest]` — defer to a future test-infra phase that consolidates `operator_module/tests/` and `mcp-server/tests/` under a unified config.
- CI wiring (GitHub Actions / make targets) — defer to a future test-infra phase.
- A custom `Template` subclass that ignores bare `$identifier` — would solve the mixing landmine but adds code surface and a non-stdlib API. Reconsider if AIDP migration hits the landmine in production and the workarounds above prove insufficient.

</deferred>

---

*Phase: 16-render-helper-and-test-scaffolding*
*Context gathered: 2026-05-06*
