---
phase: 16-render-helper-and-test-scaffolding
plan: 01
subsystem: operator
tags: [python, string-template, pytest, render, variable-substitution, kopf, operator]

# Dependency graph
requires: []
provides:
  - "Pure render(text: str, variables: Optional[Dict[str, str]]) -> str helper in operator_module/main.py"
  - "Pre-scan guard short-circuits text without ${...} placeholders or $$ escapes (backward-compat for cluster_init shell scripts)"
  - "ValueError-raising error contract (chained via 'from e') for undefined variables and malformed placeholders"
  - "operator_module/tests/ scaffolding: __init__.py, conftest.py with sys.path injection, requirements-dev.txt"
  - "12-test pytest suite covering OP-01..OP-05 + TST-01"
affects: [Phase 17 (CRD schema for spec.appStack.variables), Phase 18 (wires render() into handle_appstack_deployment + load_values_from_reference, wraps ValueError as kopf.PermanentError), Phase 20 (AIDP migration end-to-end smoke test)]

# Tech tracking
tech-stack:
  added: [stdlib string.Template, pytest>=8.0.0 (dev-only)]
  patterns:
    - "Pre-scan guard literal-substring check before stdlib delegation"
    - "Chained-exception error wrapping with `raise ValueError(...) from e`"
    - "Per-component requirements-dev.txt (split prod/dev deps)"
    - "Per-function deferred SUT import inside test bodies"

key-files:
  created:
    - "operator_module/tests/__init__.py (empty package marker)"
    - "operator_module/tests/conftest.py (sys.path injection)"
    - "operator_module/tests/test_render.py (12 unit tests)"
    - "operator_module/requirements-dev.txt (pytest>=8.0.0)"
  modified:
    - "operator_module/main.py (added `import string` + render() helper between _deep_merge and CRD Strategy Helpers section)"

key-decisions:
  - "Pre-scan guard extended to also pass through when '$$' is present (Rule 1 fix — locked OP-04 $$ escape test contradicted the literal-substring `${`-only guard otherwise)"
  - "render() raises ValueError (not kopf.PermanentError) — kopf wrapping deferred to Phase 18 per D-04"
  - "Stdlib string.Template used directly with no subclass per D-03"
  - "Empty/None variables short-circuit FIRST, then '${' substring pre-scan (with $$ exception), then Template.substitute() per D-02"
  - "DOCKERCONFIGJSON_PAYLOAD is a module-level inline literal in test_render.py (D-12 — never loaded from external repo paths at runtime)"
  - "Per-function deferred `from main import render` mirrors mcp-server/tests/test_validate_yaml.py pattern (avoids kopf decorator side effects at pytest collection)"

patterns-established:
  - "Pre-scan guard pattern: short-circuit on absence of all stdlib Template tokens (`${` AND `$$`) before delegating to Template.substitute()"
  - "Chained-exception ValueError wrapping for downstream kopf.PermanentError translation"
  - "operator_module/tests/ scaffolding mirrors mcp-server/tests/ exactly (sys.path injection, deferred imports, plain-function pytest)"
  - "Split production vs dev deps via per-component requirements-dev.txt (avoids inflating operator container image)"

requirements-completed: [OP-01, OP-02, OP-03, OP-04, OP-05, TST-01]

# Metrics
duration: 7m54s
completed: 2026-05-06
---

# Phase 16 Plan 01: render() Helper and Test Scaffolding Summary

**Pure single-pass `${VAR}` substitution helper using stdlib `string.Template`, with pre-scan guard for backward-compat with shell-script content in cluster_init manifests, plus operator_module/tests/ initialized with 12 passing pytest cases.**

## Performance

- **Duration:** 7m54s
- **Started:** 2026-05-06T10:26:24Z
- **Completed:** 2026-05-06T10:34:18Z
- **Tasks:** 3
- **Files modified:** 1 (operator_module/main.py)
- **Files created:** 4 (tests/__init__.py, tests/conftest.py, tests/test_render.py, requirements-dev.txt)

## Accomplishments

- Added pure `render(text, variables) -> str` helper to `operator_module/main.py` (38 insertions, 0 deletions, no other functions touched)
- Initialized `operator_module/tests/` package with conftest mirroring `mcp-server/tests/conftest.py` (drops MagicMock/pytest fixtures because Phase 16 only does path setup)
- Created 12-test pytest suite covering all D-13 required cases — `pytest operator_module/tests/test_render.py` passes 12/12 in <1s
- Split pytest into new `operator_module/requirements-dev.txt` so production operator image stays minimal (kopf, kr8s, kubernetes only)
- Confirmed `cluster_init/app-store-cluster-init.yaml` shell scripts (`$CRDS`, `$CRD`, `$MISSING`, `$GATEWAY_API_URL`) are byte-identical-preserved through render() with both `{}` and `{'namespace': 'default'}` (the locked TST-01 backward-compat regression)

## Task Commits

Each task was committed atomically on `worktree-agent-a70391cf78cbfee0f`:

1. **Task 1: Add render() helper to operator_module/main.py** — `473f721` (feat)
2. **Task 2: Create operator_module/tests/ scaffolding and requirements-dev.txt** — `bf5f767` (chore)
3. **Task 3: Create test_render.py with full TST-01 coverage** — `34f8f85` (test)

## Files Created/Modified

- `operator_module/main.py` — modified (38 lines added at lines 8 and 253-289). Adds `import string` to stdlib import block and a single new top-level `def render(text: str, variables: Optional[Dict[str, str]]) -> str:` helper between `_deep_merge` (ends line 249) and the `# ===================== CRD Strategy Helpers =====================` section header (now line 290). No other code changed.
- `operator_module/requirements-dev.txt` — NEW (1 line: `pytest>=8.0.0`). Dev-only; production `requirements.txt` unchanged.
- `operator_module/tests/__init__.py` — NEW (0 bytes). Empty package marker matching `mcp-server/tests/__init__.py`.
- `operator_module/tests/conftest.py` — NEW (15 lines). sys.path injection of `operator_module/` so tests can `from main import render`.
- `operator_module/tests/test_render.py` — NEW (142 lines). 12 pytest functions; module-level `DOCKERCONFIGJSON_PAYLOAD` inline literal; per-function deferred `from main import render` to avoid kopf decorator side effects at collection.

## render() Function Shape

**Signature** (locked by CONTEXT.md "Specifics"):

```python
def render(text: str, variables: Optional[Dict[str, str]]) -> str:
```

**Check ordering** (D-02):

1. `if not variables: return text` — empty/None variables short-circuit (handles the production reconcile of pre-v5.0 CRs, the most common path)
2. `if '${' not in text:` — pre-scan substring guard. If absent, the helper still falls through to Template.substitute() **iff** `$$` is present (OP-04 escape requirement). Otherwise returns unchanged. The cluster_init shell-script content (bare `$CRDS`, `$CRD`, etc.) hits this branch.
3. `string.Template(text).substitute(variables)` — strict mode, no subclass.

**Error contract** (D-04, D-05, D-06):

- `KeyError` from substitute() → `raise ValueError(f"Undefined variable: ${{{name}}}") from e`
- `ValueError` from substitute() (malformed `${}`, `${123}`, etc.) → `raise ValueError(f"Malformed placeholder in template: {e}") from e`
- Phase 18 will catch these `ValueError`s and wrap as `kopf.PermanentError(...)` with component context.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Extend pre-scan guard to also fall through when `$$` is present | Plan must_haves required both the `${`-only literal pre-scan AND that `render('price is $$5', {'x':'y'})` returns `'price is $5'`. With the literal pre-scan only, `$$` strings never reach Template and the OP-04 escape test fails. The fix nests `if '$$' not in text: return text` inside the `if '${' not in text:` block — this preserves the locked literal substring `if '${' not in text:` (still grep-matchable on a single line) while letting `$$`-only strings reach Template for escape collapse. cluster_init has zero `$$` occurrences, so backward-compat is preserved. |
| Inline `DOCKERCONFIGJSON_PAYLOAD` constant in test_render.py | D-12 anti-pattern: never load AIDP secret content from `/Users/christopherjenkins/git/aidp/...` at runtime. Module-level literal is portable, deterministic, and self-contained. |
| Drop the literal external-repo path from a comment | Acceptance criterion `grep -c "/Users/christopherjenkins/git/aidp" operator_module/tests/test_render.py` requires 0; reworded the existing comment to "any external repo path" to honor the lock. |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] Pre-scan guard extended to handle `$$` escape**

- **Found during:** Task 1 (verifying render() inline assertions before commit)
- **Issue:** The plan's locked implementation `if '${' not in text: return text` makes `render('price is $$5', {'x':'y'})` return the input unchanged because `$$` does not contain the substring `${`. This contradicts the plan's own must_have: `"render('price is $$5', {'x': 'y'}) returns 'price is $5' (D-13 $$ escape; OP-04)"` and would cause `test_render_double_dollar_escape` to fail. Without the fix, the locked success criterion #5 (`pytest operator_module/tests/test_render.py passes all 12 tests`) is unattainable.
- **Fix:** Added a nested `if '$$' not in text: return text` inside the `if '${' not in text:` block. The check ordering remains as D-02 specifies (empty-vars first, then `${` pre-scan, then substitute). The `${`-substring literal still appears verbatim on its own line, satisfying the structural acceptance criterion. cluster_init/app-store-cluster-init.yaml contains zero `$$` occurrences (`grep -c '\$\$'` = 0), so the backward-compat regression test still passes.
- **Files modified:** operator_module/main.py
- **Verification:** All 12 pytest cases pass, including `test_render_double_dollar_escape` (`render('price is $$5', {'x':'y'}) == 'price is $5'`) AND `test_render_cluster_init_unchanged` (cluster_init bytes-identical for both `{}` and `{'namespace': 'default'}`).
- **Committed in:** 473f721 (Task 1 commit)

**2. [Rule 2 — Missing Critical] Inline-comment scrubbed of external repo path**

- **Found during:** Task 3 (running structural acceptance criteria after writing test_render.py)
- **Issue:** A comment originally read `# NOT loaded from /Users/christopherjenkins/git/aidp at runtime` — but the plan's acceptance criterion `grep -c "/Users/christopherjenkins/git/aidp" operator_module/tests/test_render.py` returns 0. The comment was meant to document the D-12 anti-pattern; instead it tripped the acceptance check.
- **Fix:** Reworded to "any external repo path at runtime" — preserves the documentation intent without the external path string.
- **Files modified:** operator_module/tests/test_render.py
- **Verification:** `grep -c "/Users/christopherjenkins/git/aidp" operator_module/tests/test_render.py` returns 0; pytest still passes 12/12.
- **Committed in:** 34f8f85 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug, 1 Rule 2 critical correctness)
**Impact on plan:** Both auto-fixes were necessary to satisfy the plan's own success criteria (#5 pytest pass, structural acceptance regex). No scope creep — both fixes are within the locked behavior contract; only the pre-scan guard implementation was extended to make the plan internally consistent.

## Issues Encountered

None unrelated to the deviation rules. The pre-scan guard inconsistency was the only real issue and was resolved cleanly via Rule 1.

## Threat Surface Scan

No new threat surface introduced beyond what the plan's `<threat_model>` already documented (T-16-01 through T-16-04, all `accept`). render() is dead code from a security-posture standpoint until Phase 18 wires it after kopf admission validation. The single `cluster_init/app-store-cluster-init.yaml` disk read inside the regression test is repo-local trusted content. No threat flags raised.

## Test Invocation

```bash
pip install -r operator_module/requirements-dev.txt
pytest operator_module/tests/test_render.py
# expected: 12 passed
```

## TDD Gate Compliance

Plan-frontmatter is `type: execute` (not `type: tdd`), but two tasks (Task 1 and Task 3) carry `tdd="true"` markers. The plan author structured Task 1 as implementation + inline-assertion verification, and Task 3 as the formal test suite. While this inverts strict RED→GREEN ordering at the per-task level, the plan-level outcome holds: a `feat(...)` implementation commit (473f721) is followed by a `test(...)` commit (34f8f85) that exercises the contract end-to-end and passes 12/12. No REFACTOR commit was needed (render() is a single small function with no duplication to consolidate).

## Next Phase Readiness

- **Phase 17 (CRD schema):** Independent of Phase 16. Can proceed in parallel.
- **Phase 18 (wire render() into operator paths):** Ready. The hand-off contract is:
  - Import path: `from main import render` (already in same module)
  - Signature: `render(text: str, variables: Optional[Dict[str, str]]) -> str`
  - Empty-vars guard: callers can pass `variables=None` to disable substitution entirely (locks `handle_helm_deployment` non-wiring per TST-05)
  - Error handling: catch `ValueError` and re-raise as `kopf.PermanentError(f"... in component {comp_name}.kubernetesManifest: {ve}")` — `from ve` to chain
  - Auto-default `${namespace}`: Phase 18 should merge `{'namespace': cr.metadata.namespace}` with user-supplied `spec.appStack.variables` (user values win on conflict per PRD)
  - Wiring sites: `handle_appstack_deployment` (line 551 in current main.py) for `kubernetesManifest` strings, and `load_values_from_reference` (line 352) for ConfigMap/Secret-loaded `valuesFiles` content (BEFORE `yaml.safe_load`)
  - Phase 18 must NOT modify `handle_helm_deployment` single-chart path
- **Phase 19 (validator soft-warnings):** Independent. Can proceed in parallel with 17/18.
- **Phase 20 (AIDP migration):** Blocked on Phase 18 deployment. The dockerconfigjson `$oauthtoken` mixing landmine documented in CONTEXT.md "Deferred" remains a Phase 18/20 concern (not a Phase 16 deliverable).

## Self-Check: PASSED

Verified before write:
- `operator_module/main.py` — exists, contains `def render(`, `import string`, no other modifications
- `operator_module/tests/__init__.py` — exists, 0 bytes
- `operator_module/tests/conftest.py` — exists, contains `Path(__file__).resolve().parents[1]`
- `operator_module/tests/test_render.py` — exists, all 12 test functions defined, 12 deferred imports of render
- `operator_module/requirements-dev.txt` — exists, 1 line `pytest>=8.0.0`
- Commits 473f721, bf5f767, 34f8f85 present in git log on `worktree-agent-a70391cf78cbfee0f`
- `pytest operator_module/tests/test_render.py` returns 12 passed
- `git diff main..HEAD operator_module/requirements.txt` is empty (production deps unchanged)
- No project-root pytest.ini, pyproject.toml, or setup.py introduced

---

*Phase: 16-render-helper-and-test-scaffolding*
*Plan: 01*
*Completed: 2026-05-06*
