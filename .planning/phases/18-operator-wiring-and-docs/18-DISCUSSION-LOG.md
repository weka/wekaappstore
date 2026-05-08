# Phase 18: Operator Wiring and Docs - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 18-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-08
**Phase:** 18-operator-wiring-and-docs
**Areas discussed:** Fetch-error classification (OP-11), README placement & worked example, TST-03 backward-compat snapshot strategy, Helper extraction style

---

## Fetch-error classification (OP-11)

### Q1: Which classes of failure should escalate to kopf.TemporaryError(delay=30) vs PermanentError?

| Option | Description | Selected |
|--------|-------------|----------|
| Only resource missing (404) | Just NotFoundError → TemporaryError; everything else PermanentError. | |
| Missing + transient API errors | 404, connection, timeout, 5xx → TemporaryError; auth/RBAC + yaml.safe_load → PermanentError. | ✓ |
| All kr8s exceptions → Temporary | Any kr8s/API failure → TemporaryError; loses signal on RBAC misconfig. | |

**User's choice:** Missing + transient API errors (Recommended).
**Notes:** "cluster wobble → retry; bad CR → fail loudly."

### Q2: How should the TemporaryError message look?

| Option | Description | Selected |
|--------|-------------|----------|
| Resource-only context | kind/namespace/name only, no component context. | |
| Resource + component context | Names component + valuesFiles index. | ✓ |
| Resource + component + retry hint | Above + RBAC verification hint; potential info-leak risk. | |

**User's choice:** Resource + component context (Recommended).

### Q3: Apply fetch-error upgrade to both call paths or only AppStack?

| Option | Description | Selected |
|--------|-------------|----------|
| Both paths | TemporaryError fires regardless of caller; helm path gets bonus fix. | ✓ |
| AppStack path only | Wrapper or flag so helm path keeps silent-{} behavior. | |
| Both, but log a warning on helm path | Both + extra warning logged when called from non-variables context. | |

**User's choice:** Both paths (Recommended).
**Notes:** Aligns with OP-11 wording; helm-path silent-{} fix is a welcome bonus.

### Q4: How should render-failure during CM/Secret loading be wrapped?

| Option | Description | Selected |
|--------|-------------|----------|
| PermanentError, named variable + ref | Names variable, component, source resource and key. | ✓ |
| PermanentError, minimal | Skips component context to keep it short. | |
| Re-raise ValueError, let caller wrap | Spreads error-formatting logic across call sites. | |

**User's choice:** PermanentError, named variable + ref (Recommended).

---

## README placement & worked example

### Q1: Where in README.md should the variable-substitution docs live?

| Option | Description | Selected |
|--------|-------------|----------|
| New top-level section after Common configuration | `## Variable substitution in AppStack manifests` between Common configuration and Upgrading. | ✓ |
| Subsection under Common configuration | `### Variable substitution` nested. | |
| Separate file: docs/VARIABLES.md | Keep README terse; full docs in dedicated file. | |

**User's choice:** New top-level section after Common configuration (Recommended).

### Q2: How extensive should the worked example be?

| Option | Description | Selected |
|--------|-------------|----------|
| Single AIDP-style multi-component CR | Two-three components using ${namespace}, ${milvusHost}, valuesFiles. | ✓ |
| Reference syntax table + minimal example | Compact; readers go to AIDP repo for full examples. | |
| Full migration walkthrough (before/after) | Most pedagogical but longest; overlaps with Phase 20 PR. | |

**User's choice:** Single AIDP-style multi-component CR (Recommended).

### Q3: How to present DOC-05 no-recursion warning?

| Option | Description | Selected |
|--------|-------------|----------|
| Callout block + worked-wrong/right pair | `> Note:` callout, # WRONG vs # CORRECT side-by-side. | ✓ |
| Inline bullet under syntax table | One line in a 'Limitations' bullet list. | |
| Footnote at the bottom of the section | 'Known limitations' subsection separated from syntax docs. | |

**User's choice:** Callout block + worked-wrong/right pair (Recommended).

### Q4: How prescriptive should DOC-06 be on operator-control fields?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard recommendation: drop targetNamespace | Strong nudge toward portable pattern. | ✓ |
| Neutral list of non-templated fields | Lets readers decide; risk of users keeping hardcoded targetNamespace. | |
| Tabular: field → templated? → recommendation | Densest format with all info in one table. | |

**User's choice:** Hard recommendation: drop targetNamespace (Recommended).

---

## TST-03 backward-compat snapshot strategy

### Q1: Which fixture should the snapshot test use?

| Option | Description | Selected |
|--------|-------------|----------|
| Existing ai-research.yaml fixture | Already in tree; multi-component; no `variables:` block. | ✓ |
| cluster_init/app-store-cluster-init.yaml | Real production CR; covers shell scripts; pulls non-test file into test path. | |
| Synthetic minimal fixture | Maximally isolated and stable; doesn't exercise real complexity. | |

**User's choice:** Existing ai-research.yaml fixture (Recommended).
**Notes:** Phase 19 will add ai-research-portable.yaml as sibling.

### Q2: What gets snapshot-asserted?

| Option | Description | Selected |
|--------|-------------|----------|
| Both: merged values dict + manifest string | Locks the entire substitution path's no-op behavior. | ✓ |
| Manifest tempfile only | Simpler; skips valuesFiles render path. | |
| Render call-input identity | Stub render() and assert call args; cheapest but indirect. | |

**User's choice:** Both: merged values dict + manifest string (Recommended).

### Q3: Mocking strategy?

| Option | Description | Selected |
|--------|-------------|----------|
| Mock subprocess.run + kr8s + HelmOperator | Standard pytest pattern; works with existing conftest.py. | ✓ |
| Refactor handle_appstack_deployment to be more testable | Cleaner test surface but adds refactor scope. | |
| Snapshot fixture files captured once | Concrete files in tests/snapshots/; reviewable in PRs. | |

**User's choice:** Mock subprocess.run + kr8s + HelmOperator (Recommended).
**Notes:** Snapshot baselines still live in `operator_module/tests/snapshots/ai-research/` (combination of options 1 and 3).

### Q4: TST-05 (handle_helm_deployment non-wiring) test shape?

| Option | Description | Selected |
|--------|-------------|----------|
| Mock-based call assertion | Patch + assert + inspect.getsource static check. Two-layered. | ✓ |
| Inspect-based static check only | Pure static; no runtime mocking; faster and simpler. | |
| Behavior test: helm path with variables-bearing CR fails | Behavioral proof but more complex setup. | |

**User's choice:** Mock-based call assertion (Recommended).

---

## Helper extraction style

### Q1: Inline render() at each site, single helper, or two-helper split?

| Option | Description | Selected |
|--------|-------------|----------|
| Extract _render_or_raise helper | Single helper with source_desc kwarg; ~10 lines + 3 per site. | ✓ |
| Inline at each call site | Self-contained per site; risk of drift in error format. | |
| Two helpers: one per surface | More targeted but more API surface to test. | |

**User's choice:** Extract _render_or_raise helper (Recommended).

### Q2: Where exactly in handle_appstack_deployment is the variables dict built?

| Option | Description | Selected |
|--------|-------------|----------|
| Top of function, after enabled_components filter | Before resolve_dependencies; key validation pre-empts deployment. | ✓ |
| Inside resolve_dependencies callsite, just before for loop | Closer to consumption; still pre-loop. | |
| Lazy: build the first time render() is needed | Avoids build for AppStacks with no rendering; but partial-deployment risk. | |

**User's choice:** Top of function, after enabled_components filter (Recommended).

### Q3: load_values_from_reference signature evolution?

| Option | Description | Selected |
|--------|-------------|----------|
| Add `variables: Optional[Dict[str, str]] = None` | Plus optional comp_name kwarg for richer error context. | ✓ |
| Add variables only, no comp_name | Terser errors; component context wrapping done at callsite. | |
| Two-function split: load_values_raw + load_values_rendered | Cleanest separation but duplicates kr8s fetch logic. | |

**User's choice:** Add `variables: Optional[Dict[str, str]] = None` (Recommended).
**Notes:** Helm path callsite at line 923 is NOT touched (still positional 4-arg call).

### Q4: OP-12 field='spec' filter — apply as-spec'd or wrap?

| Option | Description | Selected |
|--------|-------------|----------|
| Apply as-spec'd, no extra wrapping | Single-line decorator change at line 1053. | ✓ |
| Add field='spec' AND old != new check | Belt-and-suspenders with `when=` predicate. | |
| Apply field='spec' AND audit other handlers | Audit-and-fix scope creep. | |

**User's choice:** Apply as-spec'd, no extra wrapping (Recommended).

---

## Claude's Discretion

- Exact placement of `_render_or_raise` helper inside `operator_module/main.py` (top-level, near `render()`).
- Exact phrasing of error messages within the formats locked above.
- Snapshot baseline file format (`values_<comp>.json` vs `values_<comp>.yaml`).
- README syntax table layout (pipes vs code-block).
- Pinning vs. building the regex pattern (`re.compile` once vs. inline).
- Chart.yaml version bump cadence (likely 0.1.62 → 0.1.63 for this phase).

## Deferred Ideas

- Per-component variable overrides → future phase if AIDP asks.
- `when=` predicate on @kopf.on.update — paranoia; field='spec' is sufficient.
- field='spec' audit on @kopf.on.create / @kopf.on.delete — out of scope for v5.0.
- Refactor `_prepare_component_artifacts` pure helper — defer to operator-cleanup phase.
- Snapshot test for cluster_init/app-store-cluster-init.yaml — Phase 16 covers it.
- Migration walkthrough in README — overlaps with Phase 20 PR description.
- CI wiring for operator_module/tests/ — future test-infra phase.
- `status.conditions[type=VariablesResolved]` observability field — already deferred to V51-02.
- Default-value syntax `${VAR:-default}` — already deferred to V51-03.
- Templating `targetNamespace` — already deferred to V51-01.
- Bare `$identifier` mixing landmine — inherited from Phase 16; documented in error messages.
