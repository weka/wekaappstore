# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v2.0 — OpenClaw MCP Tool Integration

**Shipped:** 2026-03-23
**Phases:** 5 (6-10) | **Plans:** 11 | **Tests:** 103

### What Was Built
- 8-tool MCP server with flat agent-friendly JSON (inspect_cluster, inspect_weka, list_blueprints, get_blueprint, get_crd_schema, validate_yaml, apply, status)
- SKILL.md: 12-step agent workflow with validate-retry, re-inspect-before-apply, negative examples
- Mock agent harness with description-based tool selection (3 scenarios)
- Container deployment: Dockerfile, CI/CD, README, openclaw.json
- v1.0 backend-brain cleanup: 12 deprecated files removed

### What Worked
- `_impl(injectable) / register_*(mcp)` pattern enabled isolated testing without MCP framing
- Flat 2-key depth contract enforced by `check_depth()` prevented response bloat early
- Wave-based parallel execution allowed Plans 08-01 and 08-03 to run simultaneously (no file conflicts)
- Mock harness proving tool descriptions are sufficient for routing — validated SKILL.md design before live testing
- Gap closure phase (10) caught 3 integration defects that static verification missed

### What Was Inefficient
- Phase 10 defects (logger.warning TypeError, LOG_LEVEL not wired, PYTHONPATH missing) should have been caught by Phase 6/9 tests — regression tests added retroactively
- NemoClaw alpha schema blocker persists — placeholder docs may need significant revision

### Patterns Established
- `_RegistryCapture` stub pattern: reused across mock harness, openclaw.json generator, and tests — extracts tool metadata without starting FastMCP
- `ops_log` pattern for asserting side effects: mocked methods append (op_type, kwargs) tuples to shared list
- Description-based tool selection: keyword matching on `@mcp.tool()` descriptions proves agent routing works
- Drift detection tests: generated config vs live tool registrations — catches desync automatically

### Key Lessons
1. Integration checker after all phases catches wiring bugs that per-phase verification misses — always run audit before milestone completion
2. `file=sys.stderr` on `logger.warning()` is a common Python mistake — logger already sends to stderr via handler, `file=` is a print() kwarg
3. Env var config that's declared but not consumed is invisible to unit tests — integration tests or startup smoke tests catch it

### Cost Observations
- Model mix: sonnet for research/execution/verification agents, opus for orchestration
- All 11 plans executed in ~50 minutes total across 5 phases
- Phase 10 (3 bug fixes) took 12 minutes including TDD red/green cycle

---

## Milestone: v4.0 — App Categories on Home Screen

**Shipped:** 2026-04-21
**Phases:** 1 (Phase 15) | **Plans:** 3 | **LOC:** +162/-56 in one file

### What Was Built
- Three category filter cards (AIDP, WARP, Partner) above the existing App Catalog grid on the WEKA App Store home page
- Client-side filter with `history.replaceState` URL hash deep-link support (`/#category=<key>`)
- Native `<button>` + `aria-pressed` keyboard accessibility; mobile-responsive stacking
- New `AppShell` component lifts `ThemeProvider`, pure prop-based `Catalog`, `Categories` + `EmptyState` components — all in the single-file CDN-React IIFE pattern (no build step, no new dependencies)

### What Worked
- **PRD → research → plan → execute pipeline** moved from input PRD to shipped feature in ~1h40min single session
- **Milestone-level research (Stack/Features/Architecture/Pitfalls + SUMMARY) fed directly into plans** — `--skip-research` at the phase level was appropriate; research artifacts were more useful at milestone scope for a tightly-scoped single-phase milestone
- **Grep-level pitfall encoding** — the 5 critical pitfalls from PITFALLS.md (JSX forbidden, `component:'a'` absent, `replaceState` not on mount, `startsWith('#category=')` parser, single `ThemeProvider`) were each encoded as `grep` checks in plan verify blocks. All passed first try in 15-03; zero revisions needed.
- **3-step build order authoritative from ARCHITECTURE.md** — data prep → structural refactor → feature landing. Each step independently verifiable with pixel-identical output up to the final step.
- **Gray-area discussion (A, B, C, D) done upfront in discuss-phase** meant planner had zero ambiguity about page structure, heading copy, descriptions, and scroll behavior.

### What Was Inefficient
- **`roadmap update-plan-progress` CLI corrupted Phase 11 `Plans:` line twice** during 15-01 and 15-02 finalization — the regex used for locating phase rows is too broad. Manual revert required both times. Not a blocker but worth reporting/fixing upstream.
- **`milestone complete` CLI miscounted phases (5) and plans (12)** — tallied everything in `.planning/phases/` instead of scoping to the milestone's phase range. MILESTONES.md entry required manual correction with real stats (1 phase, 3 plans, 8 tasks) and accomplishments (CLI returned empty `accomplishments: []`). Fresh for next milestone.
- **`milestone complete` CLI did not delete `REQUIREMENTS.md`** per the workflow spec — had to `rm` manually after verifying archive.
- **`phase complete` CLI did not update REQUIREMENTS.md traceability Status column** — all 14 rows stayed "Pending" after `phase complete`; had to fix in the audit step inline.
- **Plan grep invariants scan their own comment text** — 15-03 executor had to rephrase three inline comments because they contained the exact strings the grep invariants were checking for (`component: 'a'`, `history.replaceState`, `pushState`). Valid deviation but indicates the grep patterns are too coarse — they should scope to code vs. comments.
- **OpenClaw 8B Llama retry for v3.0 E2E (same day)** burned ~2h hunting model/config issues before we rescoped v3.0 and deferred E2E to v3.1 — lesson captured in v3.0-KNOWN-ISSUES.md rather than repeated here.

### Patterns Established
- **Grep-as-verification for anti-patterns** — every critical "do NOT do X" becomes a `grep -cE "..." | grep -qE "^0$"` check in the plan's verify block. Enforceable by execution agent, not just human review.
- **`--skip-research` at phase level when milestone research is sufficient** — milestone-level SUMMARY.md + PITFALLS.md + ARCHITECTURE.md was enough input for the planner. A phase-level RESEARCH.md would have duplicated content.
- **Three-step build order for single-file UI changes** — data → structural → feature. Each step renders pixel-identically up to the final step, making regression detection trivial.
- **Gray-area discussion BEFORE planning** — four small decisions (mount strategy, heading copy, descriptions, scroll behavior) made upfront kept planner output tight and verifiable.
- **Grey-area decision references in plan locked_decisions blocks** — each plan's YAML frontmatter lists the Gray Areas it implements verbatim, so execute-phase doesn't re-open them.

### Key Lessons
1. **Small, tightly-scoped frontend milestones can ship in one session** if PRD + research + gray-area discussion are done upfront and the stack is locked down. The entire v4.0 cycle from "submit PRD" to "audit passed" was under 2 hours.
2. **Grep-level pitfall verification is the right defense for "looks done but isn't" anti-patterns** in a no-build codebase. The ThemeProvider count, JSX count, component:'a' count, replaceState location checks all caught real mistakes that visual QA might have missed.
3. **GSD CLI has bookkeeping bugs to file** — `roadmap update-plan-progress`, `milestone complete` stats, `phase complete` traceability updates all had issues this milestone. All trivially patchable by hand, but not self-healing.
4. **Docstring/comment content vs code content collision** is a real pattern to watch — grep invariants must scope to code-only regions when the invariant string could legitimately appear in documentation.

### Cost Observations
- Model mix: sonnet for all agent roles (research, synthesis, planning, checking, execution, verification)
- Single session: ~1h40min wall clock from "submit PRD" to "audit passed"
- 18 commits, 9 touching the phase directory
- 8 agent spawns: 4 parallel researchers, 1 synthesizer, 1 planner, 1 plan-checker, 3 executors + 3 continuations, 1 verifier

---

## Cross-Milestone Trends

| Metric | v2.0 | v4.0 |
|--------|------|------|
| Phases | 5 | 1 |
| Plans | 11 | 3 |
| Tests | 103 | 0 (frontend — grep invariants + visual verify) |
| LOC | 4,628 | +162 / -56 (single file) |
| Audit defects | 3 (all fixed) | 0 |
| Audit re-check | passed | passed (first pass) |
| Timeline | ~50 min exec + surrounding work | ~1h40min end-to-end |

**Trend notes:**
- v4.0 was dramatically tighter in scope than v2.0 — single-file frontend vs. multi-module Python server. The GSD pipeline compressed accordingly (1 phase, 3 plans, no test files).
- Grep-level invariants replaced traditional test coverage for v4.0 because the codebase has no frontend test harness. This worked well for a tightly-scoped single-file change; won't scale to larger UI work.
