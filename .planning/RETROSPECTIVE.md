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

## Cross-Milestone Trends

| Metric | v2.0 |
|--------|------|
| Phases | 5 |
| Plans | 11 |
| Tests | 103 |
| LOC | 4,628 |
| Audit defects | 3 (all fixed) |
| Audit re-check | passed |
