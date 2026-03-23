---
phase: 7
slug: validation-apply-and-status-tools
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config file** | None — tests run from `mcp-server/` with PYTHONPATH |
| **Quick run command** | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q` |
| **Full suite command** | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q`
- **After every plan wave:** Run `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | MCPS-07 | unit | `pytest mcp-server/tests/test_validate_yaml.py -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | MCPS-08 | unit | `pytest mcp-server/tests/test_apply_tool.py -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | MCPS-09 | unit | `pytest mcp-server/tests/test_status_tool.py -x` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | AGNT-02 | integration | `pytest mcp-server/tests/test_mock_agent.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `mcp-server/tests/test_validate_yaml.py` — stubs for MCPS-07 (validate_yaml accepts/rejects)
- [ ] `mcp-server/tests/test_apply_tool.py` — stubs for MCPS-08 (apply gate, success, error)
- [ ] `mcp-server/tests/test_status_tool.py` — stubs for MCPS-09 (status found, not found, empty)
- [ ] `mcp-server/tests/test_mock_agent.py` — stubs for AGNT-02 (harness happy path, bypass, failure)

*Existing Phase 6 infrastructure (conftest.py, fixtures) covers shared needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `mcp dev server.py` lists all 8 tools | MCPS-07/08/09 | Requires MCP Inspector UI | Run `mcp dev mcp-server/server.py`, verify 8 tools visible |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
