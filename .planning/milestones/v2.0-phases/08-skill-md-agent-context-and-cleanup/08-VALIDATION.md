---
phase: 08
slug: skill-md-agent-context-and-cleanup
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 08 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | mcp-server/tests/ (existing) |
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
| 08-01-01 | 01 | 1 | AGNT-01 | content | `grep -c "validate-retry" mcp-server/SKILL.md` | ❌ W0 | ⬜ pending |
| 08-01-02 | 01 | 1 | AGNT-01 | content | `grep -c "re-inspect" mcp-server/SKILL.md` | ❌ W0 | ⬜ pending |
| 08-02-01 | 02 | 1 | AGNT-01 | unit | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/test_mock_agent.py -x -q` | ✅ | ⬜ pending |
| 08-02-02 | 02 | 1 | AGNT-03 | content | `test -f mcp-server/openclaw.json` | ❌ W0 | ⬜ pending |
| 08-03-01 | 03 | 2 | CLEAN-01 | deletion | `test ! -f app-store-gui/webapp/planning/session_service.py` | ✅ | ⬜ pending |
| 08-03-02 | 03 | 2 | CLEAN-02 | deletion | `grep -c "planning_session" app-store-gui/webapp/main.py` | ✅ | ⬜ pending |
| 08-03-03 | 03 | 2 | CLEAN-03 | integration | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements. SKILL.md and openclaw.json are new files validated by content checks, not test framework.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| SKILL.md reads clearly as agent workflow | AGNT-01 | Prose quality | Read SKILL.md, verify workflow steps are unambiguous |
| openclaw.json matches NemoClaw format | AGNT-03 | Schema not published | Compare against NemoClaw docs when available |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
