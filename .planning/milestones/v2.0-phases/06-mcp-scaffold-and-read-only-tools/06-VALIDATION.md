---
phase: 6
slug: mcp-scaffold-and-read-only-tools
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `cd mcp-server && python -m pytest tests/ -x -q` |
| **Full suite command** | `cd mcp-server && python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd mcp-server && python -m pytest tests/ -x -q`
- **After every plan wave:** Run `cd mcp-server && python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 0 | MCPS-01 | unit | `pytest mcp-server/tests/test_server.py::test_tool_list -x` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | MCPS-02 | unit | `pytest mcp-server/tests/test_inspect_cluster.py::test_flat_response -x` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | MCPS-03 | unit | `pytest mcp-server/tests/test_inspect_weka.py::test_flat_response -x` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | MCPS-04 | unit | `pytest mcp-server/tests/test_blueprints.py::test_list_blueprints -x` | ❌ W0 | ⬜ pending |
| 06-01-05 | 01 | 1 | MCPS-05 | unit | `pytest mcp-server/tests/test_blueprints.py::test_get_blueprint -x` | ❌ W0 | ⬜ pending |
| 06-01-06 | 01 | 1 | MCPS-06 | unit | `pytest mcp-server/tests/test_crd_schema.py::test_get_crd_schema -x` | ❌ W0 | ⬜ pending |
| 06-01-07 | 01 | 2 | MCPS-10 | unit | `pytest mcp-server/tests/test_response_depth.py -x` | ❌ W0 | ⬜ pending |
| 06-01-08 | 01 | 2 | MCPS-11 | unit | `pytest mcp-server/tests/test_logging.py::test_no_stdout -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `mcp-server/tests/__init__.py` — test package init
- [ ] `mcp-server/tests/conftest.py` — shared mocked K8s API fixtures
- [ ] `mcp-server/tests/test_server.py` — tool list smoke test
- [ ] `mcp-server/tests/test_inspect_cluster.py` — flat output contract tests
- [ ] `mcp-server/tests/test_inspect_weka.py` — flat output contract tests
- [ ] `mcp-server/tests/test_blueprints.py` — scanner + list + get tests
- [ ] `mcp-server/tests/test_crd_schema.py` — schema shape tests
- [ ] `mcp-server/tests/test_response_depth.py` — depth-2 contract enforcer
- [ ] `mcp-server/tests/test_logging.py` — stderr-only contract test
- [ ] `mcp-server/requirements.txt` — `mcp[cli]>=1.26.0`, `kubernetes>=27.0.0`, `PyYAML>=6.0.1`, `pytest>=8.0.0`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `mcp dev server.py` starts and lists tools | MCPS-01 | Requires MCP dev inspector UI | Run `mcp dev mcp-server/server.py`, verify 5 tools listed in inspector |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
