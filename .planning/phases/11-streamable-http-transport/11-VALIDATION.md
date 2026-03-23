---
phase: 11
slug: streamable-http-transport
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | mcp-server/pytest.ini |
| **Quick run command** | `cd mcp-server && python -m pytest tests/ -x -q` |
| **Full suite command** | `cd mcp-server && python -m pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd mcp-server && python -m pytest tests/ -x -q`
- **After every plan wave:** Run `cd mcp-server && python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | XPORT-01 | unit | `cd mcp-server && python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |
| 11-01-02 | 01 | 1 | XPORT-02 | unit | `cd mcp-server && python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |
| 11-01-03 | 01 | 1 | XPORT-03 | integration | `cd mcp-server && python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |
| 11-01-04 | 01 | 1 | XPORT-04 | unit | `cd mcp-server && python -m pytest tests/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `mcp-server/tests/test_http_transport.py` — stubs for XPORT-01, XPORT-02, XPORT-03
- [ ] `mcp-server/tests/test_openclaw_config.py` — update existing transport assertion for XPORT-04

*Existing infrastructure covers framework and fixtures.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `curl localhost:8080/health` returns 200 | XPORT-01 | Confirms real port binding | Start server with `MCP_TRANSPORT=http`, run `curl -s -o /dev/null -w "%{http_code}" localhost:8080/health` |

*Integration tests use Starlette TestClient for automated coverage of the same endpoint.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
