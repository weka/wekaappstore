---
phase: 09
slug: deployment-and-registration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 09 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x + docker build |
| **Config file** | mcp-server/tests/ (existing) |
| **Quick run command** | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q` |
| **Full suite command** | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds (tests) + ~30 seconds (docker build) |

---

## Sampling Rate

- **After every task commit:** Run `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/ -x -q`
- **After every plan wave:** Run full suite + `docker build -f mcp-server/Dockerfile .`
- **Before `/gsd:verify-work`:** Full suite must be green + image builds successfully
- **Max feedback latency:** 35 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | DEPLOY-01 | build | `docker build -f mcp-server/Dockerfile . -t test` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | DEPLOY-02 | unit | `cd mcp-server && PYTHONPATH=.:../app-store-gui python -m pytest tests/test_config.py -x -q` | ❌ W0 | ⬜ pending |
| 09-02-01 | 02 | 1 | DEPLOY-03, DEPLOY-04 | content | `test -f mcp-server/README.md` | ❌ W0 | ⬜ pending |
| 09-03-01 | 03 | 2 | DEPLOY-01 | ci | `test -f .github/workflows/mcp-server.yml` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `mcp-server/Dockerfile` — container image build
- [ ] `mcp-server/tests/test_config.py` — env var validation tests
- [ ] `mcp-server/README.md` — registration documentation
- [ ] `.github/workflows/mcp-server.yml` — CI workflow

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Image starts MCP server on stdio | DEPLOY-01 | Requires Docker runtime | `docker run --rm -i test echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' \| docker run --rm -i test` |
| NemoClaw registration works | DEPLOY-04 | Schema not published | Follow README NemoClaw section when available |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 35s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
