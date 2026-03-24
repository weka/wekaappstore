---
phase: 12
slug: nemoclaw-eks-topology
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-24
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | bash scripts + pytest (existing) |
| **Config file** | none — scripts created in Wave 0 |
| **Quick run command** | `cd mcp-server && python -m pytest tests/ -x -q` |
| **Full suite command** | `cd mcp-server && python -m pytest tests/ -v && bash scripts/validate-topology.sh` |
| **Estimated runtime** | ~30 seconds (pytest) + ~15 seconds (smoke test) |

---

## Sampling Rate

- **After every task commit:** Run `cd mcp-server && python -m pytest tests/ -x -q`
- **After every plan wave:** Run `cd mcp-server && python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite + smoke test script must pass
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | NCLAW-01 | smoke | `bash scripts/validate-topology.sh` | ❌ W0 | ⬜ pending |
| 12-01-02 | 01 | 1 | NCLAW-03 | smoke | `bash scripts/validate-topology.sh` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/validate-topology.sh` — smoke test covering NCLAW-01 (pod Running, GPU allocated) and NCLAW-03 (loopback reachable)
- [ ] `scripts/install-agent-sandbox.sh` — operator install helper (optional, reusable)

*Note: NCLAW-01 and NCLAW-03 are live infrastructure requirements that cannot be automated in pytest unit tests. The smoke test script is the automated verification artifact.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| NemoClaw pod Running with GPU | NCLAW-01 | Requires live EKS cluster with GPU node | `kubectl get pods -l app=openclaw` shows Running; `kubectl describe pod` shows `nvidia.com/gpu: 1` allocated |
| Loopback to sidecar port reachable | NCLAW-03 | Requires live pod with shared network namespace | `kubectl exec <pod> -- curl -s localhost:8080/health` returns 200 (after Phase 13 wires sidecar) |

*Smoke test script automates these checks but requires cluster access.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
