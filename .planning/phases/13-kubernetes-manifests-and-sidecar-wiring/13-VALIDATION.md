---
phase: 13
slug: kubernetes-manifests-and-sidecar-wiring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-24
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | bash scripts + kubectl dry-run + pytest (existing) |
| **Config file** | none — scripts created in this phase |
| **Quick run command** | `kubectl apply --dry-run=client -n wekaappstore -f k8s/agent-sandbox/` |
| **Full suite command** | `kubectl apply --dry-run=client -n wekaappstore -f k8s/agent-sandbox/ && cd mcp-server && python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds (dry-run) + ~15 seconds (pytest) |

---

## Sampling Rate

- **After every task commit:** Run `kubectl apply --dry-run=client -n wekaappstore -f k8s/agent-sandbox/`
- **After every plan wave:** Run full suite (dry-run + pytest)
- **Before `/gsd:verify-work`:** Live cluster checks via `scripts/validate-phase13.sh`
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | K8S-02, K8S-05 | smoke | `kubectl apply --dry-run=client -f k8s/agent-sandbox/mcp-rbac.yaml` | ❌ W0 | ⬜ pending |
| 13-01-02 | 01 | 1 | K8S-01, K8S-03, K8S-04, K8S-05, NCLAW-02 | smoke | `kubectl apply --dry-run=client -f k8s/agent-sandbox/openclaw-sandbox.yaml` | ✅ | ⬜ pending |
| 13-01-03 | 01 | 1 | NCLAW-04 | smoke | `kubectl apply --dry-run=client -f k8s/agent-sandbox/mcp-skill-configmap.yaml` | ❌ W0 | ⬜ pending |
| 13-02-01 | 02 | 2 | K8S-01..05, NCLAW-02, NCLAW-04 | integration | `bash scripts/validate-phase13.sh` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `k8s/agent-sandbox/mcp-rbac.yaml` — ServiceAccount, ClusterRole, ClusterRoleBinding (K8S-02, K8S-05)
- [ ] `k8s/agent-sandbox/mcp-skill-configmap.yaml` — SKILL.md as ConfigMap (NCLAW-04)
- [ ] `scripts/validate-phase13.sh` — live cluster validation script for all requirements

*Existing `openclaw-sandbox.yaml` is modified in-place (already exists from Phase 12).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| MCP sidecar /health returns 200 before tool registration | K8S-03 | Requires live pod with sidecar running | `kubectl logs openclaw-sandbox -c mcp-sidecar -n wekaappstore \| grep health` |
| openclaw.json visible in pod logs | K8S-05 | Requires live pod after init container runs | `kubectl exec openclaw-sandbox -c openclaw -n wekaappstore -- cat /home/node/.openclaw/openclaw.json` |
| Blueprints accessible at BLUEPRINTS_DIR | K8S-04 | Requires live pod with git-sync running | `kubectl exec openclaw-sandbox -c mcp-sidecar -n wekaappstore -- ls $BLUEPRINTS_DIR` |

*`scripts/validate-phase13.sh` automates these live checks.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 20s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
