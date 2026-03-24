---
phase: 14
slug: end-to-end-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | bash scripts + kubectl assertions + manual chat transcript |
| **Config file** | none — scripts are standalone |
| **Quick run command** | `bash scripts/validate-phase13.sh --live wekaappstore` |
| **Full suite command** | Quick run + `kubectl get wekaappstores -n wekaappstore` + chat transcript review |
| **Estimated runtime** | ~30 seconds (automated) + manual chat session |

---

## Sampling Rate

- **After every task commit:** Run `bash scripts/validate-phase13.sh --live wekaappstore`
- **After every plan wave:** Full evidence set: kubectl outputs + chat transcript
- **Before `/gsd:verify-work`:** All 4 E2E evidence files present + chat transcript shows SKILL.md steps
- **Max feedback latency:** 30 seconds (automated checks)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | prereq | smoke | `bash scripts/validate-phase14-prereqs.sh` | ❌ W0 | ⬜ pending |
| 14-02-01 | 02 | 2 | E2E-01 | manual+kubectl | `kubectl get nodes -o wide` | ✅ | ⬜ pending |
| 14-02-02 | 02 | 2 | E2E-02 | manual+kubectl | `kubectl exec $POD -c weka-mcp-sidecar -- ls /app/blueprints/` | ✅ | ⬜ pending |
| 14-02-03 | 02 | 2 | E2E-03 | manual+kubectl | `kubectl get wekaappstores -n wekaappstore` | ✅ | ⬜ pending |
| 14-02-04 | 02 | 2 | E2E-04 | manual+kubectl | `kubectl get wekaappstore <name> -o jsonpath='{.status.appStackPhase}'` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/validate-phase14-prereqs.sh` — confirms Phase 13 sidecar healthy + OSS Rag blueprint present in catalog
- [ ] `evidence/` directory — storage location for chat transcripts and kubectl outputs

*E2E requirements are predominantly manual validation. Automated commands provide kubectl-side evidence.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Agent returns real cluster data via chat | E2E-01 | Requires interactive agent chat session | Ask agent about cluster state; compare response to kubectl output |
| Agent lists blueprints via chat | E2E-02 | Requires interactive agent chat session | Ask agent to list blueprints; verify OSS Rag appears |
| Agent generates, validates, applies CR | E2E-03 | Requires interactive agent chat + user approval | Follow SKILL.md 12-step workflow; approve apply |
| Agent reports full deployment success | E2E-04 | Requires operator reconciliation + agent status check | Ask agent for deployment status; wait for Ready |

*All E2E requirements require human-in-the-loop chat interaction. This is by design — the milestone goal is validating the agent chat experience.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
