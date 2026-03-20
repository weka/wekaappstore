---
phase: 02
slug: cluster-and-weka-inspection-signals
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-20
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none - `app-store-gui/requirements.txt` installs the runner |
| **Quick run command** | `cd app-store-gui && python -m pytest tests/planning -q` |
| **Full suite command** | `cd app-store-gui && python -m pytest tests/planning -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd app-store-gui && python -m pytest tests/planning -q`
- **After every plan wave:** Run `cd app-store-gui && python -m pytest tests/planning -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | CLSTR-05 | unit | `cd app-store-gui && python -m pytest tests/planning/test_inspection_contract.py -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | PLAN-04 | unit | `cd app-store-gui && python -m pytest tests/planning/test_inspection_contract.py -q` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | SAFE-04 | unit | `cd app-store-gui && python -m pytest tests/planning/test_inspection_contract.py -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | CLSTR-01 | integration | `cd app-store-gui && python -m pytest tests/planning/test_cluster_inspection.py -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | CLSTR-02 | integration | `cd app-store-gui && python -m pytest tests/planning/test_cluster_inspection.py -q` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 2 | CLSTR-03 | integration | `cd app-store-gui && python -m pytest tests/planning/test_cluster_inspection.py -q` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | CLSTR-04 | integration | `cd app-store-gui && python -m pytest tests/planning/test_weka_inspection.py -q` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | SAFE-03 | integration | `cd app-store-gui && python -m pytest tests/planning/test_weka_inspection.py -q` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 3 | SAFE-01 | integration | `cd app-store-gui && python -m pytest tests/planning/test_inspection_integration.py -q` | ❌ W0 | ⬜ pending |
| 02-04-02 | 04 | 3 | SAFE-02 | integration | `cd app-store-gui && python -m pytest tests/planning/test_inspection_integration.py -q` | ❌ W0 | ⬜ pending |
| 02-04-03 | 04 | 3 | CLSTR-05 | integration | `cd app-store-gui && python -m pytest tests/planning/test_inspection_integration.py -q` | ❌ W0 | ⬜ pending |
| 02-04-04 | 04 | 3 | PLAN-04 | integration | `cd app-store-gui && python -m pytest tests/planning/test_inspection_integration.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers framework installation and baseline pytest execution from Phase 1.
- `app-store-gui/tests/planning/test_inspection_contract.py` - add snapshot and fit-signal contract tests.
- `app-store-gui/tests/planning/test_cluster_inspection.py` - add mocked Kubernetes inspection tests.
- `app-store-gui/tests/planning/test_weka_inspection.py` - add mocked WEKA inspection tests.
- `app-store-gui/tests/planning/test_inspection_integration.py` - add request-level correlation and stage-classification tests.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Inspection data looks sensible against a live cluster with mixed CPU and GPU nodes | CLSTR-02 | Automated tests will mock Kubernetes responses rather than depend on a real cluster topology | Run the app against a representative cluster, call the new inspection endpoint or helper path, and confirm GPU type, memory, and partial-data blockers match observed node labels and allocatable values |
| WEKA filesystem inventory matches the operator-visible cluster state | CLSTR-04 | Automated tests should not require a live WEKA cluster or operator installation | Against a cluster with WEKA operator resources installed, call the WEKA inspection path and compare reported filesystems and capacities with visible `WekaCluster` status |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-03-20
