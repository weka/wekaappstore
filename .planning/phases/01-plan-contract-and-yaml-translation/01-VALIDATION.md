---
phase: 1
slug: plan-contract-and-yaml-translation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-20
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py tests/planning/test_compiler.py tests/planning/test_apply_gateway.py -q` |
| **Full suite command** | `cd app-store-gui && python -m pytest tests/planning -q` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py tests/planning/test_compiler.py tests/planning/test_apply_gateway.py -q`
- **After every plan wave:** Run `cd app-store-gui && python -m pytest tests/planning -q`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | PLAN-06 | setup | `cd app-store-gui && python -m pytest --version` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | PLAN-07 | setup | `cd app-store-gui && python -m pytest tests/planning -q` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | PLAN-08 | setup | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py tests/planning/test_compiler.py tests/planning/test_apply_gateway.py -q` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | PLAN-02 | unit | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py -q -k models` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | PLAN-03 | unit | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py -q` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 2 | PLAN-06 | unit | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py -q` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 2 | APPLY-06 | integration | `cd app-store-gui && python -m pytest tests/planning/test_apply_gateway.py -q -k gateway` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 2 | APPLY-07 | integration | `cd app-store-gui && python -m pytest tests/planning/test_apply_gateway.py -q` | ❌ W0 | ⬜ pending |
| 1-03-03 | 03 | 2 | APPLY-07 | integration | `cd app-store-gui && python -m pytest tests/planning/test_apply_gateway.py -q` | ❌ W0 | ⬜ pending |
| 1-04-01 | 04 | 3 | PLAN-08 | unit | `cd app-store-gui && python -m pytest tests/planning/test_compiler.py -q` | ❌ W0 | ⬜ pending |
| 1-04-02 | 04 | 3 | APPLY-06 | integration | `cd app-store-gui && python -m pytest tests/planning/test_plan_contract.py tests/planning/test_compiler.py tests/planning/test_apply_gateway.py -q` | ❌ W0 | ⬜ pending |
| 1-04-03 | 04 | 3 | APPLY-07 | integration | `cd app-store-gui && python -m pytest tests/planning -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `app-store-gui/tests/planning/test_plan_contract.py` — contract and validation coverage for PLAN-02, PLAN-03, PLAN-06, PLAN-07
- [ ] `app-store-gui/tests/planning/test_compiler.py` — canonical YAML translation coverage for PLAN-08
- [ ] `app-store-gui/tests/planning/test_apply_gateway.py` — shared handoff coverage for APPLY-06 and APPLY-07
- [ ] `app-store-gui/tests/conftest.py` — shared fixtures for plan payloads and compiled `WekaAppStore` documents
- [ ] `pytest` in `app-store-gui/requirements.txt` — first Python application-level test harness entry

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Canonical YAML is readable and stable for humans reviewing previews | PLAN-08 | Automated tests can assert structure but not review ergonomics | Compile representative plans and inspect the emitted YAML for consistency and clarity |
| Shared apply handoff still aligns with existing operator expectations | APPLY-06 / APPLY-07 | Final confidence depends on runtime semantics beyond mocked tests | Apply a representative canonical `WekaAppStore` in a disposable environment and confirm it follows the existing runtime path |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
