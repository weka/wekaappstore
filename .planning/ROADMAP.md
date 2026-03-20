# Roadmap: NemoClaw Agent Planning For WEKA App Store

**Created:** 2026-03-20
**Mode:** YOLO
**Granularity:** Standard

## Overview

This roadmap adds a NemoClaw-driven planning workflow to the existing WEKA App Store without changing the runtime execution model. It sequences the work so contract and validation safety land before model integration and UI exposure.

## Phase Summary

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 1 | Plan Contract And YAML Translation | Validate structured plans and render canonical `WekaAppStore` YAML on the existing runtime path | PLAN-02, PLAN-03, PLAN-06, PLAN-07, PLAN-08, APPLY-06, APPLY-07 |
| 2 | Cluster And WEKA Inspection Signals | Add bounded inspection inputs and safety signals for fit decisions | CLSTR-01, CLSTR-02, CLSTR-03, CLSTR-04, CLSTR-05, PLAN-04, SAFE-01, SAFE-02, SAFE-03, SAFE-04 |
| 3 | Conversational Planning Sessions | Deliver chat-first planning with follow-up turns and supported-family matching | CHAT-01, CHAT-02, CHAT-03, CHAT-04, CHAT-05, PLAN-01 |
| 4 | Review, Approval, And Apply Gating | Let users review fit and YAML and require explicit approval before apply | PLAN-05, APPLY-01, APPLY-02, APPLY-03, APPLY-04, APPLY-05 |
| 5 | Maintainer Draft Authoring And Test Hardening | Add maintainer draft blueprint generation and mocked end-to-end coverage | AUTHR-01, AUTHR-02, AUTHR-03, SAFE-05 |

## Phase Details

### Phase 1: Plan Contract And YAML Translation

**Goal:** Establish the deterministic structured-plan contract, YAML translation, and safe existing-runtime handoff before NemoClaw-generated plans can submit anything.

**Plan progress:** 4 of 4 plans completed (`01-01`, `01-02`, `01-03`, `01-04`)

**Requirements:**
- PLAN-02
- PLAN-03
- PLAN-06
- PLAN-07
- PLAN-08
- APPLY-06
- APPLY-07

**Success criteria:**
1. Backend accepts a structured planning payload and validates it without any model dependency.
2. Backend renders canonical `WekaAppStore` YAML from validated plan data.
3. Validated plans still hand off to the existing apply and operator execution path.
4. Invalid plans are rejected with deterministic errors tied to repo and operator contract rules.

**Completed plan highlights:**
- `01-01` established the pytest harness and seeded planning/apply gateway fixtures.
- `01-02` established the typed structured-plan contract and layered validation rules for deterministic acceptance, warnings, and rejection.
- `01-03` extracted the shared YAML apply gateway and verified `WekaAppStore` runtime-path compatibility with mocked seam tests.
- `01-04` compiled validated plans into canonical `WekaAppStore` YAML and wired `main.py` preview/apply helpers through the shared gateway.

### Phase 2: Cluster And WEKA Inspection Signals

**Goal:** Build the bounded inspection layer and safety signals needed for trustworthy cluster-fit and storage-fit decisions.

**Requirements:**
- CLSTR-01
- CLSTR-02
- CLSTR-03
- CLSTR-04
- CLSTR-05
- PLAN-04
- SAFE-01
- SAFE-02
- SAFE-03
- SAFE-04

**Success criteria:**
1. Backend can produce bounded inspection snapshots for namespaces, storage classes, GPU, CPU, RAM, and WEKA state.
2. Fit signals include freshness and confidence metadata and fail closed when required data is incomplete.
3. NemoClaw-facing tool calls are bounded, auditable, and read-only.
4. The system can classify failures by stage: inspection, validation, YAML generation, or apply handoff.

### Phase 3: Conversational Planning Sessions

**Goal:** Deliver the chat-first planning workflow with follow-up turns, supported-family matching, and safe session management.

**Requirements:**
- CHAT-01
- CHAT-02
- CHAT-03
- CHAT-04
- CHAT-05
- PLAN-01

**Success criteria:**
1. Users can start a planning session from the UI and submit free-text requests.
2. The system can match a supported blueprint family or explicitly say none fit.
3. Users can answer follow-ups, review session history, and restart or abandon drafts safely.

### Phase 4: Review, Approval, And Apply Gating

**Goal:** Expose fit, validation, and YAML review to users and keep all applies behind an explicit approval gate.

**Requirements:**
- PLAN-05
- APPLY-01
- APPLY-02
- APPLY-03
- APPLY-04
- APPLY-05

**Success criteria:**
1. Users can review the plan, YAML, validation results, and fit rationale before apply.
2. Multi-blueprint oversubscription is detected and blocks apply.
3. No apply occurs without explicit approval.

### Phase 5: Maintainer Draft Authoring And Test Hardening

**Goal:** Add maintainer draft blueprint generation and the mocked test coverage required for safer rollout.

**Requirements:**
- AUTHR-01
- AUTHR-02
- AUTHR-03
- SAFE-05

**Success criteria:**
1. Maintainers can request draft blueprints without applying them.
2. Draft output is reviewable separately and follows repo conventions.
3. The workflow is testable with mocked NemoClaw, Kubernetes, and WEKA inputs.

## Phase Ordering Rationale

- Phase 1 comes first because the planner needs a stable contract and canonical YAML path before any UI or model integration is trustworthy.
- Phase 2 comes before Phase 3 so chat and planning decisions are driven by bounded, auditable cluster and WEKA signals.
- Phase 4 explicitly separates review and apply gating from conversational flow so the approval boundary stays visible and testable.
- Phase 5 combines maintainer authoring with test hardening because both reuse the same plan and validation pipeline and both are safer once the user-facing flow already works.

## Coverage

| Phase | Requirement Count |
|-------|-------------------|
| 1 | 7 |
| 2 | 10 |
| 3 | 6 |
| 4 | 6 |
| 5 | 4 |

**Total v1 requirements:** 33
**Mapped requirements:** 33
**Unmapped requirements:** 0 ✓

---
*Roadmap created: 2026-03-20*
*All v1 requirements covered: yes*
