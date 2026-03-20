# Requirements: NemoClaw Agent Planning For WEKA App Store

**Defined:** 2026-03-20
**Core Value:** Users can describe what they want to deploy, and the system turns that into a safe, validated WEKA App Store installation plan that actually fits the target cluster before anything is applied.

## v1 Requirements

### Chat Planning

- [ ] **CHAT-01**: User can start a NemoClaw planning session from the WEKA App Store UI.
- [ ] **CHAT-02**: User can submit a free-text install request in a chat-style interface.
- [ ] **CHAT-03**: User can answer follow-up questions from NemoClaw within the same planning session.
- [ ] **CHAT-04**: User can see prior prompts, agent responses, and unanswered follow-up questions for the current session.
- [ ] **CHAT-05**: User can restart or abandon a draft planning session without applying changes.

### Cluster Inspection

- [x] **CLSTR-01**: The backend can provide a bounded cluster summary for planning sessions, including namespaces and storage classes.
- [x] **CLSTR-02**: The backend can provide bounded GPU inventory data including available GPU count, GPU model, and GPU memory capacity.
- [x] **CLSTR-03**: The backend can provide bounded CPU and RAM availability data relevant to blueprint scheduling decisions.
- [x] **CLSTR-04**: The backend can provide WEKA storage inspection data including storage capacity, available space, and existing filesystems.
- [x] **CLSTR-05**: The backend can distinguish complete versus partial inspection data and expose confidence or freshness signals to the planner and UI.

### Planning And Fit Validation

- [ ] **PLAN-01**: The system can map a natural-language request to a supported blueprint family or explicitly report that no supported family fits the request.
- [x] **PLAN-02**: The system can produce a structured installation plan before producing or previewing YAML.
- [x] **PLAN-03**: The structured plan includes blueprint family, namespace strategy, component configuration, prerequisites, unresolved questions, and reasoning summary.
- [x] **PLAN-04**: The structured plan includes cluster-fit findings for GPU type, GPU count, GPU memory, CPU, RAM, and WEKA storage.
- [ ] **PLAN-05**: The system can assess whether multiple requested blueprints can coexist on the same cluster without oversubscribing required resources.
- [x] **PLAN-06**: The backend can validate a structured plan against repo-specific `WekaAppStore` and operator constraints before apply.
- [x] **PLAN-07**: The backend can reject malformed or unsupported agent output with deterministic validation errors.
- [x] **PLAN-08**: The backend can render canonical `WekaAppStore` YAML from a validated structured plan.

### Review And Apply

- [ ] **APPLY-01**: User can review the proposed installation plan before apply.
- [ ] **APPLY-02**: User can review the generated `WekaAppStore` YAML before apply.
- [ ] **APPLY-03**: User can review validation results, including blocking cluster-fit and storage-fit issues, before apply.
- [ ] **APPLY-04**: User can see the hardware and storage rationale used for the fit decision, including GPU, CPU, RAM, and WEKA findings.
- [ ] **APPLY-05**: The system requires explicit user approval before submitting generated YAML to the cluster.
- [x] **APPLY-06**: Approved plans are submitted through the existing backend apply path rather than bypassing the current app store/operator contract.
- [x] **APPLY-07**: The runtime execution path remains the existing `WekaAppStore` CRD and operator reconciliation flow.

### Blueprint Authoring

- [ ] **AUTHR-01**: Maintainer can request a draft blueprint definition from NemoClaw without applying it to the cluster.
- [ ] **AUTHR-02**: Draft blueprint output is compatible with repo conventions for app stack components, dependencies, namespaces, and readiness checks.
- [ ] **AUTHR-03**: Draft blueprint output is reviewable as an artifact separate from the end-user install workflow.

### Observability And Safety

- [x] **SAFE-01**: Planning sessions, validation runs, and apply handoffs include stable correlation identifiers.
- [x] **SAFE-02**: Failure responses identify whether the failure came from inspection, agent output, plan validation, YAML generation, or cluster apply.
- [x] **SAFE-03**: Agent-callable tools are bounded and auditable, with no unrestricted direct `kubectl`, `helm`, or shell execution.
- [x] **SAFE-04**: The system fails closed when required inspection data is missing or too incomplete to make a reliable fit decision.
- [ ] **SAFE-05**: The integration can be tested with mocked NemoClaw responses and mocked Kubernetes and WEKA inspection results.

## v2 Requirements

### Blueprint Expansion

- **BPX-01**: The system supports a broader multi-family blueprint catalog beyond the pilot family.
- **BPX-02**: The system can recommend alternative blueprint families when the first choice does not fit.

### Policy And Hardening

- **HARD-01**: The system applies richer policy rules for tenant, quota, and placement constraints beyond baseline resource fit.
- **HARD-02**: The system preserves detailed audit history of planning sessions and approved plan artifacts for long-term review.
- **HARD-03**: The system provides stronger rollback and partial-failure guidance after runtime execution failures.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Unrestricted direct `kubectl`, `helm`, or shell execution by NemoClaw | Violates the bounded-tool safety model and bypasses existing backend and operator controls |
| Automatic apply immediately after plan generation | Removes required human review and approval safeguards |
| Replacing the `WekaAppStore` CRD or current operator reconciliation model | The project builds on the existing runtime contract rather than redesigning it |
| Generic Kubernetes copilot behavior outside supported blueprint installs | Would create major scope and safety drift away from the app-store product |
| Broad GUI authentication redesign | Important separately, but not required to deliver this planning workflow |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CHAT-01 | Phase 3 | Pending |
| CHAT-02 | Phase 3 | Pending |
| CHAT-03 | Phase 3 | Pending |
| CHAT-04 | Phase 3 | Pending |
| CHAT-05 | Phase 3 | Pending |
| CLSTR-01 | Phase 2 | Complete |
| CLSTR-02 | Phase 2 | Complete |
| CLSTR-03 | Phase 2 | Complete |
| CLSTR-04 | Phase 2 | Complete |
| CLSTR-05 | Phase 2 | Complete |
| PLAN-01 | Phase 3 | Pending |
| PLAN-02 | Phase 1 | Complete |
| PLAN-03 | Phase 1 | Complete |
| PLAN-04 | Phase 2 | Complete |
| PLAN-05 | Phase 4 | Pending |
| PLAN-06 | Phase 1 | Complete |
| PLAN-07 | Phase 1 | Complete |
| PLAN-08 | Phase 1 | Complete |
| APPLY-01 | Phase 4 | Pending |
| APPLY-02 | Phase 4 | Pending |
| APPLY-03 | Phase 4 | Pending |
| APPLY-04 | Phase 4 | Pending |
| APPLY-05 | Phase 4 | Pending |
| APPLY-06 | Phase 1 | Complete |
| APPLY-07 | Phase 1 | Complete |
| AUTHR-01 | Phase 5 | Pending |
| AUTHR-02 | Phase 5 | Pending |
| AUTHR-03 | Phase 5 | Pending |
| SAFE-01 | Phase 2 | Complete |
| SAFE-02 | Phase 2 | Complete |
| SAFE-03 | Phase 2 | Complete |
| SAFE-04 | Phase 2 | Complete |
| SAFE-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after completing Phase 1 plan execution*
