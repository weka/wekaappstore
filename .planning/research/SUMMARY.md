# Project Research Summary

**Project:** NemoClaw Agent Planning For WEKA App Store
**Domain:** Bounded conversational blueprint planning for a brownfield Kubernetes app-store platform
**Researched:** 2026-03-20
**Confidence:** MEDIUM

## Executive Summary

This project is not a generic Kubernetes copilot. It is a bounded planning layer for the existing WEKA App Store that must convert user intent into validated `WekaAppStore` app stacks without bypassing the current backend and operator execution model. The research is consistent across stack, feature, architecture, and pitfalls: keep NemoClaw inside the FastAPI/backend boundary, keep the plan contract structured and typed, and keep mutation authority in deterministic code paths.

The strongest product requirement is not chat by itself. It is believable preflight reasoning about whether a requested blueprint fits the current cluster and WEKA environment. That means GPU type, GPU count, GPU memory, CPU, RAM, storage class, WEKA capacity, and existing filesystems all need to be surfaced as first-class validation signals rather than hidden model reasoning.

The main implementation risk is not model integration. It is contract drift and brownfield fragility: duplicated apply logic, weak validation boundaries, thin automated tests, and live cluster inspection paths that could become slow or misleading if reused naively. The roadmap should therefore start with backend extraction, typed contracts, validation, and inspection adapters before it adds full NemoClaw orchestration and chat UX.

## Key Findings

### Recommended Stack

The recommended stack stays close to the current repo: Python, FastAPI, Jinja, Kubernetes Python client, PyYAML, a narrow WEKA API adapter, and a dedicated NemoClaw adapter. Typed Pydantic models should define the structured plan, tool payloads, validation result, and planning session state. This keeps the backend as the enforcement point for the `WekaAppStore` contract while avoiding a premature microservice split or frontend rewrite.

**Core technologies:**
- Python and FastAPI: extend the current backend without introducing a second service boundary too early
- Pydantic contracts: keep plan and tool outputs structured, versioned, and testable
- Kubernetes Python client and WEKA API adapter: provide deterministic bounded inspection instead of raw cluster exec
- PyYAML translation layer: generate canonical `WekaAppStore` YAML server-side after validation

### Expected Features

The table-stakes product shape is a bounded install planner that can map natural language to supported blueprint families, ask only necessary follow-up questions, validate fit before apply, and explain why a plan fits or fails. The differentiators are coexistence-aware planning across multiple resources, repo-contract-aware output, and maintainer-facing draft blueprint authoring. The anti-features are broad Kubernetes copilot behavior, unrestricted exec, auto-apply, and YAML-first contracts.

**Must have (table stakes):**
- Goal-to-blueprint matching for supported blueprint families
- Minimal clarification loop for missing install-critical inputs
- Structured plan before YAML
- Server-validated `WekaAppStore` YAML preview
- Clear fit / no-fit result with GPU, CPU, RAM, and WEKA storage explanation
- Explicit prerequisite detection and review-before-apply

**Should have (competitive):**
- Coexistence-aware multi-blueprint fit assessment
- Tradeoff-aware recommendations such as smaller model or different filesystem choices
- Maintainer draft blueprint authoring using repo conventions

**Defer (v2+):**
- Broader blueprint family expansion beyond the pilot once the planner contract is stable
- Advanced policy and hardening features after the baseline workflow is validated

### Architecture Approach

The architecture recommendation is to place NemoClaw in a planning subsystem inside the FastAPI/backend boundary. That subsystem should own chat sessions, bounded inspection, structured plan generation, validation, and YAML preview. The current `WekaAppStore` CRD and operator remain the only execution model after explicit approval.

**Major components:**
1. Chat/UI layer — captures intent, displays plan state, validation, and approval
2. Planning session service — manages conversation state, inspection snapshots, plan drafts, and approval state
3. NemoClaw adapter — normalizes model interaction and bounded tool usage
4. Inspection adapters — Kubernetes and WEKA read-only summaries
5. Plan validator / translator — enforces operator contract and renders canonical YAML
6. Existing apply gateway and operator runtime — unchanged execution path after approval

### Critical Pitfalls

1. **Contract drift between NemoClaw output and `WekaAppStore` reality** — keep the backend JSON-first and validate against repo/operator rules before YAML preview or apply.
2. **Namespace behavior drift in generated installs** — unify the existing duplicated apply paths before adding planner-generated YAML.
3. **Using total hardware instead of schedulable capacity** — build tool outputs around allocatable and confidence-scored fit data, not raw installed resources.
4. **Treating WEKA inspection as static truth** — timestamp responses and revalidate storage-dependent assumptions immediately before apply.
5. **Weak observability between planning and runtime** — add stable correlation IDs, structured stage logging, and deterministic failure categorization.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Contracts And Backend Extraction
**Rationale:** Brownfield fragility is the biggest delivery risk.
**Delivers:** Shared apply helper extraction, typed plan schema, validation contracts, and canonical YAML translation.
**Addresses:** structured plan before YAML, server-side validation, operator contract fidelity.
**Avoids:** contract drift and namespace-handling regressions.

### Phase 2: Inspection And NemoClaw Backend Integration
**Rationale:** The model should not be integrated before bounded tools and fit inputs exist.
**Delivers:** Kubernetes capacity adapters, WEKA API adapters, planning sessions, NemoClaw adapter, and mocked end-to-end planning loop.
**Uses:** typed contracts and extracted backend services from Phase 1.
**Implements:** planning service, inspection adapters, orchestration adapter.

### Phase 3: Chat UX And Review Flow
**Rationale:** UI should sit on a deterministic backend payload, not drive backend design.
**Delivers:** chat interface, follow-up turns, validation display, hardware/storage rationale, YAML preview, and approval states.
**Implements:** chat/UI layer and state transitions.

### Phase 4: Pilot Blueprint Family
**Rationale:** Fit logic needs one concrete blueprint family before broadening.
**Delivers:** one supported end-to-end blueprint family with encoded fit rules and acceptance tests.
**Addresses:** real cluster-fit reasoning rather than generic prompts.

### Phase 5: Maintainer Authoring Mode
**Rationale:** Reuse the validated planner and translator after install planning works.
**Delivers:** draft blueprint generation, export/review workflow, and repo-convention checks.

### Phase 6: Hardening And Expansion
**Rationale:** Expansion before hardening would multiply existing failure modes.
**Delivers:** broader test coverage, better observability, policy tightening, and more blueprint families.

### Phase Ordering Rationale

- Validation and translation must come before model integration to prevent prompt-led backend drift.
- Inspection adapters must come before chat UI so the conversation is backed by trustworthy signals.
- The pilot blueprint should come before family expansion so fit logic becomes reusable backend logic instead of prompt folklore.
- Hardening must precede broad expansion because the repo already has fragile apply and reconcile paths.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Exact Kubernetes sources for GPU memory and schedulable capacity, plus WEKA API freshness and auth model
- **Phase 4:** Pilot blueprint-specific sizing and prerequisite rules
- **Phase 6:** Observability and policy hardening patterns that fit this runtime model

Phases with standard patterns:
- **Phase 1:** Typed validation and service extraction are straightforward brownfield backend work
- **Phase 3:** Chat and approval UI flow is standard once backend contracts are stable

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Strong repo-local fit, but exact NemoClaw deployment details still need implementation decisions. |
| Features | HIGH | The PRD is explicit about user needs, boundaries, and fit-explanation expectations. |
| Architecture | HIGH | The current repo shape strongly supports backend-first integration in front of the apply path. |
| Pitfalls | HIGH | Risks line up closely with known codebase concerns and PRD guardrails. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Authoritative GPU memory and schedulable-capacity source: define whether this comes from node allocatable data, device plugin metrics, or another adapter.
- WEKA multi-cluster or multi-tenant targeting: define how the backend selects the correct WEKA context for inspection.
- Planning session persistence: choose whether v1 uses in-memory, file-backed, or store-backed session state.
- Maintainer authoring output target: define how draft blueprints are stored or exported without creating silent repo mutations.

## Sources

### Primary
- `.planning/PROJECT.md`
- `.planning/PRD-nemoclaw-integration.md`
- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STACK.md`
- `.planning/codebase/CONCERNS.md`

---
*Research completed: 2026-03-20*
*Ready for roadmap: yes*
