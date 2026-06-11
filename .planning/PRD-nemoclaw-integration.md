# PRD: NemoClaw Integration For Agent-Driven Blueprint Planning

**Project:** WEKA App Store
**Date:** 2026-03-20
**Status:** Draft for GSD PRD Express Path
**Primary Goal:** Add a NemoClaw-powered planning layer and chat experience that lets users describe what they want to install in natural language, evaluates cluster and storage capacity, and converts that intent into validated `WekaAppStore` appstack YAML compatible with the operator in this repository.

## Problem Statement

The current app store flow is blueprint-centric and form-centric. Users choose from a fixed set of app options and provide a limited set of parameters. That works for known blueprints with stable inputs, but it does not handle:

- users who do not know which blueprint to choose
- users who describe goals in natural language instead of infrastructure terms
- users who need a conversational interface in the web app to evaluate cluster fit before install
- environment-specific adaptation to different Kubernetes cluster layouts
- maintainers who want help generating new appstack YAML blueprints compatible with the `WekaAppStore` CRD and operator behavior in this repo

The project needs an agent-assisted planning path that can:

- understand user intent
- inspect cluster capabilities through bounded tools
- inspect GPU, CPU, RAM, and WEKA storage capacity before selecting a plan
- choose or synthesize a compatible installation plan
- produce operator-compatible `WekaAppStore` YAML
- present a reviewable plan before cluster mutation

## Product Outcome

Users can describe a desired installation in plain English, such as:

- "Install OpenFold into a namespace for biology workloads and use the right WEKA storage configuration"
- "Set up an OSS RAG stack in a namespace for testing with the best storage class available"
- "I want to deploy a blueprint for GPU-backed inference into this cluster, but adapt it to what is already installed"

The system responds with:

- a web-based chat conversation that can ask clarification questions only when required
- a structured proposed installation plan
- a generated `WekaAppStore` appstack YAML preview
- validation results against cluster capabilities, storage capacity, and repo constraints
- an explicit apply step that submits the YAML through the existing app-store/operator pipeline

Maintainers can also use the system to generate draft reusable blueprint files for this repo, constrained to the `WekaAppStore` CRD and the current operator contract.

## Users

### Primary Users

- platform users installing blueprints from the WEKA App Store UI
- cluster admins who need blueprint installs adapted to their Kubernetes environment

### Secondary Users

- maintainers authoring new blueprint files for the app store
- internal operators troubleshooting why a blueprint will or will not fit a given cluster

## In Scope

- NemoClaw integration as a planning and reasoning layer
- natural-language install requests through a web app chat interface
- cluster inspection through tightly scoped tools
- cluster hardware-fit assessment across GPU, CPU, and RAM resources
- WEKA API inspection for storage capacity and existing filesystems
- generation of structured installation plans
- generation of `WekaAppStore` YAML compatible with this repo's CRD/operator
- preview, validation, and approval before apply
- support for generating draft blueprint source files for maintainers
- initial support for one pilot blueprint family plus the shared appstack generation pipeline

## Out of Scope

- allowing NemoClaw unrestricted direct `kubectl` or `helm` execution in v1
- replacing the existing operator reconciliation model
- replacing the existing `WekaAppStore` CRD with a new API
- full autonomous remediation of cluster problems
- broad multi-user auth redesign for the GUI
- generic Kubernetes authoring outside the WEKA app store/operator model

## Existing System Facts This PRD Locks In

- The GUI already supports applying blueprint content from rendered YAML strings.
- The operator already reconciles `WekaAppStore` resources with `spec.appStack.components[]`.
- The operator contract supports Helm components, raw Kubernetes manifests, dependencies, readiness checks, and target namespaces.
- The safest integration point is in front of the current apply path, not as a replacement for the operator.
- The NemoClaw layer must produce output constrained to this repo's CRD/operator schema.

## Proposed Solution

Introduce an agent planning workflow in the GUI and backend with three explicit layers:

### 1. Conversation Layer

The user enters a natural-language request in a chat interface in the app store UI. The system sends the request, current cluster context, and blueprint catalog metadata to NemoClaw.

### 2. Planning Layer

NemoClaw uses bounded tools to:

- inspect cluster capabilities
- inspect available GPU count, GPU model, and GPU memory capacity
- inspect available CPU and RAM capacity relevant to blueprint placement
- inspect namespaces and storage classes
- inspect WEKA storage capacity and existing filesystems through the WEKA API
- inspect known blueprint templates and supported variables
- assess whether one or more requested blueprints can fit on the current cluster at the same time
- ask follow-up questions when critical information is missing
- return a structured plan and draft YAML

### 3. Execution Layer

The WEKA app store backend:

- validates the structured plan
- validates generated YAML against repo-specific rules
- renders or normalizes the final `WekaAppStore` YAML
- presents a preview to the user
- applies only after explicit approval through the existing blueprint apply path

## Architectural Principles

### Principle 1: NemoClaw Plans, Existing Backend Executes

NemoClaw should not be the source of truth for cluster mutation. It should produce plans and drafts. The backend remains responsible for validation and application.

### Principle 2: Structured Output Before YAML

The preferred contract between NemoClaw and the backend is structured JSON representing an appstack plan. YAML is derived or validated server-side. Raw YAML-only generation is allowed only as a secondary output.

### Principle 3: Bounded Tooling

NemoClaw must only receive narrowly defined tools such as:

- cluster summary
- GPU inventory and memory summary
- CPU and RAM availability summary
- namespace listing
- storage class listing
- WEKA capability inspection
- WEKA capacity and filesystem inspection through the WEKA API
- blueprint catalog listing
- blueprint schema inspection
- appstack validation
- draft submission

### Principle 4: Explicit Human Approval

The system must not apply a generated installation plan without showing the generated result and validation status first.

### Principle 5: Repo Contract Fidelity

All generated plans and YAML must align to the `WekaAppStore` CRD in `weka-app-store-operator-chart/templates/crd.yaml` and operator behavior in `operator_module/main.py`.

## User Stories

### Install Planning

- As a platform user, I want to describe what I want to install in plain English so I do not need to understand every blueprint parameter ahead of time.
- As a platform user, I want the system to inspect my cluster and recommend a compatible install plan so I do not have to manually map my environment to blueprint inputs.
- As a platform user, I want the system to explain whether the cluster has enough GPU memory, CPU, RAM, and WEKA storage for the blueprint I want to install so I can avoid failed deployments.
- As a platform user, I want the system to tell me when my request cannot be satisfied and why so I can correct the request or cluster prerequisites.

### Blueprint Selection

- As a platform user, I want the system to choose the correct blueprint family based on my goal so I do not need to browse multiple fixed forms first.
- As a platform user, I want the system to ask me for missing decisions only when required so the interaction stays efficient.
- As a platform user, I want to use a chat interface in the web app to iterate on installation options until the plan fits the available cluster resources.

### Blueprint Authoring

- As a maintainer, I want NemoClaw to generate draft `WekaAppStore` appstack YAML for new blueprints so I can accelerate blueprint creation while preserving operator compatibility.
- As a maintainer, I want generated blueprints to include explicit dependencies, namespaces, readiness checks, and values structure compatible with the operator so that drafts are usable with minimal rework.

### Safety And Review

- As a cluster admin, I want to review the generated plan and YAML before apply so that the system does not make unexpected changes.
- As a maintainer, I want deterministic validation errors when agent output is invalid so I can harden the prompt/tooling contract over time.

## Functional Requirements

### Agent Entry Point

- [ ] **NCLAW-01**: The GUI must provide a NemoClaw-driven install workflow separate from or replacing the current fixed-form parameter flow for supported blueprints.
- [ ] **NCLAW-02**: The workflow must accept a free-text install request from the user.
- [ ] **NCLAW-03**: The workflow must support follow-up question and answer turns before plan generation completes.
- [ ] **NCLAW-04**: The web app must provide a persistent chat-style interface for the planning session, including prior user prompts, NemoClaw responses, and follow-up questions.

### Bounded Tooling

- [ ] **NCLAW-05**: The backend must expose bounded tool endpoints or plugin functions for cluster summary, namespaces, storage classes, WEKA-related capabilities, and blueprint catalog metadata.
- [ ] **NCLAW-06**: The backend must expose a validation tool that checks a structured appstack plan against repo/operator constraints before apply.
- [ ] **NCLAW-07**: Tool responses must be structured and deterministic enough for agent consumption.
- [ ] **NCLAW-08**: The backend must expose a Kubernetes resource inspection tool that reports available GPU count, GPU model, and GPU memory capacity.
- [ ] **NCLAW-09**: The backend must expose a Kubernetes resource inspection tool that reports available CPU and RAM capacity relevant to blueprint scheduling decisions.
- [ ] **NCLAW-10**: The backend must expose a WEKA API-backed tool that reports storage capacity, available space, and existing filesystems.
- [ ] **NCLAW-11**: The tooling contract must support multi-blueprint fit assessment so NemoClaw can evaluate whether multiple requested blueprints can be installed on the same cluster without oversubscribing GPU, CPU, RAM, or storage.

### Plan Generation

- [ ] **NCLAW-12**: NemoClaw must be able to return a structured plan containing blueprint family, target namespaces, component configuration, prerequisite status, unresolved questions, and reasoning summary.
- [ ] **NCLAW-13**: The structured plan must include a cluster-fit assessment covering GPU type, GPU count, GPU memory, CPU, RAM, and WEKA storage findings relevant to the requested blueprint or blueprints.
- [ ] **NCLAW-14**: The backend must transform or validate that structured plan into a `WekaAppStore` YAML document.
- [ ] **NCLAW-15**: Generated YAML must be valid against the current `WekaAppStore` CRD schema.
- [ ] **NCLAW-16**: Generated YAML must support the operator's current appstack execution contract, including `helmChart`, `kubernetesManifest`, `values`, `dependsOn`, `targetNamespace`, `waitForReady`, and `readinessCheck` where applicable.

### Preview And Approval

- [ ] **NCLAW-17**: The UI must show the proposed installation plan before apply.
- [ ] **NCLAW-18**: The UI must show the generated YAML before apply.
- [ ] **NCLAW-19**: The UI must show validation output, including cluster-fit issues and missing prerequisites.
- [ ] **NCLAW-20**: The UI must surface the hardware and storage rationale used by NemoClaw, including GPU model, GPU memory, CPU, RAM, and WEKA filesystem/capacity signals.
- [ ] **NCLAW-21**: The user must explicitly approve the plan before the backend applies it to the cluster.

### Apply Path

- [ ] **NCLAW-22**: The approved generated YAML must be submitted through the existing backend apply path rather than bypassing the current app store/operator contract.
- [ ] **NCLAW-23**: The apply path must preserve the current status and reconciliation behavior of the operator.

### Blueprint Authoring

- [ ] **NCLAW-24**: Maintainers must be able to use the integration to generate draft reusable blueprint source files in appstack YAML format.
- [ ] **NCLAW-25**: Draft blueprint generation must support repo-specific conventions for templated values, namespace handling, and operator-compatible component definitions.
- [ ] **NCLAW-26**: Generated draft blueprint files must be reviewable without direct cluster apply.

### Observability

- [ ] **NCLAW-27**: The backend must log request, plan, validation, and apply stages with stable correlation identifiers.
- [ ] **NCLAW-28**: Failure responses must identify whether the failure came from tool execution, plan validation, YAML generation, or cluster apply.

## Non-Functional Requirements

- [ ] **NCLAW-NF-01**: The integration must default to least privilege and avoid giving the agent unrestricted execution rights.
- [ ] **NCLAW-NF-02**: The system must fail closed when agent output is malformed or validation fails.
- [ ] **NCLAW-NF-03**: The system must preserve deterministic validation rules independent of model output variability.
- [ ] **NCLAW-NF-04**: The architecture must allow blueprint family expansion after the first pilot without redesigning the core planning contract.
- [ ] **NCLAW-NF-05**: The integration must be testable with mocked NemoClaw responses and mocked Kubernetes, WEKA API, and Helm interactions.
- [ ] **NCLAW-NF-06**: Capacity assessments must be auditable so a user or operator can see which hardware and storage signals informed a fit or rejection decision.

## UX Requirements

- [ ] **NCLAW-UX-01**: The install UI must have an "agent" mode with a free-text prompt input.
- [ ] **NCLAW-UX-02**: The install UI must present the interaction as a chat conversation rather than a single one-shot prompt.
- [ ] **NCLAW-UX-03**: The UI must show what information the agent used, including cluster signals and selected blueprint family.
- [ ] **NCLAW-UX-04**: The UI must distinguish between "draft plan", "validated plan", and "applied plan".
- [ ] **NCLAW-UX-05**: The UI must allow the user to edit or restart the request if the agent's proposed plan is not acceptable.
- [ ] **NCLAW-UX-06**: The UI must surface when the agent needs one or more follow-up answers before it can continue.
- [ ] **NCLAW-UX-07**: The UI must clearly explain when a requested model or blueprint does not fit because of insufficient GPU memory, GPU type, CPU, RAM, or WEKA storage capacity.

## Technical Design Constraints

### Backend Constraints

- The implementation must fit into the current FastAPI app in `app-store-gui/webapp/main.py` unless a limited refactor is required.
- Shared blueprint apply logic should be consolidated rather than duplicated further.
- Existing apply behavior for YAML file paths and rendered YAML strings must continue to work.

### Operator Constraints

- No CRD-breaking changes in v1.
- Generated plans must remain compatible with the current `WekaAppStore` operator semantics.
- The operator remains the only supported runtime reconciler for generated appstack installs.

### Security Constraints

- The integration must not ship with unrestricted direct execution from the agent to cluster admin commands.
- Any NemoClaw execution or tool actions must remain scoped and auditable.
- Secrets must not be exposed to the agent unless explicitly required and sanitized for the task.

### Deployment Constraints

- The design should allow NemoClaw to run as an external service or separately hosted component.
- The WEKA app store deployment must continue to function when NemoClaw integration is disabled.

## Data Contracts

### Structured Agent Plan Contract

The system must define and enforce a structured output contract that includes at minimum:

- request summary
- selected blueprint family
- confidence / ambiguity status
- target namespace or namespace strategy
- prerequisite findings
- GPU fit assessment including GPU type, count, and memory capacity findings
- CPU and RAM fit assessment
- selected storage class and rationale
- selected WEKA filesystem or storage inputs when required
- WEKA storage capacity and existing filesystem findings
- component list with deployment method
- values/variables to inject
- multi-blueprint coexistence assessment when applicable
- unresolved questions
- warnings and blockers
- rendered or renderable appstack payload

### Validation Contract

Validation must check at minimum:

- required top-level `WekaAppStore` fields
- `appStack.components` presence and shape
- component naming uniqueness
- Helm chart field completeness
- namespace resolution
- dependency references
- readiness-check field validity
- cluster-scoped resource handling
- GPU, CPU, RAM, and WEKA storage fit checks for supported blueprint families
- supported blueprint family inputs

## Acceptance Criteria

### Core Feasibility

1. A supported user can submit a natural-language install request from the UI.
2. The backend sends the request to NemoClaw through a bounded integration layer.
3. NemoClaw returns a structured plan or explicit follow-up questions.
4. The backend validates the plan and generates operator-compatible `WekaAppStore` YAML.
5. The UI shows the chat conversation, plan, validation status, and YAML preview before apply.
6. On approval, the backend applies the generated YAML through the existing apply path.
7. The operator reconciles the resulting `WekaAppStore` resource without requiring a new reconciliation model.

### Cluster Adaptation

1. For a cluster with multiple namespaces and storage classes, the agent can recommend a target namespace and storage class for a supported blueprint family.
2. The agent can identify available GPU count, GPU model, and GPU memory capacity and use that data to determine whether a requested model or blueprint can fit.
3. The agent can identify CPU and RAM availability and use that data when evaluating whether one or more blueprints can coexist on the cluster.
4. The agent can query WEKA through the WEKA API to determine storage capacity and existing filesystems relevant to the requested install.
5. If a required WEKA, GPU, CPU, RAM, or storage prerequisite is missing, the system blocks apply and explains the missing prerequisite.
6. If the user request is ambiguous, the system asks targeted follow-up questions before plan generation completes.

### Blueprint Authoring

1. A maintainer can request a new draft blueprint from NemoClaw for a supported blueprint family or app pattern.
2. The system returns draft appstack YAML compatible with the current `WekaAppStore` schema.
3. The draft includes valid component definitions, dependency ordering, and readiness structure where applicable.
4. The draft can be saved for manual review without cluster mutation.

### Safety

1. The system does not apply cluster changes without explicit user approval.
2. Invalid model output is rejected by deterministic validation.
3. Logs clearly identify which stage failed when the flow does not succeed.

## Implementation Phases

### Phase 1: Domain Contract And Validation Layer

Deliverables:

- define structured agent output schema
- implement backend validation for structured appstack plans
- refactor shared blueprint apply logic into reusable helper(s)
- add YAML normalization and CRD/operator compatibility checks

Phase success criteria:

- backend can accept a structured plan payload and produce validated `WekaAppStore` YAML without calling NemoClaw
- invalid plans are rejected with actionable errors

### Phase 2: NemoClaw Integration Backend

Deliverables:

- add NemoClaw client/integration module
- add bounded tool endpoints or plugin functions
- add Kubernetes capacity inspection tools for GPU, CPU, and RAM
- add WEKA API inspection tools for storage capacity and existing filesystems
- add orchestration endpoint for agent planning sessions
- support follow-up questions and resumed planning

Phase success criteria:

- backend can run an end-to-end plan session using mocked or real NemoClaw responses
- plan sessions return structured results without direct cluster mutation

### Phase 3: UI Agent Experience

Deliverables:

- add agent mode to app store UI
- add chat interface, prompt input, follow-up question handling, plan preview, validation display, and approval controls
- show generated YAML and warnings before apply

Phase success criteria:

- a user can complete a full planning conversation and reach a reviewed install plan in the UI

### Phase 4: Pilot Blueprint Family

Deliverables:

- integrate one pilot blueprint family end-to-end
- recommended pilot: `openfold`, because it already has environment-specific inputs and existing template flow
- encode cluster-fit checks for pilot prerequisites

Phase success criteria:

- users can successfully request and install the pilot blueprint through the agent workflow

### Phase 5: Blueprint Authoring Mode

Deliverables:

- maintainer-facing path to generate draft blueprint files
- save/export generated draft appstack YAML
- include repo-specific template placeholders and conventions

Phase success criteria:

- maintainers can generate and review operator-compatible draft blueprints without hand-authoring from scratch

### Phase 6: Hardening And Expansion

Deliverables:

- automated tests for planning, validation, and apply handoff
- metrics and structured logs
- support for more blueprint families
- stronger policy rules and guardrails

Phase success criteria:

- multiple blueprint families are supported
- integration is testable and safe to extend

## Recommended First Technical Tasks

1. Extract shared document mutation/apply logic from the duplicated blueprint apply functions in `app-store-gui/webapp/main.py`.
2. Define a typed internal schema for "agent appstack plan" distinct from raw YAML.
3. Implement deterministic validation and translation from structured plan to `WekaAppStore` YAML.
4. Add a backend-only mocked agent endpoint to prove the flow before wiring real NemoClaw.
5. Add UI plan preview and approval flow.
6. Integrate real NemoClaw once backend contracts are stable.
7. Pilot with `openfold`.

## Risks

### Model Output Drift

NemoClaw may produce inconsistent output unless the tool contract and structured schema are strict.

Mitigation:

- require structured output
- validate deterministically
- reject malformed plans

### Cluster Safety

Agent-generated output could propose invalid or dangerous installs.

Mitigation:

- no direct unrestricted exec in v1
- explicit preview and approval
- server-side validation before apply

### Current Code Fragility

The existing GUI apply logic is duplicated and namespace handling is fragile.

Mitigation:

- address refactor and validation before expanding apply paths

### Limited Test Coverage

The repo currently lacks automated test coverage for key deploy flows.

Mitigation:

- add focused unit and integration-style tests as part of the implementation phases

## Dependencies

- NemoClaw runtime/service availability
- a defined NemoClaw tool/plugin integration approach
- current FastAPI backend endpoints and Kubernetes access
- WEKA API access and credentials model
- current `WekaAppStore` CRD/operator contract
- blueprint catalog metadata for supported blueprint families

## Open Questions

- Will NemoClaw be hosted inside the cluster, outside the cluster, or both?
- What exact auth model will protect the agent planning endpoints?
- Should the first version support only one blueprint family or a constrained multi-family catalog?
- Should maintainers author blueprint drafts entirely through the UI, or through a separate maintainer endpoint/workflow?
- Do we want the structured plan to be persisted as a CR annotation or separate audit record?
- What is the authoritative source for available GPU memory: node labels, device plugin metrics, or a separate accelerator inventory feed?
- What is the authoritative source for WEKA filesystem and capacity data in environments with multiple WEKA clusters or tenants?

## Explicitly Deferred

- autonomous direct helm/kubectl execution by the agent
- generalized cluster repair flows
- replacing the operator
- unrestricted free-form blueprint generation without schema validation

## Definition Of Done

This PRD is complete when the implementation delivers:

- a NemoClaw-backed agent planning flow in the WEKA app store UI
- bounded backend tools for cluster/context inspection
- bounded backend tools for GPU, CPU, RAM, and WEKA storage inspection
- deterministic structured-plan validation
- generation of valid `WekaAppStore` appstack YAML
- human review and approval before apply
- successful handoff to the existing operator-based execution path
- maintainer support for draft blueprint authoring

---
*Prepared for GSD PRD Express Path*
*Suggested usage: $gsd-plan-phase <phase-number> --prd .planning/PRD-nemoclaw-integration.md*
