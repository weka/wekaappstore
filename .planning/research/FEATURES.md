# Features Research: NemoClaw Blueprint Installation Assistant

## Scope

Bounded conversational planning for the existing WEKA App Store product shape:

- chat-first planning in the GUI
- structured plan generation before YAML
- server-side validation against the current `WekaAppStore` CRD and operator contract
- explicit user approval before the existing apply path is used
- maintainer draft blueprint authoring without bypassing review

This research assumes v1 remains a blueprint planner layered onto the current catalog, YAML preview, and operator-driven execution model. It does not assume a general Kubernetes copilot.

## Table Stakes

| Feature | Why it is table stakes in this product | Complexity / dependencies |
|---|---|---|
| Goal-to-blueprint matching | Users will expect to describe an outcome and be routed to a supported blueprint family without first understanding repo internals. | Medium. Depends on clean blueprint catalog metadata, supported variable schemas, and prompt/tool constraints. |
| Minimal clarification loop | The assistant must ask follow-up questions only for missing install-critical inputs such as namespace, model choice, secrets, storage target, or sizing intent. | Medium. Depends on a structured notion of required vs optional fields per blueprint family. |
| Structured plan before YAML | The PRD already locks in structured output as the preferred contract. Users need a draft plan that can be validated before CR generation. | Medium. Depends on a stable plan schema and server-side normalization. |
| Server-validated `WekaAppStore` YAML preview | In this product, trust comes from seeing the exact CR/appstack that will be submitted through the existing apply path. | Medium. Depends on CRD-aware validation and YAML rendering from structured plan data. |
| Clear fit / no-fit result | Users need a direct answer on whether the request fits the cluster and why, before any apply step. | Medium-High. Depends on bounded Kubernetes inspection plus WEKA API inspection. |
| Explanation of GPU, CPU, RAM, and storage findings | Hardware-fit explanation is not optional for AI blueprints. Users expect the assistant to show what signals it used and what blocked or enabled the plan. | High. Depends on trustworthy cluster summary tools, WEKA storage data, and a consistent explanation format. |
| Explicit prerequisite detection | The assistant must flag missing secrets, unsupported storage classes, missing WEKA CSI/operator setup, or namespace/preinstall requirements instead of letting installs fail later. | Medium. Depends on blueprint metadata, cluster inspection, and repo-specific validation rules. |
| Review-before-apply guardrail | The current product already has a safe apply path. The assistant must preserve human approval rather than silently applying changes. | Low-Medium. Depends on UI state separation for draft, validated, and approved plan states. |
| Failure mode that stays inside supported scope | If the request cannot be satisfied, the assistant must fail closed with a concrete reason and a smaller next step. | Medium. Depends on deterministic validation categories and refusal behavior. |
| Maintainer draft authoring tied to repo conventions | Maintainers will expect draft blueprint generation to respect existing appstack semantics, dependency ordering, namespaces, and readiness checks. | Medium-High. Depends on schema templates, repo conventions, and authoring validation separate from apply. |

## Differentiators

| Feature | Why it differentiates this assistant | Complexity / dependencies |
|---|---|---|
| Multi-resource fit synthesis across Kubernetes and WEKA | Many assistants can summarize cluster state; fewer can combine GPU, CPU, RAM, storage class, filesystem, and WEKA capacity into one installability decision. | High. Depends on joining K8s and WEKA signals into a single fit model. |
| Repo-contract-aware planning | A planner that understands the actual `WekaAppStore` operator contract is more valuable than a generic YAML generator. | High. Depends on codifying the operator-supported fields and common failure cases. |
| Bounded explanation of tradeoffs, not just pass/fail | Users benefit when the assistant can say "this fits if you pick the smaller model / different namespace / different filesystem" instead of only rejecting. | Medium-High. Depends on blueprint-specific sizing tiers and policy-safe alternatives. |
| Coexistence-aware planning for multiple requested blueprints | The PRD explicitly mentions multi-blueprint fit. Explaining contention before install is a meaningful product differentiator. | High. Depends on resource accounting across requested plans, not single-blueprint heuristics. |
| Maintainer mode that converts conversation into draft reusable appstack blueprints | This turns the assistant into a blueprint authoring accelerator, not only an end-user installer. | High. Depends on reusable templates, linting/validation, and clear separation from cluster apply. |
| Explanation grounded in current cluster and current product limits | Users will trust an assistant that says "supported in this app store" or "not supported by the current operator" instead of pretending every Kubernetes pattern is available. | Medium. Depends on accurate capability metadata and explicit unsupported-case handling. |

## Anti-Features

| Anti-feature | Why it should be avoided | Complexity / dependency signal |
|---|---|---|
| Open-ended general Kubernetes copilot behavior | This product is an app-store planner. Broad cluster authoring creates safety risk and scope drift away from blueprint installs. | High risk. Would require much broader tools, auth, and policy control. |
| Unrestricted `kubectl`, `helm`, or shell access from NemoClaw | Violates the PRD safety model and bypasses the existing backend/operator execution path. | Explicitly out of scope. |
| Auto-apply after plan generation | Removes the review boundary that the current product relies on for safe mutation. | Low implementation cost, high product risk. |
| Raw YAML-only agent contract | Makes validation and normalization brittle and increases prompt sensitivity. Structured plan first is the safer product shape. | Medium-High maintenance burden if allowed. |
| Endless conversational exploration | Users are here to install or draft a blueprint, not to have a long advisory chat. The assistant should converge quickly or fail clearly. | Medium UX risk and token cost. |
| Inventing unsupported blueprint families or operator fields | A bounded assistant must not fabricate capabilities that the repo and operator do not support. | High correctness risk. |
| Pretending resource fit is guaranteed scheduling success | Cluster state is dynamic. The assistant should present fit as a validated preflight assessment, not a hard runtime guarantee. | High trust risk if overstated. |
| Hiding assumptions behind a "recommended" answer | Users need to see what model size, replica count, filesystem, and namespace assumptions were used. | Medium trust risk. |
| Autonomous remediation of cluster problems | Detecting missing prerequisites is useful; attempting to fix them autonomously is outside the current product shape. | Explicitly out of scope. |
| Direct commit/write of maintainer drafts into production blueprint paths without review | Draft generation is useful; silent repository mutation is not. | Medium operational risk. |

## User Expectations For Fit Explanations

### GPU

- Show requested versus observed GPU facts: count, model/type, and memory per GPU where available.
- Distinguish hard blockers from preferences. Example: "requires 80 GB GPU memory" is a blocker; "A100 preferred for throughput" is not.
- If the assistant proposes a smaller model or different blueprint because of GPU limits, say so explicitly.
- If GPU data is partial or unavailable, say the fit result is provisional rather than pretending certainty.

### CPU and RAM

- Explain CPU and RAM fit in workload terms, not only raw totals. Users expect to know whether the plan is blocked by requested components, replicas, or sidecars.
- Present available capacity, requested capacity, and any buffer assumption used by the validator.
- Separate cluster-wide visibility from schedulable certainty. A preflight estimate should not be presented as a scheduler guarantee.

### Storage and WEKA Fit

- Explain both Kubernetes-side and WEKA-side fit.
- Kubernetes-side expectations: storage class compatibility, RWX/RWO expectations, namespace targeting, and any CSI/operator prerequisite.
- WEKA-side expectations: target filesystem, available capacity, existing filesystems, and whether the blueprint requires a specific filesystem rather than any WEKA-backed storage.
- Make storage mismatch concrete. Example: wrong filesystem, insufficient free capacity, or missing WEKA integration should each produce different messages.

### Explanation Format

Recommended minimum explanation payload per plan:

- requested resources and assumptions
- observed cluster and WEKA signals
- fit verdict: fits, fits with changes, or does not fit
- blocking reasons
- optional adjustments that would make the plan fit
- confidence note when data is incomplete or heuristic

## Recommendations For Requirements Drafting

1. Treat "bounded install planner" as the core product, not "chatbot for Kubernetes."
2. Make fit explanation a first-class requirement alongside plan generation, because users will judge the feature mainly on whether the resource reasoning is believable.
3. Keep the plan contract strongly typed and let the backend own normalization, validation, and YAML rendering.
4. Require unsupported-case handling that names the specific boundary: unsupported blueprint family, missing prerequisite, operator contract mismatch, or cluster/storage capacity shortfall.
5. Split maintainer authoring from end-user install planning in UX and validation rules, even if they share the same planning engine.

## Notes For Sequencing

- Lowest-complexity path to value: one pilot blueprint family, structured plan schema, deterministic validation, and explanation UI.
- Highest-risk dependency: accurate and explainable fit data for GPU, CPU, RAM, and WEKA storage.
- Most important scope control: keep NemoClaw inside bounded inspection and draft generation, with no direct execution authority.
