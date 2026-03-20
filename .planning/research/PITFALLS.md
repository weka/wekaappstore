# NemoClaw Integration Pitfalls

Concise failure-mode checklist for roadmap and phase planning.

## 1. Agent output drifts away from the actual `WekaAppStore` contract

- Why this matters: The PRD requires structured plans first, but the current runtime still depends on the CRD schema and `operator_module/main.py` semantics. NemoClaw can produce plausible plans that are invalid for this repo.
- Warning signs:
  - Generated plans omit `appStack.components[]`, `helmChart.name`, `targetNamespace`, or readiness fields the operator expects.
  - The agent returns YAML that "looks right" but fails CRD validation or reconcile-time dependency ordering.
  - The backend starts accepting YAML-first outputs because JSON contract enforcement is weak.
- Prevention strategy:
  - Make the backend contract JSON-first and versioned.
  - Validate against both the CRD shape and repo-specific operator rules before any YAML preview/apply.
  - Normalize generated plans server-side instead of trusting model-produced YAML structure.
  - Add fixtures for valid and invalid plans taken from real blueprints in this repo.
- Phase: Phase 1, reinforced in Phase 2 and Phase 6.

## 2. Namespace handling breaks generated installs in subtle ways

- Why this matters: `app-store-gui/webapp/main.py` already has duplicated namespace mutation logic and only rewrites `components[].targetNamespace` when the field already exists. Agent-generated plans will amplify that fragility.
- Warning signs:
  - Same plan behaves differently when applied from rendered YAML versus file-backed YAML.
  - Components land in the CR namespace instead of the intended workload namespace.
  - Multi-component stacks partially deploy because one component inherited a default namespace unexpectedly.
- Prevention strategy:
  - Refactor both blueprint apply paths into one shared helper before adding agent apply handoff.
  - Define a single namespace resolution policy for CR namespace, `appStack.defaultNamespace`, `components[].targetNamespace`, and raw manifests.
  - Add tests for namespaced and cluster-scoped multi-doc blueprints, including components without explicit `targetNamespace`.
- Phase: Phase 1.

## 3. Capacity tools report cluster totals instead of schedulable capacity

- Why this matters: The PRD asks NemoClaw to decide fit for GPU, CPU, RAM, and storage. In this codebase, live cluster summaries already do broad scans; naive totals will overstate what can actually be scheduled.
- Warning signs:
  - Plans are approved even though the cluster is fragmented, tainted, or already committed.
  - "Available" GPU memory is derived from node inventory, not allocatable/schedulable workload headroom.
  - Users see successful planning followed by runtime Pending pods.
- Prevention strategy:
  - Define tool outputs around schedulable/allocatable capacity, not just installed hardware.
  - Record whether values are raw totals, allocatable amounts, or heuristic estimates.
  - Fail closed when capacity confidence is low or the signal is incomplete.
  - Start with one pilot blueprint family and encode its concrete fit rules before broadening the planner.
- Phase: Phase 2, refined in Phase 4 and Phase 6.

## 4. WEKA storage inspection is treated as static truth

- Why this matters: The planner will use WEKA API responses to size installs, but WEKA free space and existing filesystem state can change between planning and apply.
- Warning signs:
  - Plans are cached or reused without a freshness window.
  - The UI shows storage-fit as definitive even though the underlying filesystem state is minutes old.
  - Apply fails because the target filesystem no longer exists or available capacity dropped after planning.
- Prevention strategy:
  - Put timestamps and source metadata on every WEKA capacity result.
  - Revalidate WEKA-dependent constraints immediately before apply.
  - Distinguish "observed state" from "guaranteed reservation"; do not present capacity as reserved unless the backend actually reserves it.
  - Return deterministic validation errors when plan-time and apply-time WEKA checks diverge.
- Phase: Phase 2, reinforced in Phase 3 and Phase 6.

## 5. Bounded tools quietly become a second mutation path

- Why this matters: The PRD explicitly says NemoClaw must not get unrestricted `kubectl` or `helm`, but this repo already mixes Python clients, `kr8s`, `kubectl`, and `helm` in runtime paths. Tool creep can erode that boundary.
- Warning signs:
  - Tool endpoints start creating namespaces, secrets, or draft resources "for convenience."
  - Planning sessions call backend helpers that reuse deploy/apply functions instead of read-only inspection functions.
  - The distinction between "plan", "validate", and "apply" is not visible in logs.
- Prevention strategy:
  - Separate read-only inspection adapters from mutation adapters at the module boundary.
  - Enforce least-privilege RBAC for planning-specific service calls.
  - Add explicit allowlists for agent-callable tools and reject everything else.
  - Emit audit logs showing every tool call and whether it was read-only or mutating.
- Phase: Phase 2, hardened in Phase 6.

## 6. Follow-up question flow loses determinism across turns

- Why this matters: The PRD needs multi-turn planning, but the backend and UI are currently request-centric. Without a stable planning session model, follow-up answers can invalidate earlier assumptions silently.
- Warning signs:
  - The same conversation yields different plans after a page refresh or retry.
  - Users answer a follow-up question and earlier hardware/storage assumptions disappear from the final plan.
  - Correlation between prompt, tool outputs, validation, and apply preview is missing in logs.
- Prevention strategy:
  - Store a structured planning session state with immutable snapshots of tool results used for each draft.
  - Require the backend to regenerate and revalidate the whole plan after every follow-up answer.
  - Use stable correlation IDs across chat, tool calls, validation, YAML preview, and apply handoff.
- Phase: Phase 2 and Phase 3.

## 7. UI treats draft plans as validated plans

- Why this matters: The PRD requires clear separation between draft, validated, and applied states. If the UI collapses them, users will over-trust agent prose.
- Warning signs:
  - The chat response is visually stronger than validation errors or missing-prerequisite warnings.
  - The UI does not show which cluster and WEKA signals were actually used.
  - Users can click apply from a conversational response before server-side validation completes.
- Prevention strategy:
  - Model explicit plan states in the backend and surface them in the UI.
  - Block apply until validation passes and the user reviews the normalized YAML preview.
  - Show stale/incomplete signal warnings alongside the plan, not hidden in logs.
- Phase: Phase 3.

## 8. Pilot blueprint rules stay implicit instead of becoming reusable planner logic

- Why this matters: The PRD calls for a pilot blueprint family first. If fit logic lives only in prompts or ad hoc conditionals, expansion will become inconsistent and brittle.
- Warning signs:
  - OpenFold-specific sizing assumptions exist only in prompt text or agent examples.
  - Adding a second blueprint requires copy-pasting prompt instructions instead of extending a rule layer.
  - Validation and planner reasoning disagree for the same blueprint family.
- Prevention strategy:
  - Encode pilot blueprint prerequisites and fit rules in backend data structures or validators, not just prompts.
  - Keep blueprint metadata, required inputs, and fit heuristics in versioned files the backend can test.
  - Use the pilot to define the extension pattern for future blueprint families.
- Phase: Phase 4, with supporting groundwork in Phase 1 and Phase 2.

## 9. Maintainer authoring mode produces repo-incompatible drafts

- Why this matters: The repo already has conventions around namespaces, component ordering, readiness checks, and values structure. Draft YAML that ignores those conventions will create review churn instead of saving time.
- Warning signs:
  - Generated blueprint files omit `dependsOn`, readiness checks, or repo-specific templating placeholders.
  - Drafts work syntactically but do not match the patterns used by existing blueprints and operator behavior.
  - Maintainers have to hand-rewrite most generated files before review.
- Prevention strategy:
  - Treat authoring mode as a separate validated output profile, not the same contract as install planning.
  - Seed generation with repo examples and enforce lint/validation rules against repo conventions.
  - Export drafts with explicit TODO markers for unresolved maintainer decisions instead of inventing values.
- Phase: Phase 5.

## 10. Observability is too weak to debug planner-versus-runtime failures

- Why this matters: The existing codebase already has fragile deploy and reconcile paths with limited automated verification. NemoClaw adds another failure layer unless stages are separable in logs and tests.
- Warning signs:
  - Operators cannot tell whether failure came from NemoClaw, a tool call, validation, YAML normalization, GUI apply, or operator reconcile.
  - The same user report requires reading both GUI logs and operator logs without a shared correlation ID.
  - Regression fixes are made by prompt tweaking because no deterministic reproduction exists.
- Prevention strategy:
  - Log plan session ID, request ID, generated plan version, validation result, and applied CR name on every transition.
  - Add mocked tests that isolate tool failure, malformed agent output, validation rejection, and apply/reconcile failure.
  - Preserve the exact normalized plan and validation artifacts used for each apply attempt.
- Phase: Phase 6, with minimum correlation ID groundwork in Phase 2.

## 11. Hardening is deferred until after new blueprint families are added

- Why this matters: The current repo already has thin test coverage, fragile namespace behavior, and mixed execution paths. If expansion comes before hardening, the planner will multiply existing failure modes.
- Warning signs:
  - New blueprint families are being added while Phase 1 validation and Phase 2 tool contracts are still changing.
  - Team confidence depends on manual cluster testing for every planner change.
  - Roadmap items for tests and policy rules are phrased as "later cleanup."
- Prevention strategy:
  - Gate blueprint-family expansion on passing validation, mocked planner tests, and at least one end-to-end pilot workflow.
  - Keep Phase 6 work mandatory for rollout, not optional polish.
  - Refuse new family onboarding until the extension contract is stable and tested.
- Phase: Phase 6.
