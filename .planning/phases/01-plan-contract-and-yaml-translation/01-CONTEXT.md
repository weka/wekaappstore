# Phase 1: Plan Contract And YAML Translation - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 defines the deterministic structured-plan contract, validation behavior, and canonical `WekaAppStore` YAML translation path that NemoClaw-generated plans must pass through before anything can reach the existing apply and operator runtime path. This phase is not about chat UX, cluster and WEKA inspection tooling, or broader agent orchestration behavior beyond what is necessary to define and validate the contract.

</domain>

<decisions>
## Implementation Decisions

### Structured plan contract
- The backend requires a rich structured plan before YAML translation is attempted.
- The minimum required plan shape includes blueprint family, namespace strategy, components, prerequisites, fit findings, and a reasoning summary.
- The plan must name a supported blueprint family explicitly. Generic app patterns are not sufficient for Phase 1 translation.
- Install plans must contain concrete resolved values or explicit blockers. TODO-style placeholders are not allowed in install-plan YAML translation.

### Unresolved-question handling
- The backend defines what counts as install-critical missing information.
- NemoClaw may flag unresolved questions, but it does not decide what is blocking.
- If unresolved install-critical questions remain, translation and apply are blocked.

### Validation behavior
- Any structured plan that violates the `WekaAppStore` CRD or operator contract hard-fails validation.
- Safe, inferable missing data such as default namespace or release name may be normalized by the backend, but the normalization must be surfaced as a warning.
- Backend validation is authoritative when planner reasoning and validator findings disagree.
- Plans with warnings but no hard blockers may still produce a YAML preview.

### YAML normalization policy
- The compiler should produce stable canonical output for equivalent valid plans.
- Explicit valid user or planner intent, such as component names and namespace choices, should be preserved rather than rewritten to repo defaults.
- Optional fields should be injected only when they matter to runtime behavior, validation clarity, or deterministic output.
- The YAML preview target for Phase 1 is a single canonical `WekaAppStore` resource.

### Claude's Discretion
- Exact schema type names and module boundaries for the plan validator and compiler
- Exact YAML field ordering as long as output is stable and canonical
- Exact warning payload format as long as it clearly surfaces normalization decisions

</decisions>

<specifics>
## Specific Ideas

- Rich structured plan first, YAML second
- Backend rule, planner flags unresolved questions
- Hard fail contract breaks, warn on safe normalization
- Single canonical `WekaAppStore` YAML preview artifact

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `apply_blueprint_with_namespace()` in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py): existing file-backed YAML apply path with namespace and CR handling logic that should inform the shared apply gateway.
- `apply_blueprint_content_with_namespace()` in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py): existing rendered-content apply path that duplicates much of the file-backed behavior and is a strong candidate for consolidation.
- `handle_appstack_deployment()` in [main.py](/Users/christopherjenkins/git/wekaappstore/operator_module/main.py): authoritative runtime behavior for `appStack.components[]`, dependency handling, namespace resolution, Helm versus raw manifest execution, and readiness checks.
- `weka-app-store-operator-chart/templates/crd.yaml`: canonical schema reference for the `WekaAppStore` contract the Phase 1 validator and compiler must target.

### Established Patterns
- The repo currently favors direct Python helpers and `Dict[str, Any]` payloads over dedicated typed models, so Phase 1 can introduce typed plan and validation models as a deliberate tightening of the contract.
- The GUI and operator both use broad exception handling with logging rather than deep custom exception hierarchies, so validation errors should be explicit and operationally clear.
- Existing apply behavior already mutates namespaces and annotations during apply, which means Phase 1 should define a stricter and more predictable normalization policy before adding planner-generated YAML.

### Integration Points
- New Phase 1 work should connect first to the existing FastAPI backend in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py), because that is where current blueprint application and YAML handling already live.
- The compiled canonical YAML must remain compatible with the `WekaAppStore` runtime path consumed by [main.py](/Users/christopherjenkins/git/wekaappstore/operator_module/main.py).
- The shared apply gateway should become the single handoff point for file-backed, rendered, and planner-generated YAML.

</code_context>

<deferred>
## Deferred Ideas

- Chat UX details and session experience belong to Phase 3.
- Kubernetes and WEKA inspection tooling belong to Phase 2.
- Multi-blueprint coexistence enforcement belongs to Phase 4.

</deferred>

---
*Phase: 01-plan-contract-and-yaml-translation*
*Context gathered: 2026-03-20*
