# NemoClaw Planning Architecture

## Recommendation

Integrate NemoClaw as a new planning layer inside the FastAPI GUI/backend boundary, not inside the CRD or operator. The planning layer should own conversation state, bounded cluster/WEKA inspection, structured plan generation, and pre-apply validation. The existing `WekaAppStore` CRD and Kopf operator should remain the execution contract: a `WekaAppStore` resource is created only after the user approves a validated plan.

This keeps mutation authority in deterministic backend and operator code, matches the PRD, and avoids making the operator responsible for conversational or partially-complete intent.

## Target Component Boundaries

### 1. Chat/UI Layer
- Location: existing FastAPI/Jinja GUI, with new agent-mode routes and templates.
- Responsibility: capture user intent, display plan/validation/YAML preview, collect follow-up answers, and require explicit approval.
- Must not: inspect Kubernetes directly from the browser or apply generated YAML without backend validation.

### 2. Planning Session Service
- Location: new backend module extracted from `app-store-gui/webapp/main.py`.
- Responsibility: manage planning sessions, turn history, correlation IDs, selected blueprint family, unresolved questions, and final plan state.
- Output contract: structured `InstallPlan` JSON plus validation results and rendered `WekaAppStore` YAML preview.
- Persistence: start with backend-managed session storage; do not persist draft plans as CRs in v1.

### 3. NemoClaw Orchestrator Adapter
- Location: new backend service boundary called by the planning session service.
- Responsibility: build NemoClaw prompts, expose only bounded tools, normalize responses, reject malformed output, and map follow-up questions back into the session.
- Must return structured data first. YAML generation should be derived server-side from the structured plan, with model-generated YAML treated as advisory only.

### 4. Inspection Adapter Layer
- Location: new backend modules wrapping current Kubernetes and future WEKA API reads.
- Responsibility: provide deterministic tool results for:
  - cluster summary
  - namespace list
  - storage classes
  - GPU inventory and GPU memory summary
  - CPU and RAM availability summary
  - WEKA capacity and filesystem summary
  - blueprint catalog and blueprint schema metadata
- Design rule: NemoClaw calls these adapters through narrow functions, not through unrestricted `kubectl`, `helm`, or raw Kubernetes clients.

### 5. Plan Validator / Translator
- Location: new backend module shared by chat preview and final apply.
- Responsibility:
  - validate plan shape and required fields
  - validate cluster-fit and storage-fit claims against tool outputs
  - enforce repo/operator contract rules for `appStack.components[]`
  - translate structured plan into canonical `WekaAppStore` YAML
- This should become the single entry point before both preview and apply.

### 6. Existing Apply Gateway
- Location: existing GUI apply path, but refactored behind one helper.
- Responsibility: submit approved canonical YAML through the existing manifest apply flow that creates `WekaAppStore` resources.
- Required change: unify `apply_blueprint_with_namespace()` and `apply_blueprint_content_with_namespace()` behind one internal manifest application service before NemoClaw work adds a third path.

### 7. Existing Runtime Layer
- Components: `WekaAppStore` CRD and `operator_module/main.py`.
- Responsibility: unchanged runtime reconciliation of approved resources into Helm releases and Kubernetes manifests.
- V1 stance: no new execution authority in NemoClaw and no major CRD redesign.

## CRD And Operator Fit

### `WekaAppStore` CRD
- Keep `WekaAppStore` as the runtime submission format.
- Use the existing `spec.appStack.components[]` contract as the canonical execution target.
- Avoid storing draft conversational state in the CRD. Drafts are not declarative desired runtime state and would create noisy or invalid reconcile attempts.
- Minimal v1 schema change is optional, not required. If provenance must be preserved, prefer metadata labels/annotations added at submission time rather than new planning fields in `spec`.

### Operator
- Keep the operator focused on reconcile only.
- Do not add NemoClaw calls, cluster-fit logic, or follow-up question handling to the operator.
- The operator should continue consuming validated `WekaAppStore` specs and reporting `appStackPhase` and `componentStatus`.
- If later needed, add only execution-adjacent improvements such as clearer status messages or provenance annotations surfaced in status.

## End-To-End Data Flow

1. User starts an agent-mode install session in the GUI.
2. FastAPI creates a planning session with a correlation ID and initial user intent.
3. Planning session service calls bounded inspection adapters for cluster, storage, blueprint, and WEKA facts needed for the first reasoning pass.
4. NemoClaw receives:
   - user intent
   - bounded tool outputs
   - blueprint catalog/schema metadata
   - operator/CRD constraints
5. NemoClaw returns either:
   - follow-up questions, or
   - a structured `InstallPlan` draft with fit rationale.
6. Plan validator checks:
   - schema completeness
   - allowed blueprint/operator constructs
   - namespace and component rules
   - fit claims against current inspection data
7. Backend translates the validated plan into canonical `WekaAppStore` YAML for preview.
8. GUI shows:
   - conversation transcript
   - structured plan summary
   - validation results
   - YAML preview
9. On approval, backend submits the canonical YAML through the existing apply gateway.
10. Kubernetes stores the `WekaAppStore` resource and the operator reconciles it as today.
11. GUI polls existing CR/operator status endpoints for runtime progress.

## Suggested Internal Contracts

### Planning Session
- `session_id`
- `correlation_id`
- `user_intent`
- `conversation_turns[]`
- `inspection_snapshot`
- `draft_plan`
- `validation_result`
- `rendered_wekaappstore_yaml`
- `approval_state`

### Structured Plan
- `blueprint_family`
- `target_namespace`
- `components[]`
- `values_overrides`
- `dependencies`
- `fit_assessment`
- `weka_storage_assessment`
- `prerequisites`
- `unresolved_questions`
- `reasoning_summary`

### Validation Result
- `plan_valid`
- `operator_contract_valid`
- `cluster_fit_valid`
- `storage_fit_valid`
- `errors[]`
- `warnings[]`

## Build Order For Roadmap

1. Extract shared backend services before adding NemoClaw.
   The current GUI monolith and duplicated apply paths are the biggest integration hazard. Create reusable modules for manifest apply, blueprint catalog access, and cluster inspection first.

2. Add deterministic inspection adapters and validation contracts.
   Build the bounded Kubernetes and WEKA inspection layer plus a plan validator before integrating the model. This creates testable interfaces and prevents prompt work from driving backend shape.

3. Introduce structured plan translation to canonical `WekaAppStore` YAML.
   Server-side translation must exist before chat UX goes live, otherwise YAML generation semantics will drift from operator reality.

4. Add the NemoClaw adapter and mocked planning loop.
   Start with mocked/model-stubbed responses, follow-up handling, and failure paths. Prove the contract before full model integration.

5. Add the chat UI and approval flow.
   Wire the frontend only after the backend can produce deterministic plan, validation, and preview payloads.

6. Add apply handoff and runtime observability.
   Reuse the existing apply path, add correlation IDs to plan/apply logs, and connect plan sessions to CR submission events.

7. Expand to maintainer blueprint-authoring mode after install planning works.
   Reuse the same planner, validator, and YAML translator, but output draft files instead of applying.

## Risks From Current Code Fragility

### GUI Monolith Risk
- `app-store-gui/webapp/main.py` is already ~2,176 lines and mixes routing, cluster IO, templating, and apply logic.
- Adding NemoClaw directly into this file will increase coupling and make roadmap phases hard to verify.
- Implication: service extraction is not optional early work.

### Duplicated Apply Logic Risk
- File-based and string-based blueprint apply flows already duplicate namespace mutation and CR submission behavior.
- NemoClaw would introduce a third manifest source unless these paths are unified first.
- Implication: refactor apply into one canonical backend path before any planning-generated YAML is allowed to submit.

### Weak Automated Verification Risk
- Current concerns already identify thin coverage across GUI deploy paths, operator reconcile logic, and chart rendering.
- NemoClaw integration increases the number of edge cases without changing the operator contract.
- Implication: roadmap should put contract tests around plan validation, YAML translation, and apply handoff before broad feature expansion.

### Runtime/Inspection Cost Risk
- Existing GUI status endpoints already do live cluster-wide reads.
- Reusing that style for agent tools will create slow, expensive planning turns and inconsistent answers.
- Implication: inspection adapters need narrow queries, caching, and stable response schemas.

### Operator Partial-Failure Risk
- The operator stops `appStack` execution on first failure and leaves partial installs behind.
- NemoClaw can improve preflight fit, but it does not remove reconcile-time partial failure.
- Implication: roadmap should keep operator-side status clarity and partial-failure handling visible as follow-on hardening work.

## Decision Summary

- Put NemoClaw between the GUI and the existing apply path, not in the operator.
- Keep planning state outside the CRD until approval.
- Make structured plan JSON the primary contract; derive `WekaAppStore` YAML server-side.
- Treat bounded inspection and validation services as prerequisites, not follow-up polish.
- Sequence roadmap work around backend extraction and validation first, then model integration, then UI, then maintainer authoring.
