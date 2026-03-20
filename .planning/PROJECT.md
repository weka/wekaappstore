# NemoClaw Agent Planning For WEKA App Store

## What This Is

This project extends the existing WEKA App Store codebase with a NemoClaw-driven planning workflow that helps users install blueprints through a conversational interface instead of fixed forms alone. It builds on the current FastAPI web app, Kubernetes operator, and `WekaAppStore` CRD by adding agent-assisted plan generation, cluster fit assessment, and review-before-apply for blueprint installs.

The primary users are platform users and cluster admins deploying blueprints into Kubernetes clusters, plus maintainers who need faster ways to author operator-compatible blueprint definitions.

## Core Value

Users can describe what they want to deploy, and the system turns that into a safe, validated WEKA App Store installation plan that actually fits the target cluster before anything is applied.

## Requirements

### Validated

- ✓ Users can browse and deploy blueprint content through the existing WEKA App Store web UI and backend apply flow. — existing
- ✓ The system can create and reconcile `WekaAppStore` resources through the current operator-driven execution model. — existing
- ✓ The operator supports multi-component `appStack` deployments with Helm charts, raw manifests, dependencies, target namespaces, and readiness checks. — existing
- ✓ Blueprint content can be sourced externally and applied as rendered YAML strings or file-backed manifests. — existing

### Active

- [ ] Add a NemoClaw-backed chat workflow in the web app for conversational blueprint planning and follow-up questions.
- [ ] Add bounded backend tools for Kubernetes inspection, including GPU count, GPU type, GPU memory, CPU availability, and RAM availability.
- [ ] Add bounded backend tools for WEKA API inspection, including storage capacity, available space, and existing filesystems.
- [ ] Generate structured installation plans and operator-compatible `WekaAppStore` YAML from NemoClaw output.
- [ ] Validate cluster fit, storage fit, and operator contract compatibility before allowing apply.
- [ ] Preserve explicit user review and approval before the existing backend apply path is invoked.
- [ ] Support maintainer-facing draft blueprint generation compatible with repo conventions and operator semantics.

### Out of Scope

- Autonomous unrestricted `kubectl` or `helm` execution by NemoClaw — v1 must keep mutation authority in the existing backend and operator paths.
- Replacing the `WekaAppStore` CRD or the current operator reconciliation contract — the new workflow must build on the current runtime model, not swap it out.
- Broad authentication redesign for the GUI — valuable, but not the scope of this planning initiative.
- General Kubernetes authoring outside the WEKA app store/operator model — this project is for bounded blueprint planning and installation.

## Context

The codebase is already a brownfield Kubernetes application bundle with a FastAPI and Jinja web UI in `app-store-gui/webapp/main.py`, a Kopf-based operator in `operator_module/main.py`, and a Helm chart in `weka-app-store-operator-chart/`. The current architecture already supports applying blueprint manifests that create `WekaAppStore` custom resources and reconciling those resources into Helm releases and Kubernetes manifests.

Recent codebase mapping identified several constraints that matter to this effort: blueprint apply logic is duplicated in the GUI, namespace handling is fragile, runtime validation is mostly imperative, and automated test coverage is currently limited. Those issues make it important to introduce structured plan validation before increasing the complexity of the apply flow.

The new PRD defines a NemoClaw integration that adds a chat-first UX in the web app and bounded inspection tools so the system can reason about GPU inventory, GPU memory, CPU, RAM, and WEKA storage capacity before proposing blueprint installs. This is especially important for large LLM deployments and multi-blueprint coexistence on the same Kubernetes cluster.

## Constraints

- **Tech stack**: Must fit into the existing FastAPI GUI, Kopf operator, Helm chart, and `WekaAppStore` CRD architecture — avoids introducing a second execution model.
- **Compatibility**: No CRD-breaking changes in v1 — current operator semantics must continue to work for generated app stacks.
- **Safety**: NemoClaw must use bounded, auditable tools only — unrestricted cluster mutation is explicitly out of scope.
- **Data dependency**: Cluster-fit decisions depend on trustworthy Kubernetes resource signals and WEKA API responses — capacity logic is only as good as those inputs.
- **Quality**: Existing test coverage is thin in key deploy paths — validation and tests need to be added early to reduce regression risk.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on the existing WEKA App Store repo as a brownfield project | The current GUI, operator, and CRD already implement the safe apply path and runtime contract the new workflow needs | — Pending |
| Use NemoClaw as a planning layer, not the execution authority | Keeps cluster mutation in deterministic backend/operator code paths and reduces safety risk | — Pending |
| Make the user interaction chat-first in the web app | The PRD requires iterative planning, follow-up questions, and explanation of cluster-fit constraints | — Pending |
| Treat GPU, CPU, RAM, and WEKA storage inspection as first-class bounded tools | Capacity assessment is central to deciding whether blueprint installs and large models can fit the cluster | — Pending |
| Preserve explicit human review before apply | This aligns with the existing safety posture and prevents silent cluster mutations from agent output | — Pending |

---
*Last updated: 2026-03-20 after initialization*
