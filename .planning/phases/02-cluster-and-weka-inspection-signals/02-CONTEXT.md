# Phase 2: Cluster And WEKA Inspection Signals - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 2 adds the bounded, read-only inspection signals NemoClaw needs to assess whether a blueprint request can fit the target cluster before planning proceeds. This phase covers Kubernetes resource inspection for GPU, GPU memory, CPU, RAM, namespaces, and storage classes, plus WEKA inspection for storage capacity and existing filesystems. It does not cover chat UX, user approval UX, or changing the apply/runtime model established in Phase 1.

</domain>

<decisions>
## Implementation Decisions

### Fit assessment model
- GPU fit may use cluster-wide totals rather than strict per-node placement modeling in Phase 2.
- GPU memory must be exposed grouped by GPU type or model, such as A10 or L40, rather than only as one cluster-wide total.
- CPU and RAM insufficiency are hard blockers for planning rather than advisory-only warnings.
- When a user is planning multiple blueprints in one request flow, fit should be assessed against the bundled request rather than one blueprint at a time.

### WEKA inspection scope
- WEKA inspection should include both cluster-level storage context and filesystem inventory.
- The planner should receive total capacity and free capacity from the WEKA side, not only free space.
- Filesystem inventory should include filesystem name and capacity data.
- NemoClaw may recommend a filesystem candidate from the inspected inventory when planning a storage-dependent blueprint.

### Partial-data and freshness behavior
- If GPU count is visible but GPU type or GPU memory data is missing, GPU-dependent plans are blocked until those missing GPU facts are available.
- If WEKA inspection fails for a storage-dependent blueprint, planning is blocked until storage capacity and filesystem data are available.
- Inspection freshness may be scoped to a single planning session snapshot rather than requiring repeated refresh during the same session.
- Inspection outputs should report per-domain status for GPU, CPU/RAM, and WEKA instead of collapsing everything into one opaque confidence score.

### Scope guardrails
- Phase 2 is strictly read-only and bounded. It adds inspection signals, not cluster mutation or approval behavior.
- Phase 2 should enrich the plan and fit-signal inputs for later phases, but it should not introduce chat-session UX or review/apply screens.
- The existing backend/operator apply path remains authoritative. Inspection results inform planning only.

### Claude's Discretion
- Exact internal module boundaries for Kubernetes inspection helpers versus WEKA inspection helpers
- Exact status field names and freshness metadata shape as long as per-domain status is explicit
- Exact heuristics for deriving RAM, GPU type, and GPU memory from Kubernetes-visible signals

</decisions>

<specifics>
## Specific Ideas

- Cluster totals are acceptable for GPU fit, but GPU memory must still be broken down by GPU type.
- CPU and RAM shortages should block fit decisions.
- Multi-blueprint requests should be evaluated as one bundled resource demand.
- WEKA inspection should expose cluster storage context plus filesystem inventory with name, total capacity, and free capacity.
- NemoClaw may recommend a filesystem from the inspected WEKA options.
- Missing GPU specifics or missing WEKA storage facts should block affected plans.
- Per-domain inspection status matters more than one global confidence number.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `get_cluster_status()` in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py) already inspects ready nodes, allocatable CPU, GPU device counts, requested-vs-free CPU/GPU, storage classes, and some app-store/operator presence signals. This is the natural Phase 2 starting point.
- `/cluster-info` and `/cluster-status` in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py) already expose cluster capability data to the UI, which makes them likely integration seams for bounded inspection outputs.
- `infer_requirements_from_yaml()` and the blueprint fit checks in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py) already contain lightweight requirement reasoning that can inform later fit-signal integration.
- OpenFold-related UI flows in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py) and [blueprint_openfold.html](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/templates/blueprint_openfold.html) already expose WEKA filesystem name and requested storage capacity as user inputs, which are strong anchors for WEKA inspection requirements.

### Established Patterns
- The repo already uses direct Kubernetes Python client calls in `main.py` for cluster inspection; Phase 2 should extend that read-only pattern instead of introducing shell-based probes.
- The existing cluster inspection path returns partial data and `None` values on probe failures, so Phase 2 needs explicit domain-status rules rather than more ad hoc fallbacks.
- There is no existing WEKA API client layer in the repo, so Phase 2 will need to introduce one as a bounded external integration without broadening into generic WEKA administration.

### Integration Points
- Phase 2 should connect first to the FastAPI backend in [main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py), because this is where current cluster-status and capability inspection already live.
- Inspection outputs should become inputs to the structured planning flow added in Phase 1, not a separate execution path.
- WEKA inspection likely needs a new backend service seam parallel to the Phase 1 planning helpers, with results shaped for later planner consumption rather than direct apply behavior.

</code_context>

<deferred>
## Deferred Ideas

- Exact chat presentation of inspection results belongs to Phase 3.
- Review and approval UX for fit findings belongs to Phase 4.
- Maintainer-facing authoring that reuses these inspection signals belongs to Phase 5.

</deferred>

---
*Phase: 02-cluster-and-weka-inspection-signals*
*Context gathered: 2026-03-20*
