# Phase 2 Research: Cluster And WEKA Inspection Signals

**Date:** 2026-03-20
**Status:** Complete

## Objective

Research what is needed to plan Phase 2 well: add bounded Kubernetes and WEKA inspection signals, carry freshness and confidence into fit findings, and keep all NemoClaw-facing inspection access read-only, auditable, and safely scoped.

## Key Findings

### 1. The current cluster inspection seam already exists, but it is too shallow and too monolithic

- `get_cluster_status()` in [/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/main.py) already gathers ready-node counts, allocatable CPU, GPU device totals, running pod requests, storage classes, and some app-store/operator presence checks.
- The function currently returns one flat dict with partial `None` behavior and only one coarse failure surface. That is enough for the settings UI, but not enough for planner-grade fit reasoning.
- Phase 2 should extract bounded inspection services from `main.py` instead of adding more imperative logic inline. The existing function is the proof that the repo already has permission and library patterns for read-only Kubernetes inspection.

### 2. GPU fit requires new modeling beyond the current total-device count

- The current inspection seam only exposes total GPU device counts and free counts derived from pod requests.
- The phase context requires GPU type and GPU memory by GPU model, with fail-closed behavior when only partial GPU facts are visible.
- Kubernetes-visible sources will likely need to come from node labels and allocatable/capacity fields rather than scheduler placement simulation. Phase 2 can stay within scope by reporting bounded inventory and domain completeness rather than solving node-level placement.

### 3. CPU and RAM availability need planner-facing snapshot semantics, not UI-only totals

- Current code already computes requested-vs-free CPU using pod requests on ready nodes, but it does not expose RAM availability in the same planner-facing shape.
- Phase 2 should normalize CPU, RAM, GPU, namespace, storage-class, and WEKA findings into a single inspection snapshot with domain statuses, timestamps, and explicit completeness signals.
- The roadmap and context make CPU and RAM blockers, not advisory warnings, so the resulting fit model must fail closed when those domains are missing or stale.

### 4. WEKA inspection should be read-only and probably Kubernetes-first for the first implementation

- There is no existing WEKA API client layer in the repo.
- The repo already contains WEKA operator CRDs and references to `wekaclusters.weka.weka.io`, including status fields for filesystem capacity in the checked-in CRD definitions under [/Users/christopherjenkins/git/wekaappstore/weka-csi-config/weka-operator/crds](/Users/christopherjenkins/git/wekaappstore/weka-csi-config/weka-operator/crds).
- The lowest-risk Phase 2 plan is to inspect WEKA state through bounded Kubernetes reads first:
  - operator/CRD presence from `/cluster-info`
  - relevant `WekaCluster` custom resources and status fields
  - filesystem inventory and capacity data if surfaced in CR status
- If CR status is incomplete, the system should return partial-data signals and block storage-dependent plans instead of expanding into broad WEKA admin capabilities.

### 5. The current structured plan contract is too narrow for Phase 2

- The typed planning contract in [/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/planning/models.py](/Users/christopherjenkins/git/wekaappstore/app-store-gui/webapp/planning/models.py) currently models `fit_findings` as:
  - `status`
  - `notes`
- That shape is sufficient for Phase 1 placeholder fit reasoning, but not for Phase 2 requirement coverage.
- Phase 2 planning should extend the contract so fit findings can carry:
  - per-domain results for GPU, CPU, RAM, namespaces, storage classes, and WEKA
  - freshness metadata
  - confidence/completeness flags
  - blocking reasons
  - optional correlation or provenance identifiers for auditability

### 6. Failure taxonomy and auditability should be introduced as explicit structures, not log-message conventions

- Requirement `SAFE-02` requires failures to be classified by stage: inspection, validation, YAML generation, or apply handoff.
- Requirement `SAFE-01` requires stable correlation identifiers across planning sessions, validation, and apply handoff.
- Phase 2 should introduce shared typed structures for:
  - correlation IDs propagated through request handling and planning services
  - stage-tagged failures or diagnostics
  - audit records for bounded tool calls or inspection reads
- This should be implemented as lightweight backend utilities and response fields, not a full observability platform.

### 7. The safest NemoClaw tool surface is a narrow internal adapter over inspection services

- The roadmap calls for bounded, auditable, read-only agent-callable tools.
- Phase 2 should avoid exposing raw Kubernetes clients or shell commands to any agent surface.
- Instead, create narrow service functions that accept only supported inspection intents and return typed snapshots or typed failure objects. That preserves a bounded contract and keeps audit logging centralized.

## Recommended Implementation Shape

### Backend modules

Recommended new backend units inside `app-store-gui/webapp/`:
- `inspection/models.py` for typed inspection snapshots and domain statuses
- `inspection/cluster.py` for Kubernetes namespace, storage-class, CPU, RAM, and GPU reads
- `inspection/weka.py` for bounded WEKA-related inspection via Kubernetes-visible resources
- `inspection/tools.py` or `planning/inspection_tools.py` for NemoClaw-safe read-only tool wrappers
- `planning/fit_signals.py` or a `planning/models.py` extension for planner-facing fit findings derived from inspection snapshots
- `planning/audit.py` or similar for correlation IDs and stage-tagged diagnostics

### Integration seams

- `/cluster-info` should remain a simple installation-capability endpoint.
- `get_cluster_status()` should be refactored to use shared inspection services rather than remain the only source of truth.
- Structured-plan validation and compilation from Phase 1 should stay intact; Phase 2 should enrich inputs to that flow rather than create a second planner path.

### Data-shape recommendations

Inspection snapshot should include:
- snapshot timestamp
- correlation ID
- domain objects for namespaces, storage classes, CPU, RAM, GPU, and WEKA
- per-domain status: `complete`, `partial`, `unavailable`, or similar
- freshness/completeness metadata and blocking reasons

Fit findings should include:
- overall fit decision
- per-domain findings
- blockers vs non-blocking notes
- provenance linking findings to the snapshot timestamp and correlation ID

### Boundary rules

Phase 2 should do:
- read-only Kubernetes and WEKA-adjacent inspection
- typed fit-signal generation
- bounded tool wrappers and auditable logging
- fail-closed classification when required facts are missing

Phase 2 should not do:
- chat UX
- review/apply screens
- broad WEKA administration or mutation
- unrestricted model tools
- multi-blueprint coexistence scoring beyond the explicitly required bundled-demand inputs that later phases will own more fully

## Testing Implications

Highest-value tests for this phase:
- unit tests for snapshot models, completeness logic, freshness logic, and fail-closed fit derivation
- mocked Kubernetes inspection tests for namespace, storage class, CPU, RAM, and GPU aggregation
- mocked WEKA CR inspection tests for capacity and filesystem inventory extraction
- integration tests proving planner-facing fit findings include per-domain status, freshness, and blockers
- request-level tests proving correlation IDs and stage-classified errors are returned consistently
- bounded-tool tests proving inspection helpers remain read-only and do not expose arbitrary command execution

Existing pytest infrastructure from Phase 1 is sufficient. Phase 2 does not need a new framework, but it does need new fixtures and mocks for Kubernetes and WEKA inspection results.

## Validation Architecture

Nyquist-relevant validation strategy for this phase:

- Keep the fastest feedback loop around mocked inspection services and typed fit-finding derivation.
- Ensure every plan has at least one targeted pytest command that runs in seconds and validates the exact requirement slice it changes.
- Add integration-style tests around the FastAPI seams that expose inspection snapshots or fit findings, including correlation ID propagation and stage-tagged errors.
- Verify read-only bounded-tool behavior with mocked service adapters rather than live cluster dependencies.
- Treat live-cluster inspection as manual-only verification for this phase; automated checks should stay deterministic and mocked.

## Open Planning Risks

- `phase_req_ids` were not populated by the GSD init tool, so requirement coverage must be enforced manually from `ROADMAP.md` and `REQUIREMENTS.md`.
- GPU model and GPU memory data may not be uniformly available in every Kubernetes environment, so plan tasks must define how incomplete GPU metadata is surfaced and blocked.
- WEKA filesystem inventory availability may depend on operator CR status richness; if fields are absent, the implementation must degrade to explicit partial-data blockers instead of guessing.
- `main.py` is already large, so refactoring shared inspection logic into focused modules is important to keep Phase 2 execution tractable.

## Planning Recommendations

1. Start by extending the typed contract and tests for inspection snapshots, fit findings, and failure taxonomy.
2. Extract Kubernetes inspection logic from `main.py` into bounded services for namespaces, storage classes, CPU, RAM, and GPU inventory.
3. Add WEKA inspection through read-only Kubernetes-visible seams and expose partial-data blockers when status is incomplete.
4. Add correlation IDs, stage-classified failures, and bounded audit logging at the service and request seams.
5. Wire planner-facing fit findings and API responses through the new inspection services without changing the Phase 1 YAML/apply contract.

---
*Phase 2 research completed: 2026-03-20*
