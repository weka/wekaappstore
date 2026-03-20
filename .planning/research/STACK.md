# Stack Research: NemoClaw Planning Layer

**Date:** 2026-03-20
**Scope:** Standard 2026 stack for adding NemoClaw-driven conversational planning, bounded cluster inspection, and capacity-aware blueprint installation to this existing WEKA App Store codebase.

## Executive Recommendation

Use a **Python-native planning layer inside the existing FastAPI service**:

- **Model/runtime edge:** NVIDIA NIM behind an OpenAI-compatible API
- **Agent orchestration:** NVIDIA NeMo Agent Toolkit (`nvidia-nat`)
- **Safety/policy rails:** NVIDIA NeMo Guardrails
- **Typed contracts:** Pydantic v2 models for tool IO, plan IO, and YAML compilation
- **Conversation transport:** FastAPI endpoints plus SSE streaming
- **Session persistence:** PostgreSQL + SQLAlchemy 2 + Alembic
- **Cluster inspection:** Kubernetes Python client + `kr8s`, Metrics API (`metrics.k8s.io`), Node allocatable data, NVIDIA GPU Operator/NFD labels, and DCGM exporter when present
- **WEKA inspection:** Thin typed WEKA REST client in the FastAPI app
- **Execution boundary:** Keep apply/mutation in the existing backend and `WekaAppStore` operator path

This is the best fit for this repo because the current architecture is already:

- FastAPI + Jinja + browser `fetch`/`EventSource`
- Python-only in both GUI and operator
- CRD/operator-driven for actual cluster mutation
- light on durable backend infrastructure today

## Standard Stack

| Layer | Use | Why it fits this repo | Confidence |
|---|---|---|---|
| LLM serving | **NVIDIA NIM** via `/v1/chat/completions` or `/v1/responses` | Keeps NemoClaw/NVIDIA alignment while preserving a standard API edge and future swapability | High |
| Agent runtime | **NeMo Agent Toolkit (`nvidia-nat`)** | NVIDIA-native, framework-agnostic, MCP-capable, and avoids bolting a second agent stack onto an existing Python app | Medium-High |
| Guardrails | **NeMo Guardrails** | Topic/tool/output rails belong in policy code, not prompts; strong fit for bounded inspection and fail-closed planning | High |
| Data contracts | **Pydantic v2** | Best fit for this Python brownfield; use one schema system across API, tools, validation, and YAML compiler | High |
| API/UI transport | **FastAPI + SSE** | Existing app already uses SSE and server-rendered pages; lowest-risk way to add chat streaming and status updates | High |
| Session store | **PostgreSQL + SQLAlchemy 2 + Alembic** | Durable multi-turn chat, approval checkpoints, audit logs, and resumable planning need storage not present today | Medium-High |
| Cluster inspection | **Kubernetes Python client + `kr8s` + Metrics API + node labels/allocatable** | Reuses repo libraries and avoids shelling out from the planner | High |
| GPU inspection | **NVIDIA GPU Operator/NFD labels + DCGM exporter when available** | Needed for GPU model and memory-aware fit checks; Metrics API alone is not enough | High |
| WEKA inspection | **Typed internal WEKA REST adapter** | Keep WEKA calls deterministic and auditable inside the existing backend | High |
| Plan compilation | **Deterministic compiler from structured plan -> `WekaAppStore` spec/YAML** | Preserves operator contract and keeps YAML generation out of the model trust boundary | High |
| Async/background work | **FastAPI background tasks first; introduce Temporal only if v2 durability demands it** | Separate workflow infra is premature for this repo today | Medium |
| Observability | **OpenTelemetry traces/log correlation, export to Langfuse/Phoenix/OTel backend if desired** | Gives request-to-tool-to-plan auditability without committing the repo to a single vendor UI | Medium |

## Prescriptive Architecture

### 1. Keep one execution model

Use NemoClaw as a planner only. Do **not** let it apply manifests, run `helm`, or run `kubectl`.

Recommended boundary:

1. User chats with FastAPI
2. FastAPI planning service calls Nemo/NAT tools
3. NAT returns a **structured install plan**
4. Backend validates and compiles the plan into `WekaAppStore`
5. User reviews plan + YAML
6. Existing backend apply path submits the CR
7. Existing Kopf operator reconciles it

Why this fits the repo:

- The current product already has a safe CRD/operator execution path
- The PRD explicitly keeps mutation authority out of NemoClaw
- The operator already knows how to reconcile `appStack.components[]`

Confidence: High

### 2. Add a planning subsystem inside the FastAPI app, not a new microservice

Create a bounded planning package in the GUI service:

- `app-store-gui/webapp/planning/api.py`
- `app-store-gui/webapp/planning/service.py`
- `app-store-gui/webapp/planning/models.py`
- `app-store-gui/webapp/planning/tools/`
- `app-store-gui/webapp/planning/compiler.py`
- `app-store-gui/webapp/planning/store.py`

Service boundaries:

- **Conversation API**
  - create session
  - append user turn
  - stream assistant/tool events over SSE
  - approve/reject plan
- **Planner service**
  - assembles context
  - invokes NAT workflow
  - records tool results
  - emits draft plan
- **Tool registry**
  - cluster summary
  - namespace/storage class inventory
  - CPU/RAM capacity
  - GPU inventory/memory
  - WEKA capacity/filesystems
  - blueprint catalog/schema lookup
  - plan validation
- **Compiler**
  - structured plan -> repo-compatible `WekaAppStore` dict
  - YAML preview rendering
- **Store**
  - chat turns
  - tool snapshots
  - plan revisions
  - approval state

Why this fits the repo:

- avoids cross-service auth and deployment complexity
- keeps planning close to existing blueprint/application logic
- preserves Helm chart simplicity for v1

Confidence: High

### 3. Use NVIDIA NIM as the model edge, but code to the standard API surface

Recommendation:

- Run NemoClaw through **NVIDIA NIM** if available
- Wrap it behind an internal `PlannerLLMClient`
- Standardize on OpenAI-compatible request/response envelopes

Why:

- NIM exposes `/v1/chat/completions` and `/v1/responses`
- this reduces lock-in to a vendor-specific SDK surface
- it keeps a future fallback path open if NemoClaw packaging changes

Important brownfield guidance:

- The repo should not leak NIM-specific payload shapes through route handlers
- Only the planning adapter should know which model is actually behind the endpoint

Confidence: High

### 4. Use NeMo Agent Toolkit, but keep the workflow simple

Recommendation:

- Use `nvidia-nat` for the planning loop
- Start with a **tool-calling or reasoning agent**
- Keep tools local/in-process in Python for v1
- Use MCP only if tool isolation becomes a real requirement later

Why:

- NAT is explicitly framework-agnostic and designed to sit beside an existing stack
- it supports MCP and observability, but does not force a full replatform
- this repo does not need a large multi-agent graph to answer bounded install questions

Do first:

- one planner agent
- one validation pass
- one compiler
- one approval gate

Do later only if needed:

- agent-to-agent delegation
- external MCP servers
- planner/router split

Confidence: Medium-High

### 5. Put safety in Guardrails and validators, not in prompts

Recommendation:

- Use **NeMo Guardrails** for:
  - allowed topic scope
  - tool allow-listing
  - unsafe action blocking
  - required approval gate language
  - explanation shaping for rejection reasons
- Pair that with deterministic Python validators

Validation layers:

1. tool input/output schemas
2. plan schema validation
3. repo/operator compatibility validation
4. cluster/storage fit validation
5. explicit human approval state

Why this fits the repo:

- existing validation is imperative and thin
- the PRD requires fail-closed behavior
- guardrails alone are not sufficient; they need deterministic validators behind them

Confidence: High

### 6. Keep the UI server-rendered and stream over SSE

Recommendation:

- Keep Jinja templates
- Add a dedicated chat page for agent planning
- Stream assistant/tool/progress events with SSE
- Use plain browser `fetch` + `EventSource`

Why this fits the repo:

- the current app already uses SSE endpoints and browser `EventSource`
- this avoids introducing React/Next/Vite build complexity into a repo that does not need it
- chat, plan preview, and approval are all straightforward in a server-rendered UI

Suggested UX pieces:

- left pane: conversation
- right pane: live cluster findings / plan draft / YAML preview
- explicit plan states: `draft`, `validated`, `blocked`, `approved`, `applied`

Confidence: High

### 7. Capacity inspection should be API-first and layered

Recommendation:

- Use **Kubernetes API + Metrics API + vendor telemetry**
- Do not rely on a single source for capacity

Use this inspection layering:

1. **Node allocatable and labels**
   - source: core Kubernetes APIs
   - use for schedulable ceilings and hardware classes
2. **CPU/RAM current usage**
   - source: `metrics.k8s.io`
   - use for near-real-time pressure
3. **GPU count and schedulable presence**
   - source: `nvidia.com/gpu` allocatable/limits model
4. **GPU model/memory**
   - source: NFD/vendor labels, GPU Operator integrations, DCGM exporter if installed
5. **WEKA capacity/filesystems**
   - source: WEKA API

Why:

- Metrics API only gives CPU and memory
- Kubernetes device plugins expose schedulable GPU resources, but not rich memory telemetry
- GPU type/memory-aware planning needs vendor labels and telemetry, not just pod spec math

Confidence: High

### 8. Model fit assessment should use a normalized capacity snapshot

Recommendation:

- Build a `CapacitySnapshot` model that combines:
  - cluster timestamp
  - namespaces examined
  - node inventory
  - CPU allocatable/usage/headroom
  - memory allocatable/usage/headroom
  - GPU count/model/memory/free estimate
  - WEKA total/available/filesystem inventory
  - inspection confidence flags

Then have the planner reason over the snapshot, not raw live tool calls everywhere.

Why this fits the repo:

- makes behavior testable
- gives auditable evidence for every rejection
- lets the compiler/validator reuse the same normalized view

Confidence: High

### 9. Persist sessions and approvals in PostgreSQL

Recommendation:

- Add PostgreSQL for:
  - planning sessions
  - chat turns
  - tool snapshots
  - plan revisions
  - approvals
  - audit events

Minimal tables:

- `planning_session`
- `planning_message`
- `planning_tool_snapshot`
- `planning_plan_revision`
- `planning_approval`
- `planning_event`

Why:

- this feature introduces conversational state and review-before-apply
- file or in-memory storage will become brittle immediately
- Postgres is standard, durable, and operationally predictable

Confidence: Medium-High

## What To Use For Each Concern

### Agent and Planning

- **Use:** `nvidia-nat`
- **Also use:** Pydantic models for every tool and every plan object
- **Pattern:** one bounded planner workflow, not a general multi-agent platform
- **Confidence:** Medium-High

### Conversation UI

- **Use:** FastAPI routes + SSE + Jinja templates + vanilla JS
- **Pattern:** optimistic local rendering, server as source of truth
- **Confidence:** High

### Cluster Inspection

- **Use:** Kubernetes Python client for typed API access
- **Keep:** `kr8s` where it already simplifies object traversal
- **Pattern:** read-only, bounded, timeout-controlled inspection adapters
- **Confidence:** High

### GPU Details

- **Use:** NVIDIA GPU Operator/NFD labels and DCGM exporter when present
- **Pattern:** degrade gracefully if only `nvidia.com/gpu` count is available
- **Confidence:** High

### Storage Inspection

- **Use:** WEKA REST adapter with explicit request/response models
- **Pattern:** one aggregated storage summary tool, not many tiny model-facing calls
- **Confidence:** High

### Validation and Compilation

- **Use:** deterministic Python compiler + Pydantic validation
- **Pattern:** model returns structured intent; backend owns final YAML
- **Confidence:** High

## What NOT To Use

### Do not add a second frontend stack

- **Do not use:** React/Next/Vite rewrite for v1
- **Why not:** The current app is server-rendered and already streams events. A SPA rewrite adds risk without solving the core planning problem.
- **Confidence:** High

### Do not let the model execute cluster mutations

- **Do not use:** model-triggered `kubectl`, `helm`, or shell tools
- **Why not:** Directly conflicts with the PRD and the repo’s safe operator-based execution model.
- **Confidence:** High

### Do not adopt LangChain/LlamaIndex/CrewAI as a primary new dependency

- **Do not use:** a second general-purpose orchestration framework on top of NAT
- **Why not:** NAT already gives the NVIDIA-native agent/runtime layer. Doubling orchestration frameworks increases complexity and debugging surface in a repo that is currently simple.
- **Confidence:** Medium-High

### Do not use Prometheus as the primary planning API

- **Do not use:** ad hoc PromQL as the main source of truth for fit checks
- **Why not:** Prometheus is useful for observability, but the planner needs bounded, typed, permissioned capacity snapshots. Kubernetes APIs plus WEKA APIs should be authoritative; Prometheus can be supplementary.
- **Confidence:** High

### Do not use a vector database for pilot blueprint selection

- **Do not use:** RAG-first retrieval over blueprint YAML for the initial release
- **Why not:** The blueprint catalog is structured and relatively bounded. Deterministic metadata extraction is safer and easier to validate than semantic retrieval in v1.
- **Confidence:** High

### Do not introduce Temporal in v1 unless long-running resumability becomes a proven problem

- **Do not use:** a workflow engine by default
- **Why not:** It adds another stateful control-plane component. This repo is not there yet operationally. Start with Postgres-backed session state and explicit approval transitions.
- **Confidence:** Medium

## Brownfield-Specific Rationale

These recommendations are grounded in the current repo shape:

- The GUI is already **FastAPI + Jinja + browser JS**, so SSE chat is a natural extension.
- The operator is already the **runtime authority**, so planning should stop at validated CR generation.
- The repo already depends on the **Kubernetes Python client** and `kr8s`, so bounded inspection should stay Python/API-first.
- The current codebase has **thin validation and limited tests**, so typed schemas and deterministic compilers matter more than sophisticated agent autonomy.
- The current deployment story is **Helm-based and compact**, so adding multiple new services should be avoided until proven necessary.

Confidence: High

## Implementation Pattern To Standardize On

Use this contract:

1. `UserIntent`
2. `CapacitySnapshot`
3. `BlueprintCandidate`
4. `InstallPlanDraft`
5. `ValidatedInstallPlan`
6. `CompiledWekaAppStoreSpec`
7. `RenderedYAMLPreview`

Planner responsibilities:

- interpret user goal
- ask follow-up questions only when required
- select blueprint family
- propose parameterization
- explain fit/non-fit

Backend responsibilities:

- gather tool data
- enforce schema and policy
- score fit deterministically
- compile YAML
- apply only after approval

Confidence: High

## Sources And Notes

Primary sources used for current-stack recommendations:

- NVIDIA NeMo Agent Toolkit overview: framework-agnostic runtime, MCP, observability, evaluation, UI  
  https://docs.nvidia.com/nemo/agent-toolkit/
- NVIDIA NeMo Guardrails: programmable guardrails, rail types, tracing/OpenTelemetry support  
  https://docs.nvidia.com/nemo/guardrails/latest/
- NVIDIA NIM for LLMs API reference: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`  
  https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
- FastAPI `StreamingResponse` docs for streaming server responses  
  https://fastapi.tiangolo.com/advanced/custom-response/
- Kubernetes resource metrics pipeline and Metrics API  
  https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/
- Kubernetes GPU scheduling, device plugins, and NFD-based labeling  
  https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- Node Feature Discovery feature labels  
  https://kubernetes-sigs.github.io/node-feature-discovery/master/usage/features.html
- NVIDIA GPU Operator getting started  
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
- NVIDIA DCGM exporter for GPU telemetry in Kubernetes  
  https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html

Inference note:

- As of **2026-03-20**, I found stable official NVIDIA documentation for **NeMo Agent Toolkit, Guardrails, and NIM**, but not a clear public standalone SDK/API reference specifically branded as **NemoClaw**. Because of that, the recommended stack treats NemoClaw as the planning/model layer and anchors implementation on the stable public NVIDIA components and API surfaces above.
