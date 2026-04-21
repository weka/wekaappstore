# Roadmap: OpenClaw MCP Tools For WEKA App Store

## Milestones

- ✅ **v2.0 OpenClaw MCP Tool Integration** - Phases 6-10 (shipped 2026-03-22)
- ✅ **v3.0 Live EKS Deployment** - Phases 11-13 (shipped 2026-04-21, rescoped from original v3.0 "...and Agent Testing")
- 🔜 **v3.1 E2E Chat Validation** - Phase 14 deferred here along with 4 known issues from v3.0 retry (see `.planning/v3.0-KNOWN-ISSUES.md`)
- 🚧 **v4.0 App Categories on Home Screen** - Phase 15 (in progress)

## Phases

### 🚧 v4.0 App Categories on Home Screen (In Progress)

**Milestone Goal:** Introduce three top-level app categories — AIDP, WARP, and Partner — as selectable filter cards on the home screen. Client-side filter of the existing catalog grid, URL hash sync for deep links, keyboard accessibility, and empty state for zero-match categories. Single-file change to `app-store-gui/webapp/templates/index.html` — no build step, no new dependencies.

**Phase structure rationale:** Three research steps map to three plans inside one phase. All three steps operate on the same single file, share the same IIFE scope, and are strictly sequential. No step is independently shippable as a user-visible milestone that justifies its own phase boundary. One phase (Phase 15), three plans, is the correct granularity.

- [ ] **Phase 15: App Categories Feature** - Add category data, extract AppShell, build Categories component with filter/hash/a11y

**Phase Details:**

### Phase 15: App Categories Feature
**Goal**: Users can filter the WEKA App Store catalog by category (NeuralMesh AIDP, WARP, Partner) using selectable cards above the grid, with URL deep-link support and keyboard accessibility — all delivered as a single-file change to `index.html`
**Depends on**: Phase 14 (v4.0 is a GUI milestone independent of v3.1 EKS work)
**Requirements**: CAT-01, CAT-02, CAT-03, FIL-01, FIL-02, FIL-03, VIS-01, VIS-02, URL-01, URL-02, URL-03, A11Y-01, A11Y-02, A11Y-03
**Success Criteria** (what must be TRUE):
  1. User sees three category cards — "AIDP", "WARP", "Partner" — rendered in that left-to-right order above the App Catalog grid on the home page
  2. User clicks "WARP" and the catalog grid filters in place to show exactly 4 blueprints; user clicks "WARP" again and the grid returns to all 5 blueprints
  3. User clicks "Partner" and sees the inline message "No apps in this category yet." (no blank grid, no error)
  4. User loads `/#category=warp` directly and arrives at the WARP-filtered view without a flash of the unfiltered state; pressing Back exactly once leaves the site
  5. User navigating by keyboard can Tab to each category card and toggle it with Enter or Space; screen reader announces each card as a pressed/unpressed button via `aria-pressed`
**Plans**: 3 plans

**Implementation notes (critical pitfalls encoded from PITFALLS.md):**
- JSX is forbidden — every element uses `h()` only. Grep check on new code: `<[A-Z]` must return zero matches before each plan is marked done.
- `component: 'a'` must NOT appear on category `CardActionArea` — category cards are toggle buttons, not links. DOM must show `<button aria-pressed="...">`.
- `history.replaceState` must never fire on mount — initialization reads hash only; writing on mount corrupts the back stack.
- Hash parser must use `startsWith('#category=')` to avoid collision with existing `#catalog` and `#planning-studio` scroll anchors.
- `ThemeProvider` must lift from `Catalog` to `AppShell` — both `Categories` and `Catalog` must render inside one shared `ThemeProvider`.

Plans:
- [ ] 15-01-PLAN.md — Data preparation: add `category` field to all 5 `items[]` entries; define `CATEGORIES` constant inside IIFE. Page renders pixel-identically. Requires Chris's sign-off on blueprint → category mapping (Open Question 1 from research SUMMARY.md). Covers: CAT-01.
- [ ] 15-02-PLAN.md — AppShell extraction + Catalog refactor: extract `ThemeProvider`+`CssBaseline` into new `AppShell`; convert `Catalog` to accept `items` prop; rename mount point to `app-root`; replace `render(h(Catalog))` with `render(h(AppShell))`. No `useState`, no filter, no new UI — page renders pixel-identically. Structural prerequisite for Plan 15-03.
- [ ] 15-03-PLAN.md — Categories component, filter state, hash sync, accessibility: write `Categories` and `EmptyState` components; add `useState` lazy initializer for `selectedCategory`; compute `filteredItems` and `counts` inline; add `useEffect` hash-write via `replaceState`; wire all pitfall mitigations. This is the only plan that introduces user-visible behavior. Covers: CAT-02, CAT-03, FIL-01, FIL-02, FIL-03, VIS-01, VIS-02, URL-01, URL-02, URL-03, A11Y-01, A11Y-02, A11Y-03.

---

<details>
<summary>✅ v2.0 OpenClaw MCP Tool Integration (Phases 6-10) - SHIPPED 2026-03-22</summary>

8-tool MCP server shipped with 103 tests, SKILL.md, mock agent harness, Dockerfile, CI/CD, and deprecated v1.0 code removed.

See MILESTONES.md for full v2.0 summary.

</details>

### ✅ v3.0 Live EKS Deployment (Shipped 2026-04-21, rescoped)

**Milestone Goal (rescoped):** Deploy OpenClaw/NemoClaw and the MCP server to EKS, register tools via Streamable HTTP sidecar. Infrastructure-only — agent chat validation (E2E-01..04) and Phase 14 moved to v3.1.

**Rescope context:** A 2026-04-21 retry of the Phase 14 chat session surfaced real code and config gaps — MCP inspect wrappers never call `load_incluster_config()`; init container's openclaw.json is missing required runtime keys; 8B model perseveration on error state. Full root-cause analysis in `.planning/v3.0-KNOWN-ISSUES.md`. Infrastructure deliverables proven functional via direct MCP invocation — only the end-to-end chat experience blocked.

- [x] **Phase 11: Streamable HTTP Transport** - Add HTTP transport mode to MCP server (code-only, no cluster needed) (completed 2026-03-23)
- [x] **Phase 12: NemoClaw EKS Topology** - Deploy NemoClaw/OpenClaw to EKS using agent-sandbox CRD; validate topology before manifests (completed 2026-03-24)
- [x] **Phase 13: Kubernetes Manifests and Sidecar Wiring** - Author complete K8s manifest set; wire MCP sidecar into OpenClaw pod (completed 2026-03-24)
- [~] **Phase 14: End-to-End Validation** — descoped; Plan 14-01 infra prep retained as shipped. Plan 14-02 chat session moves to v3.1 along with FIX-01..04 and E2E-01..04.

## Phase Details

### Phase 11: Streamable HTTP Transport
**Goal**: MCP server runs in dual-mode: stdio (default) and Streamable HTTP, selected by env var, fully validated locally before any cluster work begins
**Depends on**: Nothing (Phase 10 complete)
**Requirements**: XPORT-01, XPORT-02, XPORT-03, XPORT-04
**Success Criteria** (what must be TRUE):
  1. `curl localhost:8080/health` returns HTTP 200 when server starts with `MCP_TRANSPORT=http`
  2. `MCP_TRANSPORT=stdio` (default) starts the server exactly as before; all 103 existing tests pass unchanged
  3. `MCP_TRANSPORT=http` starts the server in Streamable HTTP mode on the port set by `MCP_PORT`
  4. Tool calls over HTTP return the same flat JSON responses as stdio (depth contract preserved)
  5. `openclaw.json` points to `http://localhost:8080/mcp` with `"transport": "streamable-http"` replacing the stdio startup block
**Plans:** 2/2 plans complete

Plans:
- [ ] 11-01-PLAN.md — Dual-mode transport in config.py and server.py with health endpoint and tests
- [ ] 11-02-PLAN.md — Update openclaw.json, generator, test assertions, and Dockerfile EXPOSE

### Phase 12: NemoClaw EKS Topology
**Goal**: NemoClaw/OpenClaw is running and reachable on EKS using the experimental agent-sandbox CRD approach; topology confirmed and documented before any manifests are written
**Depends on**: Phase 11
**Requirements**: NCLAW-01, NCLAW-03
**Success Criteria** (what must be TRUE):
  1. NemoClaw/OpenClaw pod is Running in EKS cluster (`kubectl get pods` shows Ready)
  2. NemoClaw egress policy explicitly allows loopback access so sidecar port is reachable
  3. GPU node group and NVIDIA GPU Operator confirmed operational (agent container starts without GPU errors)
  4. Topology decision (agent-sandbox CRD approach) documented as a Key Decision in PROJECT.md
**Plans:** 2/2 plans complete

Plans:
- [ ] 12-01-PLAN.md — Create Sandbox CR manifest, Secret templates, operator install script, and smoke test script
- [x] 12-02-PLAN.md — Deploy to EKS, validate topology, and write TOPOLOGY.md reference for Phase 13
 (completed 2026-03-24)
### Phase 13: Kubernetes Manifests and Sidecar Wiring
**Goal**: Complete Kubernetes manifest set authored and applied; MCP sidecar running inside the OpenClaw pod with correct RBAC, startup ordering, and runtime-generated openclaw.json
**Depends on**: Phase 12
**Requirements**: K8S-01, K8S-02, K8S-03, K8S-04, K8S-05, NCLAW-02, NCLAW-04
**Success Criteria** (what must be TRUE):
  1. MCP sidecar container starts after NemoClaw pod readiness; `kubectl logs` shows no startup race errors
  2. `kubectl logs <mcp-sidecar>` shows `/health` returning 200 before OpenClaw attempts tool registration
  3. Blueprint YAML files are accessible inside the sidecar at `BLUEPRINTS_DIR` via volume mount
  4. `openclaw.json` is generated at pod startup from env vars (not baked into the image); correct URL and transport visible in pod logs
  5. `weka-mcp-server-sa` ServiceAccount exists with scoped ClusterRole (not reusing operator's service account)
**Plans:** 3/3 plans complete

Plans:
- [ ] 13-01-PLAN.md — RBAC manifest (SA + ClusterRole + ClusterRoleBinding) and SKILL.md ConfigMap
- [ ] 13-02-PLAN.md — Update Sandbox CR with init container, MCP sidecar, git-sync, and volumes; create validation script
- [ ] 13-03-PLAN.md — Deploy manifests to EKS cluster, run live validation, human verification

### Phase 14: End-to-End Validation — descoped to v3.1
**Status**: Plan 14-01 infrastructure prep retained in v3.0 as shipped. Plan 14-02 chat session moved to v3.1.
**Depends on**: Phase 13
**Original requirements**: E2E-01, E2E-02, E2E-03, E2E-04 — now deferred to v3.1
**Rescope reason**: 2026-04-21 retry surfaced code gap (inspect tool wrappers miss `load_incluster_config`), config gap (init container openclaw.json minimal), and model reliability gap (Llama 3.1 8B). Full details in `.planning/v3.0-KNOWN-ISSUES.md`.
**Plans:**
- [x] 14-01-PLAN.md — Infrastructure prep: prereq validation, Service+HTTPRoute manifests, evidence capture scripts (shipped; artifacts remain useful for v3.1 retry)
- [~] 14-02-PLAN.md — Descoped. Will re-execute in v3.1 after FIX-01..04 are addressed.

## Progress

**Execution Order:** 11 → 12 → 13 → 14 → 15

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 6. MCP Scaffold and Read-Only Tools | v2.0 | 3/3 | Complete | 2026-03-22 |
| 7. Validation, Apply, and Status Tools | v2.0 | 2/2 | Complete | 2026-03-22 |
| 8. SKILL.md, Agent Context, and Cleanup | v2.0 | 3/3 | Complete | 2026-03-22 |
| 9. Deployment and Registration | v2.0 | 2/2 | Complete | 2026-03-22 |
| 10. Integration Bug Fixes | v2.0 | 1/1 | Complete | 2026-03-22 |
| 11. Streamable HTTP Transport | v3.0 | 2/2 | Complete | 2026-03-24 |
| 12. NemoClaw EKS Topology | v3.0 | 2/2 | Complete | 2026-03-24 |
| 13. Kubernetes Manifests and Sidecar Wiring | v3.0 | 3/3 | Complete | 2026-03-24 |
| 14. End-to-End Validation | v3.1 | 1/2 | Descoped → v3.1 | 2026-04-21 |
| 15. App Categories Feature | 2/3 | In Progress|  | - |
