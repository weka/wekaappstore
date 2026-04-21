# Requirements: OpenClaw MCP Tools For WEKA App Store

**Defined:** 2026-03-23
**Core Value:** OpenClaw can inspect, reason about, validate, and safely install WEKA App Store blueprints through bounded MCP tools without needing custom backend planning logic.

## v4.0 Requirements

Requirements for the **App Categories on Home Screen** milestone. Adds three top-level app categories (AIDP, WARP, Partner) as selectable filter cards on the WEKA App Store home page.

**Source PRD:** `.planning/PRD-gui-app-categories.md`
**Research:** `.planning/research/SUMMARY.md`
**Scope:** Single-file change to `app-store-gui/webapp/templates/index.html`. No new dependencies, no build step, no backend changes.

### Blueprint → Category Mapping

| Blueprint | Category | Reason |
|---|---|---|
| OSS RAG | warp | RAG platform blueprint |
| NVIDIA RAG | warp | RAG platform blueprint |
| NVIDIA VSS | warp | WARP-family per product decision |
| OpenFold | warp | WARP-family per product decision |
| AI Agent for Enterprise Research | aidp | Headline NeuralMesh AI Data Platform offering |

Result: AIDP=1 app, WARP=4 apps, Partner=0 apps (empty state intentional).

### Category Data (CAT)

- [x] **CAT-01**: Each blueprint `item` in `index.html` has a single `category` field with value in `{aidp, warp, partner}`
- [ ] **CAT-02**: User sees three category cards — labeled "AIDP", "WARP", "Partner" — rendered in that left-to-right order between Planning Studio and App Catalog
- [ ] **CAT-03**: User sees each category card styled consistently with existing catalog cards (same glassmorphism, border radius, fonts, dark theme)

### Filter Behavior (FIL)

- [ ] **FIL-01**: User clicks a category card and the App Catalog grid below filters in place to show only apps in that category
- [ ] **FIL-02**: User clicks the currently-selected category card and returns to the default "All" view (5 apps)
- [ ] **FIL-03**: User viewing a category with zero matching apps sees the inline empty-state message "No apps in this category yet."

### Visual State (VIS)

- [ ] **VIS-01**: User sees the selected category card visually emphasized with purple border and glow matching the WEKA primary color
- [ ] **VIS-02**: User sees unselected category cards at reduced opacity (0.7) when a category is active; all cards at full opacity when showing All

### URL State Sync (URL)

- [ ] **URL-01**: User can deep-link to a pre-filtered view by loading `/#category=<key>` where `<key>` is one of `aidp`, `warp`, or `partner`
- [ ] **URL-02**: User clicking Back after one or more category interactions leaves the page in a single press (no history pollution from category toggles)
- [ ] **URL-03**: User arriving with an unknown, malformed, or unrelated hash sees the default "All" view (hash parser ignores non-`#category=` hashes)

### Responsive and Accessibility (A11Y)

- [ ] **A11Y-01**: User on a mobile viewport (≤768px) sees category cards stacked vertically, each remaining tappable
- [ ] **A11Y-02**: User navigating by keyboard can focus each category card via Tab and toggle selection via Enter or Space
- [ ] **A11Y-03**: Screen-reader users hear the selected/unselected state of each category card via native `aria-pressed` semantics on a `<button>` element

## v4.0 Future Requirements (v4.1+)

Deferred after v4.0 ships. Candidates for a v4.1 polish milestone.

- **CAT-04**: Live "N apps" count Chip on each category card (deferred from PRD "Should pass" tier)
- **UX-01**: Explicit "Show all" affordance when a category is active
- **UX-02**: 150ms opacity fade on grid during category switch
- **UX-03**: CTA copy for Partner empty state (pending PMM decision and partnership pipeline)
- **A11Y-04**: `aria-label` with spelled-out "NeuralMesh AI Data Platform" on the AIDP category card
- **DATA-01**: Move `items[]` array out of `index.html` into a backend source of truth
- **CAT-05**: Multi-value category field (array) to allow a blueprint to belong to multiple categories (requires catalog growth beyond ~15 items)

## v4.0 Out of Scope

Explicitly excluded from v4.0. Each entry has a reason to prevent re-introduction.

| Feature | Reason |
|---|---|
| New React build pipeline (Vite/Webpack/npm) | PRD locks the stack to CDN React + no build step |
| New CDN dependencies | All required MUI/React APIs already in the loaded bundle (see research STACK.md) |
| Backend routes or API endpoints for categories | Filter is client-side only; no server involvement |
| Search input on the catalog | Not included; out of scope per PRD |
| Sort controls on the catalog | Not included; out of scope per PRD |
| Multi-select filtering | v4.0 is single-select; array category field is v4.1+ |
| Tag-based secondary filters | `tags[]` remain informational Chips on the card |
| Per-user persistence of last-selected category | Hash-only state; no localStorage |
| `history.pushState` per click | Pollutes back stack; `replaceState` is correct (research PITFALLS.md) |
| Per-category icons or images on the cards | Simpler is better for a 3-card row; not worth the design overhead |
| Tooltip descriptions that duplicate card copy | Anti-pattern; redundant |
| "Recently viewed" surface | Catalog too small (5 items) to warrant |
| Server-side rendering of filtered results | Flask template is static; filter runs in React on the client |
| Category authorization / gating | Every user sees every category; no RBAC layer |
| Renaming the "AI Agent for Enterprise Research" blueprint | Blueprint title stays; only category assignment changes |
| Changes to blueprint detail pages, Planning Studio, `/settings` | Scope is the home page only |

## Traceability

All 14 v4.0 requirements map to Phase 15. Phase 15 has three plans; the plan column indicates which plan first delivers each requirement.

| Requirement | Phase | Plan | Status |
|-------------|-------|------|--------|
| CAT-01 | Phase 15 | 15-01 (Data Preparation) | Pending |
| CAT-02 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| CAT-03 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| FIL-01 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| FIL-02 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| FIL-03 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| VIS-01 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| VIS-02 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| URL-01 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| URL-02 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| URL-03 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| A11Y-01 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| A11Y-02 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |
| A11Y-03 | Phase 15 | 15-03 (Categories + Filter + A11Y) | Pending |

**Coverage:**
- v4.0 requirements: 14 total
- Mapped to phases: 14 (all to Phase 15)
- Unmapped: 0

---

## v3.0 Requirements

Requirements for the Live EKS Deployment milestone (scoped to infrastructure delivery).

**Note:** Original v3.0 scope included E2E-01..04 (agent chat validation). After the 2026-04-21 retry session surfaced code and config gaps, these four requirements were moved to v3.1 along with the bugs that blocked them. See Future Requirements section and `.planning/v3.0-KNOWN-ISSUES.md`.

Each requirement below maps to a roadmap phase.

### MCP Transport

- [x] **XPORT-01**: MCP server supports Streamable HTTP transport on a configurable port alongside existing stdio
- [x] **XPORT-02**: `MCP_TRANSPORT` env var selects transport mode (`stdio` default, `http` for sidecar deployment)
- [x] **XPORT-03**: Health endpoint (`/health`) returns 200 when server is ready for tool calls
- [x] **XPORT-04**: HTTP transport operates in stateless mode (no session ID dependency)

### Kubernetes Deployment

- [x] **K8S-01**: Sidecar container spec defines the MCP server container for inclusion in OpenClaw pod
- [x] **K8S-02**: ServiceAccount and RBAC ClusterRole grant minimum permissions for all 8 tools (read cluster/WEKA, create WekaAppStore CRs)
- [x] **K8S-03**: Readiness probe on health endpoint prevents tool discovery before server is ready
- [x] **K8S-04**: Blueprint manifests are available to the sidecar via shared volume mount
- [x] **K8S-05**: `openclaw.json` is generated at pod startup from environment variables (not baked into image)

### NemoClaw/OpenClaw Setup

- [x] **NCLAW-01**: NemoClaw/OpenClaw deployed to EKS using experimental agent-sandbox CRD approach
- [x] **NCLAW-02**: MCP server registered with OpenClaw via Streamable HTTP transport (`http://localhost:<port>/mcp`)
- [x] **NCLAW-03**: NemoClaw egress policy explicitly allows loopback access to MCP sidecar port
- [x] **NCLAW-04**: SKILL.md loaded by agent at registration time

## Future Requirements (v3.1 — E2E Chat Validation)

Deferred from v3.0 after 2026-04-21 retry session discovered code and config gaps blocking the E2E chat experience. See `.planning/v3.0-KNOWN-ISSUES.md` for root causes.

### E2E Validation (moved from v3.0)

- **E2E-01**: Agent can inspect cluster resources and WEKA storage through chat
- **E2E-02**: Agent can list and describe blueprints through chat
- **E2E-03**: Agent can generate, validate, and apply a WekaAppStore CR through the full SKILL.md workflow
- **E2E-04**: Agent reports deployment status after apply

### Prerequisite fixes for E2E retry

- **FIX-01**: `mcp-server/tools/inspect_cluster.py` and `inspect_weka.py` call `load_incluster_config()` before using the kubernetes-python client (Issue 1)
- **FIX-02**: Init container openclaw.json includes `tools.exec.host`, `agents.defaults.sandbox.mode`, and model provider config (Issue 2)
- **FIX-03**: Phase 13/14 validation scripts invoke each MCP tool over HTTP and assert non-null response data (Nyquist gap)
- **FIX-04**: Evaluate upgrading the NIM model from Llama 3.1 8B to a larger variant for reliable multi-step tool use (Issue 3)

### Other v3.1+ candidates

- **LIVE-02**: SKILL.md tuning based on real agent behavior
- **LIVE-03**: Multi-blueprint coexistence assessment tool
- **LIVE-04**: Maintainer-facing draft blueprint authoring through agent
- **PROD-01**: Production hardening (TLS, auth, rate limiting for HTTP transport)
- **PROD-02**: Monitoring and alerting for MCP server health

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom chat UI | OpenClaw/NemoClaw provides conversation interface natively |
| SSE transport | Deprecated April 2026; Streamable HTTP is the correct transport |
| Multi-tenant MCP server | Single-operator deployment; multi-client is v4.0+ |
| NemoClaw production hardening | Early preview software; focus on proving the chain works |
| Dedicated EC2 VM deployment | Using agent-sandbox CRD approach instead |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| XPORT-01 | Phase 11 | Complete |
| XPORT-02 | Phase 11 | Complete |
| XPORT-03 | Phase 11 | Complete |
| XPORT-04 | Phase 11 | Complete |
| K8S-01 | Phase 13 | Complete |
| K8S-02 | Phase 13 | Complete |
| K8S-03 | Phase 13 | Complete |
| K8S-04 | Phase 13 | Complete |
| K8S-05 | Phase 13 | Complete |
| NCLAW-01 | Phase 12 | Complete |
| NCLAW-02 | Phase 13 | Complete |
| NCLAW-03 | Phase 12 | Complete |
| NCLAW-04 | Phase 13 | Complete |
| E2E-01 | Phase 14 (descoped → v3.1) | Deferred |
| E2E-02 | Phase 14 (descoped → v3.1) | Deferred |
| E2E-03 | Phase 14 (descoped → v3.1) | Deferred |
| E2E-04 | Phase 14 (descoped → v3.1) | Deferred |

**Coverage:**
- v3.0 requirements: 13 total (after rescope)
- Mapped to phases: 13 (all satisfied)
- Deferred to v3.1: 4 (E2E-01..04)

---
*Requirements defined: 2026-03-23*
*Last updated: 2026-04-21 — v4.0 traceability table populated by roadmapper; all 14 REQ-IDs mapped to Phase 15 with plan-level assignments.*
