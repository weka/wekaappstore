# Milestones

## v7.0 Secret Management, WEKA Storage Integration & Dynamic Blueprint System (Shipped: 2026-06-17)

**Phases completed:** 6 phases (21-26), 18 plans
**Timeline:** 2026-06-11 → 2026-06-17 (6 days, 176 commits)
**Files changed:** 95 | +21,355 / -576 lines
**Requirements delivered:** 45/46 — CRD-01..06, OPS-01..09, GUI-01..15, API-01..08, SDK-01..05, DYN-01..06, DYN-08; DYN-07 Partial (external repo)
**Archive:** `.planning/milestones/v7.0-ROADMAP.md`

**Note:** Encompasses both v6.0 (Secret Management, Phases 21-25) and v7.0 (Dynamic Blueprint System, Phase 26) — archived together as a continuous sprint.

**Key accomplishments:**
- `WarpCredential` CRD with typed schema, enum admission validation (`nvidia-ngc` | `huggingface` | `weka-storage`), status subresource, namespace-scoped RBAC for operator secret management
- Operator reconciler auto-derives type-specific secrets: NGC → API key + docker pull secret; HuggingFace → token; WEKA Storage → username/token/endpoint; idempotent with kopf status conditions
- `/api/credentials` CRUD REST API + `/api/weka/overview` proxy (60s server-side cache, `?bust=1` bypass); legacy `/api/secret/{nvidia,huggingface}` endpoints retired
- Settings page overhaul: Credential Management section (3 types, inline add forms, traffic-light states polling every 2s), WEKA Storage Overview panel (capacity bar, filesystem table, backend IP grid)
- `_credential_macros.html` Jinja2 macro SDK (`credential_select` + `weka_storage_select` with `warpSyncEndpoint` JS); `_get_credentials_by_type` helper auto-injected into all blueprint template contexts
- Dynamic blueprint discovery: `parse_x_variables` + `find_blueprint` scanner + generic `blueprint.html`; `/deploy-stream` refactored to generic `variables` JSON dict; hardcoded `app_map` and all 7 per-variable positional params removed from `main.py`

**Tech debt captured:**
- DYN-07 production blueprint migration (oss-rag, openfold, nvidia) deferred to external `warp-blueprints` repo
- v3.1 E2E chat validation still deferred (see `.planning/v3.0-KNOWN-ISSUES.md`)
- v5.0 Phases 19-20 (Validator Soft-Warning, AIDP Migration Smoke Test) unstarted

---

## v4.0 App Categories on Home Screen (Shipped: 2026-04-21)

**Phases completed:** 1 phase (Phase 15), 3 plans, 8 tasks
**Timeline:** 2026-04-21 (single session, ~1h 40min from start to audit)
**Commits:** 9 commits touching Phase 15 artifacts; 18 total including milestone init, research, and audit
**LOC:** `app-store-gui/webapp/templates/index.html` +162/-56 (single file)
**Requirements delivered:** 14/14 — CAT-01..03, FIL-01..03, VIS-01..02, URL-01..03, A11Y-01..03
**Audit:** `.planning/milestones/v4.0-MILESTONE-AUDIT.md` — passed, 10/10 must-haves, 5/5 critical pitfalls mitigated

**Key accomplishments:**
- Three top-level app-family categories (AIDP, WARP, Partner) rendered as selectable filter cards above the App Catalog grid on the home screen
- Blueprint → category mapping: AIDP=1 app (AI Agent for Enterprise Research), WARP=4 apps (OSS RAG, NVIDIA RAG, NVIDIA VSS, OpenFold), Partner=0 apps with intentional empty state
- Client-side filter with URL hash deep-link support (`/#category=<key>` via `history.replaceState`, one-back-press-leaves-site guaranteed)
- Full keyboard accessibility: native `<button>` via `CardActionArea`, `aria-pressed` state, Tab/Enter/Space toggles
- Mobile-responsive: cards stack vertically below `md` breakpoint
- Clean architecture: new `AppShell` component lifts `ThemeProvider`, pure prop-based `Catalog` component, new `Categories` + `EmptyState` components — all in the existing single-file CDN-React IIFE pattern (no build step, no new dependencies)
- Five critical pitfalls from research encoded as grep-level verification and mitigated in live code (JSX forbidden, `component:'a'` absent on category cards, `replaceState` never on mount, hash parser uses `startsWith('#category=')`, exactly one `ThemeProvider` in the file)

**Tech debt captured (not blockers):**
- No phase VALIDATION.md (Nyquist) — research was skipped since milestone-level research already produced the authoritative 3-step build order and pitfalls; matches v3.0 pattern
- Pre-existing unrelated duplicate `async function refreshAuthStatus()` at `index.html` lines 436 and 529 (infrastructure script block, outside the React IIFE) — worth cleaning up in a future polish pass
- 3 inline comments rephrased during 15-03 to avoid tripping the plan's own grep invariants — documented in 15-03 SUMMARY

**Post-ship patches (applied after v4.0 was archived and tagged):**

- **2026-04-21 — AIDP blueprint rebrand and category card sizing.** The single blueprint in the AIDP category was renamed from "AI Agent for Enterprise Research" to "NeuralMesh AIDP" with a new product-positioned description ("Embed data on your NeuralMesh storage automatically and keep it in-sync with your data"), `comingSoon` banner removed, and `href` changed from `/blueprint/ai-agent-enterprise-research` to `/blueprint/neuralmesh-aidp`. Flask detail template renamed to match (`blueprint_neuralmesh-aidp.html`) with title/h1/description updated; `app_map` placeholder comment in `main.py` updated to new slug. Category cards in the home-screen Categories row shrunk ~20-25% (subtitle2 title, caption description, tighter padding `p: 1.25`, grid spacing `1.5` instead of `2.5`) to look more like a tab row and less like full cards.
  - Reverses v4.0 REQUIREMENTS.md Out-of-Scope item "Renaming the AI Agent for Enterprise Research blueprint" (was excluded during planning; product decision changed after shipping).
  - All changes scoped to `app-store-gui/webapp/templates/index.html`, `blueprint_neuralmesh-aidp.html` (renamed), and a one-line comment in `main.py`. No new dependencies.

---

## v3.0 Live EKS Deployment (Shipped: 2026-04-21, rescoped from "...and Agent Testing")

**Phases completed:** 3 full (11, 12, 13) + 1 partial (14-01 infra prep)
**Commits across v3.0 phases:** 40+
**Requirements delivered:** 13/17 — XPORT-01..04, K8S-01..05, NCLAW-01..04

**Key accomplishments:**
- Streamable HTTP transport for MCP server (port 8080, stateless, `/health` endpoint)
- OpenClaw/NemoClaw deployed to EKS GPU node (NVIDIA A10G) via experimental agent-sandbox CRD
- Complete Kubernetes manifest set: dedicated RBAC (`weka-mcp-server-sa` + scoped ClusterRole), SKILL.md ConfigMap, MCP sidecar wired into the Sandbox CR with init-container-generated openclaw.json, git-sync blueprint catalog
- Plan 14-01 infrastructure prep shipped: prereq validation script, evidence capture script, Gateway API Service/HTTPRoute manifests

**Rescoped out:** E2E-01..04 agent chat validation. A 2026-04-21 retry session proved infrastructure works (direct MCP tool invocation returns structured responses) but surfaced code bug in inspect tool wrappers, config gap in init container, and small-model reliability issues. See `.planning/v3.0-KNOWN-ISSUES.md` for full analysis. Moved to v3.1 along with four prerequisite fixes (FIX-01..04).

---

## v2.0 OpenClaw MCP Tool Integration (Shipped: 2026-03-22)

**Phases completed:** 5 phases (6-10), 11 plans
**Lines of code:** 4,628 Python (mcp-server/)
**Commits:** 63 across v2.0 phases
**Tests:** 103 passing

**Key accomplishments:**
- 8-tool MCP server with flat agent-friendly JSON responses (inspect_cluster, inspect_weka, list_blueprints, get_blueprint, get_crd_schema, validate_yaml, apply, status)
- SKILL.md: 12-step agent workflow with validate-retry loop, re-inspect-before-apply, and negative YAML examples
- Mock agent harness with description-based tool selection exercising full inspect→validate→apply chain
- Approval-gated apply tool with boolean identity check preventing unapproved CR creation
- Deprecated v1.0 backend-brain code removed (12 files, planning routes, pruned exports)
- Container-ready: Dockerfile, GitHub Actions CI/CD to wekachrisjen Docker Hub, README with OpenClaw/NemoClaw registration docs

---

