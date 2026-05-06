# Roadmap: OpenClaw MCP Tools For WEKA App Store

## Milestones

- ✅ **v2.0 OpenClaw MCP Tool Integration** - Phases 6-10 (shipped 2026-03-22)
- ✅ **v3.0 Live EKS Deployment** - Phases 11-13 (shipped 2026-04-21, rescoped from original v3.0 "...and Agent Testing")
- 🔜 **v3.1 E2E Chat Validation** - Phase 14 deferred here along with 4 known issues from v3.0 retry (see `.planning/v3.0-KNOWN-ISSUES.md`)
- ✅ **v4.0 App Categories on Home Screen** - Phase 15 (shipped 2026-04-21)
- 🔜 **v5.0 AppStack Variable Substitution** - Phases 16-20

## Phases

<details>
<summary>✅ v4.0 App Categories on Home Screen (Phase 15) — SHIPPED 2026-04-21</summary>

3-card app-family filter row (AIDP, WARP, Partner) above the catalog grid with URL hash deep-link support and full keyboard accessibility. Single-file change to `app-store-gui/webapp/templates/index.html` — no build step, no new dependencies.

- [x] Phase 15: App Categories Feature (3/3 plans) — completed 2026-04-21

See `.planning/milestones/v4.0-ROADMAP.md` for full phase details and `.planning/MILESTONES.md` for shipping notes.

</details>

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

### v5.0 AppStack Variable Substitution

**Milestone Goal:** Add `spec.appStack.variables` to the `WekaAppStore` CR. The operator performs a single `${VAR}` substitution pass over `kubernetesManifest` strings and `valuesFiles` content (loaded from ConfigMaps/Secrets) before they are applied or merged into Helm values. Blueprints become portable across namespaces and environments without external pre-render tooling.

- [x] **Phase 16: render() Helper and Test Scaffolding** — Pure `render()` function with pre-scan backward-compat guard; new `operator_module/tests/` directory; no live operator paths touched (completed 2026-05-06)
- [ ] **Phase 17: CRD Schema Additive Update** — `spec.appStack.variables` optional map added to CRD; admission-validated as string-only; independently deployable
- [ ] **Phase 18: Operator Wiring and Docs** — Wire `render()` into `handle_appstack_deployment` and `load_values_from_reference`; key-name validation; fetch-error upgrade; `field='spec'` guard; user-facing README section
- [ ] **Phase 19: Validator Soft-Warning and Portable Fixture** — Validator accepts `variables:` block without error; soft-warns on hardcoded DNS / `namespace:` literals; `ai-research-portable.yaml` fixture
- [ ] **Phase 20: AIDP Migration Smoke Test** — Follow-up PR in separate `aidp` repo; end-to-end cluster verification that feature works in production

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
**Plans:** 3/3 plans complete

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

---

### Phase 16: render() Helper and Test Scaffolding
**Goal**: A tested, standalone `render()` function with the pre-scan backward-compat guard exists in `operator_module/main.py`, and the new `operator_module/tests/` directory is initialized — no live operator reconcile paths are modified in this phase
**Depends on**: Nothing (Phase 15 complete; this phase touches no live paths)
**Requirements**: OP-01, OP-02, OP-03, OP-04, OP-05, TST-01
**Success Criteria** (what must be TRUE):
  1. `render("$CRDS && $CRD", {})` returns the string unchanged (pre-scan guard; OP-01 verified) — existing `cluster_init/` shell-script manifests cannot regress after upgrade
  2. `render("hello ${NAME}", {"NAME": "world"})` returns `"hello world"`; `render("price is $$5", {"x": "y"})` returns `"price is $5"` (OP-02, OP-04 verified)
  3. `render("value: ${UNDEF}", {"x": "y"})` raises a descriptive error naming `UNDEF`; `render("bad: ${}", {"x": "y"})` also raises a descriptive error — both `KeyError` and `ValueError` are caught (OP-03 verified). Malformed-placeholder examples pass non-empty variables to bypass the D-02 empty-vars short-circuit; in production the variables dict always contains the auto-default `${namespace}` key so this is the realistic path.
  4. `render("no-tokens", None)` and `render("no-tokens", {})` both return `"no-tokens"` unchanged (OP-05 verified)
  5. `pytest operator_module/tests/test_render.py` passes all cases including JSON-safety check (TST-01 verified)
**Plans:** 1/1 plans complete

Plans:
- [x] 16-01-PLAN.md — render() helper added to operator_module/main.py + operator_module/tests/ scaffolding (__init__.py, conftest.py, test_render.py) + operator_module/requirements-dev.txt
**UI hint**: no

### Phase 17: CRD Schema Additive Update
**Goal**: The `WekaAppStore` CRD schema accepts an optional `spec.appStack.variables` map of string values; the updated CRD can be applied to the cluster independently of any operator code change
**Depends on**: Nothing (independently deployable; can ship in parallel with Phase 16)
**Requirements**: CRD-01, CRD-02, CRD-03
**Success Criteria** (what must be TRUE):
  1. `kubectl apply -f weka-app-store-operator-chart/templates/crd.yaml` succeeds without error on a live cluster (CRD-01 verified)
  2. A CR with `spec.appStack.variables: {namespace: foo, milvusHost: milvus.foo.svc.cluster.local}` passes Kubernetes admission validation (CRD-03 verified)
  3. A CR with `spec.appStack.variables: {count: 42}` (integer value) is rejected at admission with a type error (CRD-03 verified)
  4. `kubectl explain wekastoreapp.spec.appStack.variables` shows the description including `${VAR}` syntax, `$$` escape, `${namespace}` auto-default, and identifier-name requirement (CRD-02 verified)
  5. Existing CRs without `variables:` continue to pass admission and reconcile identically — the field is optional with no `required:` constraint (CRD-01 verified)
**Plans**: TBD

### Phase 18: Operator Wiring and Docs
**Goal**: The `render()` helper is wired into both substitution sites in `handle_appstack_deployment` and `load_values_from_reference`; `${namespace}` auto-defaults to CR namespace; key-name validation, fetch-error upgrade, and `field='spec'` guard are in place; README documents the feature; the non-wiring of `handle_helm_deployment` is locked by a test
**Depends on**: Phase 16 (render() must exist and be tested); Phase 17 (CRD must accept variables before new CRs can be submitted)
**Requirements**: OP-06, OP-07, OP-08, OP-09, OP-10, OP-11, OP-12, TST-02, TST-03, TST-05, DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06
**Success Criteria** (what must be TRUE):
  1. A `WekaAppStore` CR with `metadata.namespace: staging` and no `variables:` field applies with byte-identical Helm values dict and manifest tempfile content compared to pre-Phase-18 — backward-compat snapshot test passes (TST-03, OP-06 verified)
  2. A CR with `metadata.namespace: staging` and a `kubernetesManifest:` containing `namespace: ${namespace}` causes all resources to be created in `staging`; a `kopf.PermanentError` naming the variable and component is raised when `${unset}` appears in a manifest (OP-07, DOC-04 verified)
  3. A ConfigMap referenced via `valuesFiles:` containing `host: ${milvusHost}` deep-merges into Helm values with the resolved value given `variables: {milvusHost: milvus.staging.svc.cluster.local}`; a ConfigMap or Secret that is missing surfaces as `kopf.TemporaryError(delay=30)` rather than a silent empty dict (OP-08, OP-11 verified)
  4. A CR with `variables: {my-host: foo}` (hyphenated key) raises `kopf.PermanentError` at variables-dict build time with a message identifying `my-host` as invalid; `handle_helm_deployment` single-chart path does not receive `variables` wiring and its unit test passes (OP-09, OP-10, TST-05 verified)
  5. README contains a worked `${VAR}` example, `$$` password example, `${namespace}` auto-default explanation, strict-failure documentation using fully-resolved values (not the cross-referencing PRD example), and explicit callout that operator-control fields are not templated (DOC-01..06 verified)
**Plans**: TBD

### Phase 19: Validator Soft-Warning and Portable Fixture
**Goal**: The MCP server validator accepts CRs with `spec.appStack.variables` without spurious errors; it soft-warns operators when manifests contain hardcoded DNS names or namespace literals that could be parameterized; a portable sample blueprint fixture demonstrates the recommended pattern
**Depends on**: Nothing (no operator code dependency; can ship in parallel with Phases 17-18)
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, TST-04
**Success Criteria** (what must be TRUE):
  1. `validate_yaml(cr_with_variables_block)` returns `valid: true` with no schema error for a CR containing a well-formed `spec.appStack.variables` map (VAL-01 verified)
  2. `validate_yaml(cr_with_hardcoded_dns)` returns `valid: true` but includes a soft-warning message suggesting `${milvusHost}` for a manifest containing `milvus.rag.svc.cluster.local` (VAL-02 verified)
  3. `validate_yaml(cr_with_namespace_literal)` returns `valid: true` but includes a soft-warning when `namespace: rag` appears inside a `kubernetesManifest` and the CR's `metadata.namespace` is not `rag` (VAL-03 verified)
  4. `validate_yaml(cr_with_invalid_key)` returns `valid: false` with an error message identifying the offending key name (e.g., `my-host` does not match `[_a-zA-Z][_a-zA-Z0-9]*`) (VAL-04 verified)
  5. `validate_yaml(cr_with_integer_variable)` returns `valid: false` with an error on the non-string value; `ai-research-portable.yaml` fixture is accepted without errors or warnings (VAL-05, TST-04 verified)
**Plans**: TBD

### Phase 20: AIDP Migration Smoke Test
**Goal**: The AIDP blueprint (`aidp` repo) is migrated to use `${namespace}`, `${milvusHost}`, and `${postgresHost}` variables with fully-resolved values; applying the migrated CR with a different namespace deploys all components into that namespace with no other file changes — end-to-end production verification of the v5.0 feature

**IMPORTANT — separate repository:** All deliverables for this phase live in `/Users/christopherjenkins/git/aidp`, NOT in `wekaappstore`. Executor agents must NOT modify any files inside the `wekaappstore` repo for this phase. The phase ships as a separate PR against the `aidp` repo.

**Depends on**: Phases 16, 17, 18 deployed to cluster (CRD schema updated + operator wiring live)
**Requirements**: MIG-01, MIG-02, MIG-03, MIG-04, MIG-05
**Success Criteria** (what must be TRUE):
  1. `aidp/appstack/weka-aidp-appstack.yaml` declares `spec.appStack.variables` with `milvusHost` and `postgresHost` as fully-resolved string values (e.g., `milvus.aidp-prod.svc.cluster.local`) — not cross-referencing `${namespace}` inside a variable value (MIG-01 verified)
  2. All 17 inline `namespace: rag` literals across `kubernetesManifest:` blocks are replaced with `namespace: ${namespace}`; PV/PVC `claimRef.namespace: rag` is replaced with `${namespace}` (MIG-02, MIG-03 verified)
  3. DNS literals in `aidp/appstack/aidp-site-config.yaml` — including `milvus.rag.svc.cluster.local` and `space-manager-postgres.rag.svc.cluster.local` — are replaced with `${milvusHost}` and `${postgresHost}` references (MIG-04 verified)
  4. `kubectl apply -f appstack/weka-aidp-appstack.yaml` with `metadata.namespace: aidp-test` deploys all components into `aidp-test` with no other file changes; command output is captured as acceptance evidence in the PR description (MIG-05 verified)
  5. No `rag` namespace literal remains in `weka-aidp-appstack.yaml` or `aidp-site-config.yaml` after migration (MIG-02..MIG-04 collectively verified)
**Plans**: TBD

## Progress

**Execution Order:** 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20
(Phase 17 can be deployed in parallel with Phase 16; Phase 19 can be worked in parallel with Phases 17-18; Phase 20 requires Phases 16-18 deployed)

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
| 15. App Categories Feature | v4.0 | Complete | 2026-04-21 | 2026-04-21 |
| 16. render() Helper and Test Scaffolding | v5.0 | 1/1 | Complete   | 2026-05-06 |
| 17. CRD Schema Additive Update | v5.0 | 0/TBD | Not started | - |
| 18. Operator Wiring and Docs | v5.0 | 0/TBD | Not started | - |
| 19. Validator Soft-Warning and Portable Fixture | v5.0 | 0/TBD | Not started | - |
| 20. AIDP Migration Smoke Test | v5.0 | 0/TBD | Not started | - |
