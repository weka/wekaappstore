# Roadmap: OpenClaw MCP Tools For WEKA App Store

## Milestones

- ✅ **v2.0 OpenClaw MCP Tool Integration** - Phases 6-10 (shipped 2026-03-22)
- ✅ **v3.0 Live EKS Deployment** - Phases 11-13 (shipped 2026-04-21, rescoped from original v3.0 "...and Agent Testing")
- 🔜 **v3.1 E2E Chat Validation** - Phase 14 deferred here along with 4 known issues from v3.0 retry (see `.planning/v3.0-KNOWN-ISSUES.md`)
- ✅ **v4.0 App Categories on Home Screen** - Phase 15 (shipped 2026-04-21)
- 🔜 **v5.0 AppStack Variable Substitution** - Phases 16-20
- 🔜 **v6.0 Secret Management & WEKA Storage Integration** - Phases 21-25

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
- [x] **Phase 17: CRD Schema Additive Update** — `spec.appStack.variables` optional map added to CRD; admission-validated as string-only; independently deployable (completed 2026-05-06)
- [x] **Phase 18: Operator Wiring and Docs** — Wire `render()` into `handle_appstack_deployment` and `load_values_from_reference`; key-name validation; fetch-error upgrade; `field='spec'` guard; user-facing README section (completed 2026-05-08)
- [ ] **Phase 19: Validator Soft-Warning and Portable Fixture** — Validator accepts `variables:` block without error; soft-warns on hardcoded DNS / `namespace:` literals; `ai-research-portable.yaml` fixture
- [ ] **Phase 20: AIDP Migration Smoke Test** — Follow-up PR in separate `aidp` repo; end-to-end cluster verification that feature works in production

### v6.0 Secret Management & WEKA Storage Integration

**Milestone Goal:** Give App Store administrators a first-class credential management system — named, multi-key storage for NGC/HuggingFace/WEKA credentials via a new `WarpCredential` CRD, automatic secret derivation by the operator, a blueprint Jinja2 macro SDK for credential selection, and live WEKA storage visibility on the Settings page.

- [x] **Phase 21: WarpCredential CRD and Helm RBAC** — `WarpCredential` CRD defined in Helm chart; operator service account has Secret CRUD permissions scoped to the App Store namespace (completed 2026-06-11)
- [x] **Phase 22: Operator WarpCredential Reconciler** — Operator reconciles `WarpCredential` CRs, deriving correct secrets per type and maintaining `status` conditions; idempotent (completed 2026-06-11)
- [ ] **Phase 23: Backend Credentials API and WEKA Overview Proxy** — GUI backend exposes `/api/credentials` CRUD endpoints and `/api/weka/overview` proxy; old secret endpoints removed
- [ ] **Phase 24: Settings GUI Overhaul** — Settings page restructured with Credential Management first, per-type credential lists with traffic-light states, inline add forms, and WEKA Storage Overview panel
- [ ] **Phase 25: Blueprint Credential Selector SDK** — Blueprint install pages render credential dropdowns and WEKA endpoint fields using Jinja2 macros; `credentials_by_type` injected automatically into all blueprint template contexts

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
**Plans:** 1/1 plans complete

Plans:
- [x] 17-01-PLAN.md — Insert spec.appStack.variables schema block, bump Chart.yaml to 0.1.62, ship verify-crd.sh with 4 dry-run fixtures + --apply mode + kubectl explain keyword check

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
**Plans:** 5/5 plans complete

Plans:
- [x] 18-01-PLAN.md — Operator wiring (handle_appstack_deployment, load_values_from_reference, _render_or_raise helper, field='spec' decorator, Chart.yaml bump 0.1.62 -> 0.1.63)
- [x] 18-02-PLAN.md — README new top-level section ## Variable substitution in AppStack manifests (DOC-01..06)
- [x] 18-03-PLAN.md — operator_module/tests/test_appstack.py (TST-02 surface; OP-06..08, OP-10..12)
- [x] 18-04-PLAN.md — operator_module/tests/test_helm_non_wiring.py (TST-05; OP-09 non-wiring lock)
- [x] 18-05-PLAN.md — operator_module/tests/test_backward_compat_snapshot.py + snapshots/ai-research/ baselines (TST-03)

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

---

### Phase 21: WarpCredential CRD and Helm RBAC
**Goal**: The `WarpCredential` CRD is defined in the Helm chart and the operator's service account has Secret CRUD permissions scoped to the App Store namespace
**Depends on**: Nothing (independently deployable)
**Requirements**: CRD-01, CRD-02, CRD-03, CRD-04, CRD-05, CRD-06
**Success Criteria** (what must be TRUE):
  1. `kubectl apply -f weka-app-store-operator-chart/templates/crd.yaml` succeeds; `kubectl get crd warpcredentials.warp.io` returns the CRD
  2. A CR with `spec.type: invalid-type` is rejected at admission; a CR with valid type+secretRef is accepted
  3. A `WarpCredential` CR with `spec.type: weka-storage` and `spec.endpoint: https://weka-cluster:14000` is accepted; same CR without `spec.endpoint` is also accepted (field is optional)
  4. `kubectl explain warpcredential.status` shows `conditions`, `derivedSecrets`, `lastSyncTime`, and `wekaEndpoint` fields
  5. Helm chart deploys the new `Role` + `RoleBinding` granting operator Secret CRUD in the App Store namespace; `helm lint` and `helm template` pass without error
**Plans**: 2 plans

Plans:
- [x] 21-01-PLAN.md — WarpCredential CRD schema (spec + status subresource)
- [x] 21-02-PLAN.md — Namespace-scoped Secret CRUD Role + RoleBinding + Chart.yaml 0.1.64 bump

### Phase 22: Operator WarpCredential Reconciler
**Goal**: The operator reconciles `WarpCredential` CRs, deriving the correct secrets for each credential type and maintaining `status` conditions; derived secrets are idempotent
**Depends on**: Phase 21 (CRD must exist before handler can be registered)
**Requirements**: OPS-01, OPS-02, OPS-03, OPS-04, OPS-05, OPS-06, OPS-07, OPS-08, OPS-09, API-08
**Success Criteria** (what must be TRUE):
  1. Creating a `WarpCredential` of type `nvidia-ngc` results in both `warp-<name>-apikey` (Opaque, key `NGC_API_KEY`) and `warp-<name>-docker` (type `kubernetes.io/dockerconfigjson`) in the App Store namespace within the kopf retry window
  2. Creating a `WarpCredential` of type `huggingface` results in `warp-<name>-token` (Opaque, key `HF_API_KEY`)
  3. Creating a `WarpCredential` of type `weka-storage` results in `warp-<name>-token` (Opaque, keys `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`); `status.wekaEndpoint` is set to `spec.endpoint`
  4. Manually deleting a derived secret and triggering a reconcile restores it (idempotency)
  5. A `WarpCredential` whose referenced Secret does not exist results in `kopf.TemporaryError` (not a crash); `status.conditions[KeyReady].status = "False"` with reason `KeyMissing`
  6. Deleting a `WarpCredential` CR leaves all `warp-<name>-*` secrets intact; operator logs a warning
  7. No key values appear in operator logs at any log level (`pytest operator_module/tests/test_warp_credential.py` verifies derivation logic without network access)
**Plans:** 3/3 plans complete

Plans:
**Wave 1**
- [x] 22-01-PLAN.md — Pure derivation helpers (_derive_ngc_payloads, _derive_hf_payload, _derive_weka_payload), kr8s I/O wrappers (_read_source_secret, _apply_secret_idempotent), status-condition builder, _VALID_WARPCRED_TYPES constant, delete_warpcredential handler with optional=True (OPS-04/05/06 payloads, OPS-08, OPS-09 plumbing, API-08)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 22-02-PLAN.md — reconcile_warpcredential stacked-decorator handler (create / update field=spec / resume) with type dispatch, status patch on every failure branch, success status writes including wekaEndpoint (OPS-01/02/03/07, OPS-04/05/06 wiring, OPS-09)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 22-03-PLAN.md — operator_module/tests/test_warp_credential.py: pure-helper tests + idempotency tests + handler-path tests + delete-handler test + caplog-based API-08 assertion (validates all 7 ROADMAP success criteria)

### Phase 23: Backend Credentials API and WEKA Overview Proxy
**Goal**: The GUI backend exposes the `/api/credentials` CRUD endpoints and `/api/weka/overview` proxy; old secret endpoints removed
**Depends on**: Phase 22 (operator must be able to reconcile before the GUI API is useful; independently codeable but depends on CRD+operator for live testing)
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08
**Success Criteria** (what must be TRUE):
  1. `GET /api/credentials` returns a JSON array with correct shape for each WarpCredential CR in the namespace (name, displayName, type, ready, optional fields per type)
  2. `POST /api/credentials` with valid body creates a `warp-cred-<slug>` Secret and a `WarpCredential` CR; slug collision is handled by appending a suffix
  3. `DELETE /api/credentials/<name>` removes the CR and raw Secret; derived secrets remain
  4. `GET /api/credentials?type=nvidia-ngc` returns only NGC-type credentials with `ready: true`
  5. `GET /api/weka/overview?credential=<name>` returns the structured JSON schema (capacity, filesystems, backendNodes, fetchedAt); second request within 60 seconds returns cached data (same `fetchedAt`); `?bust=1` bypasses cache
  6. `GET /api/secret/nvidia` and `GET /api/secret/huggingface` return 404
  7. No token values appear in response bodies or server logs
**Plans:** 3/4 plans executed

Plans:
**Wave 1**
- [x] 23-01-PLAN.md — Remove deprecated /api/secret/{huggingface,nvidia} handlers from main.py; strip HuggingFace + NVIDIA HTML sections and JS from settings.html; rewire loadBlueprints namespace fallback (API-07)

**Wave 2** *(blocked on Wave 1 completion — same main.py file ownership)*
- [x] 23-02-PLAN.md — /api/credentials CRUD: GET list + ?type filter, POST with slug + collision + raw Secret + WarpCredential CR, DELETE preserving derived secrets; slug + response helpers; _CREDENTIAL_TYPE_KEYS constant (API-01, API-02, API-03, API-04, API-08)

**Wave 3** *(blocked on Wave 2 completion — same main.py file ownership)*
- [x] 23-03-PLAN.md — /api/weka/overview proxy: _resolve_weka_credential_secret + _weka_login + _weka_get_json helpers; _assemble_weka_overview pure transform; 60s namespace-scoped cache; ?bust=1 bypass; 502 on WEKA failures (API-05, API-06, API-08)

**Wave 4** *(blocked on Waves 2 + 3 — tests exercise handlers from both)*
- [ ] 23-04-PLAN.md — app-store-gui/tests/test_credentials_api.py: pure-helper tests + GET/POST/DELETE handler tests + WEKA overview assembler + cache TTL + bust + namespace scoping + no-token-leak assertions (all API requirements)

### Phase 24: Settings GUI Overhaul
**Goal**: The Settings page is restructured with Credential Management first, per-type credential lists with traffic-light states, inline add forms, and the WEKA Storage Overview panel
**Depends on**: Phase 23 (needs the API endpoints to be functional for full interaction)
**Requirements**: GUI-01, GUI-02, GUI-03, GUI-04, GUI-05, GUI-06, GUI-07, GUI-08, GUI-09, GUI-10, GUI-11, GUI-12, GUI-13, GUI-14, GUI-15
**Success Criteria** (what must be TRUE):
  1. On page load, Credential Management section appears above all other Settings sections; three type sub-sections (NGC, HuggingFace, WEKA) each show stored credentials or "(none stored)"
  2. Clicking `[+ Add]` on NGC type expands an inline form with Name and Key fields; clicking `[+ Add]` on WEKA expands a 4-field form (Name, Username, API Token, Endpoint); only one form open per type at a time
  3. After saving a new credential, the row enters amber "Verifying..." state; polls every 2 seconds; transitions to green when operator sets `KeyReady=True` (or shows red with error message if operator reports failure)
  4. In green state the row shows display name + Ready badge + Delete button only — no key input visible
  5. Clicking Delete removes the row; derived secrets remain in cluster
  6. With one WEKA Storage credential registered, the WEKA Storage Overview panel appears below Credential Management showing a capacity bar, filesystem table (human names, utilisation bars, >=90% amber), and backend IP grid
  7. With zero WEKA credentials, the panel is replaced by a "No WEKA Storage credential configured" hint
  8. `[Refresh]` button triggers a fresh WEKA API call (bypasses 60s cache); "Last updated" shows actual data age
**Plans**: TBD
**UI hint**: yes

### Phase 25: Blueprint Credential Selector SDK
**Goal**: Blueprint install pages can render credential dropdowns and WEKA endpoint fields using Jinja2 macros; `credentials_by_type` is injected automatically into all blueprint template contexts
**Depends on**: Phase 23 (needs credentials API); can be developed in parallel with Phase 24
**Requirements**: SDK-01, SDK-02, SDK-03, SDK-04, SDK-05
**Success Criteria** (what must be TRUE):
  1. A blueprint template using `{% from "_credential_macros.html" import credential_select %}` and `{{ credential_select(type="nvidia-ngc", field_name="ngc_credential") }}` renders a `<select>` populated with all ready NGC credentials
  2. When no credentials of the requested type are ready, `credential_select` renders a hint paragraph with a link to `/settings#credentials`
  3. `{{ weka_storage_select() }}` renders a credential dropdown + endpoint `<input>` pair; each option has a `data-endpoint` attribute; changing selection updates the endpoint field via `warpSyncEndpoint` JavaScript
  4. All blueprint install page route handlers inject `credentials_by_type` dict into their template context; the tokenvisor and glocomp blueprint templates serve as reference examples updated to use the macro
  5. If the Kubernetes API is unreachable when fetching credentials, `credentials_by_type` is an empty dict and macros degrade to hint mode — no 500 error on blueprint page load
**Plans**: TBD

## Progress

**Execution Order:** 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20
(Phase 17 can be deployed in parallel with Phase 16; Phase 19 can be worked in parallel with Phases 17-18; Phase 20 requires Phases 16-18 deployed)

v6.0 Execution Order: 21 → 22 → 23 → 24/25 (Phases 24 and 25 can be developed in parallel after Phase 23)

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
| 17. CRD Schema Additive Update | v5.0 | 1/1 | Complete   | 2026-05-06 |
| 18. Operator Wiring and Docs | v5.0 | 5/5 | Complete   | 2026-05-08 |
| 19. Validator Soft-Warning and Portable Fixture | v5.0 | 0/TBD | Not started | - |
| 20. AIDP Migration Smoke Test | v5.0 | 0/TBD | Not started | - |
| 21. WarpCredential CRD and Helm RBAC | v6.0 | 2/2 | Complete    | 2026-06-11 |
| 22. Operator WarpCredential Reconciler | v6.0 | 3/3 | Complete    | 2026-06-11 |
| 23. Backend Credentials API and WEKA Overview Proxy | v6.0 | 3/4 | In Progress|  |
| 24. Settings GUI Overhaul | v6.0 | 0/TBD | Not started | - |
| 25. Blueprint Credential Selector SDK | v6.0 | 0/TBD | Not started | - |
