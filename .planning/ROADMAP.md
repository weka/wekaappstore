# Roadmap: v8.0 Guided Install Wizard — WEKA Operator, CSI & Storage Classes

**Milestone:** v8.0
**Created:** 2026-06-24
**Granularity:** coarse
**Core Value:** A customer can stand up the full WEKA storage stack (operator, CSI driver, client, storage classes) from the App Store install wizard by answering a short form — no manual `kubectl`/`helm`/base64 work.
**Authoritative spec:** `.planning/PRD-install-wizard-weka-storage-stack.md`

Phase numbering continues from the previous milestone (last phase was 26). This milestone runs phases **27–31**. Prior milestones (v2.0–v7.0, phases 6–26) are archived under `.planning/milestones/`.

## Phases

- [ ] **Phase 27: Install Blueprint Authoring** - Parameterized `app-store-install.yaml` AppStack CR encoding the full ordered install with `stringData` secrets and idempotent components.
- [ ] **Phase 28: Operator Helm Auth & CRD Discovery** - Operator authenticates to quay before OCI chart pulls and stops caching empty-CRD failures — the highest-risk de-risk, built before the frontend.
- [ ] **Phase 29: Backend Wiring & Secret Safety** - Server-side dockerconfigjson/endpoint derivation, raised per-blueprint SSE deadline, and secret redaction in annotation + SSE stream.
- [ ] **Phase 30: Wizard Stepper & Live Progress** - Multi-step `welcome.html` form replacing the hard-block, with per-stage progress and chaining to cluster-init.
- [ ] **Phase 31: End-to-End Verification** - Fresh-cluster full install, failure/retry, idempotent re-run, and secret-leak gate.

## Phase Details

### Phase 27: Install Blueprint Authoring
**Goal**: A single parameterized `WekaAppStore` CR exists that, when rendered and applied, installs the entire WEKA storage stack in the correct dependency order with no hand-encoded secrets.
**Depends on**: Nothing (first phase of this milestone; the declarative contract every later phase targets)
**Requirements**: INST-01, INST-02, INST-03, INST-04, INST-05, INST-06, INST-07, INST-08, INST-09, INST-10
**Success Criteria** (what must be TRUE):
  1. `cluster_init/app-store-install.yaml` renders to valid YAML via the GUI `[[ var ]]` path with sample variables, and the operator's `resolve_dependencies` topo-sorts the components into the order quay-secrets → operator → (node-label-job / weka-client-secret / csi-wekafs) → weka-client / csi-api-secret → storageclasses.
  2. Every wizard-generated Secret in the blueprint uses `stringData` (no hand-`base64`), and the quay `dockerconfigjson` is injected as one `data` variable — no committed `data:` field decodes with a trailing newline (round-trip verified).
  3. The `weka-client` (WekaClient CR) component declares `dependsOn: [weka-operator, weka-client-secret]` and the `weka-operator` component carries `readinessCheck: {type: deployment}`, so the WekaClient apply cannot 404 on a missing CRD; the WEKA operator installs from `oci://quay.io/weka.io/helm/weka-operator` at `[[ operator_version ]]` and CSI from the public `csi-wekafs` repo into the `csi-wekafs` namespace.
  4. The three StorageClasses share `secretName`/`secretNamespace` matching the `csi-wekafs-api-secret`, `storageclass-wekafs-dir-api` is the cluster default, and a brownfield existing-default is demoted or skipped rather than creating two defaults.
  5. The node-label Job uses `kubectl label nodes --all weka.io/supports-clients=true --overwrite` and is a no-op on re-run, and an `x-variables` block maps 1:1 to the wizard form fields (GUI-derived vars excluded from validation).
**Plans**: TBD

### Phase 28: Operator Helm Auth & CRD Discovery
**Goal**: The operator pod can pull and install the quay OCI operator chart on a fresh cluster that has only the in-namespace pull secrets, and reliably discovers the chart's CRDs.
**Depends on**: Phase 27 (the blueprint defines the chart ref and version the operator must authenticate to pull)
**Requirements**: OPA-01, OPA-02
**Success Criteria** (what must be TRUE):
  1. On a fresh cluster with only the quay `dockerconfigjson` secrets present (no host/operator prior `helm registry login`), the operator's `helm show crds` and `helm install oci://quay.io/weka.io/helm/weka-operator` succeed without a quay `401`.
  2. The operator materializes quay auth (`helm registry login` or `--registry-config` from the quay credentials) before every `oci://quay.io/...` chart operation, and the quay password never appears in process args or operator logs.
  3. `discover_chart_crds` no longer memoizes an empty-CRD result on fetch failure — a transient/auth blip does not permanently cache "no CRDs," so a fixed-auth retry installs the WEKA CRDs without an operator pod restart.
  4. After operator install, the WEKA CRDs (`weka.weka.io`) are registered and a subsequent `WekaClient` apply does not 404.
**Plans**: TBD

### Phase 29: Backend Wiring & Secret Safety
**Goal**: The backend locates and serves the new blueprint, derives the multi-sink variables server-side, survives a long install without false-failing, and never leaks secret values.
**Depends on**: Phase 27 (blueprint to locate and render), Phase 28 (operator auth proven, so the raised deadline targets a real install path)
**Requirements**: PROG-02, SEC-01
**Success Criteria** (what must be TRUE):
  1. `/deploy-stream?app_name=app-store-install` locates the blueprint, preserves the components' fixed namespaces (namespace-preserve special-case extended from cluster-init), and renders to valid YAML.
  2. The server builds `quay_dockerconfigjson` from posted quay creds such that `auths["quay.io"]["auth"]` decodes to exactly `user:pass` with no trailing bytes, and `split_endpoints` produces both the `joinIpPorts` YAML-list and comma-joined `endpoints` forms — both verified by unit tests.
  3. The SSE deadline is raised (per-blueprint as needed) and keepalive/reconnect is robust, so a long operator+CSI+WekaClient install does not surface a false "timed out" while the CR is still progressing.
  4. Quay and WEKA secret values never appear in the `warp.io/gui-variables` CR annotation (variable allowlist excludes `*password*`/`*token*`/`*secret*`/`quay_dockerconfigjson`) nor in emitted SSE component messages (redaction).
**Plans**: TBD

### Phase 30: Wizard Stepper & Live Progress
**Goal**: A customer completes a multi-step web form and watches the storage stack install stage-by-stage, then is redirected to the App Store after cluster-init.
**Depends on**: Phase 27 (blueprint + x-variables shape), Phase 29 (backend derivation, deadline, redaction)
**Requirements**: WIZ-01, WIZ-02, WIZ-03, WIZ-04, WIZ-05, WIZ-06, WIZ-07, WIZ-08, PROG-01, PROG-03
**Success Criteria** (what must be TRUE):
  1. The customer progresses through a multi-step form (node prerequisites → quay credentials → WEKA connection → WEKA credentials → review) with inline validation (required fields, `host:port` endpoint format, version tags) blocking an invalid forward step, and the old single-button prerequisite hard-block is gone — the wizard installs the operator/CSI instead of requiring them present.
  2. Step 1 shows the required `KubeletConfiguration` (CPU manager `static`, `strict-cpu-reservation`) + hugepage snippet as copy-paste text gated behind an "I have applied node prerequisites" checkbox; the App Store never writes node config or restarts kubelet.
  3. The customer enters quay credentials (username, masked password, operator version default `v1.13.0`), WEKA connection (one or more `host:port` endpoints, image version tag, `http`/`https` scheme dropdown), and WEKA credentials (org default `Root`, username, masked password); the review step shows a masked summary plus a namespace selector before submit.
  4. The install view shows each stage transitioning Pending → In-progress → Done/Failed driven by the existing `componentStatus` SSE events; on a stage failure the customer sees a clear, specific error and can retry the install.
  5. After `app-store-install` reaches `Ready` the wizard chains the untouched `app-store-cluster-init` CR as the final stage and redirects to the App Store when cluster-init reaches `Ready`.
**Plans**: TBD
**UI hint**: yes

### Phase 31: End-to-End Verification
**Goal**: The full chained-CR install is proven against the PRD success criteria on a real cluster — first install, failure recovery, idempotent re-run, and no secret leakage.
**Depends on**: Phase 30 (the complete wizard path), and transitively all prior phases
**Requirements**: E2E-01, E2E-02, E2E-03, SEC-02
**Success Criteria** (what must be TRUE):
  1. On a cluster with no WEKA operator/CSI, completing the wizard installs the operator, CSI driver, WekaClient, all secrets, and three StorageClasses, with exactly one default StorageClass (`storageclass-wekafs-dir-api`).
  2. A forced stage failure surfaces a clear error and the customer-triggered retry resumes successfully (apply-or-patch is non-destructive).
  3. Re-running the completed wizard on an already- or partially-installed cluster is idempotent — the node-label Job is a no-op, secrets/CRs re-patch cleanly, and no duplicate or destructive action occurs.
  4. No secret values appear in the browser SSE log box, the operator pod logs, or `kubectl get wekaappstore -o yaml` (verifies SEC-01).
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 27. Install Blueprint Authoring | 0/0 | Not started | - |
| 28. Operator Helm Auth & CRD Discovery | 0/0 | Not started | - |
| 29. Backend Wiring & Secret Safety | 0/0 | Not started | - |
| 30. Wizard Stepper & Live Progress | 0/0 | Not started | - |
| 31. End-to-End Verification | 0/0 | Not started | - |

---
*Roadmap created: 2026-06-24*
*Coverage: 28/28 v1 requirements mapped*
