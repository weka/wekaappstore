# Requirements: v8.0 Guided Install Wizard — WEKA Operator, CSI & Storage Classes

**Defined:** 2026-06-24
**Core Value:** A customer can stand up the full WEKA storage stack (operator, CSI driver, client, storage classes) from the App Store install wizard by answering a short form — no manual `kubectl`/`helm`/base64 work.
**Authoritative spec:** `.planning/PRD-install-wizard-weka-storage-stack.md`

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Install Blueprint (INST) — parameterized AppStack that installs the WEKA storage stack

- [ ] **INST-01**: A parameterized `cluster_init/app-store-install.yaml` `WekaAppStore` blueprint with an `x-variables` block encodes the full ordered install via `appStack.components[]` (`dependsOn` + `readinessCheck`).
- [ ] **INST-02**: quay `dockerconfigjson` pull secrets are created in `weka-operator-system` and `default` from the GUI-built payload.
- [ ] **INST-03**: The WEKA Operator and its CRDs are installed from `oci://quay.io/weka.io/helm/weka-operator` at the customer-chosen version, gated by operator-deployment readiness.
- [ ] **INST-04**: All cluster nodes are labeled `weka.io/supports-clients=true` via an idempotent Job.
- [ ] **INST-05**: A `WekaClient` CR and its `weka-client-cluster-dev` secret are created (joinIpPorts list + image tag), gated behind operator CRD + deployment readiness so the apply cannot 404 on a missing CRD.
- [ ] **INST-06**: The WEKA CSI driver is installed from the public `csi-wekafs` chart into the `csi-wekafs` namespace.
- [ ] **INST-07**: The `csi-wekafs-api-secret` is created with `endpoints` (comma-joined), `scheme`, `organization`, `username`, and `password`.
- [ ] **INST-08**: Three StorageClasses are created with consistent `secretName`/`secretNamespace`, and `storageclass-wekafs-dir-api` is set as the cluster default — with detection that demotes or skips when a default StorageClass already exists.
- [ ] **INST-09**: All wizard-generated Secrets use `stringData` (no hand-base64), eliminating the trailing-newline encoding bug class.
- [ ] **INST-10**: After `app-store-install` reaches `Ready`, the existing (untouched) `app-store-cluster-init` CR runs as the final stage; the customer is redirected to the App Store when cluster-init reaches `Ready`.

### Operator Helm Auth (OPA) — operator-side changes for authenticated quay OCI pulls

- [x] **OPA-01**: The operator authenticates to quay before pulling the operator OCI chart (`helm registry login` / `--registry-config`) using the supplied quay credentials, so `helm show crds` / `helm install oci://quay.io/...` succeed from inside the operator pod.
- [x] **OPA-02**: `discover_chart_crds` no longer memoizes an empty-CRD failure result, so operator CRDs are discovered reliably (preventing a later WekaClient 404).

### Wizard Form & UX (WIZ) — multi-step `welcome.html` form

- [x] **WIZ-01**: The customer progresses through a multi-step form (node prerequisites → quay credentials → WEKA connection → WEKA credentials → review) before install begins.
- [x] **WIZ-02**: Step 1 shows the required `KubeletConfiguration` (CPU manager `static`, `strict-cpu-reservation`) and hugepage snippet as copy-paste text, gated behind an "I have applied node prerequisites" checkbox; the App Store never writes node config or restarts kubelet (Decision A1).
- [x] **WIZ-03**: The customer enters quay registry username, password (masked), and operator chart version (default `v1.13.0`).
- [x] **WIZ-04**: The customer enters WEKA connection details: join endpoints (one or more `host:port`), WEKA container image version tag, and API scheme via an `http`/`https` dropdown.
- [x] **WIZ-05**: The customer enters WEKA cluster credentials: organization (default `Root`), username, and password (masked).
- [x] **WIZ-06**: A review step shows a summary with secrets masked plus a namespace selector before the customer submits.
- [x] **WIZ-07**: Inputs are validated before submit (required fields, `host:port` endpoint format, version tags), with clear inline errors.
- [x] **WIZ-08**: The old single-button prerequisite hard-block is removed; the wizard installs the operator/CSI instead of requiring them already present.

### Progress & Streaming (PROG)

- [x] **PROG-01**: The install view shows each install stage transitioning Pending → In-progress → Done/Failed, driven by the existing `componentStatus` SSE events.
- [x] **PROG-02**: The deploy SSE deadline is raised (per-blueprint as needed) and keepalive/reconnect is robust, so a long operator+CSI+WekaClient install does not false-fail on a still-healthy deployment.
- [ ] **PROG-03**: On a stage failure the customer sees a clear, specific error and can retry the install.

### Secret Safety & Idempotency (SEC)

- [ ] **SEC-01**: Quay and WEKA secret values never appear in the CR `warp.io/gui-variables` annotation (variable allowlist) nor in the SSE log stream (message redaction).
- [ ] **SEC-02**: Re-running the wizard on a partially- or fully-installed cluster is non-destructive (apply-or-patch resources, idempotent label Job).

### End-to-End Verification (E2E)

- [ ] **E2E-01**: On a cluster with no WEKA operator/CSI, completing the wizard results in the operator, CSI driver, WekaClient, all secrets, and three StorageClasses installed, with `storageclass-wekafs-dir-api` marked default.
- [ ] **E2E-02**: The failure→retry path works, and a wizard re-run on an already-installed cluster is idempotent.
- [ ] **E2E-03**: No secret values appear in logs or the SSE stream (verifies SEC-01).

## v2 Requirements

Deferred to a future milestone. Tracked, not in this roadmap.

### Node Automation (NODE)

- **NODE-01**: Optionally auto-apply worker-node kubelet/hugepage config via a privileged DaemonSet (the A2 option deferred in favor of A1's documented prerequisite).

### Day-2 Operations (DAY2)

- **DAY2-01**: Upgrade or uninstall the WEKA operator / CSI driver from the GUI.
- **DAY2-02**: Rotate WEKA or quay credentials post-install.

## Out of Scope

Explicitly excluded for v8.0. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Auto-restarting kubelet on worker nodes | Disruptive and distro-specific; Decision A1 keeps node config a documented, customer-applied prerequisite |
| Changing the cluster-init component set (`app-store-cluster-init.yaml`) | Runs as-is, just later (Decision D) — out of scope to modify |
| WEKA backend cluster provisioning | We connect *clients* to an existing NeuralMesh cluster; we do not create the storage cluster |
| Multi-cluster / multiple WEKA backends in one wizard run | Single-target for v8.0 |
| Non-quay / air-gapped operator image sources | Only the quay OCI path is supported in v8.0 |
| Free-text / advanced raw-YAML overrides in the wizard | Keeps the form bounded and validated |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 27 | Pending |
| INST-02 | Phase 27 | Pending |
| INST-03 | Phase 27 | Pending |
| INST-04 | Phase 27 | Pending |
| INST-05 | Phase 27 | Pending |
| INST-06 | Phase 27 | Pending |
| INST-07 | Phase 27 | Pending |
| INST-08 | Phase 27 | Pending |
| INST-09 | Phase 27 | Pending |
| INST-10 | Phase 27 | Pending |
| OPA-01 | Phase 28 | Complete |
| OPA-02 | Phase 28 | Complete |
| PROG-02 | Phase 29 | Complete |
| SEC-01 | Phase 29 | Pending |
| WIZ-01 | Phase 30 | Complete |
| WIZ-02 | Phase 30 | Complete |
| WIZ-03 | Phase 30 | Complete |
| WIZ-04 | Phase 30 | Complete |
| WIZ-05 | Phase 30 | Complete |
| WIZ-06 | Phase 30 | Complete |
| WIZ-07 | Phase 30 | Complete |
| WIZ-08 | Phase 30 | Complete |
| PROG-01 | Phase 30 | Complete |
| PROG-03 | Phase 30 | Pending |
| E2E-01 | Phase 31 | Pending |
| E2E-02 | Phase 31 | Pending |
| E2E-03 | Phase 31 | Pending |
| SEC-02 | Phase 31 | Pending |

**Coverage:**
- v1 requirements: 28 total (INST 10, OPA 2, WIZ 8, PROG 3, SEC 2, E2E 3)
- Mapped to phases: 28/28 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-24*
*Last updated: 2026-06-24 after roadmap creation (phases 27–31; 28/28 mapped)*
