# Phase 27: Install Blueprint Authoring - Context

**Gathered:** 2026-06-24 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Author `cluster_init/app-store-install.yaml` — a single parameterized `WekaAppStore` CR that installs the entire WEKA storage stack (quay pull secrets → operator → node-labels / WekaClient secret / CSI driver → WekaClient CR / CSI API secret → StorageClasses) in the correct dependency order using `stringData` secrets and `x-variables` substitution. This is the declarative contract that every later phase targets. No frontend, no operator changes, no backend wiring in this phase.
</domain>

<decisions>
## Implementation Decisions

### Component Ordering & Dependency Graph

- **D-01:** The topo-sort order is: `quay-secret-operator-ns` + `quay-secret-default-ns` (parallel, no deps) → `weka-operator` (depends on both quay secrets, `readinessCheck: {type: deployment}`) → parallel group: `weka-node-label-sa`, `weka-client-secret`, `csi-wekafs` (all depend on `weka-operator`) → `weka-node-label-rbac` (depends on `weka-node-label-sa`) → `weka-node-label-job` (depends on `weka-node-label-rbac`) → parallel: `csi-api-secret` (depends on `csi-wekafs`) + `weka-client` (depends on `weka-operator` + `weka-client-secret`) → `storageclass-demote-job` (depends on `csi-api-secret`) → `storageclasses` (depends on `storageclass-demote-job`).

- **D-02:** The `weka-operator` component uses `helmChart.repository: "oci://quay.io/weka.io/helm"`, `helmChart.name: "weka-operator"`, `helmChart.releaseName: "weka-operator"`, `helmChart.version: "[[ operator_version ]]"`, `targetNamespace: weka-operator-system`. The OCI path `repository/name` is assembled by the operator at `main.py:1113` — confirm this splits correctly as `oci://quay.io/weka.io/helm/weka-operator`.

- **D-03:** The `weka-client` (WekaClient CR) component uses `kubernetesManifest` sourced from `wekaClientCR-online.yaml` with `[[ join_ip_ports_list ]]` (YAML-array form, produced server-side in Phase 29) for `joinIpPorts` and `[[ weka_image_version ]]` for the image tag. It declares `dependsOn: [weka-operator, weka-client-secret]` and has NO `readinessCheck` (the WekaClient operator manages its own reconcile loop; we cannot wait on it here).

- **D-04:** Two separate `quay-pull-secret` components — one in `weka-operator-system`, one in `default` — each using `type: kubernetes.io/dockerconfigjson` and `stringData: {".dockerconfigjson": "[[ quay_dockerconfigjson ]]"}`. One var, two Secret manifests in the same `kubernetesManifest` block (separated by `---`) is fine since the operator's `apply_gateway` splits on document boundaries.

### StorageClass Default Handling (INST-08)

- **D-05:** Brownfield default detection is handled by a `storageclass-demote-job` component (Job + dedicated SA + ClusterRole with `get/list/patch` on `storageclasses`). The Job script: check for existing `is-default-class: "true"` annotation and patch it to `"false"` before the StorageClass manifests component runs. The `storageclasses` component (three StorageClass docs in one manifest block) depends on `storageclass-demote-job` with `readinessCheck: {type: job}` on the demote job. The three StorageClasses are authored with `storageclass-wekafs-dir-api` carrying `storageclass.kubernetes.io/is-default-class: "true"`.

### x-variables Block Design

- **D-06:** The `x-variables` block exposes exactly these keys (all required unless defaulted):
  - `operator_version` — default `v1.13.0` (v-prefix confirmed)
  - `weka_image_version` — no default, required
  - `join_ip_ports` — comma-delimited `host:port` string entered by user; server-side (Phase 29) splits into `join_ip_ports_list` (YAML-array `["h:p", ...]`) and `endpoints_csv` (comma-joined for CSI secret)
  - `weka_endpoint_scheme` — default `http`
  - `weka_org` — default `Root`
  - `weka_username` — required
  - `weka_password` — required
  - `quay_dockerconfigjson` — `validate: false`; GUI-built base64 JSON, never user-typed; single token that renders into `stringData[".dockerconfigjson"]`

  Server-side derived vars (`join_ip_ports_list`, `endpoints_csv`) are NOT in `x-variables` — they are injected as extra render vars by Phase 29 backend before the Jinja2 pass.

### Node Label Job Service Account

- **D-07:** Three separate components in sequence: `weka-node-label-sa` (ServiceAccount in `kube-system`) → `weka-node-label-rbac` (ClusterRoleBinding using `cluster-admin`, matching the `gateway-api-crds` precedent in `app-store-cluster-init.yaml`) → `weka-node-label-job` (Job running `kubectl label nodes --all weka.io/supports-clients=true --overwrite`, `readinessCheck: {type: job}`). All three depend on `weka-operator` to ensure they run after operator install stage is complete.

### CSI Driver Helm Chart Reference

- **D-08:** CSI driver component: `helmChart.repository: "https://weka.github.io/csi-wekafs"`, `helmChart.name: "csi-wekafsplugin"`, `helmChart.releaseName: "csi-wekafs"`, `targetNamespace: csi-wekafs`. No `helmChart.version` pinned in Phase 27 (leave unpinned or use latest stable `2.8.7` — researcher found index has versions 0.6.2–2.8.7). No auth needed (public repo).

### stringData & Secret Encoding

- **D-09:** ALL wizard-generated Secrets in the blueprint use `stringData` — no `data:` fields with pre-encoded values. The one exception is `quay_dockerconfigjson` which is already a fully-formed JSON string injected into `stringData[".dockerconfigjson"]` (not double-encoded). The existing `weka-csi-config/*.yaml` files with `data:` base64 are reference-only (for understanding field shapes); the new blueprint uses `stringData` throughout.

### Claude's Discretion

- Whether the weka-operator and CSI components carry `waitForReady: true` (likely yes, matching cluster-init pattern) vs relying solely on `readinessCheck`.
- Exact operator chart values (node tolerations, resource limits) — leave as defaults; no `valuesFiles` block in Phase 27 unless required by the chart.
- CSI driver chart version pinning — can be left unpinned for Phase 27 (pinned in a follow-on).
- Whether `csi-api-secret` and `weka-client` components should be in the same parallel group or sequenced — either works as long as `storageclass-demote-job` depends on `csi-api-secret`.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `.planning/PRD-install-wizard-weka-storage-stack.md` — authoritative spec with resolved Decisions A–E
- `.planning/ROADMAP.md` — Phase 27 success criteria (5 items), component ordering in SC1
- `.planning/REQUIREMENTS.md` — INST-01..INST-10 acceptance criteria
- `cluster_init/app-store-cluster-init.yaml` — canonical blueprint pattern (SA+CRB+Job idiom, helmChart field shape, readinessCheck syntax, kubernetesManifest multi-doc separator)
- `weka-csi-config/wekaClientCR-online.yaml` — WekaClient CR field shape (joinIpPorts, image, imagePullSecret, nodeSelector, wekaSecretRef)
- `weka-csi-config/weka-client-cluster-dev.yaml` — WekaClient secret field shape (org, username, password)
- `weka-csi-config/csi-wekafs-api-secret.yaml` — CSI API secret field shape (username, password, organization, endpoints, scheme)
- `weka-csi-config/storageclass-wekafs-dir-api.yaml` — default StorageClass shape + `is-default-class` annotation
- `weka-csi-config/storageclass-wekafs-dir-api-retain.yaml` — Retain policy StorageClass shape
- `weka-csi-config/storageclass-wekafs-fs-api.yaml` — FS StorageClass shape
- `weka-csi-config/quay-secret.yaml` — quay dockerconfigjson Secret field shape
- `operator_module/main.py` — `resolve_dependencies` (Kahn's topo-sort, ~line 824), `discover_chart_crds` (OCI path assembly, ~line 675), `handle_helm_install` (OCI vs HTTPS chart logic, ~line 1113)
- `app-store-gui/webapp/main.py` — `parse_x_variables` (~line 1694), `find_blueprint` (~line 1801), `/deploy-stream` endpoint (~line 2839)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `cluster_init/app-store-cluster-init.yaml` — full blueprint pattern with SA+CRB+Job, helmChart (HTTPS repo), readinessCheck pod/deployment/job, multi-doc kubernetesManifest, `waitForReady: true` conventions
- `weka-csi-config/*.yaml` — six static YAML files that become parameterized blueprint component bodies; fields are the authoritative source of truth for field names
- `operator_module/main.py` `readinessCheck` dispatcher — supports `type: pod`, `type: deployment`, `type: job`; `type: deployment` waits for `available >= desired` replicas
- `app-store-gui/webapp/main.py` `parse_x_variables` — reads top-level `x-variables` dict from blueprint, returns `{}` on all failure paths; used by the deploy-stream for required-field gating

### Established Patterns

- `[[var]]` Jinja2 delimiters — GUI rendering layer; `${VAR}` is operator-side only
- SA + ClusterRoleBinding + Job triple-component sequence (lines 130-226 of `app-store-cluster-init.yaml`) — the canonical way to run a privileged idempotent `kubectl` operation inside an AppStack
- `dependsOn: []` — empty list means no deps (runs immediately); omit key or use empty list equivalently
- `targetNamespace` on a component overrides the CR's metadata namespace for that component's apply
- Multi-doc `kubernetesManifest` (separated by `---`) — the `apply_gateway` splits on document boundaries; both docs in one component share the same `dependsOn`/`readinessCheck`

### Integration Points

- `find_blueprint` in `main.py` (~line 1801) needs to locate `cluster_init/app-store-install.yaml` — Phase 29 will wire this; Phase 27 just authors the file in the right location
- `deploy-stream` renders the blueprint with `[[ var ]]` Jinja2 substitution before applying — the blueprint must be valid YAML after substitution with sample values
- `ClusterInitMiddleware` watches the `app-store-cluster-init` CR name specifically — unchanged; chaining is a Phase 30/frontend concern
</code_context>

<specifics>
## Specific Ideas

- The `quay_dockerconfigjson` variable is the pre-built JSON string that goes into `stringData[".dockerconfigjson"]` — the template looks like:
  ```yaml
  stringData:
    .dockerconfigjson: "[[ quay_dockerconfigjson ]]"
  ```
  The GUI builds this in Phase 29 by base64-encoding `user:pass` and wrapping it in the auths JSON — the blueprint author does not need to implement the builder, just use the token.

- CSI chart version: researcher confirmed `2.8.7` is the latest stable. Phase 27 blueprint can either pin to `2.8.7` or omit the version (helm defaults to latest). Pinning is safer for reproducibility — recommend pinning to `2.8.7` with a comment to update.

- Operator version default `v1.13.0` (v-prefix confirmed as correct). The `x-variables` block should set `default: v1.13.0` for `operator_version`.
</specifics>

<deferred>
## Deferred Ideas

None — analysis stayed within Phase 27 scope.

Operator-side helm registry login (needed for authenticated quay OCI pulls) is scoped to Phase 28, not here. The Phase 27 blueprint simply references `oci://quay.io/weka.io/helm/weka-operator` — the auth mechanism is Phase 28's responsibility.
</deferred>
