# Stack Research — v8.0 Guided Install Wizard (WEKA Storage Stack)

**Domain:** Brownfield Kubernetes app (WEKA App Store) — fold the WEKA Operator + CSI + StorageClass install into the existing AppStack/SSE wizard
**Researched:** 2026-06-24
**Confidence:** HIGH (chart coordinates, CRD groups, and integration points verified against the repo + WEKA docs / Artifact Hub)

> **Scope note:** This is a SUBSEQUENT milestone on an existing system. The AppStack engine, OCI helm install, `helm show crds` discovery, `dockerconfigjson` builder, `[[ var ]]` Jinja2 substitution, `/deploy-stream` SSE, and the kubernetes Python apply gateway **already exist and work**. This file documents only what is NEW or version-pinned for v8.0, the exact K8s/Helm objects involved, and an explicit do-NOT-add list. There is **no language/package-manager dependency to add** — see "What NOT to Use."

---

## Recommended Stack

### Core Technologies (external chart/image coordinates the wizard installs)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| WEKA Operator Helm chart | `v1.13.0` (PRD default; field-overridable via `[[ operator_version ]]`) | Installs the operator Deployment + the 6 `weka.weka.io` CRDs that make `WekaClient` a valid kind | OCI ref `oci://quay.io/weka.io/helm/weka-operator`. The operator already takes the OCI path (skips `helm repo add`, `main.py:118`) and discovers CRDs via `helm show crds` (`main.py:679`). Bundled copy in-repo is `v1.9.1` (`weka-csi-config/weka-operator/Chart.yaml`) — **stale, do not rely on it; install the pinned version from quay at runtime.** |
| WEKA-in-container image | tag only, e.g. `5.1.0.605` (field `[[ weka_image_version ]]`) | The WEKA client runtime image referenced by the `WekaClient` CR (`image: quay.io/weka.io/weka-in-container:<tag>`) | Customer must match the tag to their WEKA backend NeuralMesh version; pulled from quay using the same `quay-io-secret`. Wizard collects only the tag and concatenates. |
| WEKA CSI driver Helm chart | `csi-wekafsplugin` `2.8.x` (latest verified `2.8.7` on Artifact Hub) | Installs `csi-wekafs-controller` (Deployment) + node DaemonSet; provides `provisioner: csi.weka.io` | Public chart, repo `https://weka.github.io/csi-wekafs` (Artifact Hub repo id `csi-wekafs/csi-wekafsplugin`). **No auth** — do not attach a pull secret. Since 0.7.0 it is API-based: StorageClasses must carry the `csi.storage.k8s.io/*-secret-name/-namespace` params (already present in the `weka-csi-config/storageclass-*.yaml` files). Pin a `version:` in the helmChart component for reproducibility. |

### Kubernetes API objects the wizard creates (the real "stack" — all standard kinds, no new CRDs authored)

| Object | apiVersion / kind | Source template (to parameterize) | Notes |
|--------|-------------------|-----------------------------------|-------|
| Quay pull secret ×2 | `v1` `Secret`, `type: kubernetes.io/dockerconfigjson` | `weka-csi-config/quay-secret.yaml` | One in `weka-operator-system`, one in `default`. GUI builds the `.dockerconfigjson` (Decision B) and injects as `[[ quay_dockerconfigjson ]]`. Covers BOTH chart pull and image pull → that is why no `helm registry login` is needed (Decision C). |
| WEKA Operator | (helm release) | — | AppStack `helmChart` OCI component; `readinessCheck: deployment` in `weka-operator-system`. Installs the 6 CRDs below. |
| Operator CRDs (×6) | `apiextensions.k8s.io/v1` `CustomResourceDefinition` | bundled at `weka-csi-config/weka-operator/crds/` (reference only) | `wekaclients`, `wekaclusters`, `wekacontainers`, `driveclaims`, `wekapolicies`, `wekamanualoperations` — all group `weka.weka.io`. Installed by the operator chart via `helm show crds`; the `WekaClient` apply MUST be gated behind these existing. |
| Node label job | `batch/v1` `Job` (+ `ServiceAccount` + `ClusterRoleBinding` to `cluster-admin`) | new — precedent `gateway-api-crds-job` in `app-store-cluster-init.yaml:131-226` | `kubectl label nodes --all weka.io/supports-clients=true`. Must be idempotent / no-op on re-run. `readinessCheck: job`. |
| WekaClient | `weka.weka.io/v1alpha1` `WekaClient` | `weka-csi-config/wekaClientCR-online.yaml` | Fields: `image`, `imagePullSecret: quay-io-secret`, `joinIpPorts: ["host:port", ...]` (YAML list form), `nodeSelector: {weka.io/supports-clients: "true"}`, `wekaSecretRef: weka-client-cluster-dev`, `port`, `agentPort`, `network.udpMode`. **`dependsOn` operator + `readinessCheck` on operator deployment**, or the apply 404s on a missing CRD. |
| WekaClient secret | `v1` `Secret` (use `stringData`) | `weka-csi-config/weka-client-cluster-dev.yaml` | Keys `org`, `username`, `password`. Referenced by `wekaSecretRef`. |
| CSI API secret | `v1` `Secret` (use `stringData`) | `weka-csi-config/csi-wekafs-api-secret.yaml` | Keys `username`, `password`, `organization`, `endpoints` (comma-joined `host:port` string), `scheme` (`http`/`https`). In ns `csi-wekafs`. Consumed by all StorageClasses. |
| StorageClasses (×3) | `storage.k8s.io/v1` `StorageClass` | `storageclass-wekafs-dir-api.yaml` (default), `-dir-api-retain.yaml`, `-fs-api.yaml` | `provisioner: csi.weka.io`; `is-default-class: "true"` only on dir-api. `*-secret-name=csi-wekafs-api-secret` / `*-secret-namespace=csi-wekafs` must match the CSI API secret. |

### Existing runtime capabilities reused (DO NOT re-build)

| Capability | Where | Used for |
|-----------|-------|----------|
| OCI helm install (skips `helm repo add`) | `operator_module/main.py:118` | Operator chart from quay |
| `helm show crds` CRD discovery | `operator_module/main.py:679` `discover_chart_crds` | Operator CRDs |
| `dockerconfigjson` build pattern + `_b64` (standard padded) | `operator_module/main.py:476-509`, `:1719` | Quay pull secret (Decision B) — GUI mirrors this logic |
| `dependsOn` + `readinessCheck` (pod/deployment/job) | AppStack engine | Ordering operator→label→client→csi→SCs |
| `[[ var ]]` Jinja2 substitution + `x-variables` validation | `parse_x_variables`, `/deploy-stream` `main.py:2839` | New blueprint variables |
| `find_blueprint` special-case | `main.py:1801-1807` | Add `app-store-install` next to the `cluster-init` special case |
| kubernetes Python apply gateway (create→409→patch, cluster-scoped kinds incl. StorageClass) | `planning/apply_gateway.py` | Idempotent re-run safety |

## Installation

```bash
# NOTHING to npm/pip install. All "installation" is runtime YAML the wizard renders.

# Operator (runtime, executed by the operator via the AppStack helmChart component):
#   chart: oci://quay.io/weka.io/helm/weka-operator   version: v1.13.0   ns: weka-operator-system
#   (pull auth via quay-io-secret dockerconfigjson in-namespace — NOT `helm registry login`)

# CSI driver (runtime, AppStack helmChart component):
#   repo: https://weka.github.io/csi-wekafs   name: csi-wekafsplugin   version: 2.8.x   ns: csi-wekafs
#   (public — no pull secret)

# New files to author this milestone:
#   cluster_init/app-store-install.yaml        # parameterized WekaAppStore CR + x-variables
#   parameterized weka-csi-config/* templates  # stringData secrets, [[ var ]] tokens
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| OCI helm install + in-namespace `dockerconfigjson` secret for quay auth (Decision C) | `helm registry login quay.io` before install | Never for this milestone — the operator runs helm via subprocess in its own pod; a login there would not persist nor cover image pulls. The dockerconfigjson approach covers both chart and image pulls. Avoid. |
| Single parameterized `WekaAppStore` CR + chained cluster-init CR (Decision D) | One mega-CR including cluster-init | Never — PRD locks cluster-init untouched and run last as a separate CR; merging risks regressing existing behavior. |
| `stringData` secrets from form input | Pre-base64 `data:` fields | Never — pre-encoding by hand caused the trailing-newline bug (`echo` vs `printf`) already fixed 2026-06-24. `stringData` lets K8s encode and eliminates the bug class. |
| Documented manual kubelet/hugepage prereq + confirm checkbox (Decision A1) | Privileged DaemonSet that writes `KubeletConfiguration` and restarts kubelet | Out-of-scope follow-on. Node-root config + kubelet restart is too risky for an in-cluster app to own automatically this milestone. |
| Pin operator `version: v1.13.0` (field-overridable) | Track `latest` from quay | Never — reproducible installs require a pinned chart version; bundled `v1.9.1` in-repo is stale and must not be the source of truth. |
| Job running `kubectl label nodes --all` | Operator-managed node labeling | The Job mirrors the proven `gateway-api-crds-job` precedent (cluster-admin SA) and is idempotent; no operator change needed. |

## Worker-node prerequisite (documented, NOT installed — Decision A1)

Step 1 of the wizard shows a copy-paste `KubeletConfiguration` snippet gated behind an "I have applied node prerequisites" checkbox. The App Store does **not** write node config or restart kubelet.

| Setting | Value | Source confidence |
|---------|-------|-------------------|
| `cpuManagerPolicy` | `static` | HIGH — WEKA EKS client docs |
| `cpuManagerPolicyOptions: strict-cpu-reservation` | `"true"` | MEDIUM — named in PRD; WEKA docs confirm static policy + reserved CPU, exact option string per WEKA client docs / support |
| `systemReserved.cpu` | reserved core(s) (docs show `"1"`) | HIGH — WEKA EKS docs |
| hugepages | `25000` (per PRD; docs phrase as ~1.5 GiB for client core) | MEDIUM — PRD-specified count; WEKA docs give the rationale, exact count is deployment/core-count dependent |
| node label | `weka.io/supports-clients=true` | HIGH — verified in WekaClient CR + WEKA docs |

> Flag for requirements: the exact `strict-cpu-reservation` option key and the `25000` hugepage count are the two LOWest-confidence values. Treat the PRD as authoritative for the wizard's displayed snippet. Because the snippet is informational only (customer applies it on their nodes), a slightly off default is low-blast-radius — the customer adapts it to their node sizing.

## What NOT to Use / NOT to Add

- **No new build step.** `welcome.html` stays CDN-React (MUI + Tailwind via unpkg), no bundler/transpile. The stepper is plain `useState` step state or MUI `Stepper` — same pattern as the v4.0 single-React-root approach. (Locked by milestone context + repo CLAUDE.md "no build step".)
- **No `helm registry login` and no operator auth change** (Decision C). The in-namespace quay `dockerconfigjson` secrets cover chart + image pulls. The OCI path at `main.py:118` is used as-is.
- **No new Python/JS package dependency.** Everything reuses FastAPI/Jinja, the kubernetes client, kopf, the existing `_b64`/dockerconfigjson helpers, and the SSE stream.
- **No new CRD authored.** `WekaClient`/`WekaCluster`/etc. CRDs come FROM the WEKA operator chart; the App Store's only CRD remains `WekaAppStore` (`warp.io/v1alpha1`). Do not vendor or hand-maintain the 6 `weka.weka.io` CRDs (the bundled `weka-csi-config/weka-operator/crds/` is reference only — stale at v1.9.1).
- **Do NOT trust the in-repo bundled operator chart version (`v1.9.1`).** Install the pinned `v1.13.0` (overridable) from quay at runtime.
- **No pull secret for the CSI chart** — it is a public GitHub-Pages repo.
- **No air-gapped / non-quay operator registry support** (PRD non-goal). The dockerconfigjson hardcodes the `quay.io` auth host.
- **No Helm SDK.** The operator shells out to `helm`/`kubectl` by design — match that; do not introduce a Python Helm binding.
- **No day-2 ops** (upgrade/uninstall/rotate) — PRD non-goal.

## Integration Points (for the roadmapper)

1. **New blueprint** `cluster_init/app-store-install.yaml` — parameterized `WekaAppStore` CR with `x-variables` and `[[ var ]]` tokens; ordered `components[]` per PRD §"Install Sequence". This is the bulk of the work.
2. **`find_blueprint` (`main.py:1801`)** — add `app-store-install` alongside the `cluster-init` special case (or rely on `x-variables` discovery).
3. **Derived-variable helper** — parse the single endpoints field into BOTH the YAML-list form (`["host:port", ...]` for WekaClient `joinIpPorts`) and the comma-joined string (`csi-wekafs-api-secret.endpoints`).
4. **GUI `quay_dockerconfigjson` builder** — mirror `_derive_ngc_payloads` logic at `main.py:476` but for the `quay.io` host (`auth = b64(user:pass)`, standard padding D-12); inject as one var; never template raw creds.
5. **SSE deadline** — operator + CSI + WekaClient readiness can exceed the 15-min cap (`main.py:2956`); raise the deadline + keepalive for this blueprint.
6. **Frontend** — convert `welcome.html` to a stepper, drop the prerequisite hard-block in `handleInitialize`, drive stage rows from `component` SSE events, chain `app-store-install` → `cluster-init`, redirect when cluster-init reaches `Ready` (Decision E, middleware unchanged).
7. **Secrets via `stringData`** — all three credential secrets; no manual base64.

## Sources

- Repo files (HIGH): `weka-csi-config/wekaClientCR-online.yaml`, `quay-secret.yaml`, `csi-wekafs-api-secret.yaml`, `storageclass-*.yaml`, `weka-csi-config/weka-operator/Chart.yaml` + `crds/`, `operator_module/main.py` (`:118`, `:476-509`, `:679`), `app-store-gui/webapp/main.py` (`:1801`), `cluster_init/app-store-cluster-init.yaml`, `.planning/PRD-install-wizard-weka-storage-stack.md`.
- [WEKA Operator GitHub](https://github.com/weka/weka-operator) — OCI ref `oci://quay.io/weka.io/helm/weka-operator`, `helm show crds` / `helm pull` flow (HIGH).
- [WEKA — Deploy the WEKA client on Amazon EKS](https://docs.weka.io/kubernetes/weka-operator-deployments/deploy-the-weka-client-on-amazon-eks) — `cpuManagerPolicy: static`, `systemReserved.cpu`, hugepages rationale, `weka.io/supports-clients=true`, `WekaClient` `weka.weka.io/v1alpha1` fields (HIGH for label/CRD/policy; MEDIUM for exact hugepage count / strict-cpu-reservation key).
- [csi-wekafsplugin on Artifact Hub](https://artifacthub.io/packages/helm/csi-wekafs/csi-wekafsplugin) — latest `2.8.7`, repo `https://weka.github.io/csi-wekafs`, API-based since 0.7.0 (HIGH).
