# PRD: Guided Install Wizard — WEKA Operator, CSI Driver & Storage Classes

**Status:** Draft (for review)
**Author:** Christopher Jenkins
**Date:** 2026-06-24
**Area:** Web GUI (`welcome.html` + `main.py`) · AppStack blueprint · Operator helm/secret handling · `weka-csi-config/` templates

---

## Problem

Today the App Store install flow is a **single-button initializer**. The `/welcome` page (`welcome.html`):

1. Calls `/cluster-info` to detect whether the WEKA Operator (6 CRDs) and CSI driver (`csi-wekafs-controller` deployment) are **already present**, and shows two read-only status dots.
2. Refuses to proceed unless both are present (`handleInitialize` hard-blocks: *"you need to install the WEKA Operator and WEKA CSI driver first"*).
3. On Initialize, applies the fixed `cluster_init/app-store-cluster-init.yaml` WekaAppStore CR (monitoring, Envoy, Gateway CRDs) via `/deploy-stream?app_name=cluster-init`, streams progress, and redirects to `/` when the CR reaches `Ready`.

So the operator and CSI driver are **prerequisites the customer must install by hand** (via the docs links shown on the page) before they can use the App Store at all. That hand-install is a long, error-prone sequence of `kubectl`/`helm` commands plus several base64-encoded secret files and StorageClass YAMLs — exactly the friction the App Store exists to remove.

We want to **fold that entire WEKA storage-stack install into the wizard**: collect the handful of customer-specific variables in a web form, install the operator + CSI + StorageClasses through the existing AppStack mechanism with live per-stage progress, and only then run the existing cluster-init as the **final** step before redirecting to the App Store.

---

## Goals

1. Replace the single Initialize button with a **multi-step web form** that collects every customer-specific variable needed to stand up the WEKA storage stack.
2. Install, in the correct dependency order, via a parameterized AppStack `WekaAppStore` CR: worker-node kubelet/hugepage config, quay pull secret, WEKA Operator (+CRDs), node labels, `WekaClient` CR + its secret, CSI driver, the CSI API secret, and three StorageClasses (one set as the cluster default).
3. Show **live per-stage progress** across all install stages, reusing the existing `componentStatus` SSE stream.
4. Run the **existing cluster-init** (monitoring/Envoy/etc.) as the **last** stage, then redirect to the App Store UI — preserving current end-state behavior.
5. Turn the static templates in `weka-csi-config/` into parameterized blueprint components with `[[ variable ]]` substitution.

## Non-Goals (this PRD)

- Changing the cluster-init component set (`app-store-cluster-init.yaml`) itself — it runs as-is, just later.
- Day-2 operations: upgrading/uninstalling the operator or CSI, rotating WEKA credentials, editing StorageClasses after install.
- Supporting non-quay operator registries or air-gapped/offline operator images.
- Multi-cluster / multiple WEKA backend targets in one wizard run.
- WEKA backend cluster provisioning — we connect **clients** to an existing WEKA NeuralMesh cluster; we do not create the storage cluster.

---

## Current State (verified in code)

| Concern | Where | Behavior today |
|---|---|---|
| Route gating | `main.py:40-129` `ClusterInitMiddleware`; exempt paths `main.py:43` | Blocks the GUI until the `app-store-cluster-init` WekaAppStore CR is `Ready`. |
| Prerequisite detection | `/cluster-info` `main.py:2361` | Returns `weka_operator_installed` (checks 6 CRDs) and `weka_csi_installed` (finds `csi-wekafs-controller`). |
| Init form | `welcome.html` | One namespace dropdown + Initialize button; two status dots; log box; progress bar. |
| Deploy + progress | `/deploy-stream` `main.py:2839-3011` | Renders blueprint with Jinja2 `[[ ]]` delimiters from a `variables` JSON, applies docs, polls `status.componentStatus[]` and emits `component`/`complete` SSE events. |
| Apply engine | `planning/apply_gateway.py` | Uses the Kubernetes Python client; special-cases WekaAppStore CRs (create→409→patch); handles cluster-scoped kinds incl. `StorageClass`, `Namespace`, `CustomResourceDefinition`. |
| Helm execution | `operator_module/main.py` | The **operator** runs all helm/kubectl via subprocess. Supports OCI charts (`oci://`, skips `helm repo add` at `:118`), CRD discovery via `helm show crds` (`:675`), `valuesFiles` from ConfigMap/Secret, `dependsOn`, and `readinessCheck` (pod/deployment/job). |
| Secret/base64 helpers | `operator_module/main.py:435-594` | Existing helpers build standard padded base64 and `dockerconfigjson` for pull secrets (locked decisions D-11/D-12/D-13). |
| Credentials API | `/api/credentials` `main.py:1099` | Creates `WarpCredential` CRs + raw secrets (pattern reusable for quay creds). |

**Key takeaway:** the AppStack mechanism already does almost everything we need — ordered components, OCI helm installs, CRD install, readiness gating, and live progress. The new work is mostly **(a) a parameterized blueprint**, **(b) a multi-step form**, and **(c) resolving a few gaps** called out under Open Design Decisions.

---

## Proposed UX — Multi-Step Wizard

`welcome.html` becomes a stepper. The right-hand card keeps the "Target Cluster" header; the status dots become **live install stages** rather than prerequisite checks.

**Step 1 — Worker Node Prerequisites**
Explains the kubelet/hugepage requirement (CPU manager `static` policy, `strict-cpu-reservation`, hugepages `25000` for clients). See **Open Decision A** for whether this is automated or a documented manual confirm.

**Step 2 — Operator Registry Credentials** (quay.io)
- Quay username (`QUAY_USERNAME`)
- Quay password / robot token (`QUAY_PASSWORD`), masked
- Operator chart version (default `v1.13.0`)

**Step 3 — WEKA Cluster Connection**
- Join endpoints — `joinIpPorts`: list of `host:port` (e.g. `192.168.1.1:14000`). One input accepting comma/newline-separated entries.
- WEKA container image version — the tag after `weka-in-container:` (e.g. `5.1.0.605`)
- API scheme — **dropdown**: `http` / `https`

**Step 4 — WEKA Cluster Credentials**
- Organization (`org`), default `Root`
- WEKA username
- WEKA password (masked)

**Step 5 — Review & Install**
Read-only summary (secrets masked) + namespace selector (kept from today). Submit triggers the install stream.

**Install view** — a vertical stage list, each row Pending → In-progress → Done/Failed, driven by `component` SSE events, plus the existing scrolling log box. On the terminal cluster-init `Ready`, show success and redirect to the App Store (`redirect_url` or `/`), exactly as today.

---

## Install Sequence → Components

The wizard renders a **single parameterized `WekaAppStore` CR** (`cluster_init/app-store-install.yaml`) whose `appStack.components[]` encode the full ordered sequence below using `dependsOn` + `readinessCheck`. The customer's 9-step manual procedure maps as follows:

| # | Manual step | Component(s) | Mechanism | Notes / gaps |
|---|---|---|---|---|
| 1 | Kubelet config + hugepages on workers | (none — documented prereq) | Wizard Step 1 shows the required `KubeletConfiguration` snippet + a confirm checkbox (**Decision A1**) | Node-level/root work stays the customer's responsibility; the App Store does not restart kubelet. |
| 2 | quay docker-registry secrets in `weka-operator-system` + `default` | `quay-pull-secret` | `Secret` manifests (`kubernetes.io/dockerconfigjson`); GUI builds the `dockerconfigjson` (**Decision B**) and injects it as one `[[ quay_dockerconfigjson ]]` var | Reuse operator helper logic at `:476`. Two copies — one per namespace. |
| 3 | Install operator CRDs + operator chart from quay | `weka-operator` | AppStack `helmChart` `oci://quay.io/weka.io/helm/weka-operator`, `version: [[ operator_version ]]`; operator already does `helm show crds` (`:675`) | **No `helm registry login` needed** (**Decision C**): the quay `dockerconfigjson` secrets in the namespaces cover chart + image pulls. `readinessCheck: deployment` in `weka-operator-system`. |
| 4 | Label all nodes `weka.io/supports-clients=true` | `node-label-job` | `Job` running `kubectl label nodes --all` (precedent: `gateway-api-crds-job` uses cluster-admin SA) `readinessCheck: job` | depends on operator install. |
| 5 | `WekaClient` CR (`joinIpPorts`, image tag) | `weka-client` | `kubernetesManifest` from `wekaClientCR-online.yaml`, `joinIpPorts: [[ ... ]]`, `image: ...:[[ weka_image_version ]]` | depends on operator Ready (CRD `weka.weka.io/v1alpha1` must exist). `imagePullSecret: quay-io-secret`. |
| 6 | `weka-client-cluster-dev` secret (`org`,`username`,`password`) | `weka-client-secret` | `Secret` — **use `stringData`** so values aren't pre-encoded | referenced by the WekaClient `wekaSecretRef`. Must exist before/with step 5. |
| 7 | `csi-wekafs-api-secret` (username/password/organization/endpoints/scheme) | `csi-api-secret` | `Secret` `stringData`; `endpoints` = comma-joined `joinIpPorts`; `scheme` from dropdown | reused by all StorageClasses. |
| 8 | Install CSI driver | `csi-wekafs` | AppStack `helmChart` `csi-wekafs/csi-wekafsplugin` into `csi-wekafs` ns; `readinessCheck: deployment csi-wekafs-controller` | public repo — no auth. |
| 9 | Three StorageClasses (dir-api as **default**) | `storageclasses` | `kubernetesManifest` (3 docs) from the `storageclass-*.yaml` files; `is-default-class: "true"` on `storageclass-wekafs-dir-api` | `secretName`/`secretNamespace` must match step 7 (`csi-wekafs-api-secret` / `csi-wekafs`). |
| 10 | **(existing)** cluster-init | **chained** — separate `app-store-cluster-init.yaml`, untouched | runs **last** as a second CR (**Decision D**) | monitoring/Envoy/etc., then redirect when it reaches `Ready` (**Decision E**). |

### Variable → Field Mapping

| Wizard field | WekaClient (`wekaClientCR-online.yaml`) | `weka-client-cluster-dev` secret | `csi-wekafs-api-secret` | quay secret |
|---|---|---|---|---|
| Join endpoints | `joinIpPorts` (YAML list) | — | `endpoints` (comma-joined string) | — |
| WEKA image version | `image: ...:<ver>` | — | — | — |
| Scheme (http/https) | — | — | `scheme` | — |
| Org | — | `org` | `organization` | — |
| WEKA username | — | `username` | `username` | — |
| WEKA password | — | `password` | `password` | — |
| Quay username | — | — | — | `username`/`email`/`auth` |
| Quay password | — | — | — | `password`/`auth` |

One input drives multiple destinations — the GUI is responsible for producing both the **YAML list** form (`["host:port", ...]`) and the **comma-joined string** form of the endpoints from a single field.

---

## Technical Design

### Backend
- **New blueprint** `cluster_init/app-store-install.yaml` — the parameterized WekaAppStore CR above, with an `x-variables` block (so the existing `parse_x_variables` / required-field validation in `/deploy-stream` applies). Templated with the existing `[[ var ]]` Jinja2 delimiters.
- **`find_blueprint` (`main.py:1801`)** — add `app-store-install` alongside the existing `cluster-init` special-case path, or rely on x-variables discovery.
- **Reuse `/deploy-stream`** unchanged for the per-component progress contract. The wizard POSTs/streams `variables` JSON exactly like other blueprints. (The `cluster-init` namespace-preservation special-case at `:1931`/`:2943` likely applies to `app-store-install` too — these components target fixed namespaces.)
- **Endpoint transform** — a small helper that parses the endpoints field into list + comma-joined forms and injects both as derived variables.
- **Secrets as `stringData`** — avoids encoding user input by hand and sidesteps the trailing-newline bug noted below.

### Frontend (`welcome.html`)
- Convert the React component to a stepper (MUI `Stepper` or simple step state). Keep the dark WEKA theme.
- Remove the prerequisite hard-block in `handleInitialize`; the dots become live stages fed by `component` events. (`/cluster-info` can stay as an informational "already installed?" hint and to allow skipping if detected.)
- Submit builds the `variables` object and opens `EventSource('/deploy-stream?app_name=app-store-install&...')`; map `component` events to stage rows; on `complete ok:true` for the final stage, redirect.

### Operator
- **No change required for quay auth** (Decision C). Confirm the operator's existing ordering correctly gates the `WekaClient` apply behind operator-CRD install + operator-deployment readiness. CRD install, readiness checks, and valuesFiles are already supported.

---

## Resolved Decisions

**A → A1. Worker-node kubelet/hugepage config is a documented prerequisite.**
Wizard Step 1 displays the required `KubeletConfiguration` (CPU manager `static`, `strict-cpu-reservation`, hugepages `25000` for clients) as a copy-paste snippet, gated behind an "I have applied node prerequisites" checkbox. The App Store does **not** write node config or restart kubelet. (Auto-config via privileged DaemonSet is a possible follow-on, out of scope here.)

**B → Build the quay secret in the GUI.**
The GUI assembles the `kubernetes.io/dockerconfigjson` payload (base64 `auth` = `user:pass`, wrapping JSON) from the Step 2 quay username/password, reusing the operator helper logic at `operator_module/main.py:476`, and injects it as a single `[[ quay_dockerconfigjson ]]` variable. Raw quay creds are never templated into the manifest text. Two `quay-pull-secret` copies are created — one in `weka-operator-system`, one in `default`.

**C → REVISED (2026-06-24, after v8.0 research): operator-side `helm registry login` IS required.**
The original decision (rely solely on in-namespace `dockerconfigjson` secrets) was based on manual `helm install` success — but that worked because the *workstation* had `~/.docker/config.json` from a prior login. The **operator pod** runs `helm show crds` / `helm install oci://quay.io/...` from its own helm process, which authenticates via the helm registry config, **not** the Kubernetes `dockerconfigjson` pull secrets (those only cover the kubelet's image pulls). No existing component pulls from an authenticated OCI registry, so this path is untested and would 401 on a fresh cluster. **v8.0 scopes operator-side helm registry auth for quay (`helm registry login` / `--registry-config`) from the start**, and fixes the related `discover_chart_crds` `@lru_cache` poisoning (an empty-CRD failure result is memoized, which then 404s the WekaClient apply). The `dockerconfigjson` pull secrets are still needed — for the operator/client *image* pulls.

**D → Two chained CRs.**
`app-store-cluster-init.yaml` stays untouched. The wizard applies the new `app-store-install` WekaAppStore CR first and waits for it to reach `Ready`, then applies the existing `cluster-init` CR as the final stage.

**E → cluster-init `Ready` remains the single redirect gate.**
`ClusterInitMiddleware` is unchanged; the redirect to the App Store fires when the `app-store-cluster-init` CR reaches `Ready` — which now naturally comes last.

---

## Risks & Notes

- **Trailing-newline bug in existing templates — FIXED (2026-06-24).** The committed base64 values in `weka-client-cluster-dev.yaml` and `csi-wekafs-api-secret.yaml` decoded with a trailing `\n` (`echo "x" | base64` instead of `printf`/`echo -n`). Re-encoded the affected fields (`username`, `password`, `endpoints`, `scheme`) to decode cleanly. The generated blueprint will use `stringData` from form input, eliminating this bug class going forward.
- **Credential handling.** Quay token and WEKA password are sensitive. They must never be logged (the SSE log box echoes component names/messages — ensure no secret values leak), and should transit over the form POST only. Consider storing the WEKA creds as a `WarpCredential` (existing pattern) for reuse by storage-aware blueprints later.
- **Long-running install.** Operator + CSI + WekaClient readiness can exceed the current 15-minute SSE cap (`:2956`). May need a higher deadline and robust keepalive for this blueprint.
- **`WekaClient` readiness.** Step 5 depends on the operator CRDs existing and operator pods being Ready — `dependsOn` + `readinessCheck: deployment` on the operator must gate it, or the CR apply 404s on a missing CRD.
- **Idempotency / re-run.** A customer retrying after a partial failure must be safe: apply-or-patch (already how WekaAppStore CRs and secrets behave), and the node-label Job / DaemonSet must be no-ops on re-run.

---

## Success Criteria

1. From a cluster with **no** WEKA operator/CSI, a customer completes the wizard form and the operator, CSI driver, `WekaClient`, all secrets, and three StorageClasses are installed — with `storageclass-wekafs-dir-api` marked default.
2. The install view shows each stage transitioning through Pending → In-progress → Done, and surfaces a clear error + retry on failure.
3. After the storage stack is up, the existing cluster-init runs and, on `Ready`, the customer is redirected to the App Store UI.
4. No secret values appear in logs or the SSE stream.
5. Re-running the wizard on an already-installed cluster is safe (no duplicate/destructive actions).

---

## Phased Implementation (proposal)

1. **Blueprint** — author `cluster_init/app-store-install.yaml` with `x-variables` and all components; parameterize the `weka-csi-config/` templates. Verify with `helm template`-equivalent dry run / operator unit tests.
2. **Operator** — confirm CRD-install + operator-readiness ordering correctly gates the `WekaClient` apply (no auth change needed, per Decision C).
3. **Backend** — wire `app-store-install` into `find_blueprint`; add endpoint transform (list + comma-joined) and the GUI `quay_dockerconfigjson` builder; raise the SSE deadline for the longer install.
4. **Frontend** — convert `welcome.html` to the stepper + live stage list; add the Step 1 node-prereq snippet + confirm checkbox; remove the prerequisite hard-block; chain `app-store-install` → `cluster-init`.
5. **E2E** — fresh-cluster install, failure/retry, re-run idempotency, redirect.

---

*Decisions A–E are resolved. Ready to proceed to implementation on approval.*
