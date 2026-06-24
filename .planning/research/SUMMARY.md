# Project Research Summary

**Project:** WEKA App Store — v8.0 Guided Install Wizard (WEKA Storage Stack)
**Domain:** Brownfield integration of a multi-step install wizard + parameterized AppStack blueprint into an existing FastAPI/Jinja GUI + Kopf operator + `WekaAppStore` CR system
**Researched:** 2026-06-24
**Confidence:** HIGH on the integration map; **the single highest-risk item (operator quay chart-pull auth) is UNRESOLVED and contradicts the PRD's Decision C — it must be settled by a go/no-go spike before the roadmap is committed.**

## Executive Summary

v8.0 folds the entire WEKA storage-stack install (operator + CSI + WekaClient + secrets + 3 StorageClasses, then the existing cluster-init) into the App Store's `/welcome` page, replacing the single Initialize button with a 5-step web wizard. The good news, confirmed across STACK, FEATURES, and ARCHITECTURE research with HIGH confidence: the AppStack runtime **already does almost all of this** — ordered, dependency-gated, CRD-installing, readiness-checked deployment with live `componentStatus` SSE progress. There is **no new language/framework/package to add**. The work is overwhelmingly additive: one new parameterized blueprint (`cluster_init/app-store-install.yaml`), parameterized `weka-csi-config/` templates using `[[ var ]]` + `stringData`, a stepper frontend in `welcome.html`, and a few backend helpers (`build_quay_dockerconfigjson`, `split_endpoints`, a raised SSE deadline, `find_blueprint` entry). The shape matches established infra-installer UX (Rancher/Longhorn/Vault), so most features are table stakes, not novel invention.

**However, the four researchers DISAGREE on a load-bearing point, and the disagreement must be surfaced as the #1 milestone risk.** STACK and ARCHITECTURE both conclude "Decision C is sound — the quay `dockerconfigjson` pull secrets cover chart + image pulls, no operator change needed." PITFALLS concludes (HIGH confidence, citing `operator_module/main.py:136-178` and `:673-703` plus the Helm OCI registry docs) that **this is wrong as written**: the operator installs the chart by shelling out to `helm install oci://quay.io/...` and `helm show crds oci://...` *inside its own pod*, which authenticates only via the helm registry config (`helm registry login` / `--registry-config`) — **NOT** via Kubernetes `imagePullSecrets`, which the kubelet consumes for *container image* pulls only. These are two separate auth subsystems. The plausible reconciliation: the user's MANUAL `helm` success worked because their **workstation** had a `~/.docker/config.json` from a prior `docker/helm login`; the operator **pod has no such login**, and **no existing cluster-init component pulls from an authenticated OCI registry** (cluster-init uses public `oci://docker.io/envoyproxy`), so this path is genuinely **untested**. If PITFALLS is right, a fresh-cluster operator install fails with a quay `401` at the very first chart touch — making the feature dead-on-arrival — and the empty CRD result gets cached (Pitfall 8), compounding it.

**Recommendation:** Do **not** treat Decision C as settled. Front-load a small **go/no-go spike** that runs the operator pod against quay with *only* the dockerconfigjson secrets present (no host helm login) and observes whether `helm show crds`/`helm install` succeed or `401`. The spike outcome determines whether the operator/E2E phase stays "confirm ordering" (Decision C holds) or **expands to implement operator-side helm registry auth for OCI quay charts** (write `--registry-config` from the quay secret on every OCI op, plus fix the `discover_chart_crds` failure-caching). All other risks — CRD-404 ordering, secret leakage, 900s SSE deadline, default-StorageClass conflict, node-label Job idempotency, the A1 copy-paste-only node prereq — are well-understood, config-or-backend-level, and carry forward into specific phases below.

## Key Findings

### Recommended Stack

No dependencies to add — all "installation" is runtime YAML the wizard renders through the existing engine (see [STACK.md](STACK.md)). The external chart/image coordinates the wizard *installs* are pinned and field-overridable.

**Core technologies (what the wizard installs, not what we build with):**
- **WEKA Operator Helm chart `v1.13.0`** (`oci://quay.io/weka.io/helm/weka-operator`, field-overridable via `[[ operator_version ]]`) — installs the operator Deployment + the 6 `weka.weka.io` CRDs that make `WekaClient` a valid kind. **Bundled in-repo copy is stale `v1.9.1` — do not rely on it; pull the pinned version from quay at runtime.** This is the chart whose pull auth is the #1 open risk.
- **WEKA CSI driver `csi-wekafsplugin` 2.8.x** (public repo `https://weka.github.io/csi-wekafs`, latest `2.8.7`) — installs `csi-wekafs-controller` + node DaemonSet; `provisioner: csi.weka.io`. **No auth — do not attach a pull secret.** API-based since 0.7.0; StorageClasses must carry the `csi.storage.k8s.io/*-secret-name/-namespace` params.
- **WEKA-in-container image** (tag only, e.g. `5.1.0.605` via `[[ weka_image_version ]]`) — the WekaClient runtime image from quay, pulled by the kubelet using the dockerconfigjson secret (this path *is* covered by the pull secret — the chart path is the contested one).

**Reused as-is (do NOT rebuild):** OCI helm install (skips `helm repo add`), `helm show crds` CRD discovery, the `_b64`/dockerconfigjson builder pattern, `dependsOn` + `readinessCheck`, `[[ var ]]` Jinja2 + `x-variables`, the `/deploy-stream` SSE, and the kubernetes apply gateway (create->409->patch).

### Expected Features

The v8.0 wizard maps cleanly onto established infra-installer conventions (see [FEATURES.md](FEATURES.md)): gather prereqs -> collect credentials in grouped steps -> review masked summary -> submit -> watch real per-stage progress -> hand off to the product. Most behaviors are table stakes precisely because the underlying AppStack machinery already exists.

**Must have (table stakes — all P1 for v8.0):**
- 5-step stepper UI replacing the single Initialize button (node-prereq confirm -> quay creds -> WEKA connection -> WEKA creds -> review/install)
- Per-step forward-blocking validation (host:port list, masked secrets, version tags, scheme dropdown)
- Live per-stage progress list mapped from the existing `component` SSE events (honest status, not a synthetic timer)
- Error surface on partial failure + user-triggered Retry that preserves form state (re-run is idempotent via apply-or-patch)
- Chained `app-store-install` -> `cluster-init` -> redirect on cluster-init `Ready` (preserves today's end-state; middleware unchanged)
- Secrets never appear in logs/SSE (release gate — Success Criterion 4)

**Should have (differentiators):**
- **Single field -> multiple sinks:** one endpoints field yields both `joinIpPorts: ["h:p",...]` (WekaClient) and `endpoints: "h:p,h:p"` (CSI secret); one credential set fans out to multiple secrets. This is the core "enter once" value.
- **GUI-built `quay_dockerconfigjson`** — removes the worst manual step (base64/JSON wrangling)
- **`stringData` secrets** — eliminates the trailing-newline base64 bug class entirely
- **Copy-paste `KubeletConfiguration` snippet + confirm checkbox** (Decision A1) — the one thing the wizard *can't* do for the user, de-risked

**Defer (v1.x / v2+):**
- Prereq-detection skip path (auto-skip steps when operator/CSI already present)
- Store WEKA creds as a `WarpCredential` for downstream blueprint reuse
- Privileged DaemonSet to auto-apply node kubelet/hugepage config (high-risk node mutation — explicitly out of scope)
- Day-2 ops (upgrade/uninstall/rotate/edit), air-gapped/non-quay registries, multi-cluster

### Architecture Approach

This is an integration map, not greenfield (see [ARCHITECTURE.md](ARCHITECTURE.md)). The frontend modifies `welcome.html` into a 5-step stepper that chains two `EventSource('/deploy-stream')` streams (`app-store-install` then `cluster-init`). The backend adds a `find_blueprint` entry, two derived-variable helpers, extends the cluster-init namespace-preserve special-case to `app-store-install`, and raises the SSE deadline. The new blueprint is a **single** parameterized `WekaAppStore` CR whose `appStack.components[]` encode the ordered install; the operator's existing topo-sort + sequential, readiness-gated component loop does the rest.

**Major components:**
1. **`cluster_init/app-store-install.yaml`** (NEW) — single parameterized `WekaAppStore` CR, 9 components (2 quay secrets -> operator -> node-label Job / weka-client-secret / csi-wekafs -> weka-client / csi-api-secret -> storageclasses), `x-variables` schema, `[[ var ]]` + `stringData`. This is the bulk of the work.
2. **`welcome.html`** (MODIFIED) — 5-step stepper, drop the prereq hard-block, live stage rows from `component` SSE events, chain the two CR streams, redirect on cluster-init `Ready`.
3. **`main.py` backend** (MODIFIED) — `find_blueprint` entry; `build_quay_dockerconfigjson()` + `split_endpoints()` (derive **server-side**); namespace-preserve special-case; raised SSE deadline; secret-redaction on the annotation + SSE message.
4. **`operator_module/main.py`** — ARCHITECTURE/STACK say UNCHANGED (Decision C); **PITFALLS says this must change for quay chart-pull auth.** Scope is gated on the spike (see Risks).

### Critical Pitfalls

Top items from [PITFALLS.md](PITFALLS.md), ordered by severity:

1. **Operator quay chart-pull auth (Decision C contested — #1 OPEN RISK).** `helm install`/`helm show crds` for `oci://quay.io/...` run inside the operator pod and authenticate via the helm registry config, NOT `imagePullSecrets`. A fresh cluster with only the dockerconfigjson secrets likely `401`s at the first chart touch. **Resolve via a go/no-go spike before committing the roadmap** (`operator_module/main.py:136-178`, `:673-703`; Helm OCI docs). Avoidance if confirmed: operator writes `--registry-config` from the quay secret on every OCI op.
2. **`discover_chart_crds` `@lru_cache` poisons CRD discovery on failure** (`operator_module/main.py:673`, `:685-688`). Returns empty `set()` on *any* `helm show crds` failure and caches it against `(chart, version)` — so after a transient/auth blip the operator believes the chart ships no CRDs forever (until pod restart), and WekaClient then 404s. Fix: don't cache failures (distinguish "fetched empty" from "fetch failed"); bundle with the auth fix.
3. **WekaClient CR applied before its CRD is served -> 404** (`apply_gateway.py:301,323` only handles 409, not 404). Gate via `dependsOn: [weka-operator]` **plus** `readinessCheck: deployment` on the operator (transitively guarantees CRDs are served), and optionally a belt-and-suspenders "wait-for-crd Established" poll.
4. **Secret leakage into the CR annotation and SSE stream.** `/deploy-stream` stamps the *entire* `variables` dict onto the `warp.io/gui-variables` annotation (`main.py:2935`) in cleartext, and emits operator `componentStatus` messages verbatim (`main.py:2984`). Fix: allowlist/denylist excludes `*password*`/`*token*`/`*secret*`/`quay_dockerconfigjson` from the annotation; redact emitted SSE messages; never pass creds as helm `--set`. **Release gate (Success Criterion 4).**
5. **900s SSE deadline too short.** `/deploy-stream` caps at 15 min (`main.py:2956`); a cold-cache multi-GB `weka-in-container` pull + cluster join alone can exceed it, showing a false "failed" on a still-progressing install. Fix: raise/per-blueprint deadline (45-60 min); make timeout non-destructive (reconnect-and-resume from `componentStatus`); document ingress `proxy-read-timeout` + `proxy_buffering off`; confirm operator `HELM_CMD_TIMEOUT`.
6. **Default-StorageClass conflict.** `storageclass-wekafs-dir-api` hard-codes `is-default-class: "true"`; on a brownfield cluster with an existing default this creates two defaults -> non-deterministic PVC binding (surfaces later, in workloads). Fix: detect an existing default at review; warn/offer to switch; verify SC `*-secret-name/-namespace` match the CSI API secret.
7. **Node-label Job not idempotent on re-run.** Plain `kubectl label` fails with "already has value"; a same-name `Job` is immutable on re-apply. Fix: `--overwrite` + delete-then-create / treat completed Job as success. Verify the precedent `gateway-api-crds-job` is itself idempotent. (Success Criterion 5.)
8. **Auto-restarting kubelet for node prereqs is forbidden (Decision A1 protects this).** CPU-manager `none`->`static` requires deleting `/var/lib/kubelet/cpu_manager_state` + kubelet restart; automating it can brick every node. Keep the copy-paste snippet + confirm checkbox; the App Store never mutates node config. The pitfall is *re-introducing* automation later. NOTE: A1 is **copy-paste-only — no auto kubelet restart**; if the customer skips it, WekaClient pods sit unready with insufficient-hugepages — surface that clearly.

## Implications for Roadmap

The dependency-aware build order is clear: the blueprint is the contract everything else targets; the operator-auth question is the single hard de-risk that must come first; backend and frontend consume the blueprint; E2E validates the chained-CR end state. **The one deviation from the PRD's phasing: a spike precedes Phase 2, and Phase 2's scope is conditional on the spike.**

### Phase 0 (NEW): Quay Helm-Auth Go/No-Go Spike
**Rationale:** The researchers disagree on Decision C and it is load-bearing — if PITFALLS is right, the feature is dead-on-arrival. No downstream phase can be safely scoped until this is settled. The user's manual success likely used a workstation `~/.docker/config.json` the operator pod does not have; no existing cluster-init component exercises an authenticated OCI pull, so the path is untested.
**Delivers:** A definitive answer to: *with only the quay `dockerconfigjson` secrets present (no host/operator helm login), does the operator pod's `helm show crds` / `helm install oci://quay.io/weka.io/helm/weka-operator` succeed or 401?*
**How:** Run the operator pod (or a pod with the same helm/registry config) against quay using only the in-namespace dockerconfigjson; observe. If 401 -> Decision C is wrong; Phase 2 expands.
**Gate:** Spike outcome sets Phase 2 scope. Treat as the #1 open risk.

### Phase 1: Blueprint Authoring
**Rationale:** The parameterized CR is the contract every other phase targets; author it first so backend/frontend/operator/E2E have a fixed target.
**Delivers:** `cluster_init/app-store-install.yaml` — all 9 components with `dependsOn`/`readinessCheck`, `x-variables` schema, parameterized `weka-csi-config/` content as `[[ var ]]` + `stringData`.
**Addresses:** Parameterized app-store-install blueprint, stringData secrets, the install-sequence components.
**Avoids:** Pitfall 2 (`stringData`, not hand-base64); Pitfall 3 (dockerconfigjson injected as one `data` var, not `stringData`); Pitfall 7 (node-label Job `--overwrite` + re-run strategy); Pitfall 6 (default-SC secret-name/namespace match); Pitfall 3-CRD wiring (`dependsOn` + `readinessCheck: deployment` on operator).
**Verify:** `yaml.safe_load_all(render(...))` succeeds; topo order correct (operator `resolve_dependencies`).

### Phase 2: Operator / E2E Ordering — SCOPE CONDITIONAL ON PHASE 0
**Rationale:** De-risk the single hard integration hazard before UI work. The PRD scopes this as "confirm ordering, no auth change needed" — **that scope is only valid if Phase 0 confirms Decision C.**
**Delivers (if spike confirms Decision C holds):** Confirmation that the operator chart *installs* (not `--skip-crds`) the WEKA CRDs and that `readinessCheck: deployment` gates WekaClient correctly; add a wait-for-crd Job only if the operator goes Available before CRDs are served.
**Delivers (if spike shows 401 — the likely-per-PITFALLS case):** **Operator-side helm registry auth for OCI quay charts** — materialize `--registry-config` from the quay secret on every `helm install`/`helm upgrade`/`helm show crds` for `oci://quay.io/...` refs (`operator_module/main.py:136-178`, `:673-703`), **plus** fix `discover_chart_crds` to not cache failures (Pitfall 8). This is a real operator code change contradicting Decision C — the phase scope EXPANDS from "confirm ordering" to "implement operator-side helm registry auth for quay."
**Uses:** WEKA operator chart `v1.13.0` from quay; existing OCI install path.
**Avoids:** Pitfalls 1, 3-CRD (404), 8.
**Verify:** Fresh cluster with *only* the pull secrets -> operator chart installs (no 401); WekaClient applies without 404; CRDs present.

### Phase 3: Backend Wiring
**Rationale:** Backend consumes the blueprint and produces the derived/secure variables the frontend will POST.
**Delivers:** `find_blueprint` entry; `build_quay_dockerconfigjson()` (mirror `operator_module/main.py:476-509`, `_b64`); `split_endpoints()` -> (YAML-list, CSV) — both derived **server-side**; extend cluster-init namespace-preserve special-case to `app-store-install`; raise/per-blueprint SSE deadline + non-destructive timeout; **secret allowlist on the `warp.io/gui-variables` annotation + SSE message redaction.**
**Implements:** Endpoint transform, GUI dockerconfigjson builder, secret-leak guard, raised SSE deadline.
**Avoids:** Pitfall 3 (dockerconfigjson assembly + round-trip test); Pitfall 4 (annotation `main.py:2935` + SSE message `:2984` redaction — release gate); Pitfall 5 (existing-default-SC detection); Pitfall 5-SSE (deadline + reconnect). Also consider a POST variant / server-side staging so creds don't ride a GET query string into proxy/access logs.
**Verify:** Unit tests for both helpers (`auths["quay.io"]["auth"]` decodes to exactly `user:pass`; endpoints both forms); `/deploy-stream?app_name=app-store-install` locates + renders.

### Phase 4: Frontend Stepper
**Rationale:** Consumes the blueprint + backend; last build step before E2E.
**Delivers:** `welcome.html` -> 5-step stepper (Step 1 node-prereq snippet + confirm checkbox), drop the `handleInitialize` hard-block, live stage rows from `component` events, chain `app-store-install` -> `cluster-init`, redirect on cluster-init `Ready`.
**Addresses:** Stepper UI, per-step validation, live progress list, error+retry, node-prereq snippet, secret masking in review.
**Avoids:** Pitfall 8 (A1 confirm checkbox — never auto-restart kubelet); Pitfall 4 (mask secrets in review/replay).
**Verify:** Form builds correct `variables{}`; stage rows update; redirect fires.

### Phase 5: E2E
**Rationale:** Validates the chained-CR end state against PRD Success Criteria.
**Delivers:** Fresh-cluster full install -> 3 StorageClasses (dir-api default) -> cluster-init -> redirect; failure/retry; **re-run idempotency**; secret-leak check on the log box + CR YAML.
**Avoids:** Pitfall 6 (exactly one default SC); Pitfall 7 (idempotent re-run — Job, SC, chained CRs); Pitfall 4 (no creds in `kubectl get wekaappstore -o yaml` or the browser log); Pitfall 5 (cold-cache install completes without false timeout).
**Verify:** PRD Success Criteria 1-5.

### Phase Ordering Rationale
- **Phase 0 (spike) is non-negotiable and first** because the entire operator phase scope, and arguably the feature's viability, hinges on a contested decision no existing component exercises.
- The blueprint (1) is the declarative contract; building it before backend/frontend prevents churn.
- The operator/E2E de-risk (2) comes before UI so the single hard hazard (chart auth + CRD ordering) is settled before consuming code is written.
- Backend (3) and frontend (4) are parallelizable once the blueprint lands; E2E (5) is the integration gate.

### Research Flags

Phases likely needing deeper research / a spike during planning:
- **Phase 0:** This *is* the research — a hands-on go/no-go spike against quay, not a literature review. Highest priority.
- **Phase 2 (Operator):** If the spike shows 401, the helm-registry-auth implementation (where/how the operator reads the quay secret, `--registry-config` materialization, `discover_chart_crds` cache fix) needs `/gsd:plan-phase --research-phase`. Also confirm the operator chart's `crdsStrategy`/`--skip-crds` heuristic doesn't skip the WEKA CRDs.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Blueprint):** Mirrors the existing `cluster-init` blueprint + `gateway-api-crds-job` precedent; well-understood `[[ var ]]`/`stringData`/`dependsOn` patterns.
- **Phase 3 (Backend):** Helpers mirror `_derive_ngc_payloads`; SSE/annotation paths are existing code.
- **Phase 4 (Frontend):** Single-React-root + CDN constraint, existing `/deploy-stream` contract; standard stepper pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Chart coordinates, CRD groups, integration points verified against repo + WEKA docs / Artifact Hub. Lowest-confidence values are the *informational* node-prereq snippet (`strict-cpu-reservation` key, `25000` hugepages) — low blast radius since the customer applies them. |
| Features | HIGH | Maps to authoritative PRD (decisions A-E) + verified `/deploy-stream` SSE contract; UX framing from established installers (MEDIUM, not re-verified live). |
| Architecture | HIGH | All claims verified against current source. The one stated "operator: no code change" conclusion is **contested by Pitfalls** — see gap below. |
| Pitfalls | HIGH | Verified against repo code (exact line refs) and official Helm/Kubernetes/WEKA docs. The Decision C reversal is the most consequential finding and is well-evidenced. |

**Overall confidence:** HIGH on the integration mechanics; **MEDIUM-to-resolve on the operator quay chart-pull auth** until the Phase 0 spike runs.

### Gaps to Address

- **Operator quay chart-pull auth (Decision C) — THE gap.** STACK/ARCHITECTURE say "no operator change"; PITFALLS says the operator must do `helm registry login`/`--registry-config` because `imagePullSecrets` don't authenticate `helm pull`. Reconciliation: the user's manual success used a workstation login the operator pod lacks; no existing component exercises an authenticated OCI pull, so it's untested. **Handle:** Phase 0 go/no-go spike before committing the roadmap; Phase 2 scope is conditional on the result. Do not let the roadmap assume Decision C is settled.
- **`discover_chart_crds` failure-caching** is coupled to the auth gap (a 401 on the first call caches an empty CRD set permanently). Fix alongside the auth resolution.
- **SSE creds-in-GET-query-string** leakage to proxy/access logs — `/deploy-stream` is GET today; evaluate a POST variant or short-lived server-side variable staging in Phase 3.
- **Exact node-prereq values** (`strict-cpu-reservation` option key, `25000` hugepage count) are MEDIUM confidence; the PRD is authoritative for the displayed snippet and the customer adapts it — low blast radius.
- **Operator `HELM_CMD_TIMEOUT`** vs worst-case chart install (CRD/hook waits) — confirm it's >= the chart's install time, independent of the SSE deadline.

## Sources

### Primary (HIGH confidence)
- Repo code: `operator_module/main.py` (`:116-118` OCI repo-add skip, `:136-178` `_install_chart`/`_upgrade_chart` pass no helm auth, `:441`/`:476-509` `_b64`/`_derive_ngc_payloads` dockerconfigjson reference, `:673-703` `discover_chart_crds` no-auth + `lru_cache` + swallow-failure, `:824` `resolve_dependencies`, `:865` `wait_for_component_ready`, `:1002-1254` sequential component loop, `:1161` CRD-skip heuristic)
- Repo code: `app-store-gui/webapp/main.py` (`:40-129` `ClusterInitMiddleware`, `:1801` `find_blueprint`, `:1694` `parse_x_variables`, `:2361` `/cluster-info`, `:2839-3011` `/deploy-stream` incl. `:2935` annotation stamping, `:2956` 900s deadline, `:2963` keepalive, `:2984` SSE message passthrough)
- Repo code: `app-store-gui/webapp/planning/apply_gateway.py` (`:24` StorageClass cluster-scoped, `:206-224` namespace normalization, `:301`/`:323` 409-only handling — no 404)
- Repo templates: `weka-csi-config/{wekaClientCR-online.yaml, weka-client-cluster-dev.yaml, csi-wekafs-api-secret.yaml, quay-secret.yaml, storageclass-*.yaml}`, `weka-csi-config/weka-operator/Chart.yaml` + `crds/`, `cluster_init/app-store-cluster-init.yaml`
- `.planning/PRD-install-wizard-weka-storage-stack.md` — decisions A1/B/C/D/E, trailing-newline fix note, risks, success criteria
- [WEKA Operator deployments — WEKA docs](https://docs.weka.io/kubernetes/weka-operator-deployments) — `oci://quay.io/weka.io/helm/weka-operator`, requires QUAY creds + quay secret in both namespaces, **not anonymous-pullable**
- [Use OCI-based registries — Helm](https://helm.sh/docs/topics/registries/) — `helm registry login` authenticates chart pulls; distinct from kubelet `imagePullSecrets`
- [Control CPU Management Policies on the Node — Kubernetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) — policy change requires delete `cpu_manager_state` + kubelet restart (why A1 stays manual)
- [csi-wekafsplugin on Artifact Hub](https://artifacthub.io/packages/helm/csi-wekafs/csi-wekafsplugin) — latest `2.8.7`, public repo, API-based since 0.7.0
- [WEKA — Deploy the WEKA client on Amazon EKS](https://docs.weka.io/kubernetes/weka-operator-deployments/deploy-the-weka-client-on-amazon-eks) — CPU-manager static, `systemReserved.cpu`, hugepages, `weka.io/supports-clients=true`, WekaClient `weka.weka.io/v1alpha1` fields

### Secondary (MEDIUM confidence)
- [Pulling Helm charts from private OCI registry — argo-cd #21060](https://github.com/argoproj/argo-cd/discussions/21060) — `imagePullSecrets` do not authenticate helm chart pulls (corroborates Pitfall 1)
- Established infra-installer UX conventions (Rancher, Longhorn/OpenEBS, Vault/Consul guided setup) — table-stakes/anti-feature framing

### Tertiary (LOW confidence — needs validation)
- Exact `strict-cpu-reservation` option key and `25000` hugepage count — PRD-specified; informational snippet, customer adapts (low blast radius)

---
*Research completed: 2026-06-24*
*Ready for roadmap: yes — pending the Phase 0 quay-helm-auth go/no-go spike, which gates Phase 2 scope*
