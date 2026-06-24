# Architecture Research — v8.0 Guided Install Wizard Integration

**Domain:** Brownfield integration of a multi-step install wizard + parameterized AppStack blueprint into the existing WEKA App Store (FastAPI/Jinja GUI + Kopf operator + `WekaAppStore` CR).
**Researched:** 2026-06-24
**Confidence:** HIGH (all claims verified against current source — `main.py`, `operator_module/main.py`, `apply_gateway.py`, `cluster_init/`, `weka-csi-config/`)

This is an **integration map**, not a greenfield architecture. The AppStack runtime already does ordered, dependency-gated, CRD-installing, readiness-checked deployment with live `componentStatus` progress. The v8.0 work is overwhelmingly **additive**: one new parameterized blueprint, parameterized `weka-csi-config/` components, a multi-step form, and three small backend helpers. The operator needs **no code change** (verified — Decision C holds).

---

## System Overview — Where the Wizard Plugs In

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BROWSER                                                                   │
│  welcome.html  (MODIFIED: single-button → 5-step stepper + live stage list)│
│     │ Step 1 node-prereq confirm  Step 2 quay creds  Step 3 WEKA conn      │
│     │ Step 4 WEKA creds  Step 5 review + namespace                         │
│     │  builds variables{} incl. GUI-derived quay_dockerconfigjson,         │
│     │  join_ip_ports (YAML-list form) + endpoints_csv (comma form)         │
│     ▼                                                                      │
│  EventSource('/deploy-stream?app_name=app-store-install&variables=...')    │
│     │ … on complete ok:true →                                             │
│     ▼                                                                      │
│  EventSource('/deploy-stream?app_name=cluster-init&variables=...')         │
│     │ … on complete ok:true → redirect to redirect_url or "/"             │
└──────────────────────────────────────────────────────────────────────────┘
                                   │ SSE
┌──────────────────────────────────┼─────────────────────────────────────────┐
│  FASTAPI GUI  (app-store-gui/webapp/main.py)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ ClusterInitMiddleware (UNCHANGED)  — /deploy-stream already exempt     │ │
│  │   gate = app-store-cluster-init CR appStackPhase == "Ready"           │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ find_blueprint()  (MODIFIED: add app-store-install special-case OR    │ │
│  │   rely on x-variables discovery in cluster_init/app-store-install.yaml)│ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ /deploy-stream  (MOSTLY UNCHANGED: render [[ ]] Jinja2, apply, poll    │ │
│  │   status.componentStatus → SSE component events; raise 900s deadline)  │ │
│  │   namespace-preserve special-case (":2943") must also match           │ │
│  │   app-store-install (components target FIXED namespaces)              │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │ NEW helpers: build_quay_dockerconfigjson(), split_endpoints()         │ │
│  │ /cluster-info (UNCHANGED endpoint; UI no longer hard-blocks on it)    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│       │ apply_blueprint_documents_with_namespace → ApplyGateway (UNCHANGED) │
│       ▼  create→409→patch WekaAppStore CR via CustomObjectsApi             │
└────────┼────────────────────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  KOPF OPERATOR  (operator_module/main.py)  — NO CODE CHANGE                 │
│  handle_appstack_deployment:                                                │
│    resolve_dependencies() topo-sort → deploy IN ORDER, one at a time        │
│    helmChart → install_or_upgrade (OCI skips repo add) → wait_for_ready     │
│    kubernetesManifest → kubectl apply (server-side, CRD-aware) → Ready      │
│    patches status.componentStatus after each component (live progress)      │
└──────────────────────────────────────────────────────────────────────────┘
         │ kubectl/helm subprocess
         ▼
   Cluster: quay secrets → weka-operator(+CRDs) → node labels → WekaClient(+secret)
            → csi-wekafs → csi-api-secret → 3 StorageClasses  ║  then cluster-init
```

---

## The New CR: `cluster_init/app-store-install.yaml` — Component Dependency Graph

A **single** parameterized `WekaAppStore` (`metadata.name: app-store-install`) whose `appStack.components[]` encode the ordered install. The operator's `resolve_dependencies()` (Kahn topo-sort, `operator_module/main.py:824`) deploys strictly in dependency order, **one component at a time**, blocking on `wait_for_component_ready` for any component with `waitForReady: true`.

```
quay-pull-secret-operator-ns   quay-pull-secret-default-ns
   (Secret, weka-operator-system)  (Secret, default)
        │                               │
        └───────────────┬───────────────┘
                        ▼
                  weka-operator
        (helmChart oci://quay.io/weka.io/helm/weka-operator,
         version [[ operator_version ]], readinessCheck:deployment
         in weka-operator-system, waitForReady:true)   ◄── CRDs installed here
                        │
          ┌─────────────┼──────────────────────────┐
          ▼             ▼                           ▼
   node-label-job   weka-client-secret        csi-wekafs
   (Job kubectl     (Secret stringData,       (helmChart csi-wekafs/
    label --all,     weka-client-cluster-dev)  csi-wekafsplugin, ns csi-wekafs,
    readiness:job)        │                    readiness:deployment
          │               ▼                     csi-wekafs-controller)
          │          weka-client                     │
          └─────────►(WekaClient CR,                 ▼
                      kubernetesManifest)        csi-api-secret
                      depends weka-operator       (Secret stringData,
                      + weka-client-secret         csi-wekafs-api-secret)
                                                       │
                                                       ▼
                                                 storageclasses
                                                 (3 docs, dir-api = default)
                                                 depends csi-wekafs + csi-api-secret
```

### Component table (recommended `dependsOn` / `readinessCheck`)

| Component | Type | targetNamespace | dependsOn | readinessCheck | Notes |
|-----------|------|-----------------|-----------|----------------|-------|
| `quay-pull-secret-operator` | manifest (Secret, dockerconfigjson) | `weka-operator-system` | — | none | from `[[ quay_dockerconfigjson ]]` |
| `quay-pull-secret-default` | manifest (Secret) | `default` | — | none | second copy (Decision B) |
| `weka-operator` | helmChart OCI | `weka-operator-system` | both quay secrets | `deployment` | **CRD-install gate**; OCI skips repo add (`operator:118`) |
| `node-label-job` | manifest (Job) | `default`/`kube-system` | `weka-operator` | `job` | needs cluster-admin SA (mirror `gateway-api-crds-job`) |
| `weka-client-secret` | manifest (Secret `stringData`) | `default` | `weka-operator` | none | `org/username/password` |
| `weka-client` | manifest (WekaClient CR) | `default` | `weka-operator`, `weka-client-secret` | none | **must wait for operator CRDs** — see risk below |
| `csi-wekafs` | helmChart (public) | `csi-wekafs` | `weka-operator` | `deployment csi-wekafs-controller` | no auth |
| `csi-api-secret` | manifest (Secret `stringData`) | `csi-wekafs` | `csi-wekafs` | none | `endpoints`=`[[ endpoints_csv ]]`, `scheme` |
| `storageclasses` | manifest (3 StorageClass docs) | cluster-scoped | `csi-wekafs`, `csi-api-secret` | none | `dir-api` is default |

> The operator resolves `targetNamespace` for helm via component → `namespaces` map → `defaultNamespace` → CR ns (`operator:1126-1134`). For manifests it uses `component.targetNamespace` or CR ns (`operator:1202`). StorageClass/Namespace/CRD are recognized cluster-scoped by `apply_gateway.CLUSTER_SCOPED_KINDS` and by `_apply_manifest_multi_ns` namespace-arg suppression in the operator.

---

## Critical Integration Risk: WekaClient CR Applied Before Its CRD Exists (404)

**This is the one ordering hazard that must be designed for, and it is solvable with config only — no operator change.**

How the operator processes components (verified `operator:1068-1254`):
- Components run **sequentially in topo order**. The loop does not start component N+1 until component N reaches a terminal phase.
- For `helmChart` components with `waitForReady: true` (default is `True`, `operator:1187`), the loop **blocks** in `wait_for_component_ready` until the configured `readinessCheck` passes (e.g. `kubectl wait --for=condition=available deployment/...`).
- For `kubernetesManifest` components, apply is `kubectl apply -f -` (`operator:399`) — **server-side and CRD-aware**. There is **no** readiness wait for manifests; they mark `Ready` immediately on a zero exit code.

Therefore the safe gate for the `WekaClient` manifest is:

1. `weka-client.dependsOn: [weka-operator]` (and `weka-client-secret`), AND
2. `weka-operator` has `waitForReady: true` + `readinessCheck: { type: deployment, name: <operator-deploy>, namespace: weka-operator-system }`.

Because the operator chart's own controller **Deployment cannot become `Available` until its CRDs are registered and served** (the operator pod registers/serves `weka.weka.io/v1alpha1`), gating WekaClient behind operator-deployment readiness transitively guarantees the `WekaClient` CRD is installed *and being served* by the API server before the WekaClient `kubectl apply` runs. This closes the 404 race.

**Failure modes to verify during E2E (PITFALLS feed):**
- Operator chart with `crdsStrategy: Auto` and `--skip-crds` heuristics (`operator:1161`, `should_skip_crds_for_component`): confirm the operator chart actually *installs* the WEKA CRDs (it must NOT be skipped). If the chart ships CRDs in `crds/`, helm installs them by default; verify `helm show crds` discovery doesn't trip the skip path for this chart.
- `readinessCheck.name` must match the **actual** operator Deployment name in `weka-operator-system` (don't rely on the label-selector fallback for a chart whose labels you haven't verified). Use `type: deployment` + explicit `name`.
- If the operator chart marks "Available" before CRDs are served (rare, but possible with separate CRD-install jobs), add a tiny `node-label-job`-style "wait-for-crd" Job between operator and weka-client as a belt-and-suspenders gate. **Flag for the operator/E2E phase.**

---

## New vs Modified — Explicit Inventory

### NEW files
| File | What it is |
|------|------------|
| `cluster_init/app-store-install.yaml` | Parameterized `WekaAppStore` CR + `x-variables` block + ConfigMaps for any helm `valuesFiles`. Uses `[[ var ]]` Jinja2 delimiters. |
| (parameterized) `weka-csi-config/*.yaml` content | Folded into the blueprint's `kubernetesManifest` strings (or referenced); converted to `[[ var ]]` + `stringData`. |

### MODIFIED files
| File | Change | Anchor |
|------|--------|--------|
| `app-store-gui/webapp/templates/welcome.html` | Replace single-button init with 5-step stepper; remove `handleInitialize` prereq hard-block; status dots → live stage rows driven by `component` SSE events; chain two `EventSource` streams. | whole React component (503 lines) |
| `app-store-gui/webapp/main.py` `find_blueprint` | Add `app-store-install` → `cluster_init/app-store-install.yaml` special-case (mirror `cluster-init` at `:1806`), OR rely purely on `x-variables` discovery (the file lives under `cluster_init/`; dir-name match `parent_dir == app_name` would require the dir renamed — an explicit special-case is cleaner). | `main.py:1801` |
| `app-store-gui/webapp/main.py` `/deploy-stream` | Extend the `cluster-init` namespace-preserve special-case (`ns_for_apply = "" if app_name in {"cluster-init","app-store-install"}`) so fixed component namespaces are preserved (`:2943`); keep required-field validation ON for app-store-install (it has real required fields — do NOT add it to the validation exemption at `:2874`); raise SSE `deadline` from 900s for the longer install (`:2956`). | `main.py:2874`, `:2943`, `:2956` |
| `app-store-gui/webapp/main.py` NEW helpers | `build_quay_dockerconfigjson(user, password)` (mirror operator logic `operator:476-509`; produce `kubernetes.io/dockerconfigjson`) and `split_endpoints(raw)` → `(yaml_list, csv)`; inject both into `user_vars` before render (server-side, so secrets/derivation aren't trusted from the browser — recommended). | `main.py` new functions |

### UNCHANGED (verified — do not touch)
| Component | Why it stays |
|-----------|--------------|
| `operator_module/main.py` | OCI install, CRD discovery, `dependsOn` topo-sort, `readinessCheck` (pod/deployment/job), `valuesFiles`, `${VAR}` render, `stringData` passthrough (kubectl apply), per-component `componentStatus` patching — all already present. Decision C: quay `dockerconfigjson` secrets cover chart+image pulls; no `helm registry login`. |
| `ClusterInitMiddleware` (`main.py:40-129`) | Gate is `app-store-cluster-init` CR `appStackPhase == "Ready"`. Since cluster-init runs **last** (Decision D/E), the gate naturally fires at the right moment. `/deploy-stream` already in `exempt_paths` (`:43`). |
| `apply_gateway.py` `ApplyGateway` | Already handles WekaAppStore create→409→patch, cluster-scoped kinds incl. `StorageClass`/`CustomResourceDefinition`/`Namespace`, and per-component `targetNamespace` override. |
| `cluster_init/app-store-cluster-init.yaml` | Runs as-is, just later (Non-Goal: don't change its component set). |
| `/cluster-info`, `/cluster-status` endpoints | Stay; `/cluster-info` becomes an informational "already installed?" hint instead of a hard gate. |

---

## Data Flow — Variable Fan-out (single form field → multiple sinks)

One browser form, one `variables{}` JSON, rendered once by `/deploy-stream` with `[[ ]]` delimiters (`main.py:2918`). The GUI derives the fan-out **before** render so the blueprint stays declarative:

```
Step 3 "Join endpoints" textarea (comma/newline)
   └─ split_endpoints() ──► join_ip_ports  = ["h:p", ...]  (YAML list literal in WekaClient.spec.joinIpPorts)
                            endpoints_csv   = "h:p,h:p,..." (string in csi-api-secret.endpoints)

Step 2 quay user+pass
   └─ build_quay_dockerconfigjson() ──► quay_dockerconfigjson  (full .dockerconfigjson JSON → both Secret copies)

Step 3 image version  ──► weka_image_version  → WekaClient.spec.image: quay.io/.../weka-in-container:[[ weka_image_version ]]
Step 3 scheme dropdown ──► scheme             → csi-api-secret.scheme
Step 2 operator version ──► operator_version  → weka-operator helmChart.version
Step 4 org/user/pass   ──► org/username/password → weka-client-secret (stringData) + csi-api-secret (org→organization)
Step 5 namespace        ──► namespace (note: components target FIXED namespaces; namespace var mostly cosmetic here)
```

**Secret hygiene (PITFALLS feed):** quay token + WEKA password must never reach the SSE log box. The SSE stream emits only component `name`/`phase`/`message` (`main.py:2980-2985`) — safe as long as `stringData` secret values never appear in a component `message`. Derive `quay_dockerconfigjson` server-side from POSTed creds (don't round-trip the raw base64 through the browser URL). Note `/deploy-stream` is **GET** today — long dockerconfigjson + creds in a query string is a leak risk to proxy/access logs; **flag: may need a POST variant or short-lived server-side staging of variables.**

---

## SSE / Two-Chained-CR Control Flow (frontend)

```
stage list = union(app-store-install components, cluster-init components)
open EventSource(app-store-install)
  on 'init'      → render app-store-install stage rows
  on 'component' → set row name→phase (Pending/Installing/Ready/Failed)
  on 'complete' ok:false → show error + Retry (re-open same stream; apply-or-patch is idempotent)
  on 'complete' ok:true  → close; open EventSource(cluster-init)
open EventSource(cluster-init)
  on 'component' → cluster-init rows
  on 'complete' ok:true → success → redirect (redirect_url || "/")
```

This matches the existing contract exactly; the only new frontend logic is (a) stepper state, (b) chaining the second stream on the first's success, (c) mapping `component` events to a vertical stage list instead of two dots. Note: cluster-init emits no per-component progress today — `/deploy-stream` reports it complete immediately after apply (`main.py:2948`) because the wizard then relies on the polling gate; the cluster-init stage rows will jump straight to done once the CR is applied, with `ClusterInitMiddleware`/`is_cluster_initialized_anywhere` confirming `Ready` for the redirect.

---

## Patterns to Follow (established in this repo)

- **`[[ var ]]` for GUI substitution, `${VAR}` for operator substitution** — keep them separate (Key Decision, v7.0). The blueprint is rendered once by the GUI with `[[ ]]`; do **not** also push these into `spec.appStack.variables` for `${VAR}` unless a helm `valuesFile` needs them.
- **`x-variables` self-describing schema** — gives free required-field + format validation in `/deploy-stream` (`main.py:2884`). Author it so wizard form fields map 1:1, plus the GUI-derived vars (`quay_dockerconfigjson`, `join_ip_ports`, `endpoints_csv`) marked `validate: false` or omitted from the schema.
- **`stringData` over `data`** — eliminates the base64 trailing-newline bug class (PRD risk, already fixed in committed files). Author all wizard-generated secrets with `stringData`.
- **Idempotent re-run** — WekaAppStore + Secrets are create→409→patch (`apply_gateway:300/322`). The `node-label-job` and any wait-for-crd Job must be no-ops on re-run (deterministic name; `kubectl label --all` is idempotent; mirror the `gateway-api-crds-job` "check then apply, exit 0 if present" pattern).

## Anti-Patterns to Avoid

- **Don't add operator code for quay auth.** Decision C verified: OCI path skips `helm repo add` (`operator:118`); namespace dockerconfigjson secrets cover chart + image pulls.
- **Don't put readinessCheck on the WekaClient manifest** — operator manifests have no readiness wait; gate via `dependsOn` on the operator helm component instead.
- **Don't override component namespaces from the wizard.** Pass `ns_for_apply = ""` for `app-store-install` so `_normalize_document_namespace` (`apply_gateway:206`) does not rewrite the fixed `weka-operator-system` / `csi-wekafs` targets.
- **Don't widen `/deploy-stream` GET with secrets in the query string** without considering URL leakage to proxy/access logs.

---

## Suggested Dependency-Aware Build Order

Ordered so each phase is independently verifiable and later phases depend only on earlier ones.

1. **Blueprint authoring** — write `cluster_init/app-store-install.yaml` (all 9 components + `x-variables`), parameterize `weka-csi-config/` templates into it with `[[ ]]` + `stringData`. Verify the dependency graph topo-sorts (operator `resolve_dependencies` unit test) and renders to valid YAML with sample vars. *Verify: `yaml.safe_load_all(env.render(...))` succeeds; topo order correct.*
2. **Operator/E2E ordering confirmation** — confirm the operator chart installs (not skips) WEKA CRDs and that `readinessCheck: deployment` on the operator gates WekaClient correctly (the 404-race). Add a wait-for-crd Job only if the operator deployment can go Available before CRDs are served. *Verify: WekaClient applies without 404 on a fresh cluster; no operator code diff.*
3. **Backend wiring** — `find_blueprint` entry; `build_quay_dockerconfigjson`; `split_endpoints`; extend the `cluster-init` namespace-preserve special-case to `app-store-install`; raise SSE deadline. *Verify: GUI unit tests for the two helpers; `/deploy-stream?app_name=app-store-install` locates the blueprint and renders.*
4. **Frontend stepper** — convert `welcome.html` to 5-step form + Step 1 node-prereq snippet/checkbox; remove prereq hard-block; live stage list; chain `app-store-install` → `cluster-init`; redirect on cluster-init Ready. *Verify: form builds correct `variables{}`; stage rows update; redirect fires.*
5. **E2E** — fresh cluster (no operator/CSI) → full install → 3 StorageClasses (dir-api default) → cluster-init → redirect. Failure/retry. Re-run idempotency. Secret-leak check on the log box. *Verify: PRD Success Criteria 1–5.*

Build order rationale: the blueprint (1) is the contract everything else targets; the operator-ordering confirmation (2) de-risks the single hard integration hazard before any UI work; backend (3) and frontend (4) consume the blueprint; E2E (5) validates the chained-CR end state. Phases 1–4 are parallelizable across blueprint/operator vs backend/frontend once (1) lands.

---

## Sources

- `app-store-gui/webapp/main.py` — `ClusterInitMiddleware` (:40-129), `find_blueprint` (:1801), `parse_x_variables` (:1694), `/deploy-stream` (:2839-3011), `/cluster-info` (:2361), `/cluster-status` (:2432). HIGH (current source).
- `app-store-gui/webapp/planning/apply_gateway.py` — create→409→patch (:260-336), cluster-scoped kinds (:17-34), namespace normalization (:206-224). HIGH.
- `operator_module/main.py` — `resolve_dependencies` topo-sort (:824), `wait_for_component_ready` (:865), `handle_appstack_deployment` sequential loop (:1002-1254), manifest `kubectl apply` (:389-403), OCI repo-add skip (:116-118), CRD-skip heuristic (:1161), per-component progress patch (:1247). HIGH.
- `cluster_init/app-store-cluster-init.yaml` — precedent for cluster-admin SA + Job pattern (`gateway-api-crds-*`), OCI helm (`oci://docker.io/envoyproxy`), readinessCheck shapes. HIGH.
- `weka-csi-config/{wekaClientCR-online.yaml, weka-client-cluster-dev.yaml, csi-wekafs-api-secret.yaml, quay-secret.yaml, storageclass-*.yaml}` — source manifests to parameterize. HIGH.
- `.planning/PRD-install-wizard-weka-storage-stack.md` — resolved decisions A1/B/C/D/E. HIGH (authoritative spec).
