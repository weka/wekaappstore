# Feature Research

**Domain:** Guided multi-step install wizard for infrastructure (Kubernetes operator + CSI driver + credentials + storage classes), web-based, with live per-stage progress
**Researched:** 2026-06-24
**Confidence:** HIGH

This research is scoped to the **v8.0 Guided Install Wizard** described in `.planning/PRD-install-wizard-weka-storage-stack.md`. The PRD has already resolved decisions A–E, so this document does not re-open those. Its job is to categorize the wizard's behaviors as **table stakes / differentiators / anti-features**, note complexity, and pin each to its dependency on the existing GUI/SSE machinery (`/deploy-stream`, `componentStatus` events, `parse_x_variables`, `welcome.html`).

> Prior-milestone feature research (v5.0 `${VAR}` substitution) previously occupied this file; v3.0/v4.0 are archived under `.planning/research/v3.0/` and `v4.0/`. This file now reflects the active v8.0 milestone.

Domain comparables consulted (conceptually, from established install-wizard UX conventions): Rancher cluster/driver install flows, Longhorn/OpenEBS storage-class setup, the kubeadm/`kubectl` "prereq → install → verify" pattern, Vault/Consul guided setup, and generic credential-collection installers (Docker registry login, cloud-provider connect screens). These inform what users *expect* from such a wizard versus what reliably backfires.

---

## How these wizards typically work (expected behavior)

A guided infra installer almost universally follows the shape: **(1) gather prereqs and confirm the environment is ready → (2) collect credentials and connection details across a small number of grouped steps → (3) review a masked summary → (4) submit and watch an ordered, per-stage progress list resolve to Done/Failed → (5) on terminal success, hand the user off to the product.** Steps are grouped by *concern* (prereqs, registry auth, target connection, target auth), not by underlying resource — the user should never see "now creating secret #2 of 3." Validation is per-step and blocks forward navigation. Progress is driven by the *backend's* real status, not a fake timer. Failure stops the run at the failing stage, shows which stage failed and why, and offers a safe retry that re-runs from a clean apply rather than forcing a full form re-entry. The v8.0 design matches this shape exactly, which is why most of it is table stakes rather than novel.

---

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes / Dependencies |
|---------|--------------|------------|----------------------|
| Multi-step stepper UI (5 steps) replacing the single Initialize button | Any infra installer collecting >3 fields groups them into steps; a single giant form feels broken | MEDIUM | New frontend in `welcome.html` (MUI `Stepper` or simple step state). Existing single-React-root + CDN constraint applies. No new deps. |
| Per-step forward-blocking validation | Users expect "Next" to be disabled / error until the current step is valid; don't fail at submit time on a field from step 2 | MEDIUM | Pure frontend. Each step validates its own fields before advancing. |
| Host:port endpoint list input (`joinIpPorts`) accepting comma/newline-separated entries | Multi-endpoint join targets are normal for clustered storage; one field that accepts several is the convention | MEDIUM | Frontend parse + validate each entry as `host:port`. Backend helper must derive **both** YAML-list form `["h:p", ...]` and comma-joined string from one field (PRD "Endpoint transform"). |
| Masked secret inputs (quay password/token, WEKA password) with show/hide toggle | Password fields are masked by default everywhere; show/hide is standard | LOW | Frontend `type=password` + toggle. |
| Scheme dropdown (`http`/`https`) | Enumerated choice → dropdown, not free text; prevents typos that break the CSI API secret | LOW | Already have MUI `Select` in `welcome.html`. Feeds `scheme` in csi-api-secret. |
| Version/tag inputs with sensible defaults (operator `v1.13.0`, WEKA image `5.1.0.605`) | Users expect prefilled defaults they can override, not a blank required field | LOW | Frontend default values; backend treats as plain `[[ var ]]`. |
| Read-only review step with secrets masked | Confirm-before-apply is universal; showing masked secrets proves they were captured without leaking | LOW | Frontend. Reuse namespace selector already in `welcome.html`. |
| Live per-stage progress list (Pending → In-progress → Done/Failed per stage) | The product's whole value is replacing a long manual `kubectl`/`helm` sequence; users must *see* each stage land | MEDIUM | **Reuses existing `componentStatus` SSE** — `/deploy-stream` already emits `{type:"component", name, phase, message}` on each component phase change (main.py:2980). Map each `name` to a stage row. No backend stream change. |
| Scrolling log box alongside the stage list | Already present today; users expect raw detail available when a stage stalls | LOW | Reuse existing `.log-container` + log event handling in `welcome.html`. |
| Clear error surface on partial failure (which stage, why) | On failure users must know *what* failed, not just "failed" | LOW | `/deploy-stream` already emits `{type:"complete", ok:false, message:"<comp>: <msg>"}` on `appStackPhase==Failed` (main.py:2990-2994). Render to the failing stage row. |
| Retry after failure without re-entering the whole form | Re-typing credentials after a transient failure is the #1 abandonment cause | MEDIUM | Keep form state in React; re-open `EventSource('/deploy-stream?app_name=app-store-install')`. Apply-or-patch on WekaAppStore CRs + secrets makes re-run safe (PRD risk note). |
| Idempotent re-run on an already-partially-installed cluster | Operators *will* re-run; destructive/duplicate actions on re-run are unacceptable | MEDIUM | WekaAppStore CR + secrets already create→409→patch (apply_gateway.py). **Node-label Job must be a no-op on re-run** (`kubectl label --all` is idempotent; verify). |
| Final chained cluster-init → redirect | Preserves today's end-state; cluster-init `Ready` is the single redirect gate (Decision E) | MEDIUM | After `app-store-install` reaches `Ready`, apply the untouched `app-store-cluster-init` CR; redirect on its `Ready` via existing `/cluster-status` poll. |
| Secrets never appear in logs / SSE stream | Leaking a quay token or WEKA password into the log box is a security incident | MEDIUM | SSE only emits component `name`/`message` — verify component messages never echo secret values. GUI builds `dockerconfigjson`; raw creds never templated into manifest text (Decision B). |
| Cluster-name / target header + namespace selector | Confirms *where* the install lands; both exist today | LOW | Keep from current `welcome.html` (`/cluster-info`, `/api/namespaces`). |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes / Dependencies |
|---------|-------------------|------------|----------------------|
| Single field driving multiple destinations (endpoints → WekaClient YAML list + CSI secret comma-string; one credential set → multiple secrets) | User enters connection details once; wizard fans them out to WekaClient CR, weka-client secret, and csi-api-secret. Removes the most error-prone part of the manual procedure | MEDIUM | Backend endpoint transform + variable→field mapping (PRD table). The value-add over a raw form. |
| Prereq detection used as a *skip hint* rather than a hard block | If `/cluster-info` shows operator/CSI already present, let the user skip ahead instead of failing — turns the old hard-block into helpful guidance | LOW | Reuse `/cluster-info` `weka_operator_installed` / `weka_csi_installed`. Remove the `handleInitialize` hard-block (PRD frontend note). |
| Copy-paste `KubeletConfiguration` snippet on the node-prereq step | The node config (CPU manager `static`, `strict-cpu-reservation`, hugepages `25000`) is the one thing the wizard *can't* do for the user; handing them the exact snippet + a confirm checkbox de-risks it | LOW | Frontend static content + checkbox gate (Decision A1). |
| Stage list that reflects *real* operator status, not a synthetic timer | Honest progress (driven by `componentStatus`) builds trust; many installers fake it | LOW | Already how `/deploy-stream` works — differentiator is *exposing* it as a labeled stage list. |
| GUI-built `dockerconfigjson` for the quay pull secret | User types username/password; wizard assembles the docker-registry secret payload — no manual base64/JSON wrangling (the manual procedure's worst step) | MEDIUM | Reuse operator helper logic at `operator_module/main.py:476`. Inject as one `[[ quay_dockerconfigjson ]]` var. |
| `stringData` secrets instead of pre-encoded base64 | Eliminates the trailing-newline base64 bug class entirely (PRD risk note, fixed 2026-06-24) | LOW | Author secret components with `stringData` in the new blueprint. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Auto-configure / restart kubelet & hugepages on worker nodes** | "The wizard should do *everything*" | Requires privileged node-level/root access and a kubelet restart that can disrupt running workloads; out of the App Store's safety envelope | Decision A1: show the `KubeletConfiguration` snippet + confirm checkbox. App Store does not write node config or restart kubelet. (Privileged DaemonSet is a possible *future* follow-on, explicitly out of scope.) |
| **`helm registry login` step / storing registry creds in helm config** | Mirrors the manual `helm registry login quay.io` muscle memory | Adds an auth surface and operator change for no benefit — the in-namespace `dockerconfigjson` secrets already cover chart + image pulls | Decision C: rely on the quay `dockerconfigjson` secrets via `imagePullSecret`. No operator auth change. |
| **Provisioning the WEKA backend storage cluster** | "Install WEKA end-to-end" | The wizard connects *clients* to an existing NeuralMesh cluster; provisioning the backend is a wholly different, dangerous operation | Non-goal. Connect clients only; require an existing cluster (join endpoints). |
| **Day-2 ops in the wizard (upgrade/uninstall operator, rotate creds, edit StorageClasses)** | Natural "while we're here" scope creep | Bloats a first-run installer with destructive lifecycle operations; different risk profile and audience | Out of scope per PRD Non-Goals. Day-2 belongs in Settings/credential management later. |
| **Multi-cluster / multiple WEKA backend targets in one run** | "Configure all our clusters at once" | Multiplies form state, validation, and failure modes; the install is per-cluster by nature | Non-goal. One backend target per wizard run. |
| **Air-gapped / non-quay operator registries** | Enterprise/offline asks | Each alternate registry path is its own auth + image-mirror project | Out of scope per PRD. Assume quay.io. |
| **Free-text scheme / free-text "advanced YAML" override field** | Power users want to tweak the generated CR | Defeats validation, invites the exact base64/typo bugs the wizard removes, and produces unsupported configs | Constrain to the dropdown + typed fields. Generated CR is the contract. |
| **Synthetic/percentage progress bar driven by a timer** | Looks reassuring during long installs | Lies to the user; operator readiness (operator + CSI + WekaClient) can exceed the timer and the bar desyncs from reality | Drive the stage list from real `componentStatus` phases; use an indeterminate bar only while a stage is genuinely in-progress. |
| **Auto-retry-forever on stage failure** | "Just keep trying" | Masks real misconfig (bad creds, unreachable endpoints) and burns the SSE deadline | Stop at the failing stage, surface the message, offer an explicit user-triggered Retry. |

---

## Feature Dependencies

```
[Multi-step stepper UI]
    └──feeds──> [Endpoint transform (list + comma-joined)]
                    └──feeds──> [Parameterized app-store-install blueprint (x-variables)]
                                    └──applied by──> [/deploy-stream (unchanged stream contract)]
                                                         └──drives──> [Live per-stage progress list]
                                                                          └──on Ready──> [Chained cluster-init]
                                                                                             └──on Ready──> [Redirect to App Store]

[GUI-built dockerconfigjson] ──feeds──> [Parameterized app-store-install blueprint]
[Per-step validation] ──gates──> [Review step] ──gates──> [Submit / install stream]
[stringData secrets] ──enables──> [Idempotent re-run] (apply-or-patch, no encode drift)
[Prereq detection (/cluster-info)] ──enhances──> [Multi-step stepper UI] (skip hint)

[Auto-configure kubelet] ──conflicts──> [Node-prereq confirm checkbox (A1)]
[helm registry login] ──conflicts──> [GUI-built dockerconfigjson + imagePullSecret (Decision C)]
```

### Dependency Notes

- **Live progress requires the parameterized blueprint to expose components by name.** `/deploy-stream` emits one `component` event per `status.componentStatus[]` entry on phase change (main.py:2974-2985). Each component's `name` in `app-store-install.yaml` becomes a stage-list label — choose human-readable names (e.g. `weka-operator`, `csi-wekafs`, `storageclasses`).
- **Stage ordering is enforced by the operator, not the frontend.** `dependsOn` + `readinessCheck` in the blueprint gate WekaClient behind operator-CRD install + operator-deployment readiness (PRD risk: WekaClient apply 404s on a missing CRD). The frontend only *reflects* the order it observes.
- **Endpoint transform is the lynchpin of the "enter once" differentiator.** One field must yield both `joinIpPorts: ["h:p", ...]` (YAML list, WekaClient) and `endpoints: "h:p,h:p"` (string, csi-api-secret). Implement as a backend derived-variable helper before rendering.
- **Idempotent re-run depends on stringData + apply-or-patch + idempotent Job.** WekaAppStore CRs and secrets already patch on 409; the node-label `Job` (`kubectl label nodes --all`) must be verified idempotent so re-run is a no-op.
- **Redirect depends on the chained cluster-init reaching `Ready`.** `ClusterInitMiddleware` is unchanged (Decision E); the existing `/cluster-status` poll in `welcome.html` (lines 171-215) drives the final redirect. Keep that machinery.
- **Long-install risk vs the 15-minute SSE cap.** `/deploy-stream` hardcodes a 900s deadline (main.py:2956) with a `: ping` keepalive (main.py:2963). Operator + CSI + WekaClient readiness may exceed this — raising the deadline for this blueprint is a likely required backend change, flagged as a phase risk.

---

## MVP Definition

### Launch With (v1 — the v8.0 milestone)

- [ ] 5-step stepper (node prereq confirm → quay creds → WEKA connection → WEKA creds → review/install) — the core UX shift
- [ ] Per-step forward-blocking validation (host:port list, masked secrets, version tags, scheme dropdown) — prevents submit-time failures
- [ ] Backend endpoint transform (list + comma-joined derived variables) — the "enter once" value
- [ ] GUI-built `quay_dockerconfigjson` variable — removes the worst manual step (Decision B)
- [ ] Parameterized `cluster_init/app-store-install.yaml` with `x-variables` + ordered components, `stringData` secrets — the install payload
- [ ] Live per-stage progress list mapped from existing `component` SSE events — honest progress
- [ ] Error surface + user-triggered Retry preserving form state — recover from partial failure
- [ ] Chained `app-store-install` → `cluster-init` → redirect on `Ready` — preserves end-state (Decisions D/E)
- [ ] Secret-leak guard: no credential values in logs/SSE — security requirement
- [ ] Raised SSE deadline + keepalive for the longer install — operational requirement

### Add After Validation (v1.x)

- [ ] Prereq-detection skip path (auto-skip operator/CSI steps when `/cluster-info` reports them present) — trigger: users with pre-installed operators hit friction
- [ ] Store WEKA creds as a `WarpCredential` for reuse by storage-aware blueprints — trigger: downstream blueprints want the connection
- [ ] Per-stage "view logs" expander / copyable error detail — trigger: support load on diagnosing stalled stages

### Future Consideration (v2+)

- [ ] Privileged DaemonSet to auto-apply node kubelet/hugepage config — defer: high-risk node mutation, out of v8.0 scope
- [ ] Air-gapped / non-quay registry support — defer: separate enterprise effort
- [ ] Day-2 ops (upgrade/uninstall/rotate/edit) — defer: different audience and risk profile

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| 5-step stepper UI | HIGH | MEDIUM | P1 |
| Per-step validation | HIGH | MEDIUM | P1 |
| Endpoint transform (list + comma-joined) | HIGH | MEDIUM | P1 |
| GUI-built dockerconfigjson | HIGH | MEDIUM | P1 |
| Parameterized app-store-install blueprint | HIGH | HIGH | P1 |
| Live per-stage progress list (reuse SSE) | HIGH | LOW | P1 |
| Error + Retry preserving form state | HIGH | MEDIUM | P1 |
| Chained cluster-init → redirect | HIGH | MEDIUM | P1 |
| Secret-leak guard | HIGH | LOW | P1 |
| Raised SSE deadline | MEDIUM | LOW | P1 |
| Idempotent re-run (verify Job no-op) | MEDIUM | LOW | P1 |
| Node-prereq snippet + confirm checkbox | MEDIUM | LOW | P1 |
| Prereq-detection skip path | MEDIUM | LOW | P2 |
| Store WEKA creds as WarpCredential | MEDIUM | MEDIUM | P2 |
| Auto-configure kubelet (DaemonSet) | LOW | HIGH | P3 |

**Priority key:** P1 must-have for v8.0 launch · P2 add when possible · P3 future.

## Competitor Feature Analysis

| Feature | Rancher (driver/cluster install) | Longhorn/OpenEBS (storage class setup) | Our Approach |
|---------|----------------------------------|----------------------------------------|--------------|
| Step grouping | Multi-step by concern | Single-page form | 5 steps grouped by concern (prereq/registry/connection/creds) |
| Credential entry | Masked, validated per provider | Minimal | Masked quay + WEKA creds, GUI-assembled into K8s secrets |
| Progress display | Real status polling per resource | Helm status | Real `componentStatus` SSE per ordered stage |
| Node prereqs | Doc links / pre-checks | Pre-flight checks | Copy-paste snippet + confirm checkbox (no auto-config) |
| Default storage class | Manual annotation | Toggle | `storageclass-wekafs-dir-api` set default automatically |
| Idempotent re-run | Reconcile-based | Helm upgrade | Apply-or-patch CRs/secrets; idempotent label Job |

## Sources

- `.planning/PRD-install-wizard-weka-storage-stack.md` (authoritative spec; decisions A–E resolved) — HIGH
- `app-store-gui/webapp/main.py` `/deploy-stream` SSE contract (lines 2837-2999), `parse_x_variables`/`find_blueprint` (1694-1881), `componentStatus` polling — HIGH (verified in code)
- `app-store-gui/webapp/templates/welcome.html` current single-button flow, `/cluster-info`, `/cluster-status` poll, log box — HIGH (verified in code)
- `.planning/PROJECT.md` v8.0 milestone scope and prior-milestone machinery — HIGH
- Established infra-installer UX conventions (Rancher, Longhorn/OpenEBS, Vault/Consul guided setup, Docker registry login) for table-stakes/anti-feature framing — MEDIUM (domain knowledge, not re-verified against live docs)

---
*Feature research for: guided infra install wizard (WEKA operator + CSI + credentials + storage classes)*
*Researched: 2026-06-24*
