# Pitfalls Research

**Domain:** Guided install-wizard automation for the WEKA storage stack (operator + CSI + WekaClient + secrets + StorageClasses) installed via the existing AppStack `WekaAppStore` mechanism, fronted by an SSE-streamed multi-step web form.
**Researched:** 2026-06-24
**Confidence:** HIGH (most findings verified against the repo's own code and official Helm/WEKA/Kubernetes docs; the few WebSearch-only items are flagged inline)

These are the failure modes specific to *adding* this install automation to the brownfield App Store. The single most important finding — verified against the repo's operator code and the WEKA/Helm docs — is that **Decision C ("no `helm registry login`") is wrong as written**: the dockerconfigjson pull secrets cover *image* pulls but not the operator's *chart* pull. That is Pitfall 1 below and should be treated as the highest-risk item in v8.0.

---

## Critical Pitfalls

### Pitfall 1: `helm registry login` is required for the operator's chart pull — pull secrets do NOT cover it (Decision C is incorrect)

**What goes wrong:**
The operator installs the WEKA operator chart by shelling out to `helm install <name> oci://quay.io/weka.io/helm/weka-operator --version vX.Y.Z` and, before that, runs `helm show crds oci://...` to discover CRDs (`operator_module/main.py:679`, `discover_chart_crds`). Both commands run **inside the operator pod's own helm process**, which authenticates using the helm registry config (`~/.config/helm/registry/config.json`) — populated only by `helm registry login`. The quay `kubernetes.io/dockerconfigjson` secrets that Decision B/C create are consumed by the **kubelet** when pulling *container images* referenced by pods (the operator Deployment image, the weka-in-container image). They have **zero effect** on `helm pull` / `helm show crds`. Verified facts: (a) the operator's `_install_chart`/`_upgrade_chart`/`discover_chart_crds` pass **no** `--registry-config` and no credentials (`operator_module/main.py:136-178`, `:673-703`); (b) the WEKA operator chart on quay.io is **not anonymous-pullable** — WEKA's own docs require `QUAY_USERNAME`/`QUAY_PASSWORD` and a `helm registry login quay.io` step before pulling the chart.

So on a fresh cluster the operator install component fails at the `helm show crds` / `helm install` step with a `401 Unauthorized` / `unauthorized: access to the requested resource is not authorized` from quay — **before any pod, and therefore before any pull secret, is ever consulted.**

**Why it happens:**
Conflation of two distinct auth surfaces that *look* identical (both consume a dockerconfigjson). "We put a quay pull secret in the namespace, so quay auth is handled" is intuitively true for images and intuitively (but wrongly) extended to chart pulls. Helm chart pull auth and Kubernetes image pull auth are separate subsystems (verified — Helm docs: "use `helm registry login`"; Kubernetes imagePullSecrets are for kubelet image pulls only).

**How to avoid:**
The operator (which runs helm) must authenticate to quay before the OCI chart pull. Options, in order of preference:
1. **Have the operator run `helm registry login quay.io -u <user> -p <pass>` (or write `~/.config/helm/registry/config.json`) using the quay creds, immediately before any `oci://quay.io/...` chart operation.** This is a real operator change — it directly contradicts Decision C's "no operator change required for auth." Pass the creds to the operator via the same quay secret (mounted/read by the operator pod), not via the manifest text.
2. Alternatively, pass `--registry-config <path>` pointing at a config.json the operator materializes from the quay secret, on every `helm install`/`helm upgrade`/`helm show crds` invocation for OCI refs. This is more surgical than a global `helm registry login` and avoids mutating shared helm home state.
3. Have the GUI/operator detect a pre-existing `helm registry login` on the operator pod (rare) and skip — but do not *rely* on it.

Whatever the mechanism, the quay password must reach the operator's helm process, and `discover_chart_crds` (the very first quay touch, and `@lru_cache`d) must use the same auth or it will cache an empty CRD set on the first auth failure and never retry (see Pitfall 8).

**Warning signs:**
- Operator-install component goes `Failed` with `unauthorized` / `401` / `denied` in the operator log, on a cluster where the dockerconfigjson secrets are present and correct.
- `helm show crds` silently returns `{}` (it swallows `CalledProcessError`, `:685`), so CRD discovery quietly does nothing and the install proceeds without CRDs — then the WekaClient apply 404s later (Pitfall 4). The empty result is then cached.

**Phase to address:**
**Operator phase** (PRD "Phase 2 — Operator"). This phase is currently scoped as "confirm ordering, no auth change needed." That scope is wrong — re-scope it to *implement operator-side helm registry auth for OCI quay charts*. Flag for deeper research/spike before committing the v8.0 roadmap.

---

### Pitfall 2: `echo "x" | base64` trailing-newline corruption (the just-fixed bug) reappearing in any hand-encoded path

**What goes wrong:**
`echo "value" | base64` encodes a trailing `\n`, so the decoded secret value is `value\n`. For WEKA creds this silently breaks auth: the CSI driver and WekaClient send `password\n` / `username\n` / a scheme of `http\n` to the WEKA API and get rejected, or the endpoints string parses with a stray newline. This already bit the committed templates (`weka-client-cluster-dev.yaml`, `csi-wekafs-api-secret.yaml`) and was fixed 2026-06-24 by re-encoding `username`/`password`/`endpoints`/`scheme`. The failure is *silent at apply time* — the secret applies cleanly; only the runtime WEKA login fails, far downstream and with an opaque "authentication failed" error.

**Why it happens:**
`echo` appends a newline by default; `base64` faithfully encodes it. The base64 string still *looks* valid and the Secret object is well-formed, so nothing flags it until WEKA rejects the credential.

**How to avoid:**
- **Generated secrets must use `stringData`, not `data`** (PRD Decision; Goal 5). With `stringData`, the GUI puts the raw form value in and the API server base64-encodes it exactly — no manual `base64`, no `echo`, no newline class of bug. This is the correct call.
- For the *one* field that still requires hand-assembled base64 — the quay `.dockerconfigjson` `auth` value (Pitfall 3) — never use shell `echo|base64`. Reuse the operator's `_b64()` helper (`operator_module/main.py:441`, `base64.b64encode(s.encode()).decode()`), which has no newline. The GUI builds the whole dockerconfigjson in Python and injects it as one `[[ quay_dockerconfigjson ]]` var (Decision B) — keep it that way.
- Add a unit/round-trip test: for each generated secret, `base64decode(rendered) == raw_form_value` exactly (byte-for-byte, no trailing `\n`).

**Warning signs:**
- WEKA API auth failures at WekaClient / CSI runtime despite "correct" credentials.
- Any `| base64` in a script, Makefile, or template. Any committed `data:` field whose decode ends in a `\n` (e.g. base64 ending in `Cg==`).

**Phase to address:**
**Blueprint phase** (parameterize `weka-csi-config/` to `stringData`) + a guard test. **Backend phase** for the `quay_dockerconfigjson` builder.

---

### Pitfall 3: dockerconfigjson `auth` field assembled wrong (the field that actually authenticates)

**What goes wrong:**
A Kubernetes `kubernetes.io/dockerconfigjson` secret authenticates via `auths."quay.io".auth`, which must be `base64("username:password")` — and the *outer* `.dockerconfigjson` value is itself `base64(json)`. Three independent ways to get this wrong: (a) forget the inner `auth` and only set `username`/`password` (older Docker/containerd ignore them and fail to auth); (b) build `auth` with a newline (Pitfall 2) so `username:password\n` is sent and rejected; (c) double-encode or single-encode the wrong layer (forget the outer base64, or base64 the JSON twice). The reference implementation in this repo for NGC (`_derive_ngc_payloads`, `:499-509`) shows the correct shape: `auth = _b64(f"{user}:{pass}")`, then `.dockerconfigjson = _b64(json.dumps({"auths": {host: {username, password, auth}}}))`.

**Why it happens:**
The two-layer base64 + the `user:password` concatenation is fiddly and easy to half-remember. The image-pull failure it produces (`ImagePullBackOff: 401 unauthorized`) is generic and doesn't point at the encoding.

**How to avoid:**
- Build the dockerconfigjson in Python by analogy to `_derive_ngc_payloads`, substituting host `quay.io`, `username = QUAY_USERNAME`, `password = QUAY_PASSWORD`, `auth = _b64(f"{QUAY_USERNAME}:{QUAY_PASSWORD}")`. Include all three of `username`, `password`, `auth`.
- Inject the finished base64 dockerconfigjson as `[[ quay_dockerconfigjson ]]` into a `data:` field (it's already base64). Do **not** put it in `stringData` (that would double-encode).
- Create both copies (`weka-operator-system` and `default`) from the same computed value.
- Test: decode outer → JSON parses; `auths["quay.io"]["auth"]` decodes to exactly `QUAY_USERNAME:QUAY_PASSWORD` with no trailing bytes.

**Warning signs:**
- `ImagePullBackOff` / `401 unauthorized` pulling `quay.io/weka.io/...` images on the operator or WekaClient pods, despite a present `quay-pull-secret`.
- `kubectl get secret quay-pull-secret -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d` shows missing `auth` or malformed JSON.

**Phase to address:**
**Backend phase** (GUI dockerconfigjson builder) with a dedicated round-trip test.

---

### Pitfall 4: Applying the `WekaClient` CR before its CRD exists → 404 / "no matches for kind"

**What goes wrong:**
The `WekaClient` CR (`weka.weka.io/v1alpha1`, `wekaClientCR-online.yaml`) can only be applied after the WEKA operator's CRDs are installed *and registered in the API server's discovery cache*. If the AppStack applies it too early, `apply_gateway` gets a `404 NotFound` / `the server could not find the requested resource` / `no matches for kind "WekaClient"`. Note `apply_gateway` only special-cases `409` (already-exists → patch) for custom objects (`apply_gateway.py:301,323`); a `404` from a missing CRD is **not** handled and surfaces as a hard error, failing the component.

**Why it happens:**
CRD registration is asynchronous: even after `helm install` of the operator returns, the apiserver discovery cache can lag by seconds, and the operator Deployment (which may run conversion/validation webhooks) must be Ready. `dependsOn` alone (ordering the helm install before the CR) is necessary but **not sufficient** — you also need a readiness gate on the operator Deployment *and* ideally a poll that the CRD `weka.weka.io` is Established.

**How to avoid:**
- `dependsOn: [weka-operator]` on the `weka-client` component (PRD already plans this).
- Add `readinessCheck: { type: deployment, name: <operator deployment>, namespace: weka-operator-system }` so the WekaClient apply waits for operator pods Ready, not just for `helm install` to return.
- Belt-and-suspenders: a short poll that the `WekaClient` CRD is present/Established before applying the CR (the operator already has `list_existing_crds()`, `:707`, and `discover_chart_crds()` for exactly this). Since the operator installs CRDs from the chart, confirm those two are wired into the gating.
- Ensure the `weka-client-secret` (`wekaSecretRef: weka-client-cluster-dev`, step 6) and the `quay-pull-secret` (`imagePullSecret: quay-io-secret`) exist **before** the WekaClient CR — a WekaClient referencing a missing secret will sit unready.

**Warning signs:**
- Component `weka-client` `Failed` with `no matches for kind "WekaClient"` / `404` shortly after operator install "succeeded."
- Intermittent failures that pass on retry (classic discovery-cache lag).

**Phase to address:**
**Operator phase** (confirm CRD-Established + Deployment-Ready gating) and **Blueprint phase** (`dependsOn` + `readinessCheck` wiring).

---

### Pitfall 5: Default-StorageClass conflict — two default StorageClasses on the cluster

**What goes wrong:**
The wizard marks `storageclass-wekafs-dir-api` with `storageclass.kubernetes.io/is-default-class: "true"` (`storageclass-wekafs-dir-api.yaml:6`). On a brownfield cluster that already has a default StorageClass (EKS `gp2`/`gp3`, AKS `default`, etc.), this creates **two** defaults. Kubernetes does not reject this — it tolerates multiple defaults but the behavior for a PVC with no explicit `storageClassName` becomes effectively undefined (newer apiservers pick the most-recently-created, older ones may warn/error). The result is non-deterministic PVC binding that surfaces much later, in workloads, not in the install.

**Why it happens:**
The template hard-codes `is-default-class: "true"` and the install just applies it; nobody checks the cluster's current default.

**How to avoid:**
- Before applying, **detect an existing default StorageClass** (the App Store already has `inspect_cluster` / MCP storage-class listing; `apply_gateway` handles `StorageClass` as cluster-scoped, `:24`). If one exists, either: (a) surface a wizard warning and ask the user to confirm switching the default to wekafs, then patch the old default's annotation to `"false"` as part of install; or (b) install wekafs as non-default and let the user opt in.
- Because StorageClass is cluster-scoped and `apply_gateway` does create→409→patch, **re-running the wizard re-patches** the annotation — fine, but make sure a re-run doesn't flip a user's manual choice back (Pitfall 9).
- Also verify the StorageClass `provisioner-secret-name`/`-namespace` (`csi-wekafs-api-secret` / `csi-wekafs`, `:22-31`) match the secret actually created in step 7 — a name/namespace mismatch yields PVCs stuck `Pending` with `secret not found`, again only visible at first PVC, not at install.

**Warning signs:**
- `kubectl get sc` shows two classes annotated `is-default-class: true`.
- Later: PVCs with no `storageClassName` bind to the wrong provisioner, or the apiserver logs "multiple default StorageClasses."
- PVCs `Pending` with `failed to provision volume ... secret "csi-wekafs-api-secret" not found`.

**Phase to address:**
**Backend phase** (existing-default detection + warning) and **Blueprint phase** (secret name/namespace consistency between StorageClasses and the CSI API secret).

---

### Pitfall 6: Secrets leaking into the SSE log stream / CR annotations

**What goes wrong:**
The wizard handles the most sensitive values in the system: the quay robot token and the WEKA cluster password. Several existing code paths can leak them:
1. **CR annotation echo.** `/deploy-stream` stamps the *entire* submitted `variables` dict onto the CR as `warp.io/gui-variables` annotation (`main.py:2935`) so the blueprint page can show them later. If the wizard passes `quay_password`/`weka_password`/`quay_dockerconfigjson` as plain variables, they get written **in clear text** into a cluster annotation readable by anyone with `get wekaappstores`.
2. **SSE component messages.** The stream emits `comp.get("message","")` from operator `componentStatus` verbatim (`:2984`). If the operator logs a failed `helm install`/`kubectl apply` command line that includes a secret (e.g., a `--set` with a password, or an apply error echoing the manifest), it flows straight to the browser log box.
3. **Operator helm errors.** `_install_chart` returns `result.stderr or result.stdout` on failure (`:172`) into the component message — a quay 401 page or helm error can include the registry ref and sometimes credentials.

**Why it happens:**
The annotation-stamping and message-passthrough were built for non-secret blueprint variables in v7.0; the wizard introduces secrets into the same generic pipeline without redaction.

**How to avoid:**
- **Do not stamp secret variables onto the CR annotation.** Maintain an allowlist (or denylist) of variable names excluded from `warp.io/gui-variables` — exclude `*password*`, `*token*`, `*secret*`, `quay_dockerconfigjson`. The blueprint-page "show submitted variables" feature must mask these.
- **Never pass raw creds as `--set` to helm** (they'd appear in process args / logs). They go into `stringData` secrets and the `data` dockerconfigjson only.
- **Redact component messages before emitting SSE.** Scrub the emitted `message` for anything matching the known secret values / their base64 before `yield sse_event`. At minimum, the operator must never log the secret manifests (the existing `NEVER log` discipline, `:490`,`:520`,`:538`, must extend to the new quay/weka creds).

**Warning signs:**
- `kubectl get wekaappstore <name> -o jsonpath='{.metadata.annotations.warp\.io/gui-variables}'` shows a password.
- A password/token visible in the browser log box, operator pod logs, or `kubectl describe`.

**Phase to address:**
**Backend phase** (annotation allowlist + SSE message redaction) and **Frontend phase** (mask in the review step and any replay of variables). PRD Success Criterion 4 ("No secret values appear in logs") makes this a release gate — verify explicitly.

---

### Pitfall 7: Long install exceeds the 15-minute SSE deadline and/or proxy idle timeout

**What goes wrong:**
`/deploy-stream` caps the poll loop at `deadline = time.time() + 900` (15 min, `main.py:2956`) and on timeout emits `{type:"error", message:"Timed out waiting for components to become ready"}`. The v8.0 install is *much* longer than any prior blueprint: quay chart pulls, operator CRD+Deployment rollout, node-label Job across all nodes, **WekaClient pulling the multi-GB `weka-in-container:5.1.0.605` image and joining the backend**, CSI rollout, then StorageClasses. WekaClient readiness alone (image pull + cluster join + driver build) can easily exceed 15 minutes on a fresh node. When the deadline fires, the GUI shows a hard failure even though the install is still progressing successfully in the background — and the operator keeps going, so the user sees "failed" on a stack that then becomes Ready.

Separately, even with the existing `: ping\n\n` keepalive (`:2963`), an ingress/load-balancer **idle or total-request timeout** (e.g., nginx `proxy_read_timeout` default 60s, ALB idle 60s, many ingress `proxy-read-timeout` ~300s) can sever the SSE connection between events. The 2s keepalive defends idle timeouts but **not** a hard max-request-duration cap some proxies impose.

**Why it happens:**
The 900s constant was sized for short app-stack blueprints, not a full storage-stack bring-up including large image pulls and a cluster join. Proxy timeouts are environment-specific and invisible in dev.

**How to avoid:**
- **Raise the deadline** for `app-store-install` specifically (e.g., 45–60 min), or make it per-blueprint configurable rather than a global 900s. The PRD already flags this.
- Keep the 2s keepalive `: ping` (already present) and confirm it flushes (it does — it's `yield`ed). Document required ingress settings: `proxy-read-timeout` / idle timeout >= the deadline, and disable response buffering for `text/event-stream` (nginx `proxy_buffering off` / `X-Accel-Buffering: no` header).
- **Make timeout non-destructive:** on deadline, the GUI should treat it as "still installing — reconnect" rather than "failed." Because the operator drives state on the CR, the stream is just an observer; a disconnect/timeout should reconnect and resume from current `componentStatus`, not abort. The `complete ok:false` path (`:2994`) should remain reserved for actual operator `Failed`.
- Raise `HELM_CMD_TIMEOUT` in the operator too — `_install_chart` has its own subprocess timeout (`:161`, `helm_cmd_timeout`) that, if shorter than the chart's hook/CRD wait, fails the helm step independently of the SSE deadline.

**Warning signs:**
- Browser shows "Timed out waiting for components" while `kubectl get wekaappstore` still shows `appStackPhase: Pending`/progressing.
- "Stream connection error" in the browser mid-install (proxy cut the connection).
- Operator log: "Helm install timed out after Ns" (the operator subprocess timeout, distinct from SSE).

**Phase to address:**
**Backend phase** (raise/per-blueprint SSE deadline; reconnect-on-timeout semantics; document ingress requirements) and **Operator phase** (confirm `HELM_CMD_TIMEOUT` >= worst-case chart install).

---

### Pitfall 8: `discover_chart_crds` LRU cache poisons CRD discovery after a transient/auth failure

**What goes wrong:**
`discover_chart_crds` is `@lru_cache(maxsize=128)` (`operator_module/main.py:673`) and returns `set()` (empty) on **any** `helm show crds` failure (`:685-688`). On a fresh install the very first call hits quay; if it fails for auth (Pitfall 1) or a transient network blip, the **empty set is cached against `(chart_ref, version)`** and every subsequent reconcile reuses the cached empty result — so the operator believes the chart ships no CRDs and never installs them, even after auth is fixed. The WekaClient apply then 404s (Pitfall 4) and the only fix is restarting the operator pod to clear the cache.

**Why it happens:**
`lru_cache` is meant for a stable pure function, but `helm show crds` against a remote registry is neither pure nor reliably successful; caching the failure as a legitimate empty result conflates "no CRDs" with "couldn't fetch CRDs."

**How to avoid:**
- Don't cache failures: only memoize a **non-empty** successful result, or distinguish "fetched, empty" from "fetch failed" (raise/return sentinel on failure so it isn't cached). Since Pitfall 1 must be fixed in the same operator phase, fix the cache semantics alongside the auth fix.
- Verify auth (Pitfall 1) happens *before* the first `discover_chart_crds` call so the first call succeeds.

**Warning signs:**
- Operator installed "successfully" but `kubectl get crd | grep weka.weka.io` shows nothing; WekaClient 404s; problem persists across reconciles until operator pod restart.

**Phase to address:**
**Operator phase** (bundle with the Pitfall 1 auth fix).

---

### Pitfall 9: Non-idempotent re-run after a partial failure (node-label Job, secrets, StorageClasses, chained CRs)

**What goes wrong:**
PRD Success Criterion 5 requires re-running the wizard on a partially- or fully-installed cluster to be safe. Several components are not automatically idempotent:
1. **Node-label Job** (`node-label-job`, `kubectl label nodes --all weka.io/supports-clients=true`). A plain `kubectl label` **fails on re-run** with `label already has value` unless `--overwrite` is passed; a `Job` object with the same name also can't be re-created (immutable `spec.template`) — the second apply gets a 409 and, unlike CRs, `apply_gateway` doesn't patch arbitrary `Job` kinds (or patching an immutable Job spec fails).
2. **Secrets via `stringData`** re-apply fine (create→409→patch handles them), good.
3. **StorageClass** is largely immutable (`parameters`, `provisioner` can't be patched); a create→409→patch on an SC whose parameters changed will fail — re-running with different WEKA endpoints won't update the SC.
4. **Chained CRs** (Decision D): if `app-store-install` partially failed, re-running must re-drive the *same* CR (create→409→patch, supported for WekaAppStore) rather than spawn cluster-init prematurely. The GUI must wait for `app-store-install` Ready before applying `cluster-init` on **every** run, including re-runs.

**Why it happens:**
`kubectl label` and `Job`/`StorageClass` immutability are classic non-idempotent Kubernetes objects; the AppStack create-or-patch path is proven for CRs and Secrets but not for Jobs/immutable kinds.

**How to avoid:**
- Node-label Job: use `kubectl label nodes --all weka.io/supports-clients=true --overwrite` and give the Job a re-run strategy — delete-then-create on reconcile, `generateName`, or treat a completed Job as success and skip. Precedent: the existing `gateway-api-crds-job` (PRD step 4) — confirm *that* job is itself idempotent before copying it.
- StorageClass: on a parameters change, delete-and-recreate (cluster-scoped, no PVCs depend on the SC object's identity) rather than patch — but guard against deleting an in-use default. For v8.0's "fresh install" target this is low risk; day-2 SC re-config is out of scope (PRD Non-Goals).
- Make the wizard re-entrant: detect already-installed components via `/cluster-info` (operator CRDs + CSI deployment, `main.py:2361`) and offer to skip/resume rather than blindly re-apply.

**Warning signs:**
- Re-run fails at `node-label-job` with `Job already exists` / `field is immutable`, or `kubectl label` non-zero exit `already has value`.
- Re-run with changed endpoints leaves stale StorageClass parameters (PVCs still point at old WEKA endpoints).

**Phase to address:**
**Blueprint phase** (`--overwrite` on label job; Job re-run strategy) and **Backend/Frontend phase** (resume/skip on detected install). **E2E phase** must explicitly test re-run.

---

### Pitfall 10: Auto-restarting kubelet to apply CPU-manager/hugepage config (why Decision A1 keeps it manual)

**What goes wrong:**
Applying the WEKA node prerequisites — CPU Manager `static` policy, `strict-cpu-reservation`, hugepages `25000` — **cannot take effect without restarting kubelet**, and changing the CPU manager policy from `none`→`static` additionally requires *deleting the kubelet CPU manager state file* (`/var/lib/kubelet/cpu_manager_state`) or kubelet refuses to start with a "policy changed" error (verified — Kubernetes docs: drain node → stop kubelet → remove state file → edit config → start kubelet). If the App Store tried to automate this (privileged DaemonSet writing kubelet config + restarting kubelet), a mistake bricks every node's kubelet simultaneously — an outage of the entire cluster, including the App Store itself and tenant workloads. There is no safe rollback once kubelet won't start.

**Why it happens:**
The temptation to "fully automate the install" extends to node config, but node-level kubelet mutation is a fundamentally different blast radius than applying namespaced manifests.

**How to avoid:**
- **Keep Decision A1: documented manual prerequisite + confirm checkbox.** The App Store displays the required `KubeletConfiguration` snippet and gates install behind "I have applied node prerequisites." It does **not** write node config or restart kubelet.
- Optionally *verify* (read-only) that hugepages are present before declaring success — e.g., the operator/`inspect_cluster` can check `node.status.capacity["hugepages-2Mi"]` and warn if zero — verification only, never mutation.
- If a customer skips the prereq, WekaClient pods fail to get hugepages and sit unready; surface that as a clear "node prerequisites not applied" message rather than a generic pod-pending.

**Warning signs:**
- WekaClient pods `Pending`/`CrashLoopBackOff` with `insufficient hugepages` or CPU-pinning errors.
- Any proposal to add a privileged DaemonSet that restarts kubelet — reject for v8.0.

**Phase to address:**
**Frontend phase** (Step 1 snippet + confirm checkbox, already in PRD). Optionally **Operator/Backend** for read-only hugepage verification. Decision A1 already prevents the dangerous path — the pitfall is *re-introducing* automation later.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Rely on quay pull secrets for chart pull (Decision C as written) | No operator change | Install fails 401 on every fresh cluster; feature is dead-on-arrival | **Never** — must do helm registry auth in the operator |
| Hand-encode any secret with `echo\|base64` | Quick | Trailing-newline auth failures, silent until WEKA login (already bit twice) | Never — use `stringData` / `_b64()` |
| Global 900s SSE deadline reused for the long install | No new config | False "timeout" failures mid-install; users abort a working install | Only as a stopgap with a clear "still installing, reconnect" UX |
| Stamp full `variables` dict (incl. secrets) onto CR annotation | Reuse v7.0 replay feature | Cleartext creds in cluster annotations; fails Success Criterion 4 | Never for secret fields — allowlist required |
| Plain `kubectl label` (no `--overwrite`) in node-label Job | Matches manual step | Re-run fails; non-idempotent | Never — always `--overwrite` |
| `lru_cache` over `helm show crds` including failures | Avoids repeat remote calls | Caches an auth/network failure as "no CRDs"; needs pod restart | Only if failures are excluded from the cache |
| StorageClass default hard-coded `true`, no existing-default check | Simpler blueprint | Dual-default ambiguity on brownfield clusters | Only if wizard warns + lets user opt out |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| quay.io OCI **chart** pull (operator helm) | Assuming dockerconfigjson pull secret authenticates `helm pull`/`helm show crds` | `helm registry login quay.io` or `--registry-config` in the operator before any `oci://quay.io/...` op (Pitfall 1) |
| quay.io **image** pull (kubelet) | Wrong/missing `auth` field in dockerconfigjson; newline in `auth` | `auth = base64("user:pass")` no newline; include username/password/auth (Pitfall 3) |
| WEKA API (CSI + WekaClient) | Trailing `\n` in creds/scheme/endpoints from `echo\|base64` | `stringData`; round-trip test decode == input (Pitfall 2) |
| WEKA operator CRD registration | Applying `WekaClient` right after `helm install` returns | Gate on operator Deployment Ready + CRD Established, not just install order (Pitfall 4) |
| CSI public chart (`csi-wekafs/csi-wekafsplugin`) | Treating it like the quay chart (auth) | Public repo, no auth — but it *is* a non-OCI `helm repo add` path; confirm `_add_repo` runs for it |
| StorageClass ↔ CSI API secret | SC references a `secretName`/`secretNamespace` not yet created or mismatched | Apply `csi-wekafs-api-secret` in `csi-wekafs` ns before SCs; verify names match (`:22-31`) |
| Ingress/LB in front of SSE | Default proxy idle/total timeout < install duration; response buffering on | `proxy-read-timeout` >= deadline; `proxy_buffering off` / `X-Accel-Buffering: no` (Pitfall 7) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| 15-min SSE cap vs multi-GB WekaClient image pull + cluster join | "Timed out" while install still progressing | Per-blueprint deadline 45–60 min; reconnect-on-timeout | First real install on a node with cold image cache / slow registry |
| Operator `HELM_CMD_TIMEOUT` shorter than chart hook/CRD wait | "Helm install timed out after Ns" in operator log | Raise `HELM_CMD_TIMEOUT` for operator chart | Operator chart with pre-install hooks / many CRDs |
| Node-label Job over `--all` nodes on large clusters | Job slow / partial on big node counts | Readiness-gate the Job; `--overwrite`; idempotent re-run | Clusters with many/eventually-joining nodes |
| `discover_chart_crds` cached empty after transient failure | CRDs never install; persists across reconciles | Exclude failures from cache | Any flaky/auth-failing first call (Pitfall 8) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Quay token / WEKA password stamped into `warp.io/gui-variables` CR annotation | Cleartext creds readable by anyone with `get wekaappstores` | Allowlist excludes secret vars from the annotation (Pitfall 6) |
| Secret values flowing through SSE `message` / operator stderr to browser | Creds shown in browser log box, operator logs | Redact emitted messages; honor "NEVER log" contract for new creds |
| Passing creds as helm `--set` | Creds in process args / `ps` / logs | Creds only via `stringData` secrets + dockerconfigjson `data` |
| Storing WEKA creds long-term without scoping | Broad standing access | Consider `WarpCredential` pattern (existing) with least-privilege; PRD notes reuse |
| Quay robot token with more than pull scope | Token leak → registry write access | Use a pull-only robot account for `QUAY_PASSWORD` |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Hard "failed" on SSE timeout of a still-running install | User re-runs / aborts a healthy install | "Still installing — reconnecting" with resume from `componentStatus` |
| Generic pod-pending error when node prereqs were skipped | User can't tell it's a node-config issue | Detect missing hugepages; message "apply node prerequisites (Step 1)" |
| Silent dual-default StorageClass | Later workloads bind to wrong storage, no install-time signal | Warn at review step if a default SC already exists; let user choose |
| One endpoints field feeding both YAML-list and comma-string forms, mis-parsed | WekaClient `joinIpPorts` or CSI `endpoints` malformed | Single helper produces both forms; validate `host:port` per entry before submit |
| Re-run blindly re-applies everything | Confusing failures on immutable Job/SC | Detect installed components via `/cluster-info`, offer skip/resume |

## "Looks Done But Isn't" Checklist

- [ ] **Operator chart install:** Often missing — `helm registry login`/`--registry-config` for quay; verify on a cluster that has *only* the pull secret (no host helm login) (Pitfall 1)
- [ ] **dockerconfigjson:** Often missing the inner `auth` field or has a trailing newline — verify `auths["quay.io"]["auth"]` decodes to exactly `user:pass` (Pitfall 3)
- [ ] **Generated secrets:** Often decode with a trailing `\n` — verify byte-for-byte round trip from `stringData` (Pitfall 2)
- [ ] **WekaClient gating:** Often only `dependsOn`, not Deployment-Ready + CRD-Established — verify the CR apply waits, no 404 on a fresh cluster (Pitfall 4)
- [ ] **Default StorageClass:** Often ignores a pre-existing default — verify `kubectl get sc` shows exactly one default after install (Pitfall 5)
- [ ] **Secret redaction:** Often missing from CR annotation + SSE — verify `kubectl get wekaappstore -o yaml` and the browser log contain no creds (Pitfall 6, Success Criterion 4)
- [ ] **SSE deadline:** Often left at 900s — verify a real cold-cache install completes without a false timeout (Pitfall 7)
- [ ] **Idempotent re-run:** Often untested — verify a second wizard run on an installed cluster is a no-op / clean resume (Pitfall 9, Success Criterion 5)
- [ ] **CRD cache:** Often caches failures — verify a fixed-auth retry actually installs CRDs without operator restart (Pitfall 8)

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Operator chart 401 (no helm login) | LOW once fixed | Add operator helm auth; restart operator pod to clear `discover_chart_crds` cache; re-reconcile CR |
| Trailing-newline secret | LOW | Re-apply secret via `stringData`; restart/refresh the consuming pod (CSI/WekaClient) |
| WekaClient 404 (CRD not yet ready) | LOW | Operator auto-retries (`TemporaryError`); add Deployment-Ready gate so it stops happening |
| Dual-default StorageClass | LOW | `kubectl annotate sc <name> storageclass.kubernetes.io/is-default-class=false --overwrite` on the unwanted SC |
| Cleartext creds in CR annotation | MEDIUM | Rotate the exposed quay token / WEKA password; strip the annotation; ship the allowlist fix |
| Cached empty CRD set | LOW | Restart operator pod (clears `lru_cache`); ship the no-cache-on-failure fix |
| Non-idempotent Job on re-run | LOW | Delete the failed Job; re-apply with `--overwrite`; ship the idempotent-Job fix |
| Bricked kubelet from auto node-config | HIGH | Per-node manual recovery (delete CPU state file, fix config, restart kubelet) — avoided entirely by Decision A1 |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Helm chart pull needs registry login (Decision C wrong) | Operator phase (re-scope) | Fresh cluster with only pull secrets: operator chart installs (no 401) |
| 2. `echo\|base64` trailing newline | Blueprint phase (`stringData`) | Round-trip decode == input, no `\n` |
| 3. dockerconfigjson `auth` assembly | Backend phase (GUI builder) | `auths["quay.io"]["auth"]` decodes to `user:pass` exactly |
| 4. WekaClient before CRD → 404 | Operator + Blueprint phase | No `no matches for kind WekaClient` on fresh install |
| 5. Default-StorageClass conflict | Backend (detect) + Blueprint (secret match) | Exactly one default SC; PVC binds to wekafs as intended |
| 6. Secret leakage (annotation + SSE) | Backend (redact) + Frontend (mask) | No creds in CR YAML or browser log (Success Criterion 4) |
| 7. SSE 15-min deadline + proxy timeout | Backend phase | Cold-cache install completes without false timeout |
| 8. CRD-discovery cache poisoning | Operator phase (with #1) | Fixed-auth retry installs CRDs without pod restart |
| 9. Non-idempotent re-run | Blueprint + Backend/Frontend; E2E test | Second run on installed cluster is safe (Success Criterion 5) |
| 10. Auto kubelet restart danger | Frontend phase (A1 confirm) | App Store never mutates node/kubelet; manual prereq gate present |

## Sources

- Repo code (HIGH): `operator_module/main.py` — `_install_chart`/`_upgrade_chart` pass no helm auth (`:136-178`), `discover_chart_crds` no-auth + `lru_cache` + swallow-failure (`:673-703`), `_b64`/`_derive_ngc_payloads` dockerconfigjson reference (`:441`,`:499-509`), "NEVER log" credential contract (`:490`,`:520`,`:538`); `_add_repo` skips repo add for OCI (`:115-134`)
- Repo code (HIGH): `app-store-gui/webapp/main.py` — `/deploy-stream` 900s deadline (`:2956`), keepalive ping (`:2963`), CR annotation variable stamping (`:2935`), SSE message passthrough (`:2984`), helm-error passthrough into message; `/cluster-info` prereq detection (`:2361`)
- Repo code (HIGH): `app-store-gui/webapp/planning/apply_gateway.py` — create→409→patch only handles 409 not 404 (`:301`,`:323`); StorageClass cluster-scoped (`:24`)
- Repo templates (HIGH): `storageclass-wekafs-dir-api.yaml` hard-coded `is-default-class:"true"` + secret refs (`:6`,`:22-31`); `csi-wekafs-api-secret.yaml` `data:` fields; `wekaClientCR-online.yaml` `wekaSecretRef`/`imagePullSecret`/CRD `weka.weka.io/v1alpha1`
- PRD (HIGH): `.planning/PRD-install-wizard-weka-storage-stack.md` — Decisions A1/B/C/D/E, trailing-newline fix note, risks section
- [Use OCI-based registries — Helm](https://helm.sh/docs/topics/registries/) (HIGH) — `helm registry login` authenticates chart pulls; distinct from kubelet imagePullSecrets
- [Pulling Helm charts from private OCI registry — argo-cd #21060](https://github.com/argoproj/argo-cd/discussions/21060) (MEDIUM) — imagePullSecrets do not authenticate helm chart pulls
- [WEKA Operator deployments — WEKA docs](https://docs.weka.io/kubernetes/weka-operator-deployments) (HIGH) — operator chart at `oci://quay.io/weka.io/helm/weka-operator` requires QUAY creds + quay secret in `weka-operator-system` and `default`; not anonymous-pullable
- [Best practices for WEKA stateless client and Kubernetes — WEKA docs](https://docs.weka.io/best-practice-guides/best-practices-for-weka-stateless-client-and-kubernetes) (HIGH) — hugepages + CPU manager static + strict-cpu-reservation prerequisites
- [Control CPU Management Policies on the Node — Kubernetes](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) (HIGH) — changing CPU manager policy requires drain → stop kubelet → delete `/var/lib/kubelet/cpu_manager_state` → restart kubelet (why auto-restart is dangerous)

---
*Pitfalls research for: WEKA storage-stack guided install wizard (v8.0)*
*Researched: 2026-06-24*
