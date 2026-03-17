# Codebase Concerns

**Analysis Date:** 2026-03-17

## Tech Debt

**Blueprint apply logic duplicated across file and string paths:**
- Issue: `apply_blueprint_with_namespace()` and `apply_blueprint_content_with_namespace()` in `app-store-gui/webapp/main.py` implement nearly the same multi-document YAML mutation and apply flow.
- Why: The GUI supports both static blueprint files and rendered Jinja2 blueprint content, and the second path was added by copying the first.
- Impact: Fixes to namespace handling, CR detection, annotation behavior, or error handling must be made twice and can drift.
- Fix approach: Extract the shared document iteration and apply logic into one internal helper that accepts parsed docs or a loader callback.

**Operator deploy flow mixes client libraries and shell tools:**
- Issue: `operator_module/main.py` uses Kubernetes Python clients, `kr8s`, `helm`, and `kubectl` subprocesses in the same reconciliation path.
- Why: Different features were added pragmatically as requirements expanded from pod launches to multi-component AppStack installs.
- Impact: Error handling, retries, auth behavior, and observability differ by code path, making reconciliation harder to reason about and test.
- Fix approach: Consolidate around a narrower execution model, ideally wrapping Helm/kubectl interactions behind one adapter with structured logging and typed errors.

**Repository includes generated release artifacts alongside source:**
- Issue: `docs/` contains many packaged chart archives and the GitHub Pages index, while `.gitignore` only excludes two local directories in `.gitignore`.
- Why: The repo doubles as the source tree and the published Helm repository.
- Impact: Routine source changes are mixed with release artifacts, increasing review noise and the chance of stale packages or accidental binary churn.
- Fix approach: Separate packaging/publishing from source development, or at minimum add automation that rebuilds `docs/index.yaml` deterministically and validates it in CI.

## Known Bugs

**Operator chart points Kopf at a file path that the Dockerfile does not create:**
- Symptoms: The operator container is likely to fail at startup when rendered from the Helm chart because the chart runs `kopf ... /app/operator.py`, but the Docker build copies `main.py` to `/app` and sets a different default command.
- Trigger: Install the chart from `weka-app-store-operator-chart/templates/deployment.yaml` using an image built from `docker/operator.Dockerfile`.
- Files: `weka-app-store-operator-chart/templates/deployment.yaml`, `docker/operator.Dockerfile`
- Workaround: Override the image/entrypoint externally or align the image contents with the chart before deploying.
- Root cause: The chart, Dockerfile, and runtime path conventions drifted apart.

**GUI RBAC resources are bound to a service account the pod does not use:**
- Symptoms: The GUI deployment requests `serviceAccountName: {{ include "weka-app-store-operator-chart.serviceAccountName" . }}`, but the GUI-specific RBAC objects are bound to `wekaappstoregui-sa`, so the GUI pod may not receive the permissions documented later in the same template.
- Trigger: Deploy the GUI via `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
- Files: `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`
- Workaround: Manually align the deployment service account and the RoleBinding/ClusterRoleBinding subjects after install.
- Root cause: The template mixes the chart-wide operator service account helper with a separate hard-coded GUI service account.

**GUI RBAC assumes the `default` namespace even though the chart install docs recommend a dedicated namespace:**
- Symptoms: Several Role, RoleBinding, and ClusterRoleBinding subjects are hard-coded to `namespace: default`, which will break or mis-scope permissions when the chart is installed into `weka-app-store` as described in `README.md`.
- Trigger: Follow the install instructions in `README.md` and deploy to any namespace other than `default`.
- Files: `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`, `README.md`
- Workaround: Patch the rendered manifests or install only into `default`.
- Root cause: Namespace values in the GUI RBAC template are not parameterized.

## Security Considerations

**Operator RBAC is intentionally broad and close to cluster-admin for some resource groups:**
- Risk: `weka-app-store-operator-chart/values.yaml` grants cluster-wide create/update/delete on many core and extension resources, plus `bind` and `escalate` on RBAC resources in `extraRules`.
- Current mitigation: `rbac.permissive` defaults to `false`, so the full wildcard `*/*/*` rule is not enabled by default.
- Recommendations: Replace broad default permissions with the minimum verified rule set per supported blueprint family, and gate optional high-privilege rules behind explicit feature flags.

**GUI has a write-capable cluster role and an unauthenticated sync path by default:**
- Risk: `/sync` in `app-store-gui/webapp/main.py` only enforces bearer auth when `SYNC_TOKEN` is set, while `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml` sets `SYNC_TOKEN` to an empty string and grants cluster-wide write permissions for namespaces, ConfigMaps, and `warp.io` CRs.
- Current mitigation: The sync repo is fixed to `https://github.com/weka/warp-blueprints.git` in the chart template, limiting where sync pulls from by default.
- Recommendations: Require auth for `/sync`, default `SYNC_TOKEN` to a generated secret or disable the endpoint entirely, and scope GUI write privileges to the smallest practical namespace set.

**Runtime binary acquisition is not pinned or verified strongly enough:**
- Risk: `_ensure_git_sync_binary()` in `app-store-gui/webapp/main.py` downloads `git-sync` binaries from GitHub at runtime if the binary is not found, but it only validates executability and does not verify checksums or signatures.
- Current mitigation: `docker/webapp.Dockerfile` already copies `git-sync` from the official `registry.k8s.io/git-sync/git-sync` image, so the fallback is not expected on the happy path.
- Recommendations: Remove runtime downloads in production images, or verify checksums/signatures for every fetched asset before execution.

## Performance Bottlenecks

**Cluster status and readiness endpoints do live cluster-wide API scans:**
- Problem: `get_cluster_status()` and related endpoints list nodes, pods across all namespaces, CRDs, storage classes, and custom resources on demand.
- Measurement: No latency or scale measurements are checked into the repo.
- Cause: The GUI computes cluster state directly from live Kubernetes API calls rather than caching or pre-aggregating status.
- Improvement path: Cache expensive reads per request class, page or narrow list calls where possible, and move repeated cluster summaries behind a background refresher.

**Blueprint sync and deployment paths block on subprocesses and file IO:**
- Problem: `/sync` and the operator reconcile flow call `subprocess.run()` for `git-sync`, `helm`, and `kubectl`, while the GUI deploy stream reads and renders entire manifests before apply.
- Measurement: No runtime metrics or timeout histograms are present in the repo.
- Cause: Long-running cluster operations are performed synchronously in request/reconcile paths.
- Improvement path: Emit structured timing metrics, move expensive operations behind jobs/tasks where possible, and introduce bounded concurrency plus retries with backoff.

## Fragile Areas

**Namespace override behavior in blueprint application:**
- Why fragile: `app-store-gui/webapp/main.py` mutates `metadata.namespace` and selectively rewrites `spec.appStack.components[].targetNamespace`, but only if the original field already exists.
- Common failures: Components without `targetNamespace` keep their original target; cluster-init is treated specially in `/deploy` and `/deploy-stream`; behavior differs depending on whether the blueprint came from a file or rendered content.
- Safe modification: Change namespace logic only with representative multi-doc blueprints and cluster-scoped resources in hand.
- Test coverage: No automated tests were found for these flows in the repository.

**AppStack reconciliation stops on first component failure:**
- Why fragile: `handle_appstack_deployment()` in `operator_module/main.py` executes components serially, mutates status incrementally, and aborts the rest of the stack after the first failed component.
- Common failures: Partial installs leave previously created Helm releases or manifests behind, and troubleshooting depends on reading operator logs rather than durable per-step records.
- Safe modification: Preserve status schema, add rollback/cleanup rules deliberately, and validate against both create and update handlers because both reuse the same deployment logic.
- Test coverage: No unit or integration tests were found under the repository for create/update/delete reconciliation behavior.

**GUI deployment template contains several hard-coded infrastructure assumptions:**
- Why fragile: `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml` hard-codes the GUI image tag, AWS load balancer annotations including subnet IDs, a one-shot git-sync init container, and fixed service account names.
- Common failures: Non-AWS installs inherit irrelevant annotations, namespace changes break RBAC, and image upgrades require direct template edits instead of values overrides.
- Safe modification: Parameterize provider-specific and image settings in `values.yaml` before further template expansion.
- Test coverage: Only Helm's default `templates/tests/test-connection.yaml` exists; no chart rendering or install validation is checked in.

## Scaling Limits

**GUI runtime capacity is single-instance by configuration:**
- Current capacity: `deploy-app-store-gui.yaml` sets `replicas: 1` and `UVICORN_WORKERS=1`.
- Limit: The web UI has no horizontal scaling or shared session/state coordination defined in the chart.
- Symptoms at limit: Slow responses, dropped SSE streams, or stalled sync/deploy requests under concurrent use.
- Scaling path: Expose replica/worker counts through `values.yaml`, add readiness-safe multi-replica behavior, and instrument the HTTP endpoints before scaling.

**Operator reconciliation is effectively single-controller and sequential per stack:**
- Current capacity: `weka-app-store-operator-chart/values.yaml` defaults `replicaCount: 1`, and `handle_appstack_deployment()` processes enabled components in order.
- Limit: Large AppStacks or many simultaneous `WekaAppStore` resources will serialize Helm/kubectl work through one operator pod.
- Symptoms at limit: Slow reconcile turnaround, longer failure recovery, and more chance of timeout in `helm` or `kubectl wait`.
- Scaling path: Add leader-election-safe multi-replica support, bound per-resource concurrency, and measure reconcile durations before increasing throughput expectations.

## Dependencies at Risk

**Build-time CLI downloads are architecture-specific and drift-prone:**
- Risk: `docker/operator.Dockerfile` downloads Linux `amd64` builds of both `kubectl` and Helm regardless of target architecture, while the chart defaults reference a `multi-arch` image in `values.yaml`.
- Impact: ARM builds or runtimes can fail unexpectedly even if the Python code itself is portable.
- Migration plan: Use architecture-aware build args or vendor CLIs from multi-arch base images rather than hard-coded `amd64` URLs.

**Python dependencies are minimum-version only:**
- Risk: `app-store-gui/requirements.txt` and `operator_module/requirements.txt` specify lower bounds but no upper pins or lockfiles.
- Impact: Rebuilds can pull newer FastAPI, Kubernetes client, Kopf, or kr8s versions that change behavior without any repository change.
- Migration plan: Introduce locked dependency sets per image and validate them with automated smoke tests before releases.

## Missing Critical Features

**Automated verification for chart, GUI, and operator behavior:**
- Problem: The repository contains application code, operator logic, and Helm templates, but no unit tests, integration tests, or CI configuration beyond Helm's sample `test-connection.yaml`.
- Current workaround: Manual cluster deployments and ad hoc troubleshooting through logs.
- Blocks: Safe refactoring of reconcile logic, namespace override behavior, and chart/RBAC changes.
- Implementation complexity: Medium; the code is modular enough for targeted tests, but subprocess-heavy paths will need fakes or test harnesses.

**Configuration parameterization for environment-specific deployment:**
- Problem: Several operational choices are hard-coded directly in templates and code, including GUI image tag, AWS load balancer annotations/subnets, blueprint repo URL, and service account namespace assumptions.
- Current workaround: Patch manifests or maintain local forks for each environment.
- Blocks: Repeatable installs across clouds, namespaces, and release channels.
- Implementation complexity: Medium; most fixes are chart-value plumbing rather than deep runtime rewrites.

## Test Coverage Gaps

**GUI request handlers and cluster mutation paths:**
- What's not tested: Endpoints in `app-store-gui/webapp/main.py` that create secrets, submit CRs, initialize/uninitialize the cluster, stream logs, sync blueprints, and render/apply templated manifests.
- Risk: Permission mismatches, namespace bugs, and cluster-side regressions can ship unnoticed.
- Priority: High
- Difficulty to test: Moderate; Kubernetes client calls and subprocesses need to be mocked or exercised in a disposable cluster.

**Operator reconcile, readiness, and deletion logic:**
- What's not tested: `handle_appstack_deployment()`, `handle_helm_deployment()`, `wait_for_component_ready()`, and delete handlers in `operator_module/main.py`.
- Risk: Partial installs, broken upgrades, and namespace-specific delete failures will surface only in live clusters.
- Priority: High
- Difficulty to test: Moderate to High; Helm and kubectl subprocesses need integration-style fakes or envtest-style harnesses.

**Helm chart rendering assumptions:**
- What's not tested: Cross-template consistency between `values.yaml`, operator deployment, GUI deployment, RBAC bindings, and README install instructions.
- Risk: Template drift can leave the published chart nonfunctional even if individual source files look valid in isolation.
- Priority: High
- Difficulty to test: Low to Moderate; `helm template` plus policy assertions would catch several issues already visible in static inspection.

---

*Concerns audit: 2026-03-17*
*Update as issues are fixed or new ones discovered*
