# Architecture

**Analysis Date:** 2026-03-17

## Pattern Overview

**Overall:** Kubernetes application bundle composed of a Kopf-based operator, a FastAPI/Jinja web UI, and a Helm chart that deploys both.

**Key Characteristics:**
- Control-plane heavy design centered on a namespaced custom resource, `WekaAppStore`, defined in `weka-app-store-operator-chart/templates/crd.yaml`.
- Two Python runtimes with distinct roles: reconciliation logic in `operator_module/main.py` and user-facing cluster orchestration in `app-store-gui/webapp/main.py`.
- Packaging and deployment are first-class concerns: the repository includes Helm source in `weka-app-store-operator-chart/`, publishable chart artifacts in `docs/`, and container build definitions in `docker/`.
- Blueprint content is treated as external runtime data. The GUI expects manifests from `warp-blueprints` through `git-sync` and `BLUEPRINTS_DIR` rather than storing those application blueprints in this repository (`app-store-gui/webapp/main.py`, `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`).

## Layers

**Packaging and Deployment Layer:**
- Purpose: Define how the operator and GUI are installed into a cluster.
- Contains: Helm chart metadata and templates, chart defaults, Dockerfiles, published chart index and archives.
- Location: `weka-app-store-operator-chart/`, `docker/`, `docs/`.
- Depends on: Kubernetes, Helm, container registries.
- Used by: Cluster administrators installing the app store.

**Custom Resource Contract Layer:**
- Purpose: Define the declarative API the system reconciles.
- Contains: The `WekaAppStore` CRD schema, including single-chart installs, legacy pod mode, and multi-component `appStack` deployments.
- Location: `weka-app-store-operator-chart/templates/crd.yaml`.
- Depends on: Kubernetes CRD machinery.
- Used by: The operator and the GUI when creating or reading `warp.io/v1alpha1` resources.

**Operator Reconciliation Layer:**
- Purpose: Observe `WekaAppStore` objects and converge cluster state.
- Contains: Kopf handlers, Helm command wrapper, CRD strategy helpers, dependency ordering, manifest apply/delete helpers, status patching.
- Location: `operator_module/main.py`.
- Depends on: `kopf`, `kr8s`, Kubernetes Python client, local `helm` and `kubectl` binaries.
- Used by: Kubernetes events triggered by CR create, update, and delete operations.

**GUI/API Layer:**
- Purpose: Provide the web UI and HTTP endpoints for cluster status, secrets, blueprint listing, deployment, initialization, and sync.
- Contains: FastAPI routes, middleware, Jinja template rendering, SSE progress streams, Kubernetes API helpers, manifest application helpers.
- Location: `app-store-gui/webapp/main.py`, `app-store-gui/webapp/templates/`.
- Depends on: Kubernetes Python client, template files, runtime blueprint manifests.
- Used by: Browser clients and cluster operators.

**Operational Asset Layer:**
- Purpose: Store cluster bootstrap examples, WEKA CSI examples, and other YAML inputs that support deployment or local testing.
- Contains: Cluster-init manifests, Gateway route examples, WEKA CSI config YAMLs, ad hoc PVC samples.
- Location: `cluster_init/`, `weka-csi-config/`, `test-pvc.yaml`, `test-pvc-pod.yaml`.
- Depends on: Cluster-specific Kubernetes APIs and installed operators.
- Used by: Human operators and, in one case, the GUI bootstrap flow references `cluster_init/app-store-cluster-init.yaml`.

## Data Flow

**Helm Installation Flow:**

1. A cluster admin installs `weka-app-store-operator-chart` from `weka-app-store-operator-chart/` or the packaged repo in `docs/index.yaml`.
2. The chart renders the CRD from `weka-app-store-operator-chart/templates/crd.yaml`, the operator deployment from `weka-app-store-operator-chart/templates/deployment.yaml`, and the GUI deployment from `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
3. The operator container starts `kopf run ... /app/operator.py` and watches `WekaAppStore` resources according to `watch.*` values in `weka-app-store-operator-chart/values.yaml`.
4. The GUI container starts Uvicorn against `app-store-gui/webapp/main.py` and mounts blueprint manifests via `git-sync` and `BLUEPRINTS_DIR`.

**Blueprint Submission Flow:**

1. A user loads the GUI root route or a blueprint page served by `app-store-gui/webapp/main.py`.
2. The UI checks initialization and cluster capability through helpers such as `is_cluster_initialized_anywhere()` and `get_cluster_status()`.
3. A deployment request hits `/deploy` or `/deploy-stream`, which maps a blueprint name to a manifest path and applies YAML via `apply_blueprint_with_namespace()` in `app-store-gui/webapp/main.py`.
4. The applied manifest creates or updates a `warp.io/v1alpha1` `WekaAppStore` resource.
5. The operator receives the event in `create_warrpappstore_function()` or `update_warrpappstore_function()` in `operator_module/main.py`.
6. The operator chooses `handle_appstack_deployment()`, `handle_helm_deployment()`, or `handle_pod_deployment()` based on the CR spec and patches status back onto the resource.
7. The GUI polls `/cluster-status`, `/api/blueprints`, or streams `/init-logs` and `/deploy-stream` to expose progress to the user.

**AppStack Reconciliation Flow:**

1. A `WekaAppStore` spec with `appStack.components` is validated against the CRD schema in `weka-app-store-operator-chart/templates/crd.yaml`.
2. `handle_appstack_deployment()` in `operator_module/main.py` resolves dependencies and iterates enabled components in order.
3. Each component is installed either through Helm using `HelmOperator.install_or_upgrade()` or through raw YAML application.
4. Optional readiness checks wait on pods, deployments, statefulsets, or jobs before advancing.
5. Component-level status is aggregated into `status.componentStatus` and `status.appStackPhase`.

**State Management:**
- Persistent state lives in Kubernetes resources: `WekaAppStore` objects, Helm releases, namespaces, ConfigMaps, Secrets, and related workloads.
- Both Python processes are mostly stateless between requests or events, aside from small in-process caches such as `_config_loaded` and `_last_ready_cache` in `app-store-gui/webapp/main.py`.
- Temporary execution state is written to temp files for Helm values and manifest application in `operator_module/main.py`.

## Key Abstractions

**WekaAppStore Resource:**
- Purpose: The system’s declarative contract for installing either a single chart, a legacy pod, or a multi-component app stack.
- Examples: `cluster_init/app-store-cluster-init.yaml`, the schema in `weka-app-store-operator-chart/templates/crd.yaml`.
- Pattern: Kubernetes custom resource.

**HelmOperator:**
- Purpose: Encapsulate repo setup, install, upgrade, uninstall, and status lookup around the Helm CLI.
- Examples: `install_or_upgrade()`, `_install_chart()`, `_upgrade_chart()`, `uninstall()` in `operator_module/main.py`.
- Pattern: Thin command-wrapper service around subprocess execution.

**Blueprint Manifests:**
- Purpose: External YAML definitions that the GUI applies to create `WekaAppStore` resources or supporting objects.
- Examples: `cluster_init/app-store-cluster-init.yaml`; runtime blueprints loaded from the external repo referenced in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
- Pattern: File-backed declarative input.

**FastAPI Route Handlers:**
- Purpose: Bridge browser actions to Kubernetes API calls and template rendering.
- Examples: `/deploy`, `/deploy-stream`, `/cluster-info`, `/sync`, `/init-cluster` in `app-store-gui/webapp/main.py`.
- Pattern: Boundary/controller layer with embedded orchestration logic.

## Entry Points

**Operator Runtime:**
- Location: `operator_module/main.py`
- Triggers: `kopf` process started by `weka-app-store-operator-chart/templates/deployment.yaml` and `docker/operator.Dockerfile`.
- Responsibilities: Watch `wekaappstores`, reconcile Helm releases or manifests, patch CR status, uninstall on delete.

**GUI Runtime:**
- Location: `app-store-gui/webapp/main.py`
- Triggers: Uvicorn process started by `docker/webapp.Dockerfile`.
- Responsibilities: Serve HTML, expose JSON and SSE endpoints, inspect cluster state, apply manifests, and manage runtime blueprint sync.

**Cluster Bootstrap Manifest:**
- Location: `cluster_init/app-store-cluster-init.yaml`
- Triggers: Applied through the GUI `/init-cluster` route or manually with `kubectl`.
- Responsibilities: Seed a `WekaAppStore` app stack for monitoring, Gateway API setup, Envoy routing, and related bootstrap components.

## Error Handling

**Strategy:** Boundary handlers catch broad exceptions, translate them into JSON error responses or retryable Kopf errors, and use logging plus CR status fields for operator observability.

**Patterns:**
- The operator raises `kopf.PermanentError` for invalid specs and `kopf.TemporaryError` for failed Helm operations that should be retried (`operator_module/main.py`).
- The GUI wraps most Kubernetes and file operations in `try/except` blocks and returns structured `JSONResponse` payloads with `ok`, `error`, and optional HTTP status details (`app-store-gui/webapp/main.py`).
- Readiness and initialization state are exposed explicitly via `/readyz`, `/healthz`, `/cluster-status`, and CR status conditions rather than only logs.

## Cross-Cutting Concerns

**Logging:**
- Both runtimes use Python logging from their main modules (`operator_module/main.py`, `app-store-gui/webapp/main.py`).
- The GUI also exposes operator progress indirectly by streaming pod logs in `/init-logs`.

**Validation:**
- Structural validation begins in the CRD OpenAPI schema in `weka-app-store-operator-chart/templates/crd.yaml`.
- Runtime validation is mostly imperative: route handlers and reconciliation functions inspect required fields and environment-dependent conditions in Python.

**Authentication and Authorization:**
- There is no end-user auth layer in the web UI code.
- Access control is delegated to Kubernetes RBAC defined in `weka-app-store-operator-chart/templates/clusterrole.yaml` and the RBAC resources embedded in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.

**External Tooling Dependencies:**
- The operator relies on local `helm` and `kubectl` binaries and shells out through `subprocess` (`operator_module/main.py`, `docker/operator.Dockerfile`).
- The GUI can rely on a bundled or downloaded `git-sync` binary for blueprint refreshes (`app-store-gui/webapp/main.py`, `docker/webapp.Dockerfile`).

*Architecture analysis: 2026-03-17*
*Update when major patterns change*
