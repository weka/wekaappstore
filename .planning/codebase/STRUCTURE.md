# Codebase Structure

**Analysis Date:** 2026-03-17

## Directory Layout

```text
wekaappstore/
├── .planning/codebase/              # Generated repository mapping documents
├── app-store-gui/                   # FastAPI web application source and Python deps
│   └── webapp/                      # Runtime module, templates, and assets
├── cluster_init/                    # Bootstrap manifests referenced by the GUI
│   └── routes/                      # Example Gateway API HTTPRoute manifests
├── docker/                          # Container build definitions for operator and GUI
├── docs/                            # Published Helm repository index and packaged charts
├── operator_module/                 # Kopf operator implementation and Python deps
├── weka-app-store-operator-chart/   # Source Helm chart for deploying operator and GUI
│   ├── templates/                   # Helm-rendered Kubernetes resources
│   └── charts/                      # Helm dependency/cache directory
├── weka-csi-config/                 # WEKA CSI and storage-related example manifests
│   ├── tmp/                         # Scratch and alternate sample YAMLs
│   └── weka-operator/               # Nested chart/config fragment for WEKA operator setup
├── README.md                        # Repository-level usage and publishing guide
├── test-pvc.yaml                    # Manual PVC example
└── test-pvc-pod.yaml                # Manual pod+PVC example
```

## Directory Purposes

**`app-store-gui/`:**
- Purpose: Source tree for the web UI/backend.
- Contains: Python dependency manifest in `app-store-gui/requirements.txt` and the runtime package in `app-store-gui/webapp/`.
- Key files: `app-store-gui/webapp/main.py`, `app-store-gui/webapp/templates/index.html`, `app-store-gui/webapp/templates/welcome.html`.
- Subdirectories: `webapp/templates/` for Jinja HTML files; no separated routers or service modules are present.

**`operator_module/`:**
- Purpose: Source tree for the Kubernetes operator.
- Contains: A single large Python module and its dependency manifest.
- Key files: `operator_module/main.py`, `operator_module/requirements.txt`.
- Subdirectories: None.

**`weka-app-store-operator-chart/`:**
- Purpose: Helm source-of-truth for cluster installation.
- Contains: `Chart.yaml`, default values, and templates for CRD, RBAC, operator deployment, service, service account, and GUI deployment.
- Key files: `weka-app-store-operator-chart/Chart.yaml`, `weka-app-store-operator-chart/values.yaml`, `weka-app-store-operator-chart/templates/crd.yaml`, `weka-app-store-operator-chart/templates/deployment.yaml`, `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
- Subdirectories: `templates/` for rendered resources; `charts/` exists but no checked-in dependency charts were found in the file list.

**`docs/`:**
- Purpose: Helm repository publication output.
- Contains: `index.yaml` plus versioned `.tgz` chart packages.
- Key files: `docs/index.yaml`, `docs/weka-app-store-operator-chart-0.1.54.tgz`.
- Subdirectories: None.

**`cluster_init/`:**
- Purpose: Cluster bootstrap manifests and related route examples.
- Contains: `WekaAppStore` bootstrap YAML, Gateway/Grafana route manifests, and supporting notes.
- Key files: `cluster_init/app-store-cluster-init.yaml`, `cluster_init/ai-edge-gateway.yaml`, `cluster_init/routes/appstore-route.yaml`.
- Subdirectories: `cluster_init/routes/` for HTTPRoute examples.

**`weka-csi-config/`:**
- Purpose: Sample manifests and values related to WEKA CSI and storage integration.
- Contains: StorageClass YAMLs, secrets, WEKA client manifests, temporary variants, and a nested `weka-operator` chart fragment.
- Key files: `weka-csi-config/blueprint-default-values.yaml`, `weka-csi-config/storageclass-wekafs-fs-api.yaml`, `weka-csi-config/wekaClientCR-online.yaml`, `weka-csi-config/weka-operator/Chart.yaml`.
- Subdirectories: `tmp/` for alternate or transient YAMLs; `weka-operator/` for chart/config files.

**`docker/`:**
- Purpose: Build recipes for the two runtime images.
- Contains: One Dockerfile for the operator and one for the GUI.
- Key files: `docker/operator.Dockerfile`, `docker/webapp.Dockerfile`.
- Subdirectories: None.

**`.planning/codebase/`:**
- Purpose: Generated repository mapping output.
- Contains: Architecture and structure documents for planning workflows.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.
- Subdirectories: None at present.

## Key File Locations

**Entry Points:**
- `operator_module/main.py`: Kopf operator entry module with create, update, and delete handlers for `warp.io/v1alpha1` `wekaappstores`.
- `app-store-gui/webapp/main.py`: FastAPI application entry module serving HTML, JSON APIs, and SSE streams.
- `cluster_init/app-store-cluster-init.yaml`: Declarative bootstrap entry for a cluster-initialization app stack.

**Configuration:**
- `weka-app-store-operator-chart/values.yaml`: Helm defaults controlling image tags, RBAC, watch scope, service settings, and GUI enablement.
- `weka-app-store-operator-chart/Chart.yaml`: Chart metadata and versioning.
- `operator_module/requirements.txt`: Operator Python dependencies.
- `app-store-gui/requirements.txt`: GUI Python dependencies.

**Core Logic:**
- `operator_module/main.py`: Helm orchestration, YAML application, dependency resolution, CRD strategy, and status reconciliation.
- `app-store-gui/webapp/main.py`: UI routes, Kubernetes inspection, blueprint application, git-sync management, and health/readiness handling.

**Packaging and Publish Artifacts:**
- `docker/operator.Dockerfile`: Operator image build, including `kubectl` and `helm`.
- `docker/webapp.Dockerfile`: GUI image build, including `git-sync`.
- `docs/index.yaml`: Published Helm repo index.

**Operational Assets:**
- `cluster_init/`: Bootstrap and route manifests used by operators and the GUI.
- `weka-csi-config/`: Storage and CSI examples not loaded directly by the chart code in this repository.
- `test-pvc.yaml`: Manual PVC example.
- `test-pvc-pod.yaml`: Manual pod/PVC example.

**Documentation:**
- `README.md`: Installation, configuration, upgrade, uninstall, and publishing instructions.
- `.planning/codebase/ARCHITECTURE.md`: Conceptual architecture map.
- `.planning/codebase/STRUCTURE.md`: Physical layout map.

## Naming Conventions

**Files:**
- `main.py`: Used as the primary runtime module in both `operator_module/` and `app-store-gui/webapp/`.
- `*.yaml`: Dominant format for Helm templates, CRDs, sample manifests, and published Helm index data.
- `*.html`: Jinja templates for blueprint and settings pages in `app-store-gui/webapp/templates/`.
- `*.Dockerfile`: Container build definitions in `docker/`.

**Directories:**
- Kebab-case is used for most top-level feature and packaging directories, such as `app-store-gui/` and `weka-app-store-operator-chart/`.
- Snake_case also appears in operational asset directories, notably `cluster_init/` and `operator_module/`.

**Special Patterns:**
- `templates/` inside `weka-app-store-operator-chart/` contains Helm templates, while `templates/` inside `app-store-gui/webapp/` contains HTML views, so the same directory name has different semantics depending on subsystem.
- `docs/` is not prose documentation here; it is the checked-in Helm repository output referenced by `README.md`.
- The repository includes generated or transient artifacts in version control, such as `docs/*.tgz`, `app-store-gui/webapp/__pycache__/`, and files under `weka-csi-config/tmp/`.

## Where to Add New Code

**New Operator Behavior:**
- Primary code: `operator_module/main.py`.
- Configuration surface: `weka-app-store-operator-chart/templates/crd.yaml` and `weka-app-store-operator-chart/values.yaml` if the behavior needs new spec or deployment settings.
- Container packaging: `docker/operator.Dockerfile` if new binaries or system packages are required.

**New GUI Endpoint or Page:**
- Implementation: `app-store-gui/webapp/main.py`.
- HTML template: `app-store-gui/webapp/templates/`.
- Container packaging: `docker/webapp.Dockerfile` if new runtime tools are required.

**New Cluster-Install Resource:**
- Helm-managed install path: `weka-app-store-operator-chart/templates/`.
- Bootstrap-only manifest: `cluster_init/` if it is part of initialization content rather than the chart itself.

**New Published Chart Version:**
- Source changes: `weka-app-store-operator-chart/`.
- Published artifacts: `docs/index.yaml` and a new `docs/weka-app-store-operator-chart-<version>.tgz`.

**New Storage or WEKA Integration Example:**
- Sample manifests: `weka-csi-config/`.
- Ad hoc cluster test manifests: repository root next to `test-pvc.yaml` if they remain one-off examples.

## Special Directories

**`docs/`:**
- Purpose: Generated Helm repository artifacts for GitHub Pages consumption.
- Source: Produced from `weka-app-store-operator-chart/` and referenced in the publishing steps in `README.md`.
- Committed: Yes.

**`weka-csi-config/tmp/`:**
- Purpose: Temporary or alternate storage-related YAML samples.
- Source: Hand-maintained sample manifests; no generation script is present in this repository.
- Committed: Yes.

**`app-store-gui/webapp/__pycache__/`:**
- Purpose: Python bytecode cache.
- Source: Interpreter-generated.
- Committed: Yes.

**`.planning/codebase/`:**
- Purpose: Planning artifacts produced by repository mapping workflows.
- Source: Generated during analysis tasks.
- Committed: Not determined from repository evidence alone.

*Structure analysis: 2026-03-17*
*Update when directory structure changes*
