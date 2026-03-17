# Technology Stack

**Analysis Date:** 2026-03-17

## Languages

**Primary:**
- Python 3.13 - Application code for the web UI and Kubernetes operator in `app-store-gui/webapp/main.py`, `operator_module/main.py`, and both Dockerfiles in `docker/`.

**Secondary:**
- YAML - Kubernetes manifests, Helm chart definitions, and blueprint configuration in `weka-app-store-operator-chart/`, `cluster_init/`, `weka-csi-config/`, and `docs/index.yaml`.
- HTML with Jinja templating - Server-rendered UI templates in `app-store-gui/webapp/templates/`.
- JavaScript (browser-side, CDN delivered) - Inline UI behavior inside templates such as `app-store-gui/webapp/templates/index.html` and `app-store-gui/webapp/templates/settings.html`.

## Runtime

**Environment:**
- Python 3.13 slim containers - Base runtime for both the web UI and operator in `docker/webapp.Dockerfile` and `docker/operator.Dockerfile`.
- Kubernetes cluster - Required execution target per `README.md` and the Helm chart under `weka-app-store-operator-chart/`.
- Helm CLI - Installed into the operator image and used by `operator_module/main.py` for release lifecycle operations.
- kubectl - Installed into the operator image and used for readiness checks and raw manifest application in `operator_module/main.py`.

**Package Manager:**
- pip - Dependencies installed from `app-store-gui/requirements.txt` and `operator_module/requirements.txt`.
- Lockfile: none found; the repository uses unpinned lower-bound requirements instead of a resolved lock file.

## Frameworks

**Core:**
- FastAPI - HTTP server and API layer for the app store UI in `app-store-gui/webapp/main.py`.
- Starlette middleware - Request gating for cluster initialization state in `app-store-gui/webapp/main.py`.
- Jinja2 - HTML template rendering in `app-store-gui/webapp/main.py` and `app-store-gui/webapp/templates/`.
- Kopf - Operator framework entrypoint and event handlers in `operator_module/main.py`.
- kr8s and Kubernetes Python client - Cluster API access from both the UI and operator in `app-store-gui/webapp/main.py` and `operator_module/main.py`.
- Helm v3 chart packaging - Deployment packaging in `weka-app-store-operator-chart/`.

**Testing:**
- No Python test framework or automated test suite is present in the repository root, `app-store-gui/`, or `operator_module/`.
- Helm chart smoke testing only appears as the chart test manifest `weka-app-store-operator-chart/templates/tests/test-connection.yaml`.

**Build/Dev:**
- Uvicorn - ASGI server used as the web UI process in `docker/webapp.Dockerfile` and declared in `app-store-gui/requirements.txt`.
- Docker - Container build and runtime packaging in `docker/webapp.Dockerfile` and `docker/operator.Dockerfile`.
- GitHub Pages style Helm repo publishing - Packaged charts and index management in `docs/index.yaml` and the publishing steps in `README.md`.

## Key Dependencies

**Critical:**
- `fastapi>=0.111.0` - Web API and route handling for the UI in `app-store-gui/requirements.txt`.
- `uvicorn[standard]>=0.30.0` - Web server process for `app-store-gui/webapp/main.py`, declared in `app-store-gui/requirements.txt`.
- `Jinja2>=3.1.4` - Template rendering for the HTML UI in `app-store-gui/requirements.txt`.
- `kopf>=1.38.0` - Operator event lifecycle framework in `operator_module/requirements.txt`.
- `kr8s>=0.17.0` - Higher-level Kubernetes object access in `operator_module/requirements.txt`.
- `kubernetes>=27.0.0` - Official Kubernetes API client used by both services in `app-store-gui/requirements.txt` and `operator_module/requirements.txt`.
- `PyYAML>=6.0.1` - Manifest and values parsing in the UI and operator, declared in `app-store-gui/requirements.txt`.

**Infrastructure:**
- Helm binary `v3.14.4` - Installed in `docker/operator.Dockerfile` and invoked throughout `operator_module/main.py`.
- git-sync `v4.5.0` - Included in `docker/webapp.Dockerfile` and also referenced by the GUI deployment in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
- kubectl - Installed in `docker/operator.Dockerfile` and used by operator readiness and manifest logic in `operator_module/main.py`.

## Configuration

**Environment:**
- Web UI configuration is primarily environment-variable driven: `BLUEPRINTS_DIR`, `GIT_SYNC_REPO`, `GIT_SYNC_BRANCH`, `GIT_SYNC_ROOT`, `GIT_SYNC_LINK`, `SYNC_TOKEN`, `KUBERNETES_AUTH_MODE`, `READINESS_TTL_SECONDS`, `READINESS_SKIP_K8S`, `KUBERNETES_CLUSTER_NAME`, `GIT_SYNC_VERSION`, and `GIT_SYNC_DOWNLOAD_URL` are read in `app-store-gui/webapp/main.py`.
- Operator configuration is also environment-variable driven, with `HELM_CMD_TIMEOUT` explicitly consumed in `operator_module/main.py`.
- Helm installation settings live in `weka-app-store-operator-chart/values.yaml`, including image, RBAC, watch scope, service, and GUI toggles.

**Build:**
- Container builds are defined in `docker/webapp.Dockerfile` and `docker/operator.Dockerfile`.
- Chart metadata and install-time defaults are defined in `weka-app-store-operator-chart/Chart.yaml` and `weka-app-store-operator-chart/values.yaml`.
- Manual Helm repository publishing uses `docs/index.yaml` plus the packaging steps in `README.md`.

## Platform Requirements

**Development:**
- Any environment that can run Python and access a Kubernetes cluster, but the documented path is containerized Kubernetes deployment rather than a local dev stack in `README.md`.
- Helm 3.10+ and `kubectl` are listed as prerequisites in `README.md`.
- Cluster admin privileges are expected for the chart because it installs RBAC and a CRD per `README.md`.

**Production:**
- Kubernetes is the primary deployment target; the repository packages the operator and GUI as containers and installs them via the Helm chart in `weka-app-store-operator-chart/`.
- The chart is published as a Helm repository in `docs/index.yaml` and consumed via the repository URL in `README.md`.
- GPU-capable Kubernetes clusters are part of the intended runtime based on `README.md`, `cluster_init/app-store-cluster-init.yaml`, and multiple `nvidia.com/gpu` references in `weka-csi-config/blueprint-default-values.yaml`.

---

*Stack analysis: 2026-03-17*
*Update after major dependency changes*
