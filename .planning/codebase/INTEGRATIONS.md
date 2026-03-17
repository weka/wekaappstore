# External Integrations

**Analysis Date:** 2026-03-17

## APIs & External Services

**Kubernetes API:**
- Kubernetes cluster control plane - Primary external system for both products.
  - Integration method: Official Kubernetes Python client plus `kr8s` object access in `app-store-gui/webapp/main.py` and `operator_module/main.py`.
  - Auth: In-cluster ServiceAccount or local kubeconfig selected by `KUBERNETES_AUTH_MODE` in `app-store-gui/webapp/main.py`.
  - Resources used: CRDs and custom objects under `warp.io`, nodes, namespaces, pods, storage classes, and other cluster resources referenced in `app-store-gui/webapp/main.py`, `operator_module/main.py`, and `weka-app-store-operator-chart/values.yaml`.

**Helm Chart Repositories:**
- Arbitrary Helm repositories and OCI registries - Operator-managed application installs.
  - Integration method: Shelling out to `helm install`, `helm upgrade`, `helm uninstall`, `helm show crds`, and `helm repo add` in `operator_module/main.py`.
  - Auth: No credential flow is implemented in code; repository access depends on Helm client environment and reachable repos.
  - Concrete examples: `https://nvidia.github.io/gpu-operator` and `oci://docker.io/envoyproxy` in `cluster_init/app-store-cluster-init.yaml`.

**Blueprint Git Repository:**
- GitHub repository `https://github.com/weka/warp-blueprints.git` - Source of deployable blueprint manifests for the GUI.
  - Integration method: `git-sync` sidecar/init-container and optional runtime sync endpoint logic in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml` and `app-store-gui/webapp/main.py`.
  - Auth: Optional shared token via `SYNC_TOKEN`; repo URL and sync parameters come from `GIT_SYNC_REPO`, `GIT_SYNC_BRANCH`, `GIT_SYNC_ROOT`, and `GIT_SYNC_LINK` in `app-store-gui/webapp/main.py`.
  - Local consumption: Synced manifests are mounted under `/app/manifests` and resolved through `BLUEPRINTS_DIR` in `app-store-gui/webapp/main.py`.

**Frontend CDN Assets:**
- Google Fonts, Tailwind CDN, React UMD, ReactDOM UMD, Emotion UMD, MUI UMD, and Babel standalone - Browser-loaded assets for the server-rendered UI templates.
  - Integration method: Direct `<script>` and `<link>` tags in `app-store-gui/webapp/templates/index.html`, `welcome.html`, `settings.html`, and blueprint-specific templates.
  - Auth: None.
  - Operational impact: UI rendering depends on third-party CDN availability at runtime in the browser.

## Data Storage

**Cluster-State Storage:**
- Kubernetes etcd-backed resources - Effective system of record for app installations, configuration, and status.
  - Connection: Kubernetes API calls from `app-store-gui/webapp/main.py` and `operator_module/main.py`.
  - Objects used: `WekaAppStore` custom resources, `ConfigMap` values files, `Secret` values files, namespaces, and workload resources.
  - Persistence model: Repository code stores deployment intent in cluster objects rather than in a separate application database.

**File Storage:**
- Local container filesystem and mounted blueprint volume - Template assets, synced manifests, and temporary Helm values files.
  - Web UI paths: `app-store-gui/webapp/templates/`, `app-store-gui/webapp/app_store_logo.png`, and mounted `/app/manifests` in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
  - Operator paths: Temporary values YAML files are written via `tempfile` in `operator_module/main.py`.
  - Additional storage integration: WEKA CSI and storage class examples live in `weka-csi-config/` and are referenced by the UI through storage-class inspection in `app-store-gui/webapp/main.py`.

**Caching:**
- None observed as a dedicated external cache.
  - Short-lived readiness caching is in-process only via environment-controlled TTL logic in `app-store-gui/webapp/main.py`.

## Authentication & Identity

**Cluster Identity:**
- Kubernetes ServiceAccounts and RBAC - Primary identity mechanism for both runtime components.
  - Web UI: Service account and cluster roles are declared in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
  - Operator: Service account and RBAC are managed by the Helm chart templates in `weka-app-store-operator-chart/templates/serviceaccount.yaml`, `clusterrole.yaml`, and `clusterrolebinding.yaml`.
  - Local development fallback: kubeconfig-based auth is supported in `app-store-gui/webapp/main.py`.

**User Authentication:**
- None found.
  - The FastAPI app does not implement user login, OAuth, or session-backed identity in `app-store-gui/webapp/main.py`.

## Monitoring & Observability

**Application Logging:**
- stdout/stderr logging only.
  - Web UI uses Python logging in `app-store-gui/webapp/main.py`.
  - Operator uses Python logging and Kopf logging in `operator_module/main.py`.

**Cluster Monitoring Stack:**
- kube-prometheus-stack, Grafana, and Prometheus Adapter - Installed as managed components of cluster initialization.
  - Integration method: Declared as `appStack.components` in `cluster_init/app-store-cluster-init.yaml`.
  - Access path: Gateway routes for Grafana are declared in `cluster_init/app-store-cluster-init.yaml` and `cluster_init/routes/grafana-route.yaml`.

**External Error Tracking / Product Analytics:**
- None found in repository source or manifests.

## CI/CD & Deployment

**Hosting:**
- Kubernetes via Helm chart - Main deployment mechanism.
  - Deployment assets: `weka-app-store-operator-chart/` plus container definitions in `docker/`.
  - Published chart repository: `https://weka.github.io/wekaappstore` referenced in `README.md` and materialized in `docs/index.yaml`.

**CI Pipeline:**
- No automated CI workflows found.
  - `.github/` contains only issue templates and no GitHub Actions workflow files.

**Artifact Distribution:**
- Docker images are pulled from external registries.
  - GUI image example: `wekachrisjen/weka-app-store-gui:v0.35` in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.
  - Combined chart image default: `wekachrisjen/weka-app-store-multi-arch:v0.9` in `weka-app-store-operator-chart/values.yaml`.
  - Build-time dependencies also pull from `registry.k8s.io`, `dl.k8s.io`, and `get.helm.sh` in both Dockerfiles.

## Environment Configuration

**Development:**
- Required cluster connectivity comes from in-cluster credentials or kubeconfig as implemented in `app-store-gui/webapp/main.py`.
- Blueprint sync behavior depends on `GIT_SYNC_REPO`, `GIT_SYNC_BRANCH`, `GIT_SYNC_ROOT`, `GIT_SYNC_LINK`, `BLUEPRINTS_DIR`, and optional `SYNC_TOKEN` in `app-store-gui/webapp/main.py`.
- Operator runtime behavior depends on `HELM_CMD_TIMEOUT` in `operator_module/main.py`.

**Staging / Production:**
- Helm values drive deployment shape through `weka-app-store-operator-chart/values.yaml`.
- Cluster-init blueprints embed environment-specific external endpoints such as Prometheus, Envoy, and Gateway API URLs in `cluster_init/app-store-cluster-init.yaml`.
- WEKA storage and runtime examples include external image registries and service endpoints in `weka-csi-config/`.

## Webhooks & Callbacks

**Incoming:**
- None found. No webhook receiver endpoints are defined in `app-store-gui/webapp/main.py` or `operator_module/main.py`.

**Outgoing:**
- None found beyond direct Kubernetes, Helm, git-sync, and browser CDN fetches.

---

*Integration audit: 2026-03-17*
*Update when adding/removing external services*
