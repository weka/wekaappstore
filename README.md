# WEKA App Store Helm Repository

This repository hosts a public Helm chart for deploying the WEKA App Store Operator and its web UI onto a Kubernetes cluster.

The WEKA App Store is designed to help user to install WEKA designed blueprints for machine learning use cases.

AI blueprints are installed by the operator as Custom Resources (CRs). Multiple Blueprints can be installed into their own namespaces as long as there is enough resources in the Kubernetes Cluster.

Helm repo URL: https://weka.github.io/wekaappstore

Chart name: weka-app-store-operator-chart


## Prerequisites
- A Kubernetes cluster (v1.24+ recommended)
- A Kubernetes cluster with NVIDIA GPU support
- kubectl configured to point at the target cluster
- Helm v3.10+ installed
- Cluster admin privileges (the chart installs RBAC resources and a CRD by default)


## Quick start
1) Add the repo and update your local cache
- helm repo add wekaappstore https://weka.github.io/wekaappstore
- helm repo update

2) Inspect available versions (optional)
- helm search repo wekaappstore/weka-app-store-operator-chart --versions

3) Install into a dedicated namespace (recommended)
- kubectl create namespace weka-app-store
- helm install weka-app-store wekaappstore/weka-app-store-operator-chart -n weka-app-store

4) Verify the deployment
- kubectl get pods -n weka-app-store
- A single pod called weka-app-store-operator should be running
- To check the CRD is installed type kubectl get crd wekaappstores.warp.io

## Common configuration
You can override values via --set or a custom values.yaml. The chart exposes these notable options (see weka-app-store-operator-chart/values.yaml for the full list):

- image.repository: Container image repo
  - Default: wekachrisjen/weka-app-store-multi-arch
- image.tag: Container image tag
  - Default: "v0.2"
- image.pullPolicy: Image pull policy
  - Default: IfNotPresent
- service.type: Kubernetes Service type
  - Default: ClusterIP (use LoadBalancer or NodePort for external access)
- service.port: Service port
  - Default: 80
- serviceAccount.create: Whether to create a ServiceAccount
  - Default: true
- serviceAccount.name: Custom SA name (empty lets Helm generate one)
  - Default: ""
- serviceAccount.automount: Automount SA token
  - Default: true
- rbac.create: Create RBAC resources
  - Default: true
- rbac.clusterWide: Grant cluster-wide permissions (ClusterRole/ClusterRoleBinding)
  - Default: true
- rbac.clusterRoleName: Custom ClusterRole name
  - Default: "" (auto-derived)
- rbac.clusterRole.rules: Base RBAC rules for the operator
  - You can add rbac.clusterRole.extraRules to extend without replacing defaults
- customResourceDefinition.create: Create the operator CRD
  - Default: true
- autoscaling.enabled: Enable HPA (templates expect values if enabled)
  - Default: false

Example: expose the service externally and change the image tag
- helm upgrade --install weka-app-store wekaappstore/weka-app-store-operator-chart \
  -n weka-app-store \
  --set service.type=LoadBalancer \
  --set image.tag=v0.3

Alternatively, create a custom values.yaml and pass -f values.yaml.


## Upgrading
Fetch the latest chart index and upgrade the release:
- helm repo update
- helm upgrade weka-app-store wekaappstore/weka-app-store-operator-chart -n weka-app-store

To upgrade with custom values:
- helm upgrade weka-app-store wekaappstore/weka-app-store-operator-chart -n weka-app-store -f my-values.yaml


## Uninstalling
Remove all release-managed resources:
- helm uninstall weka-app-store -n weka-app-store

Notes:
- If you installed the CRD via the chart, Helm will remove it with the release when managed through templates. If the CRD remains due to finalizers or external references, you may need to remove finalizers or delete the CRD manually:
  - kubectl delete crd wekaappstores.warp.io
- You can also delete the namespace if it was dedicated to this chart:
  - kubectl delete namespace weka-app-store


## Troubleshooting
- Check pod events and logs:
  - kubectl describe pods -n weka-app-store
  - kubectl logs -n weka-app-store deploy/weka-app-store-operator-chart
- Validate RBAC permissions if the operator reports access issues.
- Ensure the CRD exists if the operator manages custom resources:
  - kubectl get crd wekaappstores.warp.io

## Readiness checks for components (pods or deployments)
When using AppStack components, you can control how the operator waits for a component to become ready after installation. The operator supports waiting on either pods (default) or deployments using `kubectl wait` under the hood.

Examples:

- Wait for a specific deployment by name (recommended when you know the resource name):

  appStack:
    components:
      - name: envoy-gateway
        enabled: true
        helmChart:
          repository: oci://example.registry/charts
          name: envoy-gateway
          version: 1.2.3
        readinessCheck:
          type: deployment
          name: envoy-gateway
          namespace: envoy-gateway-system
          timeout: 300

- Wait for pods by label selector (default behavior if name is not provided):

  appStack:
    components:
      - name: my-component
        enabled: true
        helmChart:
          repository: https://example.com/helm
          name: my-chart
          version: 0.1.0
        readinessCheck:
          type: pod
          matchLabels:
            app.kubernetes.io/instance: my-component
            app.kubernetes.io/name: my-chart
          timeout: 300

Notes:
- If `readinessCheck.name` is set, the operator waits for `type/name` in the specified namespace (or the component targetNamespace if omitted).
- If `name` is not set, the operator uses the selector (or matchLabels) to wait for the corresponding resource(s). For many Helm charts the operator auto-derives a good default selector when none is supplied.
- Supported types: `pod`, `deployment`, `statefulset`, `job`.

## Publishing (maintainers)
For maintainers who need to publish a new chart version to GitHub Pages under docs/:
1) Bump version in weka-app-store-operator-chart/Chart.yaml
2) Package the chart:
   - helm package weka-app-store-operator-chart -d docs/
3) Rebuild the repo index (update the URL to your GitHub Pages path if it changes):
   - helm repo index docs --url https://weka.github.io/wekaappstore
4) Commit and push docs/ (including the new .tgz and updated index.yaml) to the default branch
5) Consumers can then helm repo update and install the new version
