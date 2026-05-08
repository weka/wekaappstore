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


## Variable substitution in AppStack manifests

The operator performs single-pass `${VAR}` substitution over `kubernetesManifest:` strings and `valuesFiles:` content (loaded from ConfigMaps and Secrets) before they are applied or merged into Helm values. This makes a single AppStack CR portable across namespaces and environments — the same blueprint deploys into `staging`, `aidp-prod`, or `aidp-test` by changing `metadata.namespace` and the `spec.appStack.variables` map, without forking the YAML.

### Syntax

| Syntax | Behavior |
| --- | --- |
| `${VAR}` | Substituted with the value of `VAR` from `spec.appStack.variables`. Strict — undefined references fail loudly. |
| `$$` | Literal dollar sign. Use this when a manifest needs a real `$` (for example a database password starting with `$`). |
| `${namespace}` | Auto-defaults to the CR's `metadata.namespace`. You can override by listing `namespace:` explicitly under `spec.appStack.variables`. |
| undefined `${VAR}` | Raises `kopf.PermanentError` naming the variable, the component, and the manifest/valuesFiles location. The CR must be fixed and re-applied. |
| invalid key (e.g., `my-host`) | Rejected at admission by the CRD schema, and again at the operator. Variable names must match the Python identifier pattern `[_a-zA-Z][_a-zA-Z0-9]*`. |

### Worked example

The example below mirrors the AIDP migration pattern: two components, `${namespace}` auto-defaulting, an explicit `${milvusHost}` consumed both inside a `kubernetesManifest` and inside a ConfigMap-loaded `valuesFiles` reference.

```yaml
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: ai-research
  namespace: aidp-prod
spec:
  appStack:
    variables:
      milvusHost: milvus.aidp-prod.svc.cluster.local
    components:
      - name: vector-db
        helmChart:
          repository: oci://registry.example.com/charts
          name: milvus
          version: 4.2.1
        valuesFiles:
          - kind: ConfigMap
            name: milvus-values
            key: values.yaml
        # The ConfigMap above contains:
        #   externalAccess:
        #     enabled: true
        #     host: ${milvusHost}
        # which the operator renders to milvus.aidp-prod.svc.cluster.local
        # before merging into Helm values.

      - name: ingress
        kubernetesManifest: |
          apiVersion: networking.k8s.io/v1
          kind: Ingress
          metadata:
            name: ai-research-ingress
            namespace: ${namespace}
          spec:
            rules:
              - host: ai-research.example.com
                http:
                  paths:
                    - path: /
                      pathType: Prefix
                      backend:
                        service:
                          name: ai-research
                          port:
                            number: 80
```

Apply the same blueprint to a different namespace by changing `metadata.namespace` (and the `milvusHost` value to match):

```yaml
metadata:
  name: ai-research
  namespace: aidp-test
spec:
  appStack:
    variables:
      milvusHost: milvus.aidp-test.svc.cluster.local
    components:
      # ... unchanged ...
```

### Variable values are NOT recursively resolved

> **Note:** Variable values are taken literally — no recursive resolution. The operator runs a single substitution pass; `${...}` tokens that appear inside variable VALUES are NOT expanded.

```yaml
# WRONG (this does NOT work):
spec:
  appStack:
    variables:
      milvusHost: milvus.${namespace}.svc.cluster.local   # nested ${} is NOT expanded
```

```yaml
# CORRECT (use fully-resolved values):
spec:
  appStack:
    variables:
      milvusHost: milvus.aidp-prod.svc.cluster.local
```

If you have many environments, generate the fully-resolved CR per environment from your own templating tool (Helm chart wrapper, kustomize, sed, etc.) — the operator deliberately stops at single-pass substitution to keep failure modes predictable.

### Operator-control fields are NOT templated

The fields below are operator-control fields and are NOT subject to `${VAR}` substitution. Putting `${...}` in any of them is a no-op that will silently fail to resolve at deploy time:

- `helmChart.repository`, `helmChart.name`, `helmChart.version`
- `releaseName`
- `targetNamespace`
- `readinessCheck.*` (`type`, `name`, `namespace`, `timeout`)

**Recommendation:** Omit `targetNamespace` and let the operator default to `metadata.namespace`. Templating is not supported on this field; setting it pins the component to a specific namespace and defeats the portability the variables block provides.

### Errors

- **Undefined variable** (e.g., `${unset}` appears in a manifest with no matching key in `spec.appStack.variables`): `kopf.PermanentError` is raised. The error message names the variable, the component, and the source location. The reconcile does NOT retry — the CR must be edited and re-applied.
- **Missing referenced ConfigMap or Secret** (e.g., `valuesFiles[].name` does not exist in the cluster yet): `kopf.TemporaryError(delay=30)` is raised. The operator retries every 30 seconds — useful when the ConfigMap is created after the CR.
- **Malformed `${...}`** (e.g., `${}`, `${123}`, bare `$` at end of line): `kopf.PermanentError` is raised. The error message names the malformed placeholder.
- **Invalid variable key** (e.g., `my-host` with a hyphen): `kopf.PermanentError` is raised at the start of reconcile, before any deployment work. Variable names must match `[_a-zA-Z][_a-zA-Z0-9]*`.


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
