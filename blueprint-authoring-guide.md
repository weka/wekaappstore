# WEKA App Store: Blueprint Authoring Guide

This guide explains how to create a blueprint for the WEKA App Store. It is written for engineers who understand Kubernetes and Helm but have not worked with the App Store before. It covers the complete specification for the `WekaAppStore` and `WarpCredential` custom resources, the blueprint YAML file format, the two-layer variable substitution system, and the credential management system. Working through the guide from start to finish will give you enough knowledge to author a blueprint that installs multiple Kubernetes applications in the correct order, uses registry and storage credentials securely, and presents an install form to the operator that collects only what is needed.

---

## Table of Contents

1. [How the App Store Works](#1-how-the-app-store-works)
2. [Blueprint File Structure](#2-blueprint-file-structure)
3. [The x-variables Block — GUI Install Form Schema](#3-the-x-variables-block--gui-install-form-schema)
3b. [The x-requirements Block — Cluster Compatibility](#3b-the-x-requirements-block--cluster-compatibility)
4. [The WekaAppStore Custom Resource](#4-the-wekaappstore-custom-resource)
5. [appStack — Multi-Component Stack Specification](#5-appstack--multi-component-stack-specification)
6. [appStack Variables — Operator-Level Substitution](#6-appstack-variables--operator-level-substitution)
7. [Defining Components](#7-defining-components)
8. [Helm Chart Components](#8-helm-chart-components)
9. [Kubernetes Manifest Components](#9-kubernetes-manifest-components)
10. [Loading Helm Values from ConfigMaps and Secrets](#10-loading-helm-values-from-configmaps-and-secrets)
11. [Component Dependencies and Deployment Order](#11-component-dependencies-and-deployment-order)
12. [Readiness Checks](#12-readiness-checks)
13. [Namespace Management](#13-namespace-management)
14. [Credential Management — The WarpCredential Resource](#14-credential-management--the-warpcredential-resource)
15. [Two-Layer Variable Substitution Reference](#15-two-layer-variable-substitution-reference)
16. [Inspecting Resources with kubectl](#16-inspecting-resources-with-kubectl)
17. [Lifecycle — Create, Update, Delete](#17-lifecycle--create-update-delete)
18. [Complete Worked Example](#18-complete-worked-example)
19. [Troubleshooting Reference](#19-troubleshooting-reference)

---

## 1. How the App Store Works

The WEKA App Store is a Kubernetes operator. It extends the Kubernetes API with two custom resource types — `WekaAppStore` and `WarpCredential` — and watches for those resources to appear or change. When they do, it takes action: running Helm commands, applying manifests, or creating derived Kubernetes Secrets.

The operator runs as a pod in the `wekaappstore` namespace. It has cluster-level permissions to create Helm releases, apply manifests, read Secrets and ConfigMaps, and create Secrets in any namespace. Blueprints are stored in a separate git repository (`warp-blueprints`) that is mounted into the GUI pod via a `git-sync` sidecar. The GUI reads blueprint files from that mount and presents them to the user; the operator never reads blueprint files directly — it only sees the Kubernetes resources that result from applying them.

The journey from a blueprint YAML file to a running application stack involves two distinct phases:

**Phase 1 — GUI rendering.** When a user opens a blueprint page and clicks Install, the browser submits the form. The GUI loads the raw blueprint YAML file from disk, treats it as a Jinja2 template with `[[ ]]` variable delimiters, and renders it with the values the user entered. The result is a concrete Kubernetes YAML document — no `[[ ]]` tokens remain. The GUI then submits that document to the Kubernetes API server, which stores it in etcd as a `WekaAppStore` custom resource.

**Phase 2 — Operator reconciliation.** The operator detects the new `WekaAppStore` resource and reads its `spec`. It validates the variable declarations, resolves the component dependency graph, then deploys each component in order. For Helm components it runs `helm install` or `helm upgrade`. For raw manifest components it runs `kubectl apply`. After deploying each component, it optionally waits for readiness before moving to the next. Throughout this process it writes progress into the resource's `status` fields, which you can query with `kubectl`.

```
Blueprint YAML file (warp-blueprints repo)
        │
        │  1.  User fills install form in the GUI browser.
        │      GUI renders blueprint with [[ ]] tokens replaced.
        ▼
Installed blueprint stored in Kubernetes
(all [[ ]] tokens already resolved — App Store never sees them)
        │
        │  2.  Operator pod detects the new resource.
        │      Validates variables. Resolves dependency order.
        ▼
For each component (in dependency order):
  ┌─ helmChart component ──────────────────────────────────────────┐
  │  Operator substitutes ${VAR} into valuesFiles content.         │
  │  Merges inline values + valuesFiles values.                    │
  │  Runs: helm install/upgrade <release> <chart> --values <file>  │
  └────────────────────────────────────────────────────────────────┘
  ┌─ kubernetesManifest component ─────────────────────────────────┐
  │  Operator substitutes ${VAR} into the manifest string.         │
  │  Runs: kubectl apply -f <rendered manifest>                    │
  └────────────────────────────────────────────────────────────────┘
        │
        │  Waits for readiness (unless waitForReady: false)
        ▼
  status.appStackPhase = "Ready"
```

The two rendering passes — `[[ ]]` in the GUI layer and `${ }` in the operator layer — use different syntax intentionally. Section 15 covers both in full.

---

## 2. Blueprint File Structure

A blueprint is a single YAML file stored in the `warp-blueprints` repository. It contains two logical sections in one document:

- **`x-variables`** — a top-level key read only by the GUI. It describes the form fields to show on the blueprint install page and provides the schema for validating user input before deployment.
- **The `WekaAppStore` resource** — a standard Kubernetes resource document that the GUI submits to the cluster after rendering. This is what the operator acts on.

The file name (without the `.yaml` extension) becomes the blueprint's identifier in the GUI. A file at `my-blueprint/my-blueprint.yaml` or simply `my-blueprint.yaml` is accessible at `/blueprint/my-blueprint` in the GUI. The `x-variables` block makes the file discoverable — the GUI scanner only treats a file as a blueprint if it contains a non-empty `x-variables` key.

```yaml
# ── Section 1: GUI form schema ────────────────────────────────────────────────
x-variables:
  namespace:
    type: string
    required: true
    description: "Kubernetes namespace to deploy into"
  storage_class:
    type: string
    required: false
    description: "StorageClass for persistent volumes"
    placeholder: "e.g. weka-storageclass"
  ngc_credential:
    type: credential
    credential_type: nvidia-ngc
    required: true
    description: "NVIDIA NGC API key for pulling container images"

# ── Section 2: Kubernetes resource ───────────────────────────────────────────
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
  namespace: [[ namespace ]]        # GUI-layer token — resolved before Kubernetes sees this
spec:
  appStack:
    variables:
      storage_class: "[[ storage_class ]]"
      ngc_secret: "warp-[[ ngc_credential ]]-docker"
    components:
      - name: my-service
        helmChart:
          repository: https://charts.example.com
          name: my-chart
          version: "1.2.3"
        values:
          storageClass: "${storage_class}"     # operator-layer token — resolved at deploy time
```

The `[[ ]]` tokens exist only in the file on disk. By the time the document reaches the Kubernetes API, all `[[ ]]` tokens have been replaced with real values. The `${ }` tokens exist in the stored Kubernetes resource and are resolved by the operator when it reconciles.

---

## 3. The x-variables Block — GUI Install Form Schema

The `x-variables` block is a YAML map at the top level of the blueprint file. Each key in the map defines one field in the install form. The key name becomes the variable name used in `[[ ]]` tokens elsewhere in the file.

When the GUI renders a blueprint page, it reads this block and builds a form. Fields with `required: true` must be filled before the Install button becomes active. Fields of `type: credential` render as dropdowns populated with `WarpCredential` resources already registered in the App Store. If no credentials of the matching type exist, the Install button stays disabled and a link to the Settings page is shown instead.

### x-variables field properties

| Property | Type | Required | Description |
|---|---|---|---|
| `type` | string | Yes | The field type. Must be `string` or `credential`. |
| `required` | boolean | No | When `true`, the form disables the Install button until this field has a non-empty value. Defaults to `false`. |
| `description` | string | No | Help text rendered below the input field on the install form. |
| `placeholder` | string | No | Gray hint text shown inside the empty input box. Only applies to `type: string`. |
| `validate` | boolean | No | Set to `false` to skip the GUI's input-format validation for this field. By default fields whose name looks like a URL (`*_url`, `*_uri`, `*_endpoint`, `*_host`, …) are validated as `http(s)://` URLs; use `validate: false` for a bare host/IP or any value that should not be URL-checked. |
| `credential_type` | string | Required when `type` is `credential` | The category of credential to list. Must be one of `nvidia-ngc`, `huggingface`, or `weka-storage`. |

### `type: string`

A `type: string` field renders as a plain text input box. The user types a value and that raw string is used verbatim when rendering the blueprint template.

```yaml
x-variables:
  model_tag:
    type: string
    required: true
    description: "Container image tag for the model server"
    placeholder: "e.g. 1.0.0"
  storage_class:
    type: string
    required: false
    description: "Kubernetes StorageClass for persistent volumes. Leave empty to use the cluster default."
    placeholder: "e.g. weka-storageclass"
```

### `type: credential`

A `type: credential` field renders as a dropdown list. The dropdown is populated from `WarpCredential` resources of the specified `credential_type` that exist in the `wekaappstore` namespace. The value submitted when the user selects an entry is the `metadata.name` of that `WarpCredential` — for example, `my-ngc-key`. The blueprint author uses that name to construct the derived Secret name that the operator will have already created (see [Section 14](#14-credential-management--the-warpcredential-resource)).

```yaml
x-variables:
  ngc_credential:
    type: credential
    credential_type: nvidia-ngc
    required: true
    description: "NVIDIA NGC API key used to pull NIM container images and authenticate with the NGC model registry."
  hf_credential:
    type: credential
    credential_type: huggingface
    required: false
    description: "HuggingFace token for downloading gated models. Leave blank if no gated models are needed."
```

### How variable values flow into the blueprint

After the user submits the form, the GUI collects all field values into a dictionary. For example:

```json
{
  "namespace": "my-ai-stack",
  "storage_class": "weka-sc-api",
  "ngc_credential": "prod-ngc-key",
  "hf_credential": "prod-hf-token"
}
```

The GUI then opens the blueprint YAML file and renders it as a Jinja2 template using these values, replacing every `[[ key ]]` token with the corresponding string. The rendered document is a complete, valid Kubernetes YAML file with no `[[ ]]` tokens remaining. That rendered document is submitted directly to the Kubernetes API.

---

## 3b. The x-requirements Block — Cluster Compatibility

The blueprint page shows a **Cluster Compatibility** panel that compares the blueprint's resource needs against the cluster's currently **free** capacity (CPU cores, GPU devices, memory). The GUI determines those needs in this order:

1. **Declared (authoritative).** An optional top-level `x-requirements` block. Use this whenever the real needs cannot be read from the blueprint text — most importantly when GPU workloads are packaged in external Helm charts referenced via `valuesFiles` (the GUI cannot read those). This is the only way to show an exact GPU count for NIM/Helm-packaged blueprints.
2. **Inferred (fallback).** If there is no `x-requirements` block, the GUI sums container `resources.requests`/`limits` (`cpu`, `memory`, `nvidia.com/gpu`) across every `kubernetesManifest` and inline `helmChart.valuesContent`, multiplying by `replicas`.
3. **Honest unknown.** If a GPU is clearly needed (an `nvidia.com/gpu` limit, a `runtimeClassName: nvidia`, a `weka.io/nim-role` label, or an NVIDIA/NIM chart name) but the count cannot be determined, the page shows **"GPU required — count unknown"** rather than a misleading number. Add `x-requirements` to replace that with an exact figure.

```yaml
# ── Optional: declared resource requirements ─────────────────────────────────
x-requirements:
  cpu:
    cores: 16          # total vCPU the blueprint requests
  memory:
    gib: 64            # total memory in GiB
  gpu:
    count: 2           # total nvidia.com/gpu devices; omit to show "required, count unknown"
    model: H100        # optional, shown as a hint next to the GPU row
```

All three sub-blocks are optional. `gpu` with no `count` means "GPU required, count unknown"; `gpu: {count: 0}` means "no GPU required". A row with no declared or inferable value shows **"not specified"**.

---

## 4. The WekaAppStore Custom Resource

In the App Store, a deployed application is called a **blueprint**. When you install a blueprint through the GUI, the result is a `WekaAppStore` resource stored in Kubernetes. This resource holds everything the App Store needs to know about what to deploy: which Helm charts or manifests to apply, in what order, into which namespace, and with what values. You can think of it as the running record of a blueprint installation.

The `WekaAppStore` resource lives in API group `warp.io`, version `v1alpha1`. The plural name used with `kubectl` is `wekaappstores`. Every installed blueprint corresponds to exactly one `WekaAppStore` resource in the cluster.

The top-level `spec` supports three deployment modes. Only one mode may be active in a given resource:

| Mode | Spec field | When to use |
|---|---|---|
| Multi-component stack | `spec.appStack` | **The recommended mode for all new blueprints.** Deploys multiple components with explicit dependency ordering. |
| Single Helm chart | `spec.helmChart` + optional `spec.values` + optional `spec.valuesFiles` | Suitable for blueprints that install exactly one Helm release with no dependency management needed. |
| Legacy bare pod | `spec.image` + `spec.binary` | Creates a single Pod from an image. Not suitable for production use. |

All production blueprints use `spec.appStack`. The rest of this guide focuses on that mode.

### Top-level spec fields

| Field | Type | Required | Default | Description | Example |
|---|---|---|---|---|---|
| `appStack` | object | One of the three modes is required | — | Multi-component deployment specification. Contains the full component list, variables, and namespace settings. | See [Section 5](#5-appstack--multi-component-stack-specification) |
| `helmChart` | object | One of the three modes is required | — | Single Helm chart deployment. Use this only when you need to install one chart with no components or dependencies. | `helmChart: { repository: "https://charts.bitnami.com/bitnami", name: "redis", version: "19.6.4" }` |
| `values` | object | No | `{}` | Inline Helm values passed to `spec.helmChart`. Has no effect when using `spec.appStack` (per-component values go inside each component instead). | `values: { replicaCount: 1, auth: { enabled: false } }` |
| `valuesFiles` | array | No | `[]` | References to existing ConfigMaps or Secrets in the cluster whose content provides additional Helm values. Used with `spec.helmChart` only. | `valuesFiles: [{ kind: ConfigMap, name: my-values, key: values.yaml }]` |
| `targetNamespace` | string | No | Blueprint's own namespace | Kubernetes namespace to install the chart into. Used with `spec.helmChart` only. | `targetNamespace: "my-app"` |
| `image` | string | No | — | Container image for legacy pod mode only. | `image: "nginx:1.25"` |
| `binary` | string | No | — | Command to run in the container for legacy pod mode only. | `binary: "/bin/server"` |

### Viewing installed blueprints

```bash
# List all installed blueprints in a namespace
kubectl get wekaappstores -n <namespace>

# List all installed blueprints across the whole cluster
kubectl get wekaappstores -A

# Get the full blueprint spec and current status
kubectl get wekaappstore <name> -n <namespace> -o yaml

# Human-readable summary with events and status conditions
kubectl describe wekaappstore <name> -n <namespace>
```

---

## 5. appStack — Multi-Component Stack Specification

The `appStack` object is where you define the applications that make up the blueprint. Most blueprints deploy more than one application — for example, a vector database, a model server, and a frontend. Each of these is a **component**. The `appStack` holds the list of components plus some global settings that apply across all of them.

### appStack fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `components` | array | Yes | — | Ordered list of components to deploy. Must contain at least one entry. Each component is one Helm chart or one Kubernetes manifest. |
| `variables` | object | No | `{}` | Key-value pairs that can be referenced inside component manifests and values files using `${VAR}` syntax. See [Section 6](#6-appstack-variables--substitution-into-components). |
| `defaultNamespace` | string | No | Blueprint's own namespace | The Kubernetes namespace that all components deploy into unless they specify their own. |
| `namespaces` | object | No | `{}` | Per-component namespace overrides, keyed by component name. More specific than `defaultNamespace` but less specific than a namespace set directly on the component. |

The components in a blueprint are deployed one at a time, in order. If component B lists component A in its `dependsOn` field, then A is guaranteed to be running and healthy before B starts. Without any `dependsOn` declarations, components deploy from top to bottom in the order they are written. This is explained in detail in [Section 11](#11-component-dependencies-and-deployment-order).

### Complete appStack example

The example below shows all four top-level `appStack` fields together with representative components, so you can see how the structure looks as a whole:

```yaml
spec:
  appStack:
    defaultNamespace: my-app-namespace   # all components go here unless overridden

    namespaces:
      monitoring: monitoring             # "monitoring" component deploys into a different namespace

    variables:                           # values available to components as ${VAR}
      storage_class: "weka-storageclass-api"
      ngc_docker_secret: "warp-prod-ngc-key-docker"

    components:

      - name: database                   # component 1 — no dependsOn, deploys first
        helmChart:
          repository: https://charts.bitnami.com/bitnami
          name: postgresql
          version: "15.0.0"
        values:
          primary:
            persistence:
              storageClass: "[[ storage_class ]]"
        waitForReady: true
        readinessCheck:
          type: statefulset
          name: database-postgresql
          timeout: 300

      - name: app-server                 # component 2 — waits for database to be ready
        dependsOn:
          - database
        helmChart:
          repository: https://charts.example.com
          name: my-server
          version: "1.2.0"
        valuesFiles:                     # reads a ConfigMap that contains ${VAR} tokens
          - kind: ConfigMap
            name: app-server-values
            key: values.yaml
        waitForReady: true

      - name: monitoring                 # component 3 — deploys into its own namespace
        helmChart:
          repository: https://prometheus-community.github.io/helm-charts
          name: kube-prometheus-stack
          version: "58.0.0"
        targetNamespace: monitoring      # overrides defaultNamespace for this component only
        waitForReady: false              # don't wait — just fire and move on
```

### Status fields written by the operator

After the operator processes the resource, it writes status back into the resource. You can read this with `kubectl describe` or `kubectl get -o yaml`.

| Status field | Type | Description |
|---|---|---|
| `status.appStackPhase` | string | Overall state. One of `Pending`, `Installing`, `Ready`, `Failed`, or `Degraded`. |
| `status.componentStatus` | array | Per-component state. Each entry has `name`, `phase`, `releaseName`, `message`, and `lastTransitionTime`. |
| `status.conditions` | array | Standard Kubernetes conditions. The `Ready` condition is always present. |

```bash
# Watch the appStackPhase field update in real time
kubectl get wekaappstore my-blueprint -n my-namespace -w

# Pull the per-component status
kubectl get wekaappstore my-blueprint -n my-namespace \
  -o jsonpath='{.status.componentStatus[*]}'

# Get all conditions
kubectl get wekaappstore my-blueprint -n my-namespace \
  -o jsonpath='{.status.conditions[*]}'
```

---

## 6. appStack Variables — Substitution into Components

The `appStack.variables` block is a simple list of named values that you want to reuse across the components in your blueprint. Think of it as a set of configuration knobs that sit above all the components. Once you define a variable here, you can reference it inside any component's manifest text or values file, and the App Store will substitute the real value in at deploy time.

A common use case: you have three components, and all three need to know the StorageClass name or a container pull secret. Instead of repeating the same string in each component, you define it once in `variables` and reference it everywhere.

```yaml
spec:
  appStack:
    variables:
      storage_class: "weka-storageclass-api"
      ngc_docker_secret: "warp-prod-ngc-key-docker"
      model_name: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
      replica_count: "3"    # numbers and booleans must be quoted — variables are always strings
```

### Where variables are used in a component

Variables can be referenced in two specific places within a component using `${variable_name}` syntax:

**1. Inside a `kubernetesManifest` string.** When a component deploys a raw Kubernetes manifest, the App Store scans the manifest text for `${VAR}` tokens and replaces them before applying the manifest to the cluster.

```yaml
- name: namespace-setup
  kubernetesManifest: |
    apiVersion: v1
    kind: Namespace
    metadata:
      name: ${namespace}        # ${namespace} is always available automatically
      labels:
        storage: ${storage_class}   # replaced with "weka-storageclass-api"
```

**2. Inside a `valuesFiles` ConfigMap or Secret.** When a component loads its Helm values from a ConfigMap or Secret (see [Section 10](#10-loading-helm-values-from-configmaps-and-secrets)), the App Store reads the raw text of that ConfigMap/Secret and replaces `${VAR}` tokens before parsing it as YAML. This lets you store your Helm values in the cluster and still inject blueprint-level values into them at deploy time.

```yaml
# The ConfigMap "app-config" contains this text under key "values.yaml":
#
#   imagePullSecrets:
#     - name: "${ngc_docker_secret}"
#   persistence:
#     storageClass: "${storage_class}"
#
# When the App Store reads this ConfigMap, it replaces the tokens before
# passing the content to Helm as values.

- name: my-app
  helmChart:
    repository: oci://nvcr.io/nvidia
    name: my-server
    version: "1.0.0"
  valuesFiles:
    - kind: ConfigMap
      name: app-config          # variables are substituted into this ConfigMap's text
      key: values.yaml
```

**What variables do NOT reach:** The inline `values:` block on a component is a YAML object, not plain text, so `${VAR}` tokens written there will not be substituted — they would be passed literally to Helm. If you need a variable value in Helm chart settings and you are using the inline `values:` block, use `[[ variable_name ]]` instead (the GUI-layer syntax that is resolved before the blueprint reaches the cluster). The full explanation of both substitution systems is in [Section 15](#15-two-layer-variable-substitution-reference).

### The automatic `${namespace}` variable

You always have access to `${namespace}` without declaring it. It is automatically set to the Kubernetes namespace that the blueprint was installed into (the `metadata.namespace` of the blueprint resource). This is useful in `kubernetesManifest` components where you need to reference the target namespace.

### Variable naming rules

Variable names must follow standard identifier rules. A name that violates these rules causes the blueprint to fail before any components are deployed.

| Rule | Example valid name | Example invalid name |
|---|---|---|
| Must start with a letter or underscore | `storage_class`, `_private` | `1var`, `-var` |
| Remaining characters: letters, digits, underscores only | `model_name_v2` | `storage-class`, `storage.class` |
| Values must be quoted strings | `"weka-sc"`, `"42"`, `"true"` | `42` (bare number), `true` (bare boolean) |

### What happens when a variable is missing

If a `${VAR}` token references a name that is not in the `variables` map, the blueprint fails immediately with a permanent error — the App Store does not retry, and no components are deployed. The error message names the missing variable and can be read from the blueprint status:

```bash
kubectl describe wekaappstore my-blueprint -n my-namespace | grep -A5 "Message:"
```

### Escaping a literal dollar sign

If your manifest includes shell scripts or other content that legitimately uses `${...}` syntax (not a variable reference), write `$$` to produce a literal `$`:

```yaml
kubernetesManifest: |
  apiVersion: v1
  kind: ConfigMap
  data:
    startup.sh: |
      #!/bin/bash
      export PATH=$${PATH}:/opt/bin    # $${PATH} → ${PATH} (shell variable, not blueprint variable)
      echo "Blueprint namespace: ${namespace}"  # ${namespace} → actual namespace name
```

---

## 7. Defining Components

Each entry in `spec.appStack.components` represents one deployable unit. A component is either a Helm chart installation or a raw Kubernetes manifest application. The operator processes components in dependency order, deploying and optionally waiting for each one before moving to the next.

Every component must have a `name` that is unique within the stack, and must specify exactly one of `helmChart` or `kubernetesManifest`. If both are present, the `helmChart` block takes precedence and `kubernetesManifest` is ignored.

### Component common fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | — | Unique identifier for this component within the stack. Used in `dependsOn` references and in status reporting. |
| `description` | string | No | — | Free-text description. Appears in operator logs. Has no effect on deployment behaviour. |
| `enabled` | boolean | No | `true` | When `false`, the component is completely skipped during deployment and deletion. Useful for temporarily disabling a component without removing it from the spec. |
| `helmChart` | object | One required | — | Deploys a Helm chart. See [Section 8](#8-helm-chart-components). |
| `kubernetesManifest` | string | One required | — | Applies a raw Kubernetes manifest. See [Section 9](#9-kubernetes-manifest-components). |
| `values` | object | No | `{}` | Inline Helm values passed to this component's Helm chart. Not applicable to manifest components. |
| `valuesFiles` | array | No | `[]` | References to ConfigMaps or Secrets whose content is merged into Helm values. See [Section 10](#10-loading-helm-values-from-configmaps-and-secrets). |
| `targetNamespace` | string | No | Resolved from stack (see Section 13) | The Kubernetes namespace this component deploys into. |
| `dependsOn` | array of strings | No | `[]` | Names of other components that must reach Ready status before this component is deployed. |
| `waitForReady` | boolean | No | `true` | When `true`, the operator waits for the component to become ready before proceeding to the next. |
| `readinessCheck` | object | No | type: pod, timeout: 300s | Custom configuration for how readiness is determined. See [Section 12](#12-readiness-checks). |

### Enabled vs. disabled components

When `enabled: false`, the component is excluded from the enabled-components list before dependency resolution runs. Components that depend on a disabled component are not affected — the disabled component is simply absent from the graph. If a component with `enabled: true` lists a disabled component in its `dependsOn`, the deployment proceeds without waiting for the disabled component.

---

## 8. Helm Chart Components

A `helmChart` component instructs the operator to run `helm install` (for a new release) or `helm upgrade` (for an existing one). The operator writes Helm values to a temporary file, runs the command, then deletes the temporary file. The Helm timeout defaults to 900 seconds and can be changed with the `HELM_CMD_TIMEOUT` environment variable on the operator pod.

### helmChart fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `repository` | string | No | — | Helm repository URL. For HTTP/S repositories this is the index URL (e.g., `https://charts.bitnami.com/bitnami`). For OCI registries use the `oci://` prefix. The operator runs `helm repo add` automatically for HTTP/S URLs. |
| `name` | string | Yes | — | The Helm chart name within the repository (e.g., `redis`, `kube-prometheus-stack`). For OCI charts this is the chart name appended to the repository path. |
| `version` | string | No | Latest | The chart version to install. Leaving this unset always fetches the latest available version, which may cause unexpected changes on re-deployment. Pinning a version is strongly recommended for production blueprints. |
| `releaseName` | string | No | Component `name` | The Helm release name. This is the name Helm tracks the installation under and what appears in `helm list`. If not set, the operator uses the component's `name` value. |
| `crdsStrategy` | string | No | `Auto` | How to handle CRDs bundled in the chart. One of `Auto`, `Install`, or `Skip`. See below. |

### Repository URL formats

The operator handles three styles of chart reference, determined automatically from the `repository` and `name` values:

**Standard HTTP/S repository:**
```yaml
helmChart:
  repository: https://prometheus-community.github.io/helm-charts
  name: kube-prometheus-stack
  version: "58.0.0"
```
The operator runs `helm repo add <derived-name> <url>` and then `helm repo update` before installing. The chart reference becomes `<repo-name>/kube-prometheus-stack`.

**OCI registry:**
```yaml
helmChart:
  repository: oci://nvcr.io/nvidia/blueprint
  name: rag-server
  version: "2.3.0"
```
The operator does not call `helm repo add` for OCI URLs (Helm does not support it). The full chart reference becomes `oci://nvcr.io/nvidia/blueprint/rag-server`. Authentication for private OCI registries must be pre-configured in the cluster as a Kubernetes docker registry secret.

**Direct archive URL:**
```yaml
helmChart:
  name: https://example.com/charts/my-chart-1.0.0.tgz
```
When `name` ends in `.tgz` and contains `://`, the operator passes it directly to Helm as the chart reference.

### CRD strategy

Helm charts often bundle Custom Resource Definitions in a `crds/` directory. When you re-deploy a blueprint (e.g., as an upgrade), Helm may refuse to re-install CRDs that already exist in the cluster. The `crdsStrategy` field controls this behaviour.

| Value | Behaviour | When to use |
|---|---|---|
| `Auto` (default) | The operator runs `helm show crds` to inspect the chart, then checks whether any of those CRD names already exist in the cluster. If any do, it passes `--skip-crds` to Helm. If none do, CRDs are installed normally. | Most situations. Safe for both first-time installs and upgrades. |
| `Install` | Helm always installs CRDs. Equivalent to running `helm install` without `--skip-crds`. | First-time cluster bootstrap when you are certain no CRDs from this chart are present. |
| `Skip` | Helm always skips CRDs. Equivalent to `helm install --skip-crds`. | Clusters where CRDs are managed by a separate CRD-only chart, a GitOps tool, or a manual bootstrap step. |

### Inline values

The `values` field is a YAML object that is serialized to a temporary values file and passed to Helm as `--values`. It follows standard Helm values file format and supports nested objects.

```yaml
- name: prometheus
  helmChart:
    repository: https://prometheus-community.github.io/helm-charts
    name: kube-prometheus-stack
    version: "58.0.0"
  values:
    alertmanager:
      enabled: false
    prometheus:
      prometheusSpec:
        scrapeInterval: "15s"
    grafana:
      adminPassword: "initial-password"
      persistence:
        enabled: true
        size: "10Gi"
```

Inline `values` are the base layer. Any values loaded from `valuesFiles` are deep-merged on top, with later entries in the `valuesFiles` list taking precedence over earlier ones. See [Section 10](#10-loading-helm-values-from-configmaps-and-secrets) for the full merge order.

### Inspecting Helm releases created by the operator

```bash
# List all Helm releases in a namespace
helm list -n <namespace>

# Show the full status of a specific release
helm status <release-name> -n <namespace>

# Show the computed values in use for a release
helm get values <release-name> -n <namespace>

# Show the rendered manifests for a release
helm get manifest <release-name> -n <namespace>

# Show the history of a release (all upgrades)
helm history <release-name> -n <namespace>
```

---

## 9. Kubernetes Manifest Components

A `kubernetesManifest` component instructs the operator to apply raw Kubernetes YAML using `kubectl apply`. The manifest content is written as a YAML literal block scalar — a multi-line string — directly inside the component spec.

This is useful for Kubernetes resources that are not packaged as Helm charts: Namespaces, ServiceAccounts, ClusterRoleBindings, ConfigMaps, Secrets, custom operator CRs, or any other resource you need to exist before or after a Helm release.

The operator performs `${VAR}` substitution on the entire manifest string before writing it to a temporary file. After substitution, the operator runs `kubectl apply -f <tmpfile> -n <target-namespace>`. For cluster-scoped resources (Namespaces, ClusterRoles, ClusterRoleBindings, etc.), the `-n` flag has no effect — Kubernetes routes them correctly based on the resource's `kind`.

```yaml
- name: namespace-setup
  kubernetesManifest: |
    apiVersion: v1
    kind: Namespace
    metadata:
      name: ${namespace}
      labels:
        managed-by: weka-app-store
    ---
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: my-app-worker
      namespace: ${namespace}
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: RoleBinding
    metadata:
      name: my-app-worker-binding
      namespace: ${namespace}
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: ClusterRole
      name: view
    subjects:
      - kind: ServiceAccount
        name: my-app-worker
        namespace: ${namespace}
```

Multiple Kubernetes documents within a single manifest string are separated by `---`. All documents in the string are applied in a single `kubectl apply` call.

### Empty manifests

If the manifest string is empty, consists only of whitespace, or contains only YAML comment lines (lines starting with `#`), the component is skipped silently. The operator marks its phase as `Ready` with the message `Skipped: Empty manifest (placeholder component)`. This behaviour makes it possible to add a placeholder component that will be filled in later without breaking the deployment.

---

## 10. Loading Helm Values from ConfigMaps and Secrets

Rather than embedding all Helm values inline in the blueprint spec, you can store values in a Kubernetes ConfigMap or Secret in the cluster and reference them from the component. The operator fetches the named resource, reads the specified key, and merges the content into the component's Helm values.

This approach is useful when values are large (a full NVIDIA NIM configuration runs to hundreds of lines), when values contain sensitive data that should not be stored in a blueprint file, or when the same values need to be shared across multiple blueprint deployments and updated centrally.

```yaml
- name: rag-server
  helmChart:
    repository: oci://nvcr.io/nvidia/blueprint
    name: rag-server
    version: "2.3.0"
  values:
    replicaCount: 1           # inline value — base layer
  valuesFiles:
    - kind: ConfigMap
      name: rag-config
      key: values.yaml
    - kind: Secret
      name: rag-credentials
      key: secrets.yaml
      namespace: credential-store    # read from a different namespace
```

### valuesFiles entry fields

| Field | Type | Required | Description |
|---|---|---|---|
| `kind` | string | Yes | The type of Kubernetes resource to read from. Must be `ConfigMap` or `Secret`. |
| `name` | string | Yes | The `metadata.name` of the ConfigMap or Secret to read. |
| `key` | string | Yes | The key within the ConfigMap's `data` map (or the Secret's `data` map) whose value contains the Helm values YAML. |
| `namespace` | string | No | The namespace to look up the resource in. Defaults to the component's target namespace. |

For a `ConfigMap`, the operator reads `data[key]` as a plain string. For a `Secret`, the operator base64-decodes `data[key]` to get the plain text string. In both cases, the string must be valid YAML when parsed after variable substitution.

### Values merge order

The final values passed to Helm are the result of deep-merging all sources in this order, with each source overriding the previous:

```
1. Helm chart defaults (from the chart itself)
          ↓ overridden by
2. component.values (inline YAML object in the spec)
          ↓ overridden by
3. valuesFiles[0] content (first referenced ConfigMap/Secret)
          ↓ overridden by
4. valuesFiles[1] content
          ↓ overridden by
5. ... (additional valuesFiles entries)
```

Deep merge means that nested objects are merged recursively, not replaced wholesale. If `component.values` sets `grafana.enabled: false` and a `valuesFiles` entry sets `grafana.adminPassword: "abc"`, the result contains both `grafana.enabled: false` and `grafana.adminPassword: "abc"`.

### Variable substitution in valuesFiles content

When the App Store reads a ConfigMap or Secret for `valuesFiles`, it treats the raw text as a template and substitutes any `${VAR}` tokens before parsing the text as YAML. This means you can create a ConfigMap or Secret in your cluster that contains `${VAR}` placeholders, and the App Store will fill them in at deploy time using the values from `appStack.variables`.

For this to work, the variable name inside the ConfigMap must match a key declared in `spec.appStack.variables`. The variables block is where you forward the user's form input (captured with `[[ ]]`) into names the App Store can use when reading the ConfigMap.

**Example: NGC pull secret injected via ConfigMap**

This example shows how to pass the name of an NGC Docker pull secret into a Helm chart's values file without hardcoding it. The user picks an NGC credential from the install form; the blueprint constructs the derived secret name and stores it in `variables`; the ConfigMap references it with `${ngc_docker_secret}`.

```yaml
# ── In the blueprint spec ─────────────────────────────────────────────────────
spec:
  appStack:
    variables:
      # GUI layer resolves [[ ngc_credential ]] to the registration name, e.g. "prod-ngc-key".
      # The result stored in the cluster is: ngc_docker_secret: "warp-prod-ngc-key-docker"
      ngc_docker_secret: "warp-[[ ngc_credential ]]-docker"

    components:
      - name: nim-server
        helmChart:
          repository: oci://nvcr.io/nim/meta
          name: llama3-8b-instruct
          version: "1.0.0"
        valuesFiles:
          - kind: ConfigMap
            name: nim-values      # ConfigMap must already exist in the target namespace
            key: values.yaml

# ── ConfigMap "nim-values", key "values.yaml" (pre-created in the cluster) ────
# imagePullSecrets:
#   - name: "${ngc_docker_secret}"    # App Store substitutes this with "warp-prod-ngc-key-docker"
# model:
#   ngcAPIKey: ""
```

**Example: StorageClass injected via ConfigMap (non-secret)**

Variables are not only for secrets. Any value the user provides can be forwarded through `appStack.variables` and substituted into a ConfigMap. Here the user-supplied StorageClass name flows into a database values file:

```yaml
# ── In the blueprint spec ─────────────────────────────────────────────────────
spec:
  appStack:
    variables:
      storage_class: "[[ storage_class ]]"    # user typed "weka-storageclass-api"

    components:
      - name: database
        helmChart:
          repository: https://charts.bitnami.com/bitnami
          name: postgresql
          version: "15.0.0"
        valuesFiles:
          - kind: ConfigMap
            name: db-values
            key: values.yaml

# ── ConfigMap "db-values", key "values.yaml" (pre-created in the cluster) ─────
# primary:
#   persistence:
#     enabled: true
#     storageClass: "${storage_class}"    # becomes "weka-storageclass-api" at deploy time
#     size: "50Gi"
# readReplicas:
#   persistence:
#     storageClass: "${storage_class}"    # same variable, referenced twice
```

If substitution produces content that is not valid YAML, the App Store raises a permanent error identifying the ConfigMap or Secret and the parse failure.

### Error handling for missing resources

If the referenced ConfigMap or Secret does not exist when the operator tries to read it, the operator raises a **temporary error** and retries after 30 seconds. This means the operator will keep retrying until the resource appears — useful if the ConfigMap or Secret is being created by an earlier component in the same blueprint. Once the resource exists and is readable, reconciliation continues normally.

```bash
# Verify the ConfigMap the operator will read
kubectl get configmap rag-config -n my-namespace -o yaml

# Check that the key exists and contains valid YAML
kubectl get configmap rag-config -n my-namespace \
  -o jsonpath='{.data.values\.yaml}'
```

---

## 11. Component Dependencies and Deployment Order

The operator builds a deployment order for the enabled components before starting any work. It uses topological sort (Kahn's algorithm) to determine which components can be deployed in sequence given the `dependsOn` relationships declared.

Without any `dependsOn` declarations, components are deployed in the order they appear in the `components` list, one at a time, with the operator waiting for each to reach readiness before starting the next.

With `dependsOn`, the order is determined by the dependency graph. A component will not start deploying until all components listed in its `dependsOn` have reached `Ready` status. This guarantees, for example, that a database is fully running before an application server that needs it starts deploying.

```yaml
components:
  - name: redis               # no dependsOn — deploys first
    helmChart: { ... }

  - name: milvus              # no dependsOn — deploys second (after redis is ready)
    helmChart: { ... }

  - name: rag-server          # depends on both — deploys only after both are Ready
    dependsOn:
      - redis
      - milvus
    helmChart: { ... }

  - name: frontend            # depends on rag-server — deploys last
    dependsOn:
      - rag-server
    helmChart: { ... }
```

### Dependency rules and errors

| Condition | Operator behaviour |
|---|---|
| `dependsOn` names a component that does not exist | Permanent error at the start of reconciliation. No components are deployed. |
| Circular dependency (A → B → A) | Permanent error at the start of reconciliation. No components are deployed. |
| A component in `dependsOn` has `enabled: false` | The disabled component is absent from the graph. The dependency reference is silently ignored. |
| A component in `dependsOn` fails during deployment | The dependent components are not started. The overall stack phase is set to `Failed`. |

### Deletion order

When a `WekaAppStore` resource is deleted, the operator uninstalls components in the reverse of the deployment order. Components that were deployed last are removed first. This ensures that dependent applications are removed before the infrastructure they depend on.

```bash
# Watch deletion progress
kubectl get wekaappstore my-blueprint -n my-namespace -w

# After deletion completes, verify Helm releases are gone
helm list -n my-namespace

# Verify manifest-applied resources are gone
kubectl get all -n my-namespace
```

---

## 12. Readiness Checks

After deploying a component, the operator optionally waits for it to report as ready before proceeding to the next component. This wait is implemented using `kubectl wait`, which polls the Kubernetes API until the specified condition is met or the timeout expires.

Setting `waitForReady: false` disables this wait entirely — the operator moves to the next component immediately after `helm install`/`kubectl apply` returns, regardless of whether pods are up or healthy.

When `waitForReady: true` (the default), the operator uses the `readinessCheck` configuration to determine what to wait for. If no `readinessCheck` block is specified, the operator constructs a default selector of `app.kubernetes.io/instance=<releaseName>,app.kubernetes.io/name=<chartName>` and waits for pods matching that selector to reach the `Ready` condition.

### readinessCheck fields

**Common fields — always available:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `type` | string | No | `pod` | The kind of Kubernetes resource to wait on. One of `pod`, `deployment`, `statefulset`, or `job`. |
| `namespace` | string | No | Component's target namespace | The namespace to check resources in. Use this when the resources being waited on are in a different namespace than where the chart was installed. |
| `timeout` | integer | No | `300` | How many seconds to wait before treating the component as failed and stopping the deployment. |
| `gracePeriodSeconds` | integer | No | `5` | How many seconds to wait after Helm returns before starting the readiness check. Gives Kubernetes time to create the initial resources. |

**Targeting — choose exactly one of the following three options:**

> You must tell the App Store *which* running resource to check. Pick one approach — if you set `name`, the other two are ignored; `selector` and `matchLabels` do the same thing in different syntax formats.

| Option | Field | Type | When to use |
|---|---|---|---|
| **Option A — exact name** | `name` | string | You know the exact name of the Deployment, StatefulSet, or Pod created by the chart (e.g., `my-release-redis-master`). This is the most reliable approach when the name is predictable. |
| **Option B — label filter (string)** | `selector` | string | You do not know the exact name but know the labels. Write as a comma-separated string: `app.kubernetes.io/instance=redis,app.kubernetes.io/name=redis`. |
| **Option C — label filter (map)** | `matchLabels` | object | Same as Option B but written as a YAML key-value map instead of a string. Use whichever format you find easier to read. |

If none of the three targeting options is set, the App Store constructs a default selector from the Helm release name and chart name, which works for most standard Helm charts.

### Check type behaviour

| `type` value | `kubectl wait` condition | Typical use case |
|---|---|---|
| `pod` | `condition=ready` | Most workloads. Waits until all matching pods report their containers as running and passing health checks. |
| `deployment` | `condition=available` | Deployments with replicas. Waits until the deployment has the minimum available replicas. |
| `statefulset` | `condition=ready` | StatefulSets (databases, caches). Waits until all pods in the set are ready. |
| `job` | `condition=complete` | One-off tasks. Waits until the job exits successfully. |

### Option A — Targeting by exact name

When you know the exact name of the resource created by the Helm chart, this is the simplest and most reliable approach. Run the commands below after the chart is installed once to discover the name, then put it in the blueprint.

```yaml
readinessCheck:
  type: deployment
  name: my-release-my-chart    # exact Deployment name — use Option B or C if you don't know it
  timeout: 600
```

```bash
# Discover the exact name after a test install
kubectl get deployments -n my-namespace
kubectl get statefulsets -n my-namespace
kubectl get pods -n my-namespace
```

### Option B — Targeting by label filter (string format)

When the chart generates resource names dynamically (e.g. from the Helm release name), you cannot hardcode the exact name. Instead, filter by labels. Most Helm charts apply standard labels that you can rely on:

```yaml
readinessCheck:
  type: deployment
  selector: "app.kubernetes.io/instance=my-release,app.kubernetes.io/name=my-chart"
  timeout: 300
```

### Option C — Targeting by label filter (map format)

Identical to Option B, but written as a YAML map instead of a comma-separated string. Use whichever format is easier to read for your blueprint:

```yaml
readinessCheck:
  type: deployment
  matchLabels:
    app.kubernetes.io/instance: my-release
    app.kubernetes.io/name: my-chart
  timeout: 300
```

```bash
# Find the actual labels applied to a resource (to use in Option B or C)
kubectl get deployments -n my-namespace --show-labels
kubectl get pods -n my-namespace --show-labels
```

### Automatic fallback behaviour

When no resources match the configured selector, the operator attempts two automatic fallbacks before giving up:

1. If the `type` was not `deployment`, the operator retries once using `deployment` with the same selector.
2. If that also finds nothing, the operator tries once more using the simpler selector `app=<componentName>` on pods.

If neither fallback finds matching resources, the operator marks the component as failed and stops the stack deployment.

```bash
# Debug a readiness check that is timing out
kubectl get pods -n my-namespace --show-labels
kubectl get deployments -n my-namespace --show-labels
kubectl get events -n my-namespace --sort-by='.lastTimestamp'
```

---

## 13. Namespace Management

The operator resolves the target namespace for each component by checking five sources in priority order, using the first non-empty value it finds:

| Priority | Source | How to set it |
|---|---|---|
| 1 (highest) | `component.targetNamespace` | Set directly on the component object |
| 2 | `component.namespace` | Alias field on the component; identical to `targetNamespace` |
| 3 | `appStack.namespaces[componentName]` | A map on the stack, keyed by component name |
| 4 | `appStack.defaultNamespace` | A single default for the whole stack |
| 5 (lowest) | Blueprint's own namespace | The namespace the blueprint was installed into (`metadata.namespace` of the installed blueprint resource). |

When the operator determines the target namespace for a component, it checks whether that namespace exists and creates it if it does not.

The `readinessCheck.namespace` field provides a separate override specifically for the readiness check — useful when a Helm chart installs its workloads into a different namespace than the `targetNamespace` where the chart was installed.

### Stack-level namespace fields

```yaml
spec:
  appStack:
    defaultNamespace: ai-stack          # applies to all components unless overridden

    namespaces:                         # per-component overrides by name
      monitoring: monitoring            # component "monitoring" deploys into "monitoring"
      database: database-prod           # component "database" deploys into "database-prod"

    components:
      - name: database
        # resolves to: database-prod (from namespaces map)
        helmChart: { ... }

      - name: app-server
        # resolves to: ai-stack (from defaultNamespace)
        helmChart: { ... }

      - name: monitoring
        # resolves to: monitoring (from namespaces map)
        helmChart: { ... }

      - name: frontend
        targetNamespace: public         # resolves to: public (highest priority)
        helmChart: { ... }
```

---

## 14. Credential Management — API Keys and Secrets

The App Store has a built-in credential management system. Rather than embedding API keys or passwords directly into a blueprint, you register a credential once in the App Store Settings, and blueprints reference it by name. This means a single API key can be shared across many blueprints, and rotating the key only requires updating it in one place.

Credentials are stored as `WarpCredential` entries in the `wekaappstore` namespace. Each entry holds a pointer to a raw Kubernetes Secret that contains the actual key or token. When the App Store processes a credential registration, it reads that raw secret and produces one or more **derived secrets** — pre-formatted versions of the credential ready for direct use by workloads. For example, an NVIDIA NGC API key produces both a plain-text key secret (for mounting as an environment variable) and a Docker pull secret (for pulling NGC container images).

The derived secrets stay in sync automatically. If one is deleted accidentally, the App Store recreates it on the next sync. If the source secret's value changes after rotating a key, updating the credential registration causes all derived secrets to be refreshed.

Credential registrations are `WarpCredential` resources in the Kubernetes API (`warp.io/v1alpha1`). In the App Store UI they appear on the Settings page, and in the install form they appear in credential dropdown fields. The plural name used with `kubectl` is `warpcredentials`.

### Step 1 — Store the raw key in a Kubernetes Secret

Before registering a credential in the App Store, store the actual API key or token in an ordinary Kubernetes Secret in the `wekaappstore` namespace. The App Store reads this secret when deriving the formatted credential secrets.

```bash
# For an NVIDIA NGC key
kubectl create secret generic raw-ngc-key \
  --from-literal=NGC_API_KEY=nvapi-xxxx-yyyy-zzzz \
  -n wekaappstore

# For a HuggingFace token
kubectl create secret generic raw-hf-token \
  --from-literal=HF_API_KEY=hf_xxxxxxxxxxxx \
  -n wekaappstore

# For WEKA storage credentials
kubectl create secret generic raw-weka-creds \
  --from-literal=WEKA_API_USERNAME=admin \
  --from-literal=WEKA_API_TOKEN=my-api-token \
  --from-literal=WEKA_API_ENDPOINT=https://weka-cluster.example.com:14000 \
  -n wekaappstore
```

### Step 2 — Register the credential in the App Store

Once the raw secret exists, create a `WarpCredential` entry to register it. This is what makes the credential appear in the App Store Settings page and in blueprint install form dropdowns.

### Credential registration fields

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | Yes | The credential category. Determines which ready-to-use secrets are produced. Must be one of `nvidia-ngc`, `huggingface`, or `weka-storage`. |
| `displayName` | string | Yes | A human-readable label shown in the Settings page and in blueprint install form dropdowns. Make this descriptive enough for the person installing the blueprint to know which credential to select. |
| `secretRef.name` | string | Yes | The name of the Kubernetes Secret (created in Step 1) that holds the raw API key or token. |
| `secretRef.key` | string | Yes | The key within that secret whose value is the raw credential. For `weka-storage`, set this to `WEKA_API_TOKEN` — the App Store reads all three WEKA keys automatically. |
| `endpoint` | string | No | For `weka-storage` only. The WEKA management API URL (e.g., `https://weka-cluster:14000`). Overrides the endpoint stored in the source secret if both are present. |

### Credential type: `nvidia-ngc` — NVIDIA NGC API key

Registers an NVIDIA NGC API key. The App Store produces two ready-to-use secrets from this one key.

| Derived secret name | What it is | How to use it in a blueprint |
|---|---|---|
| `warp-<name>-apikey` | A plain-text secret containing the `NGC_API_KEY`. | Mount as an environment variable into pods that need the key to download models or access the NGC API. |
| `warp-<name>-docker` | A Docker image pull secret pre-configured for `nvcr.io`. | Reference as an `imagePullSecret` on any Deployment, Pod, or namespace that pulls images from the NGC container registry. |

```yaml
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: prod-ngc-key
  namespace: wekaappstore
spec:
  type: nvidia-ngc
  displayName: "Production NGC API Key"
  secretRef:
    name: raw-ngc-key
    key: NGC_API_KEY
```

### Credential type: `huggingface` — HuggingFace token

Registers a HuggingFace Hub access token for downloading gated models. The App Store produces one ready-to-use secret.

| Derived secret name | What it is | How to use it in a blueprint |
|---|---|---|
| `warp-<name>-token` | A plain-text secret containing the `HF_API_KEY`. | Mount as an environment variable (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`) into model-loading containers. |

```yaml
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: prod-hf-token
  namespace: wekaappstore
spec:
  type: huggingface
  displayName: "Production HuggingFace Token"
  secretRef:
    name: raw-hf-token
    key: HF_API_KEY
```

### Credential type: `weka-storage`

Used for connecting to the WEKA storage cluster REST API. The source Secret must contain exactly three keys: `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, and `WEKA_API_ENDPOINT`. The operator creates one derived Secret containing all three values.

| Derived Secret name | Kubernetes Secret type | Contents | Typical use |
|---|---|---|---|
| `warp-<name>-token` | `Opaque` | Three keys: `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT` (all base64-encoded) | Mount all three as environment variables into workloads that need to authenticate with the WEKA management API. |

```yaml
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: prod-weka-cluster
  namespace: wekaappstore
spec:
  type: weka-storage
  displayName: "Production WEKA Storage Cluster"
  secretRef:
    name: raw-weka-creds
    key: WEKA_API_TOKEN
  endpoint: "https://weka-cluster.example.com:14000"
```

### Credential registration status

After the App Store processes a credential registration, it writes back a status summary. This is what you check when verifying that a credential is ready to use.

| Status field | Type | Description |
|---|---|---|
| `status.conditions` | array | Health conditions. The `KeyReady` condition is present for all types and tells you whether the raw key was read successfully. The `DockerSecretReady` condition is additionally present for `nvidia-ngc` credentials. |
| `status.derivedSecrets` | array | The list of ready-to-use secrets that were created from this registration, with each entry showing the secret name and type. |
| `status.lastSyncTime` | string (ISO 8601) | Timestamp of the most recent successful sync. |
| `status.wekaEndpoint` | string | For `weka-storage` only. The resolved WEKA endpoint URL. |

### Viewing registered credentials

```bash
# List all registered credentials
kubectl get warpcredentials -n wekaappstore

# See full status including the list of derived secrets
kubectl describe warpcredential prod-ngc-key -n wekaappstore

# List the ready-to-use secrets produced from registered credentials
kubectl get secrets -n wekaappstore | grep "^warp-"

# Inspect a derived docker pull secret (shows metadata but not the raw data)
kubectl describe secret warp-prod-ngc-key-docker -n wekaappstore

# Check that a derived API key secret has the expected key name
kubectl get secret warp-prod-ngc-key-apikey -n wekaappstore \
  -o jsonpath='{.data}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"

# Check why a credential failed to sync
kubectl get warpcredential prod-ngc-key -n wekaappstore \
  -o jsonpath='{.status.conditions[0]}'
```

### Using registered credentials in blueprints

In the `x-variables` block, declare a `type: credential` field. When the user installs a blueprint and selects a credential from the dropdown, the value submitted is the registration name (the `metadata.name` of the credential entry — e.g., `prod-ngc-key`). In the `spec`, you use that name to construct the name of the derived secret you want to reference.

The name of each derived (ready-to-use) secret follows a fixed pattern based on the registration name. If the credential was registered as `prod-ngc-key`, the derived secrets will be named:

- `nvidia-ngc` API key: `warp-prod-ngc-key-apikey`
- `nvidia-ngc` Docker pull secret: `warp-prod-ngc-key-docker`
- `huggingface` token: `warp-prod-ngc-key-token`
- `weka-storage` token: `warp-prod-ngc-key-token`

In general, the pattern is:

- `nvidia-ngc` API key secret: `warp-<registrationName>-apikey`
- `nvidia-ngc` docker pull secret: `warp-<registrationName>-docker`
- `huggingface` token secret: `warp-<registrationName>-token`
- `weka-storage` token secret: `warp-<registrationName>-token`

```yaml
x-variables:
  ngc_credential:
    type: credential
    credential_type: nvidia-ngc
    required: true

spec:
  appStack:
    variables:
      # Construct the derived secret names from the credential name
      ngc_docker_secret: "warp-[[ ngc_credential ]]-docker"
      ngc_apikey_secret: "warp-[[ ngc_credential ]]-apikey"
    components:
      - name: nim-server
        helmChart:
          repository: oci://nvcr.io/nim/meta
          name: llama3-8b-instruct
          version: "1.0.0"
        values:
          imagePullSecrets:
            - name: "${ngc_docker_secret}"    # passed as Helm value
          nim:
            ngcAPIKey: ""                     # key is in the secret, not inline
          envFrom:
            - secretRef:
                name: "${ngc_apikey_secret}"  # mount the key as an env var
```

### Derived secrets persist when a credential registration is removed

Removing a credential registration from the App Store does not delete the derived secrets it produced. This is intentional — any blueprints that were already installed and are actively using those secrets continue to function normally. To fully clean up the derived secrets after removing a credential registration, delete them manually:

```bash
kubectl delete secret warp-prod-ngc-key-apikey warp-prod-ngc-key-docker -n wekaappstore
```

---

## 15. Two-Layer Variable Substitution Reference

There are two independent variable substitution passes in the App Store system. They run at different points in time, use different syntax, and substitute into different parts of the document. Understanding the boundary between them is essential for authoring correct blueprints.

### Comparison of the two layers

| | GUI layer | App Store layer |
|---|---|---|
| **When** | When the user clicks Install in the browser | When the App Store deploys each component in the cluster |
| **Performed by** | The GUI web server (Jinja2 template engine) | The App Store operator (Python `string.Template`) |
| **Input document** | The raw blueprint YAML file from disk | The installed blueprint stored in the cluster |
| **Syntax** | `[[ variable_name ]]` | `${variable_name}` |
| **Variable values come from** | The install form submitted by the user | `spec.appStack.variables` in the installed blueprint |
| **Substitution scope** | Anywhere in the YAML document — including `metadata`, field names, values, and nested structures | Only: `kubernetesManifest` strings, and the raw text content of `valuesFiles` references |
| **Inline `values` objects** | Yes — `[[ ]]` tokens in `values` objects are resolved by Jinja2 | No — inline `values` are YAML objects, not strings; `${ }` tokens in `values` objects are passed verbatim to Helm |
| **Undefined variable behaviour** | Jinja2 renders an empty string | Permanent error — deployment stops and does not retry |
| **After resolution** | No `[[ ]]` tokens exist in the submitted document | No `${ }` tokens remain in the deployed manifests and values |

### When to use each layer

**Use `[[ variable_name ]]` when the value needs to be baked into the blueprint before it is saved.**

The `[[ ]]` syntax is resolved the moment the user clicks Install. Use it for anything that must be fixed at that instant:

- **Setting the deployment namespace.** The namespace field in `metadata` must be known before the blueprint is saved. Write `namespace: [[ namespace ]]` so the namespace the user chose is embedded in the blueprint.
- **Building an API key secret name.** The App Store names derived secrets using the registration name the user selects (e.g., `warp-prod-ngc-key-docker`). Because you don't know the registration name until the user picks it, you build it at install time: `"warp-[[ ngc_credential ]]-docker"`.
- **Anything that goes inside an inline `values:` block.** The `values:` block is a fixed YAML structure, not plain text — the `${ }` system does not process it. If you need a user-supplied value inside `values:`, write it directly with `[[ ]]`.
- **Any setting that needs to be in the blueprint's own metadata.** Resource names, labels, and annotations must be fully resolved before the blueprint is stored.

**Use `${variable_name}` when the value should be filled in during each deployment.**

The `${ }` syntax is resolved later — at the point when the App Store is actually deploying a component. Use it for values that are injected into the component's deployment settings:

- **Inside a Kubernetes manifest.** If a component creates raw Kubernetes resources (a Namespace, a ConfigMap, a Deployment), write `${variable_name}` inside that manifest text and the App Store will fill it in before applying the manifest to the cluster.
- **Inside a values file stored in a ConfigMap or Secret.** If you have a ConfigMap in your cluster that holds Helm chart settings, you can write `${variable_name}` tokens in that ConfigMap and the App Store will replace them when it reads the ConfigMap during deployment. This is useful for large values files that would be unwieldy to write inline in the blueprint.
- **Anywhere you want to reuse a value across multiple components.** Declare it once in `appStack.variables` and reference it with `${ }` in as many components as you need.

### Common pattern: forwarding a user value through both layers

The most common pattern is to capture a user's form input with `[[ ]]`, store it in `spec.appStack.variables`, and then consume it in manifests or values files with `${ }`:

```yaml
x-variables:
  storage_class:
    type: string
    required: true

spec:
  appStack:
    variables:
      storage_class: "[[ storage_class ]]"     # GUI layer stores user input into the CR
    components:
      - name: database
        helmChart:
          repository: https://charts.bitnami.com/bitnami
          name: postgresql
          version: "15.0.0"
        valuesFiles:
          - kind: ConfigMap
            name: db-values
            key: values.yaml
            # ConfigMap content: "primary:\n  persistence:\n    storageClass: \"${storage_class}\""
            # Operator layer substitutes ${storage_class} when reading the ConfigMap content
```

### Common pattern: credential name to derived secret name

Credential form fields return the WarpCredential resource name. The blueprint author constructs the derived Secret name in the GUI layer, then uses it in the operator layer:

```yaml
x-variables:
  ngc_credential:
    type: credential
    credential_type: nvidia-ngc

spec:
  appStack:
    variables:
      # [[ ngc_credential ]] is resolved by GUI to e.g. "prod-ngc-key"
      # resulting in: ngc_docker_secret: "warp-prod-ngc-key-docker"
      ngc_docker_secret: "warp-[[ ngc_credential ]]-docker"
    components:
      - name: workload
        kubernetesManifest: |
          apiVersion: apps/v1
          kind: Deployment
          spec:
            template:
              spec:
                imagePullSecrets:
                  - name: ${ngc_docker_secret}    # operator resolves this
```

### Values objects do not support operator-layer substitution

Inline `values` objects in the component spec are passed directly to Helm as YAML data. The operator does not scan them for `${ }` tokens. If you write `storageClass: "${storage_class}"` in an inline `values` block, Helm receives the literal string `${storage_class}`, not the resolved value.

To use a variable value in a Helm chart, either:
1. Use the `[[ ]]` GUI layer directly in the inline `values` block (because the GUI layer renders the whole document including `values` objects), or
2. Move the setting into a `valuesFiles` ConfigMap where operator-layer substitution applies.

```yaml
# Option 1 — GUI layer in inline values (resolves before Kubernetes sees the document)
values:
  storageClass: "[[ storage_class ]]"

# Option 2 — Operator layer via valuesFiles (resolves at deploy time in the cluster)
valuesFiles:
  - kind: ConfigMap
    name: my-values
    key: values.yaml
    # ConfigMap content: "storageClass: \"${storage_class}\""
```

---

## 16. Inspecting Resources with kubectl

This section collects the most useful kubectl commands for observing operator-managed resources.

### WekaAppStore resources

```bash
# List all WekaAppStore resources across all namespaces
kubectl get wekaappstores -A

# List resources in a specific namespace
kubectl get wekaappstores -n my-namespace

# Get full YAML including status
kubectl get wekaappstore my-blueprint -n my-namespace -o yaml

# Watch status changes in real time during deployment
kubectl get wekaappstore my-blueprint -n my-namespace -w

# Get the overall appStackPhase
kubectl get wekaappstore my-blueprint -n my-namespace \
  -o jsonpath='{.status.appStackPhase}'

# Get per-component status as a table
kubectl get wekaappstore my-blueprint -n my-namespace \
  -o jsonpath='{range .status.componentStatus[*]}{.name}{"\t"}{.phase}{"\t"}{.message}{"\n"}{end}'

# Get the Ready condition message
kubectl get wekaappstore my-blueprint -n my-namespace \
  -o jsonpath='{.status.conditions[?(@.type=="Ready")].message}'

# Stream operator logs while a deployment runs
kubectl logs -n wekaappstore -l app=weka-app-store-operator -f
```

### WarpCredential resources

```bash
# List all WarpCredential resources
kubectl get warpcredentials -n wekaappstore

# Get full YAML including derived secret list
kubectl get warpcredential prod-ngc-key -n wekaappstore -o yaml

# Check credential status
kubectl describe warpcredential prod-ngc-key -n wekaappstore

# Check the KeyReady condition
kubectl get warpcredential prod-ngc-key -n wekaappstore \
  -o jsonpath='{.status.conditions[?(@.type=="KeyReady")].status}'

# List the derived secrets for a credential
kubectl get warpcredential prod-ngc-key -n wekaappstore \
  -o jsonpath='{range .status.derivedSecrets[*]}{.name}{"\t"}{.type}{"\n"}{end}'

# See when the credential was last synced
kubectl get warpcredential prod-ngc-key -n wekaappstore \
  -o jsonpath='{.status.lastSyncTime}'
```

### Derived Secrets

```bash
# List all derived secrets in the wekaappstore namespace
kubectl get secrets -n wekaappstore | grep "^warp-"

# Describe a docker pull secret (shows type and metadata, not the encoded data)
kubectl describe secret warp-prod-ngc-key-docker -n wekaappstore

# Verify the keys in an Opaque secret
kubectl get secret warp-prod-ngc-key-apikey -n wekaappstore \
  -o jsonpath='{.data}' | python3 -m json.tool | grep -o '"[^"]*":' | tr -d '":"'

# Verify the dockerconfigjson secret is well-formed (shows registry and username)
kubectl get secret warp-prod-ngc-key-docker -n wekaappstore \
  -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d | python3 -m json.tool
```

### Helm releases created by the operator

```bash
# List all Helm releases in a namespace
helm list -n my-namespace

# See the status of a specific release
helm status my-release -n my-namespace

# See the computed values Helm is using
helm get values my-release -n my-namespace

# See the full rendered manifests
helm get manifest my-release -n my-namespace

# Roll back a release to a previous revision
helm rollback my-release <revision> -n my-namespace

# See upgrade history
helm history my-release -n my-namespace
```

### Operator pod

```bash
# Check the operator pod is running
kubectl get pods -n wekaappstore

# Stream operator logs
kubectl logs -n wekaappstore deployment/weka-app-store-operator -f

# Search operator logs for errors related to a blueprint
kubectl logs -n wekaappstore deployment/weka-app-store-operator \
  | grep -i "my-blueprint\|PermanentError\|TemporaryError"
```

---

## 17. Lifecycle — Create, Update, Delete

### Create

When a `WekaAppStore` resource is first applied to the cluster, the operator's create handler runs. It executes the following steps in order:

1. **Variable validation.** All keys in `spec.appStack.variables` are checked against the naming pattern `[_a-zA-Z][_a-zA-Z0-9]*`. All values are checked to be strings. Any violation raises a permanent error immediately — no deployment work begins.

2. **Dependency resolution.** The operator builds a directed acyclic graph from the `dependsOn` declarations and topologically sorts the enabled components. Circular dependencies or references to non-existent components raise a permanent error.

3. **Status initialisation.** The operator sets `status.appStackPhase` to `Installing` and writes an initial `conditions` entry.

4. **Component deployment loop.** For each component in resolved order: deploy (via Helm or kubectl), optionally wait for readiness, then update `status.componentStatus` for that component. If any component fails, the loop stops and the overall phase is set to `Failed`.

5. **Final status.** If all components succeed, `status.appStackPhase` is set to `Ready`.

### Update

When the `spec` of an existing `WekaAppStore` resource changes, the operator's update handler runs. It re-executes the entire deployment loop from step 1. Each Helm component runs `helm upgrade` (not re-install). Each manifest component runs `kubectl apply` again, which is idempotent. Components that did not change are still processed — there is no diff-based optimisation.

### Delete

When a `WekaAppStore` resource is deleted, the operator's delete handler runs. It resolves the dependency graph and iterates through components in **reverse deployment order** — the last-deployed component is removed first. For Helm components it runs `helm uninstall`. For manifest components it runs `kubectl delete -f <manifest> --ignore-not-found`.

If a Helm release or manifest resource has already been removed from the cluster by other means, the `--ignore-not-found` flag and the equivalent Helm behaviour ensure that the deletion handler completes without errors.

### Operator restart

When the App Store pod restarts (rolling update, node eviction, manual restart), it re-processes every existing installed blueprint and every registered credential. For credentials this may re-create derived secrets that were deleted while the App Store was down. For installed blueprints, the App Store does not automatically re-run a full deployment on restart — it only processes blueprints that change after the restart.

---

## 18. Complete Worked Example

This example deploys a four-component AI RAG (Retrieval-Augmented Generation) stack: a Redis cache, a Milvus vector database, a NIM large language model server, and a RAG application. It demonstrates credential integration, component dependencies, custom readiness checks, variable forwarding, and valuesFiles.

### Prerequisites

The following Kubernetes Secrets must exist in the `wekaappstore` namespace before the WarpCredential resources are created:

```bash
# NVIDIA NGC key
kubectl create secret generic raw-ngc-key \
  --from-literal=NGC_API_KEY=nvapi-REPLACE-WITH-REAL-KEY \
  -n wekaappstore

# HuggingFace token (optional — used if any gated models are needed)
kubectl create secret generic raw-hf-token \
  --from-literal=HF_API_KEY=hf_REPLACE-WITH-REAL-TOKEN \
  -n wekaappstore
```

Then create the WarpCredential resources via the App Store Settings page or with `kubectl apply`:

```yaml
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: prod-ngc-key
  namespace: wekaappstore
spec:
  type: nvidia-ngc
  displayName: "Production NVIDIA NGC Key"
  secretRef:
    name: raw-ngc-key
    key: NGC_API_KEY
---
apiVersion: warp.io/v1alpha1
kind: WarpCredential
metadata:
  name: prod-hf-token
  namespace: wekaappstore
spec:
  type: huggingface
  displayName: "Production HuggingFace Token"
  secretRef:
    name: raw-hf-token
    key: HF_API_KEY
```

```bash
# Verify derived secrets were created
kubectl get secrets -n wekaappstore | grep "^warp-prod-"
# Expected output:
# warp-prod-ngc-key-apikey   Opaque                      1      ...
# warp-prod-ngc-key-docker   kubernetes.io/dockerconfigjson   1      ...
# warp-prod-hf-token-token   Opaque                      1      ...
```

### The blueprint file

Save this as `rag-stack/rag-stack.yaml` in the `warp-blueprints` repository:

```yaml
# ── Install form schema ───────────────────────────────────────────────────────
x-variables:
  namespace:
    type: string
    required: true
    description: "Kubernetes namespace to deploy the RAG stack into. It will be created if it does not exist."
    placeholder: "e.g. rag-production"
  storage_class:
    type: string
    required: true
    description: "StorageClass for persistent volumes (Redis, Milvus, ingestor data). Use a WEKA StorageClass for best performance."
    placeholder: "e.g. weka-storageclass-api"
  ngc_credential:
    type: credential
    credential_type: nvidia-ngc
    required: true
    description: "NVIDIA NGC credential used to pull NIM container images and provide the NGC API key to model-serving containers."
  hf_credential:
    type: credential
    credential_type: huggingface
    required: false
    description: "HuggingFace credential for downloading gated models. Leave blank if no gated models are required."

# ── Kubernetes resource ───────────────────────────────────────────────────────
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: rag-stack
  namespace: [[ namespace ]]
spec:
  appStack:
    defaultNamespace: [[ namespace ]]

    # Operator-layer variables. All GUI-layer values are forwarded here
    # so the operator can inject them into kubernetesManifest strings
    # and valuesFiles content using ${VAR} syntax.
    variables:
      storage_class: "[[ storage_class ]]"
      ngc_docker_secret: "warp-[[ ngc_credential ]]-docker"
      ngc_apikey_secret: "warp-[[ ngc_credential ]]-apikey"
      hf_token_secret: "warp-[[ hf_credential ]]-token"

    components:

      # ── 1. Redis cache ────────────────────────────────────────────────────
      # No dependencies. Deploys first. Used by the RAG server for
      # session caching and the NV-Ingest message queue.
      - name: redis
        helmChart:
          repository: https://charts.bitnami.com/bitnami
          name: redis
          version: "19.6.4"
          releaseName: rag-redis
          crdsStrategy: Auto
        values:
          auth:
            enabled: false
          master:
            persistence:
              enabled: true
              storageClass: "[[ storage_class ]]"   # GUI layer — works in inline values
              size: "8Gi"
          replica:
            replicaCount: 0                          # standalone mode, no replicas
        waitForReady: true
        readinessCheck:
          type: statefulset
          name: rag-redis-master
          timeout: 300
          gracePeriodSeconds: 10

      # ── 2. Milvus vector database ─────────────────────────────────────────
      # No dependencies. Deploys after Redis is ready.
      - name: milvus
        helmChart:
          repository: https://zilliztech.github.io/milvus-helm
          name: milvus
          version: "4.2.12"
          releaseName: milvus
          crdsStrategy: Auto
        values:
          cluster:
            enabled: false
          standalone:
            persistence:
              persistentVolumeClaim:
                storageClass: "[[ storage_class ]]"
                size: "100Gi"
          etcd:
            replicaCount: 1
          minio:
            mode: standalone
            persistence:
              storageClass: "[[ storage_class ]]"
              size: "50Gi"
        waitForReady: true
        readinessCheck:
          type: deployment
          selector: "app.kubernetes.io/instance=milvus,app.kubernetes.io/name=milvus"
          timeout: 600
          gracePeriodSeconds: 15

      # ── 3. NIM LLM model server ───────────────────────────────────────────
      # Depends on nothing (can start while Milvus initialises).
      # Pulling and loading large model weights takes several minutes.
      - name: nim-llm
        helmChart:
          repository: oci://nvcr.io/nim/meta
          name: llama3-8b-instruct
          version: "1.0.0"
          releaseName: nim-llm
          crdsStrategy: Skip
        values:
          imagePullSecrets:
            - name: "[[ ngc_credential ]]"    # GUI layer constructs the name inline
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              nvidia.com/gpu: 1
        valuesFiles:
          - kind: ConfigMap
            name: nim-llm-values
            key: values.yaml
            # ConfigMap "nim-llm-values" must exist in the target namespace
            # with key "values.yaml" containing operator-layer ${VAR} tokens:
            #   imagePullSecrets:
            #     - name: "${ngc_docker_secret}"
            #   model:
            #     ngcAPIKey: ""
            #     ngcAPIKeySecret:
            #       name: "${ngc_apikey_secret}"
            #       key: NGC_API_KEY
        waitForReady: true
        readinessCheck:
          type: deployment
          name: nim-llm
          timeout: 1800      # model weight loading can take 20+ minutes

      # ── 4. RAG application server ─────────────────────────────────────────
      # Depends on both Redis, Milvus, and nim-llm.
      # Deploys only after all three dependencies are Ready.
      - name: rag-server
        dependsOn:
          - redis
          - milvus
          - nim-llm
        helmChart:
          repository: oci://nvcr.io/nvidia/blueprint
          name: rag-server
          version: "2.3.0"
          releaseName: rag-server
          crdsStrategy: Auto
        values:
          imagePullSecrets:
            - name: "[[ ngc_credential ]]"
          service:
            type: NodePort
        valuesFiles:
          - kind: ConfigMap
            name: rag-server-values
            key: values.yaml
            # This ConfigMap contains the full RAG server configuration with
            # ${ngc_docker_secret} and ${storage_class} tokens that the operator substitutes.
        waitForReady: true
        readinessCheck:
          type: deployment
          name: rag-server
          timeout: 600
```

### Deploying and monitoring

```bash
# After the blueprint appears in the GUI and the user fills the form, the operator
# creates the WekaAppStore resource. Monitor deployment progress:

# Watch the overall phase
kubectl get wekaappstore rag-stack -n rag-production -w

# Watch individual pod startup
kubectl get pods -n rag-production -w

# Stream operator logs
kubectl logs -n wekaappstore deployment/weka-app-store-operator -f

# After deployment, check all components reached Ready
kubectl get wekaappstore rag-stack -n rag-production \
  -o jsonpath='{range .status.componentStatus[*]}{.name}{"\t"}{.phase}{"\n"}{end}'

# Check all Helm releases
helm list -n rag-production

# Verify the NIM LLM service is reachable
kubectl get svc -n rag-production
```

---

## 19. Troubleshooting Reference

### Deployment fails immediately without deploying any components

The operator validates variables and resolves dependencies before touching the cluster. A failure here means the spec itself is invalid.

```bash
kubectl describe wekaappstore my-blueprint -n my-namespace | grep -A 10 "Conditions:"
```

| Error message | Cause | Fix |
|---|---|---|
| `Invalid variable key '<name>': must match Python identifier syntax` | Variable name contains a hyphen, dot, or starts with a digit | Rename: use underscores instead of hyphens, e.g., `storage_class` not `storage-class` |
| `Invalid variable value for '<name>': must be a string` | A variable value is an unquoted number or boolean | Wrap the value in quotes: `"42"` not `42`, `"true"` not `true` |
| `Dependency resolution failed: Component 'x' depends on unknown component 'y'` | `dependsOn` references a name that does not exist in the components list | Check spelling — component names are case-sensitive |
| `Circular dependency detected` | Component A depends on B, and B depends on A (directly or through a chain) | Remove the circular reference from `dependsOn` |
| `appStack.components is required and cannot be empty` | The `components` list is empty or absent | Add at least one component |

### A component is marked Failed after Helm ran successfully

The Helm install succeeded but the readiness check timed out. The component is deployed in the cluster but not yet healthy.

```bash
# Check pod status in the target namespace
kubectl get pods -n my-namespace

# Check pod logs for startup errors
kubectl logs -n my-namespace deployment/my-release-my-chart

# Check events for scheduling or image pull problems
kubectl get events -n my-namespace --sort-by='.lastTimestamp'

# Check what the readiness check is actually waiting on
kubectl get deployments -n my-namespace --show-labels
kubectl get pods -n my-namespace --show-labels
```

Common causes:
- The `timeout` in `readinessCheck` is too short for the image to pull and start. Increase it.
- The `name` in `readinessCheck` does not match the actual Deployment name created by Helm. Run `kubectl get deployments -n my-namespace` to find the correct name.
- The label selector in `readinessCheck.selector` does not match any resources. Run `kubectl get pods -n my-namespace --show-labels` to see actual labels.
- The pod is failing due to an image pull error (wrong secret name, credential not synced). Check pod events.

### WarpCredential stuck with condition `KeyReady: False`

```bash
kubectl describe warpcredential my-cred -n wekaappstore
```

| Condition reason | Meaning | Fix |
|---|---|---|
| `KeyMissing` | The source Secret named in `spec.secretRef.name` does not exist, or does not contain the key named in `spec.secretRef.key` | Verify the Secret exists: `kubectl get secret <name> -n wekaappstore`. Verify the key: `kubectl get secret <name> -n wekaappstore -o jsonpath='{.data}'` |
| `EmptyKey` | The source Secret exists and has the key, but the value is empty or whitespace | Update the source Secret with a valid non-empty credential value |
| `UnknownType` | `spec.type` is not one of `nvidia-ngc`, `huggingface`, `weka-storage` | Correct the `type` field |
| `InvalidSpec` | `spec.secretRef.name` or `spec.secretRef.key` is missing | Add the missing field |
| `SecretWriteError` | The operator failed to write the derived Secret to the API server | Check operator pod logs for the detailed error |

### Derived secrets missing after credential is Ready

Derived secrets persist in the `wekaappstore` namespace. If they were deleted manually, trigger a re-sync to recreate them:

```bash
# Trigger re-reconcile by adding a harmless annotation
kubectl annotate warpcredential my-cred -n wekaappstore \
  warp.io/force-reconcile="$(date +%s)" --overwrite

# Wait a few seconds, then verify
kubectl get secrets -n wekaappstore | grep "^warp-my-cred"
```

### Blueprint not appearing in the GUI

The GUI discovers blueprints by walking the `BLUEPRINTS_DIR` directory tree and looking for YAML files with a non-empty `x-variables` block at the top level.

```bash
# On the GUI pod, check what BLUEPRINTS_DIR is set to
kubectl exec -n wekaappstore deployment/weka-app-store-gui -- \
  python3 -c "from webapp.main import BLUEPRINTS_DIR; print(BLUEPRINTS_DIR)"

# Verify your blueprint file is present
kubectl exec -n wekaappstore deployment/weka-app-store-gui -- \
  find /app/manifests -name "*.yaml" | head -20

# Verify the x-variables block is present and parseable
kubectl exec -n wekaappstore deployment/weka-app-store-gui -- \
  python3 -c "
import yaml, sys
with open('/app/manifests/my-blueprint/my-blueprint.yaml') as f:
    d = yaml.safe_load(f)
print('x-variables:', list((d.get('x-variables') or {}).keys()))
"
```

Common causes:
- The `x-variables` key is absent or set to `null` / `{}`.
- The file name stem or parent directory name does not match the expected URL slug.
- The `git-sync` sidecar has not pulled the latest commit. Check its logs: `kubectl logs -n wekaappstore -c git-sync deployment/weka-app-store-gui`.

### Helm upgrade fails with "cannot re-use a name that is still in use"

A Helm release is stuck in a failed or pending state. Clean it up:

```bash
# Check the state of the release
helm list -n my-namespace -a     # -a shows failed/pending releases

# Roll back to the last good revision
helm rollback my-release -n my-namespace

# If rollback also fails, force-uninstall and let the operator re-install
helm uninstall my-release -n my-namespace

# Then touch the WekaAppStore resource to trigger re-reconciliation
kubectl annotate wekaappstore my-blueprint -n my-namespace \
  warp.io/force-reconcile="$(date +%s)" --overwrite
```

### ${VAR} token appears literally in a deployed resource

This means the token was in an inline `values` block rather than in a `kubernetesManifest` string or a `valuesFiles` content string. The operator only resolves `${ }` in those two specific contexts.

Move the setting to a ConfigMap referenced by `valuesFiles`, or use the GUI-layer `[[ ]]` syntax directly in the inline `values` block instead.
