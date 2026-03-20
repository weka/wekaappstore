# WEKA App Store MCP Server — Agent Workflow

## Overview

The WEKA App Store MCP server enables an AI agent to inspect a Kubernetes cluster, discover
available blueprints, generate a valid WekaAppStore manifest, validate it, and safely apply
it to the cluster. The server exposes 8 tools that form a structured workflow. Agents must
follow this workflow in order — skipping or reordering steps leads to unsafe or incorrect
deployments.

All tool calls are bounded: they read cluster state or apply a single validated manifest.
No tool modifies the cluster except `apply`, and `apply` requires explicit user confirmation.

---

## Workflow

The numbered steps below define the required execution order. Do not skip steps or
reorder them without explicit user instruction.

**Step 1 — `inspect_cluster`**
Call `inspect_cluster` FIRST. This returns available CPU, memory, GPU, namespaces, and
storage classes. You need this to understand whether the cluster can host the intended
blueprint. Record `cpu_cores_free`, `memory_gib_free`, `gpu_total`, and `storage_classes`
for use in blueprint selection.

**Step 2 — `inspect_weka`**
Call `inspect_weka` to check WEKA storage availability. If the blueprint requires WEKA
filesystems (storage class `wekafs`), verify `free_capacity_gib` is sufficient. This step
is required whenever the blueprint uses WEKA-backed persistent storage.

**Step 3 — `list_blueprints`**
Call `list_blueprints` after inspecting the cluster. This returns a catalog of available
WekaAppStore blueprints with component names and counts. Use the resource data from
Step 1 to mentally filter blueprints that fit the cluster. Use the blueprint `name` field
to proceed to Step 4.

**Step 4 — `get_blueprint`**
Call `get_blueprint` with the name from Step 3. This returns full blueprint details:
all components, Helm chart references, target namespaces, and prerequisites. Read the
prerequisites carefully — some blueprints require existing cluster infrastructure. Record
the component list for the user summary in Step 10.

**Step 5 — `get_crd_schema`**
Call `get_crd_schema` BEFORE generating YAML. This returns the WekaAppStore CRD OpenAPI
v3 schema and 1-2 example manifests. Study the schema to understand valid fields,
required fields, and correct camelCase naming. Refer to the examples for structural patterns.
Do NOT generate YAML without reading the schema first.

**Step 6 — Generate YAML**
Using the CRD schema from Step 5 and the blueprint details from Step 4, generate a
WekaAppStore manifest. Requirements:
- `apiVersion: warp.io/v1alpha1`
- `kind: WekaAppStore`
- `metadata.name` must be set
- Use camelCase field names (e.g., `helmChart`, `appStack`) — never snake_case
- Include at least one deployment method: `helmChart`, `appStack`, or `image`
- Do NOT include v1.0 planning model fields (see Negative Examples below)

**Step 7 — `validate_yaml`**
Call `validate_yaml` with the generated YAML before doing anything else. Do not call
`apply` if `validate_yaml` returns `valid=false`. Fix errors and retry (see
Validate-Retry Loop section).

**Step 8 — Validate-Retry Loop**
If `validate_yaml` returns `valid=false`, read each error's `code`, `path`, and `message`.
Fix the YAML and call `validate_yaml` again. Maximum 3 attempts. If still failing after
3 attempts, stop, present the full YAML and all errors to the user, and ask for help.
Do NOT call `apply` while errors remain.

**Step 9 — Re-run `inspect_cluster` (MANDATORY before apply)**
After `validate_yaml` returns `valid=true` and before calling `apply`, ALWAYS re-run
`inspect_cluster`. Cluster state may have changed since Step 1 — nodes may have become
unschedulable, namespace quotas may have changed, or required storage classes may have
been removed. If resources are now insufficient, inform the user and do not proceed.

**Step 10 — Present plan to user and wait for explicit approval**
Show the user:
- The validated YAML (full text)
- The resource `metadata.name` and target namespace
- The deployment method (helmChart / appStack / image)
- The list of components that will be created
- A summary of the cluster resources that will be consumed

Wait for the user to explicitly approve. Do not infer approval from prior messages.
Do not call `apply` until the user says to proceed.

**Step 11 — `apply`**
Call `apply` with `confirmed=true` ONLY after the user has explicitly approved in Step 10.
Pass the validated YAML text and target namespace. `confirmed` must be a boolean `true` —
string `"true"` or integer `1` are rejected. Any other value returns
`error: approval_required` without creating any K8s resources.

**Step 12 — `status`**
Call `status` after a successful `apply` to monitor deployment progress. Pass the resource
name and namespace from the manifest. If `appStackPhase` is null or missing, the operator
may not have reconciled yet — wait 10-30 seconds and call `status` again. Repeat until
`appStackPhase` is `Ready` or `Failed`. Report final status to the user.

---

## Validate-Retry Loop

When `validate_yaml` returns `valid=false`:

1. Read each error in the `errors` list. Each error has:
   - `code`: machine-readable error type (e.g., `v1_only_field`, `invalid_api_version`)
   - `path`: YAML path where the error was found (e.g., `doc[0].spec.blueprint_family`)
   - `message`: human-readable description of what is wrong

2. Fix the specific field or value indicated by `path` and `code`.

3. Call `validate_yaml` again with the corrected YAML.

4. Repeat up to 3 total attempts.

5. If still `valid=false` after 3 attempts: STOP. Present the full YAML, all errors, and
   the attempted fixes to the user. Ask for guidance. Do NOT call `apply`.

Common fixes:
- `invalid_api_version`: Change `apiVersion` to `warp.io/v1alpha1`
- `missing_required_field` on `metadata.name`: Add a descriptive name
- `v1_only_field`: Remove the snake_case field entirely (see Negative Examples)
- `missing_deployment_method`: Add one of `helmChart`, `appStack`, or `image` under `spec`

---

## Re-Inspect Before Apply

**MANDATORY RULE:** After `validate_yaml` returns `valid=true` and before calling `apply`,
always re-run `inspect_cluster`.

Why: Cluster state may have changed between your initial inspection and the time you are
ready to apply. Another workload may have claimed resources, a node may have gone unready,
or a storage class may have been removed. Applying without re-inspecting risks deploying
to a cluster that can no longer host the blueprint successfully.

If re-inspection reveals insufficient resources:
1. Do NOT call `apply`
2. Inform the user of the specific resource shortage
3. Ask whether to wait and retry, resize the blueprint, or abandon the deployment

---

## Negative Examples

### v1.0 Field Mistakes

These snake_case fields are from the v1.0 planning model and are NEVER valid in a
WekaAppStore spec. `validate_yaml` will return `v1_only_field` errors if they are present.

```yaml
# WRONG — v1.0 fields rejected by validator
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  blueprint_family: ai-inference       # WRONG: v1.0 field
  fit_findings: {matched: true}        # WRONG: v1.0 field
  namespace_strategy: shared           # WRONG: v1.0 field
  reasoning_summary: "GPU workload"    # WRONG: v1.0 field
  request_summary: "Deploy NIM"        # WRONG: v1.0 field
  unresolved_questions: []             # WRONG: v1.0 field
```

Use the camelCase CRD fields instead:

```yaml
# CORRECT
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
  namespace: default
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-app
    version: 1.0.0
    releaseName: my-app-release
```

### Wrong apiVersion

```yaml
# WRONG
apiVersion: apps/v1          # WRONG: must be warp.io/v1alpha1
apiVersion: warp.io/v1       # WRONG: must include alpha1 suffix
apiVersion: v1alpha1         # WRONG: missing group prefix

# CORRECT
apiVersion: warp.io/v1alpha1
```

### Missing metadata.name

```yaml
# WRONG — name is required
metadata:
  namespace: default
  # no name!

# CORRECT
metadata:
  name: my-deployment
  namespace: default
```

### Skipping validate_yaml Before Apply

```
# WRONG sequence:
# get_blueprint -> generate YAML -> apply   (no validate_yaml step!)

# CORRECT sequence:
# get_blueprint -> get_crd_schema -> generate YAML -> validate_yaml -> apply
```

### Applying Without Inspecting Cluster First

```
# WRONG: call apply without ever calling inspect_cluster
# CORRECT: inspect_cluster -> ... -> inspect_cluster (again before apply) -> apply
```

### Setting confirmed=true Without User Approval

```python
# WRONG: agent decides on its own that it's OK to apply
apply(yaml_text=..., namespace=..., confirmed=True)  # before user approved!

# CORRECT: show user the plan, wait for explicit approval, then set confirmed=True
```

---

## Tool Reference

| Tool | When to Call | Returns |
|------|--------------|---------|
| `inspect_cluster` | Step 1 (FIRST) and Step 9 (before apply) | CPU, memory, GPU, namespaces, storage classes |
| `inspect_weka` | Step 2 (when WEKA storage needed) | WEKA cluster capacity, filesystem inventory |
| `list_blueprints` | Step 3 (after inspect_cluster) | Catalog of available blueprints |
| `get_blueprint` | Step 4 (after list_blueprints) | Full blueprint spec and components |
| `get_crd_schema` | Step 5 (before generating YAML) | CRD OpenAPI schema and example manifests |
| `validate_yaml` | Step 7 (after generating YAML, before apply) | valid bool, structured errors list |
| `apply` | Step 11 (after user approval, confirmed=true only) | applied bool, applied kinds list |
| `status` | Step 12 (after apply, repeat until Ready) | CR deployment phase and conditions |
