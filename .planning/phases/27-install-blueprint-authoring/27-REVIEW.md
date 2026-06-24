---
phase: 27-install-blueprint-authoring
reviewed: 2026-06-24T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - cluster_init/app-store-install.yaml
  - operator_module/tests/test_install_blueprint.py
findings:
  critical: 3
  warning: 2
  info: 2
  total: 7
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-06-24
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Two deliverables reviewed: the `app-store-install.yaml` blueprint defining a 12-component install stack, and `test_install_blueprint.py` which provides cluster-free verification of that blueprint.

The test suite is well-structured and covers topo ordering, YAML validity, default StorageClass uniqueness, and `stringData` discipline. However three blockers were found: a missing dependency edge that causes the WekaClient DaemonSet to deploy before nodes are labeled; a variable name mismatch (`join_ip_ports` declared in `x-variables` but never used by the template, which references the different `join_ip_ports_list`); and two Job readiness checks that will silently fail to wait because `readinessCheck.name` is missing and the auto-derived label selector does not match any label on the deployed Job objects.

---

## Critical Issues

### CR-01: WekaClient CR applied before node-label Job runs (missing dependency edge)

**File:** `cluster_init/app-store-install.yaml:258-279`

**Issue:** The `weka-client` component declares `dependsOn: [weka-operator, weka-client-secret]` but does NOT depend on `weka-node-label-job`. Kahn's algorithm (confirmed by simulation) produces this concrete order:

```
... weka-client (index 8), storageclass-demote-job (index 9), weka-node-label-job (index 10) ...
```

The WekaClient CR uses `nodeSelector: {weka.io/supports-clients: "true"}`, but when it is applied, the node-label Job has not yet completed. The DaemonSet pods go `Pending` on all nodes and stay there until the Job runs two steps later. If the WekaClient operator treats prolonged Pending as a fatal condition, it may enter an error state before the labels arrive.

**Fix:**
```yaml
      - name: weka-client
        dependsOn:
          - weka-operator
          - weka-client-secret
          - weka-node-label-job   # add this line
```

---

### CR-02: `join_ip_ports` x-variable declared and collected but never substituted into the template

**File:** `cluster_init/app-store-install.yaml:8-10` (x-variable declaration) vs line 279 (template token)

**Issue:** The `x-variables` block declares `join_ip_ports` as `required: true`. The GUI will collect this value from the user and pass it as `join_ip_ports` during rendering. However the template at line 279 uses `[[ join_ip_ports_list ]]` — a completely different name — for the `joinIpPorts` field of the WekaClient CR. The user-supplied `join_ip_ports` value is silently discarded; `join_ip_ports_list` is a Phase-29 server-injected variable that does not yet exist in the GUI code (`grep join_ip_ports_list app-store-gui/webapp/main.py` returns nothing).

Result: if this blueprint is deployed today, the WekaClient CR's `joinIpPorts` field renders to `null` (Jinja2 silently drops undefined variables), and the WEKA client cannot join the cluster.

The test masks this by providing both keys in `SAMPLE_VARS` but never asserting that `joinIpPorts` is non-null in the rendered WekaClient manifest.

**Fix — Option A (simplest, no Phase 29 server work):** Rename the template token to match the declared x-variable, and convert the comma-separated string to a YAML sequence in a Jinja2 filter:

```yaml
  # in x-variables block — rename for clarity
  join_ips:
    required: true
    description: "Comma-delimited host:port list (e.g. 192.168.1.1:14000,192.168.1.2:14000)"
```

```yaml
  # in WekaClient kubernetesManifest (use jinja2 split filter)
  joinIpPorts: [[ join_ips.split(',') | list ]]
```

**Fix — Option B (keep Phase 29 injection intent):** Document that this blueprint is not deployable until Phase 29 is complete, and add an assertion in the test:

```python
# In test_topo_order or a new test:
weka_client_comp = next(c for c in components if c["name"] == "weka-client")
manifest = list(yaml.safe_load_all(weka_client_comp["kubernetesManifest"]))
cr = next(d for d in manifest if d.get("kind") == "WekaClient")
assert cr["spec"]["joinIpPorts"], "joinIpPorts must be non-empty; check join_ip_ports_list injection"
```

---

### CR-03: Job readiness checks will silently fall through due to missing `name` field

**File:** `cluster_init/app-store-install.yaml:201-203` (weka-node-label-job) and `294-296` (storageclass-demote-job)

**Issue:** Both Job components use `readinessCheck: {type: job, timeout: 300}` with no `name` field. The operator's `wait_for_component_ready` function (confirmed in `operator_module/main.py:885-938`) falls back to a label selector when `name` is absent:

```python
default_selector = f"app={component['name']}"
# → "app=weka-node-label-job" and "app=storageclass-demote-job"
```

The actual Job metadata in both manifests has **no labels at all**. `kubectl wait job -l app=weka-node-label-job -n kube-system --for=condition=complete --timeout=300s` will either time out waiting for a non-existent resource or exit immediately with a non-zero code — causing the operator to misreport job readiness.

Additionally, there is a component/resource name mismatch:
- Component `weka-node-label-job` → actual Job name `weka-node-label`
- Component `storageclass-demote-job` → actual Job name `storageclass-demote`

**Fix:** Add `name:` to both readiness checks so the operator uses `kubectl wait job/<name>` directly:

```yaml
      # weka-node-label-job component:
        readinessCheck:
          type: job
          name: weka-node-label      # actual Job metadata.name
          namespace: kube-system
          timeout: 300

      # storageclass-demote-job component:
        readinessCheck:
          type: job
          name: storageclass-demote  # actual Job metadata.name
          namespace: kube-system
          timeout: 300
```

---

## Warnings

### WR-01: `docker.io/bitnami/kubectl:latest` pinned to `latest` in two Job containers

**File:** `cluster_init/app-store-install.yaml:218` and `340`

**Issue:** Both the `weka-node-label` and `storageclass-demote` Jobs pull `docker.io/bitnami/kubectl:latest`. The `:latest` tag:
1. Forces `imagePullPolicy: Always` in Kubernetes (implicit for the `latest` tag), causing a Docker Hub pull on every Job Pod creation — and potential rate-limiting failures in air-gapped or rate-limited environments.
2. Makes the image non-deterministic; a future `latest` that changes the `kubectl` API or drops flags used in the shell scripts can silently break these Jobs.

**Fix:** Pin to a specific semver tag matching the cluster's Kubernetes minor version:
```yaml
image: docker.io/bitnami/kubectl:1.29   # or match cluster k8s version
```

---

### WR-02: `quay_dockerconfigjson` not declared `required: true` but has no default and no server injection yet

**File:** `cluster_init/app-store-install.yaml:23-25`

**Issue:** `quay_dockerconfigjson` is declared with `validate: false` and a description saying it is "injected by server". It has neither `required: true` nor a `default:` value. No server-side injection exists in the current GUI code (`grep quay_dockerconfigjson app-store-gui/webapp/main.py` returns zero results). If a user submits this blueprint today via the GUI without Phase-29 injection, `quay_dockerconfigjson` renders to an empty string, producing a structurally invalid `dockerconfigjson` Secret. The WEKA operator and CSI driver image pulls from `quay.io` will then fail with `ImagePullBackOff`.

**Fix (short-term):** Add `required: true` so the GUI at least forces the user to provide it manually, preventing a silent empty-string render:
```yaml
  quay_dockerconfigjson:
    required: true
    validate: false
    description: "GUI-built dockerconfigjson string for quay.io pull secret (injected by server)"
```
Longer-term, Phase-29 server injection should set this automatically and remove `required` from the user-facing schema.

---

## Info

### IN-01: `join_ip_ports` in `SAMPLE_VARS` is never consumed by the template

**File:** `operator_module/tests/test_install_blueprint.py:56`

**Issue:** `SAMPLE_VARS` includes `"join_ip_ports": "10.0.0.1:14000,10.0.0.2:14000"` which is not a template token (the template only uses `join_ip_ports_list`). The value is silently ignored by Jinja2. This is a direct consequence of the CR-02 mismatch, but it also means the comment on line 35 ("These cover every [[ ]] token") is inaccurate — `join_ip_ports` covers no token.

**Fix:** Once CR-02 is resolved by renaming the template token or renaming the x-variable, remove the orphaned key and update the comment.

---

### IN-02: `test_quay_roundtrip` validates only `quay-secret-operator-ns`, not `quay-secret-default-ns`

**File:** `operator_module/tests/test_install_blueprint.py:147-188`

**Issue:** The test verifies the dockerconfigjson round-trip only for `quay-secret-operator-ns`. The `quay-secret-default-ns` component is structurally identical and uses the same `[[ quay_dockerconfigjson ]]` token. While both components are validated indirectly by `test_stringdata_only`, the targeted round-trip assertion is not applied to the `default` namespace copy.

**Fix:** Extract the inner assertion into a helper and call it for both components, or add a `pytest.mark.parametrize` over both component names:

```python
@pytest.mark.parametrize("comp_name", ["quay-secret-operator-ns", "quay-secret-default-ns"])
def test_quay_roundtrip(comp_name):
    ...
    quay_comp = next(c for c in components if c["name"] == comp_name, None)
    ...
```

---

_Reviewed: 2026-06-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
