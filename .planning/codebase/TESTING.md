# Testing Patterns

**Analysis Date:** 2026-03-17

## Test Framework

**Runner:**
- There is no repository-level Python unit test runner configured. The repo does not include `pytest.ini`, `tox.ini`, `pyproject.toml`, or a `tests/` tree.
- The only explicit automated test artifact is a Helm chart hook at `weka-app-store-operator-chart/templates/tests/test-connection.yaml`.
- The repository contains manual validation manifests such as `test-pvc.yaml`, `test-pvc-pod.yaml`, and `cluster_init/job-test.yaml`, which suggests operational smoke testing is favored over automated application tests.

**Assertion Library:**
- No Python assertion library is present because no Python test suite is checked in.
- The Helm chart test relies on Kubernetes pod completion behavior rather than language-level assertions: the `busybox` container in `weka-app-store-operator-chart/templates/tests/test-connection.yaml` runs `wget` against the chart service.

**Run Commands:**
```bash
helm lint weka-app-store-operator-chart
helm install weka-app-store ./weka-app-store-operator-chart -n weka-app-store --create-namespace
helm test weka-app-store -n weka-app-store
kubectl apply -f test-pvc.yaml
kubectl apply -f test-pvc-pod.yaml
kubectl apply -f cluster_init/job-test.yaml
```

## Test File Organization

**Location:**
- Helm smoke coverage lives under `weka-app-store-operator-chart/templates/tests/`.
- Manual validation YAML files live at the repository root and in infrastructure directories rather than in a dedicated test package.
- No `tests/` or `__tests__/` directories exist for `app-store-gui/` or `operator_module/`.

**Naming:**
- Helm test naming follows chart conventions: `test-connection.yaml`.
- Manual validation files are named after the infrastructure behavior being exercised, for example `test-pvc.yaml` and `test-pvc-pod.yaml`.
- There is no naming convention in place for unit, integration, or end-to-end test files because those suites do not exist in the repository.

**Observed Structure:**
```text
weka-app-store-operator-chart/
  templates/
    tests/
      test-connection.yaml

cluster_init/
  job-test.yaml

test-pvc.yaml
test-pvc-pod.yaml
```

## Test Structure

**Current Pattern:**
- Tests are environment-driven rather than code-driven.
- Validation centers on deploying Kubernetes resources and observing readiness or command success.
- The chart test is a single pod hook that attempts an HTTP fetch against the deployed service.

**Helm Hook Example:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ include "weka-app-store-operator-chart.fullname" . }}:{{ .Values.service.port }}']
  restartPolicy: Never
```

## Mocking

**Framework:**
- No mocking framework is present in the repository.

**Observed Practice:**
- The codebase currently prefers real Kubernetes, Helm, and shell interactions over mocked test seams.
- This is evident in `app-store-gui/webapp/main.py`, which calls Kubernetes APIs and `subprocess.run` directly, and in `operator_module/main.py`, which shells out to `helm` and `kubectl` from production code.

**Implication for New Tests:**
- Any new automated tests will need to introduce their own mocking approach for `subprocess.run`, Kubernetes client calls, `kr8s`, and filesystem interactions because the repo has no established fixture layer yet.

## Fixtures and Factories

**Current State:**
- No shared fixtures, factories, or test data builders are checked in.
- Deployment examples and values files act as de facto manual fixtures, especially `weka-csi-config/blueprint-default-values.yaml`, `cluster_init/app-store-cluster-init.yaml`, and the chart defaults in `weka-app-store-operator-chart/values.yaml`.

**Practical Repository Fixtures:**
- Sample cluster and storage manifests under `weka-csi-config/` are the closest thing to reusable test inputs.
- Template HTML files in `app-store-gui/webapp/templates/` function as runtime assets, not test fixtures.

## Coverage

**Requirements:**
- No coverage target or enforcement is defined anywhere in the repository.
- There is no CI workflow under `.github/workflows/` to enforce tests, lint, or coverage on push.

**Configuration:**
- No code coverage tool configuration is present.
- Testability is currently inferred from whether the chart installs and whether Kubernetes resources come up successfully.

## Test Types

**Unit Tests:**
- Not present.
- Neither `app-store-gui/webapp/main.py` nor `operator_module/main.py` has neighboring unit test modules.

**Integration Tests:**
- Minimal and infrastructure-oriented.
- The Helm hook in `weka-app-store-operator-chart/templates/tests/test-connection.yaml` is the clearest integration-style check because it validates service reachability after deployment.

**End-to-End / Smoke Tests:**
- Manual cluster validation is the dominant pattern.
- `test-pvc.yaml` and `test-pvc-pod.yaml` validate storage provisioning behavior externally.
- `cluster_init/job-test.yaml` provides another cluster-executed validation manifest.

## Common Patterns

**What Gets Verified Today:**
- Helm release reachability after installation via `helm test`.
- Kubernetes storage and job behavior through manual manifest application.
- Container startup expectations indirectly through `docker/webapp.Dockerfile`, `docker/operator.Dockerfile`, and runtime probes defined in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml`.

**What Is Not Verified Automatically:**
- FastAPI route behavior in `app-store-gui/webapp/main.py`.
- Kopf reconciliation logic and Helm orchestration in `operator_module/main.py`.
- Template rendering correctness beyond what `helm lint` and a successful chart install would catch.

## Quality Gaps To Respect When Adding Tests

- New tests will be establishing the first real application-level test patterns in this repo, so they should be introduced deliberately and documented with their run command.
- The highest-value seams are the pure or near-pure helpers in `operator_module/main.py` such as `_deep_merge`, `merge_values`, `resolve_dependencies`, and CRD strategy helpers.
- For the FastAPI app, the safest first wave would isolate helper functions and route-level behavior with mocked Kubernetes clients before attempting cluster-backed end-to-end tests.
- If a new automated suite is added, it should not assume any existing CI support because none is currently present.

## Recommended Baseline Commands For Future Contributors

```bash
python -m py_compile app-store-gui/webapp/main.py operator_module/main.py
helm lint weka-app-store-operator-chart
helm template weka-app-store ./weka-app-store-operator-chart >/tmp/weka-app-store-rendered.yaml
helm install weka-app-store ./weka-app-store-operator-chart -n weka-app-store --create-namespace
helm test weka-app-store -n weka-app-store
```

*Testing analysis: 2026-03-17*
*Update when automated tests or CI are introduced*
