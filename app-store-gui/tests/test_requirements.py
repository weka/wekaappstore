from __future__ import annotations

import textwrap

import webapp.main as main


def _write(tmp_path, text):
    p = tmp_path / "bp.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


# --------------------------- declared (x-requirements) ---------------------------

def test_declared_requirements_full(tmp_path):
    path = _write(tmp_path, """
        x-requirements:
          cpu: {cores: 16}
          memory: {gib: 64}
          gpu: {count: 2, model: H100}
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec: {appStack: {components: []}}
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["source"] == "declared"
    assert r["cpu_cores"] == 16.0 and r["memory_gib"] == 64.0
    assert r["gpu_required"] and r["gpu_count_known"] and r["gpu_devices"] == 2
    assert r["gpu_model"] == "H100"


def test_declared_gpu_without_count_is_unknown(tmp_path):
    path = _write(tmp_path, """
        x-requirements:
          gpu: {model: A100}
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec: {appStack: {components: []}}
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["source"] == "declared"
    assert r["gpu_required"] and not r["gpu_count_known"] and r["gpu_devices"] is None


def test_declared_gpu_count_zero_means_not_required(tmp_path):
    path = _write(tmp_path, """
        x-requirements:
          gpu: {count: 0}
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec: {appStack: {components: []}}
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["gpu_required"] is False


# --------------------------- inference ---------------------------

def test_inferred_sums_resources_times_replicas(tmp_path):
    path = _write(tmp_path, """
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec:
          appStack:
            components:
              - name: app
                kubernetesManifest: |
                  apiVersion: apps/v1
                  kind: Deployment
                  metadata: {name: app}
                  spec:
                    replicas: 3
                    template:
                      spec:
                        containers:
                          - name: c
                            resources:
                              requests: {cpu: "2", memory: 4Gi}
                              limits: {nvidia.com/gpu: 2}
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["source"] == "inferred"
    assert r["cpu_cores"] == 6.0          # 2 * 3
    assert r["memory_gib"] == 12.0        # 4Gi * 3
    assert r["gpu_required"] and r["gpu_count_known"] and r["gpu_devices"] == 6  # 2 * 3


def test_inferred_gpu_from_helm_values_content(tmp_path):
    path = _write(tmp_path, """
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec:
          appStack:
            components:
              - name: nim
                helmChart:
                  name: nim
                  valuesContent: |
                    resources:
                      limits:
                        nvidia.com/gpu: 4
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["gpu_required"] and r["gpu_count_known"] and r["gpu_devices"] == 4


def test_heuristic_gpu_required_count_unknown(tmp_path):
    # No countable GPU resource, but a NIM/NVIDIA marker → required, count unknown.
    path = _write(tmp_path, """
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec:
          appStack:
            components:
              - name: nvidia-rag
                helmChart:
                  repository: https://helm.ngc.nvidia.com/nim
                  valuesFiles: [values-rag.yaml]
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["gpu_required"] and not r["gpu_count_known"] and r["gpu_devices"] is None


def test_no_gpu_when_absent(tmp_path):
    path = _write(tmp_path, """
        apiVersion: warp.io/v1alpha1
        kind: WekaAppStore
        metadata: {name: x, namespace: y}
        spec:
          appStack:
            components:
              - name: web
                kubernetesManifest: |
                  apiVersion: apps/v1
                  kind: Deployment
                  metadata: {name: web}
                  spec:
                    template:
                      spec:
                        containers:
                          - name: c
                            resources:
                              requests: {cpu: 500m, memory: 256Mi}
    """)
    r = main.compute_blueprint_requirements(path)
    assert r["gpu_required"] is False
    assert r["cpu_cores"] == 0.5


def test_missing_path_returns_empty():
    r = main.compute_blueprint_requirements(None)
    assert r["source"] == "none" and r["gpu_required"] is False


# --------------------------- meets (free-capacity comparison) ---------------------------

def test_meets_free_capacity():
    reqs = {"cpu_cores": 8, "memory_gib": 32, "gpu_devices": 2,
            "gpu_required": True, "gpu_count_known": True}
    status = {"cpu_cores_free": 10, "memory_gib_free": 16, "gpu_devices_free": 1}
    m = main._compute_requirement_meets(reqs, status)
    assert m["cpu"] is True       # 10 >= 8
    assert m["memory"] is False   # 16 < 32
    assert m["gpu"] is False      # 1 < 2


def test_meets_gpu_unknown_is_none():
    reqs = {"gpu_required": True, "gpu_count_known": False}
    status = {"gpu_devices_free": 4}
    assert main._compute_requirement_meets(reqs, status)["gpu"] is None


def test_meets_gpu_not_required_is_true():
    reqs = {"gpu_required": False}
    assert main._compute_requirement_meets(reqs, {})["gpu"] is True


def test_meets_unknown_when_cluster_figure_missing():
    reqs = {"cpu_cores": 4}
    assert main._compute_requirement_meets(reqs, {"cpu_cores_free": None})["cpu"] is None
