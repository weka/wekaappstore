from __future__ import annotations

from types import SimpleNamespace

import pytest

from webapp.inspection.cluster import collect_cluster_inspection, flatten_cluster_status


def _namespace(name: str) -> SimpleNamespace:
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


def _storage_class(name: str, *, default: bool) -> SimpleNamespace:
    annotations = {}
    if default:
        annotations["storageclass.kubernetes.io/is-default-class"] = "true"
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, annotations=annotations),
        provisioner="weka.csi.weka.io",
        parameters={"filesystemName": "weka-home"},
        reclaim_policy="Delete",
        volume_binding_mode="Immediate",
        allow_volume_expansion=True,
    )


def _node(
    name: str,
    *,
    cpu: str,
    memory: str,
    gpus: str = "0",
    labels: dict[str, str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels=labels or {}),
        status=SimpleNamespace(
            conditions=[SimpleNamespace(type="Ready", status="True")],
            allocatable={"cpu": cpu, "memory": memory, "nvidia.com/gpu": gpus},
            capacity={"cpu": cpu, "memory": memory, "nvidia.com/gpu": gpus},
        ),
    )


def _pod(node_name: str, *, cpu: str, memory: str, gpus: str = "0") -> SimpleNamespace:
    resources = SimpleNamespace(
        requests={
            "cpu": cpu,
            "memory": memory,
            "nvidia.com/gpu": gpus,
        }
    )
    return SimpleNamespace(
        status=SimpleNamespace(phase="Running"),
        spec=SimpleNamespace(
            node_name=node_name,
            containers=[SimpleNamespace(resources=resources)],
            init_containers=[],
        ),
    )


class _CustomObjectsApiStub:
    def list_cluster_custom_object(self, *, group: str, version: str, plural: str) -> dict:
        if group == "nvidia.com":
            return {"items": [{"metadata": {"name": "gpu-policy"}}]}
        if group == "warp.io":
            return {"items": [{"metadata": {"namespace": "default", "name": "app-store-cluster-init"}}]}
        raise AssertionError(f"Unexpected custom object lookup: {group}/{plural}")


class _AppsApiStub:
    def list_namespaced_daemon_set(self, namespace: str) -> SimpleNamespace:
        return SimpleNamespace(items=[])


class _VersionApiStub:
    def get_code(self) -> SimpleNamespace:
        return SimpleNamespace(git_version="v1.30.0")


def test_collect_cluster_inspection_returns_bounded_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "webapp.inspection.cluster.client.ApiextensionsV1Api",
        lambda: SimpleNamespace(read_custom_resource_definition=lambda name: object()),
    )
    core_api = SimpleNamespace(
        list_namespace=lambda: SimpleNamespace(items=[_namespace("ai-platform"), _namespace("weka")]),
        list_node=lambda: SimpleNamespace(
            items=[
                _node("gpu-node", cpu="32", memory="256Gi", gpus="2", labels={
                    "nvidia.com/gpu.product": "NVIDIA L40",
                    "nvidia.com/gpu.memory": "48Gi",
                }),
                _node("cpu-node", cpu="16", memory="64Gi"),
            ]
        ),
        list_pod_for_all_namespaces=lambda: SimpleNamespace(
            items=[
                _pod("gpu-node", cpu="4", memory="16Gi", gpus="1"),
                _pod("cpu-node", cpu="2", memory="8Gi"),
            ]
        ),
    )
    storage_api = SimpleNamespace(
        list_storage_class=lambda: SimpleNamespace(
            items=[
                _storage_class("wekafs", default=True),
                _storage_class("gp3", default=False),
            ]
        )
    )

    snapshot = collect_cluster_inspection(
        core_api=core_api,
        storage_api=storage_api,
        custom_objects_api=_CustomObjectsApiStub(),
        apps_api=_AppsApiStub(),
        version_api=_VersionApiStub(),
    )

    assert snapshot["domains"]["namespaces"]["status"] == "complete"
    assert snapshot["domains"]["storage_classes"]["observed"]["default"] == "wekafs"
    assert snapshot["domains"]["cpu"]["observed"]["allocatable_cores"] == 48.0
    assert snapshot["domains"]["cpu"]["observed"]["free_cores"] == 42.0
    assert snapshot["domains"]["memory"]["observed"]["free_gib"] == 296.0
    assert snapshot["domains"]["gpu"]["status"] == "complete"
    assert snapshot["domains"]["gpu"]["observed"]["inventory"] == [
        {"model": "NVIDIA L40", "count": 2, "memory_gib": 48.0}
    ]
    assert snapshot["k8s_version"] == "v1.30.0"
    assert snapshot["gpu_operator_installed"] is True
    assert snapshot["app_store_cluster_init_present"] is True

    flattened = flatten_cluster_status(snapshot)
    assert flattened["cpu_nodes"] == 1
    assert flattened["gpu_nodes"] == 1
    assert flattened["default_storage_class"] == "wekafs"
    assert flattened["inspection_snapshot"] == snapshot


def test_collect_cluster_inspection_marks_partial_and_unavailable_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "webapp.inspection.cluster.client.ApiextensionsV1Api",
        lambda: SimpleNamespace(
            read_custom_resource_definition=lambda name: (_ for _ in ()).throw(Exception("missing crd"))
        ),
    )
    core_api = SimpleNamespace(
        list_namespace=lambda: (_ for _ in ()).throw(RuntimeError("namespaces denied")),
        list_node=lambda: SimpleNamespace(items=[_node("gpu-node", cpu="8", memory="32Gi", gpus="4")]),
        list_pod_for_all_namespaces=lambda: (_ for _ in ()).throw(RuntimeError("pods denied")),
    )
    storage_api = SimpleNamespace(
        list_storage_class=lambda: (_ for _ in ()).throw(RuntimeError("storage denied"))
    )

    snapshot = collect_cluster_inspection(
        core_api=core_api,
        storage_api=storage_api,
        custom_objects_api=_CustomObjectsApiStub(),
        apps_api=_AppsApiStub(),
        version_api=_VersionApiStub(),
    )

    assert snapshot["domains"]["namespaces"]["status"] == "unavailable"
    assert snapshot["domains"]["storage_classes"]["status"] == "unavailable"
    assert snapshot["domains"]["cpu"]["status"] == "partial"
    assert snapshot["domains"]["memory"]["status"] == "partial"
    assert snapshot["domains"]["gpu"]["status"] == "partial"
    assert snapshot["domains"]["gpu"]["blockers"][0]["code"] == "gpu_metadata_incomplete"
    assert snapshot["domains"]["cpu"]["blockers"][0]["code"] == "pod_list_failed"
    assert snapshot["app_store_crd_installed"] is None
