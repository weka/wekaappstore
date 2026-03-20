from __future__ import annotations

from types import SimpleNamespace

import pytest
from kubernetes.client.rest import ApiException

from webapp.inspection.cluster import collect_cluster_inspection, flatten_cluster_status


def _resource(**kwargs):
    return SimpleNamespace(**kwargs)


def _metadata(name: str, *, namespace: str | None = None, labels: dict | None = None, annotations: dict | None = None):
    return _resource(
        name=name,
        namespace=namespace,
        labels=labels or {},
        annotations=annotations or {},
    )


def _node(item: dict) -> SimpleNamespace:
    return _resource(
        metadata=_metadata(item["name"], labels=item.get("labels")),
        status=_resource(
            conditions=[_resource(type="Ready", status="True")],
            allocatable=item["allocatable"],
            capacity=item.get("capacity", item["allocatable"]),
        ),
    )


def _pod(item: dict) -> SimpleNamespace:
    return _resource(
        metadata=_metadata(item["name"], namespace=item["namespace"]),
        status=_resource(phase=item["phase"]),
        spec=_resource(
            node_name=item["node_name"],
            containers=[
                _resource(resources=_resource(requests=container["requests"]))
                for container in item.get("containers", [])
            ],
            init_containers=[
                _resource(resources=_resource(requests=container["requests"]))
                for container in item.get("init_containers", [])
            ],
        ),
    )


def _storage_class(item: dict) -> SimpleNamespace:
    return _resource(
        metadata=_metadata(item["name"], annotations=item.get("annotations")),
        provisioner=item["provisioner"],
        parameters=item.get("parameters", {}),
        reclaim_policy=item.get("reclaim_policy"),
        volume_binding_mode=item.get("volume_binding_mode"),
        allow_volume_expansion=item.get("allow_volume_expansion"),
    )


class _RecordingCoreApi:
    def __init__(self, topology: dict, *, fail_pods: bool = False) -> None:
        self.topology = topology
        self.fail_pods = fail_pods
        self.calls: list[str] = []

    def list_namespace(self):
        self.calls.append("list_namespace")
        return _resource(items=[_resource(metadata=_metadata(name)) for name in self.topology["namespaces"]])

    def list_node(self):
        self.calls.append("list_node")
        return _resource(items=[_node(item) for item in self.topology["nodes"]])

    def list_pod_for_all_namespaces(self):
        self.calls.append("list_pod_for_all_namespaces")
        if self.fail_pods:
            raise RuntimeError("pod requests unavailable")
        return _resource(items=[_pod(item) for item in self.topology["pods"]])


class _StorageApi:
    def __init__(self, topology: dict) -> None:
        self.topology = topology

    def list_storage_class(self):
        return _resource(items=[_storage_class(item) for item in self.topology["storage_classes"]])


class _CustomObjectsApi:
    def __init__(self, topology: dict) -> None:
        self.topology = topology

    def list_cluster_custom_object(self, *, group: str, version: str, plural: str):
        if (group, version, plural) == ("nvidia.com", "v1", "clusterpolicies"):
            return {"items": self.topology["cluster_policies"]}
        if (group, version, plural) == ("warp.io", "v1alpha1", "wekaappstores"):
            return {"items": self.topology["app_store_crs"]}
        raise AssertionError(f"Unexpected custom object lookup {(group, version, plural)}")


class _AppsApi:
    def __init__(self, topology: dict) -> None:
        self.topology = topology

    def list_namespaced_daemon_set(self, namespace: str):
        return _resource(items=self.topology.get("daemon_sets", {}).get(namespace, []))


class _ApiExtensionsApi:
    def read_custom_resource_definition(self, name: str):
        if name != "wekaappstores.warp.io":
            raise AssertionError(f"Unexpected CRD lookup {name}")
        return _resource(metadata=_metadata(name))


class _VersionApi:
    def __init__(self, git_version: str) -> None:
        self.git_version = git_version

    def get_code(self):
        return _resource(git_version=self.git_version)


def test_collect_cluster_inspection_returns_planner_grade_snapshot(
    mocked_cluster_inspection_topology: dict,
) -> None:
    core_api = _RecordingCoreApi(mocked_cluster_inspection_topology)
    snapshot = collect_cluster_inspection(
        load_kube_config=lambda: None,
        core_api=core_api,
        storage_api=_StorageApi(mocked_cluster_inspection_topology),
        custom_objects_api=_CustomObjectsApi(mocked_cluster_inspection_topology),
        apps_api=_AppsApi(mocked_cluster_inspection_topology),
        apiextensions_api=_ApiExtensionsApi(),
        version_api=_VersionApi(mocked_cluster_inspection_topology["k8s_version"]),
    )

    flattened = flatten_cluster_status(snapshot)

    assert core_api.calls == ["list_namespace", "list_node", "list_pod_for_all_namespaces"]
    assert snapshot["domains"]["namespaces"]["observed"]["names"] == ["ai-platform", "default", "weka"]
    assert snapshot["domains"]["storage_classes"]["observed"]["default"] == "wekafs"
    assert snapshot["domains"]["cpu"]["observed"]["allocatable_cores"] == 48.0
    assert snapshot["domains"]["cpu"]["observed"]["free_cores"] == 36.0
    assert snapshot["domains"]["memory"]["observed"]["free_gib"] == 304.0
    assert snapshot["domains"]["gpu"]["status"] == "complete"
    assert snapshot["domains"]["gpu"]["observed"]["inventory"] == [
        {"model": "NVIDIA L40", "count": 4, "memory_gib": 48.0}
    ]
    assert flattened["gpu_devices_free"] == 2
    assert flattened["default_storage_class"] == "wekafs"
    assert flattened["inspection_snapshot"] == snapshot


def test_collect_cluster_inspection_surfaces_partial_gpu_and_pod_failures(
    mocked_cluster_inspection_topology: dict,
) -> None:
    topology = dict(mocked_cluster_inspection_topology)
    topology["nodes"] = [
        mocked_cluster_inspection_topology["nodes"][0],
        {
            "name": "gpu-node-2",
            "labels": {},
            "allocatable": {"cpu": "32", "memory": "256Gi", "nvidia.com/gpu": "2"},
            "capacity": {"cpu": "32", "memory": "256Gi", "nvidia.com/gpu": "2"},
        },
    ]

    snapshot = collect_cluster_inspection(
        load_kube_config=lambda: None,
        core_api=_RecordingCoreApi(topology, fail_pods=True),
        storage_api=_StorageApi(topology),
        custom_objects_api=_CustomObjectsApi(topology),
        apps_api=_AppsApi(topology),
        apiextensions_api=_ApiExtensionsApi(),
        version_api=_VersionApi(topology["k8s_version"]),
    )

    assert snapshot["domains"]["cpu"]["status"] == "partial"
    assert snapshot["domains"]["memory"]["status"] == "partial"
    assert snapshot["domains"]["gpu"]["status"] == "partial"
    assert snapshot["domains"]["gpu"]["blockers"][0]["code"] == "gpu_metadata_incomplete"
    assert "pod requests were unavailable" in snapshot["domains"]["cpu"]["notes"][0]


def test_collect_cluster_inspection_marks_unavailable_domains_when_namespace_lookup_fails(
    mocked_cluster_inspection_topology: dict,
) -> None:
    class _FailingNamespaceCoreApi(_RecordingCoreApi):
        def list_namespace(self):
            raise ApiException(status=403, reason="forbidden")

    snapshot = collect_cluster_inspection(
        load_kube_config=lambda: None,
        core_api=_FailingNamespaceCoreApi(mocked_cluster_inspection_topology),
        storage_api=_StorageApi(mocked_cluster_inspection_topology),
        custom_objects_api=_CustomObjectsApi(mocked_cluster_inspection_topology),
        apps_api=_AppsApi(mocked_cluster_inspection_topology),
        apiextensions_api=_ApiExtensionsApi(),
        version_api=_VersionApi(mocked_cluster_inspection_topology["k8s_version"]),
    )

    assert snapshot["domains"]["namespaces"]["status"] == "unavailable"
    assert snapshot["domains"]["namespaces"]["blockers"][0]["code"] == "namespace_list_failed"
