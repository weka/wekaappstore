from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from kubernetes import client
from kubernetes.client.rest import ApiException


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cpu_to_millicores(value: Any) -> int:
    try:
        if value is None:
            return 0
        text = str(value).strip()
        if text.endswith("m"):
            return int(float(text[:-1]))
        return int(float(text) * 1000)
    except Exception:
        return 0


def _memory_to_bytes(value: Any) -> int:
    try:
        if value is None:
            return 0
        text = str(value).strip()
        suffixes = {
            "Ki": 1024,
            "Mi": 1024**2,
            "Gi": 1024**3,
            "Ti": 1024**4,
            "Pi": 1024**5,
            "K": 1000,
            "M": 1000**2,
            "G": 1000**3,
            "T": 1000**4,
            "P": 1000**5,
        }
        for suffix, multiplier in suffixes.items():
            if text.endswith(suffix):
                return int(float(text[: -len(suffix)]) * multiplier)
        return int(float(text))
    except Exception:
        return 0


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(str(value))
    except Exception:
        return 0


def _round_gib(value: int) -> float:
    return round(value / float(1024**3), 2)


def _extract_gpu_model(labels: Dict[str, str]) -> Optional[str]:
    candidate_keys = (
        "nvidia.com/gpu.product",
        "nvidia.com/gpu.machine",
        "gpu.nvidia.com/class",
        "weka.io/gpu-model",
    )
    for key in candidate_keys:
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_gpu_memory_gib(labels: Dict[str, str], capacity: Dict[str, Any]) -> Optional[float]:
    for key in ("nvidia.com/gpu.memory", "gpu.nvidia.com/memory"):
        value = labels.get(key)
        if value:
            text = str(value).strip().lower()
            if text.endswith("gib"):
                return round(float(text[:-3]), 2)
            if text.endswith("gi"):
                return round(float(text[:-2]), 2)
            if text.endswith("gb"):
                return round(float(text[:-2]), 2)
            if text.endswith("mi"):
                return round(float(text[:-2]) / 1024.0, 2)
            try:
                numeric = float(text)
                if numeric > 1024:
                    return round(numeric / 1024.0, 2)
                return round(numeric, 2)
            except Exception:
                continue
    bytes_value = capacity.get("nvidia.com/gpu.memory")
    if bytes_value is not None:
        parsed = _memory_to_bytes(bytes_value)
        if parsed > 0:
            return _round_gib(parsed)
    return None


def _make_blocker(code: str, message: str, domain: str) -> Dict[str, str]:
    return {"code": code, "message": message, "domain": domain}


def _make_domain(
    *,
    captured_at: str,
    status: str,
    observed: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
    blockers: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "required": True,
        "freshness": {
            "captured_at": captured_at,
            "max_age_seconds": 300,
        },
        "observed": observed or {},
        "notes": notes or [],
        "blockers": blockers or [],
    }


def collect_cluster_inspection(
    *,
    load_kube_config: Optional[Callable[[], None]] = None,
    core_api: Any = None,
    storage_api: Any = None,
    custom_objects_api: Any = None,
    apps_api: Any = None,
    apiextensions_api: Any = None,
    version_api: Any = None,
) -> Dict[str, Any]:
    if load_kube_config is not None:
        load_kube_config()

    core_api = core_api or client.CoreV1Api()
    storage_api = storage_api or client.StorageV1Api()
    custom_objects_api = custom_objects_api or client.CustomObjectsApi()
    apps_api = apps_api or client.AppsV1Api()
    apiextensions_api = apiextensions_api or client.ApiextensionsV1Api()
    version_api = version_api or client.VersionApi()

    captured_at = _utc_timestamp()
    ready_node_names: set[str] = set()
    namespaces: List[str] = []
    storage_classes: List[str] = []
    default_storage_class = None
    default_storage_class_details = None
    cpu_nodes = 0
    gpu_nodes = 0
    cpu_milli_total = 0
    memory_bytes_total = 0
    gpu_devices_total = 0
    gpu_inventory: Dict[str, Dict[str, Any]] = {}
    gpu_notes: List[str] = []
    gpu_blockers: List[Dict[str, str]] = []
    app_store_crd_installed = None
    cluster_init_present = None
    applied_crs: List[str] = []
    gpu_operator_installed = None
    k8s_version = None

    domains: Dict[str, Dict[str, Any]] = {}

    try:
        namespaces = sorted(
            item.metadata.name
            for item in (core_api.list_namespace().items or [])
            if getattr(item, "metadata", None) and getattr(item.metadata, "name", None)
        )
        domains["namespaces"] = _make_domain(
            captured_at=captured_at,
            status="complete",
            observed={"names": namespaces},
        )
    except Exception as exc:
        domains["namespaces"] = _make_domain(
            captured_at=captured_at,
            status="unavailable",
            blockers=[_make_blocker("namespace_list_failed", str(exc), "namespaces")],
        )

    try:
        storage_items = storage_api.list_storage_class().items or []
        for storage_class in storage_items:
            annotations = dict(storage_class.metadata.annotations or {})
            storage_classes.append(storage_class.metadata.name)
            is_default = annotations.get("storageclass.kubernetes.io/is-default-class") or annotations.get(
                "storageclass.beta.kubernetes.io/is-default-class"
            )
            if str(is_default).lower() == "true" and default_storage_class is None:
                default_storage_class = storage_class.metadata.name
                default_storage_class_details = {
                    "name": storage_class.metadata.name,
                    "provisioner": storage_class.provisioner,
                    "parameters": dict(storage_class.parameters or {}),
                    "reclaimPolicy": getattr(storage_class, "reclaim_policy", None)
                    or getattr(storage_class, "reclaimPolicy", None),
                    "volumeBindingMode": getattr(storage_class, "volume_binding_mode", None)
                    or getattr(storage_class, "volumeBindingMode", None),
                    "allowVolumeExpansion": getattr(storage_class, "allow_volume_expansion", None)
                    or getattr(storage_class, "allowVolumeExpansion", None),
                    "annotations": annotations,
                }
        domains["storage_classes"] = _make_domain(
            captured_at=captured_at,
            status="complete",
            observed={
                "names": sorted(storage_classes),
                "default": default_storage_class,
                "default_details": default_storage_class_details,
            },
        )
    except Exception as exc:
        domains["storage_classes"] = _make_domain(
            captured_at=captured_at,
            status="unavailable",
            blockers=[_make_blocker("storage_class_list_failed", str(exc), "storage_classes")],
        )

    node_failure = None
    try:
        nodes = core_api.list_node().items or []
        for node in nodes:
            conditions = {condition.type: condition.status for condition in (node.status.conditions or [])}
            if conditions.get("Ready") != "True":
                continue
            ready_node_names.add(node.metadata.name)
            allocatable = dict(node.status.allocatable or {})
            capacity = dict(getattr(node.status, "capacity", {}) or {})
            labels = dict(node.metadata.labels or {})
            cpu_milli_total += _cpu_to_millicores(allocatable.get("cpu"))
            memory_bytes_total += _memory_to_bytes(allocatable.get("memory"))
            gpu_count = max(_safe_int(allocatable.get("nvidia.com/gpu") or 0), 0)
            gpu_devices_total += gpu_count
            if gpu_count > 0:
                gpu_nodes += 1
                model = _extract_gpu_model(labels)
                memory_gib = _extract_gpu_memory_gib(labels, capacity)
                inventory_key = model or f"unknown-{node.metadata.name}"
                inventory = gpu_inventory.setdefault(
                    inventory_key,
                    {
                        "model": model,
                        "count": 0,
                        "memory_gib": memory_gib,
                    },
                )
                inventory["count"] += gpu_count
                if inventory.get("memory_gib") is None and memory_gib is not None:
                    inventory["memory_gib"] = memory_gib
                if model is None or memory_gib is None:
                    gpu_notes.append(
                        f"Node {node.metadata.name} exposes GPU count without full model and memory metadata."
                    )
                    gpu_blockers.append(
                        _make_blocker(
                            "gpu_metadata_incomplete",
                            f"Node {node.metadata.name} is missing GPU model or memory details.",
                            "gpu",
                        )
                    )
            else:
                cpu_nodes += 1
    except Exception as exc:
        node_failure = str(exc)

    cpu_milli_used = 0
    memory_bytes_used = 0
    gpu_devices_used = 0
    pod_failure = None
    if node_failure is None:
        try:
            pods = core_api.list_pod_for_all_namespaces().items or []
            for pod in pods:
                phase = (pod.status.phase or "").lower()
                if phase in ("succeeded", "failed"):
                    continue
                if getattr(pod.spec, "node_name", None) not in ready_node_names:
                    continue
                containers = list(pod.spec.containers or [])
                init_containers = list(getattr(pod.spec, "init_containers", []) or [])
                for container in containers + init_containers:
                    resources = getattr(container, "resources", None)
                    requests = getattr(resources, "requests", None) if resources else None
                    if not isinstance(requests, dict):
                        continue
                    cpu_milli_used += _cpu_to_millicores(requests.get("cpu"))
                    memory_bytes_used += _memory_to_bytes(requests.get("memory"))
                    try:
                        gpu_devices_used += int(str(requests.get("nvidia.com/gpu") or 0))
                    except Exception:
                        pass
        except Exception as exc:
            pod_failure = str(exc)

    if node_failure is not None:
        blocker = _make_blocker("node_list_failed", node_failure, "cpu")
        domains["cpu"] = _make_domain(captured_at=captured_at, status="unavailable", blockers=[blocker])
        domains["memory"] = _make_domain(
            captured_at=captured_at,
            status="unavailable",
            blockers=[_make_blocker("node_list_failed", node_failure, "memory")],
        )
        domains["gpu"] = _make_domain(
            captured_at=captured_at,
            status="unavailable",
            blockers=[_make_blocker("node_list_failed", node_failure, "gpu")],
        )
    else:
        cpu_observed = {
            "ready_nodes": len(ready_node_names),
            "cpu_nodes": cpu_nodes,
            "gpu_nodes": gpu_nodes,
            "allocatable_cores": round(cpu_milli_total / 1000.0, 2),
            "used_cores": round(cpu_milli_used / 1000.0, 2),
            "free_cores": round(max(cpu_milli_total - cpu_milli_used, 0) / 1000.0, 2),
        }
        memory_observed = {
            "ready_nodes": len(ready_node_names),
            "allocatable_gib": _round_gib(memory_bytes_total),
            "used_gib": _round_gib(memory_bytes_used),
            "free_gib": _round_gib(max(memory_bytes_total - memory_bytes_used, 0)),
        }
        gpu_observed = {
            "ready_nodes": len(ready_node_names),
            "gpu_nodes": gpu_nodes,
            "total_devices": int(gpu_devices_total),
            "used_devices": int(gpu_devices_used),
            "free_devices": max(int(gpu_devices_total) - int(gpu_devices_used), 0),
            "inventory": sorted(gpu_inventory.values(), key=lambda item: (item.get("model") or "", item["count"])),
        }

        if pod_failure is not None:
            cpu_status = "partial"
            memory_status = "partial"
            cpu_blockers = [_make_blocker("pod_list_failed", pod_failure, "cpu")]
            memory_blockers = [_make_blocker("pod_list_failed", pod_failure, "memory")]
            cpu_notes = ["CPU totals are present but free capacity is incomplete because pod requests were unavailable."]
            memory_notes = [
                "Memory totals are present but free capacity is incomplete because pod requests were unavailable."
            ]
        else:
            cpu_status = "complete"
            memory_status = "complete"
            cpu_blockers = []
            memory_blockers = []
            cpu_notes = []
            memory_notes = []

        gpu_status = "partial" if gpu_blockers else "complete"
        if gpu_nodes == 0:
            gpu_notes = ["No ready GPU nodes were visible in the cluster snapshot."]
            gpu_blockers = []
            gpu_status = "complete"

        domains["cpu"] = _make_domain(
            captured_at=captured_at,
            status=cpu_status,
            observed=cpu_observed,
            notes=cpu_notes,
            blockers=cpu_blockers,
        )
        domains["memory"] = _make_domain(
            captured_at=captured_at,
            status=memory_status,
            observed=memory_observed,
            notes=memory_notes,
            blockers=memory_blockers,
        )
        domains["gpu"] = _make_domain(
            captured_at=captured_at,
            status=gpu_status,
            observed=gpu_observed,
            notes=sorted(set(gpu_notes)),
            blockers=gpu_blockers,
        )

    try:
        k8s_version = version_api.get_code().git_version
    except Exception:
        k8s_version = None

    try:
        custom_objects = custom_objects_api.list_cluster_custom_object(
            group="nvidia.com",
            version="v1",
            plural="clusterpolicies",
        )
        gpu_operator_installed = bool((custom_objects or {}).get("items"))
    except ApiException as exc:
        gpu_operator_installed = False if exc.status == 404 else None
    except Exception:
        gpu_operator_installed = None

    if gpu_operator_installed is False:
        try:
            found_ready = False
            for namespace in ("gpu-operator-resources", "nvidia-device-plugin", "kube-system", "gpu-operator"):
                try:
                    daemon_sets = apps_api.list_namespaced_daemon_set(namespace).items or []
                except ApiException as exc:
                    if exc.status in (403, 404):
                        continue
                    raise
                for daemon_set in daemon_sets:
                    name = daemon_set.metadata.name or ""
                    if "nvidia-device-plugin" not in name:
                        continue
                    desired = daemon_set.status.desired_number_scheduled or 0
                    ready = daemon_set.status.number_ready or 0
                    if desired and ready:
                        found_ready = True
                        break
                if found_ready:
                    break
            if found_ready:
                gpu_operator_installed = True
        except Exception:
            pass

    try:
        apiextensions_api.read_custom_resource_definition("wekaappstores.warp.io")
        app_store_crd_installed = True
        try:
            crs = custom_objects_api.list_cluster_custom_object(
                group="warp.io",
                version="v1alpha1",
                plural="wekaappstores",
            )
            applied_crs = [
                f"{(item.get('metadata', {}) or {}).get('namespace', 'default')}/{(item.get('metadata', {}) or {}).get('name')}"
                for item in (crs or {}).get("items", [])
                if (item.get("metadata", {}) or {}).get("name")
            ]
            cluster_init_present = any(entry.endswith("/app-store-cluster-init") for entry in applied_crs)
        except ApiException as exc:
            if exc.status == 404:
                applied_crs = []
                cluster_init_present = False
    except ApiException as exc:
        if exc.status == 404:
            app_store_crd_installed = False
            cluster_init_present = False
    except Exception:
        app_store_crd_installed = None

    return {
        "captured_at": captured_at,
        "domains": domains,
        "k8s_version": k8s_version,
        "gpu_operator_installed": gpu_operator_installed,
        "app_store_crd_installed": app_store_crd_installed,
        "app_store_cluster_init_present": cluster_init_present,
        "app_store_crs": applied_crs,
        "default_storage_class": default_storage_class,
        "default_storage_class_details": default_storage_class_details,
    }


def flatten_cluster_status(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    cpu_observed = snapshot.get("domains", {}).get("cpu", {}).get("observed", {})
    memory_observed = snapshot.get("domains", {}).get("memory", {}).get("observed", {})
    gpu_observed = snapshot.get("domains", {}).get("gpu", {}).get("observed", {})
    namespaces_observed = snapshot.get("domains", {}).get("namespaces", {}).get("observed", {})
    storage_observed = snapshot.get("domains", {}).get("storage_classes", {}).get("observed", {})

    return {
        "cpu_nodes": cpu_observed.get("cpu_nodes"),
        "gpu_nodes": gpu_observed.get("gpu_nodes"),
        "cpu_cores_total": cpu_observed.get("allocatable_cores"),
        "cpu_cores_used": cpu_observed.get("used_cores"),
        "cpu_cores_free": cpu_observed.get("free_cores"),
        "memory_gib_total": memory_observed.get("allocatable_gib"),
        "memory_gib_used": memory_observed.get("used_gib"),
        "memory_gib_free": memory_observed.get("free_gib"),
        "gpu_devices_total": gpu_observed.get("total_devices"),
        "gpu_devices_used": gpu_observed.get("used_devices"),
        "gpu_devices_free": gpu_observed.get("free_devices"),
        "gpu_inventory": gpu_observed.get("inventory", []),
        "visible_namespaces": namespaces_observed.get("names", []),
        "storage_classes": storage_observed.get("names", []),
        "default_storage_class": snapshot.get("default_storage_class"),
        "default_storage_class_details": snapshot.get("default_storage_class_details"),
        "gpu_operator_installed": snapshot.get("gpu_operator_installed"),
        "k8s_version": snapshot.get("k8s_version"),
        "app_store_crd_installed": snapshot.get("app_store_crd_installed"),
        "app_store_cluster_init_present": snapshot.get("app_store_cluster_init_present"),
        "app_store_crs": snapshot.get("app_store_crs", []),
        "inspection_snapshot": snapshot,
    }
