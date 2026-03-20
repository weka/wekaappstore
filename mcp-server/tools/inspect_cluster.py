"""inspect_cluster MCP tool.

Exposes cluster resource inspection as a flat MCP tool response.
Reuses collect_cluster_inspection() from webapp.inspection.cluster.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def flatten_inspect_cluster_for_mcp(snapshot: dict) -> dict:
    """Flatten collect_cluster_inspection() output to a flat MCP tool response.

    Key differences from flatten_cluster_status():
    - No 'inspection_snapshot' key
    - GPU inventory aggregated to gpu_total, gpu_models, gpu_memory_total_gib
    - Separate 'warnings' array from all blockers
    - All values reachable in <= 2 key traversals
    """
    cpu = snapshot.get("domains", {}).get("cpu", {}).get("observed", {})
    memory = snapshot.get("domains", {}).get("memory", {}).get("observed", {})
    gpu_domain = snapshot.get("domains", {}).get("gpu", {})
    gpu = gpu_domain.get("observed", {})
    namespaces = snapshot.get("domains", {}).get("namespaces", {}).get("observed", {})
    storage = snapshot.get("domains", {}).get("storage_classes", {}).get("observed", {})

    # Collect warnings from all domain blockers
    warnings: list[str] = []
    for domain in snapshot.get("domains", {}).values():
        for blocker in domain.get("blockers", []):
            msg = blocker.get("message", "")
            if msg:
                warnings.append(msg)

    if snapshot.get("gpu_operator_installed") is False:
        warnings.append("GPU operator not detected — GPU workloads may not schedule")

    # Aggregate GPU inventory
    inventory = gpu.get("inventory", [])
    gpu_models = [item.get("model") for item in inventory if item.get("model")]
    gpu_total = sum(item.get("count", 0) for item in inventory)
    gpu_memory_total_gib = sum(
        (item.get("count", 0) * (item.get("memory_gib") or 0))
        for item in inventory
    )

    return {
        "captured_at": snapshot.get("captured_at"),
        "k8s_version": snapshot.get("k8s_version"),
        "cpu_nodes": cpu.get("cpu_nodes"),
        "gpu_nodes": gpu.get("gpu_nodes"),
        "cpu_cores_total": cpu.get("allocatable_cores"),
        "cpu_cores_free": cpu.get("free_cores"),
        "memory_gib_total": memory.get("allocatable_gib"),
        "memory_gib_free": memory.get("free_gib"),
        "gpu_total": gpu_total,
        "gpu_models": gpu_models,
        "gpu_memory_total_gib": round(gpu_memory_total_gib, 2),
        "gpu_operator_installed": snapshot.get("gpu_operator_installed"),
        "visible_namespaces": namespaces.get("names", []),
        "storage_classes": storage.get("names", []),
        "default_storage_class": snapshot.get("default_storage_class"),
        "app_store_crd_installed": snapshot.get("app_store_crd_installed"),
        "app_store_cluster_init_present": snapshot.get("app_store_cluster_init_present"),
        "app_store_crs": snapshot.get("app_store_crs", []),
        "warnings": [w for w in warnings if w],
    }


def register_inspect_cluster(mcp: Any) -> None:
    """Register the inspect_cluster tool with the given FastMCP instance."""

    @mcp.tool()
    def inspect_cluster() -> dict:
        """Call this tool FIRST when you need to understand what cluster resources are
        available before blueprint selection. Returns a flat snapshot of CPU cores,
        memory, GPU devices, namespaces, and storage classes. Call before
        list_blueprints to know which blueprints can fit the cluster. Call again after
        time passes to refresh — results are not cached.

        Sequencing: inspect_cluster -> list_blueprints -> get_blueprint ->
        validate_yaml -> apply.
        """
        try:
            from webapp.inspection.cluster import collect_cluster_inspection
            snapshot = collect_cluster_inspection()
            return flatten_inspect_cluster_for_mcp(snapshot)
        except ApiException as exc:
            logger.warning("K8s API error in inspect_cluster: %s", exc)
            return {
                "captured_at": _utc_now(),
                "k8s_version": None,
                "cpu_nodes": None,
                "gpu_nodes": None,
                "cpu_cores_total": None,
                "cpu_cores_free": None,
                "memory_gib_total": None,
                "memory_gib_free": None,
                "gpu_total": 0,
                "gpu_models": [],
                "gpu_memory_total_gib": 0.0,
                "gpu_operator_installed": None,
                "visible_namespaces": [],
                "storage_classes": [],
                "default_storage_class": None,
                "app_store_crd_installed": None,
                "app_store_cluster_init_present": None,
                "app_store_crs": [],
                "warnings": [f"K8s API unavailable: {exc.status} {exc.reason}"],
            }
