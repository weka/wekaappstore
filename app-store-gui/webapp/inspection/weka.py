from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from kubernetes import client
from kubernetes.client.rest import ApiException


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_blocker(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message, "domain": "weka"}


def _read_nested(mapping: Dict[str, Any], path: List[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _filesystem_inventory(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    for path in (["filesystems"], ["stats", "filesystems"], ["filesystemStatus", "filesystems"]):
        items = _read_nested(status, path)
        if isinstance(items, list):
            inventory = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                inventory.append(
                    {
                        "name": item.get("name"),
                        "total_bytes": item.get("totalBytes") or item.get("total_bytes"),
                        "free_bytes": item.get("availableBytes")
                        or item.get("freeBytes")
                        or item.get("free_bytes"),
                    }
                )
            if inventory:
                return inventory
    return []


def collect_weka_inspection(
    *,
    load_kube_config: Optional[Callable[[], None]] = None,
    custom_objects_api: Any = None,
) -> Dict[str, Any]:
    if load_kube_config is not None:
        load_kube_config()

    custom_objects_api = custom_objects_api or client.CustomObjectsApi()
    captured_at = _utc_timestamp()

    try:
        response = custom_objects_api.list_cluster_custom_object(
            group="weka.weka.io",
            version="v1alpha1",
            plural="wekaclusters",
        )
    except ApiException as exc:
        status = "unavailable" if exc.status == 404 else "partial"
        return {
            "captured_at": captured_at,
            "domains": {
                "weka": {
                    "status": status,
                    "required": True,
                    "freshness": {"captured_at": captured_at, "max_age_seconds": 300},
                    "observed": {"clusters": [], "filesystems": []},
                    "notes": [],
                    "blockers": [_make_blocker("weka_inspection_failed", f"Unable to inspect WEKA resources: {exc}")],
                }
            },
        }
    except Exception as exc:
        return {
            "captured_at": captured_at,
            "domains": {
                "weka": {
                    "status": "partial",
                    "required": True,
                    "freshness": {"captured_at": captured_at, "max_age_seconds": 300},
                    "observed": {"clusters": [], "filesystems": []},
                    "notes": [],
                    "blockers": [_make_blocker("weka_inspection_failed", str(exc))],
                }
            },
        }

    items = (response or {}).get("items", [])
    if not items:
        return {
            "captured_at": captured_at,
            "domains": {
                "weka": {
                    "status": "unavailable",
                    "required": True,
                    "freshness": {"captured_at": captured_at, "max_age_seconds": 300},
                    "observed": {"clusters": [], "filesystems": []},
                    "notes": [],
                    "blockers": [_make_blocker("weka_cluster_missing", "No WekaCluster resources were visible.")],
                }
            },
        }

    clusters: List[Dict[str, Any]] = []
    filesystems: List[Dict[str, Any]] = []
    blockers: List[Dict[str, str]] = []
    notes: List[str] = []
    total_capacity_bytes = 0
    free_capacity_bytes = 0

    for item in items:
        metadata = item.get("metadata", {}) or {}
        status = item.get("status", {}) or {}
        cluster_total = _read_nested(status, ["stats", "capacity", "totalBytes"])
        cluster_free = _read_nested(status, ["stats", "filesystem", "totalAvailableCapacity"])
        cluster_entry = {
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "status": status.get("status"),
            "cluster_total_bytes": cluster_total,
            "cluster_free_bytes": cluster_free,
            "filesystem_capacity": _read_nested(status, ["printer", "filesystemCapacity"]),
        }
        clusters.append(cluster_entry)

        cluster_total_int = _coerce_int(cluster_total)
        cluster_free_int = _coerce_int(cluster_free)

        if cluster_total_int is not None:
            total_capacity_bytes += cluster_total_int
        else:
            blockers.append(_make_blocker("weka_capacity_missing", f"WekaCluster {metadata.get('name')} is missing total capacity bytes."))

        if cluster_free_int is not None:
            free_capacity_bytes += cluster_free_int
        else:
            blockers.append(_make_blocker("weka_free_capacity_missing", f"WekaCluster {metadata.get('name')} is missing free capacity bytes."))

        cluster_filesystems = _filesystem_inventory(status)
        if cluster_filesystems:
            filesystems.extend(cluster_filesystems)
        else:
            notes.append(f"WekaCluster {metadata.get('name')} does not expose filesystem inventory in operator-visible status.")
            blockers.append(
                _make_blocker(
                    "weka_filesystems_missing",
                    f"WekaCluster {metadata.get('name')} is missing filesystem inventory.",
                )
            )

    if blockers:
        domain_status = "partial"
    else:
        domain_status = "complete"

    return {
        "captured_at": captured_at,
        "domains": {
            "weka": {
                "status": domain_status,
                "required": True,
                "freshness": {"captured_at": captured_at, "max_age_seconds": 300},
                "observed": {
                    "clusters": clusters,
                    "filesystems": filesystems,
                    "cluster_total_bytes": total_capacity_bytes if total_capacity_bytes else None,
                    "cluster_free_bytes": free_capacity_bytes if free_capacity_bytes else None,
                },
                "notes": notes,
                "blockers": blockers,
            }
        },
    }
