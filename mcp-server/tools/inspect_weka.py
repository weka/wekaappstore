"""inspect_weka MCP tool.

Exposes WEKA storage inspection as a flat MCP tool response.
Reuses collect_weka_inspection() from webapp.inspection.weka.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bytes_to_gib(value: Any) -> float:
    """Convert bytes integer to GiB, rounded to 2 decimal places."""
    try:
        if value is None:
            return 0.0
        return round(int(value) / float(1024 ** 3), 2)
    except (TypeError, ValueError):
        return 0.0


def flatten_inspect_weka_for_mcp(snapshot: dict) -> dict:
    """Flatten collect_weka_inspection() output to a flat MCP tool response.

    Strips the domains.weka wrapper and promotes observed fields to top-level.
    All values are reachable in <= 2 key traversals.
    """
    weka_domain = snapshot.get("domains", {}).get("weka", {})
    observed = weka_domain.get("observed", {})
    blockers = weka_domain.get("blockers", [])

    # Collect warnings from blockers
    warnings: list[str] = [
        b.get("message", "") for b in blockers if b.get("message")
    ]

    # Determine primary cluster name from the clusters list
    clusters = observed.get("clusters", [])
    weka_cluster_name = clusters[0].get("name") if clusters else None
    weka_cluster_status = clusters[0].get("status") if clusters else None

    # Capacity in GiB
    total_bytes = observed.get("cluster_total_bytes")
    free_bytes = observed.get("cluster_free_bytes")
    total_gib = _bytes_to_gib(total_bytes)
    free_gib = _bytes_to_gib(free_bytes)
    used_gib = round(total_gib - free_gib, 2) if (total_bytes and free_bytes) else 0.0

    # Flatten filesystem list to {name, size_gib, used_gib}
    raw_filesystems = observed.get("filesystems", [])
    filesystems = []
    for fs in raw_filesystems:
        if not isinstance(fs, dict):
            continue
        fs_total = fs.get("total_bytes")
        fs_free = fs.get("free_bytes")
        fs_total_gib = _bytes_to_gib(fs_total)
        fs_used_gib = round(fs_total_gib - _bytes_to_gib(fs_free), 2) if (fs_total and fs_free) else 0.0
        filesystems.append({
            "name": fs.get("name"),
            "size_gib": fs_total_gib,
            "used_gib": fs_used_gib,
        })

    return {
        "captured_at": snapshot.get("captured_at"),
        "weka_cluster_name": weka_cluster_name,
        "weka_cluster_status": weka_cluster_status,
        "total_capacity_gib": total_gib,
        "used_capacity_gib": used_gib,
        "free_capacity_gib": free_gib,
        "filesystems": filesystems,
        "warnings": warnings,
    }


def register_inspect_weka(mcp: Any) -> None:
    """Register the inspect_weka tool with the given FastMCP instance."""

    @mcp.tool()
    def inspect_weka() -> dict:
        """Call this tool after inspect_cluster when the blueprint requires WEKA
        filesystems or the cluster uses a 'wekafs' storage class. Returns WEKA cluster
        capacity (total, used, free in GiB), per-filesystem breakdown, and any
        warnings about WEKA availability.

        Skip this tool only if the blueprint has no WEKA storage requirements and the
        storage class is not wekafs.

        Returns: captured_at, weka_cluster_name, weka_cluster_status,
        total_capacity_gib, free_capacity_gib, filesystems (list), warnings.

        Sequencing: inspect_cluster -> inspect_weka -> list_blueprints.
        """
        try:
            from webapp.inspection.weka import collect_weka_inspection
            snapshot = collect_weka_inspection()
            return flatten_inspect_weka_for_mcp(snapshot)
        except ApiException as exc:
            logger.warning("K8s API error in inspect_weka: %s", exc)
            return {
                "captured_at": _utc_now(),
                "weka_cluster_name": None,
                "weka_cluster_status": None,
                "total_capacity_gib": 0.0,
                "used_capacity_gib": 0.0,
                "free_capacity_gib": 0.0,
                "filesystems": [],
                "warnings": [f"K8s API unavailable: {exc.status} {exc.reason}"],
            }
