"""Blueprint scanner and tools for the WEKA App Store MCP server.

Provides:
  - scan_blueprints(): Directory-driven scanner for WekaAppStore YAML manifests
  - flatten_blueprint_summary(): Flat per-entry metadata for list_blueprints
  - flatten_blueprint_detail(): Full flat detail for get_blueprint
  - register_blueprint_tools(): Registers list_blueprints and get_blueprint with FastMCP
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import config

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def scan_blueprints(blueprints_dir: str) -> list[dict[str, Any]]:
    """Scan a directory recursively for WekaAppStore YAML manifests.

    Filters for documents where:
      - apiVersion starts with 'warp.io'
      - kind == 'WekaAppStore'

    Args:
        blueprints_dir: Absolute path to scan. Missing directory returns [].

    Returns:
        List of {"source_file": str, "manifest": dict} — internal use only.
    """
    dir_path = Path(blueprints_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("BLUEPRINTS_DIR not found or not a directory: %s", blueprints_dir)
        return []

    results: list[dict[str, Any]] = []

    yaml_files = list(dir_path.rglob("*.yaml")) + list(dir_path.rglob("*.yml"))

    for yaml_file in sorted(yaml_files):
        try:
            with yaml_file.open("r", encoding="utf-8") as fh:
                docs = list(yaml.safe_load_all(fh))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", yaml_file, exc)
            continue

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            api_version = doc.get("apiVersion", "")
            kind = doc.get("kind", "")
            if api_version.startswith("warp.io") and kind == "WekaAppStore":
                results.append({"source_file": str(yaml_file), "manifest": doc})

    return results


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------


def flatten_blueprint_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract flat metadata from a scanned blueprint entry.

    Fields returned (all at top level, <=2 key traversal from response root):
      name, namespace, component_count, component_names, source_file

    Args:
        entry: {"source_file": str, "manifest": dict}

    Returns:
        Flat dict with blueprint summary metadata.
    """
    manifest = entry["manifest"]
    meta = manifest.get("metadata", {})
    spec = manifest.get("spec", {})
    components = spec.get("appStack", {}).get("components", [])

    return {
        "name": meta.get("name", ""),
        "namespace": meta.get("namespace", "default"),
        "component_count": len(components),
        "component_names": [c.get("name", "") for c in components],
        "source_file": Path(entry["source_file"]).name,
    }


def flatten_blueprint_detail(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract full flat detail from a scanned blueprint entry.

    Each component dict is flat — helm_chart sub-dict is flattened to top-level
    helm_chart_* fields. All values reachable in <=2 key traversals.

    Args:
        entry: {"source_file": str, "manifest": dict}

    Returns:
        Flat dict with full blueprint detail.
    """
    manifest = entry["manifest"]
    meta = manifest.get("metadata", {})
    spec = manifest.get("spec", {})
    components_raw = spec.get("appStack", {}).get("components", [])
    prerequisites = spec.get("prerequisites", [])

    flat_components = []
    for c in components_raw:
        helm = c.get("helm_chart", {})
        flat_components.append({
            "name": c.get("name", ""),
            "enabled": c.get("enabled", True),
            "target_namespace": c.get("target_namespace", ""),
            "depends_on": c.get("depends_on", []),
            "helm_chart_name": helm.get("name", ""),
            "helm_chart_version": helm.get("version", ""),
            "helm_chart_repository": helm.get("repository", ""),
            "helm_chart_release_name": helm.get("release_name", ""),
            "wait_for_ready": c.get("wait_for_ready", False),
        })

    return {
        "name": meta.get("name", ""),
        "namespace": meta.get("namespace", "default"),
        "api_version": manifest.get("apiVersion", ""),
        "captured_at": _utc_now(),
        "components": flat_components,
        "prerequisites": prerequisites if isinstance(prerequisites, list) else [],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_blueprint_tools(mcp: Any) -> None:
    """Register list_blueprints and get_blueprint tools with the FastMCP instance."""

    @mcp.tool()
    def list_blueprints() -> dict:
        """Call this after inspect_cluster to discover available blueprints. Returns
        a catalog of all WekaAppStore blueprint manifests with their names, component
        counts, and component names. Cross-reference the cluster resources from
        inspect_cluster to assess which blueprints fit.

        Use the blueprint 'name' field from this list to call get_blueprint for full
        specification details.

        Returns: captured_at, count, blueprints (list of {name, namespace,
        component_count, component_names, source_file}), warnings.

        Sequencing: inspect_cluster -> list_blueprints -> get_blueprint -> get_crd_schema.
        """
        entries = scan_blueprints(config.BLUEPRINTS_DIR)
        warnings: list[str] = []

        blueprints = [flatten_blueprint_summary(e) for e in entries]

        if not blueprints:
            warnings.append(
                "No blueprints found in BLUEPRINTS_DIR — directory may be empty or not yet synced"
            )

        return {
            "captured_at": _utc_now(),
            "count": len(blueprints),
            "blueprints": blueprints,
            "warnings": warnings,
        }

    @mcp.tool()
    def get_blueprint(name: str) -> dict:
        """Call this after list_blueprints to retrieve full specification for a named
        blueprint. Returns all components (flat, with helm_chart_* fields), target
        namespaces, dependencies between components, prerequisites that must exist in
        the cluster, and any warnings.

        Read the prerequisites list carefully — some blueprints require existing
        operator installations or cluster configuration. After reading the full spec,
        call get_crd_schema to obtain the CRD schema before generating YAML.

        Returns: name, namespace, api_version, components (list), prerequisites
        (list), warnings.

        Sequencing: list_blueprints -> get_blueprint -> get_crd_schema ->
        (generate YAML) -> validate_yaml.
        """
        entries = scan_blueprints(config.BLUEPRINTS_DIR)

        for entry in entries:
            entry_name = entry["manifest"].get("metadata", {}).get("name", "")
            if entry_name == name:
                return flatten_blueprint_detail(entry)

        # Not found — return structured error with available names
        available_names = [
            e["manifest"].get("metadata", {}).get("name", "")
            for e in entries
        ]
        return {
            "captured_at": _utc_now(),
            "error": "Blueprint not found",
            "requested_name": name,
            "available_names": available_names,
            "warnings": [],
        }
