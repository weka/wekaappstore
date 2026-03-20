"""get_crd_schema MCP tool.

Reads the WekaAppStore CRD from the live cluster and returns its OpenAPI v3
schema plus 1-2 example manifests from the blueprint catalog.

Provides:
  - _get_crd_schema_impl(): Testable implementation with injectable API client
  - register_crd_schema(): Registers get_crd_schema tool with FastMCP instance
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import yaml
from kubernetes.client.rest import ApiException

import config
from tools.blueprints import scan_blueprints

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _get_crd_schema_impl(
    apiextensions_api: Any = None,
    blueprints_dir: str | None = None,
) -> dict:
    """Core implementation of get_crd_schema — injectable for testing.

    Args:
        apiextensions_api: kubernetes.client.ApiextensionsV1Api instance.
            If None, creates a real one (with config loaded).
        blueprints_dir: Path to directory containing blueprint YAML files.
            Defaults to config.BLUEPRINTS_DIR.

    Returns:
        Flat dict with captured_at, group, version, kind, schema, examples, warnings.
    """
    if blueprints_dir is None:
        blueprints_dir = config.BLUEPRINTS_DIR

    warnings: list[str] = []

    # --- Read CRD from cluster -------------------------------------------
    if apiextensions_api is None:
        from kubernetes import client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception:
                pass
        apiextensions_api = client.ApiextensionsV1Api()

    group: str | None = None
    version: str | None = None
    kind: str | None = None
    schema: dict | None = None

    try:
        crd = apiextensions_api.read_custom_resource_definition("wekaappstores.warp.io")
        group = crd.spec.group
        kind = crd.spec.names.kind

        # Extract schema from the first version entry
        versions = crd.spec.versions or []
        if versions:
            ver_entry = versions[0]
            version = ver_entry.name
            open_api_schema = ver_entry.schema.open_apiv3_schema
            if hasattr(open_api_schema, "to_dict"):
                schema = open_api_schema.to_dict()
            elif isinstance(open_api_schema, dict):
                schema = open_api_schema
            else:
                schema = {}

    except ApiException as exc:
        if exc.status == 404:
            warnings.append(
                "WekaAppStore CRD not installed — run 'kubectl apply' to install the operator first"
            )
        else:
            warnings.append(
                f"K8s API unavailable: {exc.status} {exc.reason} — cannot read CRD schema"
            )
        return {
            "captured_at": _utc_now(),
            "group": None,
            "version": None,
            "kind": None,
            "schema": None,
            "examples": [],
            "warnings": warnings,
        }

    # --- Extract examples from blueprints dir ----------------------------
    entries = scan_blueprints(blueprints_dir)
    examples: list[str] = []
    for entry in entries[:2]:
        manifest = entry.get("manifest", {})
        if manifest:
            try:
                examples.append(yaml.dump(manifest, default_flow_style=False, allow_unicode=True))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to dump blueprint manifest to YAML: %s", exc)

    if not examples:
        warnings.append(
            "No example manifests found — BLUEPRINTS_DIR may be empty or not yet synced"
        )

    return {
        "captured_at": _utc_now(),
        "group": group,
        "version": version,
        "kind": kind,
        "schema": schema,
        "examples": examples,
        "warnings": warnings,
    }


def register_crd_schema(mcp: Any) -> None:
    """Register the get_crd_schema tool with the given FastMCP instance."""

    @mcp.tool()
    def get_crd_schema() -> dict:
        """Call this tool when you need to generate WekaAppStore YAML from scratch.
        Returns the CRD OpenAPI v3 schema defining valid fields and structure, plus
        1-2 example manifests from the blueprint catalog. Use the schema to validate
        your YAML structure and the examples to pattern-match correct formatting.

        Sequencing: get_blueprint -> get_crd_schema -> (generate YAML) -> validate_yaml.
        """
        return _get_crd_schema_impl()
