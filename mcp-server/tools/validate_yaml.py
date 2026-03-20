"""validate_yaml MCP tool.

Validates WekaAppStore YAML documents against the CRD contract before applying.

Provides:
  - _validate_yaml_impl(): Testable implementation, no K8s needed (pure YAML parse)
  - register_validate_yaml(): Registers validate_yaml tool with FastMCP instance
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants — CRD contract
# ---------------------------------------------------------------------------

# v1.0 StructuredPlan model fields that must never appear in WekaAppStore spec.
# These are snake_case fields from the planning model; the CRD uses camelCase.
_V1_ONLY_SPEC_FIELDS: frozenset[str] = frozenset({
    "blueprint_family",
    "fit_findings",
    "namespace_strategy",
    "reasoning_summary",
    "request_summary",
    "unresolved_questions",
})

# Accepted apiVersion values
_VALID_API_VERSIONS: frozenset[str] = frozenset({"warp.io/v1alpha1"})

# Accepted kind values
_VALID_KINDS: frozenset[str] = frozenset({"WekaAppStore"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _validate_yaml_impl(yaml_text: str) -> dict:
    """Validate WekaAppStore YAML against CRD contract — injectable for testing.

    Args:
        yaml_text: YAML string containing one or more documents.

    Returns:
        Dict with: captured_at, valid (bool), errors (list of {code, path, message}),
        warnings (list of str).
    """
    errors: list[dict] = []
    warnings: list[str] = []

    # 1. Parse YAML — catch syntax errors immediately
    try:
        docs = list(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError as exc:
        return {
            "captured_at": _utc_now(),
            "valid": False,
            "errors": [{"code": "yaml_parse_error", "path": "$", "message": str(exc)}],
            "warnings": [],
        }

    # 2. Filter to WekaAppStore documents by kind
    weka_docs = [d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"]
    if not weka_docs:
        errors.append({
            "code": "no_wekaappstore_doc",
            "path": "$",
            "message": "No WekaAppStore document found in YAML",
        })

    # 3. Validate each WekaAppStore document
    for i, doc in enumerate(weka_docs):
        path_prefix = f"doc[{i}]"

        # Check apiVersion
        api_version = doc.get("apiVersion")
        if api_version not in _VALID_API_VERSIONS:
            errors.append({
                "code": "invalid_api_version",
                "path": f"{path_prefix}.apiVersion",
                "message": f"apiVersion must be one of {sorted(_VALID_API_VERSIONS)}",
            })

        # Check metadata.name
        metadata = doc.get("metadata")
        name = (metadata or {}).get("name") if isinstance(metadata, dict) else None
        if not name:
            errors.append({
                "code": "missing_required_field",
                "path": f"{path_prefix}.metadata.name",
                "message": "metadata.name is required",
            })

        # Check for v1.0-only snake_case fields in spec
        spec = doc.get("spec") or {}
        if isinstance(spec, dict):
            for v1_field in sorted(_V1_ONLY_SPEC_FIELDS):
                if v1_field in spec:
                    errors.append({
                        "code": "v1_only_field",
                        "path": f"{path_prefix}.spec.{v1_field}",
                        "message": (
                            f"Field '{v1_field}' is a v1.0 planning model field and "
                            "is not valid in WekaAppStore spec"
                        ),
                    })

            # Check that at least one deployment method is present
            has_deployment = any(k in spec for k in ("helmChart", "appStack", "image"))
            if not has_deployment:
                errors.append({
                    "code": "missing_deployment_method",
                    "path": f"{path_prefix}.spec",
                    "message": "spec must include at least one of: helmChart, appStack, image",
                })

    return {
        "captured_at": _utc_now(),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# FastMCP registration
# ---------------------------------------------------------------------------


def register_validate_yaml(mcp: Any) -> None:
    """Register the validate_yaml tool with the given FastMCP instance."""

    @mcp.tool()
    def validate_yaml(yaml_text: str) -> dict:
        """Call this tool BEFORE apply to verify that generated WekaAppStore YAML
        is structurally valid per the CRD contract.

        Checks: correct apiVersion (warp.io/v1alpha1), kind (WekaAppStore),
        metadata.name present, no v1.0-only planning fields (blueprint_family,
        fit_findings, namespace_strategy, etc.), and at least one deployment
        method (helmChart, appStack, or image).

        Returns valid=true with empty errors on success, or valid=false with
        a structured errors list on failure. Each error has a code, path, and
        message field to help the agent pinpoint and fix the issue.

        Sequencing: get_crd_schema -> (generate YAML) -> validate_yaml -> apply.
        Do not call apply if validate_yaml returns valid=false.
        """
        return _validate_yaml_impl(yaml_text)
