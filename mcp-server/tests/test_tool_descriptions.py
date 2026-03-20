"""Tests that tool descriptions contain required sequencing keywords.

These are keyword assertions on description strings — not NLP. They prove that
the sequencing guidance (cross-tool references, ordering hints, safety warnings)
is present in the descriptions so that an agent can select and sequence tools
correctly using description-based routing.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixture: capture all tool descriptions via registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tool_descriptions() -> dict[str, str]:
    """Return {tool_name: description_str} for all 8 registered tools."""
    from harness.mock_agent import build_tool_registry

    registry = build_tool_registry()
    return {name: entry["description"] for name, entry in registry.items()}


# ---------------------------------------------------------------------------
# Tests: all 8 tools have non-empty descriptions
# ---------------------------------------------------------------------------


EXPECTED_TOOLS = [
    "inspect_cluster",
    "inspect_weka",
    "list_blueprints",
    "get_blueprint",
    "get_crd_schema",
    "validate_yaml",
    "apply",
    "status",
]


def test_all_8_tools_registered(tool_descriptions):
    """All 8 tools must be present in the registry."""
    for tool_name in EXPECTED_TOOLS:
        assert tool_name in tool_descriptions, (
            f"Tool '{tool_name}' missing from registry. "
            f"Registered tools: {list(tool_descriptions.keys())}"
        )


@pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
def test_tool_has_non_empty_description(tool_descriptions, tool_name):
    """Each tool must have a non-empty description string."""
    desc = tool_descriptions.get(tool_name, "")
    assert desc, f"Tool '{tool_name}' has empty description"
    assert len(desc) > 20, (
        f"Tool '{tool_name}' description is suspiciously short ({len(desc)} chars): {desc!r}"
    )


# ---------------------------------------------------------------------------
# Tests: sequencing hints present in specific tool descriptions
# ---------------------------------------------------------------------------


def test_inspect_cluster_mentions_first_and_list_blueprints(tool_descriptions):
    """inspect_cluster must say it's called FIRST and reference list_blueprints."""
    desc = tool_descriptions["inspect_cluster"].lower()
    assert "first" in desc, (
        "inspect_cluster description must mention 'FIRST' to signal it's the starting tool"
    )
    assert "list_blueprints" in desc, (
        "inspect_cluster description must reference list_blueprints for sequencing"
    )


def test_inspect_weka_mentions_inspect_cluster(tool_descriptions):
    """inspect_weka must reference its relationship to inspect_cluster."""
    desc = tool_descriptions["inspect_weka"].lower()
    assert "inspect_cluster" in desc, (
        "inspect_weka description must reference inspect_cluster for ordering context"
    )


def test_list_blueprints_references_inspect_cluster_and_get_blueprint(tool_descriptions):
    """list_blueprints must reference both inspect_cluster (before) and get_blueprint (after)."""
    desc = tool_descriptions["list_blueprints"].lower()
    assert "inspect_cluster" in desc, (
        "list_blueprints description must reference inspect_cluster as the prior step"
    )
    assert "get_blueprint" in desc, (
        "list_blueprints description must reference get_blueprint as the next step"
    )


def test_get_blueprint_references_get_crd_schema(tool_descriptions):
    """get_blueprint must reference get_crd_schema as the next step after it."""
    desc = tool_descriptions["get_blueprint"].lower()
    assert "get_crd_schema" in desc, (
        "get_blueprint description must reference get_crd_schema as the next step"
    )


def test_get_crd_schema_mentions_before_generating_yaml_and_validate(tool_descriptions):
    """get_crd_schema must indicate it should be called before generating YAML."""
    desc = tool_descriptions["get_crd_schema"].lower()
    assert "before" in desc, (
        "get_crd_schema description must include 'before' to indicate ordering"
    )
    # Either "generating yaml" or "generate yaml"
    has_yaml_generation = "generating yaml" in desc or "generate yaml" in desc or "generating" in desc
    assert has_yaml_generation, (
        "get_crd_schema description must mention generating YAML as its purpose"
    )
    assert "validate_yaml" in desc or "validate" in desc, (
        "get_crd_schema description must reference validate step that follows"
    )


def test_validate_yaml_mentions_before_apply(tool_descriptions):
    """validate_yaml must reference apply as what it gates."""
    desc = tool_descriptions["validate_yaml"].lower()
    assert "before apply" in desc or ("before" in desc and "apply" in desc), (
        "validate_yaml description must mention 'before apply' to signal its gate role"
    )
    assert "apply" in desc, (
        "validate_yaml description must reference apply"
    )


def test_apply_mentions_reinspect_and_confirmed(tool_descriptions):
    """apply must mention re-inspect-before-apply and the confirmed parameter."""
    desc = tool_descriptions["apply"].lower()
    # Must mention re-inspect requirement
    has_reinspect = (
        "re-inspect" in desc
        or "re-run inspect_cluster" in desc
        or "inspect_cluster" in desc
    )
    assert has_reinspect, (
        "apply description must mention re-inspect-before-apply requirement"
    )
    # Must mention confirmed parameter requirement
    assert "confirmed" in desc, (
        "apply description must mention the confirmed parameter"
    )


def test_status_mentions_after_apply(tool_descriptions):
    """status must indicate it is called after apply."""
    desc = tool_descriptions["status"].lower()
    assert "after apply" in desc or ("after" in desc and "apply" in desc), (
        "status description must mention 'after apply' to signal its position in the workflow"
    )
