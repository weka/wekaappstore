from __future__ import annotations

import importlib

import pytest
import yaml


PHASE_ONE_REQUIREMENTS = {"PLAN-08"}


def test_compiled_fixture_is_single_wekaappstore_resource(
    compiled_wekaappstore_document: dict,
) -> None:
    assert PHASE_ONE_REQUIREMENTS == {"PLAN-08"}
    assert compiled_wekaappstore_document["apiVersion"] == "warp.io/v1alpha1"
    assert compiled_wekaappstore_document["kind"] == "WekaAppStore"
    assert compiled_wekaappstore_document["metadata"]["namespace"] == "ai-platform"
    components = compiled_wekaappstore_document["spec"]["appStack"]["components"]
    assert [component["name"] for component in components] == ["vector-db", "research-api"]


def test_compiled_yaml_fixture_round_trips_with_stable_runtime_fields(
    compiled_wekaappstore_document: dict,
    compiled_wekaappstore_yaml: str,
) -> None:
    loaded_document = yaml.safe_load(compiled_wekaappstore_yaml)

    assert loaded_document == compiled_wekaappstore_document
    assert list(loaded_document) == ["apiVersion", "kind", "metadata", "spec"]
    assert list(loaded_document["spec"]["appStack"]) == ["defaultNamespace", "components"]


def test_future_compiler_module_can_target_fixture_shape() -> None:
    compiler = pytest.importorskip(
        "webapp.planning.compiler",
        reason="Phase 1 compiler implementation has not landed yet.",
    )

    assert hasattr(compiler, "__file__")
    if hasattr(compiler, "compile_plan_to_wekaappstore"):
        assert callable(compiler.compile_plan_to_wekaappstore)
