from __future__ import annotations

import importlib

import pytest
import yaml


PHASE_ONE_REQUIREMENTS = {"PLAN-08"}


def test_compiled_fixture_is_single_wekaappstore_resource(
    compiled_wekaappstore_document: dict,
    phase_one_scope_markers: dict[str, set[str]],
) -> None:
    assert PHASE_ONE_REQUIREMENTS.issubset(phase_one_scope_markers["allowed_requirement_ids"])
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


def test_compiler_fixture_stays_within_single_resource_phase_one_scope(
    compiled_wekaappstore_document: dict,
) -> None:
    assert "fitFindings" not in compiled_wekaappstore_document["spec"]
    assert "chatSession" not in compiled_wekaappstore_document["spec"]
    assert "coexistenceAnalysis" not in compiled_wekaappstore_document["spec"]


def test_future_compiler_module_can_target_fixture_shape() -> None:
    compiler = pytest.importorskip("webapp.planning.compiler")

    assert hasattr(compiler, "__file__")
    if hasattr(compiler, "compile_plan_to_wekaappstore"):
        assert callable(compiler.compile_plan_to_wekaappstore)


def test_compiler_generates_the_expected_canonical_document(
    valid_plan_payload: dict,
    compiled_wekaappstore_document: dict,
) -> None:
    compiler = importlib.import_module("webapp.planning.compiler")
    validator = importlib.import_module("webapp.planning.validator")

    validation = validator.validate_structured_plan(valid_plan_payload)

    assert validation.valid is True
    assert validation.plan is not None
    assert compiler.compile_plan_to_wekaappstore(validation.plan) == compiled_wekaappstore_document


def test_compiler_renders_stable_yaml_for_equivalent_valid_plans(
    valid_plan_payload: dict,
    warning_plan_payload: dict,
) -> None:
    compiler = importlib.import_module("webapp.planning.compiler")

    _, base_document, base_yaml = compiler.validate_and_compile_plan(valid_plan_payload)
    validation, warning_document, warning_yaml = compiler.validate_and_compile_plan(warning_plan_payload)

    assert validation.valid is True
    assert len(validation.warnings) == 2
    assert base_document == warning_document
    assert base_yaml == warning_yaml
    assert list(yaml.safe_load(base_yaml)["spec"]["appStack"]["components"][0]["helmChart"]) == [
        "repository",
        "name",
        "version",
        "releaseName",
        "crdsStrategy",
    ]


def test_compiler_refuses_to_compile_invalid_structured_plans(
    invalid_plan_payloads: dict[str, dict],
) -> None:
    compiler = importlib.import_module("webapp.planning.compiler")

    validation, document, rendered_yaml = compiler.validate_and_compile_plan(
        invalid_plan_payloads["blocking_unresolved_question"]
    )

    assert validation.valid is False
    assert document is None
    assert rendered_yaml is None


def test_compiler_emits_a_single_yaml_document(valid_plan_payload: dict) -> None:
    compiler = importlib.import_module("webapp.planning.compiler")

    _, _, rendered_yaml = compiler.validate_and_compile_plan(valid_plan_payload)

    assert rendered_yaml is not None
    assert list(yaml.safe_load_all(rendered_yaml))[0]["kind"] == "WekaAppStore"
    assert len(list(yaml.safe_load_all(rendered_yaml))) == 1
