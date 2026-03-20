from __future__ import annotations

import importlib

import pytest


PHASE_ONE_REQUIREMENTS = {"PLAN-06", "PLAN-07"}
OUT_OF_SCOPE_TOPICS = {"chat", "cluster_inspection", "weka_inspection", "coexistence"}


def _validator_module():
    return importlib.import_module("webapp.planning.validator")


def test_valid_plan_fixture_matches_phase_one_contract(
    valid_plan_payload: dict,
    phase_one_scope_markers: dict[str, set[str]],
) -> None:
    assert PHASE_ONE_REQUIREMENTS.issubset(phase_one_scope_markers["allowed_requirement_ids"])
    assert valid_plan_payload["blueprint_family"] == "ai-agent-enterprise-research"
    assert valid_plan_payload["namespace_strategy"]["target_namespace"] == "ai-platform"
    assert valid_plan_payload["unresolved_questions"] == []
    assert {component["name"] for component in valid_plan_payload["components"]} == {
        "vector-db",
        "research-api",
    }


def test_invalid_plan_variants_cover_deterministic_validation_failures(
    invalid_plan_payloads: dict[str, dict],
) -> None:
    assert set(invalid_plan_payloads) == {
        "blocking_unresolved_question",
        "both_deployment_methods",
        "dependency_on_missing_component",
        "duplicate_component_names",
        "missing_blueprint_family",
        "missing_deployment_method",
    }
    assert "blueprint_family" not in invalid_plan_payloads["missing_blueprint_family"]
    assert invalid_plan_payloads["duplicate_component_names"]["components"][0]["name"] == (
        invalid_plan_payloads["duplicate_component_names"]["components"][1]["name"]
    )
    assert invalid_plan_payloads["dependency_on_missing_component"]["components"][1]["depends_on"] == [
        "missing-component"
    ]
    assert "kubernetes_manifest" not in invalid_plan_payloads["missing_deployment_method"]["components"][1]
    assert "helm_chart" in invalid_plan_payloads["both_deployment_methods"]["components"][1]
    assert invalid_plan_payloads["blocking_unresolved_question"]["unresolved_questions"][0]["blocking"] is True


def test_warning_fixture_only_exercises_safe_normalization(warning_case_payload: dict) -> None:
    assert warning_case_payload["expected_normalization_warnings"] == [
        {
            "path": "components[0].helm_chart.release_name",
            "message": "release_name defaults to the component name when omitted",
        },
        {
            "path": "components[0].wait_for_ready",
            "message": "wait_for_ready defaults to true when omitted",
        },
    ]
    first_component = warning_case_payload["components"][0]
    assert "release_name" not in first_component["helm_chart"]
    assert "wait_for_ready" not in first_component


def test_validator_accepts_a_valid_structured_plan_without_rewriting_explicit_intent(
    valid_plan_payload: dict,
) -> None:
    result = _validator_module().validate_structured_plan(valid_plan_payload)

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.plan is not None
    assert result.plan.blueprint_family == "ai-agent-enterprise-research"
    assert result.plan.namespace_strategy.target_namespace == "ai-platform"
    assert [component.name for component in result.plan.components] == ["vector-db", "research-api"]
    assert result.plan.components[0].helm_chart.release_name == "vector-db"
    assert result.plan.components[1].depends_on == ["vector-db"]


def test_validator_returns_explicit_normalization_warnings_for_safe_defaults(
    warning_case_payload: dict,
) -> None:
    validator = _validator_module()
    payload = {
        key: value
        for key, value in warning_case_payload.items()
        if key != "expected_normalization_warnings"
    }

    result = validator.validate_structured_plan(payload)

    assert result.valid is True
    assert result.errors == []
    assert [
        {"path": warning.path, "message": warning.message}
        for warning in result.warnings
    ] == warning_case_payload["expected_normalization_warnings"]
    assert result.plan is not None
    assert result.plan.components[0].helm_chart.release_name == "vector-db"
    assert result.plan.components[0].wait_for_ready is True


@pytest.mark.parametrize(
    ("case_name", "expected_code", "expected_path"),
    [
        ("missing_blueprint_family", "missing_required_field", "blueprint_family"),
        ("duplicate_component_names", "duplicate_component_name", "components[1].name"),
        ("dependency_on_missing_component", "unknown_dependency", "components[1].depends_on"),
        ("missing_deployment_method", "invalid_deployment_method", "components[1]"),
        ("both_deployment_methods", "invalid_deployment_method", "components[1]"),
        ("blocking_unresolved_question", "blocking_unresolved_question", "unresolved_questions[0]"),
    ],
)
def test_validator_rejects_deterministic_contract_failures(
    invalid_plan_payloads: dict[str, dict],
    case_name: str,
    expected_code: str,
    expected_path: str,
) -> None:
    result = _validator_module().validate_structured_plan(invalid_plan_payloads[case_name])

    assert result.valid is False
    assert result.plan is None
    assert (expected_code, expected_path) in {
        (error.code, error.path) for error in result.errors
    }
    assert result.errors == sorted(
        result.errors, key=lambda error: (error.path, error.code, error.message)
    )


def test_plan_contract_suite_excludes_later_phase_topics(
    phase_one_scope_markers: dict[str, set[str]],
) -> None:
    assert OUT_OF_SCOPE_TOPICS == phase_one_scope_markers["excluded_topics"]


def test_models_export_supported_phase_one_contract_constants() -> None:
    planning_models = importlib.import_module("webapp.planning.models")

    assert planning_models.DEFAULT_CRD_API_VERSION == "warp.io/v1alpha1"
    assert planning_models.DEFAULT_CRD_KIND == "WekaAppStore"
    assert planning_models.SUPPORTED_BLUEPRINT_FAMILIES == {
        "ai-agent-enterprise-research",
        "nvidia-vss",
        "openfold",
    }
    assert planning_models.SUPPORTED_READINESS_CHECK_TYPES == {
        "pod",
        "deployment",
        "statefulset",
        "job",
        "custom",
    }


def test_models_can_represent_a_valid_phase_one_structured_plan(valid_plan_payload: dict) -> None:
    planning_models = importlib.import_module("webapp.planning.models")

    first_component = valid_plan_payload["components"][0]
    second_component = valid_plan_payload["components"][1]
    plan = planning_models.StructuredPlan(
        request_summary=valid_plan_payload["request_summary"],
        blueprint_family=valid_plan_payload["blueprint_family"],
        namespace_strategy=planning_models.NamespaceStrategy(**valid_plan_payload["namespace_strategy"]),
        components=[
            planning_models.ComponentPlan(
                name=first_component["name"],
                enabled=first_component["enabled"],
                depends_on=first_component["depends_on"],
                target_namespace=first_component["target_namespace"],
                helm_chart=planning_models.HelmChartPlan(**first_component["helm_chart"]),
                values=first_component["values"],
                values_files=[
                    planning_models.ValuesFileReference(**value_file)
                    for value_file in first_component["values_files"]
                ],
                wait_for_ready=first_component["wait_for_ready"],
                readiness_check=planning_models.ReadinessCheckPlan(
                    **first_component["readiness_check"]
                ),
            ),
            planning_models.ComponentPlan(
                name=second_component["name"],
                enabled=second_component["enabled"],
                depends_on=second_component["depends_on"],
                target_namespace=second_component["target_namespace"],
                kubernetes_manifest=second_component["kubernetes_manifest"],
                wait_for_ready=second_component["wait_for_ready"],
                readiness_check=planning_models.ReadinessCheckPlan(
                    **second_component["readiness_check"]
                ),
            ),
        ],
        prerequisites=valid_plan_payload["prerequisites"],
        fit_findings=planning_models.FitFindings(**valid_plan_payload["fit_findings"]),
        unresolved_questions=[],
        reasoning_summary=valid_plan_payload["reasoning_summary"],
    )

    assert plan.namespace_strategy.target_namespace == "ai-platform"
    assert plan.components[0].helm_chart.release_name == "vector-db"
    assert plan.components[1].kubernetes_manifest.startswith("apiVersion: v1")
    assert plan.fit_findings.status == "assumed-fit"

def test_validator_module_is_part_of_the_planning_package_surface() -> None:
    planning_models = importlib.import_module("webapp.planning.models")
    validator = _validator_module()

    assert hasattr(planning_models, "__file__")
    assert hasattr(validator, "__file__")
    assert callable(validator.validate_structured_plan)
