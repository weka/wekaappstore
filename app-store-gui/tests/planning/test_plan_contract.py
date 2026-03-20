from __future__ import annotations

import importlib

import pytest


PHASE_ONE_REQUIREMENTS = {"PLAN-02", "PLAN-03", "PLAN-06", "PLAN-07"}
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
        "invalid_values_file_kind",
        "malformed_components",
        "missing_blueprint_family",
        "missing_deployment_method",
        "unsupported_blueprint_family",
        "unsupported_crds_strategy",
        "unsupported_readiness_check_type",
        "unsupported_top_level_field",
    }
    assert "blueprint_family" not in invalid_plan_payloads["missing_blueprint_family"]
    assert invalid_plan_payloads["unsupported_blueprint_family"]["blueprint_family"] == "unsupported-family"
    assert invalid_plan_payloads["duplicate_component_names"]["components"][0]["name"] == (
        invalid_plan_payloads["duplicate_component_names"]["components"][1]["name"]
    )
    assert invalid_plan_payloads["dependency_on_missing_component"]["components"][1]["depends_on"] == [
        "missing-component"
    ]
    assert "kubernetes_manifest" not in invalid_plan_payloads["missing_deployment_method"]["components"][1]
    assert "helm_chart" in invalid_plan_payloads["both_deployment_methods"]["components"][1]
    assert invalid_plan_payloads["malformed_components"]["components"] == "vector-db"
    assert "chat_session" in invalid_plan_payloads["unsupported_top_level_field"]
    assert (
        invalid_plan_payloads["unsupported_readiness_check_type"]["components"][0]["readiness_check"]["type"]
        == "service"
    )
    assert (
        invalid_plan_payloads["invalid_values_file_kind"]["components"][0]["values_files"][0]["kind"]
        == "PersistentVolumeClaim"
    )
    assert (
        invalid_plan_payloads["unsupported_crds_strategy"]["components"][0]["helm_chart"]["crds_strategy"]
        == "Always"
    )
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
    ("case_name", "expected_code", "expected_path", "expected_message"),
    [
        (
            "missing_blueprint_family",
            "missing_required_field",
            "blueprint_family",
            "missing required field 'blueprint_family'",
        ),
        (
            "unsupported_blueprint_family",
            "unsupported_blueprint_family",
            "blueprint_family",
            "unsupported blueprint family 'unsupported-family'; supported families are ai-agent-enterprise-research, nvidia-vss, openfold",
        ),
        (
            "duplicate_component_names",
            "duplicate_component_name",
            "components[1].name",
            "component name 'vector-db' must be unique",
        ),
        (
            "dependency_on_missing_component",
            "unknown_dependency",
            "components[1].depends_on",
            "component 'research-api' depends on unknown component 'missing-component'",
        ),
        (
            "missing_deployment_method",
            "invalid_deployment_method",
            "components[1]",
            "component must define exactly one deployment method: helm_chart or kubernetes_manifest",
        ),
        (
            "both_deployment_methods",
            "invalid_deployment_method",
            "components[1]",
            "component must define exactly one deployment method: helm_chart or kubernetes_manifest",
        ),
        (
            "malformed_components",
            "invalid_type",
            "components",
            "field 'components' must be a list",
        ),
        (
            "unsupported_top_level_field",
            "unsupported_field",
            "chat_session",
            "unsupported field 'chat_session' is not part of the Phase 1 structured plan contract",
        ),
        (
            "unsupported_readiness_check_type",
            "unsupported_readiness_check_type",
            "components[0].readiness_check.type",
            "unsupported readiness_check type 'service'; supported types are custom, deployment, job, pod, statefulset",
        ),
        (
            "invalid_values_file_kind",
            "unsupported_values_file_kind",
            "components[0].values_files[0].kind",
            "values_files kind must be ConfigMap or Secret",
        ),
        (
            "unsupported_crds_strategy",
            "unsupported_crds_strategy",
            "components[0].helm_chart.crds_strategy",
            "unsupported CRDs strategy 'Always'; supported values are Auto, Install, Skip",
        ),
        (
            "blocking_unresolved_question",
            "blocking_unresolved_question",
            "unresolved_questions[0]",
            "install-critical unresolved question blocks YAML generation and apply handoff",
        ),
    ],
)
def test_validator_rejects_deterministic_contract_failures(
    invalid_plan_payloads: dict[str, dict],
    case_name: str,
    expected_code: str,
    expected_path: str,
    expected_message: str,
) -> None:
    result = _validator_module().validate_structured_plan(invalid_plan_payloads[case_name])

    assert result.valid is False
    assert result.plan is None
    assert (expected_code, expected_path, expected_message) in {
        (error.code, error.path, error.message) for error in result.errors
    }
    assert result.errors == sorted(
        result.errors, key=lambda error: (error.path, error.code, error.message)
    )


def test_validator_result_serializes_to_a_stable_contract_payload(valid_plan_payload: dict) -> None:
    result = _validator_module().validate_structured_plan(valid_plan_payload)

    assert result.valid is True
    assert result.to_dict() == {
        "valid": True,
        "plan": {
            "request_summary": valid_plan_payload["request_summary"],
            "blueprint_family": valid_plan_payload["blueprint_family"],
            "namespace_strategy": valid_plan_payload["namespace_strategy"],
            "components": [
                {
                    "name": "vector-db",
                    "enabled": True,
                    "depends_on": [],
                    "target_namespace": "ai-platform",
                    "helm_chart": {
                        "repository": "https://charts.example.com/platform",
                        "name": "milvus",
                        "version": "4.2.0",
                        "release_name": "vector-db",
                        "crds_strategy": "Auto",
                    },
                    "kubernetes_manifest": None,
                    "values": {"replicaCount": 1},
                    "values_files": [
                        {
                            "kind": "ConfigMap",
                            "name": "vector-db-values",
                            "key": "values.yaml",
                        }
                    ],
                    "wait_for_ready": True,
                    "readiness_check": {
                        "type": "deployment",
                        "name": "vector-db",
                        "selector": None,
                        "match_labels": {},
                        "namespace": None,
                        "timeout": 600,
                    },
                },
                {
                    "name": "research-api",
                    "enabled": True,
                    "depends_on": ["vector-db"],
                    "target_namespace": "ai-platform",
                    "helm_chart": None,
                    "kubernetes_manifest": valid_plan_payload["components"][1]["kubernetes_manifest"],
                    "values": {},
                    "values_files": [],
                    "wait_for_ready": True,
                    "readiness_check": {
                        "type": "deployment",
                        "name": None,
                        "selector": "app=research-api",
                        "match_labels": {},
                        "namespace": None,
                        "timeout": 300,
                    },
                },
            ],
            "prerequisites": valid_plan_payload["prerequisites"],
            "fit_findings": valid_plan_payload["fit_findings"],
            "unresolved_questions": [],
            "reasoning_summary": valid_plan_payload["reasoning_summary"],
            "normalization_warnings": [],
        },
        "warnings": [],
        "errors": [],
    }


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
