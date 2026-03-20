from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _base_valid_plan() -> dict:
    return {
        "request_summary": "Deploy the enterprise research stack into the ai-platform namespace.",
        "blueprint_family": "ai-agent-enterprise-research",
        "namespace_strategy": {
            "mode": "explicit",
            "target_namespace": "ai-platform",
        },
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
                "values": {
                    "replicaCount": 1,
                },
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
                    "timeout": 600,
                },
            },
            {
                "name": "research-api",
                "enabled": True,
                "depends_on": ["vector-db"],
                "target_namespace": "ai-platform",
                "kubernetes_manifest": (
                    "apiVersion: v1\n"
                    "kind: ConfigMap\n"
                    "metadata:\n"
                    "  name: research-api-config\n"
                    "data:\n"
                    "  MODEL_PROVIDER: weka\n"
                ),
                "wait_for_ready": True,
                "readiness_check": {
                    "type": "deployment",
                    "selector": "app=research-api",
                    "timeout": 300,
                },
            },
        ],
        "prerequisites": [
            "WekaAppStore CRD installed",
            "Destination namespace exists or can be created by the existing apply path",
        ],
        "fit_findings": {
            "status": "assumed-fit",
            "notes": [
                "Cluster and WEKA inspection data is out of scope for Phase 1 fixtures.",
            ],
        },
        "unresolved_questions": [],
        "reasoning_summary": (
            "Use the supported ai-agent-enterprise-research family and preserve the explicit "
            "namespace and dependency ordering in canonical YAML."
        ),
    }


@pytest.fixture
def valid_plan_payload() -> dict:
    return _base_valid_plan()


@pytest.fixture
def invalid_plan_payloads(valid_plan_payload: dict) -> dict[str, dict]:
    missing_family = deepcopy(valid_plan_payload)
    missing_family.pop("blueprint_family")

    duplicate_component_names = deepcopy(valid_plan_payload)
    duplicate_component_names["components"][1]["name"] = "vector-db"

    dependency_on_missing_component = deepcopy(valid_plan_payload)
    dependency_on_missing_component["components"][1]["depends_on"] = ["missing-component"]

    missing_deployment_method = deepcopy(valid_plan_payload)
    missing_deployment_method["components"][1].pop("kubernetes_manifest")

    both_deployment_methods = deepcopy(valid_plan_payload)
    both_deployment_methods["components"][1]["helm_chart"] = {
        "repository": "https://charts.example.com/platform",
        "name": "research-api",
        "version": "1.0.0",
    }

    blocking_unresolved_question = deepcopy(valid_plan_payload)
    blocking_unresolved_question["unresolved_questions"] = [
        {
            "question": "What namespace should receive the install?",
            "blocking": True,
        }
    ]

    return {
        "missing_blueprint_family": missing_family,
        "duplicate_component_names": duplicate_component_names,
        "dependency_on_missing_component": dependency_on_missing_component,
        "missing_deployment_method": missing_deployment_method,
        "both_deployment_methods": both_deployment_methods,
        "blocking_unresolved_question": blocking_unresolved_question,
    }


@pytest.fixture
def warning_case_payload(valid_plan_payload: dict) -> dict:
    payload = deepcopy(valid_plan_payload)
    payload["components"][0]["helm_chart"].pop("release_name")
    payload["components"][0].pop("wait_for_ready")
    payload["expected_normalization_warnings"] = [
        {
            "path": "components[0].helm_chart.release_name",
            "message": "release_name defaults to the component name when omitted",
        },
        {
            "path": "components[0].wait_for_ready",
            "message": "wait_for_ready defaults to true when omitted",
        },
    ]
    return payload


@pytest.fixture
def phase_one_scope_markers() -> dict[str, set[str]]:
    return {
        "allowed_requirement_ids": {
            "PLAN-06",
            "PLAN-07",
            "PLAN-08",
            "APPLY-06",
            "APPLY-07",
        },
        "excluded_topics": {
            "chat",
            "cluster_inspection",
            "weka_inspection",
            "coexistence",
        },
    }


@pytest.fixture
def compiled_wekaappstore_document(valid_plan_payload: dict) -> dict:
    return {
        "apiVersion": "warp.io/v1alpha1",
        "kind": "WekaAppStore",
        "metadata": {
            "name": "ai-agent-enterprise-research-plan",
            "namespace": "ai-platform",
        },
        "spec": {
            "appStack": {
                "defaultNamespace": "ai-platform",
                "components": [
                    {
                        "name": "vector-db",
                        "enabled": True,
                        "dependsOn": [],
                        "targetNamespace": "ai-platform",
                        "helmChart": {
                            "repository": "https://charts.example.com/platform",
                            "name": "milvus",
                            "version": "4.2.0",
                            "releaseName": "vector-db",
                            "crdsStrategy": "Auto",
                        },
                        "values": {
                            "replicaCount": 1,
                        },
                        "valuesFiles": [
                            {
                                "kind": "ConfigMap",
                                "name": "vector-db-values",
                                "key": "values.yaml",
                            }
                        ],
                        "waitForReady": True,
                        "readinessCheck": {
                            "type": "deployment",
                            "name": "vector-db",
                            "timeout": 600,
                        },
                    },
                    {
                        "name": "research-api",
                        "enabled": True,
                        "dependsOn": ["vector-db"],
                        "targetNamespace": "ai-platform",
                        "kubernetesManifest": valid_plan_payload["components"][1]["kubernetes_manifest"],
                        "waitForReady": True,
                        "readinessCheck": {
                            "type": "deployment",
                            "selector": "app=research-api",
                            "timeout": 300,
                        },
                    },
                ],
            }
        },
    }


@pytest.fixture
def compiled_wekaappstore_yaml(compiled_wekaappstore_document: dict) -> str:
    return yaml.safe_dump(compiled_wekaappstore_document, sort_keys=False)


@pytest.fixture
def apply_gateway_input(compiled_wekaappstore_document: dict, compiled_wekaappstore_yaml: str) -> dict:
    return {
        "namespace_override": "ai-platform",
        "no_namespace_override": "",
        "yaml_text": compiled_wekaappstore_yaml,
        "document": compiled_wekaappstore_document,
        "expected_runtime_kind": "WekaAppStore",
        "expected_runtime_api_version": "warp.io/v1alpha1",
        "expected_apply_result": {
            "applied": ["WekaAppStore"],
        },
    }


@pytest.fixture
def builtin_manifest_document() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "research-api-settings",
            "namespace": "embedded-namespace",
        },
        "data": {
            "MODEL_PROVIDER": "weka",
        },
    }
