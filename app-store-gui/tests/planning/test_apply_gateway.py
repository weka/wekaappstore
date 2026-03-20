from __future__ import annotations

import pytest
import yaml

from webapp.planning.apply_gateway import (
    ApplyGatewayDependencies,
    apply_yaml_content_with_namespace,
)


PHASE_ONE_REQUIREMENTS = {"APPLY-06", "APPLY-07"}


def test_apply_gateway_fixture_targets_existing_runtime_path(apply_gateway_input: dict) -> None:
    payload = yaml.safe_load(apply_gateway_input["yaml_text"])

    assert PHASE_ONE_REQUIREMENTS == {"APPLY-06", "APPLY-07"}
    assert payload == apply_gateway_input["document"]
    assert payload["kind"] == apply_gateway_input["expected_runtime_kind"]
    assert payload["apiVersion"] == apply_gateway_input["expected_runtime_api_version"]
    assert apply_gateway_input["expected_apply_result"] == {"applied": ["WekaAppStore"]}
    assert list(payload) == ["apiVersion", "kind", "metadata", "spec"]


def test_apply_gateway_routes_wekaappstore_documents_through_custom_objects_api(
    apply_gateway_input: dict,
) -> None:
    operations: list[tuple] = []

    class CustomObjectsApiStub:
        def create_namespaced_custom_object(self, **kwargs):
            operations.append(("create_namespaced_custom_object", kwargs))

    dependencies = ApplyGatewayDependencies(
        load_kube_config=lambda: None,
        ensure_namespace_exists=lambda namespace: operations.append(("ensure_namespace_exists", namespace)),
        is_cluster_scoped=lambda doc: False,
        crd_scope_for=lambda group, plural: "Namespaced",
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=lambda: object(),
        custom_objects_api_factory=lambda api_client: CustomObjectsApiStub(),
        create_from_dict=lambda *args, **kwargs: pytest.fail("built-in fallback should not be used"),
    )

    result = apply_yaml_content_with_namespace(
        apply_gateway_input["yaml_text"],
        apply_gateway_input["namespace_override"],
        dependencies=dependencies,
    )

    assert result == {"applied": ["WekaAppStore"]}
    assert operations[0] == ("ensure_namespace_exists", "ai-platform")
    assert operations[1][0] == "create_namespaced_custom_object"
    payload = operations[1][1]
    assert payload["group"] == "warp.io"
    assert payload["version"] == "v1alpha1"
    assert payload["namespace"] == "ai-platform"
    assert payload["plural"] == "wekaappstores"
    assert payload["body"]["metadata"]["namespace"] == "ai-platform"
    assert payload["body"]["spec"]["appStack"]["components"][0]["targetNamespace"] == "ai-platform"


def test_apply_gateway_scope_stays_off_chat_and_inspection_paths(
    phase_one_scope_markers: dict[str, set[str]],
) -> None:
    assert {"APPLY-06", "APPLY-07"}.issubset(phase_one_scope_markers["allowed_requirement_ids"])
    assert "chat" in phase_one_scope_markers["excluded_topics"]
    assert "cluster_inspection" in phase_one_scope_markers["excluded_topics"]


def test_future_apply_gateway_module_can_wrap_existing_helpers() -> None:
    gateway = pytest.importorskip("webapp.planning.apply_gateway")

    assert hasattr(gateway, "__file__")
    assert hasattr(gateway, "apply_yaml_file_with_namespace")
    assert hasattr(gateway, "apply_yaml_content_with_namespace")
