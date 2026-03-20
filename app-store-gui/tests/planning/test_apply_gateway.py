from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from webapp.planning.apply_gateway import (
    ApplyGateway,
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


def test_apply_gateway_preserves_existing_namespaces_when_no_override_is_provided(
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
    )

    result = apply_yaml_content_with_namespace(
        apply_gateway_input["yaml_text"],
        apply_gateway_input["no_namespace_override"],
        dependencies=dependencies,
    )

    assert result == {"applied": ["WekaAppStore"]}
    assert operations[0] == ("ensure_namespace_exists", "ai-platform")
    body = operations[1][1]["body"]
    assert body["metadata"]["namespace"] == "ai-platform"
    assert body["spec"]["appStack"]["components"][0]["targetNamespace"] == "ai-platform"


def test_apply_gateway_falls_back_to_builtin_resource_apply_for_non_cr_documents(
    builtin_manifest_document: dict,
) -> None:
    operations: list[tuple] = []

    class CustomObjectsApiStub:
        def create_namespaced_custom_object(self, **kwargs):
            operations.append(("unexpected_custom_object_call", kwargs))

    def create_from_dict_stub(api_client, *, data, namespace, verbose):
        operations.append(("create_from_dict", data, namespace, verbose))
        return [(SimpleNamespace(kind=data["kind"]), None)]

    dependencies = ApplyGatewayDependencies(
        load_kube_config=lambda: None,
        ensure_namespace_exists=lambda namespace: operations.append(("ensure_namespace_exists", namespace)),
        is_cluster_scoped=lambda doc: False,
        with_last_applied_annotation=lambda doc: doc,
        api_client_factory=lambda: object(),
        custom_objects_api_factory=lambda api_client: CustomObjectsApiStub(),
        create_from_dict=create_from_dict_stub,
    )

    yaml_text = yaml.safe_dump(builtin_manifest_document, sort_keys=False)
    result = apply_yaml_content_with_namespace(
        yaml_text,
        "ai-platform",
        dependencies=dependencies,
    )

    assert result == {"applied": ["ConfigMap"]}
    assert operations[0] == ("ensure_namespace_exists", "ai-platform")
    assert operations[1] == (
        "create_from_dict",
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "research-api-settings",
                "namespace": "ai-platform",
            },
            "data": {
                "MODEL_PROVIDER": "weka",
            },
        },
        "ai-platform",
        False,
    )


def test_apply_gateway_scope_stays_off_chat_and_inspection_paths(
    phase_one_scope_markers: dict[str, set[str]],
) -> None:
    assert {"APPLY-06", "APPLY-07"}.issubset(phase_one_scope_markers["allowed_requirement_ids"])
    assert "chat" in phase_one_scope_markers["excluded_topics"]
    assert "cluster_inspection" in phase_one_scope_markers["excluded_topics"]


def test_future_apply_gateway_module_can_wrap_existing_helpers() -> None:
    gateway = pytest.importorskip("webapp.planning.apply_gateway")

    assert hasattr(gateway, "__file__")
    assert hasattr(gateway, "ApplyGateway")
    assert hasattr(gateway, "apply_yaml_file_with_namespace")
    assert hasattr(gateway, "apply_yaml_content_with_namespace")


def test_apply_gateway_wrapper_keeps_file_and_content_entrypoints_thin(
    tmp_path,
    apply_gateway_input: dict,
) -> None:
    operations: list[tuple] = []
    manifest_path = tmp_path / "planner-output.yaml"
    manifest_path.write_text(apply_gateway_input["yaml_text"], encoding="utf-8")

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
    )
    gateway = ApplyGateway(project_root=str(tmp_path), dependencies=dependencies)

    file_result = gateway.apply_file("planner-output.yaml", "ai-platform")
    content_result = gateway.apply_content(apply_gateway_input["yaml_text"], "ai-platform")

    assert file_result == {"applied": ["WekaAppStore"]}
    assert content_result == {"applied": ["WekaAppStore"]}
    assert [entry[0] for entry in operations] == [
        "ensure_namespace_exists",
        "create_namespaced_custom_object",
        "ensure_namespace_exists",
        "create_namespaced_custom_object",
    ]


def test_main_apply_structured_plan_hands_canonical_yaml_to_shared_gateway(
    monkeypatch: pytest.MonkeyPatch,
    valid_plan_payload: dict,
) -> None:
    import webapp.main as main

    captured: dict[str, str] = {}

    class GatewayStub:
        def apply_content(self, content: str, namespace: str) -> dict:
            captured["content"] = content
            captured["namespace"] = namespace
            return {"applied": ["WekaAppStore"]}

    monkeypatch.setattr(main, "PLANNING_APPLY_GATEWAY", GatewayStub())

    result = main.apply_structured_plan(valid_plan_payload)

    assert result["result"] == {"applied": ["WekaAppStore"]}
    assert result["compiled_document"]["kind"] == "WekaAppStore"
    assert captured["namespace"] == "ai-platform"
    assert yaml.safe_load(captured["content"]) == result["compiled_document"]


def test_main_apply_helpers_delegate_to_shared_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    import webapp.main as main

    calls: list[tuple[str, str, str]] = []

    class GatewayStub:
        def apply_file(self, file_path: str, namespace: str) -> dict:
            calls.append(("file", file_path, namespace))
            return {"applied": []}

        def apply_content(self, content: str, namespace: str) -> dict:
            calls.append(("content", content, namespace))
            return {"applied": []}

    monkeypatch.setattr(main, "PLANNING_APPLY_GATEWAY", GatewayStub())

    main.apply_blueprint_with_namespace("planner-output.yaml", "ai-platform")
    main.apply_blueprint_content_with_namespace("kind: ConfigMap\n", "ai-platform")

    assert calls == [
        ("file", "planner-output.yaml", "ai-platform"),
        ("content", "kind: ConfigMap\n", "ai-platform"),
    ]


def test_main_apply_structured_plan_allows_namespace_override(
    monkeypatch: pytest.MonkeyPatch,
    valid_plan_payload: dict,
) -> None:
    import webapp.main as main

    captured: dict[str, str] = {}

    class GatewayStub:
        def apply_content(self, content: str, namespace: str) -> dict:
            captured["content"] = content
            captured["namespace"] = namespace
            return {"applied": ["WekaAppStore"]}

    monkeypatch.setattr(main, "PLANNING_APPLY_GATEWAY", GatewayStub())

    result = main.apply_structured_plan(valid_plan_payload, namespace_override="review-space")

    assert result["result"] == {"applied": ["WekaAppStore"]}
    assert captured["namespace"] == "review-space"
    assert yaml.safe_load(captured["content"])["kind"] == "WekaAppStore"
