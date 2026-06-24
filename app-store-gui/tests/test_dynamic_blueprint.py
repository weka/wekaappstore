from __future__ import annotations

import asyncio
import inspect
import json as _json
import os
import shutil
import tempfile
from types import SimpleNamespace

os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")

import webapp.main as main


# ---------------------------------------------------------------------------
# parse_x_variables tests
# ---------------------------------------------------------------------------

def test_parse_x_variables_empty_string():
    """Test 1: empty string returns {}."""
    assert main.parse_x_variables("") == {}


def test_parse_x_variables_no_x_variables_key():
    """Test 2: YAML without x-variables key returns {}."""
    yaml_text = "apiVersion: warp.io/v1alpha1\nkind: WekaAppStore\n"
    assert main.parse_x_variables(yaml_text) == {}


def test_parse_x_variables_with_string_var():
    """Test 3: YAML with x-variables returns the mapped dict."""
    yaml_text = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
        '    description: "Target namespace"\n'
    )
    result = main.parse_x_variables(yaml_text)
    assert result == {
        "namespace": {
            "type": "string",
            "required": True,
            "description": "Target namespace",
        }
    }


def test_parse_x_variables_only_x_variables_returned():
    """Test 4: only the x-variables dict is returned, not apiVersion etc."""
    yaml_text = (
        "apiVersion: warp.io/v1alpha1\n"
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
    )
    result = main.parse_x_variables(yaml_text)
    assert "apiVersion" not in result
    assert "namespace" in result


def test_parse_x_variables_parse_error_returns_empty():
    """Test 5: YAML parse error returns {} without raising."""
    bad_yaml = "x-variables: {\ninvalid: [unclosed"
    result = main.parse_x_variables(bad_yaml)
    assert result == {}


def test_find_blueprint_finds_matching_yaml(tmp_path):
    """Test 6: find_blueprint returns absolute path when matching file exists."""
    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    yaml_file = app_dir / "my-app.yaml"
    yaml_file.write_text(
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
    )
    result = main.find_blueprint("my-app", blueprints_dir=str(tmp_path))
    assert result == str(yaml_file)


def test_find_blueprint_returns_none_when_not_found(tmp_path):
    """Test 7: find_blueprint returns None when no matching file exists."""
    result = main.find_blueprint("unknown", blueprints_dir=str(tmp_path))
    assert result is None


def test_find_blueprint_cluster_init_special_case():
    """Test 8: cluster-init is special-cased and returns the cluster-init YAML path."""
    result = main.find_blueprint("cluster-init", blueprints_dir="/some/dir")
    expected = os.path.join("/some/dir", "cluster_init", "app-store-cluster-init.yaml")
    assert result == expected


def test_find_blueprint_ignores_files_without_x_variables(tmp_path):
    """Test 9: find_blueprint ignores YAML files without x-variables key."""
    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    yaml_file = app_dir / "my-app.yaml"
    yaml_file.write_text("apiVersion: warp.io/v1alpha1\nkind: WekaAppStore\n")
    result = main.find_blueprint("my-app", blueprints_dir=str(tmp_path))
    assert result is None


def test_parse_x_variables_credential_type():
    """Test 10: parse_x_variables returns credential_type field for credential vars."""
    yaml_text = (
        "x-variables:\n"
        "  hf_cred:\n"
        "    type: credential\n"
        "    credential_type: huggingface\n"
        "    required: true\n"
    )
    result = main.parse_x_variables(yaml_text)
    assert result == {
        "hf_cred": {
            "type": "credential",
            "credential_type": "huggingface",
            "required": True,
        }
    }


# ---------------------------------------------------------------------------
# deploy_stream refactor tests (Task 1 — RED phase)
# ---------------------------------------------------------------------------

def _make_request_stub():
    """Create a minimal request stub suitable for deploy_stream."""
    async def _not_disconnected():
        return False

    return SimpleNamespace(
        headers={},
        cookies={},
        query_params={},
        url=SimpleNamespace(path="/deploy-stream"),
        scope={"type": "http"},
        is_disconnected=_not_disconnected,
    )


async def _collect_sse(streaming_response):
    """Collect all SSE event dicts from a StreamingResponse."""
    events = []
    async for chunk in streaming_response.body_iterator:
        chunk_str = chunk if isinstance(chunk, str) else chunk.decode()
        for line in chunk_str.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(_json.loads(line[6:]))
                except Exception:
                    pass
    return events


def _mock_appstack_ready(monkeypatch, phase="Ready", components=None):
    """Make deploy_stream's status polling return a terminal CR immediately.

    Without this, deploy_stream polls the WekaAppStore .status until appStackPhase is
    Ready/Failed — which never happens against a test's stubbed cluster — so the SSE
    generator would loop until its 15-minute cap.
    """
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    comp = components if components is not None else []

    class _StubCustomObjectsApi:
        def get_namespaced_custom_object(self, **kwargs):
            return {"status": {"appStackPhase": phase, "componentStatus": comp}}

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: _StubCustomObjectsApi())


def test_deploy_stream_signature_uses_variables_param():
    """Test 1: deploy_stream signature contains variables: str = '{}' and no per-variable positional params."""
    sig = inspect.signature(main.deploy_stream)
    params = list(sig.parameters.keys())
    assert "variables" in params, "deploy_stream must have 'variables' parameter"
    assert "storage_class" not in params, "storage_class must be removed from deploy_stream"
    assert "vllm_chat_model" not in params, "vllm_chat_model must be removed from deploy_stream"
    assert "vllm_embed_model" not in params, "vllm_embed_model must be removed from deploy_stream"
    assert "vllm_model" not in params, "vllm_model must be removed from deploy_stream"
    assert "weka_cluster_filesystem" not in params, "weka_cluster_filesystem must be removed from deploy_stream"
    assert "openfold_storage_capacity" not in params, "openfold_storage_capacity must be removed from deploy_stream"
    assert "deployment_name" not in params, "deployment_name must be removed from deploy_stream"


def test_deploy_stream_unknown_app_yields_error():
    """Test 2: Unknown app_name yields SSE error event."""
    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="unknown-blueprint-xyz", variables="{}")
        events = await _collect_sse(resp)
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, f"Expected error event, got: {events}"
        assert "Unknown app" in error_events[0]["message"]

    asyncio.run(run())


def test_deploy_stream_missing_required_variable_yields_error(tmp_path, monkeypatch):
    """Test 3: Missing required variable yields SSE error before apply."""
    yaml_content = (
        "x-variables:\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: true\n"
        '    description: "StorageClass name"\n'
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    apply_called = []
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: apply_called.append(True) or {"applied": []})

    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps({}))
        events = await _collect_sse(resp)
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, f"Expected error event for missing required var, got: {events}"
        assert "storage_class" in error_events[0]["message"]
        assert not apply_called, "apply must NOT be called when required variable is missing"

    asyncio.run(run())


def test_deploy_stream_cluster_init_exempt_from_required_validation(tmp_path, monkeypatch):
    """Test 4: cluster-init app_name skips required-field validation."""
    yaml_content = (
        "x-variables:\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: true\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: cluster-init\n"
        "  namespace: default\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "app-store-cluster-init.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: {"applied": []})

    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="cluster-init", variables=_json.dumps({"namespace": "default"}))
        events = await _collect_sse(resp)
        error_events = [e for e in events if e.get("type") == "error"]
        # cluster-init should NOT emit a required-variable-missing error
        req_errors = [e for e in error_events if "Required variable missing" in e.get("message", "")]
        assert not req_errors, f"cluster-init should skip required-field validation, got errors: {req_errors}"

    asyncio.run(run())


def test_deploy_stream_namespace_from_variables(tmp_path, monkeypatch):
    """Test 5: namespace is extracted from variables dict; defaults to 'default' if absent."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    captured_ns = []
    def mock_apply(rendered, namespace=""):
        captured_ns.append(namespace)
        return {"applied": []}
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", mock_apply)
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps({"namespace": "my-ns"}))
        await _collect_sse(resp)
        assert captured_ns, "apply should have been called"
        assert captured_ns[0] == "my-ns", f"Expected namespace 'my-ns', got '{captured_ns[0]}'"

    asyncio.run(run())


def test_deploy_stream_namespace_defaults_to_default_when_absent(tmp_path, monkeypatch):
    """Test 5b: namespace defaults to 'default' when absent from variables."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: false\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    captured_ns = []
    def mock_apply(rendered, namespace=""):
        captured_ns.append(namespace)
        return {"applied": []}
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", mock_apply)
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        # No namespace in variables
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps({}))
        await _collect_sse(resp)
        assert captured_ns, "apply should have been called"
        assert captured_ns[0] == "default", f"Expected namespace 'default', got '{captured_ns[0]}'"

    asyncio.run(run())


def test_deploy_stream_render_uses_full_variables_dict(tmp_path, monkeypatch):
    """Test 6: template.render is called with full user_vars dict (all keys from JSON)."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: true\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    storageClass: [[storage_class]]\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    applied_docs = []
    def mock_apply(docs, namespace=""):
        applied_docs.append(list(docs))
        return {"applied": []}
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", mock_apply)
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        variables = {"namespace": "prod", "storage_class": "weka-sc"}
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps(variables))
        await _collect_sse(resp)
        assert applied_docs, "apply should have been called with rendered documents"
        cr = next(d for d in applied_docs[0] if d.get("kind") == "WekaAppStore")
        assert cr["metadata"]["namespace"] == "prod"
        assert cr["spec"]["appStack"]["storageClass"] == "weka-sc"

    asyncio.run(run())


def test_deploy_stream_stamps_gui_variables_annotation(tmp_path, monkeypatch):
    """deploy_stream stamps submitted variables onto the WekaAppStore CR as an annotation."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: false\n"
        "  keycloak_url:\n"
        "    type: string\n"
        "    required: false\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    applied_docs = []
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace",
                        lambda docs, namespace="": applied_docs.append(list(docs)) or {"applied": []})
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        variables = {"namespace": "ns1", "keycloak_url": "https://kc.example.com"}
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps(variables))
        await _collect_sse(resp)
        assert applied_docs, "apply should have been called"
        cr = next(d for d in applied_docs[0] if d.get("kind") == "WekaAppStore")
        raw = cr["metadata"]["annotations"]["warp.io/gui-variables"]
        assert _json.loads(raw) == variables

    asyncio.run(run())


def test_deploy_stream_emits_component_events_from_operator_status(tmp_path, monkeypatch):
    """deploy_stream emits per-component events from the operator's componentStatus and
    completes (ok) when appStackPhase is Ready."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: false\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-app\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components:\n"
        "      - name: comp-a\n"
        "      - name: comp-b\n"
    )
    bp_file = tmp_path / "test-app.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: {"applied": ["WekaAppStore"]})
    _mock_appstack_ready(monkeypatch, phase="Ready", components=[
        {"name": "comp-a", "phase": "Ready"},
        {"name": "comp-b", "phase": "Ready"},
    ])

    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps({"namespace": "ns1"}))
        events = await _collect_sse(resp)
        comp_events = [e for e in events if e.get("type") == "component"]
        names = {e["name"] for e in comp_events}
        assert names == {"comp-a", "comp-b"}, f"Expected per-component events, got: {events}"
        assert all(e["phase"] == "Ready" for e in comp_events)
        complete = [e for e in events if e.get("type") == "complete"]
        assert complete and complete[-1].get("ok") is True, f"Expected ok complete, got: {events}"

    asyncio.run(run())


def test_deploy_stream_no_app_map_in_source():
    """Test 7: app_map dict is NOT present in deploy_stream source (removed)."""
    import inspect as _inspect
    source = _inspect.getsource(main.deploy_stream)
    assert "app_map" not in source, "deploy_stream must not contain a local app_map dict"


def test_post_deploy_uses_find_blueprint():
    """Test 8: POST /deploy route uses find_blueprint instead of app_map."""
    import inspect as _inspect
    source = _inspect.getsource(main.deploy)
    assert "app_map" not in source, "POST /deploy must not contain a local app_map dict"
    assert "find_blueprint" in source, "POST /deploy must call find_blueprint"


# ---------------------------------------------------------------------------
# Integration tests for schema-validation flow (Task 2 — tests 11-15)
# ---------------------------------------------------------------------------

def test_deploy_stream_validates_required_variables(tmp_path, monkeypatch):
    """Test 11: Required variable missing causes SSE error; apply not called."""
    yaml_content = (
        "x-variables:\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: true\n"
        '    description: "StorageClass name"\n'
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-required\n"
        "  namespace: default\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-required.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    apply_called = []
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: apply_called.append(True) or {"applied": []})

    async def run():
        request = _make_request_stub()
        resp = await main.deploy_stream(request, app_name="test-required", variables=_json.dumps({"storage_class": ""}))
        events = await _collect_sse(resp)
        error_events = [e for e in events if e.get("type") == "error"]
        assert error_events, f"Expected error event, got: {events}"
        assert "storage_class" in error_events[0]["message"]
        assert "Required variable missing" in error_events[0]["message"]
        assert not apply_called, "apply must NOT be called when required variable is empty"

    asyncio.run(run())


def test_deploy_stream_optional_variables_pass_validation(tmp_path, monkeypatch):
    """Test 12: Optional variable missing does NOT cause error."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: false\n"
        '    description: "Optional StorageClass"\n'
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: test-optional\n"
        "  namespace: [[namespace]]\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "test-optional.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: {"applied": []})
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        # namespace provided, storage_class absent (optional)
        resp = await main.deploy_stream(request, app_name="test-optional", variables=_json.dumps({"namespace": "test-ns"}))
        events = await _collect_sse(resp)
        req_errors = [e for e in events if e.get("type") == "error" and "Required variable missing" in e.get("message", "")]
        assert not req_errors, f"Optional missing var must not cause error, got: {req_errors}"

    asyncio.run(run())


def test_deploy_stream_cluster_init_exemption(tmp_path, monkeypatch):
    """Test 13: cluster-init bypasses required-field validation entirely."""
    yaml_content = (
        "x-variables:\n"
        "  storage_class:\n"
        "    type: string\n"
        "    required: true\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: cluster-init\n"
        "  namespace: default\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "app-store-cluster-init.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: {"applied": []})

    async def run():
        request = _make_request_stub()
        # storage_class is required in schema but cluster-init is exempt
        resp = await main.deploy_stream(request, app_name="cluster-init", variables=_json.dumps({"namespace": "default"}))
        events = await _collect_sse(resp)
        req_errors = [e for e in events if e.get("type") == "error" and "Required variable missing" in e.get("message", "")]
        assert not req_errors, f"cluster-init must bypass required-field validation, got: {req_errors}"

    asyncio.run(run())


def test_ai_research_fixture_has_x_variables():
    """Test 14: parse_x_variables on ai-research.yaml returns non-empty dict."""
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "../../mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml"
    )
    with open(fixture_path, "r") as f:
        content = f.read()
    schema = main.parse_x_variables(content)
    assert schema, f"ai-research.yaml must have a non-empty x-variables block, got: {schema}"
    assert isinstance(schema, dict)


def test_data_pipeline_fixture_has_x_variables():
    """Test 15: parse_x_variables on data-pipeline.yaml returns non-empty dict."""
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "../../mcp-server/tests/fixtures/sample_blueprints/data-pipeline.yaml"
    )
    with open(fixture_path, "r") as f:
        content = f.read()
    schema = main.parse_x_variables(content)
    assert schema, f"data-pipeline.yaml must have a non-empty x-variables block, got: {schema}"
    assert isinstance(schema, dict)


# ---------------------------------------------------------------------------
# Phase 29 Plan 01: NAMESPACE_PRESERVING_APPS + parse_deploy_timeout tests
# ---------------------------------------------------------------------------


def test_namespace_preserving_apps_contains_both_apps():
    """Test 16: NAMESPACE_PRESERVING_APPS set contains both cluster-init and app-store-install."""
    assert "cluster-init" in main.NAMESPACE_PRESERVING_APPS, \
        "NAMESPACE_PRESERVING_APPS must include 'cluster-init'"
    assert "app-store-install" in main.NAMESPACE_PRESERVING_APPS, \
        "NAMESPACE_PRESERVING_APPS must include 'app-store-install'"


def test_deploy_stream_app_store_install_preserves_namespace(tmp_path, monkeypatch):
    """Test 17: deploy_stream with app_name='app-store-install' passes namespace='' to apply
    even when variables carry a non-empty namespace value."""
    yaml_content = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: false\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: app-store-install\n"
        "  namespace: default\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "app-store-install.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    captured_ns = []

    def mock_apply(docs, namespace=""):
        captured_ns.append(namespace)
        return {"applied": []}

    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", mock_apply)

    async def run():
        request = _make_request_stub()
        # Pass a non-empty namespace in variables — must be ignored for app-store-install
        resp = await main.deploy_stream(
            request,
            app_name="app-store-install",
            variables=_json.dumps({"namespace": "my-ns"}),
        )
        await _collect_sse(resp)
        assert captured_ns, "apply should have been called"
        assert captured_ns[0] == "", \
            f"Expected namespace='' for app-store-install, got '{captured_ns[0]}'"

    asyncio.run(run())


def test_deploy_stream_app_store_install_exempt_from_required_validation(tmp_path, monkeypatch):
    """Test 18: deploy_stream with app_name='app-store-install' does NOT emit a
    'Required variable missing' error even when a required x-variable is absent."""
    yaml_content = (
        "x-variables:\n"
        "  quay_password:\n"
        "    type: string\n"
        "    required: true\n"
        "\n"
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: app-store-install\n"
        "  namespace: default\n"
        "spec:\n"
        "  appStack:\n"
        "    components: []\n"
    )
    bp_file = tmp_path / "app-store-install.yaml"
    bp_file.write_text(yaml_content)

    monkeypatch.setattr(main, "find_blueprint", lambda app_name, blueprints_dir=None: str(bp_file))
    monkeypatch.setattr(main, "apply_blueprint_documents_with_namespace", lambda *a, **kw: {"applied": []})

    async def run():
        request = _make_request_stub()
        # quay_password is required in schema but app-store-install is namespace-preserving exempt
        resp = await main.deploy_stream(
            request,
            app_name="app-store-install",
            variables=_json.dumps({}),
        )
        events = await _collect_sse(resp)
        req_errors = [
            e for e in events
            if e.get("type") == "error" and "Required variable missing" in e.get("message", "")
        ]
        assert not req_errors, \
            f"app-store-install must bypass required-field validation, got: {req_errors}"

    asyncio.run(run())


def test_parse_deploy_timeout_returns_blueprint_value():
    """Test 19: parse_deploy_timeout returns the blueprint's x-deploy-timeout value."""
    result = main.parse_deploy_timeout("x-deploy-timeout: 2700\n")
    assert result == 2700, f"Expected 2700, got {result}"


def test_parse_deploy_timeout_returns_default_when_absent():
    """Test 20: parse_deploy_timeout returns DEFAULT_DEPLOY_TIMEOUT_SECONDS when key is absent."""
    result = main.parse_deploy_timeout("")
    assert result == main.DEFAULT_DEPLOY_TIMEOUT_SECONDS, \
        f"Expected default {main.DEFAULT_DEPLOY_TIMEOUT_SECONDS}, got {result}"


def test_parse_deploy_timeout_returns_default_for_malformed_value():
    """Test 21: parse_deploy_timeout returns default for non-integer and non-positive values."""
    assert main.parse_deploy_timeout("x-deploy-timeout: notanumber\n") == main.DEFAULT_DEPLOY_TIMEOUT_SECONDS
    assert main.parse_deploy_timeout("x-deploy-timeout: -100\n") == main.DEFAULT_DEPLOY_TIMEOUT_SECONDS
    assert main.parse_deploy_timeout("x-deploy-timeout: 0\n") == main.DEFAULT_DEPLOY_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Phase 29 Plan 02: build_quay_dockerconfigjson + split_endpoints tests
# ---------------------------------------------------------------------------


def test_build_quay_dockerconfigjson_byte_exact():
    """Test 22: quay auth decodes to exactly user:pass with no trailing bytes."""
    import base64
    import json

    result = main.build_quay_dockerconfigjson("alice", "s3cr3t")
    parsed = json.loads(result)
    raw_auth = parsed["auths"]["quay.io"]["auth"]
    decoded = base64.b64decode(raw_auth)
    assert decoded == b"alice:s3cr3t", (
        f"Expected b'alice:s3cr3t', got {decoded!r}"
    )
    assert not decoded.endswith(b"\n"), (
        "auth must not have a trailing newline byte"
    )


def test_build_quay_dockerconfigjson_structure():
    """Test 23: build_quay_dockerconfigjson returns valid JSON with expected shape."""
    import json

    result = main.build_quay_dockerconfigjson("user", "pass")
    parsed = json.loads(result)
    assert "auths" in parsed
    assert "quay.io" in parsed["auths"]
    assert "auth" in parsed["auths"]["quay.io"]


def test_split_endpoints_single():
    """Test 24: split_endpoints with a single endpoint produces correct both forms."""
    import json

    result = main.split_endpoints("h:1")
    assert result["endpoints_csv"] == "h:1"
    assert json.loads(result["join_ip_ports_list"]) == ["h:1"]


def test_split_endpoints_multiple_with_whitespace():
    """Test 25: split_endpoints with multiple entries and surrounding whitespace trims correctly."""
    import json

    result = main.split_endpoints("a:1, b:2 , c:3")
    assert result["endpoints_csv"] == "a:1,b:2,c:3"
    assert json.loads(result["join_ip_ports_list"]) == ["a:1", "b:2", "c:3"]


def test_split_endpoints_drops_empty_entries():
    """Test 26: split_endpoints drops empty entries (e.g. trailing comma)."""
    import json

    result = main.split_endpoints("a:1,,b:2,")
    assert json.loads(result["join_ip_ports_list"]) == ["a:1", "b:2"]
    assert result["endpoints_csv"] == "a:1,b:2"


def test_split_endpoints_join_ip_ports_list_is_valid_yaml():
    """Test 27: join_ip_ports_list rendered into a YAML line parses as a valid YAML list."""
    import yaml
    from jinja2 import Environment

    result = main.split_endpoints("a:1,b:2")
    # Use the same Jinja2 Environment delimiters as deploy_stream
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    template = env.from_string("joinIpPorts: [[ join_ip_ports_list ]]")
    rendered = template.render(join_ip_ports_list=result["join_ip_ports_list"])
    parsed = yaml.safe_load(rendered)
    assert parsed == {"joinIpPorts": ["a:1", "b:2"]}, (
        f"Expected list, got: {parsed!r}"
    )


# ---------------------------------------------------------------------------
# Phase 29 Plan 03: secret-key predicate + _safe_gui_variables + SSE redaction
# ---------------------------------------------------------------------------


def test_is_secret_key_matches_password():
    """Test 28: _is_secret_key returns True for keys containing 'password' (case-insensitive)."""
    assert main._is_secret_key("weka_password")
    assert main._is_secret_key("quay_password")
    assert main._is_secret_key("PASSWORD")
    assert main._is_secret_key("my_Password_field")


def test_is_secret_key_matches_token():
    """Test 29: _is_secret_key returns True for keys containing 'token'."""
    assert main._is_secret_key("api_token")
    assert main._is_secret_key("TOKEN")
    assert main._is_secret_key("auth_token_value")


def test_is_secret_key_matches_secret():
    """Test 30: _is_secret_key returns True for keys containing 'secret'."""
    assert main._is_secret_key("client_secret")
    assert main._is_secret_key("SECRET")
    assert main._is_secret_key("my_secret_key")


def test_is_secret_key_matches_quay_dockerconfigjson():
    """Test 31: _is_secret_key returns True for exact match 'quay_dockerconfigjson'."""
    assert main._is_secret_key("quay_dockerconfigjson")


def test_is_secret_key_non_secret_keys():
    """Test 32: _is_secret_key returns False for non-secret keys."""
    assert not main._is_secret_key("weka_username")
    assert not main._is_secret_key("namespace")
    assert not main._is_secret_key("operator_version")
    assert not main._is_secret_key("join_ip_ports")
    assert not main._is_secret_key("quay_username")
    assert not main._is_secret_key("storage_class")


def test_safe_gui_variables_drops_secret_keys():
    """Test 33: _safe_gui_variables removes all secret keys from the dict."""
    user_vars = {
        "weka_password": "hunter2",
        "quay_password": "qpass",
        "quay_dockerconfigjson": '{"auths":{}}',
        "api_token": "abc123",
        "client_secret": "xyz",
        "namespace": "default",
        "weka_username": "admin",
        "operator_version": "1.0.0",
    }
    result = main._safe_gui_variables(user_vars)
    assert "weka_password" not in result
    assert "quay_password" not in result
    assert "quay_dockerconfigjson" not in result
    assert "api_token" not in result
    assert "client_secret" not in result


def test_safe_gui_variables_preserves_non_secret_keys():
    """Test 34: _safe_gui_variables keeps all non-secret keys intact."""
    user_vars = {
        "weka_password": "hunter2",
        "namespace": "default",
        "weka_username": "admin",
        "operator_version": "1.0.0",
        "join_ip_ports": "10.0.0.1:14000",
    }
    result = main._safe_gui_variables(user_vars)
    assert result["namespace"] == "default"
    assert result["weka_username"] == "admin"
    assert result["operator_version"] == "1.0.0"
    assert result["join_ip_ports"] == "10.0.0.1:14000"


def test_safe_gui_variables_does_not_mutate_input():
    """Test 35: _safe_gui_variables does not modify the original dict."""
    user_vars = {"weka_password": "secret", "namespace": "default"}
    _ = main._safe_gui_variables(user_vars)
    assert "weka_password" in user_vars, "original dict must not be mutated"


def test_redact_secrets_replaces_secret_values_with_stars():
    """Test 36: _redact_secrets replaces each secret value with *** in the message."""
    user_vars = {
        "weka_password": "hunter2",
        "namespace": "default",
    }
    msg = "Error: failed to connect with password hunter2 to cluster"
    result = main._redact_secrets(msg, user_vars)
    assert "hunter2" not in result
    assert "***" in result


def test_redact_secrets_leaves_clean_message_unchanged():
    """Test 37: _redact_secrets returns an unchanged message when no secrets present."""
    user_vars = {"weka_password": "s3cr3t", "namespace": "default"}
    msg = "Component weka-operator reached Ready phase"
    result = main._redact_secrets(msg, user_vars)
    assert result == msg


def test_redact_secrets_redacts_multiple_secret_values():
    """Test 38: _redact_secrets replaces all distinct secret values in a message."""
    user_vars = {
        "weka_password": "wpass",
        "quay_password": "qpass",
        "quay_dockerconfigjson": "someconfigjson",
        "namespace": "default",
    }
    msg = "Applying manifest with wpass and qpass and someconfigjson embedded"
    result = main._redact_secrets(msg, user_vars)
    assert "wpass" not in result
    assert "qpass" not in result
    assert "someconfigjson" not in result
    assert result.count("***") == 3


def test_redact_secrets_ignores_empty_secret_values():
    """Test 39: _redact_secrets does not replace empty string secret values."""
    user_vars = {"weka_password": "", "namespace": "default"}
    msg = "Some message with no secret"
    result = main._redact_secrets(msg, user_vars)
    assert result == msg
