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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: apply_called.append(True) or {"applied": []})

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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: {"applied": []})

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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", mock_apply)
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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", mock_apply)
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
    rendered_content = []
    def mock_apply(rendered, namespace=""):
        rendered_content.append(rendered)
        return {"applied": []}
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", mock_apply)
    _mock_appstack_ready(monkeypatch)

    async def run():
        request = _make_request_stub()
        variables = {"namespace": "prod", "storage_class": "weka-sc"}
        resp = await main.deploy_stream(request, app_name="test-app", variables=_json.dumps(variables))
        await _collect_sse(resp)
        assert rendered_content, "apply should have been called with rendered content"
        assert "prod" in rendered_content[0]
        assert "weka-sc" in rendered_content[0]

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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: {"applied": ["WekaAppStore"]})
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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: apply_called.append(True) or {"applied": []})

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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: {"applied": []})
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
    monkeypatch.setattr(main, "apply_blueprint_content_with_namespace", lambda *a, **kw: {"applied": []})

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
