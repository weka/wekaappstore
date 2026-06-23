from __future__ import annotations

import asyncio
import base64
import json
import os
import pytest
from types import SimpleNamespace

os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")

import webapp.main as main

from .conftest import (
    make_warpcred_cr_nvidia_ready,
    make_warpcred_cr_nvidia_not_ready,
    make_warpcred_cr_weka_ready,
)


# ---------------------------------------------------------------------------
# Pure-helper tests (no monkeypatching needed)
# ---------------------------------------------------------------------------

def test_make_credential_slug_normalizes_and_truncates():
    assert main._make_credential_slug("My NGC Key #1") == "my-ngc-key-1"
    long_slug = main._make_credential_slug("a" * 60)
    assert long_slug.startswith("a" * 48)
    assert len(long_slug) == 48
    with pytest.raises(ValueError):
        main._make_credential_slug("---")


def test_build_credential_response_item_omits_secret_fields():
    item = main._build_credential_response_item(make_warpcred_cr_nvidia_ready())
    expected_keys = {"name", "namespace", "type", "displayName", "ready", "lastSyncTime", "derivedSecrets", "dockerSecretReady"}
    assert set(item.keys()) == expected_keys
    assert item["ready"] is True
    assert item["dockerSecretReady"] is True
    forbidden = {"key", "apiKey", "token", "secretRef", "NGC_API_KEY", "WEKA_API_TOKEN", "WEKA_API_USERNAME", "password"}
    for f in forbidden:
        assert f not in item, f"forbidden field present: {f}"


def test_build_credential_response_item_weka_storage_exposes_endpoint_only_from_status():
    item = main._build_credential_response_item(make_warpcred_cr_weka_ready(endpoint="https://w:14000"))
    assert item["endpoint"] == "https://w:14000"
    assert "WEKA_API_TOKEN" not in json.dumps(item)


# ---------------------------------------------------------------------------
# GET /api/credentials handler tests
# ---------------------------------------------------------------------------

def _patch_list_credentials(monkeypatch, items: list) -> None:
    """Helper: patch client and load_kube_config for list_credentials tests."""
    class CoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            return {"items": items}

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: SimpleNamespace())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)


def test_list_credentials_returns_shape_without_secret_values(monkeypatch):
    _patch_list_credentials(monkeypatch, [
        make_warpcred_cr_nvidia_ready(),
        make_warpcred_cr_weka_ready(),
    ])
    response = asyncio.run(main.list_credentials(namespace="default", type=None))
    body = json.loads(response.body)
    assert body["ok"] is True
    assert len(body["items"]) == 2
    dumped = json.dumps(body)
    for forbidden in ("NGC_API_KEY", "WEKA_API_TOKEN", "WEKA_API_USERNAME", "secretRef", "apiKey"):
        assert forbidden not in dumped, f"forbidden value '{forbidden}' found in response"


def test_list_credentials_type_filter_returns_only_ready(monkeypatch):
    _patch_list_credentials(monkeypatch, [
        make_warpcred_cr_nvidia_ready(name="ngc-ok"),
        make_warpcred_cr_nvidia_not_ready(name="ngc-bad"),
        make_warpcred_cr_weka_ready(),
    ])
    response = asyncio.run(main.list_credentials(namespace="default", type="nvidia-ngc"))
    body = json.loads(response.body)
    assert body["ok"] is True
    items = body["items"]
    assert len(items) == 1
    assert items[0]["ready"] is True


# ---------------------------------------------------------------------------
# POST /api/credentials handler tests
# ---------------------------------------------------------------------------

def _make_post_stubs(monkeypatch, existing_names=None):
    """Return an ops list and patch the client for create_credential tests."""
    ops = []
    if existing_names is None:
        existing_names = []

    class CoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            return {"items": [{"metadata": {"name": n}} for n in existing_names]}

        def create_namespaced_custom_object(self, **kwargs):
            ops.append(("create_namespaced_custom_object", kwargs))

    class CoreApiStub:
        def read_namespace(self, **kwargs):
            return SimpleNamespace()

        def create_namespaced_secret(self, **kwargs):
            ops.append(("create_namespaced_secret", kwargs))

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    return ops


def test_post_credential_nvidia_creates_secret_and_cr(monkeypatch):
    ops = _make_post_stubs(monkeypatch)
    response = asyncio.run(main.create_credential(
        display_name="My NGC",
        type="nvidia-ngc",
        namespace="default",
        key="nvapi_secret",
        username=None,
        endpoint=None,
    ))
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["name"] == "my-ngc"
    assert body["type"] == "nvidia-ngc"
    assert body["displayName"] == "My NGC"
    assert "nvapi_secret" not in json.dumps(body)

    secret_op = next(o for o in ops if o[0] == "create_namespaced_secret")
    secret_body = secret_op[1]["body"]
    assert secret_body.metadata.name == "warp-cred-my-ngc"
    assert secret_body.string_data == {"NGC_API_KEY": "nvapi_secret"}

    cr_op = next(o for o in ops if o[0] == "create_namespaced_custom_object")
    cr_kwargs = cr_op[1]
    assert cr_kwargs["plural"] == "warpcredentials"
    assert cr_kwargs["body"]["spec"]["type"] == "nvidia-ngc"
    assert cr_kwargs["body"]["spec"]["secretRef"] == {"name": "warp-cred-my-ngc", "key": "NGC_API_KEY"}


def test_post_credential_slug_collision_appends_suffix(monkeypatch):
    ops = _make_post_stubs(monkeypatch, existing_names=["my-ngc"])
    response = asyncio.run(main.create_credential(
        display_name="My NGC",
        type="nvidia-ngc",
        namespace="default",
        key="nvapi_secret2",
        username=None,
        endpoint=None,
    ))
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["name"] == "my-ngc-2"

    secret_op = next(o for o in ops if o[0] == "create_namespaced_secret")
    assert secret_op[1]["body"].metadata.name == "warp-cred-my-ngc-2"
    assert "nvapi_secret2" not in json.dumps(body)


def test_post_credential_weka_storage_persists_three_keys(monkeypatch):
    ops = _make_post_stubs(monkeypatch)
    response = asyncio.run(main.create_credential(
        display_name="WEKA Primary",
        type="weka-storage",
        namespace="default",
        key="weka-tok-xyz",
        username="admin",
        endpoint="https://w:14000",
    ))
    body = json.loads(response.body)
    assert body["ok"] is True

    secret_op = next(o for o in ops if o[0] == "create_namespaced_secret")
    assert secret_op[1]["body"].string_data == {
        "WEKA_API_USERNAME": "admin",
        "WEKA_API_TOKEN": "weka-tok-xyz",
        "WEKA_API_ENDPOINT": "https://w:14000",
    }

    cr_op = next(o for o in ops if o[0] == "create_namespaced_custom_object")
    cr_body = cr_op[1]["body"]
    assert cr_body["spec"]["endpoint"] == "https://w:14000"
    assert cr_body["spec"]["secretRef"]["key"] == "WEKA_API_TOKEN"

    dumped = json.dumps(body)
    assert "weka-tok-xyz" not in dumped
    assert "WEKA_API_TOKEN" not in dumped


def test_post_credential_invalid_type_returns_400(monkeypatch):
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    response = asyncio.run(main.create_credential(
        display_name="X",
        type="bogus",
        namespace="default",
        key="somekey",
        username=None,
        endpoint=None,
    ))
    assert response.status_code == 400
    body = json.loads(response.body)
    assert "somekey" not in json.dumps(body)


def test_post_credential_weka_missing_username_returns_400(monkeypatch):
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    response = asyncio.run(main.create_credential(
        display_name="X",
        type="weka-storage",
        namespace="default",
        key="tok",
        username=None,
        endpoint="https://w:14000",
    ))
    assert response.status_code == 400
    body = json.loads(response.body)
    body_str = json.dumps(body)
    assert "username" in body_str.lower()
    assert "tok" not in body_str


# ---------------------------------------------------------------------------
# DELETE /api/credentials/<name> handler tests
# ---------------------------------------------------------------------------

def test_delete_credential_deletes_cr_then_raw_secret_preserves_derived(monkeypatch):
    ops = []

    class CoApiStub:
        def delete_namespaced_custom_object(self, **kwargs):
            ops.append(("delete_namespaced_custom_object", kwargs))

    class CoreApiStub:
        def delete_namespaced_secret(self, **kwargs):
            ops.append(("delete_namespaced_secret", kwargs))

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)

    response = asyncio.run(main.delete_credential(name="my-ngc", namespace="default"))
    body = json.loads(response.body)
    assert body["ok"] is True
    assert len(ops) == 2
    assert ops[0][0] == "delete_namespaced_custom_object"
    assert ops[0][1]["plural"] == "warpcredentials"
    assert ops[0][1]["name"] == "my-ngc"
    assert ops[1][0] == "delete_namespaced_secret"
    assert ops[1][1]["name"] == "warp-cred-my-ngc"
    derived_names = {"warp-my-ngc-apikey", "warp-my-ngc-docker", "warp-my-ngc-token"}
    for op_name, op_kwargs in ops:
        if op_name == "delete_namespaced_secret":
            assert op_kwargs["name"] not in derived_names


def test_delete_credential_idempotent_on_secret_404(monkeypatch):
    from kubernetes.client.rest import ApiException

    class CoApiStub:
        def delete_namespaced_custom_object(self, **kwargs):
            pass

    class CoreApiStub:
        def delete_namespaced_secret(self, **kwargs):
            raise ApiException(status=404)

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)

    response = asyncio.run(main.delete_credential(name="my-ngc", namespace="default"))
    body = json.loads(response.body)
    assert body["ok"] is True
    assert response.status_code == 200


def test_delete_credential_invalid_name_returns_400_without_io(monkeypatch):
    called = {"count": 0}

    def _raise_if_called():
        called["count"] += 1
        raise AssertionError("client factory should not be called")

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: _raise_if_called())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: _raise_if_called())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)

    response = asyncio.run(main.delete_credential(name="Bad..Name", namespace="default"))
    assert response.status_code == 400
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# _assemble_weka_overview pure function tests
# ---------------------------------------------------------------------------

def test_assemble_weka_overview_pure_transform_shape():
    main._weka_overview_cache.clear()
    result = main._assemble_weka_overview(
        filesystems_resp=[{"name": "fs1", "uid": "deadbeef", "total_budget": 1000, "used_total": 500}],
        cluster_resp={"capacity": {"total_bytes": 2000, "used_bytes": 800}},
        containers_resp=[
            {"role": "BACKEND", "ip": "10.0.0.1"},
            {"role": "FRONTEND", "ip": "10.0.0.2"},
            {"role": "BACKEND", "ip": "127.0.0.1"},
        ],
        fetched_at_iso="2026-06-11T00:00:00Z",
    )
    expected = {
        "capacity": {
            "totalBytes": 2000,
            "usedBytes": 800,
            "availableBytes": 1200,
            "usedPercent": 40.0,
        },
        "filesystems": [{"name": "fs1", "totalBytes": 1000, "usedBytes": 500, "usedPercent": 50.0}],
        "backendNodes": [{"ip": "10.0.0.1"}],
        "fetchedAt": "2026-06-11T00:00:00Z",
    }
    assert result == expected


def test_assemble_weka_overview_tolerates_alt_field_names():
    main._weka_overview_cache.clear()
    result = main._assemble_weka_overview(
        filesystems_resp=[{"name": "fs2", "size": 4000, "used_size": 1000}],
        cluster_resp={"capacity": {"total": 8000, "used": 2000}},
        containers_resp=[{"mode": "BACKEND", "ip_address": "10.0.0.5"}],
        fetched_at_iso="2026-06-11T00:00:00Z",
    )
    assert result["filesystems"][0]["totalBytes"] == 4000
    assert result["filesystems"][0]["usedBytes"] == 1000
    assert result["capacity"]["totalBytes"] == 8000
    assert result["capacity"]["usedBytes"] == 2000
    assert result["backendNodes"] == [{"ip": "10.0.0.5"}]


def test_assemble_weka_overview_capacity_fallback_when_no_cluster_capacity_dict():
    main._weka_overview_cache.clear()
    result = main._assemble_weka_overview(
        filesystems_resp=[
            {"name": "a", "total_budget": 2000, "used_total": 800},
            {"name": "b", "total_budget": 1000, "used_total": 400},
        ],
        cluster_resp={"unrelated": "data"},
        containers_resp=[],
        fetched_at_iso="2026-06-11T00:00:00Z",
    )
    assert result["capacity"]["totalBytes"] == 3000
    assert result["capacity"]["usedBytes"] == 1200
    assert result["capacity"].get("capacity_source") == "fallback-sum"


# ---------------------------------------------------------------------------
# GET /api/weka/overview handler tests — cache, bust, namespace scoping
# ---------------------------------------------------------------------------

def _patch_weka_overview(monkeypatch, call_count: dict, credential_name="primary", ns="default"):
    """Common monkeypatching setup for the WEKA overview tests."""
    main._weka_overview_cache.clear()

    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    monkeypatch.setattr(
        main, "_resolve_weka_credential_secret",
        lambda cred, namespace: ("https://w:14000", "admin", "weka-tok"),
    )
    monkeypatch.setattr(
        main, "_weka_resolve_bearer_token",
        lambda endpoint, token: "BEARER-TOKEN-DO-NOT-LEAK",
    )

    def fake_get_json(url: str, headers: dict, timeout: float = 15.0):
        call_count["get"] += 1
        if "fileSystems" in url:
            return [{"name": "fs1", "total_budget": 100, "used_total": 50}]
        if "cluster" in url:
            return {"capacity": {"total_bytes": 200, "used_bytes": 100}}
        if "containers" in url:
            return []
        return {}

    monkeypatch.setattr(main, "_weka_get_json", fake_get_json)


def test_weka_overview_cache_hit_avoids_refetch(monkeypatch):
    call_count = {"get": 0}
    _patch_weka_overview(monkeypatch, call_count)

    r1 = asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=0))
    r2 = asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=0))

    assert call_count["get"] == 3
    b1 = json.loads(r1.body)
    b2 = json.loads(r2.body)
    assert b1["data"]["fetchedAt"] == b2["data"]["fetchedAt"]
    assert b2.get("cached") is True
    assert "BEARER-TOKEN-DO-NOT-LEAK" not in r2.body.decode("utf-8")
    assert "weka-tok" not in r2.body.decode("utf-8")
    assert "admin" not in r2.body.decode("utf-8")


def test_weka_overview_bust_query_bypasses_cache(monkeypatch):
    import datetime as _dt

    call_count = {"get": 0}
    _patch_weka_overview(monkeypatch, call_count)

    # Control time so fetchedAt differs between calls
    _fake_times = [
        _dt.datetime(2026, 6, 11, 0, 0, 0, tzinfo=_dt.timezone.utc),
        _dt.datetime(2026, 6, 11, 0, 0, 1, tzinfo=_dt.timezone.utc),
    ]
    _time_iter = iter(_fake_times)

    class _FakeDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return next(_time_iter)

    monkeypatch.setattr(main.datetime, "datetime", _FakeDatetime)

    r1 = asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=0))
    assert call_count["get"] == 3

    r2 = asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=1))
    assert call_count["get"] == 6

    fetched1 = json.loads(r1.body)["data"]["fetchedAt"]
    fetched2 = json.loads(r2.body)["data"]["fetchedAt"]
    assert fetched1 != fetched2


def test_weka_overview_namespace_scoped_cache(monkeypatch):
    call_count = {"get": 0}
    _patch_weka_overview(monkeypatch, call_count)

    asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=0))
    asyncio.run(main.get_weka_overview(credential="primary", namespace="aidp", bust=0))

    assert call_count["get"] == 6
    assert set(main._weka_overview_cache.keys()) == {"default/primary", "aidp/primary"}


def test_weka_overview_invalid_credential_name_returns_400_without_io(monkeypatch):
    called = {"count": 0}

    def _raise_if_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("I/O helper should not be called")

    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    monkeypatch.setattr(main, "_resolve_weka_credential_secret", _raise_if_called)
    monkeypatch.setattr(main, "_weka_resolve_bearer_token", _raise_if_called)
    monkeypatch.setattr(main, "_weka_get_json", _raise_if_called)

    response = asyncio.run(main.get_weka_overview(credential="Bad..Name", namespace="default", bust=0))
    assert response.status_code == 400
    assert called["count"] == 0


def test_weka_overview_auth_failure_returns_502_without_leak(monkeypatch):
    """A rejected token / unreachable cluster makes the cluster call fail → 502, no leak."""
    main._weka_overview_cache.clear()
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    monkeypatch.setattr(
        main, "_resolve_weka_credential_secret",
        lambda cred, ns: ("https://w:14000", "admin", "weka-tok"),
    )
    monkeypatch.setattr(
        main, "_weka_resolve_bearer_token",
        lambda endpoint, token: "BEARER-TOKEN-DO-NOT-LEAK",
    )

    def fake_get_json(url: str, headers: dict, timeout: float = 15.0):
        # Simulate the cluster (and all) calls failing on a rejected token.
        raise RuntimeError("WEKA API call failed: HTTP 401 https://w:14000?session=verysecret")

    monkeypatch.setattr(main, "_weka_get_json", fake_get_json)

    response = asyncio.run(main.get_weka_overview(credential="primary", namespace="default", bust=0))
    assert response.status_code == 502
    body_str = response.body.decode("utf-8")
    assert "verysecret" not in body_str
    assert "BEARER-TOKEN-DO-NOT-LEAK" not in body_str
    assert json.loads(body_str)["ok"] is False


# ---------------------------------------------------------------------------
# _resolve_weka_credential_secret tests
# ---------------------------------------------------------------------------

def test_resolve_weka_credential_secret_decodes_base64_secret(monkeypatch):
    weka_cr = make_warpcred_cr_weka_ready(endpoint="https://weka:14000")

    encoded_data = {
        "WEKA_API_ENDPOINT": base64.b64encode(b"https://weka:14000").decode(),
        "WEKA_API_USERNAME": base64.b64encode(b"admin").decode(),
        "WEKA_API_TOKEN": base64.b64encode(b"tok-abc123").decode(),
    }

    class CoApiStub:
        def get_namespaced_custom_object(self, **kwargs):
            return weka_cr

    class CoreApiStub:
        def read_namespaced_secret(self, **kwargs):
            return SimpleNamespace(data=encoded_data)

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreApiStub())

    endpoint, username, token = main._resolve_weka_credential_secret("primary", "default")
    assert endpoint == "https://weka:14000"
    assert username == "admin"
    assert token == "tok-abc123"


def test_resolve_weka_credential_secret_missing_key_raises_runtime(monkeypatch):
    weka_cr = make_warpcred_cr_weka_ready()

    # Only provide WEKA_API_TOKEN — missing username and endpoint
    partial_data = {
        "WEKA_API_TOKEN": base64.b64encode(b"tok").decode(),
    }

    class CoApiStub:
        def get_namespaced_custom_object(self, **kwargs):
            return weka_cr

    class CoreApiStub:
        def read_namespaced_secret(self, **kwargs):
            return SimpleNamespace(data=partial_data)

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: CoreApiStub())

    with pytest.raises(RuntimeError, match="WEKA_API_USERNAME"):
        main._resolve_weka_credential_secret("primary", "default")


# ---------------------------------------------------------------------------
# _get_credentials_by_type helper tests (Plan 25-01, Task 1)
# ---------------------------------------------------------------------------

def _patch_get_credentials_by_type(monkeypatch, items: list) -> None:
    """Patch client and load_kube_config for _get_credentials_by_type tests."""
    class CoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            return {"items": items}

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)


def test_get_credentials_by_type_returns_known_keys(monkeypatch):
    """Test 1: Returns dict with exactly the three known credential type keys."""
    _patch_get_credentials_by_type(monkeypatch, [])
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert set(result.keys()) == {"nvidia-ngc", "huggingface", "weka-storage"}


def test_get_credentials_by_type_groups_ready_items(monkeypatch):
    """Test 2: Groups ready CRs by type correctly, empty list for missing types."""
    _patch_get_credentials_by_type(monkeypatch, [
        make_warpcred_cr_nvidia_ready("ngc1"),
        make_warpcred_cr_weka_ready("weka1"),
    ])
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert len(result["nvidia-ngc"]) == 1
    assert len(result["weka-storage"]) == 1
    assert len(result["huggingface"]) == 0


def test_get_credentials_by_type_returns_empty_on_api_exception(monkeypatch):
    """Test 3a: Falls back to empty dict-of-lists on ApiException without re-raising."""
    from kubernetes.client.exceptions import ApiException

    class RaisingCoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            raise ApiException(status=500, reason="Internal Error")

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: RaisingCoApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_returns_empty_on_connection_error(monkeypatch):
    """Test 3b: Falls back to empty dict-of-lists on ConnectionError without re-raising."""
    class RaisingCoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            raise ConnectionError("refused")

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: RaisingCoApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_returns_empty_on_timeout_error(monkeypatch):
    """Test 3c: Falls back to empty dict-of-lists on TimeoutError without re-raising."""
    class RaisingCoApiStub:
        def list_namespaced_custom_object(self, **kwargs):
            raise TimeoutError("timed out")

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: RaisingCoApiStub())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_drops_unknown_type(monkeypatch):
    """Test 4: CRs with unknown spec.type are silently dropped."""
    unknown_cr = {
        "apiVersion": "warp.io/v1alpha1",
        "kind": "WarpCredential",
        "metadata": {"name": "mystery", "namespace": "default"},
        "spec": {"type": "unknown-type", "displayName": "Mystery"},
        "status": {"conditions": [{"type": "KeyReady", "status": "True"}]},
    }
    _patch_get_credentials_by_type(monkeypatch, [unknown_cr])
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result["nvidia-ngc"] == []
    assert result["huggingface"] == []
    assert result["weka-storage"] == []


def test_get_credentials_by_type_filters_non_ready(monkeypatch):
    """Test 6: Non-ready CRs are filtered out by the helper (ready-filter at helper level)."""
    _patch_get_credentials_by_type(monkeypatch, [
        make_warpcred_cr_weka_ready("weka-ok"),
        make_warpcred_cr_nvidia_not_ready("ngc-bad"),
    ])
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert len(result["weka-storage"]) == 1
    assert len(result["nvidia-ngc"]) == 0


# ---------------------------------------------------------------------------
# blueprint_detail context injection tests (Plan 25-01, Task 2)
# ---------------------------------------------------------------------------

def _make_blueprint_detail_stubs(monkeypatch):
    """Patch all external calls needed for blueprint_detail route tests."""
    sentinel = {"nvidia-ngc": [{"name": "test-ngc"}], "huggingface": [], "weka-storage": []}

    monkeypatch.setattr(main, "get_auth_status", lambda: {"details": {"namespace": "test-ns"}})
    monkeypatch.setattr(main, "get_cluster_status", lambda: {"cpu_nodes": 1, "gpu_nodes": 0})

    async def _cred_stub(ns):
        return sentinel

    monkeypatch.setattr(main, "_get_credentials_by_type", _cred_stub)
    return sentinel


def test_blueprint_detail_injects_credentials_by_type(monkeypatch):
    """Test 2: blueprint_detail template context contains credentials_by_type key."""
    sentinel = _make_blueprint_detail_stubs(monkeypatch)
    request_stub = SimpleNamespace(
        headers={}, cookies={}, query_params={}, url=SimpleNamespace(path="/blueprint/openfold"),
        scope={"type": "http"},
    )
    response = asyncio.run(main.blueprint_detail(request_stub, name="openfold"))
    assert hasattr(response, "context"), "TemplateResponse should have .context attribute"
    assert "credentials_by_type" in response.context
    assert response.context["credentials_by_type"] is sentinel


def test_blueprint_detail_falls_back_to_default_namespace(monkeypatch):
    """Test 4: When get_auth_status returns {}, _get_credentials_by_type is called with 'default'."""
    called_with = []

    async def capture_ns(ns):
        called_with.append(ns)
        return {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}

    monkeypatch.setattr(main, "get_auth_status", lambda: {})
    monkeypatch.setattr(main, "get_cluster_status", lambda: {"cpu_nodes": 1, "gpu_nodes": 0})
    monkeypatch.setattr(main, "_get_credentials_by_type", capture_ns)

    request_stub = SimpleNamespace(
        headers={}, cookies={}, query_params={}, url=SimpleNamespace(path="/blueprint/openfold"),
        scope={"type": "http"},
    )
    asyncio.run(main.blueprint_detail(request_stub, name="openfold"))
    assert called_with == ["default"], f"Expected ['default'], got {called_with}"


def test_blueprint_detail_preserves_existing_context_keys(monkeypatch):
    """Test 5: All pre-existing context keys remain present after adding credentials_by_type."""
    sentinel = _make_blueprint_detail_stubs(monkeypatch)
    request_stub = SimpleNamespace(
        headers={}, cookies={}, query_params={}, url=SimpleNamespace(path="/blueprint/openfold"),
        scope={"type": "http"},
    )
    response = asyncio.run(main.blueprint_detail(request_stub, name="openfold"))
    ctx = response.context
    for key in ("request", "name", "yaml_path", "status", "requirements", "meets",
                "oss_img_b64", "aidp_img_b64", "logo_b64", "glocomp_logo_b64",
                "tokenvisor_logo_b64", "tokenvisor_arch_b64"):
        assert key in ctx, f"Missing expected context key: {key}"
