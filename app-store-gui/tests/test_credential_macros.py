from __future__ import annotations

import asyncio
import json
import os
import pytest
from types import SimpleNamespace

os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")

import webapp.main as main

from .test_credentials_api import (
    make_warpcred_cr_nvidia_ready,
    make_warpcred_cr_nvidia_not_ready,
    make_warpcred_cr_weka_ready,
)


# ---------------------------------------------------------------------------
# Stub injection helper
# ---------------------------------------------------------------------------

def _patch_list_credentials(monkeypatch, items=None, raises=None) -> None:
    """Patch CustomObjectsApi, CoreV1Api, and load_kube_config for helper tests.

    If `raises` is provided, list_namespaced_custom_object raises that exception.
    If `items` is provided, returns {"items": items}.
    """
    if raises is not None:
        exc = raises

        class CoApiStub:
            def list_namespaced_custom_object(self, **kwargs):
                raise exc

    else:
        _items = items or []

        class CoApiStub:
            def list_namespaced_custom_object(self, **kwargs):
                return {"items": _items}

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: CoApiStub())
    monkeypatch.setattr(main.client, "CoreV1Api", lambda: SimpleNamespace())
    monkeypatch.setattr(main, "load_kube_config", lambda: None)


# ---------------------------------------------------------------------------
# _get_credentials_by_type helper tests
# ---------------------------------------------------------------------------

def test_get_credentials_by_type_groups_by_type(monkeypatch):
    _patch_list_credentials(monkeypatch, items=[
        make_warpcred_cr_nvidia_ready(),
        make_warpcred_cr_weka_ready(),
    ])
    result = asyncio.run(main._get_credentials_by_type("default"))

    assert set(result.keys()) == {"nvidia-ngc", "huggingface", "weka-storage"}
    assert len(result["nvidia-ngc"]) == 1
    assert result["nvidia-ngc"][0]["name"] == "my-ngc"
    assert len(result["weka-storage"]) == 1
    assert result["weka-storage"][0]["endpoint"] == "https://weka:14000"
    assert len(result["huggingface"]) == 0


def test_get_credentials_by_type_returns_empty_lists_on_api_exception(monkeypatch):
    _patch_list_credentials(monkeypatch, raises=main.ApiException(status=500))
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_returns_empty_lists_on_connection_error(monkeypatch):
    _patch_list_credentials(monkeypatch, raises=ConnectionError("k8s down"))
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_returns_empty_lists_on_timeout_error(monkeypatch):
    _patch_list_credentials(monkeypatch, raises=TimeoutError("k8s timeout"))
    result = asyncio.run(main._get_credentials_by_type("default"))
    assert result == {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}


def test_get_credentials_by_type_drops_unknown_type(monkeypatch):
    unknown_cr = {
        "apiVersion": "warp.io/v1alpha1",
        "kind": "WarpCredential",
        "metadata": {"name": "my-unknown", "namespace": "default"},
        "spec": {
            "type": "unknown-type",
            "displayName": "Unknown Cred",
            "secretRef": {"name": "warp-cred-unknown", "key": "SECRET_KEY"},
        },
        "status": {
            "conditions": [{"type": "KeyReady", "status": "True"}],
            "derivedSecrets": [],
            "lastSyncTime": "2026-06-11T00:00:00Z",
        },
    }
    _patch_list_credentials(monkeypatch, items=[
        make_warpcred_cr_nvidia_ready(),
        unknown_cr,
    ])
    result = asyncio.run(main._get_credentials_by_type("default"))

    assert len(result["nvidia-ngc"]) == 1
    assert len(result["weka-storage"]) == 0
    assert len(result["huggingface"]) == 0
    # The unknown CR must not appear under any key
    all_names = [c["name"] for lst in result.values() for c in lst]
    assert "my-unknown" not in all_names


def test_get_credentials_by_type_filters_non_ready_credentials(monkeypatch):
    _patch_list_credentials(monkeypatch, items=[
        make_warpcred_cr_nvidia_ready(name="ngc-good"),
        make_warpcred_cr_nvidia_not_ready(name="ngc-bad"),
        make_warpcred_cr_weka_ready(name="weka-good"),
    ])
    result = asyncio.run(main._get_credentials_by_type("default"))

    assert len(result["nvidia-ngc"]) == 1
    assert result["nvidia-ngc"][0]["name"] == "ngc-good"
    assert result["nvidia-ngc"][0]["ready"] is True
    assert len(result["weka-storage"]) == 1
    assert result["weka-storage"][0]["name"] == "weka-good"
    # No entry in any list has ready=False
    assert all(c.get("ready") is True for lst in result.values() for c in lst)


# ---------------------------------------------------------------------------
# blueprint_detail context injection tests
# ---------------------------------------------------------------------------

def test_blueprint_detail_injects_credentials_by_type_into_context(monkeypatch):
    monkeypatch.setattr(main, "get_auth_status", lambda: {"details": {"namespace": "test-ns"}})
    monkeypatch.setattr(main, "get_cluster_status", lambda: {"cpu_nodes": 4, "gpu_nodes": 1})

    async def _stub_helper(ns):
        return {
            "nvidia-ngc": [{"name": "sentinel-cred", "displayName": "S", "type": "nvidia-ngc", "ready": True}],
            "huggingface": [],
            "weka-storage": [],
        }

    monkeypatch.setattr(main, "_get_credentials_by_type", _stub_helper)

    request = SimpleNamespace()
    response = asyncio.run(main.blueprint_detail(request, name="neuralmesh-aidp"))
    assert response.context["credentials_by_type"]["nvidia-ngc"][0]["name"] == "sentinel-cred"


def test_blueprint_detail_falls_back_to_default_namespace(monkeypatch):
    monkeypatch.setattr(main, "get_auth_status", lambda: {})
    monkeypatch.setattr(main, "get_cluster_status", lambda: {"cpu_nodes": 4, "gpu_nodes": 1})

    captured_ns = []

    async def _stub_helper(ns):
        captured_ns.append(ns)
        return {"nvidia-ngc": [], "huggingface": [], "weka-storage": []}

    monkeypatch.setattr(main, "_get_credentials_by_type", _stub_helper)

    asyncio.run(main.blueprint_detail(SimpleNamespace(), name="neuralmesh-aidp"))
    assert captured_ns == ["default"]
