from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")

import webapp.main as main


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

    request = SimpleNamespace(
        headers={}, cookies={}, query_params={},
        url=SimpleNamespace(path="/blueprint/neuralmesh-aidp"),
        scope={"type": "http"},
    )
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

    request = SimpleNamespace(
        headers={}, cookies={}, query_params={},
        url=SimpleNamespace(path="/blueprint/neuralmesh-aidp"),
        scope={"type": "http"},
    )
    asyncio.run(main.blueprint_detail(request, name="neuralmesh-aidp"))
    assert captured_ns == ["default"]
