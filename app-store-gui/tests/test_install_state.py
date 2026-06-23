from __future__ import annotations

import asyncio
import json

import webapp.main as main


def test_cr_gui_variables_from_annotation():
    cr = {
        "metadata": {"annotations": {"warp.io/gui-variables": json.dumps({"a": "1", "b": "2"})}},
        "spec": {"appStack": {"variables": {"ignored": "x"}}},
    }
    # Annotation wins over spec.appStack.variables.
    assert main._cr_gui_variables(cr) == {"a": "1", "b": "2"}


def test_cr_gui_variables_fallback_to_spec():
    cr = {"metadata": {}, "spec": {"appStack": {"variables": {"sc": "weka"}}}}
    assert main._cr_gui_variables(cr) == {"sc": "weka"}


def test_cr_gui_variables_empty_when_absent():
    assert main._cr_gui_variables({}) == {}
    assert main._cr_gui_variables({"metadata": {"annotations": {"warp.io/gui-variables": "not json"}}}) == {}


def test_blueprint_cr_identity(tmp_path):
    bp = tmp_path / "bp.yaml"
    bp.write_text(
        "apiVersion: warp.io/v1alpha1\n"
        "kind: WekaAppStore\n"
        "metadata:\n"
        "  name: weka-aidp\n"
        "  namespace: rag\n"
        "spec: {}\n"
    )
    assert main._blueprint_cr_identity(str(bp)) == ("weka-aidp", "rag")


def test_blueprint_cr_identity_none_for_missing():
    assert main._blueprint_cr_identity(None) == (None, None)
    assert main._blueprint_cr_identity("/no/such/file.yaml") == (None, None)


def test_wekaappstore_exists_returns_variables(monkeypatch):
    monkeypatch.setattr(main, "load_kube_config", lambda: None)

    class _Api:
        def get_namespaced_custom_object(self, **kwargs):
            return {
                "metadata": {"annotations": {"warp.io/gui-variables": json.dumps({"keycloak_url": "https://kc"})}},
                "status": {"appStackPhase": "Ready"},
            }

    monkeypatch.setattr(main.client, "CustomObjectsApi", lambda: _Api())
    resp = asyncio.run(main.wekaappstore_exists(name="weka-aidp", namespace="rag"))
    body = json.loads(resp.body)
    assert body["ok"] is True and body["exists"] is True
    assert body["phase"] == "Ready"
    assert body["variables"] == {"keycloak_url": "https://kc"}
