from __future__ import annotations

import inspect

import pytest
import yaml

from webapp import main


PHASE_ONE_REQUIREMENTS = {"APPLY-06", "APPLY-07"}


def test_apply_gateway_fixture_targets_existing_runtime_path(apply_gateway_input: dict) -> None:
    payload = yaml.safe_load(apply_gateway_input["yaml_text"])

    assert PHASE_ONE_REQUIREMENTS == {"APPLY-06", "APPLY-07"}
    assert payload == apply_gateway_input["document"]
    assert payload["kind"] == apply_gateway_input["expected_runtime_kind"]
    assert payload["apiVersion"] == apply_gateway_input["expected_runtime_api_version"]
    assert apply_gateway_input["expected_apply_result"] == {"applied": ["WekaAppStore"]}


def test_existing_apply_helpers_remain_the_phase_one_handoff_seam() -> None:
    file_apply_source = inspect.getsource(main.apply_blueprint_with_namespace)
    content_apply_source = inspect.getsource(main.apply_blueprint_content_with_namespace)

    assert "WekaAppStore" in file_apply_source
    assert "CustomObjectsApi" in file_apply_source
    assert "WekaAppStore" in content_apply_source
    assert "CustomObjectsApi" in content_apply_source


def test_future_apply_gateway_module_can_wrap_existing_helpers() -> None:
    gateway = pytest.importorskip(
        "webapp.planning.apply_gateway",
        reason="Phase 1 apply gateway implementation has not landed yet.",
    )

    assert hasattr(gateway, "__file__")
