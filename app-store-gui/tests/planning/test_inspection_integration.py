from __future__ import annotations

import importlib

import pytest


def test_build_planning_inspection_snapshot_merges_domains_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = importlib.import_module("webapp.main")

    class ToolsStub:
        def inspect(self, intent: str, *, correlation_id: str) -> dict:
            if intent == "cluster_snapshot":
                return {
                    "audit": {"intent": intent, "correlation_id": correlation_id},
                    "result": {
                        "captured_at": "2026-03-20T00:00:00Z",
                        "domains": {"cpu": {"status": "complete", "required": True, "blockers": [], "notes": []}},
                    },
                }
            if intent == "weka_storage":
                return {
                    "audit": {"intent": intent, "correlation_id": correlation_id},
                    "result": {
                        "captured_at": "2026-03-20T00:00:00Z",
                        "domains": {"weka": {"status": "complete", "required": True, "blockers": [], "notes": []}},
                    },
                }
            raise AssertionError(intent)

    monkeypatch.setattr(main, "PLANNING_INSPECTION_TOOLS", ToolsStub())

    snapshot = main.build_planning_inspection_snapshot(correlation_id="corr-merge")

    assert snapshot["correlation_id"] == "corr-merge"
    assert sorted(snapshot["domains"]) == ["cpu", "weka"]
    assert snapshot["sources"]["cluster"]["intent"] == "cluster_snapshot"
    assert snapshot["sources"]["weka"]["intent"] == "weka_storage"


def test_build_structured_plan_preview_uses_inspection_fit_findings(
    valid_plan_payload: dict,
) -> None:
    main = importlib.import_module("webapp.main")
    inspection_snapshot = {
        "captured_at": "2026-03-20T00:00:00Z",
        "correlation_id": "corr-preview",
        "domains": {
            "cpu": {"status": "complete", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {}, "notes": [], "blockers": []},
            "memory": {"status": "complete", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {}, "notes": [], "blockers": []},
            "gpu": {"status": "partial", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {"inventory": [{"count": 4}]}, "notes": [], "blockers": []},
            "namespaces": {"status": "complete", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {}, "notes": [], "blockers": []},
            "storage_classes": {"status": "complete", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {}, "notes": [], "blockers": []},
            "weka": {"status": "complete", "required": True, "freshness": {"captured_at": "2026-03-20T00:00:00Z"}, "observed": {}, "notes": [], "blockers": []},
        },
    }

    preview = main.build_structured_plan_preview(
        valid_plan_payload,
        inspection_snapshot=inspection_snapshot,
        correlation_id="corr-preview",
    )

    assert preview["valid"] is True
    assert preview["correlation_id"] == "corr-preview"
    assert preview["fit_findings"]["status"] == "blocked"
    assert preview["fit_findings"]["blockers"][0]["domain"] == "gpu"
    assert preview["inspection_snapshot"]["correlation_id"] == "corr-preview"


def test_build_structured_plan_preview_tags_yaml_generation_failures(
    monkeypatch: pytest.MonkeyPatch,
    valid_plan_payload: dict,
) -> None:
    main = importlib.import_module("webapp.main")

    monkeypatch.setattr(
        main,
        "compile_plan_to_wekaappstore",
        lambda plan: (_ for _ in ()).throw(main.PlanCompilationError("render failed")),
    )

    preview = main.build_structured_plan_preview(valid_plan_payload, correlation_id="corr-yaml")

    assert preview["valid"] is False
    assert preview["failure_stage"] == "yaml_generation"
    assert preview["errors"][0]["stage"] == "yaml_generation"
    assert "corr-yaml" in preview["errors"][0]["message"]


def test_build_structured_plan_preview_tags_inspection_failures(
    monkeypatch: pytest.MonkeyPatch,
    valid_plan_payload: dict,
) -> None:
    main = importlib.import_module("webapp.main")

    monkeypatch.setattr(
        main,
        "build_fit_findings_from_inspection",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("inspection snapshot unreadable")),
    )

    preview = main.build_structured_plan_preview(
        valid_plan_payload,
        inspection_snapshot={"captured_at": "2026-03-20T00:00:00Z", "domains": {}},
        correlation_id="corr-inspection",
    )

    assert preview["valid"] is False
    assert preview["failure_stage"] == "inspection"
    assert preview["errors"][0]["stage"] == "inspection"
    assert "corr-inspection" in preview["errors"][0]["message"]


def test_execute_structured_plan_apply_tags_apply_handoff_failures(
    monkeypatch: pytest.MonkeyPatch,
    valid_plan_payload: dict,
) -> None:
    main = importlib.import_module("webapp.main")

    class GatewayStub:
        def apply_content(self, content: str, namespace: str) -> dict:
            raise RuntimeError("apply gateway unreachable")

    monkeypatch.setattr(main, "PLANNING_APPLY_GATEWAY", GatewayStub())

    response = main.execute_structured_plan_apply(
        valid_plan_payload,
        correlation_id="corr-apply",
    )

    assert response["ok"] is False
    assert response["failure_stage"] == "apply_handoff"
    assert response["validation"]["errors"][0]["stage"] == "apply_handoff"
    assert "corr-apply" in response["validation"]["errors"][0]["message"]
