from __future__ import annotations

import pytest
from kubernetes.client.rest import ApiException

from webapp.inspection.weka import collect_weka_inspection
from webapp.planning.inspection_tools import PlanningInspectionTools


class _CustomObjectsApi:
    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    def list_cluster_custom_object(self, *, group: str, version: str, plural: str):
        self.calls.append((group, version, plural))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_collect_weka_inspection_returns_capacity_and_filesystem_inventory(
    mocked_weka_cluster_payload: dict,
) -> None:
    api = _CustomObjectsApi(mocked_weka_cluster_payload)

    snapshot = collect_weka_inspection(
        load_kube_config=lambda: None,
        custom_objects_api=api,
    )

    domain = snapshot["domains"]["weka"]
    assert api.calls == [("weka.weka.io", "v1alpha1", "wekaclusters")]
    assert domain["status"] == "complete"
    assert domain["observed"]["cluster_total_bytes"] == 2199023255552
    assert domain["observed"]["cluster_free_bytes"] == 1649267441664
    assert domain["observed"]["filesystems"][0]["name"] == "weka-home"
    assert domain["blockers"] == []


def test_collect_weka_inspection_returns_partial_when_operator_status_is_incomplete(
    mocked_weka_cluster_payload: dict,
) -> None:
    payload = {"items": [mocked_weka_cluster_payload["items"][0].copy()]}
    payload["items"][0]["status"] = {
        "status": "Ready",
        "stats": {},
    }

    snapshot = collect_weka_inspection(
        load_kube_config=lambda: None,
        custom_objects_api=_CustomObjectsApi(payload),
    )

    domain = snapshot["domains"]["weka"]
    assert domain["status"] == "partial"
    assert {blocker["code"] for blocker in domain["blockers"]} == {
        "weka_capacity_missing",
        "weka_free_capacity_missing",
        "weka_filesystems_missing",
    }


def test_collect_weka_inspection_returns_unavailable_when_operator_is_missing() -> None:
    snapshot = collect_weka_inspection(
        load_kube_config=lambda: None,
        custom_objects_api=_CustomObjectsApi(ApiException(status=404, reason="not found")),
    )

    domain = snapshot["domains"]["weka"]
    assert domain["status"] == "unavailable"
    assert domain["blockers"][0]["code"] == "weka_inspection_failed"


def test_planning_inspection_tools_limits_supported_intents_and_records_audit(
    mocked_weka_cluster_payload: dict,
) -> None:
    tools = PlanningInspectionTools(
        cluster_collector=lambda: {"domains": {"cpu": {"status": "complete"}}},
        weka_collector=lambda: collect_weka_inspection(
            load_kube_config=lambda: None,
            custom_objects_api=_CustomObjectsApi(mocked_weka_cluster_payload),
        ),
    )

    cluster_result = tools.inspect("cluster_snapshot", correlation_id="corr-cluster")
    weka_result = tools.inspect("weka_storage", correlation_id="corr-weka")

    assert cluster_result["audit"]["intent"] == "cluster_snapshot"
    assert weka_result["result"]["domains"]["weka"]["status"] == "complete"
    assert [entry["correlation_id"] for entry in tools.audit_log] == ["corr-cluster", "corr-weka"]

    with pytest.raises(ValueError, match="Unsupported inspection intent"):
        tools.inspect("kubectl_exec")
