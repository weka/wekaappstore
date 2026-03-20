from __future__ import annotations

from kubernetes.client.rest import ApiException
import pytest

from webapp.inspection.weka import collect_weka_inspection
from webapp.planning.inspection_tools import PlanningInspectionTools


class _WekaObjectsStub:
    def __init__(self, response: dict | Exception) -> None:
        self._response = response

    def list_cluster_custom_object(self, *, group: str, version: str, plural: str) -> dict:
        assert group == "weka.weka.io"
        assert plural == "wekaclusters"
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_collect_weka_inspection_reads_capacity_and_filesystems() -> None:
    snapshot = collect_weka_inspection(
        custom_objects_api=_WekaObjectsStub(
            {
                "items": [
                    {
                        "metadata": {"name": "weka-a", "namespace": "weka"},
                        "status": {
                            "status": "Ready",
                            "stats": {
                                "capacity": {"totalBytes": 5000},
                                "filesystem": {"totalAvailableCapacity": 3200},
                                "filesystems": [
                                    {
                                        "name": "weka-home",
                                        "totalBytes": 4000,
                                        "availableBytes": 2500,
                                    }
                                ],
                            },
                        },
                    }
                ]
            }
        )
    )

    weka = snapshot["domains"]["weka"]
    assert weka["status"] == "complete"
    assert weka["observed"]["cluster_total_bytes"] == 5000
    assert weka["observed"]["cluster_free_bytes"] == 3200
    assert weka["observed"]["filesystems"] == [
        {"name": "weka-home", "total_bytes": 4000, "free_bytes": 2500}
    ]


def test_collect_weka_inspection_returns_partial_when_operator_data_is_incomplete() -> None:
    snapshot = collect_weka_inspection(
        custom_objects_api=_WekaObjectsStub(
            {
                "items": [
                    {
                        "metadata": {"name": "weka-a", "namespace": "weka"},
                        "status": {"status": "Ready", "stats": {}},
                    }
                ]
            }
        )
    )

    weka = snapshot["domains"]["weka"]
    blocker_codes = {blocker["code"] for blocker in weka["blockers"]}
    assert weka["status"] == "partial"
    assert {"weka_capacity_missing", "weka_free_capacity_missing", "weka_filesystems_missing"} <= blocker_codes


def test_planning_inspection_tools_bound_intents_and_audit_log() -> None:
    tools = PlanningInspectionTools(
        cluster_collector=lambda: {
            "captured_at": "2026-03-20T00:00:00Z",
            "domains": {"cpu": {"status": "complete", "required": True, "blockers": [], "notes": []}},
        },
        weka_collector=lambda: {
            "captured_at": "2026-03-20T00:00:00Z",
            "domains": {"weka": {"status": "partial", "required": True, "blockers": [], "notes": []}},
        },
    )

    planning_snapshot = tools.inspect("planning_snapshot", correlation_id="corr-123")
    fit = tools.assess_fit(correlation_id="corr-123", required_domains=["cpu", "weka"])

    assert planning_snapshot["result"]["correlation_id"] == "corr-123"
    assert sorted(planning_snapshot["result"]["domains"]) == ["cpu", "weka"]
    assert fit["fit_findings"]["status"] == "blocked"
    assert fit["fit_findings"]["inspection_snapshot"]["correlation_id"] == "corr-123"
    assert [event["intent"] for event in tools.audit_log] == ["planning_snapshot", "planning_snapshot"]

    with pytest.raises(ValueError):
        tools.inspect("kubectl", correlation_id="corr-123")


def test_collect_weka_inspection_marks_missing_operator_as_unavailable() -> None:
    snapshot = collect_weka_inspection(
        custom_objects_api=_WekaObjectsStub(ApiException(status=404, reason="missing"))
    )

    weka = snapshot["domains"]["weka"]
    assert weka["status"] == "unavailable"
    assert weka["blockers"][0]["code"] == "weka_inspection_failed"
