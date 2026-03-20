from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi.testclient import TestClient

from webapp import main
from webapp.planning import (
    PlanningSessionFollowUpError,
    PlanningSessionService,
    PlanningSessionStateError,
)


class InspectionToolsStub:
    def __init__(self, snapshots: list[dict[str, Any]]) -> None:
        self._snapshots = [deepcopy(snapshot) for snapshot in snapshots]
        self.calls: list[dict[str, Any]] = []

    def assess_fit(
        self,
        *,
        correlation_id: str | None = None,
        required_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "correlation_id": correlation_id,
                "required_domains": list(required_domains or []),
            }
        )
        index = min(len(self.calls) - 1, len(self._snapshots) - 1)
        snapshot = deepcopy(self._snapshots[index])
        snapshot["correlation_id"] = correlation_id
        fit_status = "blocked" if snapshot.get("metadata", {}).get("fit_status") == "blocked" else "fit"
        fit_findings = {
            "status": fit_status,
            "notes": [
                "Inspection snapshot confirms the required planning domains are complete."
                if fit_status == "fit"
                else "Inspection snapshot lacks a supported-family fit for this request."
            ],
            "blockers": [] if fit_status == "fit" else ["No supported blueprint family matched the request."],
            "domains": deepcopy(snapshot["domains"]),
            "inspection_snapshot": deepcopy(snapshot),
        }
        return {
            "correlation_id": correlation_id,
            "inspection_snapshot": snapshot,
            "fit_findings": fit_findings,
        }


def _build_service(
    *,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> tuple[PlanningSessionService, InspectionToolsStub]:
    timestamps = iter(
        [
            "2026-03-20T12:00:00Z",
            "2026-03-20T12:01:00Z",
            "2026-03-20T12:02:00Z",
            "2026-03-20T12:03:00Z",
            "2026-03-20T12:04:00Z",
            "2026-03-20T12:05:00Z",
            "2026-03-20T12:06:00Z",
            "2026-03-20T12:07:00Z",
            "2026-03-20T12:08:00Z",
            "2026-03-20T12:09:00Z",
        ]
    )
    correlation_ids = iter(["corr-301", "corr-302", "corr-303", "corr-304"])
    question_ids = iter(["fq-301", "fq-302", "fq-303", "fq-304"])
    planner_calls: list[dict[str, Any]] = []
    inspection_tools = InspectionToolsStub([planning_snapshot_payload, planning_snapshot_payload])

    def planner(**kwargs: Any) -> dict[str, Any]:
        planner_calls.append(kwargs)
        payload = deepcopy(valid_plan_payload)
        payload["request_summary"] = "Deploy the enterprise research stack for domain data search."
        if len(planner_calls) == 1:
            payload["namespace_strategy"]["target_namespace"] = None
            payload["unresolved_questions"] = [
                {
                    "question": "Which namespace should receive the deployment?",
                    "field_path": "namespace_strategy.target_namespace",
                    "blocking": True,
                    "install_critical": True,
                }
            ]
            return {
                "assistant_message": "I need the target namespace before I can finish the draft.",
                "draft_summary": "Waiting on namespace selection.",
                "plan": payload,
                "follow_ups": payload["unresolved_questions"],
            }

        payload["namespace_strategy"]["target_namespace"] = "ai-platform"
        payload["unresolved_questions"] = []
        payload["reasoning_summary"] = "The draft now includes the selected ai-platform namespace."
        return {
            "assistant_message": "The draft plan now targets the ai-platform namespace.",
            "draft_summary": "Draft updated with the selected namespace.",
            "plan": payload,
            "follow_ups": [],
        }

    service = PlanningSessionService(
        session_store=planning_session_store,
        inspection_tools=inspection_tools,
        planner=planner,
        now=lambda: next(timestamps),
        correlation_id_factory=lambda: next(correlation_ids),
        question_id_factory=lambda: next(question_ids),
    )
    service._planner_calls = planner_calls  # type: ignore[attr-defined]
    return service, inspection_tools


def _install_service(service: PlanningSessionService) -> None:
    main.app.state.planning_session_service = service


def test_full_conversational_session_flow_persists_transcript_revisions_and_reload_state(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    service, inspection_tools = _build_service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        valid_plan_payload=valid_plan_payload,
    )
    _install_service(service)

    create_response = planning_test_client.post(
        "/planning/sessions",
        data={"request_text": "Deploy an enterprise research assistant for our domain data."},
        headers={"accept": "application/json"},
    )

    assert create_response.status_code == 200
    assert create_response.json() == {
        "ok": True,
        "session_id": "session-001",
        "status": "active",
        "redirect_url": "/planning/sessions/session-001",
    }

    initial_page = planning_test_client.get("/planning/sessions/session-001")
    assert initial_page.status_code == 200
    assert "Which namespace should receive the deployment?" in initial_page.text
    assert "I need the target namespace before I can finish the draft." in initial_page.text
    assert "blocked" in initial_page.text

    answer_response = planning_test_client.post(
        "/planning/sessions/session-001/follow-ups/fq-301",
        data={"answer": "Use the ai-platform namespace."},
        headers={"accept": "application/json"},
    )

    assert answer_response.status_code == 200
    assert answer_response.json() == {
        "ok": True,
        "session_id": "session-001",
        "status": "active",
        "assistant_message": "The draft plan now targets the ai-platform namespace.",
    }

    reloaded_page = planning_test_client.get("/planning/sessions/session-001")
    assert reloaded_page.status_code == 200
    assert "Use the ai-platform namespace." in reloaded_page.text
    assert "The draft plan now targets the ai-platform namespace." in reloaded_page.text
    assert "No unanswered follow-up questions." in reloaded_page.text
    assert "Draft updated with the selected namespace." in reloaded_page.text

    persisted = planning_session_store.load_session("session-001")
    assert [turn.role for turn in persisted.turns] == ["user", "assistant", "user", "assistant"]
    assert persisted.follow_ups[0].question_id == "fq-301"
    assert persisted.follow_ups[0].status == "answered"
    assert persisted.follow_ups[0].answer == "Use the ai-platform namespace."
    assert persisted.unanswered_follow_ups == []
    assert persisted.latest_revision is not None
    assert persisted.latest_revision.status == "draft"
    assert persisted.latest_revision.structured_plan["namespace_strategy"]["target_namespace"] == "ai-platform"
    assert persisted.latest_revision.fit_findings["inspection_snapshot"]["correlation_id"] == "corr-302"
    assert persisted.latest_revision.metadata["inspection_snapshot"]["correlation_id"] == "corr-302"
    assert len(persisted.draft_revisions) == 2
    assert inspection_tools.calls == [
        {
            "correlation_id": "corr-301",
            "required_domains": ["cpu", "memory", "gpu", "namespaces", "storage_classes", "weka"],
        },
        {
            "correlation_id": "corr-302",
            "required_domains": ["cpu", "memory", "gpu", "namespaces", "storage_classes", "weka"],
        },
    ]


def test_no_supported_family_and_lifecycle_operations_stay_explicit_draft_session_state(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
) -> None:
    timestamps = iter(
        [
            "2026-03-20T12:10:00Z",
            "2026-03-20T12:11:00Z",
            "2026-03-20T12:12:00Z",
            "2026-03-20T12:13:00Z",
            "2026-03-20T12:14:00Z",
        ]
    )
    correlation_ids = iter(["corr-401", "corr-402"])
    inspection_tools = InspectionToolsStub([planning_snapshot_payload])

    def planner(**_: Any) -> dict[str, Any]:
        raise AssertionError("planner should not run when the request does not match a supported family")

    service = PlanningSessionService(
        session_store=planning_session_store,
        inspection_tools=inspection_tools,
        planner=planner,
        now=lambda: next(timestamps),
        correlation_id_factory=lambda: next(correlation_ids),
    )
    _install_service(service)

    create_response = planning_test_client.post(
        "/planning/sessions",
        data={"request_text": "Install a plain PostgreSQL database for finance reporting."},
        follow_redirects=False,
    )
    assert create_response.status_code == 303
    assert create_response.headers["location"] == "/planning/sessions/session-001"

    blocked_page = planning_test_client.get("/planning/sessions/session-001")
    assert blocked_page.status_code == 200
    assert "I could not map this request to a supported blueprint family." in blocked_page.text
    assert "blocked" in blocked_page.text

    restart_response = planning_test_client.post(
        "/planning/sessions/session-001/restart",
        headers={"accept": "application/json"},
    )
    assert restart_response.status_code == 200
    assert restart_response.json() == {
        "ok": True,
        "session_id": "session-002",
        "status": "active",
        "redirect_url": "/planning/sessions/session-002",
        "restarted_from_session_id": "session-001",
    }

    abandon_response = planning_test_client.post(
        "/planning/sessions/session-002/abandon",
        headers={"accept": "application/json"},
    )
    assert abandon_response.status_code == 200
    assert abandon_response.json() == {
        "ok": True,
        "session_id": "session-002",
        "status": "abandoned",
    }

    rejected_turn = planning_test_client.post(
        "/planning/sessions/session-002/message",
        data={"message": "Try the openfold stack instead."},
        headers={"accept": "application/json"},
    )
    assert rejected_turn.status_code == 409
    assert rejected_turn.json()["detail"] == (
        "planning session 'session-002' is abandoned and cannot accept new turns"
    )

    rejected_follow_up = planning_test_client.post(
        "/planning/sessions/session-001/follow-ups/fq-999",
        data={"answer": "ai-platform"},
        headers={"accept": "application/json"},
    )
    assert rejected_follow_up.status_code == 409
    assert rejected_follow_up.json()["detail"] == (
        "planning session 'session-001' is restarted and cannot accept new turns"
    )

    original = planning_session_store.load_session("session-001")
    replacement = planning_session_store.load_session("session-002")
    assert original.status == "restarted"
    assert original.latest_revision is not None
    assert original.latest_revision.status == "blocked"
    assert original.latest_revision.structured_plan is None
    assert replacement.status == "abandoned"
    assert replacement.latest_revision is None
