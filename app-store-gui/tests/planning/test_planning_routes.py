from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi.testclient import TestClient

from webapp import main
from webapp.planning import PlanningSessionService


class InspectionToolsStub:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot

    def assess_fit(
        self,
        *,
        correlation_id: str | None = None,
        required_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        snapshot = deepcopy(self.snapshot)
        snapshot["correlation_id"] = correlation_id
        fit_findings = {
            "status": "fit",
            "notes": ["Inspection snapshot confirms the required planning domains are complete."],
            "blockers": [],
            "domains": deepcopy(snapshot["domains"]),
            "inspection_snapshot": deepcopy(snapshot),
        }
        return {
            "correlation_id": correlation_id,
            "inspection_snapshot": snapshot,
            "fit_findings": fit_findings,
        }


def _build_service(*, planning_session_store: Any, planning_snapshot_payload: dict[str, Any], valid_plan_payload: dict[str, Any]) -> PlanningSessionService:
    timestamps = iter(
        [
            "2026-03-20T11:00:00Z",
            "2026-03-20T11:01:00Z",
            "2026-03-20T11:02:00Z",
            "2026-03-20T11:03:00Z",
            "2026-03-20T11:04:00Z",
            "2026-03-20T11:05:00Z",
            "2026-03-20T11:06:00Z",
            "2026-03-20T11:07:00Z",
            "2026-03-20T11:08:00Z",
            "2026-03-20T11:09:00Z",
        ]
    )
    correlation_ids = iter(["corr-101", "corr-102", "corr-103"])
    question_ids = iter(["fq-101", "fq-102", "fq-103"])
    planner_calls: list[dict[str, Any]] = []

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
        return {
            "assistant_message": "The draft plan now targets the ai-platform namespace.",
            "draft_summary": "Draft updated with the selected namespace.",
            "plan": payload,
            "follow_ups": [],
        }

    service = PlanningSessionService(
        session_store=planning_session_store,
        inspection_tools=InspectionToolsStub(planning_snapshot_payload),
        planner=planner,
        now=lambda: next(timestamps),
        correlation_id_factory=lambda: next(correlation_ids),
        question_id_factory=lambda: next(question_ids),
    )
    service._planner_calls = planner_calls  # type: ignore[attr-defined]
    return service


def _install_service(service: PlanningSessionService) -> None:
    main.app.state.planning_session_service = service


def test_index_exposes_planning_entrypoint(planning_test_client: TestClient) -> None:
    response = planning_test_client.get("/")

    assert response.status_code == 200
    assert "Start Planning Session" in response.text
    assert "Plan With Chat" in response.text
    assert "/planning/sessions" in response.text


def test_create_and_view_planning_session_renders_persisted_transcript_and_pending_questions(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    service = _build_service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        valid_plan_payload=valid_plan_payload,
    )
    _install_service(service)

    create_response = planning_test_client.post(
        "/planning/sessions",
        data={"request_text": "Deploy an enterprise research assistant for our domain data."},
        follow_redirects=False,
    )

    assert create_response.status_code == 303
    session_url = create_response.headers["location"]
    assert session_url.endswith("/planning/sessions/session-001")

    page_response = planning_test_client.get(session_url)

    assert page_response.status_code == 200
    assert "Deploy an enterprise research assistant for our domain data." in page_response.text
    assert "I need the target namespace before I can finish the draft." in page_response.text
    assert "Which namespace should receive the deployment?" in page_response.text
    assert "Draft Status" in page_response.text
    assert "blocked" in page_response.text
    assert "Deploy Preview" not in page_response.text
    assert "Apply Plan" not in page_response.text


def test_follow_up_submission_updates_persisted_history_and_clears_pending_question(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    service = _build_service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        valid_plan_payload=valid_plan_payload,
    )
    _install_service(service)
    initial = service.start_session("Deploy an enterprise research assistant for our domain data.")

    response = planning_test_client.post(
        f"/planning/sessions/{initial.session.session_id}/follow-ups/{initial.follow_ups[0].question_id}",
        data={"answer": "Use the ai-platform namespace."},
        follow_redirects=False,
    )

    assert response.status_code == 303

    page_response = planning_test_client.get(f"/planning/sessions/{initial.session.session_id}")

    assert page_response.status_code == 200
    assert "Use the ai-platform namespace." in page_response.text
    assert "The draft plan now targets the ai-platform namespace." in page_response.text
    assert "No unanswered follow-up questions." in page_response.text
    assert "Question ID fq-101" not in page_response.text

    persisted = planning_session_store.load_session(initial.session.session_id)
    assert persisted.unanswered_follow_ups == []
    assert persisted.follow_ups[0].answer == "Use the ai-platform namespace."


def test_restart_and_abandon_routes_stay_in_planning_session_state_management(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    service = _build_service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        valid_plan_payload=valid_plan_payload,
    )
    _install_service(service)
    initial = service.start_session("Deploy an enterprise research assistant for our domain data.")

    abandon_response = planning_test_client.post(
        f"/planning/sessions/{initial.session.session_id}/abandon",
        headers={"accept": "application/json"},
    )

    assert abandon_response.status_code == 200
    assert abandon_response.json() == {
        "ok": True,
        "session_id": "session-001",
        "status": "abandoned",
    }

    restart_response = planning_test_client.post(
        f"/planning/sessions/{initial.session.session_id}/restart",
        headers={"accept": "application/json"},
    )

    assert restart_response.status_code == 200
    assert restart_response.json()["redirect_url"] == "/planning/sessions/session-002"
    replacement = planning_session_store.load_session("session-002")
    original = planning_session_store.load_session("session-001")
    assert replacement.restarted_from_session_id == "session-001"
    assert original.replacement_session_id == "session-002"


def test_planning_state_stream_is_planning_specific(
    planning_test_client: TestClient,
    planning_session_store: Any,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    service = _build_service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        valid_plan_payload=valid_plan_payload,
    )
    _install_service(service)
    initial = service.start_session("Deploy an enterprise research assistant for our domain data.")

    response = planning_test_client.get(f"/planning/sessions/{initial.session.session_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "session-state"' in response.text
    assert '"pending_follow_up_ids": ["fq-101"]' in response.text
    assert "deploy-stream" not in response.text
    assert "apply" not in response.text
