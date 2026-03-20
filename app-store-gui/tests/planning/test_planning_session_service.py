from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from webapp.planning import (
    LocalPlanningSessionStore,
    PlanningSessionFollowUpError,
    PlanningSessionService,
    PlanningSessionStateError,
)
from webapp.planning.family_matcher import SupportedFamilyMatcher


class InspectionToolsStub:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
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


def _service(
    *,
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
    planner: Any,
) -> PlanningSessionService:
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
        ]
    )
    correlation_ids = iter(["corr-001", "corr-002", "corr-003"])
    question_ids = iter(["fq-001", "fq-002", "fq-003"])
    return PlanningSessionService(
        session_store=planning_session_store,
        inspection_tools=InspectionToolsStub(planning_snapshot_payload),
        planner=planner,
        now=lambda: next(timestamps),
        correlation_id_factory=lambda: next(correlation_ids),
        question_id_factory=lambda: next(question_ids),
    )


def test_supported_family_matcher_maps_openfold_requests_deterministically() -> None:
    matcher = SupportedFamilyMatcher()

    result = matcher.match("Deploy an OpenFold protein folding workflow with a WEKA filesystem.")

    assert result.status == "matched"
    assert result.family == "openfold"
    assert "openfold" in result.matched_terms
    assert "protein folding" in result.matched_terms


def test_supported_family_matcher_reports_explicit_no_fit() -> None:
    matcher = SupportedFamilyMatcher()

    result = matcher.match("Install a generic PostgreSQL database with no AI stack.")

    assert result.status == "no_supported_family"
    assert result.family is None
    assert "Supported families:" in result.reason


def test_start_session_creates_revision_with_follow_up_and_correlation_aware_inspection(
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    planner_calls: list[dict[str, Any]] = []

    def planner(**kwargs: Any) -> dict[str, Any]:
        planner_calls.append(kwargs)
        payload = deepcopy(valid_plan_payload)
        payload["request_summary"] = "Deploy the enterprise research stack for domain data search."
        payload["unresolved_questions"] = [
            {
                "question": "Which namespace should receive the deployment?",
                "field_path": "namespace_strategy.target_namespace",
                "blocking": True,
                "install_critical": True,
            }
        ]
        return {
            "assistant_message": "I can draft the research stack, but I still need the target namespace.",
            "draft_summary": "Initial draft requires the deployment namespace before validation can continue.",
            "plan": payload,
            "follow_ups": payload["unresolved_questions"],
        }

    service = _service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        planner=planner,
    )

    transition = service.start_session("Deploy an enterprise research assistant for our domain data.")

    assert transition.correlation_id == "corr-001"
    assert transition.family_match.family == "ai-agent-enterprise-research"
    assert [turn.role for turn in transition.session.turns] == ["user", "assistant"]
    assert transition.assistant_message.startswith("I can draft the research stack")
    assert transition.draft_revision.status == "blocked"
    assert transition.draft_revision.unanswered_question_ids == ["fq-001"]
    assert transition.follow_ups[0].question_id == "fq-001"
    assert transition.follow_ups[0].status == "pending"
    assert transition.fit_findings["inspection_snapshot"]["correlation_id"] == "corr-001"
    assert transition.draft_revision.fit_findings["inspection_snapshot"]["correlation_id"] == "corr-001"
    assert planner_calls[0]["conversation"][0]["message"] == "Deploy an enterprise research assistant for our domain data."
    assert planner_calls[0]["family_match"].family == "ai-agent-enterprise-research"


def test_follow_up_answer_replays_session_and_updates_draft_with_latest_snapshot(
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
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

    service = _service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        planner=planner,
    )

    first_transition = service.start_session("Deploy an enterprise research assistant for our domain data.")
    second_transition = service.answer_follow_up(
        first_transition.session.session_id,
        question_id=first_transition.follow_ups[0].question_id,
        answer="Use the ai-platform namespace.",
    )

    assert second_transition.correlation_id == "corr-002"
    assert second_transition.family_match.family == "ai-agent-enterprise-research"
    assert second_transition.family_match.metadata["source"] == "latest_revision"
    assert [turn.role for turn in second_transition.session.turns] == ["user", "assistant", "user", "assistant"]
    assert second_transition.session.follow_ups[0].question_id == "fq-001"
    assert second_transition.session.follow_ups[0].status == "answered"
    assert second_transition.session.follow_ups[0].answer == "Use the ai-platform namespace."
    assert second_transition.session.unanswered_follow_ups == []
    assert second_transition.draft_revision.status == "draft"
    assert second_transition.draft_revision.structured_plan["namespace_strategy"]["target_namespace"] == "ai-platform"
    assert second_transition.draft_revision.fit_findings["inspection_snapshot"]["correlation_id"] == "corr-002"
    assert planner_calls[1]["conversation"][-1]["message"] == "Use the ai-platform namespace."
    assert len(planner_calls) == 2


def test_unsupported_family_returns_explicit_no_fit_without_invoking_planner(
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
) -> None:
    def planner(**_: Any) -> dict[str, Any]:
        raise AssertionError("planner should not run when the request has no supported family match")

    service = _service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        planner=planner,
    )

    transition = service.start_session("Install a plain PostgreSQL database for finance reporting.")

    assert transition.family_match.status == "no_supported_family"
    assert transition.draft_revision.status == "blocked"
    assert transition.draft_revision.structured_plan is None
    assert transition.follow_ups == []
    assert "supported blueprint family" in transition.assistant_message


def test_answer_follow_up_rejects_unknown_or_resolved_question_ids(
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    planner_calls: list[dict[str, Any]] = []

    def planner(**kwargs: Any) -> dict[str, Any]:
        planner_calls.append(kwargs)
        payload = deepcopy(valid_plan_payload)
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
        payload["unresolved_questions"] = []
        return {
            "assistant_message": "The draft plan now targets the ai-platform namespace.",
            "draft_summary": "Draft updated with the selected namespace.",
            "plan": payload,
            "follow_ups": [],
        }

    service = _service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        planner=planner,
    )

    transition = service.start_session("Deploy an enterprise research assistant for our domain data.")

    with pytest.raises(PlanningSessionFollowUpError, match="does not have follow-up 'fq-999'"):
        service.answer_follow_up(
            transition.session.session_id,
            question_id="fq-999",
            answer="Use ai-platform.",
        )

    service.answer_follow_up(
        transition.session.session_id,
        question_id=transition.follow_ups[0].question_id,
        answer="Use the ai-platform namespace.",
    )

    with pytest.raises(
        PlanningSessionFollowUpError,
        match="follow-up 'fq-001' is answered and cannot be answered again",
    ):
        service.answer_follow_up(
            transition.session.session_id,
            question_id=transition.follow_ups[0].question_id,
            answer="Use ai-platform again.",
        )


def test_process_turn_rejects_non_active_sessions(
    planning_session_store: LocalPlanningSessionStore,
    planning_snapshot_payload: dict[str, Any],
    valid_plan_payload: dict[str, Any],
) -> None:
    def planner(**_: Any) -> dict[str, Any]:
        return {
            "assistant_message": "Draft updated.",
            "draft_summary": "Draft updated.",
            "plan": deepcopy(valid_plan_payload),
            "follow_ups": [],
        }

    service = _service(
        planning_session_store=planning_session_store,
        planning_snapshot_payload=planning_snapshot_payload,
        planner=planner,
    )

    transition = service.start_session("Deploy an enterprise research assistant for our domain data.")
    planning_session_store.abandon_session(transition.session.session_id, metadata={"abandoned_by": "user"})

    with pytest.raises(
        PlanningSessionStateError,
        match="planning session 'session-001' is abandoned and cannot accept new turns",
    ):
        service.process_turn(transition.session.session_id, message="Try the openfold stack instead.")
