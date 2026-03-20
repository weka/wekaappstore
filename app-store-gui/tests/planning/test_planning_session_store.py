from __future__ import annotations

import inspect
import json

from webapp.planning import (
    LocalPlanningSessionStore,
    PlanningSessionDraftRevision,
    PlanningSessionFollowUp,
)
from webapp.planning import session_store as session_store_module


def test_create_and_load_session_persists_backend_owned_contract(planning_session_store: LocalPlanningSessionStore) -> None:
    created = planning_session_store.create_session(
        request_summary="Deploy a GPU-ready research stack.",
        metadata={"source": "chat-ui"},
    )

    reloaded = planning_session_store.load_session(created.session_id)

    assert reloaded.session_id == "session-001"
    assert reloaded.status == "active"
    assert reloaded.request_summary == "Deploy a GPU-ready research stack."
    assert reloaded.metadata == {"source": "chat-ui"}
    assert reloaded.turns == []
    assert reloaded.follow_ups == []
    assert reloaded.draft_revisions == []
    assert (planning_session_store.root_dir / "session-001.json").exists()


def test_append_turn_replays_turns_follow_ups_and_draft_revisions(
    planning_session_store: LocalPlanningSessionStore,
) -> None:
    session = planning_session_store.create_session(request_summary="Plan an OpenFold deployment.")
    follow_up = PlanningSessionFollowUp(
        question_id="fq-001",
        question="Which namespace should receive the deployment?",
        field_path="namespace_strategy.target_namespace",
        blocking=True,
        install_critical=True,
        asked_at="2026-03-20T10:02:00Z",
        revision_number=1,
    )
    draft_revision = PlanningSessionDraftRevision(
        revision_number=1,
        created_at="2026-03-20T10:02:00Z",
        summary="Initial draft requires a target namespace before validation can continue.",
        status="blocked",
        structured_plan={"blueprint_family": "openfold"},
        fit_findings={"status": "assumed-fit"},
        unanswered_question_ids=["fq-001"],
    )

    after_assistant_turn = planning_session_store.append_turn(
        session.session_id,
        role="assistant",
        message="I can draft that, but I still need the target namespace.",
        follow_ups=[follow_up],
        draft_revision=draft_revision,
    )
    after_user_turn = planning_session_store.append_turn(
        session.session_id,
        role="user",
        message="Use the ai-platform namespace.",
        question_id="fq-001",
    )

    assert [turn.turn_id for turn in after_user_turn.turns] == ["turn-001", "turn-002"]
    assert [turn.role for turn in after_user_turn.turns] == ["assistant", "user"]
    assert after_assistant_turn.latest_revision is not None
    assert after_assistant_turn.latest_revision.status == "blocked"
    assert after_assistant_turn.latest_revision.unanswered_question_ids == ["fq-001"]
    assert len(after_user_turn.follow_ups) == 1
    assert after_user_turn.follow_ups[0].status == "answered"
    assert after_user_turn.follow_ups[0].answer == "Use the ai-platform namespace."
    assert after_user_turn.unanswered_follow_ups == []


def test_restart_session_preserves_auditability_and_starts_new_active_session(
    planning_session_store: LocalPlanningSessionStore,
) -> None:
    session = planning_session_store.create_session(request_summary="Deploy Nvidia VSS.")
    planning_session_store.append_turn(
        session.session_id,
        role="assistant",
        message="I need the filesystem target before drafting.",
        follow_ups=[
            PlanningSessionFollowUp(
                question_id="fq-002",
                question="Which WEKA filesystem should back this deployment?",
                field_path="components[0].values.storage.filesystem",
                blocking=True,
                asked_at="2026-03-20T10:02:00Z",
                revision_number=1,
            )
        ],
        draft_revision=PlanningSessionDraftRevision(
            revision_number=1,
            created_at="2026-03-20T10:02:00Z",
            summary="Waiting on filesystem selection before continuing.",
            status="blocked",
            unanswered_question_ids=["fq-002"],
        ),
    )

    replacement = planning_session_store.restart_session(
        session.session_id,
        metadata={"restart_reason": "user_requested_restart"},
    )
    original = planning_session_store.load_session(session.session_id)

    assert original.status == "restarted"
    assert original.replacement_session_id == "session-002"
    assert len(original.turns) == 1
    assert len(original.draft_revisions) == 1
    assert replacement.session_id == "session-002"
    assert replacement.status == "active"
    assert replacement.restarted_from_session_id == "session-001"
    assert replacement.restart_count == 1
    assert replacement.turns == []
    assert replacement.follow_ups == []
    assert replacement.draft_revisions == []
    assert replacement.metadata["restart_reason"] == "user_requested_restart"


def test_abandon_session_marks_session_and_latest_revision_abandoned(
    planning_session_store: LocalPlanningSessionStore,
) -> None:
    session = planning_session_store.create_session(request_summary="Deploy enterprise research.")
    planning_session_store.append_turn(
        session.session_id,
        role="assistant",
        message="Drafting the plan now.",
        draft_revision=PlanningSessionDraftRevision(
            revision_number=1,
            created_at="2026-03-20T10:02:00Z",
            summary="Initial draft is ready for review.",
            status="draft",
            structured_plan={"blueprint_family": "ai-agent-enterprise-research"},
        ),
    )

    abandoned = planning_session_store.abandon_session(
        session.session_id,
        metadata={"abandoned_by": "user"},
    )
    stored_payload = json.loads(
        (planning_session_store.root_dir / f"{session.session_id}.json").read_text(encoding="utf-8")
    )

    assert abandoned.status == "abandoned"
    assert abandoned.abandoned_at == "2026-03-20T10:02:00Z"
    assert abandoned.draft_revisions[-1].status == "abandoned"
    assert abandoned.metadata["abandoned_by"] == "user"
    assert stored_payload["status"] == "abandoned"
    assert stored_payload["draft_revisions"][-1]["status"] == "abandoned"
    assert sorted(path.name for path in planning_session_store.root_dir.iterdir()) == ["session-001.json"]


def test_session_store_never_touches_apply_gateway_runtime_path() -> None:
    source = inspect.getsource(session_store_module)

    assert "apply_gateway" not in source
    assert "apply_yaml" not in source
