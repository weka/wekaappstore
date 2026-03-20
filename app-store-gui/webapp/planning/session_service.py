from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from .family_matcher import SupportedFamilyMatcher
from .models import (
    PlanningSession,
    PlanningSessionDraftRevision,
    PlanningSessionFollowUp,
    SupportedFamilyMatch,
)
from .session_store import PlanningSessionRepository
from .validator import validate_structured_plan


class PlanningDraftBuilder(Protocol):
    def __call__(
        self,
        *,
        session: PlanningSession,
        request_summary: str,
        conversation: List[Dict[str, Any]],
        family_match: SupportedFamilyMatch,
        inspection_snapshot: Mapping[str, Any],
        fit_findings: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@dataclass(slots=True)
class PlanningSessionTransition:
    correlation_id: str
    assistant_message: str
    session: PlanningSession
    family_match: SupportedFamilyMatch
    inspection_snapshot: Dict[str, Any]
    fit_findings: Dict[str, Any]
    draft_revision: PlanningSessionDraftRevision
    follow_ups: List[PlanningSessionFollowUp]


class PlanningSessionService:
    def __init__(
        self,
        *,
        session_store: PlanningSessionRepository,
        inspection_tools: Any,
        planner: PlanningDraftBuilder,
        family_matcher: Optional[SupportedFamilyMatcher] = None,
        now: Optional[Callable[[], str]] = None,
        correlation_id_factory: Optional[Callable[[], str]] = None,
        question_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._session_store = session_store
        self._inspection_tools = inspection_tools
        self._planner = planner
        self._family_matcher = family_matcher or SupportedFamilyMatcher()
        self._now = now or self._build_counter("2026-03-20T10:00:", suffix="Z")
        self._correlation_id_factory = correlation_id_factory or self._build_counter("corr")
        self._question_id_factory = question_id_factory or self._build_counter("fq")

    def start_session(
        self,
        request_text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSessionTransition:
        session = self._session_store.create_session(
            request_summary=request_text.strip(),
            metadata=dict(metadata or {}),
        )
        return self.process_turn(session.session_id, message=request_text, metadata=metadata)

    def answer_follow_up(
        self,
        session_id: str,
        *,
        question_id: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSessionTransition:
        return self.process_turn(
            session_id,
            message=answer,
            question_id=question_id,
            metadata=metadata,
        )

    def process_turn(
        self,
        session_id: str,
        *,
        message: str,
        question_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSessionTransition:
        correlation_id = self._correlation_id_factory()
        user_metadata = dict(metadata or {})
        user_metadata["correlation_id"] = correlation_id

        session = self._session_store.append_turn(
            session_id,
            role="user",
            message=message,
            question_id=question_id,
            metadata=user_metadata,
        )

        family_match = self._resolve_family_match(session)
        assessment = self._inspection_tools.assess_fit(
            correlation_id=correlation_id,
            required_domains=family_match.required_domains or None,
        )
        inspection_snapshot = deepcopy(dict(assessment["inspection_snapshot"]))
        fit_findings = deepcopy(dict(assessment["fit_findings"]))
        revision_number = (0 if session.latest_revision is None else session.latest_revision.revision_number) + 1

        if family_match.status != "matched":
            assistant_message = (
                "I could not map this request to a supported blueprint family. "
                "Please revise the request toward ai-agent-enterprise-research, nvidia-vss, or openfold."
            )
            draft_revision = PlanningSessionDraftRevision(
                revision_number=revision_number,
                created_at=self._now(),
                summary="No supported blueprint family matched the current request.",
                status="blocked",
                structured_plan=None,
                fit_findings=fit_findings,
                unanswered_question_ids=[],
                metadata={
                    "correlation_id": correlation_id,
                    "family_match": family_match.to_dict(),
                    "inspection_snapshot": inspection_snapshot,
                },
            )
            stored_session = self._session_store.append_turn(
                session_id,
                role="assistant",
                message=assistant_message,
                metadata={
                    "correlation_id": correlation_id,
                    "family_match": family_match.to_dict(),
                },
                draft_revision=draft_revision,
            )
            return PlanningSessionTransition(
                correlation_id=correlation_id,
                assistant_message=assistant_message,
                session=stored_session,
                family_match=family_match,
                inspection_snapshot=inspection_snapshot,
                fit_findings=fit_findings,
                draft_revision=draft_revision,
                follow_ups=[],
            )

        planner_output = self._planner(
            session=session,
            request_summary=session.request_summary or self._request_summary_from_session(session),
            conversation=self._conversation_payload(session),
            family_match=family_match,
            inspection_snapshot=inspection_snapshot,
            fit_findings=fit_findings,
        )
        plan_payload = self._plan_payload_from_result(
            planner_output,
            session=session,
            family_match=family_match,
            fit_findings=fit_findings,
        )
        assistant_message = str(
            planner_output.get("assistant_message")
            or "I updated the draft plan using the latest inspection snapshot."
        )
        follow_ups = self._build_follow_ups(
            planner_output=planner_output,
            session=session,
            revision_number=revision_number,
        )
        validation = validate_structured_plan(plan_payload)
        draft_status = self._draft_status(validation.valid, fit_findings, follow_ups)
        draft_summary = str(
            planner_output.get("draft_summary")
            or plan_payload.get("reasoning_summary")
            or plan_payload.get("request_summary")
            or "Updated draft plan."
        )
        if not validation.valid:
            draft_summary = "; ".join(error.message for error in validation.errors)
            assistant_message = (
                planner_output.get("assistant_message")
                or "I could not produce a valid structured draft plan from this conversation turn."
            )

        draft_revision = PlanningSessionDraftRevision(
            revision_number=revision_number,
            created_at=self._now(),
            summary=draft_summary,
            status=draft_status,
            structured_plan=(plan_payload if not validation.valid else validation.plan.to_dict()),
            fit_findings=fit_findings,
            unanswered_question_ids=[follow_up.question_id for follow_up in follow_ups],
            metadata={
                "correlation_id": correlation_id,
                "family_match": family_match.to_dict(),
                "inspection_snapshot": inspection_snapshot,
                "validation_errors": [error.to_dict() for error in validation.errors],
            },
        )
        stored_session = self._session_store.append_turn(
            session_id,
            role="assistant",
            message=assistant_message,
            metadata={
                "correlation_id": correlation_id,
                "family_match": family_match.to_dict(),
            },
            follow_ups=follow_ups,
            draft_revision=draft_revision,
        )
        return PlanningSessionTransition(
            correlation_id=correlation_id,
            assistant_message=assistant_message,
            session=stored_session,
            family_match=family_match,
            inspection_snapshot=inspection_snapshot,
            fit_findings=fit_findings,
            draft_revision=draft_revision,
            follow_ups=follow_ups,
        )

    def _plan_payload_from_result(
        self,
        planner_output: Mapping[str, Any],
        *,
        session: PlanningSession,
        family_match: SupportedFamilyMatch,
        fit_findings: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if "plan" in planner_output:
            raw_payload = planner_output["plan"]
        else:
            raw_payload = {
                key: value
                for key, value in planner_output.items()
                if key not in {"assistant_message", "draft_summary", "follow_ups"}
            }
        if not isinstance(raw_payload, Mapping):
            raise ValueError("planner output must provide a structured-plan mapping")

        payload = deepcopy(dict(raw_payload))
        unresolved_questions = payload.get("unresolved_questions")
        if unresolved_questions is None:
            unresolved_questions = planner_output.get("follow_ups", [])
        payload["unresolved_questions"] = [
            {
                "question": item["question"],
                "blocking": bool(item.get("blocking", False)),
                "install_critical": bool(item.get("install_critical", False)),
            }
            for item in unresolved_questions
        ]
        payload["request_summary"] = str(
            payload.get("request_summary")
            or session.request_summary
            or self._request_summary_from_session(session)
        )
        payload["blueprint_family"] = family_match.family
        payload["fit_findings"] = deepcopy(dict(fit_findings))
        return payload

    def _build_follow_ups(
        self,
        *,
        planner_output: Mapping[str, Any],
        session: PlanningSession,
        revision_number: int,
    ) -> List[PlanningSessionFollowUp]:
        follow_ups: list[PlanningSessionFollowUp] = []
        for item in planner_output.get("follow_ups", planner_output.get("unresolved_questions", [])):
            if not isinstance(item, Mapping):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            question_id = self._resolve_question_id(session, item)
            follow_ups.append(
                PlanningSessionFollowUp(
                    question_id=question_id,
                    question=question,
                    field_path=item.get("field_path"),
                    blocking=bool(item.get("blocking", False)),
                    install_critical=bool(item.get("install_critical", False)),
                    status=str(item.get("status") or "pending"),
                    asked_at=self._now(),
                    revision_number=revision_number,
                    metadata=deepcopy(dict(item.get("metadata") or {})),
                )
            )
        return follow_ups

    def _resolve_family_match(self, session: PlanningSession) -> SupportedFamilyMatch:
        latest_revision = session.latest_revision
        if latest_revision and isinstance(latest_revision.structured_plan, Mapping):
            family = latest_revision.structured_plan.get("blueprint_family")
            if isinstance(family, str) and family:
                return SupportedFamilyMatch(
                    status="matched",
                    family=family,
                    reason="Preserved the previously matched supported family for this session replay.",
                    required_domains=list(
                        latest_revision.metadata.get("family_match", {}).get("required_domains", [])
                    ),
                    metadata={"source": "latest_revision"},
                )
        return self._family_matcher.match(self._request_summary_from_session(session))

    @staticmethod
    def _request_summary_from_session(session: PlanningSession) -> str:
        messages = [session.request_summary or ""]
        messages.extend(turn.message for turn in session.turns if turn.role == "user")
        return " ".join(message.strip() for message in messages if message and message.strip())

    @staticmethod
    def _conversation_payload(session: PlanningSession) -> List[Dict[str, Any]]:
        return [
            {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "message": turn.message,
                "question_id": turn.question_id,
                "revision_number": turn.revision_number,
                "metadata": deepcopy(turn.metadata),
            }
            for turn in session.turns
        ]

    @staticmethod
    def _draft_status(
        validation_ok: bool,
        fit_findings: Mapping[str, Any],
        follow_ups: List[PlanningSessionFollowUp],
    ) -> str:
        if not validation_ok:
            return "blocked"
        if str(fit_findings.get("status")) == "blocked":
            return "blocked"
        if any(follow_up.status == "pending" and follow_up.blocking for follow_up in follow_ups):
            return "blocked"
        return "draft"

    def _resolve_question_id(self, session: PlanningSession, item: Mapping[str, Any]) -> str:
        existing_id = item.get("question_id")
        if isinstance(existing_id, str) and existing_id:
            return existing_id
        field_path = item.get("field_path")
        question = item.get("question")
        for follow_up in session.follow_ups:
            if follow_up.field_path == field_path and follow_up.question == question:
                return follow_up.question_id
        return self._question_id_factory()

    @staticmethod
    def _build_counter(prefix: str, *, suffix: str = "") -> Callable[[], str]:
        counter = 0

        def next_value() -> str:
            nonlocal counter
            counter += 1
            return f"{prefix}-{counter:04d}{suffix}"

        return next_value
