from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

from .models import (
    PlanningSession,
    PlanningSessionDraftRevision,
    PlanningSessionFollowUp,
    PlanningSessionTurn,
)


class PlanningSessionNotFoundError(KeyError):
    """Raised when a session record cannot be found."""


class PlanningSessionStateError(RuntimeError):
    """Raised when a session lifecycle transition is invalid for its current state."""


class PlanningSessionFollowUpError(ValueError):
    """Raised when a follow-up answer does not target a pending question."""


class PlanningSessionRepository(Protocol):
    def create_session(
        self,
        *,
        request_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession: ...

    def load_session(self, session_id: str) -> PlanningSession: ...

    def append_turn(
        self,
        session_id: str,
        *,
        role: str,
        message: str,
        question_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        follow_ups: Optional[Iterable[PlanningSessionFollowUp]] = None,
        draft_revision: Optional[PlanningSessionDraftRevision] = None,
        request_summary: Optional[str] = None,
    ) -> PlanningSession: ...

    def restart_session(
        self,
        session_id: str,
        *,
        request_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession: ...

    def abandon_session(
        self,
        session_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession: ...


class LocalPlanningSessionStore:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        now: Optional[Callable[[], str]] = None,
        session_id_factory: Optional[Callable[[], str]] = None,
        turn_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: "1970-01-01T00:00:00Z")
        self._session_id_factory = session_id_factory or self._build_counter("session")
        self._turn_id_factory = turn_id_factory or self._build_counter("turn")

    def create_session(
        self,
        *,
        request_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession:
        timestamp = self._now()
        session = PlanningSession(
            session_id=self._session_id_factory(),
            status="active",
            created_at=timestamp,
            updated_at=timestamp,
            request_summary=request_summary,
            last_activity_at=timestamp,
            metadata=dict(metadata or {}),
        )
        self._write_session(session)
        return session

    def load_session(self, session_id: str) -> PlanningSession:
        path = self._session_path(session_id)
        if not path.exists():
            raise PlanningSessionNotFoundError(session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._session_from_dict(payload)

    def append_turn(
        self,
        session_id: str,
        *,
        role: str,
        message: str,
        question_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        follow_ups: Optional[Iterable[PlanningSessionFollowUp]] = None,
        draft_revision: Optional[PlanningSessionDraftRevision] = None,
        request_summary: Optional[str] = None,
    ) -> PlanningSession:
        session = self.load_session(session_id)
        if session.status != "active":
            raise PlanningSessionStateError(
                f"planning session '{session_id}' is {session.status} and cannot accept new turns"
            )
        timestamp = self._now()
        if role == "user" and question_id is not None:
            follow_up = next(
                (item for item in session.follow_ups if item.question_id == question_id),
                None,
            )
            if follow_up is None:
                raise PlanningSessionFollowUpError(
                    f"planning session '{session_id}' does not have follow-up '{question_id}'"
                )
            if follow_up.status != "pending":
                raise PlanningSessionFollowUpError(
                    f"follow-up '{question_id}' is {follow_up.status} and cannot be answered again"
                )
        turn = PlanningSessionTurn(
            turn_id=self._turn_id_factory(),
            role=role,
            message=message,
            created_at=timestamp,
            revision_number=(None if draft_revision is None else draft_revision.revision_number),
            question_id=question_id,
            metadata=dict(metadata or {}),
        )
        session.turns.append(turn)
        if request_summary is not None:
            session.request_summary = request_summary
        if follow_ups is not None:
            session.follow_ups = self._merge_follow_ups(
                session.follow_ups,
                follow_ups,
                answered_question_id=question_id if role == "user" else None,
                answer_message=message if role == "user" else None,
                answered_at=timestamp if role == "user" else None,
            )
        elif role == "user" and question_id is not None:
            session.follow_ups = self._merge_follow_ups(
                session.follow_ups,
                (),
                answered_question_id=question_id,
                answer_message=message,
                answered_at=timestamp,
            )
        if draft_revision is not None:
            session.draft_revisions.append(draft_revision)
        session.updated_at = timestamp
        session.last_activity_at = timestamp
        self._write_session(session)
        return session

    def restart_session(
        self,
        session_id: str,
        *,
        request_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession:
        original = self.load_session(session_id)
        timestamp = self._now()
        replacement = PlanningSession(
            session_id=self._session_id_factory(),
            status="active",
            created_at=timestamp,
            updated_at=timestamp,
            request_summary=(
                original.request_summary if request_summary is None else request_summary
            ),
            restart_count=original.restart_count + 1,
            restarted_from_session_id=original.session_id,
            last_activity_at=timestamp,
            metadata=self._merge_metadata(original.metadata, metadata),
        )
        original.status = "restarted"
        original.updated_at = timestamp
        original.last_activity_at = timestamp
        original.replacement_session_id = replacement.session_id
        self._write_session(original)
        self._write_session(replacement)
        return replacement

    def abandon_session(
        self,
        session_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlanningSession:
        session = self.load_session(session_id)
        timestamp = self._now()
        session.status = "abandoned"
        session.updated_at = timestamp
        session.last_activity_at = timestamp
        session.abandoned_at = timestamp
        session.metadata = self._merge_metadata(session.metadata, metadata)
        if session.draft_revisions:
            latest_revision = session.draft_revisions[-1]
            latest_revision.status = "abandoned"
        self._write_session(session)
        return session

    @staticmethod
    def _build_counter(prefix: str) -> Callable[[], str]:
        counter = 0

        def next_value() -> str:
            nonlocal counter
            counter += 1
            return f"{prefix}-{counter:04d}"

        return next_value

    def _session_path(self, session_id: str) -> Path:
        return self.root_dir / f"{session_id}.json"

    def _write_session(self, session: PlanningSession) -> None:
        path = self._session_path(session.session_id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(session.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _merge_follow_ups(
        self,
        existing: Iterable[PlanningSessionFollowUp],
        updates: Iterable[PlanningSessionFollowUp],
        *,
        answered_question_id: Optional[str] = None,
        answer_message: Optional[str] = None,
        answered_at: Optional[str] = None,
    ) -> list[PlanningSessionFollowUp]:
        merged = {follow_up.question_id: self._copy_follow_up(follow_up) for follow_up in existing}
        for follow_up in updates:
            merged[follow_up.question_id] = self._copy_follow_up(follow_up)
        if answered_question_id and answered_question_id in merged:
            answered = merged[answered_question_id]
            answered.status = "answered"
            answered.answer = answer_message
            answered.answered_at = answered_at
        return list(merged.values())

    @staticmethod
    def _merge_metadata(
        base: Dict[str, Any],
        updates: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(base)
        if updates:
            merged.update(updates)
        return merged

    @staticmethod
    def _copy_follow_up(follow_up: PlanningSessionFollowUp) -> PlanningSessionFollowUp:
        return PlanningSessionFollowUp(**follow_up.to_dict())

    @staticmethod
    def _session_from_dict(payload: Dict[str, Any]) -> PlanningSession:
        return PlanningSession(
            session_id=payload["session_id"],
            status=payload["status"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            request_summary=payload.get("request_summary"),
            turns=[
                PlanningSessionTurn(**turn_payload)
                for turn_payload in payload.get("turns", [])
            ],
            follow_ups=[
                PlanningSessionFollowUp(**follow_up_payload)
                for follow_up_payload in payload.get("follow_ups", [])
            ],
            draft_revisions=[
                PlanningSessionDraftRevision(**revision_payload)
                for revision_payload in payload.get("draft_revisions", [])
            ],
            restart_count=payload.get("restart_count", 0),
            restarted_from_session_id=payload.get("restarted_from_session_id"),
            replacement_session_id=payload.get("replacement_session_id"),
            abandoned_at=payload.get("abandoned_at"),
            last_activity_at=payload.get("last_activity_at"),
            metadata=dict(payload.get("metadata") or {}),
        )
