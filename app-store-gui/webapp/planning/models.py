from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_CRD_API_VERSION = "warp.io/v1alpha1"
DEFAULT_CRD_KIND = "WekaAppStore"

SUPPORTED_BLUEPRINT_FAMILIES = frozenset(
    {
        "ai-agent-enterprise-research",
        "nvidia-vss",
        "openfold",
    }
)
SUPPORTED_CRDS_STRATEGIES = frozenset({"Auto", "Install", "Skip"})
SUPPORTED_READINESS_CHECK_TYPES = frozenset(
    {"pod", "deployment", "statefulset", "job", "custom"}
)
SUPPORTED_FIT_STATUSES = frozenset({"assumed-fit", "fit", "blocked"})
SUPPORTED_INSPECTION_DOMAIN_STATUSES = frozenset(
    {"complete", "partial", "unavailable", "not-required"}
)
SUPPORTED_FAILURE_STAGES = frozenset(
    {"inspection", "validation", "yaml_generation", "apply_handoff"}
)
SUPPORTED_PLANNING_SESSION_STATUSES = frozenset({"active", "restarted", "abandoned"})
SUPPORTED_PLANNING_TURN_ROLES = frozenset({"user", "assistant", "system"})
SUPPORTED_PLANNING_FOLLOW_UP_STATUSES = frozenset({"pending", "answered", "dismissed"})
SUPPORTED_PLANNING_DRAFT_STATUSES = frozenset({"draft", "validated", "blocked", "abandoned"})


@dataclass(slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class PlanValidationError(ValidationIssue):
    pass


@dataclass(slots=True)
class NormalizationWarning(ValidationIssue):
    pass


@dataclass(slots=True)
class ValuesFileReference:
    kind: str
    name: str
    key: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class HelmChartPlan:
    repository: str
    name: str
    version: str
    release_name: Optional[str] = None
    crds_strategy: str = "Auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReadinessCheckPlan:
    type: str
    name: Optional[str] = None
    selector: Optional[str] = None
    match_labels: Dict[str, str] = field(default_factory=dict)
    namespace: Optional[str] = None
    timeout: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComponentPlan:
    name: str
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    target_namespace: Optional[str] = None
    helm_chart: Optional[HelmChartPlan] = None
    kubernetes_manifest: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)
    values_files: List[ValuesFileReference] = field(default_factory=list)
    wait_for_ready: bool = True
    readiness_check: Optional[ReadinessCheckPlan] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.helm_chart is None:
            payload["helm_chart"] = None
        if self.readiness_check is None:
            payload["readiness_check"] = None
        return payload


@dataclass(slots=True)
class NamespaceStrategy:
    mode: str
    target_namespace: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass(slots=True)
class InspectionFreshness:
    captured_at: str
    max_age_seconds: Optional[int] = None
    observed_generation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FitBlocker:
    code: str
    message: str
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass(slots=True)
class InspectionDomainFinding:
    status: str
    required: bool = True
    freshness: Optional[InspectionFreshness] = None
    observed: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    blockers: List[FitBlocker] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.freshness is None:
            payload["freshness"] = None
        payload["blockers"] = [blocker.to_dict() for blocker in self.blockers]
        return payload


@dataclass(slots=True)
class InspectionSnapshot:
    captured_at: str
    correlation_id: Optional[str] = None
    domains: Dict[str, InspectionDomainFinding] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "correlation_id": self.correlation_id,
            "domains": {name: domain.to_dict() for name, domain in self.domains.items()},
        }


@dataclass(slots=True)
class FitFindings:
    status: str
    notes: List[str] = field(default_factory=list)
    blockers: List[FitBlocker] = field(default_factory=list)
    domains: Dict[str, InspectionDomainFinding] = field(default_factory=dict)
    inspection_snapshot: Optional[InspectionSnapshot] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "notes": list(self.notes),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "domains": {name: domain.to_dict() for name, domain in self.domains.items()},
            "inspection_snapshot": (
                None if self.inspection_snapshot is None else self.inspection_snapshot.to_dict()
            ),
        }


@dataclass(slots=True)
class StageFailure:
    stage: str
    message: str
    correlation_id: Optional[str] = None
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass(slots=True)
class UnresolvedQuestion:
    question: str
    blocking: bool = False
    install_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredPlan:
    request_summary: str
    blueprint_family: str
    namespace_strategy: NamespaceStrategy
    components: List[ComponentPlan]
    prerequisites: List[str]
    fit_findings: FitFindings
    unresolved_questions: List[UnresolvedQuestion]
    reasoning_summary: str
    normalization_warnings: List[NormalizationWarning] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["normalization_warnings"] = [warning.to_dict() for warning in self.normalization_warnings]
        return payload


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    plan: Optional[StructuredPlan] = None
    warnings: List[NormalizationWarning] = field(default_factory=list)
    errors: List[PlanValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": [error.to_dict() for error in self.errors],
        }


@dataclass(slots=True)
class PlanningSessionTurn:
    turn_id: str
    role: str
    message: str
    created_at: str
    revision_number: Optional[int] = None
    question_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanningSessionFollowUp:
    question_id: str
    question: str
    field_path: Optional[str] = None
    blocking: bool = False
    install_critical: bool = False
    status: str = "pending"
    asked_at: Optional[str] = None
    answered_at: Optional[str] = None
    answer: Optional[str] = None
    revision_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanningSessionDraftRevision:
    revision_number: int
    created_at: str
    summary: str
    status: str = "draft"
    structured_plan: Optional[Dict[str, Any]] = None
    fit_findings: Optional[Dict[str, Any]] = None
    unanswered_question_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanningSession:
    session_id: str
    status: str
    created_at: str
    updated_at: str
    request_summary: Optional[str] = None
    turns: List[PlanningSessionTurn] = field(default_factory=list)
    follow_ups: List[PlanningSessionFollowUp] = field(default_factory=list)
    draft_revisions: List[PlanningSessionDraftRevision] = field(default_factory=list)
    restart_count: int = 0
    restarted_from_session_id: Optional[str] = None
    replacement_session_id: Optional[str] = None
    abandoned_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request_summary": self.request_summary,
            "turns": [turn.to_dict() for turn in self.turns],
            "follow_ups": [follow_up.to_dict() for follow_up in self.follow_ups],
            "draft_revisions": [revision.to_dict() for revision in self.draft_revisions],
            "restart_count": self.restart_count,
            "restarted_from_session_id": self.restarted_from_session_id,
            "replacement_session_id": self.replacement_session_id,
            "abandoned_at": self.abandoned_at,
            "last_activity_at": self.last_activity_at,
            "metadata": dict(self.metadata),
        }

    @property
    def unanswered_follow_ups(self) -> List[PlanningSessionFollowUp]:
        return [follow_up for follow_up in self.follow_ups if follow_up.status == "pending"]

    @property
    def latest_revision(self) -> Optional[PlanningSessionDraftRevision]:
        if not self.draft_revisions:
            return None
        return self.draft_revisions[-1]


def clone_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    return dict(value)
