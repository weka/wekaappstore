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
class FitFindings:
    status: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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


def clone_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    return dict(value)
