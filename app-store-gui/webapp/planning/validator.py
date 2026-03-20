from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from .models import (
    SUPPORTED_BLUEPRINT_FAMILIES,
    SUPPORTED_CRDS_STRATEGIES,
    SUPPORTED_FIT_STATUSES,
    SUPPORTED_INSPECTION_DOMAIN_STATUSES,
    SUPPORTED_READINESS_CHECK_TYPES,
    ComponentPlan,
    FitFindings,
    FitBlocker,
    HelmChartPlan,
    InspectionDomainFinding,
    InspectionFreshness,
    InspectionSnapshot,
    NamespaceStrategy,
    NormalizationWarning,
    PlanValidationError,
    ReadinessCheckPlan,
    StructuredPlan,
    UnresolvedQuestion,
    ValidationResult,
    ValuesFileReference,
    clone_mapping,
)


_TOP_LEVEL_FIELDS = {
    "request_summary",
    "blueprint_family",
    "namespace_strategy",
    "components",
    "prerequisites",
    "fit_findings",
    "unresolved_questions",
    "reasoning_summary",
}

_COMPONENT_FIELDS = {
    "name",
    "enabled",
    "depends_on",
    "target_namespace",
    "helm_chart",
    "kubernetes_manifest",
    "values",
    "values_files",
    "wait_for_ready",
    "readiness_check",
}

_HELM_FIELDS = {"repository", "name", "version", "release_name", "crds_strategy"}
_VALUES_FILE_FIELDS = {"kind", "name", "key"}
_READINESS_FIELDS = {"type", "name", "selector", "match_labels", "namespace", "timeout"}
_NAMESPACE_MODES = {"explicit", "default"}
_FIT_FINDINGS_FIELDS = {"status", "notes", "blockers", "domains", "inspection_snapshot"}
_FIT_BLOCKER_FIELDS = {"code", "message", "domain"}
_INSPECTION_DOMAIN_FIELDS = {"status", "required", "freshness", "observed", "notes", "blockers"}
_INSPECTION_FRESHNESS_FIELDS = {"captured_at", "max_age_seconds", "observed_generation"}
_REQUIRED_INSPECTION_DOMAINS = {"cpu", "memory", "gpu", "namespaces", "storage_classes", "weka"}


def validate_structured_plan(payload: Mapping[str, Any]) -> ValidationResult:
    errors: List[PlanValidationError] = []
    warnings: List[NormalizationWarning] = []

    if not isinstance(payload, Mapping):
        return ValidationResult(
            valid=False,
            errors=[
                PlanValidationError(
                    code="invalid_type",
                    path="$",
                    message="structured plan payload must be an object",
                )
            ],
        )

    normalized = deepcopy(dict(payload))

    _validate_unknown_keys(normalized, _TOP_LEVEL_FIELDS, "", errors)
    _require_fields(
        normalized,
        [
            "request_summary",
            "blueprint_family",
            "namespace_strategy",
            "components",
            "prerequisites",
            "fit_findings",
            "unresolved_questions",
            "reasoning_summary",
        ],
        errors,
    )

    namespace_strategy = _parse_namespace_strategy(
        normalized.get("namespace_strategy"),
        errors,
    )
    fit_findings = _parse_fit_findings(normalized.get("fit_findings"), errors)
    components = _parse_components(normalized.get("components"), namespace_strategy, warnings, errors)
    unresolved_questions = _parse_unresolved_questions(
        normalized.get("unresolved_questions"), errors
    )
    prerequisites = _parse_prerequisites(normalized.get("prerequisites"), errors)
    request_summary = _require_string(normalized.get("request_summary"), "request_summary", errors)
    reasoning_summary = _require_string(
        normalized.get("reasoning_summary"), "reasoning_summary", errors
    )
    blueprint_family = _require_string(
        normalized.get("blueprint_family"), "blueprint_family", errors
    )
    if blueprint_family and blueprint_family not in SUPPORTED_BLUEPRINT_FAMILIES:
        errors.append(
            PlanValidationError(
                code="unsupported_blueprint_family",
                path="blueprint_family",
                message=(
                    f"unsupported blueprint family '{blueprint_family}'; supported families are "
                    f"{', '.join(sorted(SUPPORTED_BLUEPRINT_FAMILIES))}"
                ),
            )
        )

    _validate_component_contracts(components, errors)
    _validate_unresolved_questions(unresolved_questions, errors)

    if errors:
        return ValidationResult(
            valid=False,
            errors=_sort_issues(errors),
            warnings=_sort_issues(warnings),
        )

    plan = StructuredPlan(
        request_summary=request_summary,
        blueprint_family=blueprint_family,
        namespace_strategy=namespace_strategy,
        components=components,
        prerequisites=prerequisites,
        fit_findings=fit_findings,
        unresolved_questions=unresolved_questions,
        reasoning_summary=reasoning_summary,
        normalization_warnings=_sort_issues(warnings),
    )
    return ValidationResult(valid=True, plan=plan, warnings=_sort_issues(warnings), errors=[])


def _validate_unknown_keys(
    payload: Any,
    allowed_keys: set[str],
    prefix: str,
    errors: List[PlanValidationError],
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key in sorted(payload):
        if key not in allowed_keys:
            path = f"{prefix}.{key}" if prefix else key
            errors.append(
                PlanValidationError(
                    code="unsupported_field",
                    path=path,
                    message=f"unsupported field '{path}' is not part of the Phase 1 structured plan contract",
                )
            )


def _require_fields(
    payload: Mapping[str, Any],
    required_fields: List[str],
    errors: List[PlanValidationError],
) -> None:
    for field_name in required_fields:
        if field_name not in payload:
            errors.append(
                PlanValidationError(
                    code="missing_required_field",
                    path=field_name,
                    message=f"missing required field '{field_name}'",
                )
            )


def _require_string(value: Any, path: str, errors: List[PlanValidationError]) -> str:
    if isinstance(value, str) and value.strip():
        return value
    errors.append(
        PlanValidationError(
            code="invalid_type",
            path=path,
            message=f"field '{path}' must be a non-empty string",
        )
    )
    return ""


def _require_bool(value: Any, path: str, errors: List[PlanValidationError]) -> bool:
    if isinstance(value, bool):
        return value
    errors.append(
        PlanValidationError(
            code="invalid_type",
            path=path,
            message=f"field '{path}' must be a boolean",
        )
    )
    return False


def _require_int(value: Any, path: str, errors: List[PlanValidationError]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    errors.append(
        PlanValidationError(
            code="invalid_type",
            path=path,
            message=f"field '{path}' must be an integer",
        )
    )
    return 0


def _parse_namespace_strategy(
    payload: Any,
    errors: List[PlanValidationError],
) -> NamespaceStrategy:
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="namespace_strategy",
                message="field 'namespace_strategy' must be an object",
            )
        )
        return NamespaceStrategy(mode="explicit", target_namespace=None)

    _validate_unknown_keys(payload, {"mode", "target_namespace"}, "namespace_strategy", errors)
    mode = _require_string(payload.get("mode"), "namespace_strategy.mode", errors)
    target_namespace = payload.get("target_namespace")
    if target_namespace is not None and not isinstance(target_namespace, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="namespace_strategy.target_namespace",
                message="field 'namespace_strategy.target_namespace' must be a string when provided",
            )
        )
        target_namespace = None
    if mode and mode not in _NAMESPACE_MODES:
        errors.append(
            PlanValidationError(
                code="unsupported_namespace_strategy",
                path="namespace_strategy.mode",
                message=(
                    f"unsupported namespace strategy '{mode}'; supported modes are "
                    f"{', '.join(sorted(_NAMESPACE_MODES))}"
                ),
            )
        )
    if mode == "explicit" and not target_namespace:
        errors.append(
            PlanValidationError(
                code="missing_required_field",
                path="namespace_strategy.target_namespace",
                message="explicit namespace_strategy requires 'target_namespace'",
            )
        )
    return NamespaceStrategy(mode=mode or "explicit", target_namespace=target_namespace)


def _parse_fit_findings(payload: Any, errors: List[PlanValidationError]) -> FitFindings:
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="fit_findings",
                message="field 'fit_findings' must be an object",
            )
        )
        return FitFindings(status="", notes=[])

    _validate_unknown_keys(payload, _FIT_FINDINGS_FIELDS, "fit_findings", errors)
    status = _require_string(payload.get("status"), "fit_findings.status", errors)
    if status and status not in SUPPORTED_FIT_STATUSES:
        errors.append(
            PlanValidationError(
                code="unsupported_fit_status",
                path="fit_findings.status",
                message=(
                    f"unsupported fit_findings status '{status}'; supported values are "
                    f"{', '.join(sorted(SUPPORTED_FIT_STATUSES))}"
                ),
            )
        )
    notes = payload.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="fit_findings.notes",
                message="field 'fit_findings.notes' must be a list of strings",
            )
        )
        notes = []
    blockers = _parse_fit_blockers(payload.get("blockers", []), "fit_findings.blockers", errors)
    domains = _parse_inspection_domains(payload.get("domains", {}), "fit_findings.domains", errors)
    inspection_snapshot = _parse_inspection_snapshot(
        payload.get("inspection_snapshot"),
        errors,
    )
    fit_findings = FitFindings(
        status=status,
        notes=list(notes),
        blockers=blockers,
        domains=domains,
        inspection_snapshot=inspection_snapshot,
    )
    _validate_fit_findings_contract(fit_findings, errors)
    return fit_findings


def _parse_fit_blockers(
    payload: Any,
    path: str,
    errors: List[PlanValidationError],
) -> List[FitBlocker]:
    if not isinstance(payload, list):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be a list",
            )
        )
        return []

    blockers: List[FitBlocker] = []
    for index, item in enumerate(payload):
        blocker_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=blocker_path,
                    message=f"field '{blocker_path}' must be an object",
                )
            )
            continue
        _validate_unknown_keys(item, _FIT_BLOCKER_FIELDS, blocker_path, errors)
        domain = item.get("domain")
        if domain is not None and not isinstance(domain, str):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=f"{blocker_path}.domain",
                    message=f"field '{blocker_path}.domain' must be a string when provided",
                )
            )
            domain = None
        blockers.append(
            FitBlocker(
                code=_require_string(item.get("code"), f"{blocker_path}.code", errors),
                message=_require_string(item.get("message"), f"{blocker_path}.message", errors),
                domain=domain,
            )
        )
    return blockers


def _parse_inspection_domains(
    payload: Any,
    path: str,
    errors: List[PlanValidationError],
) -> Dict[str, InspectionDomainFinding]:
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be an object",
            )
        )
        return {}

    domains: Dict[str, InspectionDomainFinding] = {}
    for domain_name in sorted(payload):
        domain_path = f"{path}.{domain_name}"
        domain_payload = payload[domain_name]
        if not isinstance(domain_payload, Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=domain_path,
                    message=f"field '{domain_path}' must be an object",
                )
            )
            continue

        _validate_unknown_keys(domain_payload, _INSPECTION_DOMAIN_FIELDS, domain_path, errors)
        status = _require_string(domain_payload.get("status"), f"{domain_path}.status", errors)
        if status and status not in SUPPORTED_INSPECTION_DOMAIN_STATUSES:
            errors.append(
                PlanValidationError(
                    code="unsupported_inspection_domain_status",
                    path=f"{domain_path}.status",
                    message=(
                        f"unsupported inspection domain status '{status}'; supported values are "
                        f"{', '.join(sorted(SUPPORTED_INSPECTION_DOMAIN_STATUSES))}"
                    ),
                )
            )
        required = _coerce_optional_bool(domain_payload.get("required"), f"{domain_path}.required", errors, True)
        freshness = _parse_inspection_freshness(
            domain_payload.get("freshness"),
            f"{domain_path}.freshness",
            errors,
        )
        observed = clone_mapping(domain_payload.get("observed"))
        if domain_payload.get("observed") is not None and not isinstance(domain_payload.get("observed"), Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=f"{domain_path}.observed",
                    message=f"field '{domain_path}.observed' must be an object when provided",
                )
            )
            observed = {}
        notes = _parse_string_list(domain_payload.get("notes", []), f"{domain_path}.notes", errors)
        blockers = _parse_fit_blockers(domain_payload.get("blockers", []), f"{domain_path}.blockers", errors)
        domains[domain_name] = InspectionDomainFinding(
            status=status,
            required=required,
            freshness=freshness,
            observed=observed,
            notes=notes,
            blockers=blockers,
        )
    return domains


def _parse_inspection_freshness(
    payload: Any,
    path: str,
    errors: List[PlanValidationError],
) -> Optional[InspectionFreshness]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be an object when provided",
            )
        )
        return None
    _validate_unknown_keys(payload, _INSPECTION_FRESHNESS_FIELDS, path, errors)
    observed_generation = payload.get("observed_generation")
    if observed_generation is not None and not isinstance(observed_generation, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.observed_generation",
                message=f"field '{path}.observed_generation' must be a string when provided",
            )
        )
        observed_generation = None
    max_age_seconds = payload.get("max_age_seconds")
    if max_age_seconds is not None:
        max_age_seconds = _require_int(max_age_seconds, f"{path}.max_age_seconds", errors)
    return InspectionFreshness(
        captured_at=_require_string(payload.get("captured_at"), f"{path}.captured_at", errors),
        max_age_seconds=max_age_seconds,
        observed_generation=observed_generation,
    )


def _parse_inspection_snapshot(
    payload: Any,
    errors: List[PlanValidationError],
) -> Optional[InspectionSnapshot]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="fit_findings.inspection_snapshot",
                message="field 'fit_findings.inspection_snapshot' must be an object when provided",
            )
        )
        return None

    _validate_unknown_keys(
        payload,
        {"captured_at", "correlation_id", "domains"},
        "fit_findings.inspection_snapshot",
        errors,
    )
    correlation_id = payload.get("correlation_id")
    if correlation_id is not None and not isinstance(correlation_id, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="fit_findings.inspection_snapshot.correlation_id",
                message=(
                    "field 'fit_findings.inspection_snapshot.correlation_id' must be a string when provided"
                ),
            )
        )
        correlation_id = None
    return InspectionSnapshot(
        captured_at=_require_string(
            payload.get("captured_at"),
            "fit_findings.inspection_snapshot.captured_at",
            errors,
        ),
        correlation_id=correlation_id,
        domains=_parse_inspection_domains(
            payload.get("domains", {}),
            "fit_findings.inspection_snapshot.domains",
            errors,
        ),
    )


def _parse_prerequisites(payload: Any, errors: List[PlanValidationError]) -> List[str]:
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="prerequisites",
                message="field 'prerequisites' must be a list of strings",
            )
        )
        return []
    return list(payload)


def _parse_unresolved_questions(
    payload: Any,
    errors: List[PlanValidationError],
) -> List[UnresolvedQuestion]:
    if not isinstance(payload, list):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="unresolved_questions",
                message="field 'unresolved_questions' must be a list",
            )
        )
        return []

    questions: List[UnresolvedQuestion] = []
    for index, item in enumerate(payload):
        path = f"unresolved_questions[{index}]"
        if not isinstance(item, Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=path,
                    message=f"field '{path}' must be an object",
                )
            )
            continue
        _validate_unknown_keys(item, {"question", "blocking", "install_critical"}, path, errors)
        questions.append(
            UnresolvedQuestion(
                question=_require_string(item.get("question"), f"{path}.question", errors),
                blocking=_coerce_optional_bool(item.get("blocking"), f"{path}.blocking", errors, False),
                install_critical=_coerce_optional_bool(
                    item.get("install_critical"),
                    f"{path}.install_critical",
                    errors,
                    False,
                ),
            )
        )
    return questions


def _parse_components(
    payload: Any,
    namespace_strategy: NamespaceStrategy,
    warnings: List[NormalizationWarning],
    errors: List[PlanValidationError],
) -> List[ComponentPlan]:
    if not isinstance(payload, list):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path="components",
                message="field 'components' must be a list",
            )
        )
        return []
    if not payload:
        errors.append(
            PlanValidationError(
                code="missing_required_field",
                path="components",
                message="structured plan must include at least one component",
            )
        )
        return []

    components: List[ComponentPlan] = []
    for index, item in enumerate(payload):
        base_path = f"components[{index}]"
        if not isinstance(item, Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=base_path,
                    message=f"field '{base_path}' must be an object",
                )
            )
            continue

        _validate_unknown_keys(item, _COMPONENT_FIELDS, base_path, errors)
        name = _require_string(item.get("name"), f"{base_path}.name", errors)
        enabled = _coerce_optional_bool(item.get("enabled"), f"{base_path}.enabled", errors, True)
        depends_on = _parse_string_list(item.get("depends_on", []), f"{base_path}.depends_on", errors)
        target_namespace = item.get("target_namespace")
        if target_namespace is not None and not isinstance(target_namespace, str):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=f"{base_path}.target_namespace",
                    message=f"field '{base_path}.target_namespace' must be a string when provided",
                )
            )
            target_namespace = None
        if not target_namespace and namespace_strategy.target_namespace:
            target_namespace = namespace_strategy.target_namespace

        helm_chart = _parse_helm_chart(item.get("helm_chart"), base_path, name, warnings, errors)
        kubernetes_manifest = item.get("kubernetes_manifest")
        if kubernetes_manifest is not None and not isinstance(kubernetes_manifest, str):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=f"{base_path}.kubernetes_manifest",
                    message=f"field '{base_path}.kubernetes_manifest' must be a string when provided",
                )
            )
            kubernetes_manifest = None

        values = clone_mapping(item.get("values"))
        if item.get("values") is not None and not isinstance(item.get("values"), Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=f"{base_path}.values",
                    message=f"field '{base_path}.values' must be an object when provided",
                )
            )
            values = {}

        values_files = _parse_values_files(item.get("values_files", []), base_path, errors)

        wait_for_ready_value = item.get("wait_for_ready")
        if wait_for_ready_value is None:
            wait_for_ready = True
            warnings.append(
                NormalizationWarning(
                    code="defaulted_field",
                    path=f"{base_path}.wait_for_ready",
                    message="wait_for_ready defaults to true when omitted",
                )
            )
        else:
            wait_for_ready = _require_bool(
                wait_for_ready_value, f"{base_path}.wait_for_ready", errors
            )

        readiness_check = _parse_readiness_check(item.get("readiness_check"), base_path, errors)

        components.append(
            ComponentPlan(
                name=name,
                enabled=enabled,
                depends_on=depends_on,
                target_namespace=target_namespace,
                helm_chart=helm_chart,
                kubernetes_manifest=kubernetes_manifest,
                values=values,
                values_files=values_files,
                wait_for_ready=wait_for_ready,
                readiness_check=readiness_check,
            )
        )
    return components


def _parse_helm_chart(
    payload: Any,
    component_path: str,
    component_name: str,
    warnings: List[NormalizationWarning],
    errors: List[PlanValidationError],
) -> Optional[HelmChartPlan]:
    if payload is None:
        return None
    path = f"{component_path}.helm_chart"
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be an object when provided",
            )
        )
        return None

    _validate_unknown_keys(payload, _HELM_FIELDS, path, errors)
    repository = _require_string(payload.get("repository"), f"{path}.repository", errors)
    name = _require_string(payload.get("name"), f"{path}.name", errors)
    version = _require_string(payload.get("version"), f"{path}.version", errors)
    release_name = payload.get("release_name")
    if release_name is None:
        release_name = component_name
        warnings.append(
            NormalizationWarning(
                code="defaulted_field",
                path=f"{path}.release_name",
                message="release_name defaults to the component name when omitted",
            )
        )
    elif not isinstance(release_name, str) or not release_name.strip():
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.release_name",
                message=f"field '{path}.release_name' must be a non-empty string when provided",
            )
        )
        release_name = component_name

    crds_strategy = payload.get("crds_strategy", "Auto")
    if not isinstance(crds_strategy, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.crds_strategy",
                message=f"field '{path}.crds_strategy' must be a string when provided",
            )
        )
        crds_strategy = "Auto"
    if crds_strategy not in SUPPORTED_CRDS_STRATEGIES:
        errors.append(
            PlanValidationError(
                code="unsupported_crds_strategy",
                path=f"{path}.crds_strategy",
                message=(
                    f"unsupported CRDs strategy '{crds_strategy}'; supported values are "
                    f"{', '.join(sorted(SUPPORTED_CRDS_STRATEGIES))}"
                ),
            )
        )
    return HelmChartPlan(
        repository=repository,
        name=name,
        version=version,
        release_name=release_name,
        crds_strategy=crds_strategy,
    )


def _parse_values_files(
    payload: Any,
    component_path: str,
    errors: List[PlanValidationError],
) -> List[ValuesFileReference]:
    path = f"{component_path}.values_files"
    if not isinstance(payload, list):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be a list when provided",
            )
        )
        return []
    values_files: List[ValuesFileReference] = []
    for index, item in enumerate(payload):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(
                PlanValidationError(
                    code="invalid_type",
                    path=item_path,
                    message=f"field '{item_path}' must be an object",
                )
            )
            continue
        _validate_unknown_keys(item, _VALUES_FILE_FIELDS, item_path, errors)
        kind = _require_string(item.get("kind"), f"{item_path}.kind", errors)
        name = _require_string(item.get("name"), f"{item_path}.name", errors)
        key = _require_string(item.get("key"), f"{item_path}.key", errors)
        if kind and kind not in {"ConfigMap", "Secret"}:
            errors.append(
                PlanValidationError(
                    code="unsupported_values_file_kind",
                    path=f"{item_path}.kind",
                    message="values_files kind must be ConfigMap or Secret",
                )
            )
        values_files.append(ValuesFileReference(kind=kind, name=name, key=key))
    return values_files


def _parse_readiness_check(
    payload: Any,
    component_path: str,
    errors: List[PlanValidationError],
) -> Optional[ReadinessCheckPlan]:
    if payload is None:
        return None
    path = f"{component_path}.readiness_check"
    if not isinstance(payload, Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be an object when provided",
            )
        )
        return None

    _validate_unknown_keys(payload, _READINESS_FIELDS, path, errors)
    readiness_type = _require_string(payload.get("type"), f"{path}.type", errors)
    if readiness_type and readiness_type not in SUPPORTED_READINESS_CHECK_TYPES:
        errors.append(
            PlanValidationError(
                code="unsupported_readiness_check_type",
                path=f"{path}.type",
                message=(
                    f"unsupported readiness_check type '{readiness_type}'; supported types are "
                    f"{', '.join(sorted(SUPPORTED_READINESS_CHECK_TYPES))}"
                ),
            )
        )
    name = payload.get("name")
    selector = payload.get("selector")
    namespace = payload.get("namespace")
    if name is not None and not isinstance(name, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.name",
                message=f"field '{path}.name' must be a string when provided",
            )
        )
        name = None
    if selector is not None and not isinstance(selector, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.selector",
                message=f"field '{path}.selector' must be a string when provided",
            )
        )
        selector = None
    if namespace is not None and not isinstance(namespace, str):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.namespace",
                message=f"field '{path}.namespace' must be a string when provided",
            )
        )
        namespace = None
    match_labels = clone_mapping(payload.get("match_labels"))
    if payload.get("match_labels") is not None and not isinstance(payload.get("match_labels"), Mapping):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=f"{path}.match_labels",
                message=f"field '{path}.match_labels' must be an object when provided",
            )
        )
        match_labels = {}
    timeout = payload.get("timeout", 300)
    timeout = _require_int(timeout, f"{path}.timeout", errors)
    return ReadinessCheckPlan(
        type=readiness_type,
        name=name,
        selector=selector,
        match_labels={str(key): str(value) for key, value in match_labels.items()},
        namespace=namespace,
        timeout=timeout,
    )


def _parse_string_list(value: Any, path: str, errors: List[PlanValidationError]) -> List[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(
            PlanValidationError(
                code="invalid_type",
                path=path,
                message=f"field '{path}' must be a list of strings",
            )
        )
        return []
    return list(value)


def _coerce_optional_bool(
    value: Any,
    path: str,
    errors: List[PlanValidationError],
    default: bool,
) -> bool:
    if value is None:
        return default
    return _require_bool(value, path, errors)


def _validate_component_contracts(
    components: List[ComponentPlan],
    errors: List[PlanValidationError],
) -> None:
    component_names = [component.name for component in components]
    name_set = set()
    for index, component in enumerate(components):
        path = f"components[{index}]"
        if component.name in name_set:
            errors.append(
                PlanValidationError(
                    code="duplicate_component_name",
                    path=f"{path}.name",
                    message=f"component name '{component.name}' must be unique",
                )
            )
        name_set.add(component.name)

        deployment_methods = int(component.helm_chart is not None) + int(
            component.kubernetes_manifest is not None
        )
        if deployment_methods != 1:
            errors.append(
                PlanValidationError(
                    code="invalid_deployment_method",
                    path=path,
                    message="component must define exactly one deployment method: helm_chart or kubernetes_manifest",
                )
            )

        for dependency in component.depends_on:
            if dependency not in component_names:
                errors.append(
                    PlanValidationError(
                        code="unknown_dependency",
                        path=f"{path}.depends_on",
                        message=(
                            f"component '{component.name}' depends on unknown component '{dependency}'"
                        ),
                    )
                )


def _validate_unresolved_questions(
    unresolved_questions: List[UnresolvedQuestion],
    errors: List[PlanValidationError],
) -> None:
    for index, question in enumerate(unresolved_questions):
        if question.blocking or question.install_critical:
            errors.append(
                PlanValidationError(
                    code="blocking_unresolved_question",
                    path=f"unresolved_questions[{index}]",
                    message=(
                        "install-critical unresolved question blocks YAML generation and apply handoff"
                    ),
                )
            )


def _validate_fit_findings_contract(
    fit_findings: FitFindings,
    errors: List[PlanValidationError],
) -> None:
    if not fit_findings.domains:
        return

    for domain_name in _REQUIRED_INSPECTION_DOMAINS:
        domain = fit_findings.domains.get(domain_name)
        if domain is None:
            errors.append(
                PlanValidationError(
                    code="missing_required_field",
                    path=f"fit_findings.domains.{domain_name}",
                    message=f"required inspection domain '{domain_name}' is missing from fit_findings.domains",
                )
            )
            continue
        if domain.required and domain.status != "complete":
            if fit_findings.status != "blocked":
                errors.append(
                    PlanValidationError(
                        code="fit_requires_blocked_status",
                        path="fit_findings.status",
                        message=(
                            "fit_findings.status must be 'blocked' when required inspection domains are partial or unavailable"
                        ),
                    )
                )
            if not domain.blockers:
                errors.append(
                    PlanValidationError(
                        code="missing_fit_blocker",
                        path=f"fit_findings.domains.{domain_name}.blockers",
                        message=(
                            f"required inspection domain '{domain_name}' must declare at least one blocker when status is not complete"
                        ),
                    )
                )

    if fit_findings.inspection_snapshot is not None:
        snapshot_domains = fit_findings.inspection_snapshot.domains
        for domain_name, domain in fit_findings.domains.items():
            snapshot_domain = snapshot_domains.get(domain_name)
            if snapshot_domain is None:
                errors.append(
                    PlanValidationError(
                        code="missing_snapshot_domain",
                        path=f"fit_findings.inspection_snapshot.domains.{domain_name}",
                        message=(
                            f"inspection snapshot is missing fit domain '{domain_name}'"
                        ),
                    )
                )
                continue
            if snapshot_domain.status != domain.status:
                errors.append(
                    PlanValidationError(
                        code="inspection_status_mismatch",
                        path=f"fit_findings.inspection_snapshot.domains.{domain_name}.status",
                        message=(
                            f"inspection snapshot domain '{domain_name}' must match fit_findings.domains status"
                        ),
                    )
                )


def _sort_issues(issues: List[Any]) -> List[Any]:
    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))
