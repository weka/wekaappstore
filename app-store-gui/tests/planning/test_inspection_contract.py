from __future__ import annotations

import importlib
from copy import deepcopy


def _validator_module():
    return importlib.import_module("webapp.planning.validator")


def test_validator_accepts_phase_two_fit_findings_contract(
    valid_plan_payload: dict,
    complete_fit_findings: dict,
) -> None:
    payload = deepcopy(valid_plan_payload)
    payload["fit_findings"] = complete_fit_findings

    result = _validator_module().validate_structured_plan(payload)

    assert result.valid is True
    assert result.errors == []
    assert result.plan is not None
    assert result.plan.fit_findings.status == "fit"
    assert sorted(result.plan.fit_findings.domains) == [
        "cpu",
        "gpu",
        "memory",
        "namespaces",
        "storage_classes",
        "weka",
    ]
    assert result.plan.fit_findings.domains["gpu"].observed["inventory"][0]["model"] == "NVIDIA L40"
    assert result.plan.fit_findings.inspection_snapshot is not None
    assert result.plan.fit_findings.inspection_snapshot.correlation_id == "corr-phase2-complete"


def test_validator_fails_closed_when_required_inspection_domain_is_partial(
    valid_plan_payload: dict,
    blocked_gpu_fit_findings: dict,
) -> None:
    payload = deepcopy(valid_plan_payload)
    payload["fit_findings"] = blocked_gpu_fit_findings

    result = _validator_module().validate_structured_plan(payload)

    assert result.valid is True
    assert result.errors == []
    assert result.plan is not None
    assert result.plan.fit_findings.status == "blocked"
    gpu_domain = result.plan.fit_findings.domains["gpu"]
    assert gpu_domain.status == "partial"
    assert gpu_domain.blockers[0].code == "gpu_metadata_incomplete"


def test_validator_rejects_assumed_fit_when_required_inspection_data_is_partial(
    valid_plan_payload: dict,
    blocked_gpu_fit_findings: dict,
) -> None:
    payload = deepcopy(valid_plan_payload)
    payload["fit_findings"] = blocked_gpu_fit_findings
    payload["fit_findings"]["status"] = "assumed-fit"

    result = _validator_module().validate_structured_plan(payload)

    assert result.valid is False
    assert ("fit_requires_blocked_status", "fit_findings.status") in {
        (error.code, error.path) for error in result.errors
    }


def test_validator_rejects_missing_required_domain_blocker(
    valid_plan_payload: dict,
    blocked_gpu_fit_findings: dict,
) -> None:
    payload = deepcopy(valid_plan_payload)
    payload["fit_findings"] = blocked_gpu_fit_findings
    payload["fit_findings"]["domains"]["gpu"]["blockers"] = []
    payload["fit_findings"]["inspection_snapshot"]["domains"]["gpu"]["blockers"] = []

    result = _validator_module().validate_structured_plan(payload)

    assert result.valid is False
    assert ("missing_fit_blocker", "fit_findings.domains.gpu.blockers") in {
        (error.code, error.path) for error in result.errors
    }


def test_validator_rejects_snapshot_domain_status_mismatch(
    valid_plan_payload: dict,
    complete_fit_findings: dict,
) -> None:
    payload = deepcopy(valid_plan_payload)
    payload["fit_findings"] = complete_fit_findings
    payload["fit_findings"]["inspection_snapshot"]["domains"]["weka"]["status"] = "partial"

    result = _validator_module().validate_structured_plan(payload)

    assert result.valid is False
    assert ("inspection_status_mismatch", "fit_findings.inspection_snapshot.domains.weka.status") in {
        (error.code, error.path) for error in result.errors
    }


def test_fit_findings_to_dict_serializes_phase_two_metadata(
    complete_fit_findings: dict,
) -> None:
    planning_models = importlib.import_module("webapp.planning.models")

    fit_findings = planning_models.FitFindings(
        status=complete_fit_findings["status"],
        notes=complete_fit_findings["notes"],
        blockers=[],
        domains={
            name: planning_models.InspectionDomainFinding(
                status=domain["status"],
                required=domain["required"],
                freshness=planning_models.InspectionFreshness(**domain["freshness"]),
                observed=domain["observed"],
                notes=domain["notes"],
                blockers=[],
            )
            for name, domain in complete_fit_findings["domains"].items()
        },
        inspection_snapshot=planning_models.InspectionSnapshot(
            captured_at=complete_fit_findings["inspection_snapshot"]["captured_at"],
            correlation_id=complete_fit_findings["inspection_snapshot"]["correlation_id"],
            domains={
                name: planning_models.InspectionDomainFinding(
                    status=domain["status"],
                    required=domain["required"],
                    freshness=planning_models.InspectionFreshness(**domain["freshness"]),
                    observed=domain["observed"],
                    notes=domain["notes"],
                    blockers=[],
                )
                for name, domain in complete_fit_findings["inspection_snapshot"]["domains"].items()
            },
        ),
    )

    serialized = fit_findings.to_dict()

    assert serialized["status"] == "fit"
    assert serialized["domains"]["cpu"]["freshness"]["max_age_seconds"] == 300
    assert serialized["inspection_snapshot"]["correlation_id"] == "corr-phase2-complete"
