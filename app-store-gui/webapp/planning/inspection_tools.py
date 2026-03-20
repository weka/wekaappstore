from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


SUPPORTED_INSPECTION_INTENTS = frozenset({"cluster_snapshot", "planning_snapshot", "weka_storage"})
SUPPORTED_FAILURE_STAGES = frozenset(
    {"inspection", "validation", "yaml_generation", "apply_handoff"}
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PlanningInspectionTools:
    def __init__(
        self,
        *,
        cluster_collector: Callable[[], Dict[str, Any]],
        weka_collector: Callable[[], Dict[str, Any]],
    ) -> None:
        self._cluster_collector = cluster_collector
        self._weka_collector = weka_collector
        self.audit_log: List[Dict[str, Any]] = []

    def inspect(self, intent: str, *, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        if intent not in SUPPORTED_INSPECTION_INTENTS:
            raise ValueError(
                f"Unsupported inspection intent '{intent}'. Supported intents: {', '.join(sorted(SUPPORTED_INSPECTION_INTENTS))}"
            )

        if intent == "cluster_snapshot":
            result = self._cluster_collector()
        elif intent == "planning_snapshot":
            cluster_snapshot = self._cluster_collector()
            weka_snapshot = self._weka_collector()
            result = merge_inspection_results(
                (cluster_snapshot, weka_snapshot),
                correlation_id=correlation_id,
            )
        else:
            result = self._weka_collector()

        event = {
            "timestamp": _utc_timestamp(),
            "intent": intent,
            "correlation_id": correlation_id,
            "status": "ok",
        }
        self.audit_log.append(event)
        return {
            "intent": intent,
            "correlation_id": correlation_id,
            "audit": event,
            "result": result,
        }

    def assess_fit(
        self,
        *,
        correlation_id: Optional[str] = None,
        required_domains: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        inspection = self.inspect("planning_snapshot", correlation_id=correlation_id)
        snapshot = inspection["result"]
        fit_findings = derive_fit_findings_from_snapshot(
            snapshot,
            required_domains=required_domains,
        )
        return {
            "correlation_id": correlation_id,
            "audit": inspection["audit"],
            "inspection_snapshot": snapshot,
            "fit_findings": fit_findings,
        }


def merge_inspection_results(
    results: Iterable[Mapping[str, Any]],
    *,
    correlation_id: Optional[str] = None,
    captured_at: Optional[str] = None,
) -> Dict[str, Any]:
    merged_domains: Dict[str, Any] = {}

    for result in results:
        domains = result.get("domains", {})
        if not isinstance(domains, Mapping):
            continue
        for domain_name, domain in domains.items():
            merged_domains[str(domain_name)] = deepcopy(domain)

    return {
        "captured_at": captured_at or _utc_timestamp(),
        "correlation_id": correlation_id,
        "domains": merged_domains,
    }


def derive_fit_findings_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    required_domains: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    required = set(required_domains or snapshot.get("domains", {}).keys())
    domains: Dict[str, Any] = {}
    blockers: List[Dict[str, Any]] = []
    notes: List[str] = []
    status = "fit"

    raw_domains = snapshot.get("domains", {})
    if not isinstance(raw_domains, Mapping):
        raw_domains = {}

    for domain_name, raw_domain in raw_domains.items():
        if not isinstance(raw_domain, Mapping):
            continue

        domain = deepcopy(dict(raw_domain))
        domain_required = domain_name in required
        domain["required"] = domain_required
        domain.setdefault("freshness", {"captured_at": snapshot.get("captured_at")})
        domain.setdefault("observed", {})
        domain.setdefault("notes", [])
        domain.setdefault("blockers", [])

        domain_status = str(domain.get("status") or "unavailable")
        domain_blockers = [deepcopy(blocker) for blocker in domain.get("blockers", []) if isinstance(blocker, Mapping)]

        if domain_required and domain_status in {"partial", "unavailable"}:
            status = "blocked"
            if not domain_blockers:
                domain_blockers = [
                    {
                        "code": f"{domain_name}_inspection_incomplete",
                        "message": f"Required inspection domain '{domain_name}' is {domain_status}.",
                        "domain": domain_name,
                    }
                ]
            blockers.extend(domain_blockers)
            notes.append(
                f"Required inspection domain '{domain_name}' is {domain_status}; planning must fail closed."
            )

        domain["blockers"] = domain_blockers
        domains[str(domain_name)] = domain

    if status == "fit":
        notes.append("Inspection snapshot confirms the required planning domains are complete.")

    return {
        "status": status,
        "notes": notes,
        "blockers": blockers,
        "domains": deepcopy(domains),
        "inspection_snapshot": {
            "captured_at": snapshot.get("captured_at"),
            "correlation_id": snapshot.get("correlation_id"),
            "domains": deepcopy(domains),
        },
    }


def build_stage_error(stage: str, error: Exception, *, correlation_id: Optional[str] = None) -> Dict[str, str]:
    if stage not in SUPPORTED_FAILURE_STAGES:
        raise ValueError(
            f"Unsupported planning failure stage '{stage}'. Supported stages: {', '.join(sorted(SUPPORTED_FAILURE_STAGES))}"
        )

    message = str(error) or error.__class__.__name__
    if correlation_id:
        message = f"[{correlation_id}] {message}"

    return {
        "code": f"{stage}_failed",
        "path": stage,
        "message": message,
        "stage": stage,
        "correlation_id": correlation_id or "",
    }
