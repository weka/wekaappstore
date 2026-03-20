from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import yaml

from .models import DEFAULT_CRD_API_VERSION, DEFAULT_CRD_KIND, ComponentPlan, StructuredPlan, ValidationResult
from .validator import validate_structured_plan


class PlanCompilationError(ValueError):
    """Raised when a structured plan cannot be compiled into canonical YAML."""


def compile_plan_to_wekaappstore(plan: StructuredPlan) -> Dict[str, Any]:
    if plan.normalization_warnings is None:
        raise PlanCompilationError("structured plan must include validation metadata before compilation")

    default_namespace = plan.namespace_strategy.target_namespace or _first_target_namespace(plan.components)
    if not default_namespace:
        raise PlanCompilationError("structured plan must resolve a default namespace before compilation")

    metadata = {
        "name": f"{plan.blueprint_family}-plan",
        "namespace": default_namespace,
    }
    spec = {
        "appStack": {
            "defaultNamespace": default_namespace,
            "components": [_compile_component(component, default_namespace) for component in plan.components],
        }
    }
    return {
        "apiVersion": DEFAULT_CRD_API_VERSION,
        "kind": DEFAULT_CRD_KIND,
        "metadata": metadata,
        "spec": spec,
    }


def render_wekaappstore_yaml(document: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(document), sort_keys=False)


def compile_plan_to_yaml(plan: StructuredPlan) -> str:
    return render_wekaappstore_yaml(compile_plan_to_wekaappstore(plan))


def validate_and_compile_plan(payload: Mapping[str, Any]) -> tuple[ValidationResult, Dict[str, Any] | None, str | None]:
    validation = validate_structured_plan(payload)
    if not validation.valid or validation.plan is None:
        return validation, None, None

    compiled = compile_plan_to_wekaappstore(validation.plan)
    rendered = render_wekaappstore_yaml(compiled)
    return validation, compiled, rendered


def _first_target_namespace(components: Iterable[ComponentPlan]) -> str | None:
    for component in components:
        if component.target_namespace:
            return component.target_namespace
    return None


def _compile_component(component: ComponentPlan, default_namespace: str) -> Dict[str, Any]:
    compiled = {
        "name": component.name,
        "enabled": component.enabled,
        "dependsOn": list(component.depends_on),
        "targetNamespace": component.target_namespace or default_namespace,
        "waitForReady": component.wait_for_ready,
    }

    if component.helm_chart is not None:
        compiled["helmChart"] = {
            "repository": component.helm_chart.repository,
            "name": component.helm_chart.name,
            "version": component.helm_chart.version,
            "releaseName": component.helm_chart.release_name or component.name,
            "crdsStrategy": component.helm_chart.crds_strategy,
        }
    if component.kubernetes_manifest is not None:
        compiled["kubernetesManifest"] = component.kubernetes_manifest
    if component.values:
        compiled["values"] = dict(component.values)
    if component.values_files:
        compiled["valuesFiles"] = [
            {
                "kind": values_file.kind,
                "name": values_file.name,
                "key": values_file.key,
            }
            for values_file in component.values_files
        ]
    if component.readiness_check is not None:
        readiness_check: Dict[str, Any] = {
            "type": component.readiness_check.type,
            "timeout": component.readiness_check.timeout,
        }
        if component.readiness_check.name is not None:
            readiness_check["name"] = component.readiness_check.name
        if component.readiness_check.selector is not None:
            readiness_check["selector"] = component.readiness_check.selector
        if component.readiness_check.match_labels:
            readiness_check["matchLabels"] = dict(component.readiness_check.match_labels)
        if component.readiness_check.namespace is not None:
            readiness_check["namespace"] = component.readiness_check.namespace
        compiled["readinessCheck"] = readiness_check

    return compiled
