"""Backward-compat snapshot test (TST-03 / OP-06) — inline-values + helm-install path.

A WekaAppStore CR WITHOUT a `spec.appStack.variables` block must produce
byte-identical merged Helm values dicts pre/post Phase 18. Phase 16's
render() pre-scan guard makes this true at the source level; this test
locks it at the integration level for the inline-values + helm-install
path specifically.

Coverage scope:
  - Fixture: mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml
        * Multi-component AppStack (vector-db helm + research-api helm).
        * NO `variables:` block.
        * Two `helm_chart` components with INLINE `values:` only.
        * NO valuesFiles references, NO kubernetesManifest components.
        * Field names are snake_case (helm_chart, target_namespace, etc.) —
          a small in-test normalization helper bridges to the camelCase the
          operator consumes.
  - This test locks the inline-values dict serialization through
    handle_appstack_deployment -> HelmOperator.install_or_upgrade.
  - The `valuesFiles` render path is locked by Plan 18-03 tests 5+6
    (test_configmap_valuesfile_substitutes_variables /
    test_secret_valuesfile_substitutes_variables).
  - The `kubernetesManifest` no-op path is locked indirectly via Plan 18-03
    tests 1+3 (rendered string equals source string when no `${...}` tokens
    are present in the manifest).

Re-generation: `BASELINE_REGEN=1 pytest operator_module/tests/test_backward_compat_snapshot.py`
rewrites baseline files. Commit the baselines to lock the contract.

Per CONTEXT.md D-11..D-13.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    REPO_ROOT / "mcp-server" / "tests" / "fixtures" / "sample_blueprints" / "ai-research.yaml"
)
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots" / "ai-research"

REGEN = os.environ.get("BASELINE_REGEN") == "1"

# snake_case -> camelCase rename map for fixture normalization.
# The fixture predates the camelCase convention used by handle_appstack_deployment.
# Per RESEARCH.md Pitfall 6 / CONTEXT.md D-11.
SNAKE_TO_CAMEL = {
    "helm_chart": "helmChart",
    "target_namespace": "targetNamespace",
    "release_name": "releaseName",
    "crds_strategy": "crdsStrategy",
    "wait_for_ready": "waitForReady",
    "readiness_check": "readinessCheck",
    "depends_on": "dependsOn",
    "values_files": "valuesFiles",
}


def _normalize_camel(node: Any) -> Any:
    """Recursively rename snake_case keys to camelCase per SNAKE_TO_CAMEL."""
    if isinstance(node, dict):
        return {
            SNAKE_TO_CAMEL.get(k, k): _normalize_camel(v)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_normalize_camel(item) for item in node]
    return node


@pytest.fixture(scope="module")
def fixture_cr() -> dict:
    """Load and normalize the ai-research.yaml fixture once per module."""
    raw = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    return _normalize_camel(raw)


def _run_handler_capture_helm_values(cr: dict) -> dict:
    """Invoke handle_appstack_deployment under mocks; return {release_name: values_dict}.

    The HelmOperator mock records every install_or_upgrade call. The verified call
    shape at operator_module/main.py:828-838 is:
        helm_operator.install_or_upgrade(
            name=release_name,
            chart=chart_ref,
            values=merged_values,
            namespace=target_namespace,
            repository=chart_repo,
            version=chart_version,
            skip_crds=skip_crds,
        )
    We extract `values` from kwargs and key the captured dict by `release_name`
    (the value of `name` kwarg, which the operator code populates from
    `helmChart.releaseName`). The parametrize list in the test below uses these
    release_name values directly — NOT the component `name` field — because the
    capture key is release_name (W-2 fix per planner reviews).
    """
    from main import handle_appstack_deployment

    helm_calls_by_release: dict = {}

    class _MockHelmOperator:
        def __init__(self, *args, **kwargs):
            self._init_args = args
            self._init_kwargs = kwargs

        def install_or_upgrade(self, *args, **kwargs):
            values = kwargs.get("values")
            if values is None:
                for a in args:
                    if isinstance(a, dict):
                        values = a
                        break
            release_name = kwargs.get("name")
            if release_name is None and self._init_args:
                release_name = self._init_args[0]
            if release_name is None:
                raise AssertionError(
                    "Could not determine release_name from HelmOperator call. "
                    "Inspect operator_module/main.py:828-838 and update "
                    "_run_handler_capture_helm_values() accordingly."
                )
            helm_calls_by_release[release_name] = values or {}
            return (True, "deployed")

        # No-op stubs for methods main.py touches in the helm path before install
        def _extract_repo_name(self, repo):
            return "stub-repo"

        def _add_repo(self, repo_name, repo_url):
            return None

    spec = cr["spec"]
    body = cr

    with patch("main.HelmOperator", _MockHelmOperator), \
         patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.should_skip_crds_for_component", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True):
        try:
            handle_appstack_deployment(
                body=body,
                spec=spec,
                name=body.get("metadata", {}).get("name", "ai-research"),
                namespace=body.get("metadata", {}).get("namespace", "ai-platform"),
                status={},
            )
        except Exception:
            # The test cares about HelmOperator capture; any post-install
            # failure (status patches, readiness, etc.) is uninteresting if
            # we already captured all expected components.
            pass

    return helm_calls_by_release


@pytest.mark.parametrize("release_name", ["qdrant", "research-api"])
def test_helm_values_byte_identical_to_baseline(fixture_cr, release_name):
    """TST-03 / OP-06: merged Helm values for ai-research's components are
    byte-identical to the captured baseline (no variables: block in fixture).

    The parametrize list uses release_name (helm chart release names from the
    fixture's `helm_chart.release_name` — `qdrant` for the vector-db component
    and `research-api` for the research-api component). NOT component name
    values — because the capture helper keys by release_name (the value of
    the `name` kwarg passed to HelmOperator.install_or_upgrade at main.py:829).
    """
    captured = _run_handler_capture_helm_values(fixture_cr)
    assert release_name in captured, (
        f"Expected release_name {release_name!r} not captured. "
        f"Got: {sorted(captured)!r}. "
        "Inspect main.py:828-838 helm-install path and the HelmOperator mock in this file. "
        "Note: parametrize uses release_name (from helm_chart.release_name in the fixture), "
        "not component.name."
    )
    values_dict = captured[release_name]
    captured_json = json.dumps(values_dict, indent=2, sort_keys=True)

    baseline_path = SNAPSHOTS_DIR / f"values_{release_name}.json"

    if REGEN:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(captured_json + "\n", encoding="utf-8")
        return

    if not baseline_path.exists():
        pytest.fail(
            f"Baseline missing: {baseline_path}\n"
            "Re-run with BASELINE_REGEN=1 pytest operator_module/tests/test_backward_compat_snapshot.py "
            "to generate it. Then commit the baseline file."
        )

    baseline = baseline_path.read_text(encoding="utf-8").rstrip("\n")
    assert captured_json == baseline, (
        f"TST-03 VIOLATED: merged Helm values for release_name {release_name!r} "
        f"drifted from baseline.\n"
        f"Baseline file: {baseline_path}\n"
        "Re-run with BASELINE_REGEN=1 pytest operator_module/tests/test_backward_compat_snapshot.py "
        "if the change is intentional and document the drift in the PR.\n"
        f"--- expected (baseline) ---\n{baseline}\n"
        f"--- actual (current) ---\n{captured_json}"
    )
