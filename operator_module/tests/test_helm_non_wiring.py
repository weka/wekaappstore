"""TST-05: Lock the non-wiring invariant for handle_helm_deployment.

Phase 18 wires Phase 16's render() into handle_appstack_deployment but
deliberately skips the single-chart helm path. This file enforces that
invariant with TWO layers:

  1. Runtime: when handle_helm_deployment calls load_values_from_reference,
     the call uses kwarg-only form `kind=, name=, key=, namespace=` with
     zero positional args and NO `variables=`/`comp_name=`/`ref_index=` kwargs.
  2. Static: inspect.getsource(handle_helm_deployment) contains no
     `render(` substring — the helm path does not call render directly
     either.

If a future maintainer accidentally wires variables into the helm path,
one of these tests fails. Per CONTEXT.md D-14 / OP-09 / L-06.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))


def test_handle_helm_deployment_does_not_pass_variables():
    """OP-09 / L-06: handle_helm_deployment calls load_values_from_reference
    with kwarg-only form `kind=, name=, key=, namespace=`, NEVER with
    `variables=`/`comp_name=`/`ref_index=` kwargs.

    The verified call shape at operator_module/main.py:1020-1025 is multi-line
    kwargs only (zero positional args). Test asserts that exact shape.
    """
    from main import handle_helm_deployment

    # Single-chart CR with a top-level valuesFiles entry (helm path reads
    # spec['valuesFiles'], NOT spec['helmChart']['valuesFiles']).
    spec = {
        "helmChart": {
            "repository": "oci://example.registry/charts",
            "name": "milvus",
            "version": "4.2.1",
            "releaseName": "single-helm",
        },
        "valuesFiles": [
            {"kind": "ConfigMap", "name": "milvus-values", "key": "values.yaml"},
        ],
    }

    mock_load = MagicMock(return_value={})
    mock_helm_cls = MagicMock()
    mock_helm = mock_helm_cls.return_value
    mock_helm.install_or_upgrade.return_value = (True, "ok")

    with patch("main.load_values_from_reference", mock_load), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.subprocess.run"), \
         patch("main.should_skip_crds_for_component", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True), \
         patch("main.HelmOperator", mock_helm_cls):
        try:
            handle_helm_deployment(
                body={"spec": spec},
                spec=spec,
                name="single-helm",
                namespace="staging",
                status={},
            )
        except Exception:
            # Function may raise after the load_values call (e.g., HelmOperator mock
            # produces a partial result); we only care about the call shape.
            pass

    # The mock MUST have been called at least once (otherwise the test is vacuous).
    assert mock_load.call_count >= 1, (
        "handle_helm_deployment did not invoke load_values_from_reference; "
        "the test fixture may not exercise the valuesFiles path."
    )

    # Every call must use kwarg-only form `kind=, name=, key=, namespace=`
    # with NO `variables=`/`comp_name=`/`ref_index=` and zero positional args.
    # This matches the verified call shape at operator_module/main.py:1020-1025.
    for call in mock_load.call_args_list:
        args, kwargs = call
        assert "variables" not in kwargs, (
            "OP-09 / L-06 VIOLATED: handle_helm_deployment passed "
            f"`variables=` kwarg to load_values_from_reference: {kwargs!r}. "
            "The single-chart helm path must NOT be wired with variables."
        )
        assert "comp_name" not in kwargs, (
            "OP-09 / L-06 VIOLATED: handle_helm_deployment passed "
            f"`comp_name=` kwarg to load_values_from_reference: {kwargs!r}."
        )
        assert "ref_index" not in kwargs, (
            "OP-09 / L-06 VIOLATED: handle_helm_deployment passed "
            f"`ref_index=` kwarg to load_values_from_reference: {kwargs!r}."
        )
        assert len(args) == 0, (
            f"OP-09 / L-06 VIOLATED: expected zero positional args (kwarg-only "
            f"call shape `kind=, name=, key=, namespace=`); got {len(args)} "
            f"positional args: {args!r}"
        )
        assert set(kwargs.keys()) == {"kind", "name", "key", "namespace"}, (
            f"OP-09 / L-06 VIOLATED: helm-path call signature drift. "
            f"Expected exactly {{kind, name, key, namespace}}; got {sorted(kwargs)}. "
            f"Full kwargs: {kwargs!r}"
        )


def test_handle_helm_deployment_source_has_no_render():
    """OP-09 / L-06: handle_helm_deployment body does NOT call render().

    Static guard via inspect.getsource. handle_helm_deployment is NOT
    kopf-decorated (only update_warrpappstore_function and
    create_warrpappstore_function carry @kopf.on.* decorators), so
    inspect.getsource() returns the actual function body. Verified per
    RESEARCH.md Pitfall 3.
    """
    from main import handle_helm_deployment

    source = inspect.getsource(handle_helm_deployment)
    # Strip comments to keep the gate self-invalidation-proof: a future
    # comment that mentions "render(" must not flip the test green/red
    # spuriously. We check the non-comment text.
    non_comment_lines = [
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    ]
    non_comment_source = "\n".join(non_comment_lines)

    assert "render(" not in non_comment_source, (
        "OP-09 / L-06 VIOLATED: handle_helm_deployment source contains a "
        "render() call. The single-chart helm path must NOT invoke render. "
        f"Found in:\n{source}"
    )
    assert "_render_or_raise(" not in non_comment_source, (
        "OP-09 / L-06 VIOLATED: handle_helm_deployment source contains a "
        "_render_or_raise() call. The single-chart helm path must NOT "
        "invoke the render-or-raise wrapper either. "
        f"Found in:\n{source}"
    )
    assert "stack_vars" not in non_comment_source, (
        "OP-09 / L-06 VIOLATED: handle_helm_deployment source references "
        "`stack_vars`. The variables dict is built inside "
        "handle_appstack_deployment and must NOT leak into the helm path. "
        f"Found in:\n{source}"
    )
