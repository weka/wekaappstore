"""Unit tests for operator_module.main.handle_appstack_deployment substitution wiring (TST-02).

Covers OP-06 (variables-dict build with ${namespace} auto-default + explicit override),
OP-07 (kubernetesManifest render before kubectl apply, PermanentError on undefined),
OP-08 (ConfigMap and Secret valuesFiles render before yaml.safe_load),
OP-10 (invalid variable key + non-string value -> PermanentError),
OP-11 (kr8s exception -> typed kopf error dispatch),
OP-12 (@kopf.on.update is gated by field='spec').

Tests run in-process; all kr8s, subprocess, and HelmOperator interactions are mocked.
No cluster, no kubectl, no helm binary required.
"""
from __future__ import annotations

import base64
import subprocess as _real_subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))


# ----------------------------------------------------------------------
# Helper builders
# ----------------------------------------------------------------------


def _make_kubectl_run_capture():
    """Return (mock_run_callable, captured_manifests_list).

    The callable mirrors subprocess.run's signature and, for any
    `kubectl apply -f <path> ...` invocation, reads the tempfile content
    into the captured list BEFORE returning a successful CompletedProcess.
    """
    captured: list[str] = []

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 4 and cmd[0] == "kubectl" and cmd[1] == "apply" and cmd[2] == "-f":
            if cmd[3] == "-" and kwargs.get("input") is not None:
                # Per-document apply: manifest is piped via stdin (-f -).
                captured.append(kwargs["input"])
            else:
                try:
                    captured.append(Path(cmd[3]).read_text(encoding="utf-8"))
                except OSError:
                    # tempfile may have been unlinked already in some paths; ignore
                    pass
        return _real_subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    return _run, captured


def _make_kr8s_cm(data_dict):
    """Return a MagicMock with `.data = data_dict` (mimics kr8s ConfigMap)."""
    cm = MagicMock()
    cm.data = dict(data_dict)
    return cm


def _make_kr8s_secret(data_dict):
    """Return a MagicMock whose .data maps each key to base64-encoded value (mimics kr8s Secret)."""
    secret = MagicMock()
    secret.data = {
        k: base64.b64encode(v.encode("utf-8") if isinstance(v, str) else v).decode("utf-8")
        for k, v in data_dict.items()
    }
    return secret


def _make_kr8s_server_error(status_code, message="server error"):
    """Construct a kr8s.ServerError with .response.status_code set."""
    import kr8s

    err = kr8s.ServerError(message)
    response = MagicMock()
    response.status_code = status_code
    err.response = response
    return err


def _appstack_oci_helm_component(comp_name="vector-db", values_files=None):
    """Helper to build a minimal helm component spec using OCI repo (skips _add_repo path)."""
    comp = {
        "name": comp_name,
        "helmChart": {
            "repository": "oci://example.registry/charts",
            "name": "milvus",
            "version": "4.2.1",
            "releaseName": comp_name,
        },
        "waitForReady": False,
    }
    if values_files is not None:
        comp["valuesFiles"] = values_files
    return comp


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_namespace_auto_defaults_to_cr_namespace():
    """OP-06: ${namespace} auto-defaults to CR's metadata.namespace when no variables block is set."""
    from main import handle_appstack_deployment

    components = [{
        "name": "ingress",
        "kubernetesManifest": "metadata:\n  namespace: ${namespace}\n",
    }]
    spec = {"appStack": {"components": components}}

    mock_run, captured = _make_kubectl_run_capture()
    with patch("main.subprocess.run", side_effect=mock_run), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    assert len(captured) == 1
    assert "namespace: staging" in captured[0]
    assert "${namespace}" not in captured[0]


def test_explicit_namespace_override_wins():
    """OP-06: user-supplied `namespace` key in variables block overrides the CR-namespace auto-default."""
    from main import handle_appstack_deployment

    components = [{
        "name": "ingress",
        "kubernetesManifest": "metadata:\n  namespace: ${namespace}\n",
    }]
    spec = {"appStack": {
        "variables": {"namespace": "special"},
        "components": components,
    }}

    mock_run, captured = _make_kubectl_run_capture()
    with patch("main.subprocess.run", side_effect=mock_run), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    assert len(captured) == 1
    assert "namespace: special" in captured[0]
    assert "namespace: staging" not in captured[0]


def test_kubernetes_manifest_substitutes_namespace():
    """OP-07: ${VAR} in kubernetesManifest renders before kubectl apply."""
    from main import handle_appstack_deployment

    components = [{
        "name": "ingress",
        "kubernetesManifest": (
            "apiVersion: v1\n"
            "kind: Service\n"
            "metadata:\n"
            "  name: web\n"
            "  namespace: ${namespace}\n"
        ),
    }]
    spec = {"appStack": {"components": components}}

    mock_run, captured = _make_kubectl_run_capture()
    with patch("main.subprocess.run", side_effect=mock_run), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="prod",
            status={},
        )

    assert len(captured) == 1
    assert "namespace: prod" in captured[0]
    assert "${" not in captured[0]


def test_appstack_manifest_applies_cross_namespace():
    """A multi-namespace manifest applies each doc with its own namespace; the
    component target namespace is the default, and cluster-scoped kinds get no -n.

    Regression: previously the whole manifest was applied with a single
    `-n <target>`, which kubectl rejects for objects declaring a different
    namespace (the App Store credential bootstrap creates RBAC in wekaappstore)."""
    from main import handle_appstack_deployment

    components = [{
        "name": "ngc-bootstrap",
        "kubernetesManifest": (
            "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: a\n  namespace: ${namespace}\n"
            "---\n"
            "apiVersion: rbac.authorization.k8s.io/v1\nkind: Role\nmetadata:\n  name: b\n  namespace: wekaappstore\n"
            "---\n"
            "apiVersion: rbac.authorization.k8s.io/v1\nkind: ClusterRole\nmetadata:\n  name: c\n"
        ),
    }]
    spec = {"appStack": {"components": components}}

    calls = []

    def _run(cmd, *args, **kwargs):
        calls.append((list(cmd), kwargs.get("input", "")))
        return _real_subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    with patch("main.subprocess.run", side_effect=_run), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"):
        handle_appstack_deployment(
            body={"spec": spec}, spec=spec, name="x", namespace="rag", status={},
        )

    def ns_for(kind):
        for cmd, inp in calls:
            if cmd[:3] == ["kubectl", "apply", "-f"] and f"kind: {kind}" in inp:
                return cmd[cmd.index("-n") + 1] if "-n" in cmd else None
        raise AssertionError(f"no apply found for {kind}; calls={calls}")

    assert ns_for("ConfigMap") == "rag"          # ${namespace} default
    assert ns_for("Role") == "wekaappstore"      # explicit cross-namespace
    assert ns_for("ClusterRole") is None         # cluster-scoped, no -n


def test_delete_renders_namespace_in_manifest():
    """Delete path renders ${VAR} so kubectl delete's manifest namespace matches (regression).

    Without rendering, kubectl delete receives a manifest with metadata.namespace: ${namespace},
    which fails: 'the namespace from the provided object "${namespace}" does not match ...'.
    """
    from main import delete_warrpappstore_function

    components = [{
        "name": "ingress",
        "kubernetesManifest": (
            "apiVersion: v1\n"
            "kind: Service\n"
            "metadata:\n"
            "  name: web\n"
            "  namespace: ${namespace}\n"
        ),
    }]
    spec = {"appStack": {"components": components}}

    captured: list[str] = []

    def _run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 4 and cmd[0] == "kubectl" and cmd[1] == "delete" and cmd[2] == "-f":
            if cmd[3] == "-" and kwargs.get("input") is not None:
                captured.append(kwargs["input"])
            else:
                try:
                    captured.append(Path(cmd[3]).read_text(encoding="utf-8"))
                except OSError:
                    pass
        return _real_subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    with patch("main.subprocess.run", side_effect=_run), \
         patch("main.HelmOperator"):
        delete_warrpappstore_function(spec=spec, name="ai-research", namespace="rag")

    assert len(captured) == 1
    assert "namespace: rag" in captured[0]
    assert "${" not in captured[0]


def test_unknown_variable_in_manifest_left_untouched():
    """Allowlist contract (supersedes OP-07/DOC-04): an unprovided ${VAR} in a
    manifest is NOT an error — it is preserved verbatim, since it is
    indistinguishable from a legitimate shell ${VAR}. Undefined-variable
    detection moved to the variable-resolution layer.
    """
    from main import _render_or_raise

    out = _render_or_raise(
        "metadata:\n  namespace: ${unset}\n",
        {"namespace": "staging"},
        source_desc="Component 'ingress'.kubernetesManifest",
    )
    assert out == "metadata:\n  namespace: ${unset}\n"


def test_configmap_valuesfile_substitutes_variables():
    """OP-08: ConfigMap valuesFiles content with ${VAR} renders before yaml.safe_load."""
    from main import handle_appstack_deployment

    cm = _make_kr8s_cm({"values.yaml": "host: ${milvusHost}\n"})

    components = [_appstack_oci_helm_component(
        comp_name="vector-db",
        values_files=[{"kind": "ConfigMap", "name": "milvus-values", "key": "values.yaml"}],
    )]
    spec = {"appStack": {
        "variables": {"milvusHost": "milvus.aidp-prod.svc.cluster.local"},
        "components": components,
    }}

    mock_helm_cls = MagicMock()
    mock_helm = mock_helm_cls.return_value
    mock_helm.install_or_upgrade.return_value = (True, "ok")

    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", return_value=cm), \
         patch("main.HelmOperator", mock_helm_cls), \
         patch("main.should_skip_crds_for_component", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="aidp-prod",
            status={},
        )

    # The merged values dict was passed to install_or_upgrade as values=merged_values.
    install_calls = mock_helm.install_or_upgrade.call_args_list
    assert len(install_calls) == 1, f"expected 1 install call, got {len(install_calls)}"
    values_dict = install_calls[0].kwargs.get("values") or install_calls[0][1].get("values")
    assert values_dict == {"host": "milvus.aidp-prod.svc.cluster.local"}, (
        f"Expected substituted host; got {values_dict!r}"
    )


def test_secret_valuesfile_substitutes_variables():
    """OP-08: Secret valuesFiles content (base64-decoded) with ${VAR} renders before yaml.safe_load."""
    from main import handle_appstack_deployment

    secret = _make_kr8s_secret({"values.yaml": "host: ${milvusHost}\n"})

    components = [_appstack_oci_helm_component(
        comp_name="vector-db",
        values_files=[{"kind": "Secret", "name": "milvus-secret", "key": "values.yaml"}],
    )]
    spec = {"appStack": {
        "variables": {"milvusHost": "milvus.aidp-prod.svc.cluster.local"},
        "components": components,
    }}

    mock_helm_cls = MagicMock()
    mock_helm = mock_helm_cls.return_value
    mock_helm.install_or_upgrade.return_value = (True, "ok")

    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.Secret.get", return_value=secret), \
         patch("main.HelmOperator", mock_helm_cls), \
         patch("main.should_skip_crds_for_component", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="aidp-prod",
            status={},
        )

    install_calls = mock_helm.install_or_upgrade.call_args_list
    assert len(install_calls) == 1
    values_dict = install_calls[0].kwargs.get("values") or install_calls[0][1].get("values")
    assert values_dict == {"host": "milvus.aidp-prod.svc.cluster.local"}


def test_invalid_variable_key_raises_permanent_error():
    """OP-10: hyphenated key (`my-host`) is rejected at variables-dict build time."""
    import kopf
    from main import handle_appstack_deployment

    spec = {"appStack": {
        "variables": {"my-host": "foo"},
        "components": [{"name": "x", "kubernetesManifest": "x: 1\n"}],
    }}

    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"), \
         pytest.raises(kopf.PermanentError) as exc_info:
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    msg = str(exc_info.value)
    assert "my-host" in msg
    assert "Python identifier" in msg or "[_a-zA-Z]" in msg


def test_non_string_variable_value_raises_permanent_error():
    """OP-10: non-string value (e.g., integer) is rejected at variables-dict build time."""
    import kopf
    from main import handle_appstack_deployment

    spec = {"appStack": {
        "variables": {"x": 42},
        "components": [{"name": "x", "kubernetesManifest": "x: 1\n"}],
    }}

    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"), \
         pytest.raises(kopf.PermanentError) as exc_info:
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    msg = str(exc_info.value)
    assert "must be a string" in msg


def test_missing_configmap_raises_temporary_error():
    """OP-11: kr8s.NotFoundError -> kopf.TemporaryError(delay=30).

    Targets load_values_from_reference directly because handle_appstack_deployment
    has a per-component `except Exception` that swallows all errors into
    comp_status['message']. The OP-11 dispatch contract lives on
    load_values_from_reference (Phase 18 D-01..D-04).
    """
    import kopf
    import kr8s
    from main import load_values_from_reference

    with patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", side_effect=kr8s.NotFoundError("not found")), \
         pytest.raises(kopf.TemporaryError) as exc_info:
        load_values_from_reference(
            kind="ConfigMap",
            name="missing-cm",
            key="values.yaml",
            namespace="staging",
            comp_name="vector-db",
            ref_index=0,
        )

    assert getattr(exc_info.value, "delay", None) == 30
    assert "missing-cm" in str(exc_info.value)
    assert "vector-db" in str(exc_info.value)


def test_rbac_denied_raises_permanent_error():
    """OP-11: kr8s.ServerError with status_code=403 -> kopf.PermanentError (NOT TemporaryError)."""
    import kopf
    from main import load_values_from_reference

    with patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", side_effect=_make_kr8s_server_error(403, "forbidden")), \
         pytest.raises(kopf.PermanentError) as exc_info:
        load_values_from_reference(
            kind="ConfigMap",
            name="denied-cm",
            key="values.yaml",
            namespace="staging",
            comp_name="vector-db",
            ref_index=0,
        )

    msg = str(exc_info.value)
    assert "denied-cm" in msg


def test_api_timeout_raises_temporary_error():
    """OP-11: kr8s.APITimeoutError -> kopf.TemporaryError(delay=30)."""
    import kopf
    import kr8s
    from main import load_values_from_reference

    with patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", side_effect=kr8s.APITimeoutError("timed out")), \
         pytest.raises(kopf.TemporaryError) as exc_info:
        load_values_from_reference(
            kind="ConfigMap",
            name="slow-cm",
            key="values.yaml",
            namespace="staging",
            comp_name="vector-db",
            ref_index=0,
        )

    assert getattr(exc_info.value, "delay", None) == 30
    assert "slow-cm" in str(exc_info.value)


def test_malformed_yaml_raises_permanent_error():
    """OP-11: malformed YAML in ConfigMap valuesFiles -> kopf.PermanentError chained from yaml.YAMLError."""
    import kopf
    import yaml
    from main import load_values_from_reference

    malformed_cm = _make_kr8s_cm({
        "values.yaml": "key: value\nbad: {unclosed: [list, of, items\n",
    })

    with patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", return_value=malformed_cm), \
         pytest.raises(kopf.PermanentError) as exc_info:
        # No `variables` arg -> render() short-circuits via pre-scan guard;
        # yaml.safe_load is the exclusive failure point.
        load_values_from_reference(
            kind="ConfigMap",
            name="malformed-cm",
            key="values.yaml",
            namespace="staging",
            comp_name="vector-db",
            ref_index=0,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, yaml.YAMLError), (
        f"Expected kopf.PermanentError chained from yaml.YAMLError; "
        f"got chain ending with {type(cause).__name__}: {cause!r}"
    )
    assert "malformed-cm" in str(exc_info.value)


def test_handle_appstack_propagates_permanent_error_to_kopf_boundary():
    """OP-07 reconcile-boundary: kopf.PermanentError raised by render() inside the
    component loop must propagate OUT of handle_appstack_deployment, not be
    swallowed into comp_status['message']. Phase 18 closure fix added the explicit
    re-raise at main.py:899-905 before the broad `except Exception`.
    """
    import kopf
    from main import handle_appstack_deployment

    components = [{
        "name": "ingress",
        "kubernetesManifest": "metadata:\n  namespace: ${namespace}\n",
    }]
    spec = {"appStack": {"components": components}}

    # render() no longer raises on undefined vars (allowlist contract), so the
    # trigger is mocked: any kopf.PermanentError raised while processing a
    # component must propagate out, not be swallowed into comp_status['message'].
    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.HelmOperator"), \
         patch("main._render_or_raise",
               side_effect=kopf.PermanentError(
                   "Component 'ingress'.kubernetesManifest: simulated permanent failure")), \
         pytest.raises(kopf.PermanentError) as exc_info:
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    msg = str(exc_info.value)
    assert "ingress" in msg


def test_handle_appstack_propagates_temporary_error_to_kopf_boundary():
    """OP-11 reconcile-boundary: kopf.TemporaryError(delay=30) raised by
    load_values_from_reference inside the component loop must propagate OUT of
    handle_appstack_deployment so kopf can reschedule the reconcile. Pre-Phase-18
    swallow at main.py:899 hid transient cluster failures from kopf; the closure
    fix re-raises kopf.* explicitly.
    """
    import kopf
    import kr8s
    from main import handle_appstack_deployment

    components = [_appstack_oci_helm_component(
        comp_name="vector-db",
        values_files=[{"kind": "ConfigMap", "name": "missing-cm", "key": "values.yaml"}],
    )]
    spec = {"appStack": {"components": components}}

    with patch("main.subprocess.run"), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.kr8s.objects.ConfigMap.get", side_effect=kr8s.NotFoundError("not found")), \
         patch("main.HelmOperator"), \
         patch("main.should_skip_crds_for_component", return_value=False), \
         pytest.raises(kopf.TemporaryError) as exc_info:
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="ai-research",
            namespace="staging",
            status={},
        )

    assert getattr(exc_info.value, "delay", None) == 30
    assert "missing-cm" in str(exc_info.value)


def test_update_handler_has_field_spec_filter():
    """OP-12: @kopf.on.update at update_warrpappstore_function carries field='spec' filter."""
    main_path = OPERATOR_MODULE_ROOT / "main.py"
    source = main_path.read_text(encoding="utf-8")
    assert "@kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')" in source, (
        "OP-12: @kopf.on.update decorator must include field='spec' filter to prevent "
        "reconcile-storms from operator's own status patches. Expected line:\n"
        "    @kopf.on.update('warp.io', 'v1alpha1', 'wekaappstores', field='spec')"
    )
