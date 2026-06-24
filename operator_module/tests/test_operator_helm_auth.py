"""Unit tests locking the Phase 28 operator helm-auth / CRD-discovery invariants
(OPA-01 / OPA-02), implemented in plan 28-01.

Covers:
  - OPA-02 success-only CRD cache: a failed `helm show crds` is NOT memoized
    (T-28-05) and a successful result IS cached keyed by
    (chart_ref, version, registry_config_path) (D-06/D-07).
  - OPA-01 registry-config threading: `--registry-config <path>` appears in the
    helm argv for an oci://quay.io/... chart when `quay_dockerconfigjson` is in
    appStack.variables, and never appears otherwise (D-05).
  - Credential secrecy: the quay docker auth sentinel never appears in any
    captured helm argv element (T-28-01).
  - Temp-file lifecycle: the temp registry-config file is removed after
    handle_appstack_deployment returns, including the install-failure path
    (T-28-03 / D-04 try/finally).

Tests run in-process; all subprocess and kube interactions are mocked.
No cluster, no kubectl, no helm binary required.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))

from main import discover_chart_crds, handle_appstack_deployment  # noqa: E402
import main  # noqa: E402


CHART_REF = "oci://quay.io/weka.io/helm/weka-operator"
VERSION = "v1.13.0"

# A one-document CRD manifest as `helm show crds` would emit.
CRD_YAML = (
    "apiVersion: apiextensions.k8s.io/v1\n"
    "kind: CustomResourceDefinition\n"
    "metadata:\n"
    "  name: wekaclients.weka.weka.io\n"
)

# Recognizable sentinel embedded in the quay docker auth JSON. If this string
# (or the full dockerconfigjson) ever shows up in a captured helm argv element,
# the credential has leaked onto the command line (T-28-01).
SENTINEL_AUTH = "U0VOVElORUw="
QUAY_DOCKERCONFIGJSON = '{"auths":{"quay.io":{"auth":"' + SENTINEL_AUTH + '"}}}'


# ----------------------------------------------------------------------
# OPA-02: success-only CRD cache
# ----------------------------------------------------------------------


def test_failed_helm_show_crds_is_not_cached():
    """T-28-05: a failed `helm show crds` returns set() and is NOT memoized;
    the next identical call re-invokes the subprocess."""
    main._chart_crds_cache.clear()

    with patch(
        "main.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "helm"),
    ) as check_output:
        result = discover_chart_crds(CHART_REF, VERSION)

        assert result == set()
        assert (CHART_REF, VERSION, None) not in main._chart_crds_cache

        # Second call must re-attempt the subprocess (no negative memoization).
        discover_chart_crds(CHART_REF, VERSION)
        assert check_output.call_count == 2


def test_successful_helm_show_crds_is_cached():
    """D-06/D-07: a successful `helm show crds` is cached keyed by
    (chart_ref, version, registry_config_path); a second identical call is a
    cache hit and does not re-invoke the subprocess."""
    main._chart_crds_cache.clear()

    with patch(
        "main.subprocess.check_output", return_value=CRD_YAML
    ) as check_output:
        result = discover_chart_crds(CHART_REF, VERSION)

        assert "wekaclients.weka.weka.io" in result
        assert (CHART_REF, VERSION, None) in main._chart_crds_cache

        # Second identical call: served from cache, no re-invoke.
        again = discover_chart_crds(CHART_REF, VERSION)
        assert again == result
        assert check_output.call_count == 1


def test_registry_config_path_in_cache_key():
    """D-07: registry_config_path participates in the cache key, so a call with a
    path produces a distinct cache entry from one without."""
    main._chart_crds_cache.clear()

    with patch("main.subprocess.check_output", return_value=CRD_YAML):
        discover_chart_crds(CHART_REF, VERSION, registry_config_path=None)
        discover_chart_crds(CHART_REF, VERSION, registry_config_path="/tmp/x.json")

    assert (CHART_REF, VERSION, None) in main._chart_crds_cache
    assert (CHART_REF, VERSION, "/tmp/x.json") in main._chart_crds_cache
    assert len(main._chart_crds_cache) == 2


# ----------------------------------------------------------------------
# OPA-01: registry-config threading, credential secrecy, temp-file lifecycle
# ----------------------------------------------------------------------


def _quay_oci_appstack(with_quay_credential: bool):
    """Build an AppStack spec with one OCI/quay helmChart component.

    waitForReady False keeps the readiness path mocked-out simple; the OCI repo
    means `_add_repo` short-circuits (no `helm repo add` subprocess).
    """
    variables = {}
    if with_quay_credential:
        variables["quay_dockerconfigjson"] = QUAY_DOCKERCONFIGJSON

    component = {
        "name": "weka-operator",
        "helmChart": {
            "repository": "oci://quay.io/weka.io/helm",
            "name": "weka-operator",
            "version": VERSION,
            "releaseName": "weka-operator",
        },
        "waitForReady": False,
    }
    return {"appStack": {"variables": variables, "components": [component]}}


def _make_helm_run_capture(install_succeeds: bool = True):
    """Return (run_callable, check_output_callable, captured_argvs).

    Both callables record their argv (as a list) into `captured_argvs`. The
    real HelmOperator methods run against these mocks:
      - `helm status` (release-exists probe) returns rc 1 -> install path.
      - `helm install` returns rc 0 (or rc 1 when install_succeeds is False).
      - `helm show crds` (check_output) returns the CRD yaml.
    """
    captured: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "helm" and cmd[1] == "status":
            # Release does not exist -> drive the install path.
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="not found")
        rc = 0 if install_succeeds else 1
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="ok", stderr="boom")

    def _check_output(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return CRD_YAML

    return _run, _check_output, captured


def _registry_config_value(captured_argvs):
    """Return the path that immediately follows --registry-config in any captured
    argv, or None if the flag never appeared."""
    for argv in captured_argvs:
        for i, tok in enumerate(argv):
            if tok == "--registry-config" and i + 1 < len(argv):
                return argv[i + 1]
    return None


def _assert_credential_absent(captured_argvs):
    """Assert the quay auth sentinel and the full dockerconfigjson never appear
    in any captured argv element (T-28-01)."""
    for argv in captured_argvs:
        for tok in argv:
            assert SENTINEL_AUTH not in tok, f"credential sentinel leaked into argv: {argv}"
            assert QUAY_DOCKERCONFIGJSON not in tok, f"dockerconfigjson leaked into argv: {argv}"


def test_registry_config_flag_present_for_oci_quay_chart():
    """OPA-01 / T-28-01: with a quay credential present, `--registry-config <path>`
    is threaded into the helm argv, the file at that path holds the credential,
    and the credential never appears as an argv element."""
    main._chart_crds_cache.clear()
    spec = _quay_oci_appstack(with_quay_credential=True)
    run, check_output, captured = _make_helm_run_capture(install_succeeds=True)

    seen_contents: list[str] = []

    def _check_output_capture_file(cmd, *args, **kwargs):
        # Capture the registry-config file contents while it still exists.
        path = _registry_config_value([list(cmd)])
        if path and os.path.exists(path):
            with open(path) as fh:
                seen_contents.append(fh.read())
        captured.append(list(cmd))
        return CRD_YAML

    with patch("main.subprocess.run", side_effect=run), \
         patch("main.subprocess.check_output", side_effect=_check_output_capture_file), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True), \
         patch("main._patch_appstack_progress"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="weka",
            namespace="weka-operator-system",
            status={},
        )

    path = _registry_config_value(captured)
    assert path is not None, "--registry-config flag was not threaded into any helm argv"
    # The temp file content was the raw dockerconfigjson (read during the run).
    assert any(QUAY_DOCKERCONFIGJSON in c for c in seen_contents), (
        "registry-config file did not contain the quay credential"
    )
    _assert_credential_absent(captured)


def test_no_registry_config_flag_without_quay_credential():
    """D-05: with no quay credential in variables, `--registry-config` appears in
    NO captured helm argv (backward-compatible non-OCI/no-cred path)."""
    main._chart_crds_cache.clear()
    spec = _quay_oci_appstack(with_quay_credential=False)
    run, check_output, captured = _make_helm_run_capture(install_succeeds=True)

    with patch("main.subprocess.run", side_effect=run), \
         patch("main.subprocess.check_output", side_effect=check_output), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True), \
         patch("main._patch_appstack_progress"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="weka",
            namespace="weka-operator-system",
            status={},
        )

    assert _registry_config_value(captured) is None, (
        "--registry-config must not appear without a quay credential"
    )


def test_temp_registry_config_file_removed_after_return():
    """D-04 / T-28-03: the temp registry-config file is unlinked after
    handle_appstack_deployment returns (success path)."""
    main._chart_crds_cache.clear()
    spec = _quay_oci_appstack(with_quay_credential=True)
    run, check_output, captured = _make_helm_run_capture(install_succeeds=True)

    with patch("main.subprocess.run", side_effect=run), \
         patch("main.subprocess.check_output", side_effect=check_output), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True), \
         patch("main._patch_appstack_progress"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="weka",
            namespace="weka-operator-system",
            status={},
        )

    path = _registry_config_value(captured)
    assert path is not None
    assert not os.path.exists(path), "temp registry-config file should be removed after return"


def test_temp_registry_config_file_removed_on_install_failure():
    """T-28-03: the temp registry-config file is STILL removed when the helm
    install fails (try/finally exception-safe cleanup)."""
    main._chart_crds_cache.clear()
    spec = _quay_oci_appstack(with_quay_credential=True)
    run, check_output, captured = _make_helm_run_capture(install_succeeds=False)

    with patch("main.subprocess.run", side_effect=run), \
         patch("main.subprocess.check_output", side_effect=check_output), \
         patch("main._load_kube_config_once", return_value=False), \
         patch("main.wait_for_component_ready", return_value=True), \
         patch("main._patch_appstack_progress"):
        handle_appstack_deployment(
            body={"spec": spec},
            spec=spec,
            name="weka",
            namespace="weka-operator-system",
            status={},
        )

    path = _registry_config_value(captured)
    assert path is not None
    assert not os.path.exists(path), (
        "temp registry-config file should be removed even on the install-failure path"
    )
