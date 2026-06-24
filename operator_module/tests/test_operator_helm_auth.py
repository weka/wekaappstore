"""Unit tests locking the Phase 28 operator helm-auth / CRD-discovery invariants
(OPA-01 / OPA-02), implemented in plan 28-01.

Covers:
  - OPA-02 success-only CRD cache: a failed `helm show crds` is NOT memoized
    (T-28-05) and a successful result IS cached keyed by
    (chart_ref, version, registry_config_path) (D-06/D-07).

Tests run in-process; all subprocess and kube interactions are mocked.
No cluster, no kubectl, no helm binary required.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))

from main import discover_chart_crds  # noqa: E402
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
