"""Cluster-free verification of the Phase 27 install blueprint contract.

Tests render ``cluster_init/app-store-install.yaml`` through the same [[ var ]]
Jinja2 path the GUI uses, verify multi-doc YAML validity, feed the rendered
components into the operator's REAL ``resolve_dependencies``, and assert the
D-01 topo order without requiring a live cluster.

Run with:
    PYTHONPATH=operator_module pytest operator_module/tests/test_install_blueprint.py -v
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment

# conftest.py inserts operator_module/ on sys.path so this import works
from main import resolve_dependencies


# ---------------------------------------------------------------------------
# Blueprint file path
# ---------------------------------------------------------------------------

BLUEPRINT_PATH = Path(__file__).resolve().parents[2] / "cluster_init" / "app-store-install.yaml"


# ---------------------------------------------------------------------------
# Sample variables
# ---------------------------------------------------------------------------
# These cover every [[ ]] token in the blueprint:
#   - 8 x-variables keys from the YAML front matter
#   - 2 Phase-29 server-injected tokens (join_ip_ports_list, endpoints_csv)
#   - quay_dockerconfigjson built as a realistic base64 dockerconfigjson string

def _build_sample_quay_dockerconfigjson(user: str, password: str) -> str:
    """Build a realistic dockerconfigjson string the way Phase 29 will.

    Constructs: {"auths": {"quay.io": {"auth": base64(user:password)}}}
    Returns the JSON string (not base64-encoded — this is the plain JSON that
    goes into stringData['.dockerconfigjson']).
    """
    auth_token = base64.b64encode(f"{user}:{password}".encode()).decode()
    config = {"auths": {"quay.io": {"auth": auth_token}}}
    return json.dumps(config)


SAMPLE_VARS: dict[str, str] = {
    # x-variables (8 keys)
    "operator_version": "v1.13.0",
    "weka_image_version": "5.1.0.605",
    "join_ip_ports": "10.0.0.1:14000,10.0.0.2:14000",
    "weka_endpoint_scheme": "http",
    "weka_org": "Root",
    "weka_username": "admin",
    "weka_password": "secret",
    "quay_dockerconfigjson": _build_sample_quay_dockerconfigjson("user", "pass"),
    # Phase-29 server-injected tokens
    "join_ip_ports_list": '["10.0.0.1:14000"]',
    "endpoints_csv": "10.0.0.1:14000,10.0.0.2:14000",
}


# ---------------------------------------------------------------------------
# Shared render helper
# ---------------------------------------------------------------------------

def _render_blueprint() -> str:
    """Render the blueprint with SAMPLE_VARS using the GUI's exact Jinja2 delimiters."""
    raw = BLUEPRINT_PATH.read_text(encoding="utf-8")
    # Reproduce the exact Environment from app-store-gui/webapp/main.py:2918
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    template = env.from_string(raw)
    return template.render(**SAMPLE_VARS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_parses():
    """Blueprint renders with sample vars and parses as valid multi-doc YAML
    containing at least one WekaAppStore document.
    """
    rendered = _render_blueprint()
    docs = list(yaml.safe_load_all(rendered))
    assert docs, "Expected at least one YAML document"
    kinds = [d.get("kind") for d in docs if isinstance(d, dict)]
    assert "WekaAppStore" in kinds, (
        f"Expected a WekaAppStore document; found kinds: {kinds}"
    )


def test_topo_order():
    """resolve_dependencies returns a valid topo order that satisfies the D-01
    dependency edges.

    Uses index-of comparisons (a < b) rather than a single rigid sequence so
    parallel groups (which are interchangeable) do not cause false failures.
    """
    rendered = _render_blueprint()
    docs = list(yaml.safe_load_all(rendered))
    was_cr = next(
        (d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"),
        None,
    )
    assert was_cr is not None, "WekaAppStore document not found in rendered output"

    components = was_cr["spec"]["appStack"]["components"]
    ordered = resolve_dependencies(components)
    idx = {comp["name"]: i for i, comp in enumerate(ordered)}

    # D-01 required edges (a must appear before b)
    edges = [
        # Quay secrets before operator
        ("quay-secret-operator-ns", "weka-operator"),
        ("quay-secret-default-ns", "weka-operator"),
        # Operator before CRD-dependent components
        ("weka-operator", "weka-node-label-sa"),
        ("weka-operator", "weka-client-secret"),
        ("weka-operator", "csi-wekafs"),
        # Node-label chain
        ("weka-node-label-sa", "weka-node-label-rbac"),
        ("weka-node-label-rbac", "weka-node-label-job"),
        # CSI chain
        ("csi-wekafs", "csi-api-secret"),
        # WekaClient requires operator + secret
        ("weka-operator", "weka-client"),
        ("weka-client-secret", "weka-client"),
        # StorageClass chain (SC1)
        ("csi-api-secret", "storageclass-demote-job"),
        ("storageclass-demote-job", "storageclasses"),
    ]

    for a, b in edges:
        assert idx[a] < idx[b], (
            f"D-01 order violation: '{a}' (index {idx[a]}) must precede "
            f"'{b}' (index {idx[b]})"
        )


def test_quay_roundtrip():
    """Quay dockerconfigjson injects correctly and round-trips to exactly
    'user:pass' with no trailing newline.

    NOTE: This test validates a synthetic, test-local dockerconfigjson builder
    against the blueprint's single passthrough token [[ quay_dockerconfigjson ]].
    The authoritative round-trip guard against the REAL server-side quay-builder
    (the actual fix for the trailing-newline bug class) lives in Phase 29
    (ROADMAP Phase 29 SC2). This test does NOT, by itself, close the
    trailing-newline bug class for Phase 27.
    """
    rendered = _render_blueprint()
    docs = list(yaml.safe_load_all(rendered))
    was_cr = next(
        (d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"),
        None,
    )
    assert was_cr is not None

    components = was_cr["spec"]["appStack"]["components"]
    # Find the quay-secret-operator-ns component
    quay_comp = next(
        (c for c in components if c["name"] == "quay-secret-operator-ns"),
        None,
    )
    assert quay_comp is not None, "quay-secret-operator-ns component not found"

    # Parse the kubernetesManifest inside the component to get the Secret
    secret_docs = list(yaml.safe_load_all(quay_comp["kubernetesManifest"]))
    secret = next(
        (d for d in secret_docs if isinstance(d, dict) and d.get("kind") == "Secret"),
        None,
    )
    assert secret is not None, "Secret document not found in quay-secret-operator-ns manifest"

    dockerconfig_json_str = secret["stringData"][".dockerconfigjson"]
    config = json.loads(dockerconfig_json_str)
    encoded_auth = config["auths"]["quay.io"]["auth"]
    decoded = base64.b64decode(encoded_auth)

    assert decoded == b"user:pass", (
        f"Expected b'user:pass', got {decoded!r} — trailing newline or encoding error"
    )


def test_single_default_sc():
    """Exactly one StorageClass carries is-default-class == 'true' and it must
    be storageclass-wekafs-dir-api (INST-08 / SC4).
    """
    rendered = _render_blueprint()
    docs = list(yaml.safe_load_all(rendered))
    was_cr = next(
        (d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"),
        None,
    )
    assert was_cr is not None

    components = was_cr["spec"]["appStack"]["components"]
    sc_comp = next(
        (c for c in components if c["name"] == "storageclasses"),
        None,
    )
    assert sc_comp is not None, "storageclasses component not found"

    sc_docs = list(yaml.safe_load_all(sc_comp["kubernetesManifest"]))
    storage_classes = [d for d in sc_docs if isinstance(d, dict) and d.get("kind") == "StorageClass"]
    assert storage_classes, "No StorageClass documents found in storageclasses manifest"

    defaults = [
        sc for sc in storage_classes
        if sc.get("metadata", {}).get("annotations", {}).get(
            "storageclass.kubernetes.io/is-default-class"
        ) == "true"
    ]

    assert len(defaults) == 1, (
        f"Expected exactly one default StorageClass, found {len(defaults)}: "
        f"{[sc['metadata']['name'] for sc in defaults]}"
    )
    assert defaults[0]["metadata"]["name"] == "storageclass-wekafs-dir-api", (
        f"Expected default SC 'storageclass-wekafs-dir-api', got "
        f"'{defaults[0]['metadata']['name']}'"
    )


def test_stringdata_only():
    """No Opaque or dockerconfigjson Secret in the rendered output uses a
    ``data:`` field — all wizard secrets use ``stringData`` (INST-09).
    """
    rendered = _render_blueprint()
    docs = list(yaml.safe_load_all(rendered))
    was_cr = next(
        (d for d in docs if isinstance(d, dict) and d.get("kind") == "WekaAppStore"),
        None,
    )
    assert was_cr is not None

    components = was_cr["spec"]["appStack"]["components"]
    violations = []

    for comp in components:
        manifest_text = comp.get("kubernetesManifest")
        if not manifest_text:
            continue
        manifest_docs = list(yaml.safe_load_all(manifest_text))
        for doc in manifest_docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("kind") != "Secret":
                continue
            secret_type = doc.get("type", "Opaque")
            if secret_type in ("Opaque", "kubernetes.io/dockerconfigjson"):
                if "data" in doc:
                    violations.append(
                        f"Component '{comp['name']}' Secret "
                        f"'{doc.get('metadata', {}).get('name', '?')}' "
                        f"uses 'data:' instead of 'stringData:'"
                    )

    assert not violations, "Secrets must use stringData, not data:\n" + "\n".join(violations)
