"""Unit tests for operator_module.main.render().

Tests cover: pre-scan guard, $$ escape, JSON-safety (plain + substitution),
undefined-variable error, malformed-placeholder error, no-op when variables
is None or {}, multi-occurrence, and the cluster_init shell-script regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- sys.path setup (defense-in-depth; conftest.py also does this) ---
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))


# --- Module-level fixture constants -------------------------------------------

# Real AIDP imagePullSecret payload (per CONTEXT.md D-12 — inline literal,
# NOT loaded from any external repo path at runtime).
# No ${...} present, so render() must return this byte-identical via pre-scan guard.
DOCKERCONFIGJSON_PAYLOAD = (
    '{"auths": {"nvcr.io": {"username": "$oauthtoken", '
    '"password": "fake-token-not-real-credentials", '
    '"auth": "JG9hdXRodG9rZW46ZmFrZS10b2tlbi1ub3QtcmVhbC1jcmVkZW50aWFscw=="}}}'
)

# Smaller JSON literal with ${namespace} for D-12 part 2 substitution case.
SMALL_JSON_WITH_NAMESPACE = '{"namespace": "${namespace}", "key": "v"}'
SMALL_JSON_RESOLVED = '{"namespace": "aidp-prod", "key": "v"}'


# --- Tests --------------------------------------------------------------------

def test_render_returns_unchanged_when_no_braces() -> None:
    """render() short-circuits on bare-shell-style strings via pre-scan guard."""
    from main import render

    assert render("$CRDS && $CRD", {}) == "$CRDS && $CRD"


def test_render_cluster_init_unchanged() -> None:
    """Existing cluster_init/app-store-cluster-init.yaml is byte-identical pre-scan-guarded."""
    from main import render

    fixture_path = (
        Path(__file__).resolve().parents[2] / "cluster_init" / "app-store-cluster-init.yaml"
    )
    content = fixture_path.read_text(encoding="utf-8")

    assert render(content, {}) == content
    assert render(content, {"namespace": "default"}) == content


def test_render_happy_path() -> None:
    """render('hello ${NAME}', {'NAME': 'world'}) returns 'hello world'."""
    from main import render

    assert render("hello ${NAME}", {"NAME": "world"}) == "hello world"


def test_render_double_dollar_escape() -> None:
    """render('price is $$5', {'x': 'y'}) returns 'price is $5' (OP-04 $$ escape)."""
    from main import render

    assert render("price is $$5", {"x": "y"}) == "price is $5"


def test_render_dockerconfigjson_unchanged_when_no_braces() -> None:
    """JSON-safety: dockerconfigjson with no ${...} is byte-identical (D-12 part 1)."""
    from main import render

    # Plain (no variables) — empty-vars guard fires first.
    assert render(DOCKERCONFIGJSON_PAYLOAD, {}) == DOCKERCONFIGJSON_PAYLOAD
    # With variables but still no ${...} — pre-scan guard fires.
    assert (
        render(DOCKERCONFIGJSON_PAYLOAD, {"namespace": "aidp-prod"})
        == DOCKERCONFIGJSON_PAYLOAD
    )


def test_render_substitutes_namespace_in_small_json() -> None:
    """JSON-safety: smaller JSON literal with ${namespace} substitutes correctly (D-12 part 2)."""
    from main import render

    assert (
        render(SMALL_JSON_WITH_NAMESPACE, {"namespace": "aidp-prod"})
        == SMALL_JSON_RESOLVED
    )


def test_render_undefined_variable_raises_value_error() -> None:
    """render() with ${UNDEF} raises ValueError naming the variable (D-04, D-05)."""
    from main import render

    with pytest.raises(ValueError) as exc_info:
        render("value: ${UNDEF}", {"x": "y"})
    assert "UNDEF" in str(exc_info.value)


def test_render_malformed_empty_placeholder_raises_value_error() -> None:
    """render('bad: ${}', non-empty vars) raises ValueError with 'Malformed' in message (D-04, D-05)."""
    from main import render

    # NOTE: must pass non-empty variables — otherwise the empty-vars guard
    # short-circuits BEFORE Template.substitute() can detect the malformed placeholder.
    with pytest.raises(ValueError) as exc_info:
        render("bad: ${}", {"x": "y"})
    assert "Malformed" in str(exc_info.value)


def test_render_malformed_numeric_placeholder_raises_value_error() -> None:
    """render('bad: ${123}') raises ValueError with 'Malformed' in message (D-04, D-05)."""
    from main import render

    with pytest.raises(ValueError) as exc_info:
        render("bad: ${123}", {"x": "y"})
    assert "Malformed" in str(exc_info.value)


def test_render_no_op_when_variables_none() -> None:
    """render('no-tokens', None) returns 'no-tokens' (OP-05; empty-vars guard)."""
    from main import render

    assert render("no-tokens", None) == "no-tokens"


def test_render_no_op_when_variables_empty() -> None:
    """render('no-tokens', {}) returns 'no-tokens' (OP-05; empty-vars guard)."""
    from main import render

    assert render("no-tokens", {}) == "no-tokens"


def test_render_multi_occurrence() -> None:
    """render('${a} and ${a}', {'a': 'x'}) returns 'x and x' (D-13 multi-occurrence)."""
    from main import render

    assert render("${a} and ${a}", {"a": "x"}) == "x and x"
