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


def test_render_double_dollar_preserved() -> None:
    """$$ is the shell PID and is NOT collapsed to $ (supersedes OP-04 $$ escape).
    Preserved verbatim so embedded bash scripts keep working."""
    from main import render

    assert render("price is $$5", {"x": "y"}) == "price is $$5"
    assert render("pid is $$", {"namespace": "rag"}) == "pid is $$"


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


def test_render_unknown_braced_token_left_untouched() -> None:
    """${UNDEF} (not an allowlisted name) is preserved verbatim and does NOT raise —
    it is indistinguishable from a legitimate shell ${VAR}. (Supersedes D-04/D-05;
    undefined-variable detection now lives at the variable-resolution layer.)"""
    from main import render

    assert render("value: ${UNDEF}", {"x": "y"}) == "value: ${UNDEF}"


def test_render_empty_placeholder_left_untouched() -> None:
    """${} is foreign shell-ish content: preserved verbatim, no exception."""
    from main import render

    assert render("bad: ${}", {"x": "y"}) == "bad: ${}"


def test_render_numeric_placeholder_left_untouched() -> None:
    """${123} is not an allowlisted name: preserved verbatim, no exception."""
    from main import render

    assert render("bad: ${123}", {"x": "y"}) == "bad: ${123}"


def test_render_shell_manifest_only_substitutes_known_vars() -> None:
    """Regression for the AIDP outage: a Job manifest full of shell $-syntax must
    pass through untouched EXCEPT for the allowlisted ${namespace}. The previous
    strict string.Template raised 'Invalid placeholder' on the first $( / ${ / $$
    token (46 such tokens across the AIDP appstack), breaking every component."""
    from main import render

    manifest = (
        "    set -euo pipefail\n"
        '    KC_HOST="http://keycloak-http.${namespace}.svc.cluster.local"\n'
        '    TOKEN=$(curl -sf "$KC_HOST/realms/master" | jq -r .access_token)\n'
        '    for i in $(seq 1 30); do echo "$i"; done\n'
        '    local CLIENT_ID="$1"\n'
        "    dollar='$'\n"
        "    # the pattern *'${'* is a guard, not a placeholder\n"
        '    if [[ -z "${TOKEN}" ]]; then echo "pid=$$"; fi\n'
    )
    out = render(manifest, {"namespace": "rag"})

    # Only ${namespace} is substituted.
    assert "keycloak-http.rag.svc.cluster.local" in out
    assert "${namespace}" not in out
    # Every shell $-form is preserved byte-for-byte (no substitution, no raise).
    assert "$(curl" in out
    assert "$(seq 1 30)" in out
    assert '"$1"' in out
    assert "dollar='$'" in out
    assert "*'${'*" in out
    assert "${TOKEN}" in out  # unknown braced token left as shell content
    assert "pid=$$" in out    # $$ (shell PID) not collapsed


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
