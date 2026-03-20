from __future__ import annotations

from webapp.planning.family_matcher import SupportedFamilyMatcher


def test_supported_family_matcher_maps_openfold_requests_deterministically() -> None:
    matcher = SupportedFamilyMatcher()

    result = matcher.match("Deploy an OpenFold protein folding workflow with a WEKA filesystem.")

    assert result.status == "matched"
    assert result.family == "openfold"
    assert "openfold" in result.matched_terms
    assert "protein folding" in result.matched_terms


def test_supported_family_matcher_reports_explicit_no_fit() -> None:
    matcher = SupportedFamilyMatcher()

    result = matcher.match("Install a generic PostgreSQL database with no AI stack.")

    assert result.status == "no_supported_family"
    assert result.family is None
    assert "Supported families:" in result.reason
