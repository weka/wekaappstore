from __future__ import annotations

import webapp.main as main


def test_url_field_detected_by_name():
    assert main._is_url_field("keycloak_url", {"type": "string"}) is True
    assert main._is_url_field("api_endpoint", {}) is True
    assert main._is_url_field("s3_host", {}) is True
    assert main._is_url_field("weka_s3_access_key", {}) is False
    assert main._is_url_field("namespace", {}) is False


def test_url_field_detected_by_type_or_format():
    assert main._is_url_field("anything", {"type": "url"}) is True
    assert main._is_url_field("anything", {"format": "url"}) is True


def test_url_with_space_is_rejected():
    # The exact bug: "http://key cloak.example.com" — space between key and cloak.
    err = main._validate_variable_value("keycloak_url", {}, "http://key cloak.example.com")
    assert err is not None
    assert "space" in err.lower()


def test_url_without_scheme_is_rejected():
    err = main._validate_variable_value("keycloak_url", {}, "keycloak.example.com")
    assert err is not None
    assert "valid url" in err.lower()


def test_valid_url_passes():
    assert main._validate_variable_value("keycloak_url", {}, "https://keycloak.example.com") is None
    assert main._validate_variable_value("keycloak_url", {}, "http://10.0.0.5:8080/auth") is None


def test_empty_value_passes_format_check():
    # Required-ness is enforced separately; empty is not a format error.
    assert main._validate_variable_value("keycloak_url", {}, "") is None
    assert main._validate_variable_value("keycloak_url", {}, "   ") is None


def test_non_url_field_allows_spaces():
    # Non-URL fields are not whitespace-checked (may legitimately contain spaces).
    assert main._validate_variable_value("description", {}, "hello world") is None
