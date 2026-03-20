"""Integration tests for the mock agent harness.

Calls the 3 harness scenario functions and asserts expected outcomes.
No K8s network calls are made — all dependencies are fully mocked.
"""
from __future__ import annotations


def test_harness_happy_path():
    """Happy path: inspect -> list -> get -> validate -> apply -> status all succeed."""
    from harness.mock_agent import build_mock_k8s_deps, build_mock_inspection_deps, run_happy_path

    mock_apply_deps, mock_apply_ops = build_mock_k8s_deps(apply_should_succeed=True)
    mock_inspection_deps = build_mock_inspection_deps()
    mock_status_api = mock_apply_deps.custom_objects_api_factory(None)

    results = run_happy_path(mock_inspection_deps, mock_apply_deps, mock_apply_ops)

    # validate_yaml step must report valid=true
    assert results["validate"]["valid"] is True, (
        f"Expected validate valid=true, got: {results['validate']}"
    )
    # apply step must report applied=true
    assert results["apply"]["applied"] is True, (
        f"Expected apply applied=true, got: {results['apply']}"
    )
    # No exceptions during full chain
    assert "error" not in results or results.get("error") is None


def test_harness_approval_bypass():
    """Approval bypass: apply with confirmed=False returns error, no CR created."""
    from harness.mock_agent import build_mock_k8s_deps, run_approval_bypass

    mock_apply_deps, mock_apply_ops = build_mock_k8s_deps()

    result = run_approval_bypass(mock_apply_deps, mock_apply_ops)

    assert result["applied"] is False, f"Expected applied=false, got: {result}"
    assert result["error"] == "approval_required", (
        f"Expected error='approval_required', got: {result}"
    )
    # No CR was created
    assert not any(op[0] == "create" for op in mock_apply_ops), (
        f"Expected no create ops, got: {mock_apply_ops}"
    )


def test_harness_validation_failure():
    """Validation failure: v1.0 YAML rejected before apply is called."""
    from harness.mock_agent import run_validation_failure

    result = run_validation_failure()

    assert result["valid"] is False, f"Expected valid=false, got: {result}"
    error_codes = [e["code"] for e in result.get("errors", [])]
    assert "v1_only_field" in error_codes, (
        f"Expected 'v1_only_field' error code, got errors: {result.get('errors')}"
    )
