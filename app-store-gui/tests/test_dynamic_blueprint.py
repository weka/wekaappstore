from __future__ import annotations

import os
import shutil
import tempfile

os.environ.setdefault("BLUEPRINTS_DIR", "/tmp")

import webapp.main as main


# ---------------------------------------------------------------------------
# parse_x_variables tests
# ---------------------------------------------------------------------------

def test_parse_x_variables_empty_string():
    """Test 1: empty string returns {}."""
    assert main.parse_x_variables("") == {}


def test_parse_x_variables_no_x_variables_key():
    """Test 2: YAML without x-variables key returns {}."""
    yaml_text = "apiVersion: warp.io/v1alpha1\nkind: WekaAppStore\n"
    assert main.parse_x_variables(yaml_text) == {}


def test_parse_x_variables_with_string_var():
    """Test 3: YAML with x-variables returns the mapped dict."""
    yaml_text = (
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
        '    description: "Target namespace"\n'
    )
    result = main.parse_x_variables(yaml_text)
    assert result == {
        "namespace": {
            "type": "string",
            "required": True,
            "description": "Target namespace",
        }
    }


def test_parse_x_variables_only_x_variables_returned():
    """Test 4: only the x-variables dict is returned, not apiVersion etc."""
    yaml_text = (
        "apiVersion: warp.io/v1alpha1\n"
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
    )
    result = main.parse_x_variables(yaml_text)
    assert "apiVersion" not in result
    assert "namespace" in result


def test_parse_x_variables_parse_error_returns_empty():
    """Test 5: YAML parse error returns {} without raising."""
    bad_yaml = "x-variables: {\ninvalid: [unclosed"
    result = main.parse_x_variables(bad_yaml)
    assert result == {}


def test_find_blueprint_finds_matching_yaml(tmp_path):
    """Test 6: find_blueprint returns absolute path when matching file exists."""
    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    yaml_file = app_dir / "my-app.yaml"
    yaml_file.write_text(
        "x-variables:\n"
        "  namespace:\n"
        "    type: string\n"
        "    required: true\n"
    )
    result = main.find_blueprint("my-app", blueprints_dir=str(tmp_path))
    assert result == str(yaml_file)


def test_find_blueprint_returns_none_when_not_found(tmp_path):
    """Test 7: find_blueprint returns None when no matching file exists."""
    result = main.find_blueprint("unknown", blueprints_dir=str(tmp_path))
    assert result is None


def test_find_blueprint_cluster_init_special_case():
    """Test 8: cluster-init is special-cased and returns the cluster-init YAML path."""
    result = main.find_blueprint("cluster-init", blueprints_dir="/some/dir")
    expected = os.path.join("/some/dir", "cluster_init", "app-store-cluster-init.yaml")
    assert result == expected


def test_find_blueprint_ignores_files_without_x_variables(tmp_path):
    """Test 9: find_blueprint ignores YAML files without x-variables key."""
    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    yaml_file = app_dir / "my-app.yaml"
    yaml_file.write_text("apiVersion: warp.io/v1alpha1\nkind: WekaAppStore\n")
    result = main.find_blueprint("my-app", blueprints_dir=str(tmp_path))
    assert result is None


def test_parse_x_variables_credential_type():
    """Test 10: parse_x_variables returns credential_type field for credential vars."""
    yaml_text = (
        "x-variables:\n"
        "  hf_cred:\n"
        "    type: credential\n"
        "    credential_type: huggingface\n"
        "    required: true\n"
    )
    result = main.parse_x_variables(yaml_text)
    assert result == {
        "hf_cred": {
            "type": "credential",
            "credential_type": "huggingface",
            "required": True,
        }
    }
