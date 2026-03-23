"""Tests for mcp-server/config.py startup validation and env var configuration."""
from __future__ import annotations

import sys
import importlib

import pytest


def _reload_config():
    """Force a fresh import of config to pick up monkeypatched env vars."""
    if "config" in sys.modules:
        del sys.modules["config"]
    import config  # noqa: PLC0415
    return config


def test_blueprints_dir_required(monkeypatch):
    """validate_required() raises SystemExit when BLUEPRINTS_DIR is not set."""
    monkeypatch.delenv("BLUEPRINTS_DIR", raising=False)
    cfg = _reload_config()
    with pytest.raises(SystemExit) as exc_info:
        cfg.validate_required()
    assert exc_info.value.code == 1


def test_blueprints_dir_set_via_env(monkeypatch, tmp_path):
    """config.BLUEPRINTS_DIR returns the value from the env var when set."""
    expected = str(tmp_path)
    monkeypatch.setenv("BLUEPRINTS_DIR", expected)
    cfg = _reload_config()
    assert cfg.BLUEPRINTS_DIR == expected


def test_weka_endpoint_none_when_unset(monkeypatch):
    """config.WEKA_ENDPOINT is None when WEKA_ENDPOINT env var is not set."""
    monkeypatch.delenv("WEKA_ENDPOINT", raising=False)
    cfg = _reload_config()
    assert cfg.WEKA_ENDPOINT is None


def test_weka_endpoint_value_when_set(monkeypatch):
    """config.WEKA_ENDPOINT returns the env var value when set."""
    monkeypatch.setenv("WEKA_ENDPOINT", "https://weka.example.com")
    cfg = _reload_config()
    assert cfg.WEKA_ENDPOINT == "https://weka.example.com"


def test_kubeconfig_none_when_unset(monkeypatch):
    """config.KUBECONFIG is None when KUBECONFIG env var is not set."""
    monkeypatch.delenv("KUBECONFIG", raising=False)
    cfg = _reload_config()
    assert cfg.KUBECONFIG is None


def test_kubeconfig_value_when_set(monkeypatch, tmp_path):
    """config.KUBECONFIG returns the env var value when set."""
    kube_path = str(tmp_path / "config")
    monkeypatch.setenv("KUBECONFIG", kube_path)
    cfg = _reload_config()
    assert cfg.KUBECONFIG == kube_path


def test_validate_required_passes_when_blueprints_dir_set(monkeypatch, tmp_path):
    """validate_required() does not raise when BLUEPRINTS_DIR is set."""
    monkeypatch.setenv("BLUEPRINTS_DIR", str(tmp_path))
    cfg = _reload_config()
    # Should not raise
    cfg.validate_required()


# --- MCP_TRANSPORT tests ---


def test_mcp_transport_default(monkeypatch):
    """config.MCP_TRANSPORT defaults to 'stdio' when env var is not set."""
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    cfg = _reload_config()
    assert cfg.MCP_TRANSPORT == "stdio"


def test_mcp_transport_http(monkeypatch):
    """config.MCP_TRANSPORT reads 'http' when env var is set to 'http'."""
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    cfg = _reload_config()
    assert cfg.MCP_TRANSPORT == "http"


# --- MCP_PORT tests ---


def test_mcp_port_default(monkeypatch):
    """config.MCP_PORT defaults to 8080 (int) when env var is not set."""
    monkeypatch.delenv("MCP_PORT", raising=False)
    cfg = _reload_config()
    assert cfg.MCP_PORT == 8080


def test_mcp_port_custom(monkeypatch):
    """config.MCP_PORT reads custom value as int when env var is set."""
    monkeypatch.setenv("MCP_PORT", "9090")
    cfg = _reload_config()
    assert cfg.MCP_PORT == 9090


def test_mcp_port_is_int(monkeypatch):
    """config.MCP_PORT is always int type, not str."""
    monkeypatch.delenv("MCP_PORT", raising=False)
    cfg = _reload_config()
    assert isinstance(cfg.MCP_PORT, int)
