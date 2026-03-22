"""Tests to verify logging goes to stderr only and stdout is clean."""
from __future__ import annotations

import io
import logging
import sys
import subprocess
import os


def test_no_stdout_on_import():
    """Importing server.py produces zero bytes on stdout."""
    mcp_server_dir = os.path.join(os.path.dirname(__file__), "..")
    app_store_gui_dir = os.path.join(mcp_server_dir, "..", "app-store-gui")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.abspath(mcp_server_dir)}:{os.path.abspath(app_store_gui_dir)}"

    result = subprocess.run(
        [sys.executable, "-c", "import server"],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(mcp_server_dir),
        env=env,
    )
    assert result.returncode == 0, f"server import failed: {result.stderr}"
    assert result.stdout == "", f"Expected no stdout on import, got: {result.stdout!r}"


def test_log_level_env_var():
    """LOG_LEVEL env var is read by config.py and reflects the expected value."""
    import importlib
    import unittest.mock

    with unittest.mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
        import config as cfg
        importlib.reload(cfg)
        assert cfg.LOG_LEVEL == "DEBUG", (
            f"Expected config.LOG_LEVEL == 'DEBUG', got {cfg.LOG_LEVEL!r}"
        )


def test_logging_goes_to_stderr():
    """Logging calls after server import write to stderr, not stdout."""
    mcp_server_dir = os.path.join(os.path.dirname(__file__), "..")
    app_store_gui_dir = os.path.join(mcp_server_dir, "..", "app-store-gui")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.abspath(mcp_server_dir)}:{os.path.abspath(app_store_gui_dir)}"

    script = (
        "import server, logging; "
        "logging.getLogger('test').info('sentinel-message')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.path.abspath(mcp_server_dir),
        env=env,
    )
    assert result.returncode == 0, f"script failed: {result.stderr}"
    assert result.stdout == "", f"Expected no stdout, got: {result.stdout!r}"
    assert "sentinel-message" in result.stderr, (
        f"Expected 'sentinel-message' in stderr, got stderr: {result.stderr!r}"
    )
