"""Environment variable configuration for the MCP server.

All configuration is read at module import time from environment variables.
No classes needed — simple module-level constants.
"""
from __future__ import annotations

import os
import sys

# Directory where WekaAppStore blueprint YAML manifests are mounted.
# Must match the path used by the git-sync sidecar in production.
# Required: must be explicitly set via env var at runtime. Call validate_required()
# at server startup to enforce this.
BLUEPRINTS_DIR: str = os.environ.get("BLUEPRINTS_DIR", "")

# Kubernetes authentication mode.
# 'in-cluster': use ServiceAccount token (production)
# 'kubeconfig': use ~/.kube/config (local development)
KUBERNETES_AUTH_MODE: str = os.environ.get("KUBERNETES_AUTH_MODE", "in-cluster")

# Logging level — passed to logging.basicConfig in server.py.
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# Optional: WEKA cluster REST endpoint for weka-specific inspection tools.
# None if unset — tools will handle this lazily when called.
WEKA_ENDPOINT: str | None = os.environ.get("WEKA_ENDPOINT") or None

# Optional: path to a kubeconfig file for local development overrides.
# None if unset — kubernetes client falls back to in-cluster or default kubeconfig.
KUBECONFIG: str | None = os.environ.get("KUBECONFIG") or None

# Transport mode — 'stdio' (default, CI-safe) or 'http' (sidecar deployment)
MCP_TRANSPORT: str = os.environ.get("MCP_TRANSPORT", "stdio")

# HTTP listening port — only relevant when MCP_TRANSPORT=http
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8080"))


def validate_required() -> None:
    """Validate that all required env vars are set.

    Call this from the server __main__ block (not at import time) so that tests
    can import config freely without triggering a SystemExit.

    Exits with code 1 and a FATAL message if any required var is missing.
    """
    if not os.environ.get("BLUEPRINTS_DIR"):
        print(
            "FATAL: BLUEPRINTS_DIR env var is required but not set. "
            "Set BLUEPRINTS_DIR to the directory containing blueprint YAML manifests.",
            file=sys.stderr,
        )
        sys.exit(1)
