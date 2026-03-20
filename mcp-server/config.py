"""Environment variable configuration for the MCP server.

All configuration is read at module import time from environment variables.
No classes needed — simple module-level constants.
"""
from __future__ import annotations

import os

# Directory where WekaAppStore blueprint YAML manifests are mounted.
# Must match the path used by the git-sync sidecar in production.
BLUEPRINTS_DIR: str = os.environ.get("BLUEPRINTS_DIR", "/app/manifests/manifest")

# Kubernetes authentication mode.
# 'in-cluster': use ServiceAccount token (production)
# 'kubeconfig': use ~/.kube/config (local development)
KUBERNETES_AUTH_MODE: str = os.environ.get("KUBERNETES_AUTH_MODE", "in-cluster")

# Logging level — passed to logging.basicConfig in server.py.
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
