"""Shared pytest fixtures for MCP server tests.

Adds mcp-server/ and app-store-gui/ to sys.path so imports resolve, then
provides mocked K8s API fixtures and realistic sample inspection snapshots.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# --- sys.path setup -----------------------------------------------------------
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
APP_STORE_GUI_ROOT = MCP_SERVER_ROOT.parent / "app-store-gui"

for path in (str(MCP_SERVER_ROOT), str(APP_STORE_GUI_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


# --- Mocked K8s API fixtures --------------------------------------------------

@pytest.fixture
def mock_core_api() -> MagicMock:
    """MagicMock of kubernetes.client.CoreV1Api."""
    from kubernetes import client
    mock = MagicMock(spec=client.CoreV1Api)
    return mock


@pytest.fixture
def mock_storage_api() -> MagicMock:
    """MagicMock of kubernetes.client.StorageV1Api."""
    from kubernetes import client
    mock = MagicMock(spec=client.StorageV1Api)
    return mock


@pytest.fixture
def mock_custom_objects_api() -> MagicMock:
    """MagicMock of kubernetes.client.CustomObjectsApi."""
    from kubernetes import client
    mock = MagicMock(spec=client.CustomObjectsApi)
    return mock


@pytest.fixture
def mock_apps_api() -> MagicMock:
    """MagicMock of kubernetes.client.AppsV1Api."""
    from kubernetes import client
    mock = MagicMock(spec=client.AppsV1Api)
    return mock


@pytest.fixture
def mock_apiextensions_api() -> MagicMock:
    """MagicMock of kubernetes.client.ApiextensionsV1Api."""
    from kubernetes import client
    mock = MagicMock(spec=client.ApiextensionsV1Api)
    return mock


@pytest.fixture
def mock_version_api() -> MagicMock:
    """MagicMock of kubernetes.client.VersionApi."""
    from kubernetes import client
    mock = MagicMock(spec=client.VersionApi)
    return mock


# --- Realistic snapshot fixtures ----------------------------------------------

@pytest.fixture
def sample_cluster_snapshot() -> dict[str, Any]:
    """Realistic nested dict matching collect_cluster_inspection() output shape.

    Mirrors topology from app-store-gui/tests/conftest.py mocked_cluster_inspection_topology:
    - 1 CPU node (cpu-node-1): 16 cores, 128 GiB
    - 1 GPU node (gpu-node-1): 32 cores, 256 GiB, 4x NVIDIA L40 48 GiB
    """
    return {
        "captured_at": "2026-03-20T00:00:00Z",
        "k8s_version": "v1.30.1",
        "gpu_operator_installed": True,
        "app_store_crd_installed": True,
        "app_store_cluster_init_present": True,
        "app_store_crs": ["default/app-store-cluster-init"],
        "default_storage_class": "wekafs",
        "default_storage_class_details": {
            "name": "wekafs",
            "provisioner": "csi.weka.io",
            "parameters": {"filesystemName": "weka-home"},
            "reclaimPolicy": "Delete",
            "volumeBindingMode": "Immediate",
            "allowVolumeExpansion": True,
            "annotations": {"storageclass.kubernetes.io/is-default-class": "true"},
        },
        "domains": {
            "cpu": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 2,
                    "cpu_nodes": 1,
                    "gpu_nodes": 1,
                    "allocatable_cores": 48.0,
                    "used_cores": 12.0,
                    "free_cores": 36.0,
                },
                "notes": [],
                "blockers": [],
            },
            "memory": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 2,
                    "allocatable_gib": 384.0,
                    "used_gib": 80.0,
                    "free_gib": 304.0,
                },
                "notes": [],
                "blockers": [],
            },
            "gpu": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "ready_nodes": 2,
                    "gpu_nodes": 1,
                    "total_devices": 4,
                    "used_devices": 2,
                    "free_devices": 2,
                    "inventory": [
                        {"model": "NVIDIA L40", "count": 4, "memory_gib": 48.0},
                    ],
                },
                "notes": [],
                "blockers": [],
            },
            "namespaces": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {"names": ["ai-platform", "default", "weka"]},
                "notes": [],
                "blockers": [],
            },
            "storage_classes": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {"names": ["gp3", "wekafs"]},
                "notes": [],
                "blockers": [],
            },
        },
    }


@pytest.fixture
def sample_weka_snapshot() -> dict[str, Any]:
    """Realistic nested dict matching collect_weka_inspection() output shape.

    Based on mocked_weka_cluster_payload in app-store-gui/tests/conftest.py.
    Cluster 'weka-prod':
      - total capacity: 2 TiB (2199023255552 bytes)
      - free capacity:  ~1.5 TiB (1649267441664 bytes)
      - filesystem 'weka-home': 1 TiB total, ~768 GiB free
    """
    return {
        "captured_at": "2026-03-20T00:00:00Z",
        "domains": {
            "weka": {
                "status": "complete",
                "required": True,
                "freshness": {"captured_at": "2026-03-20T00:00:00Z", "max_age_seconds": 300},
                "observed": {
                    "clusters": [
                        {
                            "name": "weka-prod",
                            "namespace": "weka",
                            "status": "Ready",
                            "cluster_total_bytes": 2199023255552,
                            "cluster_free_bytes": 1649267441664,
                            "filesystem_capacity": None,
                        }
                    ],
                    "filesystems": [
                        {
                            "name": "weka-home",
                            "total_bytes": 1099511627776,
                            "free_bytes": 824633720832,
                        }
                    ],
                    "cluster_total_bytes": 2199023255552,
                    "cluster_free_bytes": 1649267441664,
                },
                "notes": [],
                "blockers": [],
            }
        },
    }
