"""Shared pytest setup for operator_module tests.

Adds operator_module/ to sys.path so tests can `from main import render`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- sys.path setup -----------------------------------------------------------
OPERATOR_MODULE_ROOT = Path(__file__).resolve().parents[1]

if str(OPERATOR_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_MODULE_ROOT))
