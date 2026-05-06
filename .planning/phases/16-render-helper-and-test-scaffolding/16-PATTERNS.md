# Phase 16: render() Helper and Test Scaffolding - Pattern Map

**Mapped:** 2026-05-06
**Files analyzed:** 5 (1 modified, 4 new)
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `operator_module/main.py` (ADD `render()`) | utility (top-level helper) | transform (str → str) | `operator_module/main.py` `_deep_merge`, `merge_values` (lines 231-249) | exact (same file, same role) |
| `operator_module/tests/__init__.py` | test (package marker) | n/a | `mcp-server/tests/__init__.py` | exact (empty file) |
| `operator_module/tests/conftest.py` | test (fixture/path setup) | n/a | `mcp-server/tests/conftest.py` (lines 1-22) | exact (sys.path injection) |
| `operator_module/tests/test_render.py` | test (pytest module) | request-response (call → assert) | `mcp-server/tests/test_validate_yaml.py` | exact (pure-function unit tests, ValueError-raising helper) |
| `operator_module/requirements-dev.txt` | config (dependency manifest) | n/a | `mcp-server/requirements.txt` line 4 | role-match (split-deps convention) |

## Pattern Assignments

### `operator_module/main.py` — ADD top-level `render(text, variables) -> str`

**Analog:** `operator_module/main.py` itself — peer helpers `_deep_merge`, `merge_values`, `discover_chart_crds`, `load_values_from_reference`. The new helper goes alongside these existing top-level utilities.

**Imports pattern** (`operator_module/main.py` lines 1-11) — already in file; only `import string` is new:
```python
import logging
import kopf
import kr8s
import subprocess
import yaml
import tempfile
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import lru_cache
```

Add a single new import. Place near the others, alphabetized within the stdlib block. Recommended addition:
```python
import string
```

**Helper docstring + signature pattern** (`operator_module/main.py` lines 231-249) — use this exact style for `render()`:
```python
def merge_values(base_values: Dict[str, Any], additional_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple values dictionaries, with later values taking precedence
    """
    result = base_values.copy()
    for values in additional_values:
        result = _deep_merge(result, values)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

Conventions to copy:
- `Dict[str, Any]` / `Optional[...]` typing from the existing `from typing import` line.
- Triple-quoted docstring; one-liner is acceptable (e.g., `_deep_merge`), multi-line works too (e.g., `merge_values`, `load_values_from_reference`). Per CONTEXT.md D (Claude's Discretion), Phase 16 prefers a multi-line docstring documenting the pre-scan guard, `$$` escape, and ValueError contract.
- Plain top-level `def` — no decorators (the `@lru_cache` decorators on `_load_kube_config_once` and `discover_chart_crds` are domain-specific; render() is pure and small enough that caching is unnecessary).

**Placement guidance** — drop `render()` immediately after `_deep_merge` (line 249) and before the `# ===================== CRD Strategy Helpers =====================` section header on line 252. This sits in the "small generic top-level helpers" cluster, away from the kube-config / CRD strategy block. Per CONTEXT.md D-Discretion, the planner has latitude here, but this is the cleanest grouping.

**Signature** (locked by CONTEXT.md "Specifics"):
```python
def render(text: str, variables: Optional[Dict[str, str]]) -> str:
```

**No call-site changes** — `handle_appstack_deployment` (line 551) and `load_values_from_reference` (line 352) MUST NOT be modified in Phase 16. They are Phase 18's responsibility.

---

### `operator_module/tests/__init__.py`

**Analog:** `mcp-server/tests/__init__.py` (1 line, 0 bytes — empty file).

```
(empty file — 0 bytes, package marker only)
```

**Pattern to copy:** The file exists purely to mark `tests/` as an importable Python package so pytest's discovery and conftest loading behave consistently. No content. Match the existing convention exactly — do not add a docstring or shebang.

---

### `operator_module/tests/conftest.py`

**Analog:** `mcp-server/tests/conftest.py` (lines 1-22) — direct template. Only the parent path comment differs.

**Module docstring + sys.path injection pattern** (lines 1-22):
```python
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
```

**Adaptation for operator_module:** Operator tests only need `operator_module/` on the path (no peer like app-store-gui to dual-import). Drop the `Any`, `MagicMock`, `pytest` imports — none are referenced in the conftest itself (Phase 16 conftest carries no fixtures, only path setup; fixtures are local to test_render.py if needed). Concrete adapted shape (planner can lock this verbatim):

```python
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
```

Conventions copied:
- `from __future__ import annotations` first (matches mcp-server convention).
- `Path(__file__).resolve().parents[1]` — same idiom as the analog (CONTEXT.md D-Discretion notes either string-based or pathlib is acceptable; pathlib matches the analog).
- Single-element insert guarded by `if str(...) not in sys.path` (mirrors the loop body in the analog).
- `# --- ... -------` block comment style for section headers.

---

### `operator_module/tests/test_render.py`

**Analog:** `mcp-server/tests/test_validate_yaml.py` — pure-function unit tests against a stdlib-only helper, including a fallback inline sys.path guard, module-level YAML/text fixture constants, and pytest plain function tests (no fixtures, no classes). Secondary analog: `mcp-server/tests/test_config.py` lines 18-24 for the `pytest.raises(...)` idiom.

**Imports + sys.path fallback pattern** (`test_validate_yaml.py` lines 1-15):
```python
"""Unit tests for validate_yaml tool.

Tests call _validate_yaml_impl() directly — no FastMCP needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- sys.path setup ---
MCP_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(MCP_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER_ROOT))
```

Conventions to copy in test_render.py:
- One-liner module docstring naming the unit under test.
- `from __future__ import annotations`.
- `import pytest` after stdlib block, blank line separator (PEP-8).
- Inline sys.path guard repeated even when `conftest.py` does the same — defense-in-depth so the test file works when run directly via `python -m pytest path/to/test_render.py`. (`test_validate_yaml.py` uses this pattern despite `conftest.py` also setting the path.)
- Defer the `from main import render` import to inside each test function (matches `from tools.validate_yaml import _validate_yaml_impl` on lines 129, 139, etc. of `test_validate_yaml.py`) — this keeps `import main` from running operator-side `kopf` registration at test collection time.

**Module-level fixture constant pattern** (`test_validate_yaml.py` lines 22-49) — for the inline JSON-safety string constant called out in CONTEXT.md D-12:
```python
VALID_HELMCHART_YAML = """\
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: my-app
spec:
  helmChart:
    repository: https://charts.example.com
    name: my-chart
    version: "1.0.0"
    releaseName: my-release
"""
```

Pattern: triple-quoted string literal at module scope, escaped first newline (`"""\`), UPPER_SNAKE constant name. Apply this pattern to define an inline `DOCKERCONFIGJSON_PAYLOAD` constant for the JSON-safety test (per CONTEXT.md D-12, NOT loaded from `/Users/christopherjenkins/git/aidp/...`).

**Plain-function test pattern with single-line docstring** (`test_validate_yaml.py` lines 127-145):
```python
def test_valid_yaml_passes() -> None:
    """Valid WekaAppStore YAML with helmChart returns valid=True and empty errors."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(VALID_HELMCHART_YAML)

    assert result["valid"] is True
    assert result["errors"] == []


def test_valid_yaml_appstack() -> None:
    """Valid WekaAppStore YAML with appStack returns valid=True."""
    from tools.validate_yaml import _validate_yaml_impl

    result = _validate_yaml_impl(VALID_APPSTACK_YAML)

    assert result["valid"] is True
    assert result["errors"] == []
```

Conventions to copy:
- `def test_<scenario>() -> None:` — explicit `-> None` return annotation.
- One-line docstring stating Given/When/Then in a single sentence (no separate Arrange/Act/Assert sections).
- Per-function deferred import.
- Blank line between Act and Assert (matches `result = ...` then blank then `assert ...` shape).
- `assert` statements directly with the `is True` / `== []` style (no `assertTrue` / `unittest`).

**`pytest.raises` pattern** (`mcp-server/tests/test_config.py` lines 18-24) — for the undefined-variable and malformed-placeholder tests:
```python
def test_blueprints_dir_required(monkeypatch):
    """validate_required() raises SystemExit when BLUEPRINTS_DIR is not set."""
    monkeypatch.delenv("BLUEPRINTS_DIR", raising=False)
    cfg = _reload_config()
    with pytest.raises(SystemExit) as exc_info:
        cfg.validate_required()
    assert exc_info.value.code == 1
```

Apply identically for the render() error tests:
```python
def test_render_undefined_variable_raises_value_error() -> None:
    """render() with ${UNDEF} raises ValueError naming the variable."""
    from main import render

    with pytest.raises(ValueError) as exc_info:
        render("value: ${UNDEF}", {"x": "y"})
    assert "${UNDEF}" in str(exc_info.value)
```
The `as exc_info` capture + post-`with` `assert "..." in str(exc_info.value)` shape matches CONTEXT.md "Specifics" section's locked error-message format `f"Undefined variable: ${{{name}}}"`.

**Cluster-init backward-compat fixture loading pattern** (CONTEXT.md "Specifics") — load from disk via pathlib so the test catches drift if the bootstrap manifest changes:
```python
def test_render_cluster_init_unchanged() -> None:
    """Existing cluster_init shell-script Job is byte-identical pre-scan-guarded."""
    from main import render

    fixture_path = Path(__file__).resolve().parents[2] / "cluster_init" / "app-store-cluster-init.yaml"
    content = fixture_path.read_text(encoding="utf-8")

    assert render(content, {}) == content
    assert render(content, {"namespace": "default"}) == content
```
Path arithmetic mirrors the conftest's `parents[1]` and the `.read_text(encoding="utf-8")` matches `tools/blueprints.py` style (used elsewhere in the codebase for fixture loads). Two assertions are explicitly required by CONTEXT.md D-11.

---

### `operator_module/requirements-dev.txt`

**Analog:** `mcp-server/requirements.txt` line 4.

**Pin format excerpt:**
```
pytest>=8.0.0
```

**Pattern to copy:** newline-terminated `<package>>=<version>` with no comments and no compound markers. Per CONTEXT.md D-07, this is a NEW dev-only file — pytest stays out of `operator_module/requirements.txt` (production image stays minimal). The full file content is one line:

```
pytest>=8.0.0
```

(Optional: add a single leading comment line if it matches local convention. `mcp-server/requirements.txt` has no header comment; `operator_module/requirements.txt` has `# WEKA App Store Operator Python Dependencies`. Match either; prefer no header to keep the dev-only file minimal.)

---

## Shared Patterns

### Module Docstring + `from __future__ import annotations`
**Source:** `mcp-server/tests/test_validate_yaml.py` line 1, `mcp-server/tests/conftest.py` line 6, `mcp-server/tests/test_blueprints.py` line 8
**Apply to:** `operator_module/tests/conftest.py` and `operator_module/tests/test_render.py`
```python
"""<one-line description of the test module>."""
from __future__ import annotations
```
Every test/conftest module in `mcp-server/tests/` opens with this two-line preamble. New operator test files must match.

### sys.path Injection via pathlib
**Source:** `mcp-server/tests/conftest.py` lines 16-21 AND `mcp-server/tests/test_validate_yaml.py` lines 13-15
**Apply to:** `operator_module/tests/conftest.py` AND `operator_module/tests/test_render.py` (defense-in-depth)
```python
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```
This idiom appears in BOTH the conftest and individual test files. The duplication is intentional: it lets test files run directly (`python -m pytest tests/test_render.py`) without depending on conftest discovery.

### Per-Function Deferred Import of System Under Test
**Source:** `mcp-server/tests/test_validate_yaml.py` lines 129, 139, 149, etc.
**Apply to:** every `test_*` function in `operator_module/tests/test_render.py`
```python
def test_X() -> None:
    """..."""
    from main import render

    result = render(...)
    assert ...
```
Importing inside the test function (rather than at module top) avoids running module-level side effects (kopf decorators in `main.py`) at pytest collection time. This pattern is consistent across `test_validate_yaml.py`, `test_blueprints.py`, and `test_config.py`.

### One-Line Docstring per Test
**Source:** `mcp-server/tests/test_validate_yaml.py` (every test), `mcp-server/tests/test_config.py` (every test)
**Apply to:** every `test_*` function in `test_render.py`
```python
def test_X() -> None:
    """<single-sentence Given/When/Then>."""
```
Pytest tests in this codebase use single-sentence docstrings, not multi-section docstrings. Match this concision.

### Per-Component requirements*.txt
**Source:** project convention — `mcp-server/requirements.txt`, `operator_module/requirements.txt`, `app-store-gui/requirements.txt`
**Apply to:** `operator_module/requirements-dev.txt`
Dependency manifests live next to the code they support, not at the project root. The new dev-only file follows this rule. No project-level `pyproject.toml`, no `setup.py`, no `pytest.ini` (per CONTEXT.md D-09 and D-10).

---

## No Analog Found

None. Every Phase 16 file maps cleanly onto an existing codebase analog.

---

## Metadata

**Analog search scope:**
- `/Users/christopherjenkins/git/wekaappstore/mcp-server/tests/` (all 14 test files + conftest)
- `/Users/christopherjenkins/git/wekaappstore/mcp-server/requirements.txt`
- `/Users/christopherjenkins/git/wekaappstore/operator_module/main.py` (1102 lines; targeted reads at top-level helper region 230-411)
- `/Users/christopherjenkins/git/wekaappstore/operator_module/__init__.py`, `mcp-server/__init__.py`, `mcp-server/tests/__init__.py` (verified empty)
- `/Users/christopherjenkins/git/wekaappstore/cluster_init/app-store-cluster-init.yaml` lines 125-164 (regression fixture region)

**Files scanned:** 9 (5 test files read in detail, 3 `__init__.py` size-verified, 1 main.py targeted reads, 1 cluster_init fixture)

**Pattern extraction date:** 2026-05-06

**Key pattern density:**
- All test imports of system-under-test deferred to inside test bodies (universal in mcp-server/tests/)
- All test files start with `from __future__ import annotations` (universal)
- All test files use single-line docstrings per test function (universal)
- pathlib `Path(__file__).resolve().parents[N]` is the only sys.path idiom in use (no `os.path.dirname` chains)
- pytest version pin uses `>=8.0.0` (one occurrence; no other versions present in the repo to disambiguate)
