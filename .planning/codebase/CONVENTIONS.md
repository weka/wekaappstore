# Coding Conventions

**Analysis Date:** 2026-03-17

## Naming Patterns

**Files:**
- Top-level areas use descriptive directory names with hyphens for deployable units and config bundles, such as `app-store-gui/`, `weka-app-store-operator-chart/`, `weka-csi-config/`, and `cluster_init/`.
- Python source files are minimal and entrypoint-oriented: `app-store-gui/webapp/main.py` and `operator_module/main.py`.
- Helm chart templates use lowercase kebab-case filenames, including `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml` and `weka-app-store-operator-chart/templates/tests/test-connection.yaml`.
- Kubernetes sample manifests also use kebab-case filenames, for example `test-pvc.yaml`, `test-pvc-pod.yaml`, and `cluster_init/routes/appstore-route.yaml`.

**Functions:**
- Python functions follow `snake_case`, including `load_kube_config`, `get_cluster_status`, `merge_values`, and `wait_for_component_ready` in `app-store-gui/webapp/main.py` and `operator_module/main.py`.
- FastAPI handlers use `snake_case` names with route-oriented suffixes such as `status_endpoint`, `auth_status_endpoint`, and `get_cluster_info` in `app-store-gui/webapp/main.py`.
- Kopf handler functions are explicit about lifecycle intent: `create_warrpappstore_function`, `update_warrpappstore_function`, and `delete_warrpappstore_function` in `operator_module/main.py`.

**Variables:**
- Module-level constants use uppercase names, for example `BASE_DIR`, `PROJECT_ROOT`, `TEMPLATES_DIR`, `STATIC_DIR`, and `BLUEPRINTS_DIR` in `app-store-gui/webapp/main.py`.
- Local variables and instance attributes use `snake_case`, including `values_file`, `helm_cmd_timeout`, `target_namespace`, and `component_statuses`.
- Private helpers use a leading underscore selectively rather than universally, such as `_config_loaded`, `_deep_merge`, `_load_kube_config_once`, `_doc_id`, and `_ensure_git_sync_binary`.

**Types and Classes:**
- Classes use `PascalCase`, for example `ClusterInitMiddleware`, `HelmOperator`, and `HelmError`.
- Type hints are used in many public helpers and methods, especially in `operator_module/main.py`, but coverage is partial rather than enforced across the repo.
- Typed dictionaries are represented with `Dict[str, Any]` and `List[Dict[str, Any]]` instead of dedicated models or dataclasses in both Python modules.

## Code Style

**Formatting:**
- Python uses 4-space indentation and conventional PEP 8 layout in `app-store-gui/webapp/main.py` and `operator_module/main.py`.
- String style is mixed. Double quotes are common in the FastAPI app, while `operator_module/main.py` often uses single quotes for shell arguments and YAML-related strings.
- The repo does not contain formatter or lint configuration such as `pyproject.toml`, `ruff.toml`, `.flake8`, or `.pre-commit-config.yaml`, so formatting is convention-driven rather than tool-enforced.
- Long modules are accepted as the current norm: `app-store-gui/webapp/main.py` and `operator_module/main.py` each centralize most application behavior in one file.

**Linting:**
- No repo-level lint runner or config is present in the repository root.
- There is no evidence of `ruff`, `flake8`, `black`, `mypy`, or `pylint` configuration in tracked files.
- Quality currently depends on manual consistency, code comments, and runtime checks rather than automated style gates.

## Import Organization

**Order:**
1. Framework and third-party imports first, for example FastAPI and Starlette imports at the top of `app-store-gui/webapp/main.py`.
2. Standard-library imports follow in grouped blocks rather than strict isort ordering, visible in both Python entrypoints.
3. Additional third-party integrations such as `kubernetes`, `jinja2`, `kopf`, and `kr8s` are added afterward as needed.

**Grouping:**
- Imports are grouped in blocks with blank lines between major sections, but not alphabetized strictly.
- `typing` imports are kept in the same general import section rather than separated into type-only groups.
- There are no internal package-relative imports between `app-store-gui/` and `operator_module/`; the deployable pieces are mostly isolated.

## Error Handling

**Patterns:**
- Boundary functions favor broad `try/except` blocks with logging and fallback behavior, especially around Kubernetes API access in `app-store-gui/webapp/main.py`.
- The FastAPI app commonly converts failures into `JSONResponse` payloads rather than propagating exceptions through custom handlers, for example in `/api/secret/huggingface`, `/api/blueprints`, `/deploy`, and `/storage-classes` within `app-store-gui/webapp/main.py`.
- The operator translates unrecoverable reconciliation problems into `kopf.PermanentError` and retryable failures into `kopf.TemporaryError` in `operator_module/main.py`.
- Helper methods frequently return `(success: bool, message: str)` tuples instead of raising domain-specific exceptions, particularly in `HelmOperator` methods in `operator_module/main.py`.

**Error Types:**
- `ApiException` from the Kubernetes client is handled explicitly in many route handlers and cluster helpers in `app-store-gui/webapp/main.py`.
- Generic `Exception` catches are common in both Python modules, often to preserve cluster-facing behavior instead of failing hard.
- Custom exception use is minimal: `HelmError` exists in `operator_module/main.py`, but most logic still reports errors through logs, tuple returns, or Kopf exceptions.

## Logging

**Framework:**
- Standard library `logging` is the only logging framework in use.
- `app-store-gui/webapp/main.py` initializes logging with `logging.basicConfig(level=logging.INFO)` and a module logger.
- `operator_module/main.py` mostly uses `logging` and `self.logger` directly rather than a shared logging wrapper.

**Patterns:**
- Logs are operational and infrastructure-oriented: cluster auth checks, CR readiness, Helm installs/upgrades, manifest application, and deployment progress.
- Both f-strings and printf-style logging are used. For example, `app-store-gui/webapp/main.py` includes `logger.info("APPLY doc[%d]: %s", idx, _doc_id(doc))`, while many other calls use f-strings.
- Error logs usually include contextual resource names such as release, namespace, selector, or component name.

## Comments and Documentation

**When to Comment:**
- Comments are used heavily to explain Kubernetes behavior, deployment assumptions, and operational edge cases.
- Multi-line explanatory comments are common in `app-store-gui/webapp/main.py` around blueprint directory discovery and Kubernetes scoping behavior.
- Helm YAML includes operator-facing notes and provider-specific guidance, especially in `weka-app-store-operator-chart/templates/deploy-app-store-gui.yaml` and `weka-app-store-operator-chart/values.yaml`.

**Docstrings:**
- Key helpers and classes in `operator_module/main.py` and parts of `app-store-gui/webapp/main.py` use docstrings, but coverage is inconsistent.
- Public intent is usually documented on cluster-facing helpers such as `load_kube_config`, `get_cluster_status`, and `install_or_upgrade`.

**TODO Comments:**
- No stable TODO convention is evident. A repository-wide scan did not surface tracked `TODO` or `FIXME` markers in source files.

## Function Design

**Shape:**
- Guard clauses and early returns are common, particularly in HTTP handlers and deployment orchestration helpers.
- Functions often combine orchestration, API calls, and response assembly in one block instead of delegating to smaller service modules.
- Nested helper functions are used when logic is tightly scoped, such as local converters and detection helpers inside `app-store-gui/webapp/main.py`.

**Parameters and Returns:**
- Type-annotated parameters are common on public helpers and route handlers.
- Return shapes are pragmatic rather than strongly modeled: plain dicts, tuples, booleans, and FastAPI response objects dominate.
- Kubernetes and Helm operations often accept raw config dictionaries instead of validated schemas.

## Module Design

**Structure:**
- The repository prefers a small number of large entrypoint modules over a layered package structure.
- `app-store-gui/webapp/main.py` contains middleware, cluster inspection, YAML application, auth checks, sync support, and FastAPI routes in one file.
- `operator_module/main.py` contains Helm helpers, dependency ordering, readiness checks, and all Kopf event handlers in one file.

**Exports and Reuse:**
- There are no barrel modules or explicit public API layers.
- Reuse happens through in-file helper functions and one primary class, `HelmOperator`, inside `operator_module/main.py`.
- Packaging is deployment-first: Dockerfiles in `docker/webapp.Dockerfile` and `docker/operator.Dockerfile` copy only the specific runtime files they need.

## Quality Implications

- The dominant convention is operational pragmatism over strict architecture. New code should match the existing direct style unless the task explicitly includes refactoring.
- Repository quality standards are implicit, not enforced by tooling. Changes should preserve existing naming, logging, and exception-handling patterns because no automated formatter or linter will correct drift.
- When adding behavior, anchor it close to the current entrypoints instead of introducing new framework layers unless the change justifies a broader structural shift.

*Convention analysis: 2026-03-17*
*Update when code style or tooling changes*
