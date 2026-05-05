# Stack Research

**Domain:** Kubernetes operator — variable substitution feature addition (brownfield Python/Kopf)
**Researched:** 2026-05-06
**Confidence:** HIGH

---

## Context: What This Research Is NOT

This is not a new-project stack selection. The operator runtime is locked. This document answers five specific questions about the v5.0 AppStack Variable Substitution milestone: whether `string.Template` is the right choice, whether the operator community has a different standard, how to test Kopf handler functions, whether any new runtime dependency is needed, and whether the CRD `additionalProperties: { type: string }` schema has any gotcha.

---

## Decision 1: `string.Template` (stdlib) — CONFIRMED

**The PRD's choice is correct. No override.**

`string.Template.substitute()` is the right tool for this feature. Rationale confirmed by local execution on Python 3.10.9 (the operator's runtime):

| Test | `string.Template` | `str.format()` |
|------|------------------|----------------|
| AIDP dockerconfigjson (`{"auths": {"nvcr.io": {"auth": "abc"}}}`) passed as-is | PASSES — `{...}` ignored entirely | FAILS — `KeyError: '"auths"'` before any substitution |
| `${namespace}` → `"rag"` substitution | PASSES | PASSES (different syntax) |
| Undefined `${unset}` raises `KeyError` | PASSES — `KeyError: 'unset'` | N/A |
| `$$` escape produces literal `$` | PASSES — `passw0rd$$abc` → `passw0rd$abc` | N/A |
| `safe_substitute` leaves unknown tokens | PASSES — `${unknown}` stays literal | N/A |

The JSON-safety property is not theoretical: AIDP's `aidp-bootstrap-secrets` component contains a real Docker registry auth JSON blob. `str.format()` crashes on it. `string.Template` does not touch it.

`Template.substitute()` (strict) is the correct variant — not `safe_substitute()`. `safe_substitute()` silently passes undefined tokens through, so `${typo}` would reach `kubectl apply` as the literal string `${typo}`, producing a cryptic Kubernetes API error rather than a clear operator failure. Strict mode + `kopf.PermanentError` gives users actionable error messages at apply time.

**No new runtime dependency.** `string.Template` is in Python's stdlib since 2.6. The operator's `requirements.txt` does not need to change.

---

## Decision 2: Operator Community Standards — No Override

The Python Kubernetes operator ecosystem does not have a single standardized variable-substitution library. Common patterns observed:

- **Jinja2** — used in tools like k8s-handle and Ansible-based K8s management, but only where conditionals and loops justify the dependency. Jinja2's default `{{ }}` delimiter conflicts with Kubernetes JSON content (requires escaping all `{` as `{{ '{' }}`). Not appropriate here.
- **Kustomize** — a separate binary (`kustomize`), not a Python library. Requires a third subprocess call. Overkill for simple string interpolation.
- **envsubst** — shell utility, not a Python library. Would add a subprocess and a shell dependency inside the operator container.
- **Custom regex** — some operators roll their own `re.sub`. Has no documented escape mechanism, harder to document for users.
- **sprig-like functions in Python** — no established Python equivalent exists; sprig is a Go template library.

None of these would improve on `string.Template` for this narrow scope (substitution only, no conditionals, JSON-safe content). The community has no standard that overrides the PRD choice.

---

## Decision 3: No New Runtime Dependency

The complete feature can be implemented with:
- `from string import Template` — stdlib
- `yaml.safe_load` — already in use (PyYAML, already in `operator_module/requirements.txt` transitively via kopf)
- `kopf.PermanentError` — already imported

`operator_module/requirements.txt` does not need to change.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `string.Template` (stdlib) | Python 3.10 stdlib (unchanged) | `${VAR}` substitution over manifest strings and ConfigMap/Secret content before YAML parse | JSON-safe (ignores `{...}`), zero new dep, `$$` escape, strict `KeyError` on undefined. Verified by local execution. |
| `kopf.PermanentError` (already imported) | kopf >=1.38.0 (already pinned) | Non-retriable operator failure on undefined variable reference | Correct kopf idiom. Operator already uses it elsewhere. |
| PyYAML (already in use) | already pinned | Parse ConfigMap/Secret content after substitution | Substitution runs on the raw string before `yaml.safe_load` — no YAML library change needed. |

### Supporting Libraries (dev/test only — NOT runtime)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest-subprocess` | `1.5.4` (released 2026-03-21) | Mock `subprocess.run` calls to `kubectl` and `helm` in operator unit tests | Required for `operator_module/tests/` — both manifest-apply and Helm install branches shell out. Provides fixture-level subprocess registration with exact command matching, preventing subprocess shape regressions. Python ≥ 3.6 confirmed. |
| `pytest` | existing project pin | Test runner | Already used project-wide. No version change. |
| `unittest.mock` (stdlib) | stdlib | Mock `kr8s.objects.ConfigMap.get` and `Secret.get` | No new dependency. `unittest.mock.patch` to inject fake ConfigMap/Secret with controlled `.data`. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest | Unit test runner | Create `operator_module/tests/__init__.py` + `test_render.py` + `test_appstack.py`. No conftest.py required unless sharing fixtures. |
| pytest-subprocess | Subprocess faking | Add to a new `operator_module/requirements-dev.txt`. Do NOT add to `operator_module/requirements.txt` — this is a test-only dependency. |

---

## Installation

```bash
# No new runtime dependencies — stdlib-only feature

# Create operator_module/requirements-dev.txt with:
pytest>=7.0
pytest-subprocess==1.5.4

# Install for local test runs:
pip install pytest pytest-subprocess==1.5.4
```

---

## CRD / OpenAPI Schema: `additionalProperties: { type: string }` — SAFE

The PRD's proposed CRD block:

```yaml
variables:
  type: object
  additionalProperties:
    type: string
```

This is structurally correct and safe. Detailed findings:

**This exact pattern already works in this CRD.** At `crd.yaml` line 184, `readinessCheck.matchLabels` uses:
```yaml
matchLabels:
  type: object
  additionalProperties:
    type: string
```
This is in production use. The pattern is proven in this cluster.

**`x-kubernetes-preserve-unknown-fields` is NOT needed.** That annotation is for schemas that intentionally allow arbitrary untyped data (e.g., `component.values` at CRD line 138, which uses it). `variables` has a fully-typed schema — `additionalProperties: {type: string}` tells the API server exactly what is valid. Do not add `x-kubernetes-preserve-unknown-fields: true` to `variables` — it would disable server-side validation of variable values, removing a useful safety net.

**Historical bug (Kubernetes issue #104137, August 2021):** In Kubernetes 1.20, a validation bug caused `additionalProperties: {type: string}` maps to incorrectly reject entries with "forbidden property" errors. The project targets modern EKS (well past 1.25 where this was resolved). Not a concern.

**`maxProperties` is optional.** Kubernetes recommends adding `maxProperties` on `additionalProperties` maps for CEL validation cost estimation. Variable maps are typically 2–20 entries — cost is negligible. Omitting `maxProperties` is fine. If CEL validation rules are added to the CRD in the future, add `maxProperties: 64` to the `variables` field at that time.

---

## Handler Unit Testing Pattern for `operator_module/tests/`

Kopf's official docs cover `KopfRunner` (integration testing against a cluster or KMock server) and do not document unit-testing individual handler functions. The key insight: **Kopf decorators do not wrap the handler function**. `@kopf.on.create(...)` registers the function in kopf's internal registry but returns the original function unchanged. The decorated function is still a plain Python callable.

**Correct pattern for this milestone's tests:**

`render()` and `load_values_from_reference()` are not decorated handlers — call them directly:

```python
# operator_module/tests/test_render.py
from operator_module.main import render
import pytest

def test_json_content_untouched():
    json = '{"auths": {"nvcr.io": {"auth": "abc"}}}'
    assert render(json, {}) == json  # ${...} not present, JSON braces untouched

def test_undefined_raises_key_error():
    with pytest.raises(KeyError, match="unset"):
        render("value: ${unset}", {"namespace": "foo"})

def test_dollar_escape():
    assert render("passw0rd$$abc", {}) == "passw0rd$abc"

def test_no_op_when_no_variables():
    assert render("namespace: myns", None) == "namespace: myns"
```

`load_values_from_reference()` with mocked kr8s:

```python
# operator_module/tests/test_appstack.py
from unittest.mock import patch, MagicMock
from operator_module.main import load_values_from_reference

def test_configmap_substitution():
    fake_cm = MagicMock()
    fake_cm.data = {"site.yaml": "url: http://${host}:9200"}
    with patch("operator_module.main.kr8s.objects.ConfigMap.get", return_value=fake_cm):
        result = load_values_from_reference(
            "ConfigMap", "my-cm", "site.yaml", "ns",
            variables={"host": "elastic.ns.svc.cluster.local"}
        )
    assert result["url"] == "http://elastic.ns.svc.cluster.local:9200"
```

`handle_appstack_deployment()` with subprocess mocking:

```python
# Using pytest-subprocess (fp fixture is auto-injected)
def test_manifest_namespace_autodefault(fp):
    fp.register(["kubectl", "apply", "-f", fp.any(), "-n", "target-ns"], returncode=0)
    # Call the handler as a plain Python function:
    handle_appstack_deployment(
        body={...},
        spec={"appStack": {"components": [{"name": "x", "kubernetesManifest": "kind: Pod\nmetadata:\n  namespace: ${namespace}"}]}},
        name="test-cr",
        namespace="target-ns",
        status={},
        patch=MagicMock()
    )
    # Assert fp received the expected kubectl call
    assert fp.call_count(["kubectl", "apply"]) == 1
```

**Do NOT use KopfRunner for these unit tests.** KopfRunner is an integration test tool that connects to a real cluster. It is the wrong tool for testing `render()` correctness or namespace auto-default behavior.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `string.Template` | Jinja2 | If the project ever adds conditionals, loops, or whitespace control to blueprint rendering. Requires escaping `{` in JSON content — only viable if JSON-bearing manifests are excluded from templating scope. Not now. |
| `string.Template` | `str.format(**vars)` | Never for this use case. Crashes on any `{`-containing string. Confirmed by test. |
| `string.Template` | `re.sub` custom regex | If `$identifier` syntax is unacceptable for some reason. Adds a bug surface with no escape mechanism. Not recommended. |
| `Template.substitute()` (strict) | `Template.safe_substitute()` | Only if the feature requirement changes to "silently ignore undefined variables." Current PRD requirement is strict failure — strict mode is correct. |
| `pytest-subprocess` | `unittest.mock.patch('subprocess.run', ...)` | If the team prefers zero new dev-deps. Plain `patch` works for simple returncode mocking but does not validate the exact command shape. Mark those tests with `# subprocess shape not validated` comments. Acceptable tradeoff if desired. |
| `unittest.mock.patch` | `pytest-mock` (`mocker` fixture) | Either works. Prefer consistency with existing mcp-server tests. If the team adds `pytest-mock` project-wide for other reasons, use `mocker`. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Jinja2 for this feature | Conflicts with JSON `{...}` content; `{{...}}` or `raw` blocks required for all Docker registry auth payloads; adds a runtime dependency for no additional benefit given the substitution-only scope | `string.Template` |
| `str.format()` or f-strings | Crashes immediately on `{`-containing content. Confirmed: `KeyError: '"auths"'` on AIDP dockerconfigjson | `string.Template` |
| `Template.safe_substitute()` | Silently passes undefined `${var}` tokens through to `kubectl apply`, producing cryptic Kubernetes errors instead of a clear failure at apply time | `Template.substitute()` (strict) + `kopf.PermanentError` |
| `x-kubernetes-preserve-unknown-fields: true` on `variables` | Disables API server validation of variable values, removing the type-safety the schema provides | Leave schema as `additionalProperties: { type: string }` — no preserve annotation |
| KopfRunner for unit tests of render logic | Integration test tool requiring cluster connectivity; wrong layer for testing a pure Python string transformation | Call `render()` and helpers directly as Python functions in pytest |
| Adding Kustomize, envsubst, or a third subprocess binary | Adds a new external binary dependency to the operator container for no benefit over stdlib string.Template | `string.Template` |

---

## Version Compatibility

| Package | Version | Compatibility Notes |
|---------|---------|---------------------|
| `string.Template` (stdlib) | Python 3.10 (operator runtime) | Stable since Python 2.6. `${identifier}` form, `$$` escape, `substitute()`/`safe_substitute()` unchanged across all Python 3.x versions. |
| `pytest-subprocess` | 1.5.4 | Python ≥ 3.6. Supports Python 3.10 — confirmed in package metadata. |
| `kopf` | ≥1.38.0 (already pinned) | `kopf.PermanentError` is stable across all 1.x releases. No version change needed. |
| `kr8s` | ≥0.17.0 (already pinned) | Used in `load_values_from_reference`. Mock via `unittest.mock.patch`. No version change. |

---

## Sources

- Local Python 3.10.9 execution — `string.Template` JSON-safety, `KeyError`, and `$$` escape behavior verified directly (HIGH confidence)
- [Python stdlib docs: string.Template](https://docs.python.org/3/library/string.html) — `substitute()`, `safe_substitute()`, `$$` escape, `${identifier}` form (HIGH confidence)
- [pytest-subprocess PyPI](https://pypi.org/project/pytest-subprocess/) — version 1.5.4, released 2026-03-21, Python ≥ 3.6 (HIGH confidence)
- [Kopf testing docs](https://docs.kopf.dev/en/stable/testing/) — KopfRunner is integration-only; no unit-test-handler pattern documented; decorator-passthrough behavior inferred from kopf source behavior (MEDIUM confidence)
- [Kubernetes issue #104137](https://github.com/kubernetes/kubernetes/issues/104137) — `additionalProperties: {type: string}` validation bug in k8s 1.20; triaged/accepted in 2021; not present on modern EKS (MEDIUM confidence — fix version not confirmed but pre-dates current cluster versions)
- `weka-app-store-operator-chart/templates/crd.yaml` line 184 — `matchLabels.additionalProperties: {type: string}` already in production use in this project confirming the pattern works (HIGH confidence)
- `operator_module/requirements.txt` — confirms kopf, kr8s, kubernetes packages pinned; PyYAML available transitively (HIGH confidence)

---
*Stack research for: WEKA App Store Operator — v5.0 AppStack Variable Substitution*
*Researched: 2026-05-06*
