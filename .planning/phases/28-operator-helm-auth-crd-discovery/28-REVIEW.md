---
phase: 28-operator-helm-auth-crd-discovery
reviewed: 2026-06-24T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - operator_module/main.py
  - operator_module/tests/test_operator_helm_auth.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 28: Code Review Report

**Reviewed:** 2026-06-24
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 28 threads an OCI `--registry-config <tempfile>` through the CRD-discovery /
install / upgrade path and replaces the prior `@lru_cache` on chart-CRD discovery
with a success-only module dict (`_chart_crds_cache`). Credential handling on the
argv is sound: the secret stays in a file referenced by path, never appearing as a
command-line token (the test `test_registry_config_flag_present_for_oci_quay_chart`
proves this, and the `--registry-config` flag carries only the path). The temp file
is unlinked in a `finally` that covers the install-failure path. The success-only
cache correctly avoids negative memoization of transient `helm show crds` failures.

However, the cache-key design defeats the cache for the exact case Phase 28 was
built for (OCI/quay charts) and grows the cache without bound, and several failure
paths are silent in a way that will make field auth failures hard to diagnose.
There is also a temp-file leak window on an exception between file creation and the
`try` whose `finally` removes it. No BLOCKER-class defects (no credential leakage,
no data loss) were found, but the WARNING items below should be fixed before ship.

## Warnings

### WR-01: `registry_config_path` in the cache key defeats caching and leaks cache entries for OCI charts

**File:** `operator_module/main.py:685, 701, 732`
**Issue:** `_chart_crds_cache` is keyed by `(chart_ref, version, registry_config_path)`.
For every OCI/quay reconcile, `handle_appstack_deployment` writes a *fresh*
`tempfile.NamedTemporaryFile` (main.py:1191-1193), producing a new random path
(`/tmp/tmpXXXX.json`) each time. That path becomes part of the cache key, so:
- The cache **never hits across reconciles** for the OCI-auth path — i.e. for the
  precise scenario Phase 28 added — so `helm show crds` is re-invoked on every
  reconcile of every quay component, defeating the purpose of the cache.
- `_chart_crds_cache` accumulates one new entry per reconcile and is **never
  evicted** (it replaced `@lru_cache(maxsize=1)` with an unbounded plain dict).
  Over the operator's lifetime this is an unbounded-growth leak keyed on dead temp
  paths.

The CRD set returned by `helm show crds` does not depend on *which* auth file is
used — only on whether auth succeeds. Including the path in the key conflates
"identity of the chart's CRDs" with "identity of this reconcile's temp file."

**Fix:** Key the cache on chart identity only, and gate caching of the success on
auth being unnecessary-or-stable. Simplest correct form:
```python
cache_key = (chart_ref, version)
if cache_key in _chart_crds_cache:
    return _chart_crds_cache[cache_key]
# ... run helm (still pass registry_config_path to the subprocess) ...
_chart_crds_cache[cache_key] = names
```
If the intent was genuinely to distinguish authed vs unauthed lookups, key on a
stable boolean (`registry_config_path is not None`) rather than the volatile path:
`cache_key = (chart_ref, version, registry_config_path is not None)`.
Note this requires updating `test_registry_config_path_in_cache_key`
(test_operator_helm_auth.py:101-112), which currently asserts the volatile-path
behavior as if it were correct — the test locks in the bug.

### WR-02: Temp registry-config file leaks if an exception is raised before the `try` block

**File:** `operator_module/main.py:1190-1196, 1244-1247`
**Issue:** The temp file is created at lines 1191-1193, but the `try:` whose
`finally:` unlinks it does not open until line 1196. Any exception thrown in the
gap — currently none, but the gap is real and fragile — would leak the file
containing the quay credential on disk. More importantly, the `with
tempfile.NamedTemporaryFile(...)` write itself (1191-1193) is outside the `try`; if
`rcf.write()` raised (e.g. disk full, or `quay_dockerconfigjson` not a str), the
partially-created file is leaked because `delete=False` and the `finally` is not yet
armed.
**Fix:** Move the temp-file creation inside the `try` (or wrap creation+use in a
single `try/finally`) so the cleanup guard covers the file from the instant it
exists:
```python
registry_config_path = None
try:
    if chart_repo and chart_repo.startswith("oci://") and "quay_dockerconfigjson" in stack_vars:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as rcf:
            rcf.write(stack_vars["quay_dockerconfigjson"])
            registry_config_path = rcf.name
    # ... CRD strategy + install_or_upgrade ...
finally:
    if registry_config_path is not None and os.path.exists(registry_config_path):
        os.unlink(registry_config_path)
```

### WR-03: `discover_chart_crds` silently swallows helm failures — OCI auth errors become invisible "no CRDs"

**File:** `operator_module/main.py:711-716`
**Issue:** Both `except subprocess.CalledProcessError:` and the bare
`except Exception:` return `set()` with **no logging at all**. The success-only
cache was added precisely so an auth/network failure is not memoized as "no CRDs"
— but the operator still has no way to tell an operator/SRE that
`helm show crds oci://quay.io/...` failed because the credential was wrong vs. the
chart genuinely has no CRDs. `subprocess.check_output` is called without
`stderr=subprocess.PIPE`, so helm's stderr is neither captured nor logged in
context. The downstream effect: `should_skip_crds_for_component` sees an empty set,
returns `False` ("nothing to skip"), and the real failure surfaces much later (or
not at all) as a confusing install error.
**Fix:** Capture stderr and log at warning level (it is helm diagnostic output, not
the credential — the credential is in the registry-config file, not on stderr):
```python
try:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    logging.warning(f"helm show crds failed for {chart_ref} (v={version}): {e.stderr or e}")
    return set()
except Exception as e:
    logging.warning(f"helm show crds errored for {chart_ref} (v={version}): {e}")
    return set()
```

### WR-04: Tests assert the credential is in the temp file only during `helm show crds`, not during install — coverage gap

**File:** `operator_module/tests/test_operator_helm_auth.py:196-226`
**Issue:** `test_registry_config_flag_present_for_oci_quay_chart` reads and asserts
the temp-file contents inside `_check_output_capture_file` (the `helm show crds`
mock) only. It never asserts the file still holds the credential at the moment
`helm install` runs (the `_run` mock at line 154 captures argv but never opens the
referenced file). If a future change unlinked or rewrote the file between CRD
discovery and install, this test would still pass while real installs lost auth.
The secrecy assertion (`_assert_credential_absent`) is good, but the
"file-actually-usable-at-install-time" invariant is unverified.
**Fix:** In the `_run` mock, when the cmd is `helm install`/`helm upgrade`, locate
the `--registry-config` path, read it while it exists, and append to `seen_contents`
(or a second list); then assert the credential was present in an install-phase read,
not just a CRD-discovery read.

## Info

### IN-01: `_make_helm_run_capture` returns a `check_output` callable that two tests discard

**File:** `operator_module/tests/test_operator_helm_auth.py:143-166, 194, 259, 284`
**Issue:** `test_registry_config_flag_present_for_oci_quay_chart` builds
`run, check_output, captured` then ignores the returned `check_output` (it patches
with a local `_check_output_capture_file` instead). The unused unpacked variable is
dead within that test. Minor readability/consistency nit, not a correctness issue.
**Fix:** Unpack as `run, _check_output, captured` in that test, or build the capture
helper without the unused return there.

### IN-02: Cache-clear coupling — every test mutates module-global `_chart_crds_cache` via `.clear()`

**File:** `operator_module/tests/test_operator_helm_auth.py:65, 85, 104, 192, 232, 257, 282`
**Issue:** Each test calls `main._chart_crds_cache.clear()` at the top to isolate
from sibling tests. This works but relies on manual discipline; a new test that
forgets the clear will be order-dependent and flaky. Consider an autouse fixture
that clears the cache (and `list_existing_crds.cache_clear()` /
`_load_kube_config_once.cache_clear()`) before each test.
**Fix:** Add to a module-level fixture:
```python
@pytest.fixture(autouse=True)
def _reset_caches():
    main._chart_crds_cache.clear()
    yield
```

### IN-03: `discover_chart_crds` caches the genuine-empty set but `should_skip_crds_for_component` cannot distinguish it from auth-failure empty

**File:** `operator_module/main.py:771-777`
**Issue:** Tied to WR-03. Because a failed lookup returns an *uncached* `set()` and a
genuine "no CRDs" returns a *cached* `set()`, both look identical to
`should_skip_crds_for_component`, which treats both as "no CRDs in chart -> nothing
to skip" (returns `False`). That is the intended conservative behavior, but it means
an auth failure during Auto-strategy CRD discovery silently downgrades to
"install CRDs" (no `--skip-crds`), which can collide with externally-managed CRDs.
Worth a one-line comment documenting that Auto-on-auth-failure intentionally errs
toward letting Helm manage CRDs, so a future reader does not "fix" it.
**Fix:** Documentation-only; add a comment at main.py:772-774 noting the
auth-failure-vs-empty conflation and the deliberate fail-toward-install choice.

---

_Reviewed: 2026-06-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
