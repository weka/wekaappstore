---
phase: 30-wizard-stepper-live-progress
plan: "01"
subsystem: backend-sse
tags: [deploy-stream, sse, appstack, componentStatus, bug-fix]
dependency_graph:
  requires: []
  provides: [PROG-01-precondition, deploy-stream-component-events-for-namespace-preserving-apps]
  affects: [app-store-gui/webapp/main.py]
tech_stack:
  added: []
  patterns: [SSE-componentStatus-poll]
key_files:
  modified:
    - path: app-store-gui/webapp/main.py
      change: "Narrowed deploy-stream guard at line 3080 from 'if not cr_name or app_name in NAMESPACE_PRESERVING_APPS:' to 'if not cr_name:'; updated stale comment"
decisions:
  - "D-13 applied: namespace-override suppression (line 3075) is orthogonal to the componentStatus poll (line 3083); they must not be conflated"
metrics:
  duration: "~3 minutes"
  completed_date: "2026-06-25"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 1
---

# Phase 30 Plan 01: Backend SSE Fix — Narrow deploy-stream Guard (D-13) Summary

**One-liner:** Narrowed deploy-stream guard from `if not cr_name or app_name in NAMESPACE_PRESERVING_APPS:` to `if not cr_name:` so app-store-install and cluster-init stream per-component SSE events instead of short-circuiting to a single immediate `complete`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Narrow line-3080 guard so appStack CRs reach componentStatus poll loop (D-13) | 66431ba | app-store-gui/webapp/main.py |

## What Was Built

A one-line guard change in `/deploy-stream` (plus an updated comment) that corrects the conflation of two unrelated concerns:

1. **Namespace-override suppression** (`main.py:3075`) — `ns_for_apply = "" if app_name in NAMESPACE_PRESERVING_APPS else namespace`. This remains unchanged; namespace-preserving apps keep their fixed per-component `targetNamespace`.

2. **componentStatus poll skip** (`main.py:3083`, was 3080) — previously gated on `not cr_name or app_name in NAMESPACE_PRESERVING_APPS`, which caused both `app-store-install` and `cluster-init` to emit a single `complete` with no `component` events. Now gated on `not cr_name` only, so any appStack CR (which by definition has a `cr_name`) reaches the poll loop.

After this change, the poll loop at lines 3087–3135 runs for all appStack CRs including namespace-preserving ones. The `get_namespaced_custom_object` call at line 3100 queries `namespace` which defaults to `"default"` (line 2990) — both `app-store-install` and `app-store-cluster-init` declare/land in `default`, so the lookup resolves correctly.

## Verification

- `python -m py_compile app-store-gui/webapp/main.py operator_module/main.py` — passed
- `grep -n 'if not cr_name:' app-store-gui/webapp/main.py` — matches line 3083 (deploy-stream guard)
- `grep -n 'app_name in NAMESPACE_PRESERVING_APPS' app-store-gui/webapp/main.py` — 2 occurrences: line 2050 (`deploy()` endpoint, pre-existing) and line 3075 (`ns_for_apply`, unchanged). The line-3080 occurrence is gone.

## Deviations from Plan

None — plan executed exactly as written. The acceptance criteria specified `grep -c` == 1 for `NAMESPACE_PRESERVING_APPS`, but there is a pre-existing occurrence at line 2050 in the separate `deploy()` endpoint. Both remaining occurrences are correct namespace-override suppressions; the problematic guard clause has been removed as intended.

## Known Stubs

None.

## Threat Flags

None. Per threat register entry T-30-09: the poll loop already wraps all emitted component messages in `_redact_secrets()`. The guard change makes already-redacted events reachable for two more apps — no new un-redacted sink is introduced.

## Self-Check: PASSED

- `app-store-gui/webapp/main.py` modified and committed at 66431ba
- Commit confirmed in git log
- `if not cr_name:` present at line 3083
- `ns_for_apply = "" if app_name in NAMESPACE_PRESERVING_APPS else namespace` unchanged at line 3075
