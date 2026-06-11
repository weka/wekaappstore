---
phase: 21-warpcredential-crd-and-helm-rbac
plan: "02"
subsystem: helm-chart-rbac
tags: [helm, rbac, kubernetes, secrets, role, rolebinding]
dependency_graph:
  requires: []
  provides: [helm-secret-manager-role, helm-secret-manager-rolebinding]
  affects: [weka-app-store-operator-chart]
tech_stack:
  added: []
  patterns: [helm-conditional-guard, namespace-scoped-rbac, serviceaccount-helper-ref]
key_files:
  created:
    - weka-app-store-operator-chart/templates/rbac.yaml
  modified:
    - weka-app-store-operator-chart/Chart.yaml
decisions:
  - "Used kind: Role (not ClusterRole) to scope Secret CRUD to .Release.Namespace per CRD-06 least-privilege requirement"
  - "Guarded with {{- if .Values.rbac.create }} matching existing chart RBAC pattern (not clusterWide guard)"
  - "ServiceAccount name resolved via weka-app-store-operator-chart.serviceAccountName helper — not hardcoded"
metrics:
  duration: "~5 minutes"
  completed: "2026-06-11T04:57:52Z"
  tasks_completed: 2
  files_changed: 2
---

# Phase 21 Plan 02: Helm RBAC — Namespace-Scoped Secret Manager Role Summary

Namespace-scoped Role and RoleBinding granting operator Secret CRUD in the App Store namespace, plus chart version bump to 0.1.64.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create rbac.yaml with namespace-scoped Role and RoleBinding | 0ab1375 | weka-app-store-operator-chart/templates/rbac.yaml |
| 2 | Bump Chart.yaml version to 0.1.64 | 934648e | weka-app-store-operator-chart/Chart.yaml |

## What Was Built

### rbac.yaml

Created `weka-app-store-operator-chart/templates/rbac.yaml` containing:

- **Role** `<fullname>-secret-manager`: namespace-scoped (kind: Role), grants `get/list/watch/create/update/patch/delete` on `secrets` resource in the core API group (`""`). Bounded to `.Release.Namespace`.
- **RoleBinding** `<fullname>-secret-manager`: binds the Role to the operator ServiceAccount (resolved via `weka-app-store-operator-chart.serviceAccountName` helper). Both subject and roleRef are in `.Release.Namespace`.

Both resources are guarded by `{{- if .Values.rbac.create }}` matching the existing chart RBAC conditional pattern.

### Chart.yaml

Version field changed from `0.1.63` to `0.1.64`. No other changes.

## Verification Results

| Check | Result |
|-------|--------|
| `helm lint weka-app-store-operator-chart` | Passed (0 failures) |
| `helm template` exits 0 | Passed |
| New `kind: Role` present | Passed — rendered at line 850 |
| New `kind: RoleBinding` present | Passed — rendered at line 882 |
| Role NOT a ClusterRole | Passed — `kind: Role` not `kind: ClusterRole` |
| secrets resource in rules | Passed — `resources: ["secrets"]` |
| write verbs present (create/patch/delete) | Passed — `verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]` |
| Chart version = 0.1.64 | Passed |
| ServiceAccount via helper (not hardcoded) | Passed — `{{ include "weka-app-store-operator-chart.serviceAccountName" . }}` |

## Deviations from Plan

None — plan executed exactly as written.

Note: The acceptance criteria specified "exactly 1 match" for `kind: Role$` and `kind: RoleBinding$`, but the chart already contained a pre-existing `wekaappstoregui-cr-manager` Role and RoleBinding in `deploy-app-store-gui.yaml`. The final counts are 2 each, with the new `secret-manager` Role+RoleBinding correctly added. This is expected behavior and not a deviation.

## Threat Model Coverage

| Threat ID | Mitigated By |
|-----------|--------------|
| T-21-04 (Elevation of Privilege — Role scope) | Used `kind: Role` (namespace-scoped), not `ClusterRole`. Secret CRUD cannot escape `.Release.Namespace`. |
| T-21-05 (Info Disclosure — verbs) | Accepted per plan — full CRUD required for Phase 22 reconciler idempotency. |
| T-21-06 (Tampering — ClusterRole overlap) | Accepted per plan — existing ClusterRole wildcard already covers this surface; new Role documents it explicitly. |

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary surfaces introduced beyond what the plan's threat model covers.

## Self-Check: PASSED

Files exist:
- weka-app-store-operator-chart/templates/rbac.yaml: FOUND
- weka-app-store-operator-chart/Chart.yaml: FOUND (version: 0.1.64)

Commits exist:
- 0ab1375: feat(21-02): add namespace-scoped Role and RoleBinding for Secret CRUD
- 934648e: chore(21-02): bump chart version to 0.1.64
