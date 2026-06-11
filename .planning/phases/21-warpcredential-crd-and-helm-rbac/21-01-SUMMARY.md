---
phase: 21-warpcredential-crd-and-helm-rbac
plan: 01
subsystem: infra
tags: [kubernetes, crd, helm, openAPIV3Schema, warp.io]

# Dependency graph
requires: []
provides:
  - "warpcredentials.warp.io CRD with full typed openAPIV3Schema"
  - "spec.type enum admission enforcement (nvidia-ngc | huggingface | weka-storage)"
  - "status subresource with conditions, derivedSecrets, lastSyncTime, wekaEndpoint"
affects:
  - "22-operator-warpcredential-handler"
  - "23-credentials-api"
  - "24-settings-gui-overhaul"
  - "25-blueprint-credential-sdk"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Multi-CRD single file: second CRD appended with --- separator, both guarded by same Helm conditional"

key-files:
  created: []
  modified:
    - "weka-app-store-operator-chart/templates/crd.yaml"

key-decisions:
  - "WarpCredential CRD placed in crd.yaml after WekaAppStore CRD with --- separator and separate {{- if .Values.customResourceDefinition.create }} guard so both CRDs are independently controlled by the same Helm values toggle"
  - "No x-kubernetes-preserve-unknown-fields on any WarpCredential field — fully typed schema ensures admission validation is complete and accurate"
  - "spec.endpoint field intentionally optional with description noting weka-storage-only use, not enforced with per-type conditional logic at CRD level"
  - "status subresource enabled via subresources.status: {} so the operator can write status fields without triggering spec reconcile loop"

patterns-established:
  - "Multi-CRD pattern: append additional CRDs to crd.yaml with --- separator, each wrapped in {{- if .Values.customResourceDefinition.create }}"

requirements-completed:
  - CRD-01
  - CRD-02
  - CRD-03
  - CRD-04
  - CRD-05

# Metrics
duration: 5min
completed: 2026-06-11
---

# Phase 21 Plan 01: WarpCredential CRD Summary

**WarpCredential CRD (warpcredentials.warp.io) added to crd.yaml with typed openAPIV3Schema enforcing nvidia-ngc/huggingface/weka-storage enum at admission and full status subresource**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-11T04:55:00Z
- **Completed:** 2026-06-11T04:57:58Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- WarpCredential CRD appended to crd.yaml following the existing WekaAppStore CRD pattern with `---` separator and same Helm conditional guard
- Admission validation: `spec.type` enum rejects any value outside `[nvidia-ngc, huggingface, weka-storage]` at the Kubernetes API server before the operator sees the CR
- Required field enforcement: `spec.displayName`, `spec.secretRef.name`, `spec.secretRef.key` all marked required in openAPIV3Schema; `spec.endpoint` is optional
- Status subresource enabled with `conditions` (type, status, reason, message, lastTransitionTime), `derivedSecrets` (name + type per derived secret), `lastSyncTime`, and `wekaEndpoint` fields

## Task Commits

1. **Task 1: Append WarpCredential CRD to crd.yaml** - `612d231` (feat)

**Plan metadata:** (final commit — see below)

## Files Created/Modified

- `weka-app-store-operator-chart/templates/crd.yaml` - WarpCredential CRD appended (102 lines added); both WekaAppStore and WarpCredential CRDs present

## Decisions Made

- Multi-CRD pattern: append with `---` separator inside same file, each CRD independently guarded by `{{- if .Values.customResourceDefinition.create }}`. This matches the plan spec exactly and mirrors the upstream Helm pattern used across Kubernetes ecosystem charts.
- Fully typed schema (no `x-kubernetes-preserve-unknown-fields`) so admission validation is complete. The existing WekaAppStore CRD uses `x-kubernetes-preserve-unknown-fields: true` on `values` fields for Helm pass-through, which is not needed here.
- `spec.endpoint` is optional at CRD level with description text explaining the weka-storage-only semantics. Per-type enforcement is handled by the operator (Phase 22), not the schema.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- WarpCredential CRD is ready for Phase 22 operator handler: `kubectl apply` will register the CRD, and the kopf handler can then watch `warpcredentials.warp.io` resources
- Phase 21 Plan 02 (Helm RBAC) can proceed independently — it adds Role/RoleBinding to the chart for Secret CRUD, no dependency on this plan beyond chart structure

---
*Phase: 21-warpcredential-crd-and-helm-rbac*
*Completed: 2026-06-11*

## Self-Check: PASSED

- [FOUND] weka-app-store-operator-chart/templates/crd.yaml - modified with WarpCredential CRD
- [FOUND] commit 612d231 - feat(21-01): add WarpCredential CRD to crd.yaml
- helm lint: 0 failures
- helm template: 2 CRDs rendered (WekaAppStore + WarpCredential)
- YAML validation: python3 yaml.safe_load_all exits 0
- enum check: nvidia-ngc, huggingface, weka-storage all present
- status fields: conditions, derivedSecrets, lastSyncTime, wekaEndpoint all present
