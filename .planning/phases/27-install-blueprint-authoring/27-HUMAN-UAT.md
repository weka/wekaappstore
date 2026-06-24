---
status: partial
phase: 27-install-blueprint-authoring
source: [27-VERIFICATION.md]
started: 2026-06-24T00:00:00Z
updated: 2026-06-24T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. CSI deployment readinessCheck selector matches live chart
expected: The `csi-wekafs` component's `readinessCheck` (type: deployment) correctly matches the actual deployment name created by the `csi-wekafsplugin` Helm chart in the `csi-wekafs` namespace. No timeout/mismatch on cluster apply.

result: [pending]

### 2. WekaClient CR schedules successfully after node labeling
expected: After the `weka-node-label-job` completes and labels nodes with `weka.io/supports-clients=true`, the WekaClient DaemonSet pods (which use that nodeSelector) schedule and start on labeled nodes. The no-readinessCheck design (D-03) is safe — the WekaClient operator reconciles independently.

result: [pending]

### 3. Storageclass-demote Job is idempotent on empty cluster
expected: When no existing StorageClass carries `is-default-class: "true"`, the demote Job exits cleanly (no error) rather than failing with a "no resources found" error. The Job completes successfully before the `storageclasses` component is applied.

result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
