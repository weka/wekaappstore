---
status: partial
phase: 30-wizard-stepper-live-progress
source: [30-VERIFICATION.md]
started: 2026-06-25T00:00:00Z
updated: 2026-06-25T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. End-to-End Install Flow on Live Cluster
expected: Stage list seeded from SSE init event; rows transition grey→yellow→green as component events arrive; complete.ok=true fires only at appStackPhase==Ready; cluster-init chain starts automatically with its stages appended to the same list; browser redirects to redirect_url from /cluster-status after cluster-init Ready.
result: [pending]

### 2. Failure + Retry Path
expected: installError renders inline in an MUI Alert (not an alert() dialog); specific error message from msg.message; Retry button visible; clicking Retry clears the error and re-opens the EventSource stream.
result: [pending]

### 3. Step-1 Next Button Disabled State
expected: Next button appears disabled until "I have applied node prerequisites on all worker nodes" checkbox is checked.
result: [pending]

### 4. Inline Validation UI Rendering
expected: Step 3 with invalid join_ip_ports stays on step 3 with helperText "Each endpoint must be host:port (e.g. 192.168.1.1:14000)"; Step 2 with invalid operator_version shows "Use a version like v1.13.0 or 1.13.0".
result: [pending]

### 5. Review Step Password Masking (Visual Confirmation)
expected: Quay Password and WEKA Password rows show •••••••• (not actual values); quay_dockerconfigjson not visible anywhere on review page.
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
