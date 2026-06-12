---
status: partial
phase: 24-settings-gui-overhaul
source: [24-VERIFICATION.md]
started: 2026-06-12T01:40:00Z
updated: 2026-06-12T01:40:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Single-open-form cross-type collapse
expected: Click NGC add-form button, then WEKA add-form button — NGC form collapses, WEKA form opens (only one form open at a time)
result: [pending]

### 2. Amber → green transition
expected: Add credential with operator running — row appears amber, transitions to green within reconcile window (~2-second polling)
result: [pending]

### 3. 30-second amber → red timeout
expected: Stop operator, add credential, wait 30 s — row transitions from amber to red with locked timeout message
result: [pending]

### 4. Page Visibility polling pause/resume
expected: Background the tab while a credential row is amber — network requests pause; return to tab — polling resumes with elapsed budget preserved
result: [pending]

### 5. WEKA Overview loading → success
expected: With live WEKA cluster credential — panel shows loading spinner, then renders capacity card, filesystem table, backend IP grid
result: [pending]

### 6. WEKA Overview error state
expected: Unreachable WEKA endpoint — locked red error banner shown, success div hidden, no spinner
result: [pending]

### 7. Filesystem amber mini-bar
expected: Filesystem at >= 90% utilisation — orange/amber bar color (not purple)
result: [pending]

### 8. Show-all toggle with > 20 filesystems
expected: "Show all (N) ▾" button appears; clicking expands to all rows; "Show top 20 ▴" collapses back
result: [pending]

### 9. Credential selector vs static label
expected: 2 ready WEKA credentials → dropdown <select>; 1 credential → static <span> with credential name
result: [pending]

### 10. Delete confirm dialog + row fade
expected: Delete button shows browser confirm with credential name; confirmed → row fades out over 150ms and disappears
result: [pending]

## Summary

total: 10
passed: 0
issues: 0
pending: 10
skipped: 0
blocked: 0

## Gaps
