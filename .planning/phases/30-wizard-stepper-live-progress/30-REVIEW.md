---
phase: 30-wizard-stepper-live-progress
reviewed: 2026-06-25T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - app-store-gui/webapp/main.py
  - app-store-gui/webapp/templates/welcome.html
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 30: Code Review Report

**Reviewed:** 2026-06-25
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 30 delivers two changes: a backend fix that removes the early-exit bypass for NAMESPACE_PRESERVING_APPS from the SSE poll loop, and a substantial frontend overhaul that replaces the old single-button welcome screen with a 5-step Material-UI wizard and a live per-component progress list driven by the `/deploy-stream` SSE endpoint.

The backend diff is small and correct in intent, but introduces a secret redaction gap on the "Failed" path that existed before this phase and is now partially (but incompletely) addressed. The frontend wizard is largely well-structured but has two correctness bugs: a namespace value that is silently dropped for both `openClusterInitStream` and `handleInitialize`, causing those deploys to always target `"default"` regardless of the user's namespace selection; and a silent connection failure in both new `onerror` handlers that leaves the user staring at an indeterminate spinner with no actionable feedback.

## Critical Issues

### CR-01: `openClusterInitStream` passes `namespace` as a URL query param that `deploy_stream` ignores — always uses "default"

**File:** `app-store-gui/webapp/templates/welcome.html:437`
**Issue:** `openClusterInitStream` constructs the EventSource URL as:
```
/deploy-stream?app_name=cluster-init&namespace=${selectedNamespace}
```
But `deploy_stream` in `main.py` declares only two parameters — `app_name` and `variables` (line 2961-2962). Namespace is extracted exclusively from inside the `variables` JSON dict (line 2990):
```python
namespace = str(user_vars.get("namespace", "default") or "default").strip() or "default"
```
The `namespace` query parameter in the URL is silently ignored by FastAPI because it is not a declared path/query parameter of the handler. The cluster-init CR is therefore always polled in namespace `"default"`, even when the user selected a different namespace on the Review step.

This means if the user selects a non-default namespace (e.g. `weka`), the `app-store-install` stage succeeds (because `buildVariables()` includes `namespace` inside the `variables` JSON), but the subsequent `openClusterInitStream` call deploys and polls `cluster-init` against `"default"`, causing the poll to either find the wrong (or no) CR and time out.

The same bug exists in `handleInitialize` (line 499) which is used by the legacy right-panel Retry button.

**Fix:** Pass `namespace` inside the `variables` JSON, not as a bare query parameter:
```javascript
// openClusterInitStream — replace line 437
function openClusterInitStream() {
  const params = new URLSearchParams({
    app_name: 'cluster-init',
    variables: JSON.stringify({ namespace: selectedNamespace }),
  });
  const es = new EventSource(`/deploy-stream?${params.toString()}`);
  // ...
}

// handleInitialize — replace line 499
const params = new URLSearchParams({
  app_name: 'cluster-init',
  variables: JSON.stringify({ namespace: selectedNamespace }),
});
const url = `/deploy-stream?${params.toString()}`;
```

---

### CR-02: `es.onerror` in `handleInstall` and `openClusterInitStream` silently swallows connection failures — user sees a frozen spinner

**File:** `app-store-gui/webapp/templates/welcome.html:430` and `486`
**Issue:** Both new SSE handlers set:
```javascript
es.onerror = () => {
  es.close();
};
```
When the SSE connection drops (network error, server restart, proxy timeout), this handler closes the connection but sets no error state. `progressActive` remains `true`, `stages` shows whatever was last received, and `installError` stays `null`. The user sees a frozen progress list with no indication that the stream ended abnormally and no Retry button.

This differs from the `handleInitialize` path, which at least calls `console.error` and closes without setting `failed=true`, but even that path only populates `errorCount` via the polling loop — not via the SSE error path. The wizard's `es.onerror` is strictly worse because there is no secondary polling loop.

**Fix:**
```javascript
// In handleInstall
es.onerror = () => {
  setInstallError('Connection to deployment stream lost. Check the server and retry.');
  setProgressActive(false);  // re-show wizard so user can retry from step review
  es.close();
};

// In openClusterInitStream
es.onerror = () => {
  setInstallError('Connection to cluster-init stream lost. Check the server and retry.');
  es.close();
};
```

## Warnings

### WR-01: `progressActive` is never reset to `false` — there is no way to return to the wizard after a connection error

**File:** `app-store-gui/webapp/templates/welcome.html:381`
**Issue:** `setProgressActive(true)` is called once in `handleInstall` (line 381) and is never reset. Once the wizard's Install button is pressed, the wizard pane is permanently hidden (`{!progressActive && ...}` at line 798) for the entire session. The only Retry path re-invokes `handleInstall()` directly (which assumes `progressActive` is already true), so if validation fails during retry (e.g. user changes nothing), the error state is set but the wizard remains hidden and the user cannot correct input without a page reload.

Strictly this is a UX deadlock, not a data-loss risk, but it degrades robustness on the failure path.

**Fix:** Reset `progressActive` to `false` in the `onerror` handler (see CR-02 fix above) or add an explicit "Back to settings" button when `installError` is set.

---

### WR-02: Namespace-preserving apps now reach the CR poll loop but use the user-supplied namespace for polling — mismatch when cluster-init lives in a fixed namespace

**File:** `app-store-gui/webapp/main.py:3103`
**Issue:** The phase 30 backend change removes `app_name in NAMESPACE_PRESERVING_APPS` from the early-exit condition (diff line ~3083). Namespace-preserving apps (`cluster-init`, `app-store-install`) now enter the CR poll loop. The comment claims these apps ARE appStacks that should be polled, which is correct in intent.

However, the poll at line 3103 uses `namespace` from `user_vars.get("namespace", "default")` (line 2990). For `cluster-init` the blueprint's `app-store-cluster-init.yaml` has a hard-coded `metadata.namespace` (e.g. `default` or `weka-app-store`) that may differ from the namespace the user provides. The CR was applied with `ns_for_apply=""` (line 3075), meaning the operator uses the namespace declared in the blueprint YAML, not the user's selection. The poll then queries the wrong namespace if the user selected a different one.

This is a latent bug surfaced by the change: the operator writes the CR to its declared namespace, but the poll reads from the user-supplied namespace.

**Fix:** After applying the blueprint, extract the actual namespace the CR was applied to from the rendered docs instead of using `user_vars.get("namespace")` for polling. Alternatively, for NAMESPACE_PRESERVING_APPS, default the poll namespace to the namespace found in the rendered CR's `metadata.namespace`.

```python
# After the docs loop that sets cr_name (around line 3068), also capture cr_namespace:
cr_namespace = None
for d in docs:
    if isinstance(d, dict) and d.get("kind") == "WekaAppStore":
        md = d.setdefault("metadata", {})
        cr_name = cr_name or md.get("name")
        cr_namespace = cr_namespace or md.get("namespace")

# Use cr_namespace for polling when set (namespace-preserving apps):
poll_namespace = cr_namespace if (app_name in NAMESPACE_PRESERVING_APPS and cr_namespace) else namespace
# Then use poll_namespace at line 3103
```

---

### WR-03: `handleInitialize` (legacy right-panel Retry) is now a dead/confusing code path — it deploys `cluster-init` directly, bypassing the full wizard flow and the `app-store-install` prerequisite

**File:** `app-store-gui/webapp/templates/welcome.html:491`
**Issue:** The old `handleInitialize` function (line 491) is retained as the Retry handler for the right-panel `failed` state (line 995). This function opens `/deploy-stream?app_name=cluster-init&...` directly, skipping the `app-store-install` wizard flow that is now the primary entry point. If the right-panel shows `failed=true` (set by the old polling interval or by a deploy-stream error), clicking Retry will attempt to re-run cluster-init without re-running app-store-install, potentially against wrong state.

Additionally, `handleInitialize` never sets `progressActive=true`, so if it's called when `progressActive` is already true (which it always will be after `handleInstall` runs), the stage list from the wizard's `progressActive` panel is still visible while `handleInitialize` updates the separate `logs`/`phase` right-panel state — confusing mixed state.

**Fix:** Replace the `handleInitialize` Retry button with a call to `handleInstall()` (or add a "Restart from beginning" page reload) for consistency with the new wizard flow:
```jsx
// Line 995: replace onClick={handleInitialize} with:
onClick={() => { window.location.reload(); }}
// OR, if the CRs are truly idempotent:
onClick={() => { setFailed(false); setInstallError(null); handleInstall(); }}
```

## Info

### IN-01: `versionRe` in `validateStep` accepts `v5.1.0.605` (4-segment) but step-2 helper text says "e.g. 5.1.0.605"

**File:** `app-store-gui/webapp/templates/welcome.html:330`
**Issue:** The version regex `/^v?\d+\.\d+(\.\d+)*$/` correctly accepts 4-segment versions like `5.1.0.605`. The helper text in step 2 (line 664) shows `e.g. 5.1.0.605`, which is consistent. However the step-2 validation error message says "Use a version like v5.1.0.605 or 5.1.0.605" (line 357) while the step-1 operator version error (line 339) says "Use a version like v1.13.0 or 1.13.0" — identical phrasing but different example versions. This is a minor inconsistency; not a bug.

**Fix:** No code change required; informational only.

---

### IN-02: `handleInitialize` no longer checks prerequisites (`wekaOperatorInstalled`, `wekaCsiInstalled`) before deploying

**File:** `app-store-gui/webapp/templates/welcome.html:491`
**Issue:** The previous implementation gated `handleInitialize` on `wekaOperatorInstalled === true && wekaCsiInstalled === true`. The new version removes that guard entirely (per the diff). The new wizard flow is intended to install those prerequisites itself, so removing the gate is correct for `handleInstall`. However, `handleInitialize` is still reachable as a Retry from the right panel (line 995) without any prerequisite check, meaning a user who somehow triggers the old code path can attempt `cluster-init` before the WEKA operator is installed.

Since this is a legacy path (see WR-03), the real fix is addressed there. Noted here for completeness.

**Fix:** Addressed by WR-03 fix. If `handleInitialize` is retained independently, restore a prerequisite guard.

---

_Reviewed: 2026-06-25_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
