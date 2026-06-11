---
phase: 23
plan: "01"
subsystem: app-store-gui
tags: [removal, api-cleanup, settings-ui, fastapi, jinja2]
dependency_graph:
  requires: []
  provides: [clean-baseline-for-plan-02, create_or_update_secret-helper]
  affects: [app-store-gui/webapp/main.py, app-store-gui/webapp/templates/settings.html]
tech_stack:
  added: []
  patterns: [surgical-removal, inline-localStorage-fallback]
key_files:
  created: []
  modified:
    - app-store-gui/webapp/main.py
    - app-store-gui/webapp/templates/settings.html
decisions:
  - "Removed /api/secret/huggingface and /api/secret/nvidia handlers entirely (no deprecation shim) per D-13; 404 is the correct post-removal behavior"
  - "Replaced getSettingsNamespace() call in loadBlueprints() with inline localStorage.getItem('selectedNamespace') || 'default' per D-14 to avoid ReferenceError"
metrics:
  duration: "160s"
  completed: "2026-06-11"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 23 Plan 01: Deprecated API and UI Removal Summary

Removal of the deprecated `/api/secret/huggingface` and `/api/secret/nvidia` FastAPI route handlers from `main.py` and all corresponding HTML/JS from `settings.html`. This clears the legacy single-key API surface so Plan 02 can introduce the new `/api/credentials` endpoints on a clean baseline.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Remove /api/secret/huggingface and /api/secret/nvidia handlers from main.py | d348839 | app-store-gui/webapp/main.py |
| 2 | Strip deprecated HTML sections and JS from settings.html and rewire loadBlueprints namespace fallback | 9f01963 | app-store-gui/webapp/templates/settings.html |

## What Was Removed

### From `app-store-gui/webapp/main.py`

| Handler | Lines Removed | Description |
|---------|---------------|-------------|
| `@app.post("/api/secret/huggingface")` + `save_huggingface_key` | 558–570 | Form-based HuggingFace API key save endpoint |
| `@app.post("/api/secret/nvidia")` + `save_nvidia_key` | 626–638 | Form-based NVIDIA API key save endpoint |

Net: 30 lines deleted. `create_or_update_secret()` helper at line 535 was retained (reused by Plan 02).

### From `app-store-gui/webapp/templates/settings.html`

| Item | Original Lines | Description |
|------|----------------|-------------|
| Section 1: HuggingFace API Key `<section>` card | 52–81 | HTML with DOM IDs: hf-api-key, hf-namespace, hf-save, hf-result, hf-secrets-list, hf-listing-scope |
| Section 2: NVIDIA API Key `<section>` card | 83–112 | HTML with DOM IDs: nvidia-api-key, nvidia-namespace, nvidia-save, nvidia-result, nvidia-secrets-list, nvidia-listing-scope |
| `getSettingsNamespace()` | 321–328 | Read hf-namespace/nvidia-namespace DOM elements |
| `setSettingsNamespace(ns)` | 330–341 | Wrote to removed DOM elements and localStorage |
| `renderSecrets(containerId, items)` | 343–360 | Rendered into removed hf-secrets-list/nvidia-secrets-list containers |
| `async function loadSecrets(ns)` | 362–383 | Called renderSecrets and getSettingsNamespace |
| hfBtn click handler block | 386–412 | Posted to /api/secret/huggingface |
| nvBtn click handler block | 414–440 | Posted to /api/secret/nvidia |
| Standalone `loadSecrets()` call | 485 | Initial call to removed function |

Net: 185 lines deleted, 1 line inserted (localStorage inline fallback).

### Edit Applied

Inside `async function loadBlueprints()` (line 546 original):

Before: `const namespace = (scope === 'current') ? getSettingsNamespace() : 'all';`
After: `const namespace = (scope === 'current') ? (localStorage.getItem('selectedNamespace') || 'default') : 'all';`

## Surviving Section Order in settings.html `<main>`

1. Kubernetes Auth Status (was Section 3)
2. Cluster Status (was Section 4)
3. Blueprint Uninstall (was Section 5)
4. Debug (was Section 6)

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

All acceptance criteria met:

- `python -m py_compile app-store-gui/webapp/main.py` exits 0
- `grep -n '/api/secret/huggingface' app-store-gui/webapp/main.py` returns no match
- `grep -n '/api/secret/nvidia' app-store-gui/webapp/main.py` returns no match
- `grep -n 'def save_huggingface_key' app-store-gui/webapp/main.py` returns no match
- `grep -n 'def save_nvidia_key' app-store-gui/webapp/main.py` returns no match
- `grep -n 'def create_or_update_secret' app-store-gui/webapp/main.py` returns exactly one match
- `grep -n '@app.get("/api/blueprints")' app-store-gui/webapp/main.py` returns exactly one match
- `grep -n '@app.get("/api/secrets")' app-store-gui/webapp/main.py` returns exactly one match
- Template passes Jinja2 parse check
- No removed DOM IDs or function names remain in settings.html
- `localStorage.getItem('selectedNamespace') || 'default'` inline fallback present exactly once
- `refreshAuthStatus()` and `setInterval(refreshAuthStatus, 10000)` retained
- Section order: Kubernetes Auth Status, Cluster Status, Blueprint Uninstall, Debug

## Known Stubs

None — this plan is removal-only; no new stubs were introduced.

## Threat Flags

None — this plan removes API surface and HTML/JS; no new network endpoints, auth paths, or file access patterns were introduced.

## Self-Check: PASSED

- [x] `app-store-gui/webapp/main.py` exists and compiles
- [x] `app-store-gui/webapp/templates/settings.html` exists and passes Jinja2 parse
- [x] Commit d348839 exists (Task 1)
- [x] Commit 9f01963 exists (Task 2)
