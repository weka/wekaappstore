# Phase 30: Wizard Stepper & Live Progress - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-25
**Phase:** 30-wizard-stepper-live-progress
**Mode:** assumptions
**Areas analyzed:** Multi-Step Form Architecture, Live Progress Display, Form Submit Flow, Cluster-Init Chain & Redirect

## Assumptions Presented

### Multi-Step Form Architecture

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Wizard extends existing `WelcomeApp` React+MUI Babel component in-place; no new Python routes | Confident | `welcome.html:18-19` (MUI CDN + Babel), `main.py:2538` (single `/welcome` route); MUI Stepper already available |
| Wizard state in `useState` hooks; secrets never persisted to `localStorage` | Confident | `welcome.html:135-137` (`localStorage` used only for selectedNamespace); Phase 29 D-09..D-10 secret safety |

### Live Progress Display

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Reuse `blueprint.html` SSE consumer pattern exactly (EventSource, init/component/complete/error) | Confident | `blueprint.html:283-346` (complete SSE consumer); `main.py:3105-3117` (componentStatus events); `main.py:191` (NAMESPACE_PRESERVING_APPS) |
| Stage failure: detect `ok:false`/`type:error` client-side, show error + Retry button that re-opens EventSource | Likely | `main.py:3122-3127` (`complete` event with `ok:false`); `welcome.html:479` (existing Retry button); operator CR upsert is idempotent |

### Form Submit Flow

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Submit quay_username, quay_password, operator_version, join_ip_ports, weka_image_version, weka_endpoint_scheme, weka_org, weka_username, weka_password to existing `/deploy-stream`; quay_dockerconfigjson NOT a form field | Confident | `app-store-install.yaml:1-37` (x-variables schema); Phase 29 D-03 (server-side derivation locked); `main.py:3044-3048` |

### Cluster-Init Chain & Redirect

| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Client opens second EventSource to `/deploy-stream?app_name=cluster-init` after app-store-install `complete.ok:true` | Likely | `welcome.html:266-296` (existing pattern); Phase 27 D-01 (two-CR chain locked); no server-side chain mechanism exists |
| ClusterInitMiddleware already exempts `/deploy-stream`; no middleware changes needed | Confident | `main.py:43` (exempt_paths list includes `/deploy-stream`) |

## Corrections Made

No corrections — all assumptions confirmed by user.

## External Research

No external research required — codebase provided sufficient evidence for all four areas.
