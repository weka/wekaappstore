# Phase 29: Backend Wiring & Secret Safety - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-24
**Phase:** 29-backend-wiring-secret-safety
**Mode:** assumptions
**Areas analyzed:** Blueprint Location & Namespace Preservation, Server-Side Variable Derivation, SSE Deadline + Keepalive/Reconnect, Secret Safety (Annotation Allowlist + SSE Redaction)

## Assumptions Presented

### Area 1 — Blueprint Location + Namespace Preservation
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `find_blueprint` already resolves `app-store-install` via generic stem-match walk; no new location code | Confident | `main.py:1811-1827`, stem match `:1826` |
| Extend cluster-init namespace-preserve special-case to `app-store-install` via `NAMESPACE_PRESERVING_APPS` set; `ns_for_apply=""` | Confident | `main.py:2943` (`ns_for_apply` ternary), `apply_gateway` override skip, exemption `~2874`; blueprint pins targetNamespace per component |

### Area 2 — Server-Side Variable Derivation
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Two pure functions near `parse_x_variables`; merge into `user_vars` before render | Confident | `main.py:1694`, render `:2920`; Phase 27 D-06 |
| `build_quay_dockerconfigjson` uses `b64encode` (not `encodebytes`); docker config.json shape | Confident | Phase 28 D-02; quay 401 risk on trailing `\n` |
| `split_endpoints` returns both list and csv forms | Confident | blueprint `:291` (joinIpPorts), `:256` (endpoints) |
| Both unit-tested in `app-store-gui/tests/` | Confident | live harness `test_dynamic_blueprint.py`, `conftest.py:11-14` |

### Area 3 — SSE Deadline + Keepalive/Reconnect
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Reuse existing `: ping` keepalive + native EventSource reconnect as-is | Likely | `main.py:2960-2963/2999` |
| Replace hardcoded `deadline = +900` with per-blueprint override | Likely | `main.py:2956/2996`; ROADMAP SC3; ~5 sequential 300s readinessChecks |

### Area 4 — Secret Safety
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| One shared secret predicate (`*password*`/`*token*`/`*secret*`/`quay_dockerconfigjson`) | Confident | REQUIREMENTS.md:48 |
| Annotation allowlist via `_safe_gui_variables` at stamp site | Confident | `main.py:2935`; read helper tolerant `:1726-1742` |
| SSE message redaction before component emit | Confident | `main.py:2980-2985`; REQUIREMENTS.md:55 (E2E-03) |

## Corrections Made

No assumptions were corrected. The user confirmed all four areas as presented.

The one open HOW (Area 3 — per-blueprint deadline mechanism) was decided in favor of a **blueprint-declared `x-deploy-timeout` key** (over a code-level `DEPLOY_DEADLINES` dict), keeping the blueprint as the declarative source of truth consistent with the `x-variables` precedent. Captured as D-08.

## External Research

None performed — analyzer flagged no gaps (all internal wiring; codebase + Phase 27/28 context sufficient).
