# Phase 28: Operator Helm Auth & CRD Discovery - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-24
**Phase:** 28-operator-helm-auth-crd-discovery
**Mode:** assumptions
**Areas analyzed:** Auth Mechanism, Auth Injection Point, discover_chart_crds Cache Fix, Scope

## Assumptions Presented

### Auth Mechanism (OPA-01)
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Use `--registry-config <tempfile>` (not `helm registry login`) | Confident | `_install_chart` line 136 — process args visible; persistent `~/.config/helm/registry/config.json` creates shared state across concurrent reconciles |
| Credential source is `stack_vars["quay_dockerconfigjson"]` | Confident | `stack_vars` line 1042 merges `appStack.variables`; Phase 27 D-06 includes `quay_dockerconfigjson` in x-variables block |
| Only `handle_appstack_deployment` path is touched | Confident | Install blueprint uses `appStack.components[]`; `handle_helm_deployment` fires on `spec.helmChart` only |

### Auth Injection Point
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| New `registry_config_path` param threads through `discover_chart_crds` → `should_skip_crds_for_component` → `install_or_upgrade` | Likely | Both `should_skip_crds_for_component` (line 1161) and `install_or_upgrade` (line 1169) run within the same component block; temp file must be live for both |

### discover_chart_crds Cache Fix (OPA-02)
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Replace `@lru_cache` with manual dict; only cache on subprocess success | Confident | `@lru_cache` at line 673 caches ALL returns; `CalledProcessError` at line 685 returns `set()` which gets memoized; retry hits cache → WekaClient 404 |
| Cache key includes `registry_config_path` | Confident | Auth-failed call (no registry config) must not pollute cache entry used by later auth'd call |

### Scope
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `operator_module/main.py` only, no new files | Confident | Single-file operator design per CLAUDE.md; `tempfile` already used in `_install_chart` |

## Corrections Made

No corrections — all assumptions confirmed.

## External Research

Not performed — codebase provided sufficient evidence for all assumptions.
