# Phase 23: Backend Credentials API and WEKA Overview Proxy - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-11
**Phase:** 23-backend-credentials-api-and-weka-overview-proxy
**Mode:** assumptions
**Areas analyzed:** Kubernetes Client, WEKA HTTP Client + Caching, WEKA Credential Secret Structure, Slug Generation, Settings Page JS Removal, Test Approach

## Assumptions Presented

### Kubernetes Client for Credentials API
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Use `CustomObjectsApi` for WarpCredential CRs; `CoreV1Api` for `warp-cred-<slug>` Secrets | Confident | `main.py:573` (`list_blueprints`), `main.py:535` (`create_or_update_secret`), `requirements.txt` |

### WEKA Overview HTTP Client and Caching Strategy
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `urllib.request` + `asyncio.to_thread`; module-level dict cache with `time.time()` TTL | Likely | `httpx` absent from `requirements.txt`; `urllib.request` at `main.py:1480`; `asyncio.to_thread` at `main.py:495`; cache pattern at `main.py:1016-1065` |

### WEKA Credential Secret Structure
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| `warp-cred-<slug>` for weka-storage stores `WEKA_API_USERNAME`, `WEKA_API_TOKEN`, `WEKA_API_ENDPOINT`; `secretRef.key` → `WEKA_API_TOKEN` | Confident | CRD `secretRef.key` single string; PRD lines 231-234 specify all three keys required |

### Slug Generation for metadata.name
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| lowercase, non-alphanumeric→hyphens, max 52 chars, collision → append `-2`, `-3` | Confident | PRD line 231 specifies this; K8s 63-char name limit |

### Settings Page JavaScript Removal
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Remove `hfBtn` handler, `nvBtn` handler, old secret list JS block, and corresponding HTML sections | Confident | `settings.html:401` posts to `/api/secret/huggingface`; `settings.html:429` posts to `/api/secret/nvidia` |

### Test Approach
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Plain pytest with stub objects, no FastAPI `TestClient` | Likely | `app-store-gui/tests/planning/test_apply_gateway.py` stub pattern; no existing route-level integration tests |

## Corrections Made

No corrections — all assumptions confirmed by user.

## External Research

Two topics flagged as needing external research (researcher must resolve before planning implementation):

1. **WEKA REST API field names** — exact response field names for `/api/v2/fileSystems`, `/api/v2/cluster`, `/api/v2/containers`; codebase has no prior WEKA REST API integration to infer from; PRD references live Swagger UI at `https://<cluster>:14000/api/v2/docs`

2. **WEKA auth token type** — whether `WEKA_API_TOKEN` is a long-lived static Bearer token (use directly) or a refresh token (requires `POST /api/v2/login/refresh` first); depends on WEKA 4.x version deployed
