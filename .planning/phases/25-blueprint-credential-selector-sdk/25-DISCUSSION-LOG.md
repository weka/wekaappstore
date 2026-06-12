# Phase 25: Blueprint Credential Selector SDK - Discussion Log (Assumptions Mode)

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-06-12
**Phase:** 25-blueprint-credential-selector-sdk
**Mode:** assumptions
**Areas analyzed:** Shared Helper Extraction, Namespace Resolution for Blueprint Route, Macro File and Jinja2 Import Mechanism, Scope of Template Macro Usage

## Assumptions Presented

### Shared Helper Extraction
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Extract `_fetch_credentials` into module-level `_get_credentials_by_type(ns)` | Likely | `main.py:530-549` inline block, SDK-05 requires same logic in `blueprint_detail` |

### Namespace Resolution for Blueprint Route
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Add `get_auth_status()` call to `blueprint_detail` for namespace resolution | Likely | `settings_page:524-528` pattern; without it, non-default namespace users always see empty credentials |

### Macro File and Jinja2 Import Mechanism
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Use `{% from '_credential_macros.html' import ... %}` not `{% include %}` | Confident | Jinja2 language rule — `{% include %}` cannot expose callable macros |

### Scope of Template Macro Usage
| Assumption | Confidence | Evidence |
|------------|-----------|----------|
| Context injection universal (all routes); macro calls deferred to template authors | Unclear → confirmed | `blueprint_glocomp-aurora.html` and `blueprint_tokenvisor-enterprise.html` have no credential inputs; user confirmed macros are the SDK, template updates are future work |

## Corrections Made

No corrections — all assumptions confirmed.

User added context: "this macro with Jinja2 will allow me to modify existing blueprints to take on this functionality you are building" — confirms that Phase 25 delivers the SDK infrastructure; per-template macro calls are authored later.

## External Research

No external research required — all evidence from codebase.
