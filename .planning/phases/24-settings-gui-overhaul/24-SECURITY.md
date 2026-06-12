---
phase: 24
slug: settings-gui-overhaul
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-12
---

# Phase 24 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| client → /settings | Browser receives Jinja2-rendered HTML with credential displayName and error strings interpolated server-side | WarpCredential metadata (displayName, type, ready state) |
| /settings → Kubernetes API | settings_page() calls list_namespaced_custom_object; attacker-controlled WarpCredential metadata.name or spec.displayName controls rendered content | Credential names, display names, ready status |
| Browser DOM ↔ /api/credentials* | API-sourced strings (displayName, error) interpolated into DOM; credential values (API keys, tokens) submitted via POST FormData | Credential secrets (POST only), credential metadata |
| Browser ↔ /api/weka/overview | Endpoint proxies WEKA REST API; filesystem names, backend IPs, and error messages from WEKA cluster rendered into DOM | WEKA cluster topology data, filesystem utilisation |
| Browser ↔ weka_storage_credentials Jinja2 emit | Credential displayName and endpoint emitted via `\| tojson` into JS literal | Credential display names, endpoint URLs |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-24-01-01 | Tampering / XSS via server-side interpolation | settings.html — server-rendered credential displayName | mitigate | Jinja2 auto-escape on; no `\|safe` filter introduced; `grep -c '\|safe'` = 0 confirmed | closed |
| T-24-01-02 | Information disclosure | settings_page() — Jinja2 context | mitigate | `_build_credential_response_item()` whitelist reused verbatim; raw key values never included in context | closed |
| T-24-01-03 | Denial of service — K8s API outage | settings_page() / _fetch_credentials() | mitigate | `except (ApiException, ConnectionError, TimeoutError)` → empty list; /settings always returns HTTP 200 | closed |
| T-24-01-04 | XSS via future JS DOM injection (deferred) | Future Plan 02/03 JS context | accept | Plan adds zero JS; Plans 02 and 03 own their esc() usage (see T-24-02-01, T-24-03-01..04) | closed |
| T-24-01-05 | CSRF on /settings | settings_page() | accept | GET-only route; no state-changing side effects | closed |
| T-24-02-01 | XSS via DOM injection | renderCredentialRow — cred.displayName, cred.error, cred.name interpolation | mitigate | All cred.* interpolations wrapped in esc(); static grep gate confirms 0 unescaped innerHTML template-literal interpolations | closed |
| T-24-02-02 | XSS via operator-injected cred.error | renderCredentialRow — red-state branch | mitigate | esc() on both innerHTML and title attribute; kopf status condition message neutralised at DOM-write boundary | closed |
| T-24-02-03 | Credential exposure via DOM | Inline add form `<input type="password">` | mitigate | Password inputs cleared via closeAllAddForms() on Save success; API response never echoes credential values | closed |
| T-24-02-04 | Credential exposure via URL | submitAddForm POST path | mitigate | FormData + POST; no credential value appears in URL or query string; GET /api/credentials carries namespace only | closed |
| T-24-02-05 | CSRF on POST/DELETE credential routes | submitAddForm + wireDeleteButton | accept | Same-origin context inside operator pod; FastAPI app does not advertise CORS; no external origin can target /api/credentials* | closed |
| T-24-02-06 | DoS via runaway polling | pollIntervals Map | mitigate | Per-row only; visibilitychange clears intervals (IDs nulled in map); beforeunload guards against null; 30-second per-row timeout | closed |
| T-24-02-07 | Race: delete pressed during amber polling | wireDeleteButton × startCredentialPoll | mitigate | Amber row renders no Delete button — race structurally prevented by DOM contract | closed |
| T-24-02-08 | Open-redirect via endpoint field | URL input validation | accept | UX-only checkValidity(); endpoint consumed by operator (Phase 22) and WEKA proxy (Phase 23) which apply _validate_weka_endpoint; no new attack surface | closed |
| T-24-03-01 | XSS via filesystem name | renderWekaSuccess — filesystem table rows | mitigate | Every `${fs.name}` interpolation wrapped in esc(); static grep gate passes | closed |
| T-24-03-02 | XSS via backend IP string | renderWekaSuccess — backend IP grid cells | mitigate | Every `${node.ip}` interpolation wrapped in esc(); static grep gate passes | closed |
| T-24-03-03 | XSS via WEKA API error message | renderWekaError banner | mitigate | `${message}` wrapped in esc() before innerHTML insertion | closed |
| T-24-03-04 | XSS via credential displayName in credential select | renderWekaControls — option value and label | mitigate | esc() on both option value and label; value also passed through encodeURIComponent in loadWekaOverview | closed |
| T-24-03-05 | SSRF via user-controlled credential slug | loadWekaOverview — ?credential= query parameter | accept | Backend _CREDENTIAL_NAME_RE.match() validates slug before lookup; arbitrary URL probing not possible via this parameter | closed |
| T-24-03-06 | Stale data after error state | WEKA overview state machine | mitigate | setWekaState('error') hides #weka-overview-success div; next successful fetch replaces innerHTML wholesale | closed |
| T-24-03-07 | Excessive cache-bust load via Refresh button | loadWekaOverview bust=1 | accept | Single-admin use; realistic click rate well within WEKA cluster capacity; server-side rate-limiting deferred to future phase if needed | closed |
| T-24-03-08 | Clock-skew negative "Last updated" timestamp | formatRelativeTime | mitigate | Math.max(0, elapsed) floors at 0; future-dated timestamps render as "Just now" | closed |
| T-24-03-09 | DoS via giant filesystems array | renderWekaSuccess — filesystem table | mitigate | Initial render capped at 20 rows; "Show all" is opt-in DOM expansion, not a fetch | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-24-01 | T-24-01-04 | Plan 01 adds zero JS; XSS defence deferred to Plans 02/03 which own esc() usage | gsd-security-auditor | 2026-06-12 |
| AR-24-02 | T-24-01-05 | /settings is GET-only; CSRF protection on POST/DELETE routes is Phase 23's responsibility (already shipped) | gsd-security-auditor | 2026-06-12 |
| AR-24-03 | T-24-02-05 | No CORS configured; same-origin context; if future CORS requirement added, per-route CSRF token needed | gsd-security-auditor | 2026-06-12 |
| AR-24-04 | T-24-02-08 | Endpoint URL validation is UX-only in the browser; SSRF defence delegated to _validate_weka_endpoint (Phase 23, now patched via CR-03 fix to cover RFC-1918 ranges) | gsd-security-auditor | 2026-06-12 |
| AR-24-05 | T-24-03-05 | SSRF via credential slug not possible — slug validated by _CREDENTIAL_NAME_RE before endpoint lookup | gsd-security-auditor | 2026-06-12 |
| AR-24-06 | T-24-03-07 | Refresh cache-bust rate not restricted; single-admin use pattern, WEKA cluster tolerates it | gsd-security-auditor | 2026-06-12 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-12 | 22 | 22 | 0 | gsd-secure-phase (plan-time register, register_authored_at_plan_time: true) |

**Note:** Code review (gsd-code-review) identified CR-03 — incomplete SSRF guard in Phase 23's `_validate_weka_endpoint` (only blocked loopback/link-local, not RFC-1918). Fixed in the same session via `fix(24): CR-03 complete SSRF guard using ipaddress.is_private`. T-24-02-08 and T-24-03-05 accepted-risk dispositions remain valid: the fix strengthens the backend guard they rely on.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter
