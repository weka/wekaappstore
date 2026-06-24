# Phase 29: Backend Wiring & Secret Safety - Context

**Gathered:** 2026-06-24 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire `app-store-gui/webapp/main.py` so the backend supports the `cluster_init/app-store-install.yaml` blueprint authored in Phase 27. Four work areas, each tied to a ROADMAP success criterion: (1) locate the blueprint and preserve component namespaces, (2) derive `quay_dockerconfigjson` and the split-endpoint vars server-side before the Jinja2 render pass, (3) raise the SSE deploy deadline per-blueprint so a long install does not false-fail, (4) keep quay/WEKA secret values out of the `warp.io/gui-variables` CR annotation and the SSE message stream. Scope is the GUI backend module plus its pytest suite — no operator changes, no blueprint authoring, no frontend wizard work.
</domain>

<decisions>
## Implementation Decisions

### Blueprint Location & Namespace Preservation (SC1)

- **D-01:** `find_blueprint("app-store-install")` resolves via the existing generic `os.walk` + stem-match path (`main.py:1811-1827`) — no new location/special-case code is needed for *finding* the file. The authored `cluster_init/app-store-install.yaml` (stem `app-store-install`, carries `x-variables`) is found by the generic walk.

- **D-02:** Extend the existing `cluster-init` namespace-preserve special-case so `app-store-install` is treated identically — `ns_for_apply = ""` so the components' fixed `targetNamespace` values and the CR's `metadata.namespace: default` are NOT overwritten by the user-selected namespace. Implement via a module-level set `NAMESPACE_PRESERVING_APPS = {"cluster-init", "app-store-install"}` as the single source of truth, checked at every site that currently special-cases `app_name == "cluster-init"`: `deploy_stream` (`main.py:~2943` and `~2948`), the `apply_gateway` namespace-override skip, and the required-field-validation exemption (`main.py:~2874`). Audit for all `== "cluster-init"` comparisons and route each through the set.

### Server-Side Variable Derivation (SC2)

- **D-03:** Add two pure module-level functions near `parse_x_variables` (`main.py:~1694`): `build_quay_dockerconfigjson(user, password)` and `split_endpoints(join_ip_ports)`. Their outputs are merged into `user_vars` **before** `template.render(**user_vars)` (`main.py:~2920`) — matching Phase 27 D-06 ("injected as extra render vars before the Jinja2 pass"). The derived vars are NOT in the blueprint's `x-variables` block.

- **D-04:** `build_quay_dockerconfigjson` uses `base64.b64encode(f"{user}:{password}".encode()).decode("ascii")` — explicitly NOT `base64.encodebytes`/`encodestring` (which insert `\n`). It wraps the result as `json.dumps({"auths": {"quay.io": {"auth": <b64>}}}, separators=(",", ":"))`. Result: `auths["quay.io"]["auth"]` base64-decodes to exactly `user:pass` with no trailing bytes. This matches the docker config.json shape Phase 28 D-02 consumes via helm `--registry-config`.

- **D-05:** `split_endpoints(join_ip_ports)` takes the user's comma-delimited `host:port` string and returns a dict `{"join_ip_ports_list": [...], "endpoints_csv": "..."}` so the caller can `user_vars.update(...)` in one line. `join_ip_ports_list` is a real Python list (rendered into the WekaClient CR `joinIpPorts` YAML array at `app-store-install.yaml:~291`); `endpoints_csv` is the comma-joined string (CSI API secret at `~256`). Each entry is whitespace-trimmed; empty entries dropped. The planner must confirm the Jinja2 `[[ join_ip_ports_list ]]` render produces valid YAML flow-sequence — if Python `repr` (single quotes) is risky, render an explicit double-quoted JSON array string instead.

- **D-06:** Both `build_quay_dockerconfigjson` and `split_endpoints` get unit tests in `app-store-gui/tests/` (import `webapp.main`, follow the existing harness). Tests must assert: the quay `auth` decodes to exactly `user:pass` with no trailing newline/bytes; `split_endpoints` produces both the list form and the csv form from representative input (single endpoint, multiple endpoints, surrounding whitespace). Run with `PYTHONPATH=mcp-server:app-store-gui pytest app-store-gui/tests/ -v`.

### SSE Deadline + Keepalive/Reconnect (SC3, PROG-02)

- **D-07:** Reuse the existing keepalive as-is — `: ping` comment-event emitted every loop iteration (~every 2s per `asyncio.sleep(2)`, `main.py:~2963/2999`), already documented as robust against idle-proxy drops. Reconnect is left to the browser-native `EventSource` auto-reconnect; the server stays idempotent (apply re-creates/updates the same CR; polling re-derives state from `.status`). No new keepalive or reconnect machinery.

- **D-08:** Replace the hardcoded `deadline = time.time() + 900` (`main.py:~2956`) with a per-blueprint override read from an `x-deploy-timeout` (seconds) key at the blueprint's top level (sibling to `x-variables`), parsed alongside `parse_x_variables`. Keeps the blueprint as the declarative source of truth (consistent with the `x-variables` precedent). Use a raised default (recommend 1800–2400s) when the key is absent. Phase 27's blueprint should carry an `x-deploy-timeout` sized for the full operator+CSI+WekaClient install. This is a flat raised cap (not progress-reset) — matches the success criterion.

### Secret Safety — Annotation Allowlist + SSE Redaction (SC4, SEC-01)

- **D-09:** Define ONE secret-key predicate shared by both redaction points: a key is secret if it matches `*password*`/`*token*`/`*secret*` (case-insensitive substring) OR equals exactly `quay_dockerconfigjson`. Apply it at the annotation stamp (`main.py:~2935`): replace `json.dumps(user_vars, ...)` with `json.dumps(_safe_gui_variables(user_vars), ...)` where `_safe_gui_variables` drops every secret key. The downstream read helper (`main.py:~1726-1742`) already tolerates a partial dict and falls back to `spec.appStack.variables`, so dropping keys does not break the read path.

- **D-10:** Add SSE message redaction: before emitting any `{"type":"component", ... "message": ...}` event (`main.py:~2984`), pass `comp.get("message")` through a redactor that replaces occurrences of the actual secret VALUES (the user-typed `weka_password`, the posted quay password, and the derived `quay_dockerconfigjson`) with `***`. Build the redactor's value-set from the same secret keys identified by the D-09 predicate so the annotation allowlist and the SSE redactor share one definition.

### Claude's Discretion

- Exact `split_endpoints` return mechanism (dict vs tuple) — dict recommended for one-line merge; either is acceptable as long as both forms are produced and tested.
- Whether `_safe_gui_variables` and the SSE redactor live as two small helpers or one combined helper module-level function — single-file GUI design preserved either way.
- The exact default value for the absent-`x-deploy-timeout` fallback (within the 1800–2400s band) and whether to log (DEBUG) when a per-blueprint timeout is applied.
- Whether the SSE redactor matches secret values literally vs. also redacting their base64 forms (literal value match is the floor; base64 form is a nice-to-have).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `.planning/REQUIREMENTS.md` — PROG-02 (SSE deadline + keepalive/reconnect), SEC-01 (annotation allowlist + SSE redaction), E2E-03 (no secrets in logs/stream)
- `.planning/ROADMAP.md` — Phase 29 section, 4 success criteria
- `.planning/PRD-install-wizard-weka-storage-stack.md` — authoritative spec
- `app-store-gui/webapp/main.py` — the file being wired:
  - `parse_x_variables` (~1694) — top-level blueprint key parser; `x-deploy-timeout` parsed alongside
  - gui-variables read helper (~1726-1742) — tolerant partial-dict read with `spec.appStack.variables` fallback
  - `find_blueprint` (~1801, generic walk ~1811-1827) — stem-match location
  - `deploy_stream` endpoint (~2839+) — render at ~2920, annotation stamp ~2935, deadline ~2956/2996, keepalive ~2963/2999, SSE component message emit ~2980-2985, required-field exemption ~2874, ns_for_apply ~2943/2948
- `cluster_init/app-store-install.yaml` — Phase 27 blueprint: `x-variables`, fixed `targetNamespace` per component, `joinIpPorts`/`endpoints`/`.dockerconfigjson` token sites; add `x-deploy-timeout` here
- `cluster_init/app-store-cluster-init.yaml` — the namespace-preserve precedent (existing `cluster-init` special-case)
- `app-store-gui/tests/test_dynamic_blueprint.py` + `conftest.py` — live pytest harness (import `webapp.main`, `_collect_sse`/`_make_request_stub`/`_mock_appstack_ready` helpers)
- `.planning/phases/27-install-blueprint-authoring/27-CONTEXT.md` — D-06 (server-side derived vars contract)
- `.planning/phases/28-operator-helm-auth-crd-discovery/28-CONTEXT.md` — D-02 (operator consumes `quay_dockerconfigjson` shape)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- The `cluster-init` namespace-preserve special-case already exists at `deploy_stream` (~2943) and in `apply_gateway`'s namespace-override skip — D-02 generalizes it to a set rather than inventing new logic.
- The SSE keepalive (`: ping` comment-event) is already implemented and documented (~2960-2963) — reuse, do not rebuild.
- The gui-variables read helper (~1726-1742) already tolerates a partial dict and falls back to `spec.appStack.variables` — so D-09's key-dropping is safe on the read side.
- `parse_x_variables` (~1694) is the established pattern for reading a top-level blueprint key — `x-deploy-timeout` parsing follows it.
- Live pytest harness in `app-store-gui/tests/` (`conftest.py` puts `app-store-gui/` on `sys.path`; tests import `webapp.main` and call functions directly).

### Established Patterns

- `[[ var ]]` Jinja2 delimiters are the GUI render layer; derived vars must be merged into `user_vars` before `template.render` (~2920).
- GUI module uses double quotes (per CONVENTIONS.md); single-file design — add module-level helpers, no new sub-packages.
- Component SSE events are dicts emitted as `data: {json}\n\n`; the message field is operator-supplied and may echo manifest content.

### Integration Points

- The GUI-built `quay_dockerconfigjson` flows through `appStack.variables` to the operator (Phase 28 D-02) for helm `--registry-config` — the byte-exact `user:pass` encoding (D-04) is load-bearing for the OCI pull.
- `join_ip_ports_list` / `endpoints_csv` (D-05) render into the WekaClient CR and CSI API secret respectively — both forms must be correct or those components fail.
- The annotation written at ~2935 is read back by the gui-variables helper and is persisted in etcd / visible via `kubectl get wekaappstore -o yaml` — hence the SEC-01 allowlist.

### Housekeeping Flag

- `.planning/codebase/TESTING.md` is stale (claims no Python unit tests). The authoritative test harness is the live `app-store-gui/tests/` pytest suite. New tests go there.
</code_context>

<specifics>
## Specific Ideas

- Quay `auth` byte-exactness is the single highest-risk detail: `base64.b64encode(f"{user}:{password}".encode()).decode("ascii")`, never `encodebytes`. The unit test must base64-decode the rendered `auth` and assert it equals exactly `b"user:pass"` (no trailing `\n`).
- The annotation allowlist and the SSE redactor must share one secret-key definition (D-09 predicate) — do not define the secret set twice.
- Phase 27's blueprint should gain an `x-deploy-timeout` key; the GUI reads it with a raised default when absent. The install spans ~5 sequential stages each with `timeout: 300` readinessChecks, so the legacy 900s cap is too tight.
</specifics>

<deferred>
## Deferred Ideas

- Progress-aware deadline extension (reset the deadline on observed component phase change) — deferred; a flat raised per-blueprint cap satisfies SC3. Note for a future phase if long installs still false-fail.
- Redacting base64-encoded forms of secrets in SSE messages (beyond literal value match) — D-10 floor is literal value match; the base64-form match is a nice-to-have, not required.
- Idempotency of re-running the wizard (SEC-02) — that is a separate requirement/phase; not in Phase 29 scope.

None of the analysis introduced scope beyond the four ROADMAP success criteria.
</deferred>
