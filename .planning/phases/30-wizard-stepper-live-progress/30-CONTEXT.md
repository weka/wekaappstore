# Phase 30: Wizard Stepper & Live Progress - Context

**Gathered:** 2026-06-25 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the current single-button prerequisite hard-block in `welcome.html` with a multi-step wizard form (node prerequisites → quay credentials → WEKA connection → WEKA credentials → review → live progress). The wizard submits to the existing `/deploy-stream` backend, displays per-stage install progress driven by `componentStatus` SSE events, chains to `cluster-init` after `app-store-install` reaches Ready, and redirects to the App Store when cluster-init reaches Ready. Scope is primarily `welcome.html` (React+MUI frontend) and any minimal new frontend routes if needed — no new Python backend routes, no operator changes, no blueprint changes — **plus one surgical backend fix** (D-13): the `/deploy-stream` generator must be corrected so `app-store-install` and `cluster-init` actually stream `component` events. This was discovered during planning (the original "frontend-only" assumption was wrong — see D-05/D-13).
</domain>

<decisions>
## Implementation Decisions

### Multi-Step Form Architecture

- **D-01:** The wizard lives entirely inside `welcome.html` as an extension of the existing `WelcomeApp` React+MUI Babel component. MUI `Stepper`/`Step`/`StepLabel` is already available (CDN-loaded `@mui/material@5.15.14`, line 18 of `welcome.html`) — no new dependencies are needed. The existing `@app.get("/welcome")` route at `main.py:2538` continues to serve the page; no new Python routes are added for wizard steps.

- **D-02:** All wizard state (current step index, per-step field values) lives in `React.useState` hooks inside the `WelcomeApp` component. Secret field values (`quay_password`, `weka_password`) are NEVER persisted to `localStorage` — in-memory only. `localStorage` is used only for `selectedNamespace` (matching the existing pattern) if a namespace selector is shown in the review step.

- **D-03:** Steps rendered in order: (1) Node Prerequisites, (2) Quay Credentials, (3) WEKA Connection, (4) WEKA Credentials, (5) Review, then the progress/install view. Steps 1–5 use `MUI Stepper` for visual navigation. The progress view replaces the stepper UI (full-width stage list). Navigation: Next/Back buttons between steps; Submit on Review triggers the install.

- **D-04:** Inline validation fires on Next/Submit click, not on blur — blocks forward navigation if required fields are empty or format-invalid (WIZ-07). Endpoint format: each `join_ip_ports` entry must match `host:port` (regex `^[^:]+:\d+$`). Version tags must match `^v?\d+\.\d+(\.\d+)?$`. Required fields: quay username+password, operator version, at least one endpoint, WEKA image version, WEKA username+password.

### Live Progress Display

- **D-05:** The per-stage install progress view reuses the `blueprint.html` SSE consumer pattern exactly — `EventSource` opened to `/deploy-stream?app_name=app-store-install&variables=<encoded>&namespace=default` (namespace is irrelevant for namespace-preserving apps but required by the endpoint). Handles `init` (populate stage list), `component` (update stage status), `complete` (success/fail branch), `error` (hard fail) — identical to `blueprint.html` lines 283–346.
  - **CORRECTION (found during planning):** The original assumption that `app-store-install` streams `component` events like `blueprint.html` was WRONG. `blueprint.html` only ever runs generic (non-namespace-preserving) blueprints, which reach the componentStatus poll loop. `app-store-install` and `cluster-init` are in `NAMESPACE_PRESERVING_APPS` (`main.py:191`) and the generator short-circuits to `complete` at `main.py:3080` **before** the poll loop — so today they emit `init` → `complete` with no `component` events. The frontend SSE consumer is still correct as written; the backend must be fixed (D-13) so the events it expects actually arrive. With D-13 applied, `complete` also fires only at `appStackPhase == Ready`/`Failed` (not immediately), so the success/fail/redirect branches become meaningful for both apps.

- **D-06:** Stage status display maps `component.phase` values from SSE events: Pending → grey, `Installing`/`Upgrading` → blue (in-progress), `Ready` → green (done), `Failed` → red (failed). The existing `sectionClass(phase)` function from `blueprint.html` or an equivalent is replicated in the wizard component.

- **D-07:** Stage failure (PROG-03): when the `complete` event has `ok: false`, or `type === 'error'`, show the failed stage name and `msg.message` inline in the progress view with a Retry button. Retry re-opens the same `EventSource` (re-submitting the same variables). No new backend endpoint — the operator's CR upsert path is idempotent (re-apply is non-destructive per Phase 27 D-09).

### Form Submit Flow

- **D-08:** The wizard Review step collects and submits exactly these fields as `variables` JSON to the existing `/deploy-stream` endpoint:

  | Wizard step | Field | x-variable key | Default |
  |-------------|-------|----------------|---------|
  | Quay Credentials | Quay username | `quay_username` | — |
  | Quay Credentials | Quay password (masked) | `quay_password` | — |
  | Quay Credentials | Operator version | `operator_version` | `v1.13.0` |
  | WEKA Connection | Endpoints (one or more `host:port`) | `join_ip_ports` | — |
  | WEKA Connection | Image version tag | `weka_image_version` | — |
  | WEKA Connection | Scheme dropdown | `weka_endpoint_scheme` | `http` |
  | WEKA Credentials | Organization | `weka_org` | `Root` |
  | WEKA Credentials | Username | `weka_username` | — |
  | WEKA Credentials | Password (masked) | `weka_password` | — |

  `quay_dockerconfigjson` is NOT a form field — it is derived server-side from `quay_username` + `quay_password` per Phase 29 D-03.

- **D-09:** The Review step (step 5) shows a masked summary: all password fields display `••••••••`; `quay_dockerconfigjson` is never shown. A namespace selector is shown (defaulting to `default`) for completeness — though `app-store-install` ignores it server-side (namespace-preserving). After review the user clicks "Install" to open the progress view and start the SSE stream.

### Cluster-Init Chain & Redirect

- **D-10:** After `/deploy-stream?app_name=app-store-install` emits `{type: "complete", ok: true}`, the client automatically opens a second `EventSource` to `/deploy-stream?app_name=cluster-init` (no `variables` parameter needed — cluster-init is namespace-preserving and takes no x-variables). The wizard transitions to a second progress section (or a second stage in the same progress list) labeled "Cluster Init".

- **D-11:** When the cluster-init `EventSource` emits `{type: "complete", ok: true}`, the client calls the existing `/cluster-status` endpoint (`main.py:2551`) to get `redirect_url`, then performs a client-side `window.location.href = redirect_url` redirect. This matches the existing redirect logic the current `welcome.html` already uses after cluster-init (lines 266-296 of `welcome.html`).

- **D-12:** The `ClusterInitMiddleware` exemption list at `main.py:43` already includes `/deploy-stream` and `/welcome` — no middleware changes needed for Phase 30.

### Backend SSE Fix (scope amendment — added during planning)

- **D-13:** The `/deploy-stream` generator at `main.py:3080` currently short-circuits to a single `complete` event for any app in `NAMESPACE_PRESERVING_APPS`, conflating two separate concerns: (a) "do not override the user-selected namespace" (`main.py:3075`, correct — these apps have fixed per-component `targetNamespace`) and (b) "do not poll `componentStatus`" (`main.py:3080`, the bug). Fix: change the line-3080 guard from `if not cr_name or app_name in NAMESPACE_PRESERVING_APPS:` to `if not cr_name:` so any appStack CR (including `app-store-install` and `cluster-init`, both of which ARE multi-component appStacks) reaches the componentStatus poll loop and streams per-stage `component` events. The namespace-override suppression at line 3075 is UNCHANGED. **Namespace-match requirement:** the poll loop queries the CR with `get_namespaced_custom_object(namespace=namespace, name=cr_name)`; for namespace-preserving apps applied with `ns_for_apply=""` the CR lives in its manifest-declared `metadata.namespace`. The executor must ensure the poll queries the namespace where the CR actually lives (both `app-store-install` and `app-store-cluster-init` declare `namespace`/land in `default`) so the lookup does not 404. This is the ONLY backend change in Phase 30 and is the precondition for PROG-01 / Success Criterion 4.

### Claude's Discretion

- Whether the two install phases (app-store-install + cluster-init) render as one combined stage list or two separate progress sections — either is acceptable, as long as per-stage granularity is visible for both.
- Exact MUI component variants (outlined vs standard inputs, step connector style) — match the existing `welcome.html` visual language.
- Whether defaulted fields (operator version, scheme, org) are shown as editable or as read-only display with an "Advanced" toggle — either acceptable; showing them as editable with pre-filled defaults is simpler.
- Whether the Node Prerequisites step (step 1) shows the KubeletConfiguration snippet as a `<pre>` block or MUI `<Code>` — match whatever looks consistent with existing `welcome.html` styling.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

- `.planning/REQUIREMENTS.md` — WIZ-01..WIZ-08 (wizard form steps + validation), PROG-01 (per-stage progress), PROG-03 (failure + retry), INST-10 (cluster-init chain + redirect)
- `.planning/ROADMAP.md` — Phase 30 section, 5 success criteria
- `.planning/PRD-install-wizard-weka-storage-stack.md` — authoritative spec; Decision A1 (node prereqs as copy-paste snippet + checkbox, never automated), Decision D (two chained CRs)
- `app-store-gui/webapp/templates/welcome.html` — file being replaced/extended; existing React+MUI Babel component, ClusterInitSSE pattern (lines 266-296), Retry button (line 479), existing `localStorage.selectedNamespace` usage
- `app-store-gui/webapp/templates/blueprint.html` — canonical SSE consumer pattern to replicate (EventSource, init/component/complete/error handling, sectionClass stage coloring; lines 283-346)
- `app-store-gui/webapp/main.py` — `@app.get("/welcome")` (~2538), `ClusterInitMiddleware` exempt_paths (~43), `/deploy-stream` endpoint (~2957), `/cluster-status` endpoint (~2551, redirect_url extraction ~2612-2634), `NAMESPACE_PRESERVING_APPS` (~191)
- `cluster_init/app-store-install.yaml` — x-variables schema (defines form fields + defaults + validate:false for quay_dockerconfigjson); lines 1-37
- `.planning/phases/29-backend-wiring-secret-safety/29-CONTEXT.md` — D-03..D-05 (server-side quay/endpoint derivation), D-08 (x-deploy-timeout), D-09..D-10 (secret redaction — why quay_dockerconfigjson never in form)
- `.planning/phases/27-install-blueprint-authoring/27-CONTEXT.md` — D-01 (two-CR chain order), D-06 (x-variables block definition)
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `welcome.html` existing `WelcomeApp` React+MUI component with Babel in-browser transform — extend rather than replace from scratch; the React/MUI boilerplate, CDN imports, and Jinja2 template wiring are already there.
- `blueprint.html` SSE consumer (lines 283-346) — copy this pattern directly for the progress view; event types (`init`, `component`, `complete`, `error`) and their payload shapes are battle-tested.
- `welcome.html` lines 266-296 — the existing ClusterInitSSE call pattern; Phase 30 turns this into the automated second-phase chain instead of a user-triggered button press.
- `/cluster-status` endpoint + redirect_url logic at `main.py:2551` — used as-is after cluster-init `complete` event.
- `sectionClass(phase)` color-mapping function from `blueprint.html` — reuse directly.

### Established Patterns

- React+MUI Babel (in-browser compile, no build step) is the established GUI frontend stack — do not introduce webpack/vite/npm; all new components must work with Babel's in-browser transform.
- `welcome.html` uses double-quoted JSX and functional components with hooks — match this style.
- `localStorage.selectedNamespace` is the only client-side persistence used in the GUI — secrets stay in-memory.
- The SSE consumer in `blueprint.html` calls `/deploy-stream` with `app_name` + `variables` (JSON-stringified dict) as query parameters.
- `NAMESPACE_PRESERVING_APPS = {"cluster-init", "app-store-install"}` at `main.py:191` — both apps have fixed `targetNamespace` per component; namespace parameter is accepted but ignored server-side.

### Integration Points

- The wizard submits to the existing `/deploy-stream` endpoint — no new Python route needed; the endpoint already handles `app-store-install` per Phase 29 work.
- `quay_username` and `quay_password` are form fields that flow as x-variables; the backend's `build_quay_dockerconfigjson(user, password)` call is already wired into `deploy_stream` per Phase 29.
- The cluster-init EventSource path is the same `/deploy-stream` endpoint with `app_name=cluster-init` — already tested and working in the current `welcome.html`.
- `ClusterInitMiddleware` at `main.py:43` already exempts `/deploy-stream` and `/welcome` — no changes needed.
</code_context>

<specifics>
## Specific Ideas

- Step 1 (Node Prerequisites): Show the required KubeletConfiguration snippet (`cpuManagerPolicy: static`, `strictCPUReservation: true`) and hugepage config as a copy-pasteable `<pre>` block. Gate the Next button behind a checkbox: "I have applied node prerequisites on all worker nodes." The App Store never modifies node config (Decision A1 is hard).
- The Review step must mask all password fields with `••••••••` — `quay_password`, `weka_password` shown masked; `quay_dockerconfigjson` not shown at all.
- On Stage failure, the error message from the SSE `complete.message` or `error.message` is shown inline in the progress view (not as a modal/alert) so the user can read the full error while deciding whether to retry.
- The "two chained installs" can be rendered as a single unified progress list: first the 10 app-store-install stages, then the cluster-init stages appended to the same list when the chain begins. This gives a seamless "one install" experience.
</specifics>

<deferred>
## Deferred Ideas

- Auto-applying worker-node kubelet/hugepage config via a privileged DaemonSet — explicitly out of scope (NODE-01 in v2 requirements; Decision A1 is final for v8.0).
- Advanced/raw YAML overrides in the wizard form — out of scope per REQUIREMENTS.md Out of Scope section.
- Animated transition between wizard steps — cosmetic; not a requirement; defer if it doesn't come for free from MUI Stepper.
- Multi-cluster / multiple WEKA backends in one wizard run — v8.0 is single-target only.
- Progress-aware deadline extension (resetting the SSE deadline on observed component phase change) — deferred per Phase 29 CONTEXT.md; flat raised cap is the floor.

None — analysis stayed within Phase 30 scope.
</deferred>
