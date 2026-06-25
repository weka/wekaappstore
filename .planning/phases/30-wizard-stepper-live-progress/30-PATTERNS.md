# Phase 30: Wizard Stepper & Live Progress - Pattern Map

**Mapped:** 2026-06-25
**Files analyzed:** 1 modified (`app-store-gui/webapp/templates/welcome.html`), 0 created
**Analogs found:** 5 / 5 (all in-repo; same React+MUI Babel stack)

> This is a frontend-only, single-file phase. All edits land in `welcome.html`. The patterns
> below are the concrete analogs the executor must replicate *inside* that file (and the
> `blueprint.html` SSE consumer it copies from). No Python, operator, or blueprint changes.

---

## File Classification

| Modified file | Role | Data Flow | Closest Analog | Match Quality |
|---------------|------|-----------|----------------|---------------|
| `app-store-gui/webapp/templates/welcome.html` (WelcomeApp component) | component (React+MUI page) | request-response (form submit) + streaming (SSE progress) | `app-store-gui/webapp/templates/blueprint.html` (SSE consumer) + existing `welcome.html` WelcomeApp (component shell + ClusterInitSSE) | exact (same stack, same endpoint) |

The single file plays two roles at once:
- **Form/wizard** (steps 1–5) → analog is `blueprint.html`'s `#deploy-form` collect-and-submit and the existing `welcome.html` namespace `Select`/`localStorage` pattern.
- **Live progress** (SSE view) → analog is `blueprint.html` lines 284–346 (the battle-tested `EventSource` consumer + `sectionClass`).
- **Chain + redirect** → analog is existing `welcome.html` `handleInitialize` (lines 251–297) + the `/cluster-status` redirect poll (lines 154–215).

---

## Pattern Assignments

### Pattern 1 — SSE consumer (progress view, D-05/D-06/D-07)

**Analog:** `app-store-gui/webapp/templates/blueprint.html` lines 279–346 (source of truth — copy this, adapt to JSX/React state).

**Endpoint + EventSource open** (`blueprint.html:279-284`):
```javascript
const params = new URLSearchParams({
  app_name: '{{ name }}',
  variables: JSON.stringify(variables),
});
const url = `/deploy-stream?${params.toString()}`;
const es = new EventSource(url);
```
For the wizard, `app_name` is the literal `'app-store-install'` and `variables` is the JSON-stringified field dict from D-08. A `namespace` key is included inside the `variables` object (the backend reads `user_vars.get("namespace")` at `main.py:2990` — it is NOT a separate query param). Namespace is ignored server-side for `app-store-install` (`NAMESPACE_PRESERVING_APPS`, `main.py:191`) but harmless to send.

**Stage-color mapping `sectionClass(phase)`** (`blueprint.html:287-297`) — reuse exactly; this is the D-06 color map:
```javascript
const sectionClass = (phase) => {
  const base = 'px-3 py-2 rounded border ';
  switch ((phase || '').toLowerCase()) {
    case 'ready':
    case 'healthy':    return base + 'border-green-500/60 bg-green-500/10 text-green-300';
    case 'failed':
    case 'error':      return base + 'border-red-500/60 bg-red-500/10 text-red-300';
    case 'installing': return base + 'border-yellow-400 bg-yellow-500/10 text-yellow-300';
    default:           return base + 'border-white/10 bg-gray-800/40'; // pending / not started
  }
};
```
> Note D-06 says "Installing/Upgrading → blue" but the analog uses yellow for `installing` and has no `upgrading` case. Match the existing analog colors (yellow = in-progress) for visual consistency, and add an `upgrading` case alongside `installing` if the operator can emit it. In a React rewrite this becomes a per-stage `status → MUI sx color` map rather than className strings, but the phase→bucket logic is identical.

**Event handling** (`blueprint.html:298-345`) — the four event types and their payload shapes (confirmed against the backend emitter):

| Event | Payload (from `main.py`) | Wizard action |
|-------|--------------------------|---------------|
| `init` | `{type:'init', items:[names...], message}` (`main.py:3016`) | Populate the stage list from `msg.items` (10 components for app-store-install), all Pending |
| `component` | `{type:'component', name, phase, message}` (`main.py:3112`) | Update the matching stage's status to `msg.phase`; if name unknown, append it (matches `blueprint.html:315-321`) |
| `complete` | `{type:'complete', ok:true|false, result, message}` (`main.py:3120` / `3127` / `3081`) | `ok !== false` → all stages green, advance chain (D-10); `ok === false` → show `msg.message` + Retry (D-07) |
| `error` | `{type:'error', message}` (`main.py:2980/3007/3011/3130/3134/3142`) | Hard fail: show `msg.message` inline + Retry (D-07) |

Reference handler body to copy (`blueprint.html:298-345`):
```javascript
es.onmessage = (ev) => {
  try {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'init') {
      // (msg.items || []).forEach(...) -> seed stage list as 'pending'
    } else if (msg.type === 'component') {
      // find stage by msg.name; li.className = sectionClass(msg.phase); title = msg.message
    } else if (msg.type === 'complete') {
      if (msg.ok !== false) { /* all green; success */ } else { /* failed: msg.message */ }
      es.close();
    } else if (msg.type === 'error') {
      // msg.message; es.close();
    }
  } catch (err) { es.close(); }
};
es.onerror = () => { es.close(); };
```
> IMPORTANT: the backend uses default SSE framing `data: {json}\n\n` (`main.py:2972-2973`) and sends `: ping\n\n` keepalive comments (`main.py:3095`). Comment lines do not fire `onmessage`, so no guard is needed — but do NOT switch to named `addEventListener('init', ...)`; all events arrive on the default `message` channel as in the analog.

**Retry (D-07):** re-run the same handler that opens the `EventSource` with the identical `variables`. Mirror the existing `welcome.html` Retry button which simply re-invokes `handleInitialize` (`welcome.html:479`):
```javascript
<Button size="small" color="inherit" onClick={handleInitialize} sx={...}>Retry</Button>
```
Idempotency is guaranteed by the operator CR upsert path (Phase 27 D-09) — re-apply is non-destructive.

---

### Pattern 2 — WelcomeApp component shell (D-01, the host for the wizard)

**Analog:** `app-store-gui/webapp/templates/welcome.html` (the file itself).

**CDN imports + MUI destructure** (`welcome.html:14-19`, `40-45`) — the wizard adds `Stepper, Step, StepLabel, TextField` (and optionally `StepContent`) to this destructure. They are all present in the `@mui/material@5.15.14` UMD bundle (`welcome.html:18`), so **no new `<script>` tags or dependencies are needed (D-01 confirmed)**:
```javascript
const {
  ThemeProvider, createTheme, CssBaseline, Button, LinearProgress,
  Typography, Box, Container, Paper, List, ListItem, ListItemText,
  ListItemIcon, Divider, Alert, AlertTitle, Chip, FormControl,
  InputLabel, Select, MenuItem
  // + Stepper, Step, StepLabel, TextField  <-- add here
} = MaterialUI;
```

**Babel + JSX conventions to match (Established Patterns):**
- Script block is `<script type="text/babel">` (`welcome.html:39`); in-browser compile, **no build step** — do not introduce webpack/vite/npm.
- Functional component with `React.useState` / `React.useEffect` hooks (`welcome.html:98-249`). Wizard step index and all field values live in `useState` (D-02).
- **Inline `sx` objects must be wrapped in Jinja `{% raw %}...{% endraw %}`** because this is a Jinja2 template — `{{` collides with Jinja. See every `sx={% raw %}{{ ... }}{% endraw %}` in `welcome.html` (e.g. lines 82, 303, 310). The executor MUST wrap all new `sx={{...}}` and inline `style={{...}}` props this way or the template will fail to render.
- Double-quoted JSX string attributes; functional sub-components like `StatusIndicator` (`welcome.html:80-96`) are the model for any new `WizardStep`/`StageRow` helper.

**Validation rules (D-04) — client-side only.** The backend does NOT enforce `host:port` or version-tag formats (`_validate_variable_value` at `main.py:1891` only checks hostname/URL shapes and skips `validate:false`). So the wizard's regexes are the sole gate:
- `join_ip_ports` each entry: `^[^:]+:\d+$`
- version tags (`operator_version`, `weka_image_version`): `^v?\d+\.\d+(\.\d+)?$`
- Required: quay username+password, operator version, ≥1 endpoint, WEKA image version, WEKA username+password.

Fire validation on Next/Submit click (not blur), block forward navigation on failure (D-04).

---

### Pattern 3 — Chain + redirect (D-10, D-11)

**Analog:** existing `welcome.html` `handleInitialize` (lines 251–297) for the cluster-init `EventSource`, and the status poll's redirect logic (lines 154–215, esp. 182–187).

**Existing cluster-init EventSource open** (`welcome.html:267-268`) — this is the second link in the chain; Phase 30 triggers it automatically after the app-store-install `complete` event instead of from a button:
```javascript
const url = `/deploy-stream?app_name=cluster-init&namespace=${selectedNamespace}`;
const es = new EventSource(url);
```
> cluster-init takes no x-variables (it is namespace-preserving); D-10 says omit the `variables` param. The existing code passes `namespace=` which is accepted but ignored — keep or drop, either is fine.

**Existing redirect pattern** (`welcome.html:182-187`) — after Ready, fetch `/cluster-status` for `redirect_url`, then `window.location.href`:
```javascript
if (status.phase === 'Ready') {
  setInitializing(false);
  clearInterval(interval);
  const targetUrl = status.redirect_url || '/';
  setTimeout(() => { window.location.href = targetUrl; }, 2000);
}
```
For D-11 the trigger is the cluster-init SSE `{type:'complete', ok:true}` (not a poll), but the redirect mechanic is identical: call `/cluster-status` (`main.py:2551`), read `redirect_url` from the JSON (`main.py:2637-2642`; `redirect_url` is `null` until cluster-init is Ready), then `window.location.href = redirect_url || '/'`.

**Chain sequencing (D-10):** on app-store-install `complete.ok===true`, close `es`, then open the cluster-init `EventSource`. Per Claude's Discretion + Specific Ideas, the cleanest UX is a single unified stage list: append cluster-init's `init` items to the existing 10 app-store-install stages so it reads as "one install."

---

### Pattern 4 — localStorage.selectedNamespace (D-02, only client persistence)

**Analog:** `welcome.html:135-136` (read) and `welcome.html:366-367` (write), plus `blueprint.html:201-203`.

Read on mount:
```javascript
const saved = localStorage.getItem('selectedNamespace');
if (saved && data.items.includes(saved)) { setSelectedNamespace(saved); }
```
Write on Select change (`welcome.html:365-368`):
```javascript
onChange={(e) => {
  setSelectedNamespace(e.target.value);
  localStorage.setItem('selectedNamespace', e.target.value);
}}
```
This is the ONLY value allowed in `localStorage`. Secret fields (`quay_password`, `weka_password`) are in-memory `useState` only — never persisted (D-02). The Review-step namespace selector (D-09) reuses this exact `FormControl`/`Select`/`MenuItem` block from `welcome.html:358-373`.

---

### Pattern 5 — MUI Stepper availability (D-01 confirmation)

**Confirmed.** `welcome.html:18` loads `@mui/material@5.15.14/umd/material-ui.development.js` — the full UMD bundle, which exports the entire component set including `Stepper`, `Step`, `StepLabel`, `StepContent`, `StepButton`, and `TextField`. They are accessed via the same `MaterialUI` global destructure (Pattern 2). No CDN change, no extra `<script>`. Use MUI default step-connector styling and `variant="outlined"` inputs to match the existing dark-theme `Select` visual language (`welcome.html:47-78` theme).

---

## Shared Patterns

### SSE event contract (cross-cutting — applies to both chained installs)
**Source:** `app-store-gui/webapp/main.py:2966-2973` (framing) and emit sites `2980, 3007-3142`.
**Apply to:** the progress view for both `app-store-install` and `cluster-init`.
- Framing: `data: {json}\n\n`, default `message` channel, `: ping\n\n` keepalives.
- Event types: `init` (seed), `component` (per-stage update; `app-store-install` only — cluster-init emits no `component` events because it has no `cr_name`/appStack status to poll, see `main.py:3080-3082` → it jumps straight to `complete`), `complete` (`ok` bool), `error`.
- `complete.message` and `component.message` are already secret-redacted server-side (`_redact_secrets`, `main.py:3116/3126`) — safe to display verbatim inline (D-07, PROG-03).

### Jinja `{% raw %}` guarding of inline JSX objects
**Source:** every `sx={% raw %}{{...}}{% endraw %}` in `welcome.html` (e.g. 82-91, 303-304, 379-387).
**Apply to:** every new `sx={{...}}` / `style={{...}}` the wizard adds. Non-negotiable — unguarded `{{` breaks Jinja rendering of the template.

### Field → x-variable mapping (form submit, D-08)
**Source:** `cluster_init/app-store-install.yaml` x-variables block (lines 1–31).
**Apply to:** the variables object posted to `/deploy-stream`.
| x-variable key | Default (from yaml) | Required |
|----------------|---------------------|----------|
| `quay_username` | — | yes (D-04) |
| `quay_password` | — | yes (D-04, masked) |
| `operator_version` | `v1.13.0` (yaml:3) | yes (D-04) |
| `join_ip_ports` | — | yes (yaml:9) |
| `weka_image_version` | — | yes (yaml:6) |
| `weka_endpoint_scheme` | `http` (yaml:17) | no |
| `weka_org` | `Root` (yaml:21) | no |
| `weka_username` | — | yes (yaml:23) |
| `weka_password` | — | yes (yaml:25, masked) |
| `namespace` | `default` | no (ignored server-side) |

`quay_dockerconfigjson` is NEVER a form field — derived server-side from `quay_username`+`quay_password` (`main.py:3044-3048`). `join_ip_ports_list` / `endpoints_csv` are server-derived too (`split_endpoints`, `main.py:1757-1770`). Submit raw `join_ip_ports` as the comma-delimited string only.

---

## No Analog Found

| Feature | Why no direct analog | Planner guidance |
|---------|----------------------|------------------|
| MUI `Stepper`-based multi-step wizard navigation (Next/Back, active step index) | No existing GUI page uses a Stepper — current `welcome.html` is single-screen, `blueprint.html` is a single form | Use standard MUI Stepper composition (`activeStep` state + `Stepper`/`Step`/`StepLabel`). MUI docs pattern; styling matches `welcome.html` theme. RESEARCH.md / MUI conventions apply. |
| Node-prerequisites copy-paste `<pre>` snippet + confirmation checkbox (step 1) | No copy-paste-gated checkbox exists in the codebase | New, but trivial: a `<pre>` block (or `Box component="pre"`) for the KubeletConfiguration snippet + a controlled MUI `Checkbox` gating the Next button. Match `welcome.html` dark styling. |
| Masked review summary (`••••••••` for passwords) | No existing masked-summary view | Render password field values as a constant bullet string in the Review step; never echo the actual secret to DOM beyond the password `TextField type="password"`. |

---

## Metadata

**Analog search scope:** `app-store-gui/webapp/templates/` (welcome.html, blueprint.html), `app-store-gui/webapp/main.py` (deploy_stream, cluster-status, NAMESPACE_PRESERVING_APPS, validation helpers), `cluster_init/app-store-install.yaml` (x-variables schema).
**Files scanned:** 4
**Pattern extraction date:** 2026-06-25
