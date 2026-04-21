# Stack Research

**Domain:** Brownfield React/MUI CDN single-file feature addition — v4.0 App Categories filter
**Researched:** 2026-04-21
**Confidence:** HIGH (all claims verified against live CDN bundle or official docs)

---

## Context: What This Research Is NOT

This is not a new-project stack selection. The PRD hard-locks the runtime to what is already in `index.html`. This document maps **which specific APIs within that locked runtime** the Categories feature needs, distinguishes what is already present vs newly introduced, and confirms no new CDN dependencies are required.

---

## Ground Truth: Existing CDN Loads (lines 17–21 of index.html)

```html
<script src="https://unpkg.com/react@18/umd/react.development.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js" crossorigin></script>
<script src="https://unpkg.com/@emotion/react@11.11.4/dist/emotion-react.umd.min.js" crossorigin></script>
<script src="https://unpkg.com/@emotion/styled@11.11.0/dist/emotion-styled.umd.min.js" crossorigin></script>
<script src="https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js" crossorigin></script>
```

Globals exposed after these loads:

| Global | Source | Version |
|--------|--------|---------|
| `window.React` | react.development.js | 18.3.1 (confirmed from bundle header) |
| `window.ReactDOM` | react-dom.development.js | 18.x |
| `window.MaterialUI` | material-ui.development.js | 5.15.14 |
| `window.emotionReact` | emotion-react.umd.min.js | 11.11.4 |
| `window.emotionStyled` | emotion-styled.umd.min.js | 11.11.0 |

---

## Question 1: MUI UMD Component Access

### UMD Global Name Confirmation

**Confirmed:** The MUI 5.15.14 UMD bundle header explicitly sets:
```javascript
global.MaterialUI = {}
```
All exported components are properties of `window.MaterialUI`. The destructuring pattern `const { Card, CardActionArea, ... } = MaterialUI;` is correct and proven by the existing code at lines 169–182 of `index.html`.

### Components Already Destructured in index.html (lines 169–182)

```javascript
const {
  createTheme,       // theme factory
  ThemeProvider,     // context provider
  CssBaseline,       // CSS reset
  Card,              // card container
  CardContent,       // card body padding wrapper
  CardActions,       // card footer action row
  CardActionArea,    // clickable button overlay on card
  Typography,        // text with variant system
  Button,            // action button
  Grid,              // 12-col responsive layout
  Box,               // generic styled div
  Chip,              // pill label
  Stack              // flex row/column
} = MaterialUI;
```

**All of these are already present in the existing destructuring.** No new MUI component names need to be added to the destructure list.

### Components Needed by the Categories Feature

| Component | UMD Access | Already in Destructure | Purpose |
|-----------|-----------|----------------------|---------|
| `Card` | `MaterialUI.Card` | YES | Outer card shell for each category |
| `CardActionArea` | `MaterialUI.CardActionArea` | YES | Clickable button wrapper (provides keyboard focus, button semantics) |
| `CardContent` | `MaterialUI.CardContent` | YES | Padded content area inside category card |
| `Chip` | `MaterialUI.Chip` | YES | "N apps" count badge |
| `Typography` | `MaterialUI.Typography` | YES | Category title (variant="h6") and description (variant="body2") |
| `Grid` | `MaterialUI.Grid` | YES | 3-column responsive layout for the category row |
| `Box` | `MaterialUI.Box` | YES | Empty-state container, layout wrappers |
| `ThemeProvider` | `MaterialUI.ThemeProvider` | YES | Shared theme context — Categories and Catalog must share one root |

**Verdict: Zero new MUI component names needed.** The existing destructure covers every component the Categories feature requires.

### CardActionArea Specifics

`CardActionArea` extends `ButtonBase`, which renders a native `<button>` element by default. Key properties:

- **HTML element:** `<button>` (implicit `role="button"`)
- **Keyboard behavior:** Enter and Space are handled natively by the `<button>` element — no custom key handler needed
- **`aria-pressed` support:** Native `<button>` elements have an implicit `button` role, so `aria-pressed` can be passed as a prop directly in JSX. `CardActionArea` passes through arbitrary HTML attributes to its root element via ButtonBase inheritance. Use: `h(CardActionArea, { 'aria-pressed': selected === categoryKey, onClick: ... })`
- **`component` prop:** Accepts a `component` prop through ButtonBase. For the categories feature, keep the default (`button`) — do NOT set `component: 'a'` (that is the pattern used on catalog cards that navigate; category cards toggle state in place and must NOT navigate)
- **`focusHighlight`:** CardActionArea renders a `<span>` as a focus highlight overlay automatically — no extra markup needed for focus visibility

**HIGH confidence** — verified against official MUI 5 API docs at mui.com/material-ui/api/card-action-area/ and mui.com/material-ui/react-card/.

---

## Question 2: React 18 APIs

### APIs Needed for Categories Feature

| API | Access via UMD Global | Already Used in index.html | Purpose |
|-----|--------------------|--------------------------|---------|
| `createElement` (aliased `h`) | `React.createElement` | YES (line 166) | All JSX-equivalent calls |
| `useState` | `React.useState` | NO — needs to be added to destructure | Selected-category state (`'all' \| 'neuralmesh-aidp' \| 'warp' \| 'partner'`) |
| `useMemo` | `React.useMemo` | YES (line 166) | Filtered items derivation |
| `useEffect` | `React.useEffect` | NO — needs to be added to destructure | Hash sync on mount (read initial hash) |
| `createRoot` | `ReactDOM.createRoot` | YES (line 167) | Already used to mount the existing Catalog |

### Additions to the Destructure Block

The existing line 166:
```javascript
const { createElement: h, useMemo } = React;
```

Must become:
```javascript
const { createElement: h, useMemo, useState, useEffect } = React;
```

That is the only change to the React destructure.

### React 18 Mounting — `createRoot` vs `ReactDOM.render`

**Critical:** The existing code already uses `createRoot` (line 167 and 305–307), which is the React 18 concurrent-mode API. `ReactDOM.render` is deprecated in React 18. The existing approach is correct.

**Architecture for shared state:** The PRD specifies and this research confirms: mount **one** React root on a single `<div>`, render an `AppShell` (or `CatalogWithCategories`) component that contains both `Categories` and `Catalog` as children under a single `ThemeProvider`. This is simpler than a two-root event-bus approach and is the standard React pattern.

Implementation shape:
```javascript
function App() {
  const [selected, setSelected] = useState('all');

  // Sync hash on mount
  useEffect(() => {
    const hash = window.location.hash;  // e.g. "#category=warp"
    const match = hash.match(/^#category=(.+)$/);
    if (match) setSelected(match[1]);
  }, []);

  const filtered = useMemo(
    () => selected === 'all' ? items : items.filter(i => i.category === selected),
    [selected]
  );

  return h(ThemeProvider, { theme },
    h(CssBaseline, null),
    h(Categories, { selected, onSelect: setSelected }),
    h(Catalog, { items: filtered })
  );
}
```

The single `createRoot` call replaces the existing one on `#catalog-root`.

**HIGH confidence** — React 18 UMD exports verified. `useState` and `useEffect` are stable React hooks present since React 16.8, fully in the React 18.3.1 bundle.

---

## Question 3: Browser APIs for URL Hash Sync

### APIs Needed

| API | Access | Browser Support | Purpose |
|-----|--------|-----------------|---------|
| `window.location.hash` | `window.location.hash` | Universal | Read initial hash on mount |
| `history.replaceState(state, '', url)` | `history.replaceState` | Baseline: widely available since July 2015 — all modern browsers | Update hash without adding a history entry |
| `hashchange` event | `window.addEventListener('hashchange', fn)` | Universal | Optional: respond to browser Back/Forward when hash changes externally |

### Implementation Pattern

**On mount** (inside `useEffect(fn, [])`):
```javascript
const hash = window.location.hash;       // "#category=warp" or "" or "#catalog"
const match = hash.match(/^#category=([a-z-]+)$/);
if (match) setSelected(match[1]);
```

**On category selection** (inside click handler):
```javascript
if (newSelected === 'all') {
  history.replaceState(null, '', window.location.pathname);  // clears hash
} else {
  history.replaceState(null, '', '#category=' + newSelected);
}
```

**Why `replaceState` not `location.hash = ...`:**
- `location.hash = '#category=warp'` triggers a `hashchange` event AND adds a new browser history entry, meaning Back walks through every category toggle. This violates PRD success criterion "Category selection does not pollute browser history."
- `history.replaceState` modifies the current history entry in place — no new entry, no `hashchange` event fired, Back leaves the page in one press.

**Hash conflict avoidance:** The regex `^#category=([a-z-]+)$` only matches hashes that start with `#category=`. The existing anchors `#catalog` and `#planning-studio` are not matched and are left untouched.

**`hashchange` listener (optional but recommended):** Adding a `hashchange` listener inside the same `useEffect` handles the edge case where the user uses Back/Forward to a previously-set hash URL (which `replaceState` alone does not re-trigger). Pattern:
```javascript
useEffect(() => {
  const sync = () => {
    const match = window.location.hash.match(/^#category=([a-z-]+)$/);
    setSelected(match ? match[1] : 'all');
  };
  sync();  // initial read
  window.addEventListener('hashchange', sync);
  return () => window.removeEventListener('hashchange', sync);
}, []);
```

**HIGH confidence** — verified against MDN docs for `history.replaceState` and `aria-pressed`.

---

## Question 4: Are Any Needed APIs Missing from the Loaded CDN Bundle?

**No. Zero gaps. The existing CDN bundle covers 100% of what the Categories feature requires.**

Verification checklist:

| Need | Status | Evidence |
|------|--------|---------|
| `MaterialUI.Card` | Present | Already destructured and used in existing Catalog |
| `MaterialUI.CardActionArea` | Present | Already destructured at line 175 — used on catalog cards as `component: 'a'` |
| `MaterialUI.CardContent` | Present | Already destructured at line 174 |
| `MaterialUI.CardActions` | Present | Already destructured at line 173 |
| `MaterialUI.Chip` | Present | Already destructured at line 179; used for tags |
| `MaterialUI.Typography` | Present | Already destructured at line 176 |
| `MaterialUI.Grid` | Present | Already destructured at line 178 |
| `MaterialUI.Box` | Present | Already destructured at line 177 |
| `MaterialUI.Stack` | Present | Already destructured at line 180 — used in Catalog |
| `React.useState` | Present in React 18 UMD | Need to add to destructure — not currently destructured |
| `React.useEffect` | Present in React 18 UMD | Need to add to destructure — not currently destructured |
| `React.useMemo` | Present in React 18 UMD | Already destructured at line 166 |
| `ReactDOM.createRoot` | Present | Already used at line 167 |
| `history.replaceState` | Browser built-in | Baseline: all modern browsers since 2015 |
| `window.location.hash` | Browser built-in | Universal |
| `aria-pressed` on `<button>` | HTML attribute | Universal — native button role supports it |

**No new `<script>` tags. No new CDN dependencies. The only code changes are inside the existing inline `<script>` block in index.html.**

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Single React root with lifted state | Two roots + CustomEvent bus | More complex, harder to reason about, PRD explicitly recommends against it |
| `history.replaceState` for hash update | `location.hash = ...` assignment | `location.hash =` adds a history entry on each click, breaking the back-button requirement |
| `hashchange` listener for hash sync | Poll `location.hash` on interval | Event-driven is zero-cost; polling is wasteful and introduces lag |
| `aria-pressed` on CardActionArea | Custom `role="button"` + `aria-pressed` on a `<div>` | CardActionArea already renders a `<button>` which has implicit button role — no need to re-declare role |
| Lift state to a shared parent component | `useContext` with a context provider | Context adds indirection with no benefit when both components are siblings in the same tree |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `ReactDOM.render(...)` | Deprecated in React 18 — triggers a warning and does not use the concurrent renderer | `ReactDOM.createRoot(el).render(...)` (already used) |
| `location.hash = '#category=warp'` | Adds a browser history entry on every click, breaking the back-button behavior | `history.replaceState(null, '', '#category=warp')` |
| `react-router` or any hash-router library | New dependency — violates PRD constraint | Native `window.location.hash` + `history.replaceState` + `hashchange` listener |
| Mounting Categories and Catalog on separate React roots | Requires an event bus for state sharing; complex and fragile | Single root on `#catalog-root` (rename or keep) wrapping both components |
| Adding `role="button"` to CardActionArea explicitly | Redundant — ButtonBase already renders a `<button>` element with implicit button role | Just pass `aria-pressed` as a prop directly |

---

## Version Compatibility

| Package | Version in Use | Compatibility Notes |
|---------|---------------|---------------------|
| `@mui/material` | 5.15.14 | `CardActionArea` present since MUI v4; `aria-pressed` pass-through confirmed via ButtonBase HTML attribute forwarding |
| `react` | 18.3.1 (confirmed from UMD banner) | `useState`, `useEffect`, `useMemo` stable since React 16.8 |
| `react-dom` | 18.x | `createRoot` is the v18 API — already in use |
| `@emotion/react` | 11.11.4 | Required peer dep for MUI 5 `sx` prop resolution — already loaded |
| `@emotion/styled` | 11.11.0 | Required peer dep for MUI 5 styled components — already loaded |

---

## Implementation Checklist for Planner

1. **Rename or extend the mount target.** The existing `<div id="catalog-root">` becomes the mount point for the unified `App` component. No new `<div>` needed unless the PRD section placement requires it (a new `<div id="app-root">` between Planning Studio and the catalog `<section>` may be cleaner).

2. **Extend the React destructure** (line 166):
   - Add `useState` and `useEffect` to `const { createElement: h, useMemo } = React;`

3. **No new MUI destructure additions needed.** All required components are already in the existing destructure block.

4. **Add `category` field to the 5 items** in the array at lines 217–251 per the PRD mapping table.

5. **Write `Categories` component** using only: `Card`, `CardActionArea`, `CardContent`, `Typography`, `Chip`, `Grid`, `Box` — all already destructured.

6. **Write `App` wrapper component** that owns `useState('all')`, `useEffect` hash sync, `useMemo` filter, and renders `ThemeProvider > Categories > Catalog`.

7. **Replace existing `root.render(h(Catalog))` call** with `root.render(h(App))`.

8. **No new `<script>` tags. No new CDN loads. No Python changes.**

---

## Sources

- `app-store-gui/webapp/templates/index.html` lines 17–21 (CDN script tags), 166–182 (existing destructure), 253–307 (existing Catalog + mount) — PRIMARY GROUND TRUTH
- MUI 5 CDN UMD bundle header at `https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js` — confirmed `global.MaterialUI = {}` export name — HIGH confidence
- MUI API docs: https://mui.com/material-ui/api/card-action-area/ — ButtonBase inheritance, `component` prop, slot structure — HIGH confidence
- MUI React Card docs: https://mui.com/material-ui/react-card/ — Card family component list — HIGH confidence
- MDN: https://developer.mozilla.org/en-US/docs/Web/API/History/replaceState — `replaceState` parameters, browser compat (baseline: all modern browsers since 2015) — HIGH confidence
- MDN: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-pressed — `aria-pressed` valid on native button elements, no explicit role attribute needed — HIGH confidence
- React docs: https://react.dev/reference/react/useState — `useState` exported from React package; accessible as `React.useState` in UMD — HIGH confidence

---
*Stack research for: v4.0 App Categories — brownfield addition to WEKA App Store GUI*
*Researched: 2026-04-21*
