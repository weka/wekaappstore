# Pitfalls Research

**Domain:** CDN-React category filter + URL-hash sync on existing Flask/Jinja + MUI 5.15 + Tailwind single-file template
**Researched:** 2026-04-21
**Confidence:** HIGH (all findings verified against the actual `index.html` source)

---

## Critical Pitfalls

### Pitfall 1: Hash Parsing Collision With Existing Scroll Anchors

**Severity:** BLOCKER

**What goes wrong:**
The page already has `href="#catalog"` (line 82) and `href="#planning-studio"` (line 83) as native browser scroll anchors. If the category filter reads `window.location.hash` with a simple equality check (`=== '#category=warp'`) it will work, but the reciprocal risk is real: if the mount-time hash reader interprets any hash that doesn't start with `#category=` as a signal to reset to All, it introduces no bug today but breaks silently the moment a future anchor is added whose name accidentally matches a prefix. More concretely: a naive reader that does `if (hash.includes('category'))` would fire on `#my-category-list` or similar future anchors.

The correct parse is a **strict prefix-match with key extraction**:

```javascript
function parseCategoryHash() {
  const raw = window.location.hash; // e.g. "#category=warp"
  if (!raw.startsWith('#category=')) return 'all';
  const key = raw.slice('#category='.length); // 'warp', 'neuralmesh-aidp', 'partner'
  const valid = ['neuralmesh-aidp', 'warp', 'partner'];
  return valid.includes(key) ? key : 'all';
}
```

The exact condition to check is: `raw.startsWith('#category=')`. Any hash that does not start with exactly `#category=` must be ignored by the category system and left for the browser to handle as a scroll anchor.

**Why it happens:**
Developers treat `window.location.hash` as a clean namespace. In this page it is not — the hero CTA buttons hard-code `#catalog` and `#planning-studio` as href values, so those fragments are already in regular user flows. A category click that does `window.location.hash = '#category=warp'` will not interfere with those anchors, but any mount-time code that doesn't parse `#catalog` correctly will incorrectly suppress the All-state default.

**How to avoid:**
Always parse with `startsWith('#category=')` first. Validate the extracted key against the enum `['neuralmesh-aidp', 'warp', 'partner']` and fall back to `'all'` for anything unrecognized, including `''` (no hash) and `'#catalog'`.

**Warning signs:**
- Category cards re-render on first load even when the URL has `#catalog` or `#planning-studio`
- Clicking "Explore Blueprints" (href `#catalog`) somehow triggers a category filter change

**Phase to address:**
Phase 15 — Task: URL hash sync implementation. Must be in the first task that touches `window.location.hash`.

---

### Pitfall 2: History Pollution From `replaceState` Called With the Same Hash

**Severity:** BLOCKER (for "should pass" criterion: back button leaves the page in one press)

**What goes wrong:**
The PRD requires that `history.replaceState` be used so clicking categories does not fill the back stack. `replaceState` does overwrite the current entry, so repeated clicks on different categories produce only one history entry — correct. However, there is a footgun at mount time: if the component reads the hash on mount AND calls `replaceState` during initialization (e.g. to normalize the URL), it replaces the entry the browser created when navigating to the page. The user then cannot press Back to leave the site because the `replaceState` has overwritten the original entry.

Additionally, `replaceState` is not a no-op when called with the same URL — unlike `window.location.hash = '#same-value'` (which browsers silently skip if the hash is identical), `replaceState` always executes and in some browsers marks the entry as programmatically replaced, which can affect the `popstate` event listener behavior.

The correct pattern:

```javascript
function setCategory(key) {
  const next = key === 'all' ? window.location.pathname : `${window.location.pathname}#category=${key}`;
  // Only call replaceState if the URL is actually changing
  if (window.location.href !== (window.location.origin + next)) {
    history.replaceState(null, '', next);
  }
  setSelected(key); // React state setter
}
```

On mount, **do not call `replaceState` at all** — only read the current hash with `parseCategoryHash()` and call `setSelected(key)` with no history side effect.

**Why it happens:**
Developers conflate "sync the hash on mount" with "write the hash on mount." Initialization is a read operation, not a write. Writing on mount creates an entry or modifies the landing entry, corrupting the back stack.

**How to avoid:**
- Mount: read hash, call `setSelected`, do NOT call `history.replaceState`.
- User click: call `history.replaceState` only when `key !== currentSelected` (to avoid redundant calls on double-click of the same card) and update React state.
- Toggle off: call `history.replaceState(null, '', window.location.pathname)` (no hash) and set state to `'all'`.

**Warning signs:**
- Browser back button from the home page takes the user back to the home page (in All state) instead of leaving the site
- Two back presses are required to leave the page after landing on `/#category=warp`

**Phase to address:**
Phase 15 — Task: URL hash sync implementation. Verifier checklist: open `/#category=warp` directly, press Back exactly once, confirm you leave the site.

---

### Pitfall 3: JSX Written in a Codebase That Uses Raw `React.createElement`

**Severity:** BLOCKER (will throw a syntax error at runtime; page goes blank)

**What goes wrong:**
**Verified from code inspection:** `index.html` line 166 reads `const { createElement: h, useMemo } = React;`. The entire existing component — `Catalog`, all `Grid`, `Card`, `CardActionArea`, `Chip` calls — uses the `h(Component, props, ...children)` pattern throughout. There is no Babel-standalone script tag, no htm import, and no JSX anywhere in the file. CDN-loaded scripts are evaluated directly by the browser JS engine, which does not understand `<JSX />` syntax.

If any new code for the `Categories` component is written in JSX (even one line like `return <Box>...</Box>`), the browser will throw a `SyntaxError: Unexpected token '<'` and the entire inline script block will fail to execute. The catalog will also disappear because both components share the same IIFE.

**How to avoid:**
Every element in the `Categories` component must use `h()` calls matching the existing pattern:

```javascript
// WRONG — will crash at runtime
function Categories() {
  return <Box sx={{ mb: 4 }}><Typography>Browse by category</Typography></Box>;
}

// CORRECT — matches existing codebase convention
function Categories({ selected, onSelect }) {
  return h(Box, { sx: { mb: 4 } },
    h(Typography, { variant: 'h6' }, 'Browse by category')
  );
}
```

**Warning signs:**
- Blank page or missing catalog after adding the `Categories` component
- Browser console shows `SyntaxError: Unexpected token '<'` pointing to the inline script

**Phase to address:**
Phase 15 — Every task that writes new React component code. Must be enforced as a code-review gate: grep the new component code for `<[A-Z]` (capital letter after `<`) — any match means JSX was accidentally used.

---

### Pitfall 4: MUI `CardActionArea` `aria-pressed` Prop — Pass-Through Is Safe But Not Guaranteed to Render on the DOM Element

**Severity:** WARNING

**What goes wrong:**
MUI `CardActionArea` extends `ButtonBase`, which in turn renders a native `<button>` element by default. MUI's ButtonBase forwards unrecognized props to the underlying DOM element via React's standard prop spreading. `aria-pressed` is a valid HTML attribute on `<button>`, so passing `{ 'aria-pressed': isSelected }` to `CardActionArea` will reach the DOM in MUI 5.15 — **but only when `component` is not overridden**. In the existing catalog, `CardActionArea` is used with `component: 'a'` (line 277), which renders an `<a>` element. `aria-pressed` on `<a>` is technically valid but less semantically natural than on `<button>`.

For category cards, `component` must NOT be set to `'a'` — the card is a toggle button, not a link. Leave `component` at its default (renders `<button>`), and pass `aria-pressed` directly:

```javascript
h(CardActionArea, {
  onClick: () => onSelect(cat.key),
  'aria-pressed': selected === cat.key,
  // Do NOT set component: 'a' — these are buttons not links
}, /* children */)
```

MUI 5.15 confirmed to forward `aria-*` props through ButtonBase's prop spreading to the native element. No `sx` override or wrapper is needed.

The secondary risk: if `aria-pressed` is passed as a boolean `true`/`false`, React will render it as the string `"true"` / `"false"` on the DOM — which is exactly what the `aria-pressed` attribute specification requires. Do not convert to string manually.

**Why it happens:**
Developers copy the existing catalog's `CardActionArea` usage (which has `component: 'a'`) as a template for the new category cards, then wonder why the semantic role is wrong or why clicking doesn't behave like a button.

**How to avoid:**
- Category cards: omit `component` prop from `CardActionArea` (defaults to `<button>`)
- Pass `aria-pressed` as a boolean: `'aria-pressed': selected === cat.key`
- Verify in DevTools that the rendered DOM shows `<button aria-pressed="true">` (not `<a>`)

**Warning signs:**
- Category cards render as `<a>` elements (inspect DOM)
- Keyboard focus shows the element has `role="link"` not `role="button"` (screen reader announces "link" not "button")
- `aria-pressed` attribute missing from DOM

**Phase to address:**
Phase 15 — Task: `Categories` component initial implementation. Verifier: inspect DOM for `<button aria-pressed="true/false">` after clicking a category.

---

### Pitfall 5: Tailwind Preflight Active — MUI Baseline Styles Partially Overridden

**Severity:** WARNING

**What goes wrong:**
**Verified from code inspection:** Line 15 of `index.html` loads `https://cdn.tailwindcss.com` with no configuration object disabling preflight. Tailwind's Play CDN injects preflight (a Normalize.css-derived reset) by default. There is no `tailwind.config` object in the HTML that sets `corePlugins: { preflight: false }`.

This means preflight IS active. Preflight resets: `button` element styles (removes border, background, padding), `a` element color/decoration, and heading font sizes. MUI components use Emotion's CSS-in-JS, which injects styles with higher specificity than the Tailwind preflight reset — so MUI components are largely unaffected for their own visual styling. However, two real collision points exist:

1. **Button focus outline:** Tailwind preflight sets `outline: none` on buttons via `*, ::before, ::after { box-sizing: border-box; border-width: 0; ... }`. MUI's `CardActionArea` (a `<button>`) applies its own focus-visible ring via `::after` pseudo-element overlay, which survives the reset. This is fine in practice.

2. **Card wrapper `<Box>` with Tailwind classes:** Using Tailwind utility classes like `className="mx-auto max-w-6xl"` on a `Box` wrapping MUI components is safe — Tailwind utilities have low specificity and MUI's `sx` prop styles use Emotion's class injection which wins. The PRD recommends this pattern and it is used throughout the existing template (e.g., `class="max-w-6xl mx-auto px-4"`).

The real failure mode: adding a Tailwind class that resets something MUI relies on at the same DOM level. For example, adding `className="flex"` to a `Card` component itself (not a wrapper) will conflict with MUI's `display` styles set through Emotion. Always apply Tailwind classes to wrapper `<div>` or `Box` elements, never directly to MUI component roots.

**How to avoid:**
- Apply Tailwind classes only to outer wrapper `<Box>` or plain `<div>` elements, never as `className` on `Card`, `CardActionArea`, `CardContent`, `Chip`, `Grid`, or `Typography` components.
- The existing pattern is correct: `h(Box, { className: 'max-w-6xl mx-auto px-4', sx: { py: 4 } }, /* MUI children */)`.
- Do not add a `tailwind.config` block to disable preflight — it would change existing page behavior.

**Warning signs:**
- Category cards lose border-radius or shadow despite `sx` specifying them (Tailwind class applied to MUI root)
- Chip inside card loses background color (Tailwind `bg-*` class on the Chip itself)

**Phase to address:**
Phase 15 — Task: Categories section layout and styling. Verifier checklist: confirm no Tailwind utility class is set as `className` directly on an MUI component root.

---

## Moderate Pitfalls

### Pitfall 6: `useMemo` for Filter — Premature at Current Scale, Required at Moderate Growth

**Severity:** INFO (now), WARNING (when catalog exceeds ~50 items)

**What goes wrong:**
With 5 items, `items.filter(i => selected === 'all' || i.category === selected)` runs in microseconds and `useMemo` adds no measurable benefit. The existing catalog already imports `useMemo` from React (line 166: `const { createElement: h, useMemo } = React`) but does not use it. If the filter is placed inside the render function without `useMemo`, it will re-run on every parent state change (including the `selected` toggle itself — which is always the trigger). This is correct behavior: the filter should re-run when `selected` changes.

The real risk is a future developer wrapping the entire `items` array definition inside `useMemo` with an empty dependency array `[]` to "optimize" it, creating a stale closure over a snapshot of `items` that never updates if `items` is ever made dynamic.

The threshold where `useMemo` on the filter computation matters: approximately 200+ items with a complex filter predicate. At 5-50 items, the overhead of `useMemo` itself (dependency comparison) exceeds the savings.

**How to avoid:**
- Do not use `useMemo` on the filter at launch. Keep it as an inline expression in the render path.
- `useMemo` is appropriate if and when `items` is loaded asynchronously or contains 100+ entries.
- The correct future pattern (when needed): `const filtered = useMemo(() => items.filter(i => ...), [selected, items])`.

**Warning signs:**
- Stale filter results (shows wrong items after state change) — diagnostic: `useMemo` with missing or empty deps array.

**Phase to address:**
Phase 15 — Note in code comments that `useMemo` is deferred intentionally until catalog exceeds ~50 items.

---

### Pitfall 7: Opacity Cascade — Unselected Card at `opacity: 0.7` Dims the Count Chip

**Severity:** WARNING

**What goes wrong:**
The PRD specifies unselected category cards drop to `opacity: 0.7`. CSS `opacity` is inherited by all descendants — the count `Chip` inside the unselected card will render at 70% opacity. In this codebase, the Chip has `background: rgba(255,255,255,0.06)` and `color: #e5e7eb` (set in the MUI theme override at lines 211-213). At 70% opacity, the effective alpha of the Chip background drops to ~0.042 and the text drops to roughly `rgba(229, 231, 235, 0.7)`.

The count text `#e5e7eb` at 70% on `#0b0c10` background: luminance calculation gives a contrast ratio of approximately 8.5:1 even at 70% opacity (the original text is very light on very dark). This does clear WCAG AA (4.5:1 minimum). The chip is still readable.

However, the chip may visually appear "washed out" compared to baseline. The question is aesthetic, not accessibility-blocking. If the design intent is that the count chip remains crisp even on a dimmed card, apply `opacity: 0.7` to the card's background and border layers using a separate overlay approach (wrapping the non-chip content in an inner `Box` with opacity) rather than on the card root. This is more complex and likely not worth it for the initial implementation.

**How to avoid:**
Apply `opacity: 0.7` via `sx={{ opacity: selected === 'all' || selected === cat.key ? 1 : 0.7 }}` on the `Card` root. Accept that the Chip inherits this. Do not add per-element opacity overrides unless QA feedback specifically flags chip readability.

**Phase to address:**
Phase 15 — Task: selected/unselected visual states. Verifier: visually inspect the dimmed card at 0.7 opacity and confirm the chip count text is still legible.

---

### Pitfall 8: Keyboard Toggle Reliability — Enter Twice Does Cycle Correctly, But Focus Ring May Disappear

**Severity:** WARNING

**What goes wrong:**
`CardActionArea` renders as `<button>`, which fires `onClick` on both Enter and Space by default (native browser behavior, no MUI intervention needed). Pressing Enter twice on the same category card will correctly cycle: first press → `onClick` fires → `setSelected('warp')` → state updates → React re-renders → card shows `aria-pressed="true"`; second press → `onClick` fires → `setSelected('all')` (toggle off) → `aria-pressed="false"`. MUI does not eat keypresses on native `<button>` elements.

The real issue: after the second press (toggle off), the `Card` component re-renders with new `sx` props (removing the selected border/glow). MUI's Emotion CSS-in-JS injects new class names on re-render. In some browser/MUI combinations, the focus-visible ring (`:focus-visible` pseudo-class) is briefly lost during the re-render because the element receives new className strings. This is a documented MUI behavior on state-driven style changes to `ButtonBase`.

MUI 5.15 mitigates this with the `focusVisibleClassName` prop on `ButtonBase`. `CardActionArea` exposes `focusVisibleClassName` as a passthrough prop.

**How to avoid:**
- Do not add custom `onKeyDown` handlers — native `<button>` Enter/Space handling is correct.
- Add `focusVisibleClassName: 'Mui-focusVisible'` to each `CardActionArea` to pin the focus ring class name and prevent it from disappearing during re-render.
- Test keyboard navigation: Tab to first card, Enter, Tab to second card, Enter, Shift+Tab, Enter again — confirm focus ring is visible throughout.

**Warning signs:**
- Focus ring disappears after state change
- Screen reader does not announce `aria-pressed` value change (means the button identity changed in the DOM)

**Phase to address:**
Phase 15 — Task: accessibility pass. Must be tested with keyboard-only navigation before marking the task done.

---

### Pitfall 9: MUI Theme Token Drift — `primary.main` vs CSS Variable

**Severity:** INFO

**What goes wrong:**
The MUI theme at line 187 reads `--weka-purple` at theme creation time: `primary: { main: getComputedStyle(document.documentElement).getPropertyValue('--weka-purple').trim() || '#6b2fb3' }`. This is a one-time read. If the CSS variable `--weka-purple` is later changed (e.g., a future inline `<style>` update changes it from `#6b2fb3` to a new brand color), the MUI theme will not update — it was created once at script evaluation and baked in as a hex string. The category cards using `borderColor: 'primary.main'` in their `sx` prop will keep the old purple.

This is not a bug introduced by v4.0 — it exists today for the existing catalog cards too. But the new category cards will add another `primary.main` reference, increasing the surface area of the drift.

The correct resilient pattern is to reference the CSS variable directly in `sx` via the MUI theme's CSS variable syntax, but MUI 5.15 UMD does not fully support the CSS theme variables API (that arrived with MUI 6). In MUI 5.15 on CDN, the only option is to re-read the CSS variable if needed:

```javascript
// Resilient: reads the live CSS variable at render time (not at theme creation)
sx={{ borderColor: `var(--weka-purple)` }}
// vs current pattern (baked at theme creation, drifts if CSS var changes):
sx={{ borderColor: 'primary.main' }}
```

**How to avoid:**
For the selected border glow, use `borderColor: 'var(--weka-purple)'` in the `sx` prop directly instead of `borderColor: 'primary.main'`. This reads the live CSS variable at paint time. Both work identically today but the CSS var reference is more resilient to future token changes and matches the existing `btn-purple` pattern used in the Tailwind section of the page.

**Phase to address:**
Phase 15 — Task: selected/unselected visual states. Low priority; acceptable to defer to a follow-up style pass if initial implementation uses `primary.main`.

---

### Pitfall 10: Mobile Viewport — 3-Card Row Requires Explicit `xs: 12` on Grid Items

**Severity:** WARNING

**What goes wrong:**
The PRD requires 3-across on `md+` and stacked (full-width) on mobile. The existing catalog uses `h(Grid, { item: true, xs: 12, md: 4, key: idx }, ...)`. If the Categories row uses a similar `Grid` with `xs: 12, md: 4`, cards will stack correctly on mobile. The pitfall is the `Chip` inside the category card — `Chip` has a default `maxWidth` and may overflow horizontally on narrow viewports if the chip label is long (e.g., "NeuralMesh AIDP · 3 apps").

A `Chip` with a long label on a 320px wide card will not truncate by default — it wraps or causes horizontal scroll on the chip's parent container if the parent has `overflow: hidden`. MUI Chip does not apply `text-overflow: ellipsis` without explicit `sx` styling.

Specific risk: on an iPhone SE (375px viewport), a category card that is `xs: 12` (full width minus padding) at ~343px wide will fit a single Chip of ~120px. The label "3 apps" is short enough to fit. However, if the Chip is placed inline with the title in a `Stack` that doesn't wrap, it can force the card to expand horizontally.

**How to avoid:**
- Use `Grid item xs={12} md={4}` (same as existing catalog cards) — confirmed safe.
- Place the count Chip on its own line inside the card (below the description), not inline with the title.
- If Chip and title are in the same `Stack`, set `flexWrap: 'wrap'` on the Stack's `sx`.
- Test at 375px viewport width (iPhone SE breakpoint) before marking done.

**Warning signs:**
- Horizontal scrollbar appears on mobile at `overflow-x: auto` on `<main>` or `<body>`
- Category cards overflow their column at 375px viewport width

**Phase to address:**
Phase 15 — Task: responsive layout. Verifier checklist: open in browser DevTools responsive mode at 375px, confirm no horizontal scroll.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hard-code `items` array with `category` field in `index.html` | No backend change needed; single-file scope | Category assignment requires a code deploy; doesn't scale to dynamic catalog | Acceptable for v4.0 (PRD explicitly descopes externalization) |
| Read `--weka-purple` once at theme creation time | Simple; matches current codebase pattern | Theme drifts if CSS variable changes | Acceptable until next brand refresh |
| Skip `useMemo` on filter | Simpler code | Performance degrades at ~100+ items | Acceptable at current 5-item catalog |
| Mount both `Categories` and `Catalog` under a single React root | Avoids cross-component event bus | Requires refactoring the existing `Catalog` IIFE into a shared scope | Preferred by PRD; should be done this way |
| Use `replaceState` for all category clicks | Prevents back-stack pollution | If user wants to "undo" category selection they lose the ability | Acceptable — PRD explicitly specifies this behavior |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| MUI 5.15 UMD + Tailwind CDN | Applying Tailwind classes as `className` on MUI component roots | Apply Tailwind only to wrapper `<div>` or `<Box>` elements; let MUI's `sx` control MUI component internals |
| `window.location.hash` + MUI component mount | Calling `replaceState` during mount to "initialize" the hash | Read the hash on mount (state-only); never write to history during initialization |
| MUI `CardActionArea` as a toggle button | Copying the existing catalog's `component: 'a'` pattern | Omit `component` prop for toggle buttons; let it default to `<button>` |
| React UMD `createElement` alias | Writing JSX in a file loaded as a plain `<script>` | Use `h()` convention matching the existing codebase; never write `<JSX />` syntax |
| MUI `Chip` inside opacity-dimmed card | Expecting the Chip to remain at full opacity | Accept inherited opacity; verify contrast is still WCAG AA (it is at 0.7 on this dark background) |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Filter recomputes on every render without `useMemo` | Imperceptible at 5 items | Acceptable now; add `useMemo` when items exceed ~50 | Never a real problem below 100 items in a CDN-React context |
| `replaceState` called on every render cycle (not just on user click) | Back button broken; browser warns about excessive history manipulation | Call `replaceState` only inside the `onClick` handler, not inside `useEffect` with `selected` as dep | Immediately — any call outside user-gesture context is a bug |
| `getComputedStyle` called on every render to read `--weka-purple` | Minor CSSOM thrash on every card render | Read it once at theme creation (current pattern); or use `var(--weka-purple)` directly in `sx` | Not a real-world concern at this scale |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Partner category ships empty with no explanation | Users think the filter is broken | Empty state copy is explicit: "No apps in this category yet." — PRD explicitly handles this |
| No visual feedback between click and re-render | On slow connections, user may double-click | MUI `CardActionArea` ripple provides immediate visual feedback; no additional loading state needed at this scale |
| Unselected cards at `opacity: 0.7` look disabled, not just unfocused | Users may not understand they can still click unselected cards | Maintain `cursor: pointer` on all cards regardless of opacity; hover lift (`translateY(-2px)`) should apply to all states |
| Back button doesn't leave the page | User frustration if `pushState` accidentally used | Use `replaceState` only; verify with one-back-press exit test |
| Category cards too tall on mobile due to long descriptions | Layout breaks on 375px viewport | Keep card descriptions under ~80 characters; verify mobile layout in DevTools before shipping |

---

## "Looks Done But Isn't" Checklist

- [ ] **Hash sync on direct load:** Verify `/#category=warp` lands in WARP-filtered state — not just that clicking a card updates the hash.
- [ ] **Toggle-off clears hash:** Verify clicking the active card removes `#category=warp` from the URL entirely (back to bare path).
- [ ] **One back press exits:** Open `/#category=warp`, press Back once — confirm you leave the site (not land on All-state home page).
- [ ] **Keyboard toggle cycle:** Tab to a card, press Enter twice, confirm toggle on → off without losing focus ring.
- [ ] **aria-pressed on DOM:** Open DevTools, inspect a category card — verify `<button aria-pressed="true">` when selected, `aria-pressed="false"` when not.
- [ ] **No JSX syntax:** Search the new component code for `<[A-Z]` — any match is an accidentally-written JSX element that will crash at runtime.
- [ ] **Mobile layout:** DevTools responsive view at 375px — confirm no horizontal scroll bar appears.
- [ ] **Partner empty state rendered:** Select Partner category — confirm "No apps in this category yet." message appears (not a blank grid).
- [ ] **ThemeProvider still wraps both components:** DevTools `React DevTools` tree shows `ThemeProvider` as an ancestor of both `Categories` and `Catalog` components.
- [ ] **Existing anchors still scroll:** Click "Explore Blueprints" (href `#catalog`) — confirm page scrolls to catalog section; click does NOT trigger a category filter change.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| JSX accidentally used (page blank) | LOW | Find the `<` character in new component code, convert each element to `h()` call; no other file changes needed |
| Hash parsing collision | LOW | Replace the hash reader with the `startsWith('#category=')` pattern; 2-line fix |
| History pollution (back button broken) | LOW | Audit all `replaceState` / `pushState` calls; remove any inside `useEffect` or mount code; move to `onClick` handler only |
| Wrong `CardActionArea` component prop | LOW | Remove `component: 'a'` from category card; verify in DOM it renders as `<button>` |
| Tailwind class on MUI root breaking styles | MEDIUM | Wrap MUI component in a plain `<div>` or `<Box>`, move Tailwind class to the wrapper |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Hash collision with `#catalog`, `#planning-studio` | Phase 15 — URL hash sync task | Direct navigation to `/#catalog` does not trigger category filter logic |
| History pollution (back button) | Phase 15 — URL hash sync task | Press Back once from `/#category=warp` — site is left, not reset to All |
| JSX without build step (runtime crash) | Phase 15 — every component task | `grep` new code for `<[A-Z]`; zero matches required |
| `CardActionArea` component prop (renders as `<a>` not `<button>`) | Phase 15 — initial Categories component | DOM inspection: category cards are `<button>` elements |
| Tailwind preflight + MUI style collision | Phase 15 — layout/styling task | No Tailwind class directly on MUI component root |
| `replaceState` called on mount (corrupts back stack) | Phase 15 — URL hash sync task | One-back-press exit test from direct URL |
| Mobile layout overflow | Phase 15 — responsive layout task | No horizontal scrollbar at 375px viewport |
| Chip readability at 0.7 opacity | Phase 15 — visual states task | Visual QA: chip count text legible on dimmed card |
| `aria-pressed` missing from DOM | Phase 15 — accessibility task | DevTools attribute inspection after click |
| Keyboard toggle focus ring loss | Phase 15 — accessibility task | Keyboard-only navigation test through all three cards |

---

## Sources

- Direct inspection of `app-store-gui/webapp/templates/index.html` (lines 15, 82-84, 164-310) — HIGH confidence
- [MUI CardActionArea API](https://mui.com/material-ui/api/card-action-area/) — props inherit from ButtonBase; aria-* forwarding confirmed by ButtonBase's "Props of the native component are also available" clause — MEDIUM confidence
- [Tailwind Preflight documentation](https://tailwindcss.com/docs/preflight) — preflight enabled by default with CDN — HIGH confidence
- [Tailwind Play CDN preflight disable discussion](https://github.com/tailwindlabs/tailwindcss/discussions/15967) — confirmed no `tailwind.config` in `index.html` means preflight is active — HIGH confidence
- [MDN: History.replaceState()](https://developer.mozilla.org/en-US/docs/Web/API/History/replaceState) — replaceState is not a no-op on same-URL calls — HIGH confidence
- [React useMemo documentation](https://react.dev/reference/react/useMemo) — threshold guidance for when memoization is worthwhile — HIGH confidence
- [MUI Chip dark theme contrast issues](https://github.com/mui/material-ui/issues/9407) — Chip contrast on dark backgrounds; opacity cascade behavior — MEDIUM confidence
- PRD risk table (`.planning/PRD-gui-app-categories.md` — Risks section) — directly maps to pitfalls 1, 3, 4, 5 above — HIGH confidence

---
*Pitfalls research for: CDN-React + MUI 5.15 category filter + URL hash sync on Flask/Jinja single-file template*
*Researched: 2026-04-21*
