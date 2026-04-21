# Architecture Research

**Domain:** Single-file CDN-React shared filter state — v4.0 App Categories milestone
**Researched:** 2026-04-21
**Confidence:** HIGH (grounded in direct inspection of `app-store-gui/webapp/templates/index.html`)

---

## Actual System State (from code inspection, not PRD assumptions)

Confirmed by reading lines 150–310 of `index.html`:

- **Single DOM mount point:** `<div id="catalog-root"></div>` at line 163, inside `<section id="catalog">`.
- **One IIFE, one `createRoot` call.** The entire React tree lives in one `<script>` block (lines 164–310). No other React roots exist on the page.
- **`ThemeProvider` owned by `Catalog`.** Lines 254–301 show `Catalog` calls `h(ThemeProvider, { theme }, ...)` internally. The theme is not lifted above the component — it is part of `Catalog`'s render tree.
- **`items` array is module-scoped inside the IIFE,** defined at lines 217–251. Not a prop — currently closed over by `Catalog`.
- **No `useState`, no `useEffect`, no `useContext`, no `createContext`** anywhere in the file. Confirmed by grep — zero hits.
- **No `StrictMode` wrapper.** `root.render(h(Catalog))` at line 307 — bare render, no `React.StrictMode` wrapping.
- **No `hashchange` listener.** No existing hash-reading or hash-writing logic anywhere in the file.
- **No `useMemo` in current `Catalog`.** The destructure at line 166 pulls `useMemo` from React but it is never called; this is dead code from an earlier draft.

---

## System Overview

### Current architecture (before this milestone)

```
index.html
  <section id="catalog">
    <div id="catalog-root">          ← single React mount point
      IIFE {
        items = [...]                ← closed over, not a prop
        function Catalog() {
          ThemeProvider              ← owns theme
            Box > Grid
              items.map(card)        ← renders all items, no filter
        }
        createRoot(catalog-root)
          .render(h(Catalog))
      }
    </div>
  </section>
```

### Target architecture (Option A — single root)

```
index.html
  <section id="app-shell-root">      ← rename or replace #catalog-root
    <div id="app-root">
      IIFE {
        items = [...]                ← unchanged, still IIFE-scoped
        CATEGORIES = [...]           ← new constant, IIFE-scoped

        function AppShell() {
          useState('all')            ← selectedCategory lives here
          ThemeProvider              ← lifted from Catalog to AppShell
            Box (column layout)
              Categories(selectedCategory, onSelect)
              Catalog(items=filtered, selectedCategory)
              [EmptyState when filtered.length === 0]
        }

        createRoot(app-root)
          .render(h(AppShell))
      }
    </div>
  </section>
```

---

## Question-by-Question Decisions

### 1. Option A (single root) vs Option B (sibling mounts + CustomEvent)

**Decision: Option A is correct. The PRD's preference is verified by the code.**

The current codebase has exactly one `createRoot` call, one IIFE, and one mount element. Option B would require creating a *second* mount point and a *second* `createRoot` call — adding net complexity. Option A extends what already exists.

**What each option requires changing in `Catalog`:**

| Change | Option A | Option B |
|--------|----------|----------|
| Remove `ThemeProvider` from `Catalog` | YES — must lift to `AppShell` | NO — each root keeps its own |
| Accept `items` as prop | YES — filter happens above; Catalog receives pre-filtered list | MAYBE — Catalog must listen to `CustomEvent` and own its own filter state |
| Add `selectedCategory` prop | YES — passed from `AppShell` (for `aria-pressed` or count display if needed) | NO — Catalog subscribes to event bus instead |
| Add new DOM element for `Categories` mount | NO — both live in same root | YES — need `<div id="categories-root">` in the HTML |
| Add `document.addEventListener('categorychange', ...)` | NO | YES — inside Catalog |
| Add `document.dispatchEvent(new CustomEvent(...))` | NO | YES — inside Categories |

**Invasiveness:** Option A requires lifting `ThemeProvider` out of `Catalog` and converting `items` from a closed-over variable to a prop. That is a small, mechanical change. Option B requires adding an event bus subscription inside `Catalog`, which introduces hidden coupling (component behavior depends on events fired from elsewhere) and makes Catalog harder to test in isolation.

**Testability / future extraction:** Option A produces a `Catalog` component with an explicit prop surface (`items` array). Future extraction to a standalone file or a build-step project is straightforward — pass the right props. Option B's event bus coupling requires extracting the listener logic as well, and the test harness must simulate DOM events rather than simply passing props. Option A is strictly better for future extraction.

---

### 2. State container shape for Option A — `useState` vs `createContext`

**Decision: `useState` in `AppShell`, pass `selectedCategory` as a prop to `Categories`. Do NOT use `createContext`.**

There are exactly 2 consumers of `selectedCategory`:
- `Categories` — needs it to show selected/unselected visual state and `aria-pressed`
- The filter expression that produces `filteredItems` — computed inline in `AppShell` before passing to `Catalog`

`Catalog` itself does not need `selectedCategory` as a prop — it only needs the already-filtered `items` array. The count on each category card is computed in `AppShell` as well, using the full `items` array before filtering.

Context is appropriate when a value must cross multiple intermediate components that do not use it (prop drilling). Here the tree is shallow: `AppShell` → `Categories` (1 hop). There is nothing being drilled through. `createContext` adds boilerplate (`createContext`, `Provider`, `useContext`) with zero benefit for a 2-level tree.

**Recommended state shape:**

```javascript
function AppShell() {
  const { useState } = React;
  const [selectedCategory, setSelectedCategory] = useState('all');

  const filteredItems = selectedCategory === 'all'
    ? items
    : items.filter(i => i.category === selectedCategory);

  const counts = CATEGORIES.reduce((acc, cat) => {
    acc[cat.key] = items.filter(i => i.category === cat.key).length;
    return acc;
  }, {});

  return h(ThemeProvider, { theme },
    h(CssBaseline, null),
    h(Box, { sx: { display: 'flex', flexDirection: 'column', gap: 4 } },
      h(Categories, { categories: CATEGORIES, selected: selectedCategory, counts, onSelect: setSelectedCategory }),
      filteredItems.length === 0
        ? h(EmptyState, null)
        : h(Catalog, { items: filteredItems })
    )
  );
}
```

---

### 3. Initial state from URL hash — where should hash-reading live?

**Decision: Lazy `useState` initializer.**

```javascript
const [selectedCategory, setSelectedCategory] = useState(() => {
  const hash = window.location.hash; // e.g. "#category=warp"
  const match = hash.match(/^#category=([\w-]+)$/);
  if (match) {
    const key = match[1];
    return CATEGORIES.some(c => c.key === key) ? key : 'all';
  }
  return 'all';
});
```

**Why lazy initializer, not `useEffect`, not `useMemo`:**

- `useEffect` runs after render — the component would flash "All" state for one render cycle before reading the hash. Visible to the user.
- `useMemo` is not an initializer — it re-runs on every render if its deps are unstable, and `window.location.hash` is not a tracked dependency. It would compute correctly on mount but is the wrong semantic.
- Lazy `useState` initializer runs exactly once at mount, synchronously, before the first render. The component starts in the correct state with zero flicker. This is the React-idiomatic pattern for computing initial state from an external source.

**StrictMode concern:** No `StrictMode` wrapper is present in the current file (verified by grep). The lazy initializer is safe regardless — `window.location.hash` is idempotent to read; calling it twice produces the same value. Even if StrictMode double-invoked the initializer (which it does in dev mode for class components' constructors, not for functional lazy initializers in React 18), the result would be the same.

---

### 4. Hash-writing logic — `useEffect`, `replaceState`, and the `hashchange` race

**Decision: One-way sync with a single `useEffect`. No `hashchange` listener needed for this feature.**

```javascript
useEffect(() => {
  if (selectedCategory === 'all') {
    history.replaceState(null, '', window.location.pathname);
  } else {
    history.replaceState(null, '', '#category=' + selectedCategory);
  }
}, [selectedCategory]);
```

**Is bidirectional sync needed?**

The PRD's success criteria require:
1. Clicking a category updates the hash — satisfied by the `useEffect` above.
2. Loading `/#category=warp` directly starts in the filtered view — satisfied by the lazy initializer in question 3.
3. Back button leaves the page in one press — satisfied by `replaceState` (not `pushState`), which means category toggles never create history entries.

The back-button case does NOT require a `hashchange` listener. The user loading `/#category=warp` is handled at initial mount by the lazy initializer. The user does not navigate between category hashes using the browser's back/forward buttons because `replaceState` is used — there is no history stack of category changes to navigate through.

**Race condition analysis:** If a `hashchange` listener were added that called `setSelectedCategory`, it would fire whenever `history.replaceState` is called in the `useEffect`. In modern browsers, `replaceState` does NOT fire `hashchange` — only `pushState` and direct navigation do. So there is no race between `replaceState` and a `hashchange` listener. However, adding the listener is still unnecessary complexity for this feature scope.

**Only add a `hashchange` listener if** the requirement changes to: "the user should be able to navigate between category views using the browser back button." That would require switching to `pushState`, adding the listener, and handling the sync loop carefully. That is explicitly out of scope per the PRD.

---

### 5. Empty-state ownership — `Catalog` or `AppShell`?

**Decision: `AppShell` owns the empty-state branch. `Catalog` should not render empty state.**

```javascript
// In AppShell render:
filteredItems.length === 0
  ? h(EmptyState, null)          // AppShell controls this branch
  : h(Catalog, { items: filteredItems })  // Catalog always has items
```

**Rationale:**

- If `Catalog` owns empty-state rendering, its API surface becomes: "receives 0..N items and conditionally renders either a grid or a message." That mixes two responsibilities and makes `Catalog`'s behavior conditional on an external filter state that it cannot see (it only sees the filtered result, not why it's empty).
- If `AppShell` owns the branch, `Catalog`'s contract is simple and stable: "given a non-empty `items` array, render a grid." If the empty-state copy or design changes later (a different message, a CTA button, an illustration), only `AppShell`/`EmptyState` changes — `Catalog` is untouched.
- `Catalog`'s testability improves: passing any non-empty array renders a grid, period. No conditional paths to exercise.

`EmptyState` can be an inline function component — no need for a separate file given the no-build-step constraint:

```javascript
function EmptyState() {
  return h(Box, { sx: { textAlign: 'center', py: 8 } },
    h(Typography, { variant: 'body1', color: 'text.secondary' },
      'No apps in this category yet.')
  );
}
```

---

### 6. Build order for implementation

**Decision: Three discrete commits, not one atomic change.**

The dependency order is strictly bottom-up: data shape must exist before filter logic, filter logic must exist before the UI components that consume it.

**Step 1 — Data preparation (no behavior change, safe to ship independently)**

Changes to `index.html`:
- Add `category: '<key>'` field to each of the 5 objects in the `items` array (lines 217–251).
- Define the `CATEGORIES` constant inside the IIFE, above `items`.
- Verify the page still renders correctly — no functional change.

This step is independently verifiable: existing catalog renders identically; the new `category` field is unused.

**Step 2 — AppShell extraction + Catalog refactor (structural, zero new UI)**

Changes to `index.html`:
- Rename `<div id="catalog-root">` to `<div id="app-root">` (or keep the ID and rename the variable — either works; renaming the DOM id is cleaner).
- Extract `ThemeProvider` + `CssBaseline` out of `Catalog` into a new `AppShell` function.
- Convert `Catalog` to accept `items` as a prop instead of closing over the module-level `items` array.
- Mount `AppShell` instead of `Catalog` on the root element.
- `AppShell` initially passes all `items` to `Catalog` unchanged, no filter yet.

This step is also independently verifiable: the rendered catalog is pixel-identical to before. No `useState`, no `selectedCategory` yet — just a structural refactor.

**Step 3 — Add `Categories` component, wire filter state, add hash sync**

Changes to `index.html`:
- Add `Categories` function component.
- Add `EmptyState` function component.
- Add `useState` + lazy initializer for `selectedCategory` in `AppShell`.
- Add filter expression `filteredItems` in `AppShell`.
- Add hash-write `useEffect`.
- Render `Categories` above the `Catalog`/`EmptyState` branch.
- Insert the `<section>` HTML container between Planning Studio and `#catalog` in the Jinja template (or move the React mount above `#catalog` — see integration point below).

This step is the only one that introduces new user-visible behavior.

---

## Component Tree

```
AppShell
  state: selectedCategory ('all' | 'neuralmesh-aidp' | 'warp' | 'partner')
  derived: filteredItems = items.filter(...)
  derived: counts = { [key]: number }
  |
  ThemeProvider (theme)            ← lifted here from Catalog
    CssBaseline
    Box (flex column)
      |
      Categories
        props: categories, selected, counts, onSelect
        renders: 3x Card/CardActionArea with aria-pressed
      |
      [conditional branch on filteredItems.length]
        if 0  → EmptyState
        if >0 → Catalog
                  props: items (pre-filtered array)
                  renders: Grid of app cards
```

---

## Component Responsibilities

| Component | Responsibility | Props In | State |
|-----------|----------------|----------|-------|
| `AppShell` | Owns filter state, reads/writes hash, computes derived data, controls rendering branch | none | `selectedCategory` |
| `Categories` | Renders 3 category cards with selected/unselected visual treatment | `categories`, `selected`, `counts`, `onSelect` | none (pure) |
| `Catalog` | Renders a grid of app cards | `items` (pre-filtered) | none (pure) |
| `EmptyState` | Renders zero-result message | none | none (pure) |

---

## DOM Integration Points

| Element | Before | After |
|---------|--------|-------|
| `<div id="catalog-root">` | Mount point for `Catalog` alone | Rename to `id="app-root"`, mount point for `AppShell` |
| `<section id="catalog">` | Contains heading + `#catalog-root` | Still contains heading + `#app-root`; Categories renders inside the React tree, not as a separate HTML section |
| `createRoot(...)` call | `createRoot(rootEl).render(h(Catalog))` | `createRoot(rootEl).render(h(AppShell))` |
| `ThemeProvider` | Inside `Catalog` | Inside `AppShell` |
| `items` array | Closed over by `Catalog` | IIFE-scoped constant; passed as prop to `Catalog` |

**Note on HTML section structure:** The PRD UX spec shows a separate HTML section for Categories. Because both `Categories` and `Catalog` are now rendered inside the same React root (`AppShell`), the HTML outside React does not need a new `<section>` element for Categories. The layout division is handled by `Box` spacing inside `AppShell`. This keeps the Jinja template change to a single line: renaming the `id` attribute on one `<div>`.

---

## Data Flow

```
window.location.hash (read once at mount)
        |
        v
useState lazy initializer → selectedCategory
        |
        +---> useEffect → history.replaceState (hash write on change)
        |
        +---> filteredItems = items.filter(i => selected === 'all' || i.category === selected)
        |
        +---> counts = CATEGORIES.reduce(...)
        |
        v
AppShell render
    |
    +---> Categories(selected=selectedCategory, counts, onSelect=setSelectedCategory)
    |         |
    |         +---> user clicks card → onSelect(key) → setSelectedCategory → re-render
    |
    +---> [filteredItems.length === 0] ? EmptyState : Catalog(items=filteredItems)
```

---

## Architectural Patterns

### Pattern: Lift state to lowest common ancestor

**What:** When two sibling components need to share state, move the state to their closest shared parent (the "lowest common ancestor"). Here, `Categories` and `Catalog` both need `selectedCategory` — their LCA is `AppShell`.

**When to use:** Any time two components need to read or write the same value. This is the standard React pattern before reaching for a state library or context.

**Trade-offs:** Requires the LCA to pass props down. Acceptable here because the tree is only 2 levels deep.

### Pattern: Derive, don't sync

**What:** Compute `filteredItems` and `counts` as derived values inside the render function rather than storing them in separate `useState` calls.

**When to use:** When a value can be deterministically computed from existing state. Storing derived values in state creates synchronization bugs (state A and state B can drift).

**Trade-offs:** Re-computes on every render. For 5 items, the cost is unmeasurable. Even at 500 items, `Array.filter` is microseconds.

### Pattern: Lazy `useState` initializer for one-time external reads

**What:** `useState(() => readExternalSource())` runs the initializer once at mount, synchronously, avoiding a render-then-correct flicker cycle.

**When to use:** When initial state must be read from `window.location`, `localStorage`, or another external source.

**Trade-offs:** None for this use case. The initializer function is garbage-collected after mount.

---

## Anti-Patterns

### Anti-Pattern: Catalog owns its own filter state

**What people do:** Add `selectedCategory` state inside `Catalog` and have it listen for events, or read the hash internally.

**Why it's wrong:** Creates hidden coupling. The filter trigger (Categories component) and the filter consumer (Catalog) must communicate through a side channel (events or global state) rather than through the React prop system. Makes both components harder to test and to reason about. Violates single-direction data flow.

**Do this instead:** `AppShell` owns `selectedCategory`. `Catalog` is a pure presentational component — it renders whatever `items` it receives. The filter happens above it.

### Anti-Pattern: `ThemeProvider` stays inside `Catalog`

**What people do:** Leave `ThemeProvider` inside `Catalog` and wrap `Categories` in a second `ThemeProvider` instance.

**Why it's wrong:** Produces two separate MUI theme contexts. Emotion's CSS-in-JS inserts two sets of style rules. The MUI team explicitly recommends one `ThemeProvider` per tree. Token updates would need to be applied in two places.

**Do this instead:** Lift `ThemeProvider` to `AppShell` — it wraps the entire rendered tree once.

### Anti-Pattern: `useEffect` to read initial hash

**What people do:** `const [cat, setCat] = useState('all'); useEffect(() => { const h = readHash(); if (h) setCat(h); }, []);`

**Why it's wrong:** Causes a visible render flash. The component renders with `'all'` first, then a synchronous DOM update via `useEffect` sets the correct category. On fast connections this can still cause a layout shift on the catalog grid.

**Do this instead:** Lazy `useState` initializer — reads hash synchronously before first render.

### Anti-Pattern: `pushState` for category toggles

**What people do:** Use `history.pushState` so each category click creates a back-button entry.

**Why it's wrong:** The PRD success criterion explicitly requires "one back press leaves the page." With `pushState`, clicking NeuralMesh AIDP → WARP → Partner creates 3 history entries; the user needs 3 back presses to leave the page.

**Do this instead:** `history.replaceState` — overwrites the current history entry on each category change. One back press always leaves the page regardless of how many category toggles occurred.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `AppShell` → `Categories` | Props (`selected`, `counts`, `onSelect`) | `onSelect` is `setSelectedCategory` passed directly — no wrapper needed |
| `AppShell` → `Catalog` | Props (`items` — pre-filtered array) | Catalog has no knowledge of filter or categories |
| `AppShell` → `EmptyState` | None — rendered by branch condition | No props needed; copy is hardcoded |
| `AppShell` ↔ `window.location.hash` | Read: lazy initializer. Write: `useEffect` → `history.replaceState` | One-way each direction; no bidirectional sync loop |

### Functions and DOM elements touched by this milestone

| Element / Function | Location in current `index.html` | Change |
|--------------------|------------------------------------|--------|
| `<div id="catalog-root">` | line 163 | Rename attribute to `id="app-root"` |
| `Catalog` function | lines 253–302 | Remove internal `ThemeProvider`; accept `items` prop instead of closure |
| `root.render(h(Catalog))` | line 307 | Change to `root.render(h(AppShell))` |
| `const rootEl = document.getElementById('catalog-root')` | line 304 | Update selector string to `'app-root'` |
| `items` array | lines 217–251 | Add `category: '<key>'` field to each object |
| IIFE destructure | line 166 | Add `useState`, `useEffect` to the React destructure |

New additions (all inside the IIFE):
- `CATEGORIES` constant (above `items`)
- `AppShell` function
- `Categories` function
- `EmptyState` function

---

## Sources

- Direct inspection of `app-store-gui/webapp/templates/index.html` (lines 1–330) — HIGH confidence
- React 18 documentation on lazy state initialization: https://react.dev/reference/react/useState#avoiding-recreating-the-initial-state — HIGH confidence
- React 18 documentation on lifting state up: https://react.dev/learn/sharing-state-between-components — HIGH confidence
- MDN: `history.replaceState` vs `pushState` and `hashchange` event behavior — HIGH confidence (replaceState does not fire hashchange in any current browser)
- PRD `.planning/PRD-gui-app-categories.md` — Implementation Notes and Success Criteria sections
- `app-store-gui/webapp/templates/index.html` grep confirms: no StrictMode, no existing useState/useEffect/useContext/createContext, no hashchange listener, no existing hash logic

---

*Architecture research for: v4.0 App Categories — single-file CDN React shared filter state*
*Researched: 2026-04-21*
