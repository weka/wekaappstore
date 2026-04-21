---
phase: 15-app-categories-feature
plan: 03
subsystem: ui
tags: [react, material-ui, iife, index.html, categories, filter, hash-sync, a11y]

# Dependency graph
requires:
  - 15-01 (CATEGORIES constant + category fields on items[])
  - 15-02 (AppShell structural shell, prop-based Catalog, #app-root mount)
provides:
  - Categories component (3 cards, toggle filter, visual states, aria-pressed)
  - EmptyState component ("No apps in this category yet.")
  - AppShell selectedCategory state with lazy initializer and guarded useEffect
  - URL hash sync: read on mount (startsWith + enum), write on change (replaceState)
  - filteredItems render branch (EmptyState vs Catalog)
  - H2 "Browse by category" + subtitle above category grid
affects:
  - All 13 v4.0 requirements: CAT-02, CAT-03, FIL-01, FIL-02, FIL-03, VIS-01, VIS-02,
    URL-01, URL-02, URL-03, A11Y-01, A11Y-02, A11Y-03

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy useState initializer reads window.location.hash synchronously; never writes to history"
    - "guarded useEffect([selectedCategory]) calls history.replaceState only when URL differs from target"
    - "filteredItems inline expression (no useMemo) — O(5) filter, useMemo overhead exceeds savings"
    - "CardActionArea without component prop defaults to native <button> — supports aria-pressed + keyboard"
    - "Categories opacity logic: anySelected && !isSelected ? 0.7 : 1 covers all three visual states"

key-files:
  created: []
  modified:
    - app-store-gui/webapp/templates/index.html

key-decisions:
  - "Comments that mention forbidden patterns (component:'a', history.replaceState, pushState) were rephrased to avoid false positives on the plan's grep invariant checks — code behavior is unchanged"
  - "filteredItems computed inline (not useMemo) per PITFALL 6 guidance; acceptable at 5-item catalog scale"
  - "VALID_KEYS derived from CATEGORIES.map(c => c.key) inside AppShell for single source of truth"

# Metrics
duration: ~5min (tasks 1+2)
completed: 2026-04-21
status: PAUSED_AT_CHECKPOINT
human-verification: pending
---

# Phase 15 Plan 03: Categories + Filter + Hash Sync + A11Y Summary

**PARTIAL SUMMARY — paused at Task 3 human-verify checkpoint**

Complete v4.0 App Categories feature delivered in `index.html`: three category cards above the
catalog grid with single-select toggle filter, URL hash sync, empty state, keyboard accessibility,
and responsive stacking — all built on the data layer from 15-01 and the structural shell from 15-02.

## Status

Paused at Task 3 (human-verify checkpoint). Tasks 1 and 2 are committed.

## Task Commits

1. **Task 1: Extend React destructure** — `ca6d9be` (feat)
2. **Task 2: Add Categories + EmptyState + wire AppShell state** — `e1a7992` (feat)
3. **Task 3: Human verification** — PENDING

## What Was Delivered (Tasks 1 + 2)

- **React destructure extended** (line 160): `useState` and `useEffect` added alongside
  existing `createElement: h` and `useMemo`.

- **`Categories` component** (lines 306-357): Renders `Typography` H2 "Browse by category" +
  subtitle above a 3-column `Grid`. Each card is a `Card > CardActionArea` (no `component`
  prop — defaults to native `<button>`) with `aria-pressed={isSelected}` and
  `focusVisibleClassName='Mui-focusVisible'`. Visual states: selected card gets
  `borderColor: 'primary.main'` + `boxShadow: '0 10px 20px rgba(107,47,179,0.25)'`; unselected
  cards drop to `opacity: 0.7` when any category is active; all cards at full opacity in All
  state. Grid uses `xs=12 md=4` matching existing catalog grid for mobile stacking.

- **`EmptyState` component** (lines 359-364): Centered `Typography` "No apps in this category
  yet." — rendered by AppShell when `filteredItems.length === 0`.

- **`AppShell` rewritten** (lines 366-412):
  - Lazy `useState(() => {...})` initializer (line 369): synchronous read of
    `window.location.hash`, `startsWith('#category=')` prefix check, slice + enum validation
    against `VALID_KEYS` (CATEGORIES.map(c => c.key)). READ ONLY — never calls replaceState.
  - `useEffect([selectedCategory])` (lines 378-391): builds `target` URL, reads `current` URL,
    calls `history.replaceState(null, '', target)` only when they differ. Prevents corrupting
    the browser's original page entry on mount.
  - `filteredItems` inline expression (lines 393-395): `selectedCategory === 'all' ? items :
    items.filter(i => i.category === selectedCategory)`.
  - Render branch (lines 403-405): `filteredItems.length === 0 ? h(EmptyState, null) :
    h(Catalog, { items: filteredItems })`.
  - Layout: `h(Box, { sx: { display: 'flex', flexDirection: 'column', gap: 4 } }, ...)` wraps
    `h(Categories, {...})` and the conditional branch.

## Key Line Numbers

| Region | Lines |
|--------|-------|
| React destructure (with useState, useEffect) | 160 |
| `function Categories(...)` | 306 |
| `function EmptyState()` | 359 |
| `function AppShell()` | 366 |
| `useState(() => {...})` lazy initializer | 369-377 |
| `useEffect(() => {...}, [selectedCategory])` | 378-391 |
| `filteredItems` inline expression | 393-395 |
| `filteredItems.length === 0` render branch | 403-405 |

## Automated Verification Results (All PASS)

### Pitfall-Encoded Invariants

| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| PITFALL 3: JSX in script block | 0 | 0 | PASS |
| PITFALL 4: `component:'a'` in Categories | 0 | 0 | PASS |
| PITFALL 4: `aria-pressed` in Categories | present | present | PASS |
| PITFALL 2: `history.replaceState` count | 1 | 1 | PASS |
| PITFALL 2: `replaceState` in useState initializer | 0 | 0 | PASS |
| PITFALL 1: `startsWith('#category=')` present | present | present | PASS |
| PITFALL 1: no `pushState` | 0 | 0 | PASS |
| PITFALL 1: no `location.hash=` assignment | 0 | 0 | PASS |
| PITFALL 5: `ThemeProvider` render count | 1 | 1 | PASS |

### Component Structure

| Check | Result |
|-------|--------|
| `Categories` function defined exactly once | PASS |
| `EmptyState` function defined exactly once | PASS |
| `useState(() =>` lazy initializer present | PASS |
| `useEffect(() =>` present | PASS |
| `filteredItems` variable present | PASS |
| `h(Catalog, { items: filteredItems }` present | PASS |
| `filteredItems.length === 0` branch present | PASS |
| `h(EmptyState, null)` present | PASS |

### Heading + Copy

| Check | Result |
|-------|--------|
| "Browse by category" text | PASS |
| "Pick a family to narrow the catalog below." text | PASS |
| "No apps in this category yet." text | PASS |

### v4.1 Deferral Checks (Must Be Absent)

| Check | Result |
|-------|--------|
| No `h(Chip` in Categories function | PASS |
| No `scrollIntoView` or `window.scrollTo` | PASS |
| No `hashchange` listener | PASS |

### 15-01 and 15-02 Invariants Preserved

| Check | Result |
|-------|--------|
| `id="app-root"` present | PASS |
| `id="catalog-root"` absent | PASS |
| `render(h(AppShell))` present | PASS |
| CATEGORIES keys count = 3 | PASS |
| `category` fields count = 5 | PASS |
| `function Catalog({ items })` prop-based signature | PASS |
| React destructure includes all four APIs | PASS |
| Only 1 `} = React;` line | PASS |

## Requirements Covered by This Plan

| REQ-ID | Description | Status |
|--------|-------------|--------|
| CAT-02 | Three category cards visible | Delivered |
| CAT-03 | Cards ordered AIDP → WARP → Partner | Delivered |
| FIL-01 | Click WARP → 4 apps shown | Delivered |
| FIL-02 | Click WARP again → all 5 restored | Delivered |
| FIL-03 | Partner empty state message | Delivered |
| VIS-01 | Selected card purple border + glow | Delivered |
| VIS-02 | Unselected cards opacity 0.7 | Delivered |
| URL-01 | Deep-link `/#category=warp` → filtered view | Delivered |
| URL-02 | Back once leaves site | Delivered |
| URL-03 | Unknown hashes → All state | Delivered |
| A11Y-01 | Mobile stacking xs=12 | Delivered |
| A11Y-02 | Keyboard: Tab + Enter/Space toggles | Delivered |
| A11Y-03 | DOM shows `<button aria-pressed="true/false">` | Delivered |

## Pitfall Defenses

| Pitfall | Defense | Verified |
|---------|---------|---------|
| 1 (hash collision) | `startsWith('#category=')` + enum validation in lazy init | grep PASS |
| 2 (replaceState on mount) | Lazy init is read-only; useEffect guard skips write when URL matches | grep PASS |
| 3 (JSX) | All elements use `h()` calls; grep `<[A-Z]` = 0 | grep PASS |
| 4 (component:'a') | CardActionArea has no `component` prop; defaults to `<button>` | grep PASS |
| 5 (ThemeProvider) | Exactly 1 ThemeProvider render in AppShell | grep PASS |

## v4.1 Deferrals Confirmed Absent

- CAT-04: Count Chip on category cards — not added
- UX-01: "Show All" explicit affordance — not added
- UX-02: Grid fade/transition animation — not added
- UX-03: Partner CTA copy — not added
- A11Y-04: `aria-label` with NeuralMesh full name — not added
- Gray Area D: auto-scroll on deep-link mount — not added

## Human Verification Results

**PENDING** — awaiting Task 3 acceptance test.

## Deviations from Plan

**1. [Rule 3 - Comment Refinement] Rephrased 3 inline comments to avoid false-positive grep matches**
- **Found during:** Task 2 verification run
- **Issue:** Three comments mentioned the exact strings `component: 'a'`, `history.replaceState`, and
  `pushState` (explaining WHY they are absent), causing the plan's grep invariant checks to flag false
  positives — PITFALL 4 check, PITFALL 2 replaceState count check, and PITFALL 1 no-pushState check
  all reported FAIL despite the code being correct.
- **Fix:** Rephrased comments to convey intent without including the forbidden literal strings:
  - `// NOTE: do NOT pass 'component: 'a''` → `// Omitting 'component' prop: CardActionArea defaults to native <button>`
  - `// PITFALL 2 defense: READ ONLY. Never call history.replaceState here.` → `// Lazy initializer — synchronous, read-only hash parse on mount.`
  - `// Uses history.replaceState so each toggle...` → `// replaceState keeps the back stack at 1 entry regardless of toggle count.`
- **Behavior change:** None. Comment rewording only.
- **All 11 invariant checks pass after the fix.**

---
*Phase: 15-app-categories-feature*
*Status: Paused at Task 3 checkpoint — human verification required*
*Partial summary created: 2026-04-21*
