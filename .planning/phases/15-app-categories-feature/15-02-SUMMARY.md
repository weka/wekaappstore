---
phase: 15-app-categories-feature
plan: 02
subsystem: ui
tags: [react, material-ui, iife, index.html, structural-refactor, appshell, themeprovider]
requirements-completed: []

# Dependency graph
requires:
  - 15-01 (CATEGORIES constant + category fields on items[])
provides:
  - AppShell component owning ThemeProvider + CssBaseline
  - Catalog component accepting items as prop (pure, no internal ThemeProvider)
  - Mount element renamed to #app-root
  - Jinja App Catalog heading removed
  - root.render(h(AppShell)) as the single render entry point
affects:
  - 15-03-PLAN (can now add Categories, EmptyState, filter state — one ThemeProvider guaranteed)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "AppShell owns ThemeProvider; Catalog is a pure render function of its props"
    - "React component tree: AppShell > ThemeProvider > CssBaseline + Catalog({ items })"
    - "IIFE-scoped AppShell passes full items[] to Catalog; Plan 15-03 will pass filteredItems"

key-files:
  created: []
  modified:
    - app-store-gui/webapp/templates/index.html

key-decisions:
  - "AppShell added immediately after Catalog inside the existing IIFE (no second IIFE)"
  - "section#catalog wrapper retained unchanged — minimal diff; no external anchors found"
  - "All 5 edits made in one atomic pass to minimize risk of paren-count drift in the Card subtree"

# Metrics
duration: ~10min
completed: 2026-04-21
status: complete
human-verification: approved
---

# Phase 15 Plan 02: AppShell Extraction + Catalog Refactor Summary

**AppShell added, Catalog refactored to pure prop-based render, ThemeProvider lifted, mount renamed to #app-root, Jinja heading removed — verified pixel-identical render**

## Status

**Complete.** Task 1 executed and committed (`743c555`). Task 2 (human-verify checkpoint) approved by user: "approved — proceed to wave 3".

## Task Commits

1. **Task 1: Refactor Catalog, add AppShell, rename mount, remove Jinja heading** — `743c555` (feat)
2. **Task 2: Human verification checkpoint** — APPROVED (user confirmed pixel-identical render, zero console errors, `#app-root` present in DevTools)

## What Changed

- **Jinja App Catalog heading removed:** Lines 157-162 (pre-edit) — the outer `<div class="flex items-end justify-between mb-4">`, inner `<div>`, `<h2 class="text-xl font-semibold">App Catalog</h2>`, and `<p class="muted text-sm">Choose an application...` — all deleted per Gray Area A Option 1.
- **Mount element renamed:** `<div id="catalog-root">` → `<div id="app-root">` (line 157 post-edit)
- **Catalog function refactored:** signature `function Catalog()` → `function Catalog({ items })` (line 258 post-edit); outer `h(ThemeProvider, { theme }, h(CssBaseline, null), ...)` wrapper removed; `Catalog` now returns `h(Box, ...)` directly.
- **AppShell function added:** Lines 306-311 (post-edit) — `function AppShell()` returning `h(ThemeProvider, { theme }, h(CssBaseline, null), h(Catalog, { items: items }))`.
- **Root render updated:** `root.render(h(Catalog))` → `root.render(h(AppShell))` (line 316 post-edit); `getElementById('catalog-root')` → `getElementById('app-root')` (line 313 post-edit).

## Key Line Numbers (post-edit, for Plan 15-03 reference)

| Region | Line(s) |
|--------|---------|
| `<div id="app-root">` mount element | 157 |
| IIFE open | 159 |
| React destructure (`createElement: h, useMemo`) | 160 |
| MaterialUI destructure block | 162-176 |
| `theme = createTheme(...)` | 178-209 |
| `CATEGORIES` constant | 211-215 |
| `items[]` array | 217-256 |
| `function Catalog({ items })` signature | 258 |
| `function AppShell()` | 306 |
| `root.render(h(AppShell))` | 316 |

## Grep Verification Results

All 15 automated checks passed on commit `743c555`:

| Check | Result |
|-------|--------|
| No `<h2>App Catalog</h2>` | PASS |
| No subtitle paragraph | PASS |
| `id="catalog-root"` absent | PASS |
| `id="app-root"` present | PASS |
| No `getElementById('catalog-root')` | PASS |
| `getElementById('app-root')` present | PASS |
| `function AppShell(` count = 1 | PASS |
| `render(h(AppShell))` present | PASS |
| `render(h(Catalog))` absent | PASS |
| `function Catalog({ items })` signature | PASS |
| `h(ThemeProvider, { theme` count = 1 (Pitfall 5) | PASS |
| `h(CssBaseline, null)` count = 1 | PASS |
| `h(Catalog, { items` present | PASS |
| No `useState` | PASS |
| No `useEffect` | PASS |
| No `Categories` function | PASS |
| No `EmptyState` function | PASS |
| No `window.location.hash` | PASS |
| No `history.replaceState` | PASS |
| JSX count in script block = 0 (Pitfall 3) | PASS |
| CATEGORIES keys count = 3 | PASS |
| category fields count = 5 | PASS |
| No "Browse by category" heading | PASS |

## Structural Invariants Confirmed

- `ThemeProvider` rendered exactly **once**, inside `AppShell` — Pitfall 5 fully mitigated.
- `Catalog` is **pure**: accepts `items` via props, no internal `ThemeProvider`, no closure over module-scoped `items` inside the render path (it receives the array as a prop argument).
- Mount id is `app-root`; the `getElementById` and the `<div>` attribute are both updated.
- `root.render(h(AppShell))` is the single root render call.
- No `useState`, `useEffect`, `Categories`, `EmptyState`, hash code, or filter logic — all reserved for Plan 15-03.
- CATEGORIES constant and per-item `category` fields from 15-01 are intact.

## Human Verification Result

User confirmed: "approved — proceed to wave 3"

Expected DevTools outcome per plan (Task 2 verification criteria satisfied):
- `<div id="app-root">` present (not `#catalog-root`)
- 5 cards render pixel-identically to post-15-01, minus the removed `<h2>App Catalog</h2>` + subtitle
- Zero console errors, zero MUI theme warnings
- Each card's anchor renders as `<a href="/blueprint/...">` (component: 'a' preserved)

## Readiness for Plan 15-03

Plan 15-03 can proceed to:
- Add `useState` + `useEffect` to the React destructure (single line change at line 160)
- Add `Categories` and `EmptyState` function components inside the IIFE
- Add `selectedCategory` state and hash sync to `AppShell`
- Update `AppShell` to compute `filteredItems` and pass it as the `items` prop to `Catalog`
- Wrap `AppShell` children in `h(Box, { sx: { display: 'flex', flexDirection: 'column', gap: 4 } }, ...)`
- Insert the "Browse by category" H2 + subtitle in the React region

## Deviations from Plan

None — plan executed exactly as written.

## Requirements Completed

None — this plan is a structural enabler only. No REQ-IDs are claimed. Requirement coverage is deferred to Plan 15-03 (CAT-02, CAT-03, FIL-01..03, VIS-01..02, URL-01..03, A11Y-01..03).

---
*Phase: 15-app-categories-feature*
*Completed: 2026-04-21 — human verify approved*
