# Project Research Summary — v4.0 App Categories

**Project:** WEKA App Store — v4.0 App Categories on Home Screen
**Domain:** Brownfield single-file CDN-React feature addition
**Researched:** 2026-04-21
**Overall confidence:** HIGH

## Executive Summary

The v4.0 App Categories feature is a focused brownfield addition to a single Jinja template (`app-store-gui/webapp/templates/index.html`). The existing page already loads React 18, MUI 5.15, and Emotion via CDN with no build step. Every component the feature requires — `Card`, `CardActionArea`, `CardContent`, `Chip`, `Typography`, `Grid`, `Box` — is already destructured and in use. The implementation is purely additive inside the existing inline `<script>` block. No new files, no new CDN dependencies, no Python changes, no build pipeline changes.

The recommended pattern is PRD Option A: a single React root wrapping a new `AppShell` component that lifts `ThemeProvider` out of `Catalog`, owns `selectedCategory` state via `useState`, and renders `Categories` and `Catalog` as siblings. The four structural moves are: (1) add `category: '<key>'` to each of the 5 `items` entries, (2) define a `CATEGORIES` constant, (3) write `AppShell`, `Categories`, and `EmptyState` components using the existing `h()` createElement convention, and (4) replace `root.render(h(Catalog))` with `root.render(h(AppShell))`. The entire changeset is confined to one file.

The top risks are all implementation-detail risks, not design or dependency risks. Writing JSX in a no-build codebase will crash the page. Calling `history.replaceState` on mount corrupts the back stack. Copying `component: 'a'` from the existing catalog `CardActionArea` onto category cards produces `<a>` elements instead of `<button>` elements, breaking toggle semantics and `aria-pressed`. Each is a one-line fix once caught, but all three are invisible until QA.

## PRD Reconciliation

Research confirms the PRD without material disagreement. No daylight between them.

| PRD Element | Research Verdict |
|---|---|
| Option A (single root, lifted ThemeProvider) | CONFIRMED — ARCHITECTURE.md independently reaches the same conclusion |
| No new CDN deps | CONFIRMED — STACK.md verified all required APIs are in the existing bundle |
| Hash fragment over query string | CONFIRMED — hash is correct for client-side-only Jinja template |
| `history.replaceState` not `pushState` | CONFIRMED — three research files converge |
| `aria-pressed` on `CardActionArea` | CONFIRMED — ButtonBase forwards `aria-*` props to native `<button>` |
| Single `category` value (not array) | CONFIRMED — array is v4.1+ |
| Default state = All | CONFIRMED — universal convention |
| Category order: NeuralMesh AIDP → WARP → Partner | CONFIRMED — first-party core → first-party extended → third-party |
| Single-file scope | CONFIRMED — all research files treat this as inviolable |

## PRD Open Questions — Resolved by Research

| # | Question | Research Answer | Confidence |
|---|---|---|---|
| 1 | Blueprint → category mapping | **Requires confirmation from Chris.** PRD's proposed defaults are internally consistent and can be shipped pending confirmation. | N/A — owner decision |
| 2 | "NeuralMesh AIDP" vs spelled-out label | Acronym-first title; one-line description provides spelled-out form; `aria-label` with full name is v4.1 polish. | HIGH |
| 3 | Partner empty state copy | "No apps in this category yet." is correct for launch. CTA is a PMM decision; defer. | HIGH |
| 4 | Default landing state | All. Field-team demo deep links to `/#category=neuralmesh-aidp` handle that use case without changing the default. | HIGH |
| 5 | Category order | NeuralMesh AIDP → WARP → Partner is correct. | MEDIUM |
| 6 | Single value vs array | Single value for v4.0. | HIGH |

**Open Question 1 is the only unresolved item.** It does not block planning or architecture work — only Step 1 (data preparation) needs owner confirmation before being marked done.

## Stack

The stack is locked. The only change to destructure blocks:

```javascript
// Line 166 — add useState and useEffect:
const { createElement: h, useMemo, useState, useEffect } = React;
```

No new MUI component names. No new `<script>` tags. No Python changes.

**In use:**
- React 18.3.1 UMD — `useState` (lazy initializer for hash init), `useEffect` (hash write after state change), `createElement` aliased as `h`
- MUI 5.15.14 UMD — `Card`, `CardActionArea`, `CardContent`, `Chip`, `Typography`, `Grid`, `Box`, `ThemeProvider`, `CssBaseline`
- `history.replaceState` + `window.location.hash` — browser built-ins
- Emotion 11.11 — already loaded as MUI peer dep

## Features

**Must have (v4.0 P1):**
- `category` field on all 5 `items[]` entries — atomic prerequisite
- 3-card `Categories` row above `#catalog`, inside the same `ThemeProvider`
- Single-select toggle (click active = return to All)
- Visual selected state: purple border + glow; unselected at `opacity: 0.7`
- Client-side filter: `items.filter(i => selected === 'all' || i.category === selected)`
- URL hash sync: `replaceState` on change; lazy `useState` initializer on mount
- Empty state: "No apps in this category yet."
- `aria-pressed` on `CardActionArea`; keyboard Enter/Space via native `<button>`
- Mobile-responsive: `xs={12} md={4}`
- Live "N apps" count `Chip` per card (PRD "Should pass" — treat as P1)

**Should have (v4.1):**
- "Show all" explicit affordance when a category is active
- 150ms opacity fade on grid during switch
- CTA copy for Partner empty state (PMM)
- `aria-label` with spelled-out "NeuralMesh AI Data Platform"

**Defer (v5+):**
- Sort controls, search input, multi-select, per-category icons, analytics

## Architecture

PRD Option A is authoritative. Component tree:

```
AppShell                  — owns selectedCategory state, ThemeProvider, hash sync
  ThemeProvider
    CssBaseline
    Box (flex column)
      Categories          — pure; receives categories, selected, counts, onSelect
      EmptyState          — pure; rendered by AppShell when filteredItems.length === 0
      Catalog             — pure; receives pre-filtered items prop
```

**State design:**
- `selectedCategory` in `AppShell` via `useState` with lazy initializer reading hash synchronously
- `filteredItems` and `counts` derived inline in render — no `useMemo` needed at 5 items
- Hash write: `useEffect([selectedCategory])` → `history.replaceState`
- `onSelect` passed to `Categories` is `setSelectedCategory` directly

**DOM changes:**
- `<div id="catalog-root">` → `id="app-root"` (or keep id, rename variable)
- `createRoot(el).render(h(AppShell))` replaces `render(h(Catalog))`
- No new HTML elements in the Jinja template

## Critical Pitfalls (BLOCKERS)

1. **JSX syntax** — One `<Box>` blanks the entire page. Every element must use `h(Component, props, children)`. Grep defense: `<[A-Z]` → zero matches required.
2. **`replaceState` on mount** — Overwrites the original browser history entry; breaks "one back press leaves the site." Initialization is read-only.
3. **`component: 'a'` on category `CardActionArea`** — Toggles, not links. Must default to `<button>`. DOM must show `<button aria-pressed="...">`, not `<a>`.
4. **Hash collision with `#catalog` / `#planning-studio`** — Parser must use `startsWith('#category=')` plus key validation against `['neuralmesh-aidp', 'warp', 'partner']`.
5. **`ThemeProvider` not lifted out of `Catalog`** — PRD success criterion 9 requires one `ThemeProvider` wrapping both components.

## Cross-Cutting Constraints (Apply to Every Task)

| Constraint | Implication |
|---|---|
| No JSX — `h()` only | Every component in `h(Component, props, children)` form |
| No new dependencies | Zero new `<script>` tags; zero new CDN loads |
| Single-file scope | All changes inside `index.html` inline `<script>` |
| `component: 'a'` forbidden on category `CardActionArea` | Toggle buttons, not links |
| `ThemeProvider` in `AppShell`, not `Catalog` | One provider wraps both |
| `replaceState` never called on mount | Initialization is read-only |

## Watch Out For (Top 5)

1. **JSX syntax** — blank page, silent failure. Grep is the defense.
2. **`component: 'a'` copy-paste** — DevTools check: DOM shows `<button>`, not `<a>`.
3. **`replaceState` in init code** — Back button test: `/#category=warp` → Back → leave the site.
4. **`ThemeProvider` left inside `Catalog`** — Category cards render in default MUI light theme.
5. **Hash collision** — Selecting category + clicking "Explore Blueprints" → category state must not reset.

## Phase 15 Build Order (authoritative)

### Step 1: Data Preparation (no behavior change)

**Delivers:** `category: '<key>'` on all 5 `items[]` entries; `CATEGORIES` constant inside the IIFE.
**Verifiable:** Page renders identically; new fields unused until Step 3.
**Research flag:** Requires Chris to confirm blueprint mapping (Open Question 1) before sign-off.

### Step 2: AppShell Extraction + Catalog Refactor (structural, zero new UI)

**Delivers:** `AppShell` owns `ThemeProvider` + `CssBaseline`. `Catalog` accepts `items` prop. `root.render(h(AppShell))`. No filter, no `Categories`, no `useState`.
**Verifiable:** Pixel-identical output; no new behavior.
**Avoids:** Pitfalls 4 and 5.

### Step 3: Categories Component, Filter State, Hash Sync

**Delivers:**
- `Categories` component (3 cards, `aria-pressed`, count `Chip`, visual states)
- `EmptyState` component
- `useState` lazy initializer for `selectedCategory`
- `filteredItems` and `counts` as derived values
- `useEffect([selectedCategory])` writing hash via `replaceState`
- React destructure extended with `useState` and `useEffect`

**Avoids:** All five critical pitfalls.

**Research flags per step:** None require `/gsd:research-phase`. All patterns fully specified.

## Confidence Assessment

| Area | Confidence | Notes |
|---|---|---|
| Stack | HIGH | All claims verified against live `index.html` source and official MUI/React docs |
| Features | HIGH | Table-stakes verified against MD3 spec, WCAG 2.2, comparable platforms; PRD open questions 2–6 resolved |
| Architecture | HIGH | Based on direct code inspection of lines 150–310 of `index.html` |
| Pitfalls | HIGH | All 10 pitfalls verified against actual source |

**Gaps to address:**

- **Open Question 1 (blueprint mapping):** OSS RAG and NVIDIA VSS placement requires Chris's confirmation. Blocks only Step 1 sign-off.
- **`focusVisibleClassName` on `CardActionArea`:** PITFALLS.md recommends `focusVisibleClassName: 'Mui-focusVisible'` to prevent focus ring loss after re-render. Implement proactively.
- **Opacity cascade on count Chip:** At `opacity: 0.7`, Chip inherits dimming. WCAG AA contrast maintained (~8.5:1 effective ratio). Acceptable; visual QA should verify.

## Sources

**Primary (HIGH):**
- `app-store-gui/webapp/templates/index.html` lines 1–330 — ground truth
- MUI 5.15.14 UMD bundle header
- `mui.com/material-ui/api/card-action-area/`
- `developer.mozilla.org/en-US/docs/Web/API/History/replaceState`
- `react.dev/reference/react/useState`
- `react.dev/learn/sharing-state-between-components`
- `w3.org/2001/tag/doc/hash-in-url`
- WCAG 2.2 / `aria-pressed` spec

**Secondary (MEDIUM):**
- Material Design 3 chips guidelines
- Remy Sharp "How tabs should work"
- LogRocket filtering UX best practices
- Pencil & Paper enterprise filtering analysis
- GitHub Marketplace, Hugging Face Models, VS Code Marketplace — comparative analysis

---

**Ready for Requirements.** The research is complete and internally consistent. Phase 15 can be planned with the 3-step build order as the authoritative task sequence. The only prerequisite before locking Step 1 is Chris's confirmation on Open Question 1.
