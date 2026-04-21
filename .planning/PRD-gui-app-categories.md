# PRD: App Categories on the WEKA App Store Home Screen

**Project:** WEKA App Store
**Date:** 2026-04-21
**Status:** Draft for GSD (Get Shit Done) intake
**Primary Goal:** Introduce three top-level app categories — **NeuralMesh AIDP**, **WARP**, and **Partner** — rendered as selectable cards on the home screen that filter the existing blueprint catalog in place, without changing the current visual language or requiring a frontend build step.

## Problem Statement

The home-screen catalog in `app-store-gui/webapp/templates/index.html` currently renders every blueprint as a flat, undifferentiated grid. As the catalog grows (OSS RAG, NVIDIA RAG, NVIDIA VSS, OpenFold, AI Agent for Enterprise Research, and future partner and WARP entries) users have no way to narrow scope to the kind of app they care about. Specifically:

- WEKA-first-party blueprints (NeuralMesh AIDP) are visually indistinguishable from partner blueprints
- WARP-class offerings (Weka AI RAG Platform / forthcoming WARP-branded blueprints) have no surfaced grouping
- Partner-contributed blueprints have nowhere to live as the roster expands
- Users scanning the page cannot quickly answer the question *"what kind of app am I looking at?"*
- Existing `tags` on each item (`NVIDIA`, `Open Source`) describe the **implementation stack**, not the **product family** — we need the product family as a primary navigation axis

The existing tag system is not a substitute: tags are descriptive metadata on the card, not a top-level filter the user can drive.

## Product Outcome

On landing at `/`, the user sees:

1. The existing hero and conversational planning section, unchanged.
2. A new **Categories** row directly above the App Catalog grid, containing three cards:
   - **NeuralMesh AIDP** — WEKA-built AI Data Platform blueprints
   - **WARP** — WEKA AI RAG Platform and WARP-branded blueprints
   - **Partner** — partner-contributed and third-party blueprints
3. An implicit fourth state, **All**, shown as the default on page load and reachable by clicking the currently-selected card again (toggle off) or via a subtle "Show all" affordance.
4. Clicking a category card filters the App Catalog grid below in place (same page, no navigation), with the selected card visually promoted (purple border glow matching the existing hover treatment) and the unselected cards dimmed slightly.
5. The selected category is reflected in the URL hash (e.g. `/#category=warp`) so deep links, bookmarks, and the browser back button behave naturally.
6. If a category has zero matching apps, the grid shows an inline empty state ("No apps in this category yet.") instead of collapsing to nothing.

No new pages, no new routes, no new backend endpoints.

## Users

### Primary Users
- Platform users browsing blueprints from the App Store home page
- WEKA field engineers demoing the catalog to customers and needing to jump to a specific family quickly

### Secondary Users
- Partner engineers who want a durable place to see their own contributions surfaced
- PMMs reviewing the home page for launch-readiness of a new WARP or partner blueprint

## In Scope

- Add a `category` field to each item in the existing hard-coded `items` array in `app-store-gui/webapp/templates/index.html` (lines ~217–251)
- Render a 3-card Categories row above the `#catalog` grid, styled consistently with the existing glassmorphism card language
- Client-side filter of the MUI `Grid` of app cards based on selected category
- Default state = **All**; clicking an active category toggles back to All
- URL hash synchronization (`#category=neuralmesh-aidp` | `#category=warp` | `#category=partner`)
- Visual selected / unselected states for the category cards
- Empty-state message when a filter yields zero apps
- Keyboard accessibility: category cards are focusable (native `<button>` or MUI `CardActionArea`) and Enter/Space toggles selection
- Basic responsive behavior: 3-across on `md+`, stacked on mobile (mirrors the existing catalog grid)

## Out of Scope

- Moving the `items` array out of `index.html` into a Python/JSON backend source of truth (can be a follow-up; this PRD preserves the current inline model so the change stays small)
- A full React build pipeline (Vite / Webpack) — continue using React + MUI via CDN with no build step
- Multi-select filtering or tag-based secondary filters (implementation `tags` remain informational Chips on the card)
- Search input on the catalog
- Server-side rendering of filtered results
- Per-user persistence of the last-selected category (URL hash only; no localStorage)
- Changes to blueprint detail pages, the planning studio, or `/settings`
- Backend category authorization / gating (every user sees every category)

## Existing System Facts This PRD Locks In

The implementation must conform to what the current GUI already is. Confirmed by direct inspection of `app-store-gui/webapp/templates/index.html`:

- **Framework:** Flask serving Jinja templates. The home page is one template: `index.html`.
- **Frontend runtime:** React 18 UMD + MUI 5.15 + Emotion, all loaded from `unpkg` CDN. No build step. No `package.json` for the webapp.
- **CSS:** TailwindCSS via `cdn.tailwindcss.com` + a small inline `<style>` block defining the WEKA design tokens.
- **Design tokens (CSS variables on `:root`):**
  - `--weka-purple: #6b2fb3`
  - `--weka-purple-dark: #552695`
  - `--weka-dark: #0B0C10`
- **Body:** background `#0b0c10`, text `#e5e7eb`, font `Inter` (loaded from Google Fonts), fallbacks `ui-sans-serif, system-ui`.
- **Card style:** `background: rgba(31,41,55,0.55); border: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(8px);` — glassmorphism over the dark-navy base.
- **Ambient glow:** two blurred background orbs (purple + blue) positioned absolutely inside `<main>`.
- **Buttons:** `.btn-purple` (filled WEKA purple) and `.btn-outline` (transparent with `rgba(255,255,255,0.2)` border).
- **Hover convention (existing catalog cards):** `transform: translateY(-2px); borderColor: primary.main; boxShadow: 0 10px 20px rgba(0,0,0,0.25)` — the Categories cards should reuse this treatment for the **selected** state so the language is consistent.
- **Catalog data shape (today):** array of objects with `{ title, desc, href, tags[], comingSoon? }` rendered through a `Catalog` React component inside the `#catalog` section.
- **MUI theme:** custom dark theme with `primary.main` bound to `--weka-purple`, `shape.borderRadius: 12`, chip background `rgba(255,255,255,0.06)`. The new category cards must be rendered inside the same `ThemeProvider` so they inherit these tokens.

Any implementation that introduces a build step, a separate React SPA, a new CSS framework, or a new component library violates this PRD.

## Categories and Mapping

### Category definitions

| Key | Display label | One-line description |
|---|---|---|
| `neuralmesh-aidp` | NeuralMesh AIDP | WEKA-built blueprints on the NeuralMesh AI Data Platform |
| `warp` | WARP | WEKA AI RAG Platform (WARP) blueprints |
| `partner` | Partner | Blueprints contributed by ecosystem partners |

### Initial mapping of existing catalog items

The current 5 items in `index.html` must each be assigned exactly one category. **Open question for the PRD owner (see below) — the proposed defaults are:**

| Blueprint (`href`) | Proposed category |
|---|---|
| `/blueprint/oss-rag` (OSS RAG) | `warp` |
| `/blueprint/nvidia-rag` (NVIDIA RAG) | `warp` |
| `/blueprint/nvidia-vss` (NVIDIA VSS) | `neuralmesh-aidp` |
| `/blueprint/openfold` (OpenFold) | `neuralmesh-aidp` |
| `/blueprint/ai-agent-enterprise-research` (AI Agent for Enterprise Research) | `neuralmesh-aidp` |

Partner starts empty on day one, which is a valid product state — the empty-state message is intentional and communicates "this is where partner apps will appear." This surfaces the container before partners populate it, rather than hiding the concept until there is content.

## UX Specification

### Layout

Insert a new section between the existing Planning Studio section and the `#catalog` section, inside `<main>`, at the same `max-w-6xl mx-auto px-4` column width as everything else:

```
┌─────────────────────────────────────────────────────────────┐
│  Browse by category                                         │
│  Pick a family to narrow the catalog below.                 │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ NeuralMesh  │  │    WARP     │  │   Partner   │          │
│  │    AIDP     │  │             │  │             │          │
│  │             │  │ WEKA AI RAG │  │ Ecosystem   │          │
│  │ WEKA-built  │  │ Platform    │  │ contributed │          │
│  │ AIDP apps   │  │ blueprints  │  │ apps        │          │
│  │             │  │             │  │             │          │
│  │  3 apps     │  │  2 apps     │  │  0 apps     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  [  App Catalog (filtered grid)  ]                          │
└─────────────────────────────────────────────────────────────┘
```

### Category card anatomy

Each category card is an MUI `Card` with a `CardActionArea` wrapping:
- Title (`Typography variant="h6"`) — category display label
- One-line description (`Typography variant="body2"` with `color: text.secondary`)
- Count chip — small MUI `Chip` showing `N apps` (computed from the filtered item list)

### Selected vs unselected state

- **Unselected:** baseline card styling (existing `card` class equivalent inside the MUI theme).
- **Selected:** purple border (`borderColor: 'primary.main'`), subtle purple glow (`boxShadow: '0 10px 20px rgba(107,47,179,0.25)'`), no dimming.
- **Other cards when one is selected:** reduce opacity to `0.7` so the selected card visibly dominates.
- **Hover (any state):** the existing `translateY(-2px)` lift.

### Filter behavior

- On mount, read `window.location.hash`. If it matches `#category=<key>`, start in that category; otherwise start in **All**.
- Clicking a category card sets state to that category and updates the hash via `history.replaceState` (no new history entry per click to keep the back button clean — one back press should leave the page, not walk through category toggles).
- Clicking the currently-selected category toggles back to **All** and clears the hash.
- The catalog grid below re-renders with `items.filter(i => selected === 'all' || i.category === selected)`.
- When filtered length is 0, render a centered `Typography` message inside the grid column: *"No apps in this category yet."*

### Accessibility

- Category cards render as buttons semantically (MUI `CardActionArea` already uses `role="button"` and is keyboard-focusable).
- Selected state is communicated via `aria-pressed="true"` on the action area, not color alone.
- Color contrast of the selected-state border and glow must remain WCAG AA against the dark background (WEKA purple `#6b2fb3` on `#0b0c10` clears 4.5:1 as a border; verify with the selected-state treatment).

## Implementation Notes (for the GSD planner, not final design)

- All changes are localized to `app-store-gui/webapp/templates/index.html`. No Python, no new static files, no new routes.
- Add a sibling React component `Categories` that lives in the same inline script block as `Catalog` and shares its `ThemeProvider`. Either (a) lift both into a single root component mounted on a single element, or (b) mount them separately but expose the selected-category state via a small event bus (e.g. `CustomEvent` on `document`). Option (a) is simpler and preferred.
- Add `category: '<key>'` to each of the 5 existing `items` per the mapping table above.
- Do **not** remove the existing `tags[]` array — implementation tags (`NVIDIA`, `Open Source`) remain on the card Chips.
- No new dependencies. No TailwindCSS config changes.

## Success Criteria

### Must pass
1. Loading `/` shows the three category cards between Planning Studio and App Catalog, in a single row on desktop.
2. Clicking **NeuralMesh AIDP** filters the catalog to only the three mapped blueprints; clicking **WARP** shows the two mapped blueprints; clicking **Partner** shows the empty-state message.
3. Clicking the selected card a second time returns to **All** (5 apps).
4. The URL hash updates to `#category=<key>` on selection and clears on toggle-off.
5. Loading `/#category=warp` directly lands the user in the WARP-filtered view.
6. The category cards visually match the existing catalog cards (same radius, blur, border, font, purple hover/selected treatment).
7. On a mobile viewport (≤768px) the category cards stack vertically and remain tappable.
8. Category cards are keyboard-focusable and toggleable via Enter/Space.
9. `ThemeProvider` still wraps both the Categories row and the Catalog grid — no orphaned MUI components rendering outside the WEKA dark theme.
10. No network requests added; no new backend routes; no build step introduced.

### Should pass
1. When a category is selected, unselected category cards drop to `opacity: 0.7`.
2. Each category card shows a live `N apps` count.
3. The empty state for Partner reads: *"No apps in this category yet."*
4. Category selection does not pollute browser history (back button leaves the page in one press).

## Open Questions

The following need confirmation from the PRD owner (Chris) before the planner locks scope:

1. **Blueprint → category mapping.** Is the proposed mapping above correct? In particular:
   - Does **OSS RAG** belong under **WARP** (it is a RAG platform) or **NeuralMesh AIDP** (it is the WEKA-built OSS stack)?
   - Does **NVIDIA VSS** belong under **NeuralMesh AIDP** or is there an argument for **Partner** given NVIDIA's role?
2. **Category display labels.** Should **NeuralMesh AIDP** be spelled out as *"NeuralMesh AI Data Platform"* in the card title, with *"NeuralMesh AIDP"* reserved as a subtitle, or is the acronym-first form fine?
3. **Partner empty state.** Is "No apps in this category yet." acceptable copy, or is there a preferred marketing line (e.g. *"Partner blueprints coming soon — talk to us about contributing."*)?
4. **Default landing state.** Should first-time landing default to **All** (proposed) or to **NeuralMesh AIDP** to foreground WEKA's own offerings?
5. **Category order.** Is **NeuralMesh AIDP → WARP → Partner** the correct left-to-right order for the card row? (Reads as: first-party AIDP, first-party WARP, third-party.)
6. **Future-proofing.** Should the `category` field on each item be a single value (proposed, simpler) or an array to allow an app to belong to multiple categories later?

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Adding categories makes the home page feel cluttered above the fold | Keep the Categories section compact (single row, short descriptions), and place it after Planning Studio so the hero still dominates first paint |
| Hard-coded category on each item ages poorly as the catalog grows | Acknowledged — follow-up work to externalize `items` to a backend source is explicitly out of scope for this PRD but should be a roadmap item |
| CDN-based React state across two sibling components drifts | Mount both Categories and Catalog under a single React root so state is shared through normal props/context |
| Partner category ships empty and looks broken | Empty state copy is explicit and intentional; treated as a feature, not a bug |
| URL-hash state conflicts with other hash anchors on the page (`#catalog`, `#planning-studio`) | Use a structured hash (`#category=warp`) and only treat hashes that match the `category=` prefix as category state; leave other anchors alone |

## Deliverables

1. Updated `app-store-gui/webapp/templates/index.html` containing:
   - `category` field on each of the 5 existing `items`
   - A new `Categories` React component rendered above the existing `Catalog` component, under the same `ThemeProvider`
   - Shared selected-category state driving the `Catalog` filter
   - URL-hash sync logic
2. No other file changes unless an Open Question resolves in a way that forces one.

## Appendix: Reference Snippets From the Current Code

From `app-store-gui/webapp/templates/index.html`:

- Design tokens (line 23): `--weka-purple: #6b2fb3; --weka-purple-dark: #552695; --weka-dark: #0B0C10;`
- Card baseline (line 28): `background: rgba(31, 41, 55, 0.55); border: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(8px);`
- MUI theme definition (lines 184–215): dark mode, primary bound to `--weka-purple`, `borderRadius: 12`, Chip tint `rgba(255,255,255,0.06)`
- Catalog data array to be extended (lines 217–251)
- Catalog render loop to be wrapped with a filter (lines 253–302)
