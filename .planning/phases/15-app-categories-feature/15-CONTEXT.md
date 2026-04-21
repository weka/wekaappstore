# Phase 15: App Categories Feature - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can filter the WEKA App Store catalog by category (AIDP, WARP, Partner) via selectable cards above the grid, with URL deep-link support and keyboard accessibility. Delivered as a single-file change to `app-store-gui/webapp/templates/index.html`. No build step. No new CDN dependencies. No backend changes. Count Chip, auto-scroll on deep-link, grid fade animation, Partner CTA, backend items externalization, and multi-value categories are explicitly deferred to v4.1+ — see REQUIREMENTS.md Future section.

</domain>

<decisions>
## Implementation Decisions

### Page Structure / Mount Strategy (Gray Area A)

**Decision: Option 1 — one section, Categories absorbs the existing heading.**

- Remove the existing Jinja "App Catalog" heading and subtitle (`<h2>App Catalog</h2>` + `<p>Choose an application...</p>`) at approximately lines 159-161 of `index.html`
- Rename the React mount element from `<div id="catalog-root">` to `<div id="app-root">`
- Update the corresponding JS reference (`document.getElementById('catalog-root')` or equivalent) to the new id
- AppShell renders a single region containing the Categories row followed by either the filtered grid or the EmptyState
- Matches research Option A (single React root, lifted ThemeProvider) at this mount point
- The wrapping `<section id="catalog">` can stay as-is, be renamed to `<section id="apps">`, or dropped — Claude's discretion (functional no-op; external scroll anchors to `#catalog` would break either way once the inside content changes, and no existing anchors pointing here were found during the scout)

### Heading Copy Inside React Region (Gray Area B)

**Decision: Option B1 — one H2 above Categories only.**

- H2 copy: "Browse by category"
- Subtitle copy (`Typography variant="body2"`, muted): "Pick a family to narrow the catalog below."
- No heading renders above the grid itself; the category cards are the navigation axis
- Matches PRD mock-up verbatim (PRD Layout section)

### Category Card Descriptions (Gray Area C)

**Decision: keep the PRD default descriptions verbatim, including the NeuralMesh reference on AIDP.**

| Category | Display label | Description |
|---|---|---|
| aidp | AIDP | WEKA-built blueprints on the NeuralMesh AI Data Platform |
| warp | WARP | WEKA AI RAG Platform (WARP) blueprints |
| partner | Partner | Ecosystem partner-contributed blueprints |

Rationale: keeping "NeuralMesh AI Data Platform" in the AIDP description makes the shortened "AIDP" label self-explaining without cluttering the title. Partner description is generic because the category is empty on launch.

### Deep-Link Scroll Behavior (Gray Area D)

**Decision: no auto-scroll on deep-link mount.**

- When the page loads with `/#category=<key>`, the hash is read synchronously (lazy `useState` initializer) and the filter is applied to the grid
- Page renders at the natural top (hero + Planning Studio still visible above)
- Auto-scroll to the Categories row on mount is intentionally deferred to v4.1 polish; will be reconsidered based on real user feedback

### Claude's Discretion

- Whether `<section id="catalog">` wrapper keeps its id, is renamed to `#apps`, or is dropped in favor of a nameless wrapping `<div>` — functional no-op
- Exact spacing/margins between Categories row and grid (use Tailwind/MUI conventions already in index.html)
- EmptyState visual treatment — centered `Typography` inside the grid column per PRD; no new custom component needed
- Focus ring styling on `CardActionArea` — use MUI default via the existing theme; revisit if visual QA shows contrast issues
- Whether `AppShell` sits inside the existing IIFE or has its own sibling IIFE — both work, pick whichever keeps the diff cleaner
- Exact names of React destructured additions (`useState`, `useEffect`) — single line change to existing destructure block at line ~166

</decisions>

<specifics>
## Specific Ideas

- PRD mock-up section "Layout" shows "Browse by category" with subtitle "Pick a family to narrow the catalog below." — use these strings verbatim
- AIDP = 1 app ("AI Agent for Enterprise Research"); WARP = 4 apps (OSS RAG, NVIDIA RAG, NVIDIA VSS, OpenFold); Partner = 0 apps (empty state shows "No apps in this category yet.")
- Category order left-to-right: AIDP → WARP → Partner (first-party AIDP → first-party RAG → third-party)
- Selected card visual: purple border (`borderColor: 'primary.main'`) + glow (`boxShadow: '0 10px 20px rgba(107,47,179,0.25)'`) + no dimming; reuse existing catalog card hover treatment (`translateY(-2px)`)
- Unselected cards when a category is active: `opacity: 0.7`
- Showing All: all 3 cards at full opacity (no card visually marked "All active"; the absence of selection is the signal)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets

- **All MUI components needed are already destructured** in the existing IIFE (around line 166 of `index.html`): `Card`, `CardContent`, `CardActions`, `CardActionArea`, `createTheme`, `ThemeProvider`, `CssBaseline`. Additional destructure needs: `Typography`, `Grid`, `Box`, `Chip` (verify each is present; STACK.md confirms they are available in the existing MUI 5.15 UMD bundle whether or not currently destructured)
- **React APIs:** `createElement` aliased as `h` and `useMemo` already destructured. Needs: `useState`, `useEffect` added to the same destructure line (one-line change)
- **Emotion + MUI dark theme** already configured; selected-state purple uses existing `primary.main` theme token (bound to `--weka-purple: #6b2fb3`)

### Established Patterns

- **Raw `React.createElement` aliased as `h()`** — no JSX, no Babel-standalone; one JSX element blanks the entire page including the existing catalog
- **Inline IIFE inside the Jinja template** — no external JS file, no build step, no `package.json` for the webapp
- **Jinja section wrappers** use `<section id="…" class="max-w-6xl mx-auto px-4 …">` — match this column width for visual alignment
- **Tailwind preflight is active** (loaded from `cdn.tailwindcss.com` with no config override) — wrap MUI components in `Box`/`div` for Tailwind classes; never put Tailwind classes directly on MUI component roots
- **Existing catalog CardActionArea uses `component: 'a'`** for navigation to blueprint detail pages — category cards must OMIT `component` to default to native `<button>` semantics and support `aria-pressed`

### Integration Points

- Jinja template: `app-store-gui/webapp/templates/index.html` — single file, single IIFE
- React mount element: currently `<div id="catalog-root">` at line ~163; rename to `#app-root`
- Existing `items[]` array at lines ~217-251: add `category: '<key>'` field to each of the 5 entries
- `createRoot` call and `root.render(h(Catalog))` at lines ~304-307: replace with `root.render(h(AppShell))`
- URL hash anchors currently in use: `#catalog`, `#planning-studio` — parser for the new `#category=<key>` scheme must use `startsWith('#category=')` plus enum validation against `['aidp', 'warp', 'partner']` to avoid collision
- No existing codebase references to the string `catalog-root` outside `index.html` itself (verify during implementation — grep before renaming)

### Constraints Applying to Every Task

- **JSX forbidden** — grep `<[A-Z]` in new code must return zero matches before sign-off on each plan
- **`component: 'a'` forbidden** on category `CardActionArea` — DOM must show `<button aria-pressed="...">`
- **`history.replaceState` must not fire on mount** — initialization is read-only
- **`ThemeProvider` lives only in AppShell** — must be removed from `Catalog` internals during the refactor plan
- **No new CDN scripts, no `package.json`, no build pipeline**

</code_context>

<deferred>
## Deferred Ideas

Captured here so they're not lost but explicitly out of v4.0 scope. Tracked in REQUIREMENTS.md v4.0 Future Requirements section unless noted.

- Count Chip "N apps" per category card — REQUIREMENTS.md CAT-04 (v4.1)
- Auto-scroll to Categories row on deep-link mount — not yet a REQ-ID; add to v4.1 list if users report the filter being hard to locate after deep-linking from external context
- Explicit "Show all" button when a category is active — REQUIREMENTS.md UX-01 (v4.1)
- 150ms grid opacity fade on category switch — REQUIREMENTS.md UX-02 (v4.1)
- Partner empty-state CTA copy (pending PMM decision) — REQUIREMENTS.md UX-03 (v4.1)
- Spelled-out `aria-label="NeuralMesh AI Data Platform"` on AIDP card — REQUIREMENTS.md A11Y-04 (v4.1)
- Backend externalization of `items[]` array — REQUIREMENTS.md DATA-01 (v4.1+)
- Multi-value category field (array instead of single string) — REQUIREMENTS.md CAT-05 (requires catalog growth beyond ~15 items)

</deferred>

---

*Phase: 15-app-categories-feature*
*Context gathered: 2026-04-21*
