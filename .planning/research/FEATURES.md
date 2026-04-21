# Feature Research

**Domain:** Top-level category filter UI for an internal app-catalog / blueprint-library (WEKA App Store v4.0)
**Researched:** 2026-04-21
**Confidence:** HIGH (toggle semantics, URL convention, count chips, empty states, accessibility); MEDIUM (label convention); LOW (anti-feature prevalence)

---

## Preamble: Research Scope

This document covers only the _new_ feature surface for v4.0: a 3-card category-filter row above the existing flat blueprint grid.
Existing features (flat grid, hero, Planning Studio, tags chips on cards) are out of scope and are treated as fixed context.

The six PRD open questions are addressed in-line within each section.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any mature catalog filter. Missing these makes the feature feel broken or unfinished.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Single-select toggle with visual active state | Every catalog UI with a tab-strip or chip row (GitHub Marketplace, Figma Community, VS Code Marketplace, Hugging Face Hub) shows the selected category as visually distinguished. Users expect exactly one category to be active or none (All). | LOW | `aria-pressed="true"` on the active card satisfies both visual and screen-reader needs. |
| Clicking the active category returns to All | Material Design 3 `FilterChip` spec and the `ChoiceChip` / `ChipGroup(singleSelection=true)` behavior both allow deselection by re-tapping the active chip. The PRD correctly models this. Comparable: Apple Music genre tabs (tap selected genre → returns to "All"), Figma Community (tap active tab → shows all). | LOW | `history.replaceState` with `''` hash clears the URL on toggle-off. One-liner in the click handler. |
| URL hash sync (`#category=<key>`) for deep links | Hugging Face Models uses `?pipeline_tag=` (query string); GitHub Marketplace uses `?category=` (query string). Both are server-rendered pages that need the server to read the parameter. The WEKA App Store filter is **client-side only**, served from a single Flask template with no per-category server route — this is the key distinction. For client-side-only state, hash fragments are the canonical convention: the server never sees the fragment, there are no spurious cache misses, and `hashchange` events drive state natively. `replaceState` (not `pushState`) is correct for filter toggles: it updates the address bar without adding a history entry per click, so one press of Back leaves the page entirely rather than walking through category history. | LOW | Parse on mount: `window.location.hash.startsWith('#category=')`. `history.replaceState(null,'','#category='+key)` on select; `history.replaceState(null,'','')` on deselect. No back-button pollution. |
| Empty state when category has zero matches | LogRocket filtering best-practices and Pencil & Paper enterprise-filter analysis both flag empty results as a critical UX moment. The user must understand why the grid is empty. Static copy is correct for a _known, intentional_ empty category (Partner ships with zero items by design). The PRD's choice of "No apps in this category yet." is the right pattern — it is informational without implying a search error. | LOW | Render a centered `Typography` block inside the grid column. No CTA needed (there is nowhere to navigate TO for a catalog that has no matching items yet). |
| Keyboard accessibility via `aria-pressed` | WCAG 2.2 (now the legal standard in ADA lawsuits as of 2024) requires toggle buttons to communicate state programmatically, not through color alone. `aria-pressed` is the correct attribute for a button that has two states (selected / not selected). Screen readers announce "NeuralMesh AIDP, button, pressed" / "not pressed" automatically when `aria-pressed` is updated. | LOW | MUI `CardActionArea` renders as `role="button"` and is keyboard-focusable; add `aria-pressed={selected === key}` explicitly. |
| Mobile-responsive stacking (3-across on md+, single column on mobile) | The existing catalog grid already stacks on mobile. A horizontal row of 3 cards that does not stack on a 375px viewport would be a regression. Users expect the same responsive contract as the rest of the page. | LOW | Mirror the existing grid's `xs={12} sm={12} md={4}` breakpoints on the Categories MUI Grid. |

**PRD Open Question 4 — Default landing state (All vs. NeuralMesh AIDP):** Table-stakes convention is "All" on first load. Every comparable catalog (GitHub Marketplace, Hugging Face, Figma Community, VS Code Marketplace) defaults to showing everything. Defaulting to a filtered state on first load would surprise users who arrived without a hash in the URL. Recommend: keep **All** as default. If the WEKA field team wants to deep-link to NeuralMesh AIDP for demos, the hash URL (`/#category=neuralmesh-aidp`) handles that without changing the default.

**PRD Open Question 5 — Category order:** Left-to-right order should follow the user's expected hierarchy: WEKA first-party → WEKA second-party product → third-party. "NeuralMesh AIDP → WARP → Partner" is the correct order. This matches how GitHub Marketplace (Apps → Actions → Models) and VS Code Marketplace (Programming Languages → Snippets → Themes) order their category families: first-party core, then extended product lines, then community/partner.

---

### Differentiators (Worth Considering)

Features that distinguish the implementation beyond the baseline. None are required for launch; all are worth scoping with the PRD owner.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Live "N apps" count chip on each category card | The PRD already includes this in the card anatomy (the wireframe shows "3 apps / 2 apps / 0 apps"). LogRocket and Pencil & Paper both recommend count indicators as strong UX — they let users predict what the filter will return before clicking. For a 3-category list the count is computed at render time from the inline `items` array (no API call). Count is always current. | LOW | `items.filter(i => i.category === key).length`. Render as MUI `Chip` inside the card. PRD marks this as a "Should pass" criterion, so it is effectively expected — treat as P1 unless scope is cut. |
| Spelled-out subtitle beneath acronym label | PRD Open Question 2 asks whether "NeuralMesh AIDP" should show the spelled-out form. The pattern on comparable platforms: Hugging Face uses task short names ("Text Classification", not "TC"); VS Code Marketplace uses full category names ("Programming Languages", not "PL"); GitHub Marketplace uses plain English names. For audience-internal acronyms (WEKA field engineers and power users know AIDP), the acronym as a title is acceptable. The differentiating option is: title = acronym ("NeuralMesh AIDP"), subtitle = spelled-out ("AI Data Platform blueprints") — this is exactly what the PRD wireframe already shows as the one-line description. The differentiator worth flagging: a `title` attribute or `aria-label` on the card that reads the full name improves accessibility for external audiences who encounter the page without context. | LOW | `aria-label="NeuralMesh AI Data Platform category"` on the `CardActionArea` provides screen-reader context without changing visible copy. |
| "Show all" explicit affordance (text link) | GitHub Marketplace shows a "Clear filter" link when a category is active. For a 3-card row where All is reachable by re-clicking the active card, an explicit "Show all" link removes the discoverability burden: users do not have to discover that clicking a selected card deactivates it. The PRD mentions "a subtle 'Show all' affordance" as part of the All state description. | LOW | A small grey text link ("Show all") appearing below the category row only when a category is active. Disappears when All is the state. |
| Transition animation on grid filter | When switching categories, the catalog grid re-renders. A 150ms opacity fade on the grid on category change reduces the perceived "flash" of cards appearing/disappearing. GitHub Marketplace and Figma Community both use subtle transitions. | LOW | CSS `transition: opacity 150ms ease` on the grid wrapper. Add a short state-driven class. No library needed. |

**PRD Open Question 3 — Partner empty state copy:** "No apps in this category yet." is the minimal table-stakes copy. The differentiating option is a call-to-action line: "Partner blueprints coming soon — talk to us about contributing." This is worth discussing with PMM/marketing before lock. It converts a dead-end state into a soft CTA. Complexity is LOW (one string change). The recommendation: go with the CTA variant if there is a concrete partnership pipeline; go with the minimal copy if Partner is an internal grouping with no public-facing recruitment intent.

**PRD Open Question 6 — Single `category` value vs. array:** Single value is correct for launch. Every comparable catalog (VS Code Marketplace, GitHub Marketplace, npm) assigns a primary category and optionally a secondary. For a 5-item catalog with 3 categories, the complexity of multi-category membership is not worth the model change. If a blueprint like "OSS RAG" genuinely straddles two families, resolve it by product decision (which family is its primary home?) rather than adding array complexity. Array support is a legitimate v4.1 follow-up once the catalog grows beyond ~15 items.

---

### Anti-Features (Explicitly Do Not Build)

Features that seem reasonable for a category filter but create concrete problems for a 3-category, ~5-item catalog.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Search input within the category filter | Larger catalogs (npm, Docker Hub) have search bars inside filtered views. Users familiar with those surfaces may ask for it. | A search bar over 5 items is absurd and clutters the page above the fold. Pencil & Paper explicitly warns: "Offering advanced filters for a 10-item list adds unnecessary complexity." The catalog grid is already fully visible in one scroll — there is nothing to search. | Add search only when the catalog exceeds ~20 items (a follow-up milestone trigger). |
| Sort controls (A–Z, newest first, most popular) | Sort is table stakes in large-catalog marketplaces (Figma Community, npm). | With 5 items, sort order is meaningless. Sorting 3 cards in a filtered NeuralMesh AIDP view by alphabetical order provides zero navigational value. Sort controls would be a misleading affordance implying depth that does not exist. | Fixed manual ordering in the `items` array controlled by the PRD owner. Externalize `items` to a backend JSON file when the catalog grows. |
| Tooltip descriptions on category cards that duplicate the card subtitle | Tooltips with expanded explanations are common in data-dense UIs (Grafana, Datadog). | The card already has a one-line description. A tooltip that repeats that same description is redundant UI noise. If the label needs explaining, fix the label — do not add a tooltip. Tooltip hover is also inaccessible on touch devices. | The one-line description on the card body IS the tooltip. Keep it visible at all times. |
| "Recently viewed" or "Suggested for you" filter state | Personalization features appear on Apple App Store, Hugging Face. | These require server-side session state or `localStorage`. The PRD explicitly rules out `localStorage` persistence. There is no session model for the catalog page. Building fake personalization (always showing the same "recent" item) misleads users. | Let the URL hash handle "return to last context" — if a field engineer bookmarks `/#category=warp`, the bookmark IS their "recently viewed" workflow. |
| Multi-select filtering (select NeuralMesh AIDP + Partner simultaneously) | Multi-select is powerful in large catalogs (Figma Community allows multiple category+tag combinations). | With 3 mutually exclusive product families, multi-select is semantically incoherent: a blueprint is either a WEKA AIDP app or a Partner app, not both. Multi-select would require union logic, checkbox UI, and a "0 apps match NeuralMesh AIDP AND Partner simultaneously" empty state. The PRD explicitly places this out of scope. | Single-select is the correct model for mutually exclusive product families. |
| `pushState` per click (polluting browser history with each category toggle) | Seems natural for "deep link each state." | Each click generates a back-button entry. Three clicks → three Back presses to leave the page. Users who use Back to leave the catalog will be confused. LogRocket and the Remy Sharp "how tabs should work" article both identify this as a known UX pitfall with hash-based tab state. `replaceState` is the documented solution. | `history.replaceState` (not `pushState`) per the PRD spec and the MDN guidance. One Back press leaves the page. |
| Category-level analytics events per hover | A/B testing infrastructure may want hover data. | Not part of this PRD scope. Adding `onMouseEnter` analytics hooks to individual category cards in an inline script block creates a maintenance burden with no current dashboard to consume the data. | Add analytics only when an analytics platform (Segment, Amplitude, GA4) is actually integrated. Track `click` events if and when that integration arrives. |
| Animated icon or SVG per category card | Icon-based category cards look polished on Hugging Face and GitHub Marketplace. | The WEKA App Store uses text-only glassmorphism cards throughout. Adding per-category SVG icons breaks the visual language of existing catalog cards, requires asset management, and would not inherit from the CDN-based MUI theme. | Rely on the card title typography and the purple selected-state border glow to provide visual differentiation. Icons are a v5 concern when the design system is formalized. |

---

## Feature Dependencies

```
[URL hash sync]
    └──requires──> [client-side category state (React useState)]
                       └──requires──> [category field on each items[] entry]

[Count chip on category card]
    └──requires──> [category field on each items[] entry]

[Empty state message]
    └──requires──> [client-side category state]
                       └──requires──> [category field on each items[] entry]

[aria-pressed selected state]
    └──enhances──> [visual selected border/glow state]

["Show all" explicit affordance (differentiator)]
    └──enhances──> [toggle-off / return to All behavior]

[Grid transition animation (differentiator)]
    └──enhances──> [client-side category state filter]
```

### Dependency Notes

- **`category` field on `items[]` is the atomic prerequisite:** every other feature — filtering, counting, empty states — derives from this single data shape change. It must land first (or simultaneously) with the `Categories` component.
- **URL hash sync requires React state, not the other way around:** The hash is written FROM state (not read into DOM imperatively). Mount reads the hash to initialize state; thereafter React drives the hash via `replaceState`.
- **`aria-pressed` enhances but does not depend on visual state:** even if CSS were broken, a screen reader user would still know which category is selected via `aria-pressed`. The two mechanisms are parallel, not sequential.

---

## MVP Definition

### Launch With (v4.0)

The minimum feature set to validate the concept. All are P1 from the PRD's "Must pass" criteria.

- [x] `category` field added to all 5 `items[]` entries per the PRD mapping table
- [x] `Categories` React component rendering 3 cards above `#catalog`, inside the same `ThemeProvider`
- [x] Single-select toggle: clicking a card sets selected category; clicking the active card returns to All
- [x] Visual selected state (purple border + glow) and unselected dimming (opacity 0.7) matching existing card language
- [x] Client-side grid filter: `items.filter(i => selected === 'all' || i.category === selected)`
- [x] URL hash sync via `history.replaceState` (no `pushState`; one Back press leaves page)
- [x] Empty state message: "No apps in this category yet." for Partner (0 apps)
- [x] `aria-pressed` on `CardActionArea`; keyboard Enter/Space toggles selection
- [x] Mobile-responsive (3-across md+, stacked mobile)

### Add After Validation (v4.1)

Add when catalog grows or business need is confirmed.

- [ ] "Show all" explicit affordance — add when user testing shows toggle-to-deselect is not discoverable enough
- [ ] Grid fade transition (150ms) — add after v4.0 ships if the rerender feels jarring
- [ ] CTA copy for Partner empty state ("Partner blueprints coming soon...") — add when a partnership pipeline exists and PMM confirms the copy
- [ ] `aria-label` spelled-out form on category cards — add if accessibility audit flags acronym labels for external audiences

### Future Consideration (v5+)

Defer until catalog is significantly larger (20+ items) or a backend data source is introduced.

- [ ] Category-level sorting — only meaningful with 10+ items per category
- [ ] Search input — only meaningful with 20+ catalog items
- [ ] Multi-category membership (array `category` field) — only meaningful when blueprints genuinely straddle families
- [ ] Per-category icons / SVG assets — requires formal design-system work
- [ ] Analytics instrumentation — requires analytics platform integration

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Category field on items[] | HIGH | LOW | P1 |
| 3-card Categories row + toggle | HIGH | LOW | P1 |
| Visual selected state (border + glow) | HIGH | LOW | P1 |
| Client-side grid filter | HIGH | LOW | P1 |
| URL hash sync (replaceState) | HIGH | LOW | P1 |
| Empty state message | HIGH | LOW | P1 |
| aria-pressed + keyboard nav | HIGH | LOW | P1 |
| Mobile responsive stacking | HIGH | LOW | P1 |
| Count chip ("N apps") | MEDIUM | LOW | P1 (PRD "Should pass") |
| Unselected opacity dimming | MEDIUM | LOW | P1 (PRD "Should pass") |
| "Show all" explicit affordance | MEDIUM | LOW | P2 |
| Grid fade transition | LOW | LOW | P2 |
| CTA copy for Partner empty state | MEDIUM | LOW | P2 (PMM decision) |
| aria-label with spelled-out name | LOW | LOW | P2 |
| Sort controls | LOW | MEDIUM | P3 (do not build in v4) |
| Search input | LOW | MEDIUM | P3 (do not build in v4) |
| Category icons / SVGs | LOW | MEDIUM | P3 |
| Multi-select | LOW | HIGH | P3 |

---

## Competitor Feature Analysis

| Feature | GitHub Marketplace | Hugging Face Hub | VS Code Marketplace | Our Approach (v4.0) |
|---------|--------------------|--------------------|---------------------|---------------------|
| Category filter mechanism | Left-sidebar category links, `?type=apps&category=ai-assisted` (query string, server-rendered) | Left-sidebar task filter, `?pipeline_tag=summarization` (query string, server-rendered) | In-editor `@category:` search syntax; web uses `?category=` (query string, server-rendered) | Hash fragment `#category=<key>` (client-side only — correct for single-template Flask app with no per-category route) |
| Deselect / return to All | "Clear filter" link appears when a category is active | Re-click task chip to deselect | Clear search query | Toggle: re-click active category card; optionally "Show all" affordance |
| Count on filter labels | No count shown | No count shown on sidebar labels | No count shown | Count chip on card ("N apps") — differentiator that comparable platforms skip but that is easily computed and surfaces value |
| Empty state | Filtered page is simply empty (no explicit message) | Shows "No results found" | No items shown | "No apps in this category yet." — intentional; Partner ships empty on day one by design |
| URL / deep-link method | Query string (server reads it) | Query string (server reads it) | Query string (server reads it) | Hash fragment (browser reads it; server sees only `/`) |
| Accessibility | `aria-current` on active sidebar item | `aria-selected` on active tab | N/A (in-editor UI) | `aria-pressed` on `CardActionArea` toggle buttons |

**Key insight from competitor analysis:** All three comparables use query strings because they are server-rendered pages. The WEKA App Store filter is purely client-side in a single Jinja template. Hash fragments are the correct convention for this architecture — not because the comparables use them, but because the comparables are not analogous in architecture. The correct analogs are client-side SPA tab systems (Gmail, Single-page dashboards with tabs), all of which use hash state.

---

## PRD Open Questions — Research Input Summary

| # | Question | Research Finding | Confidence |
|---|----------|-----------------|------------|
| 1 | Blueprint → category mapping (OSS RAG, NVIDIA VSS placement) | Not a UX research question — content ownership decision. No comparable-platform data applies. | N/A |
| 2 | "NeuralMesh AIDP" vs spelled-out label | Short acronym as title + one-line spelled-out description matches VS Code Marketplace and Figma Community patterns. Add `aria-label` with full name on card action area for screen-reader audiences unfamiliar with the acronym. | HIGH |
| 3 | Partner empty state copy | Minimal "No apps yet" is table stakes; CTA copy ("talk to us") is a differentiator worth adding when partnership pipeline is confirmed. | HIGH |
| 4 | Default landing state | "All" is universal convention. Never filter on first load without a URL parameter. | HIGH |
| 5 | Category order | First-party primary → first-party extended product → third-party. NeuralMesh AIDP → WARP → Partner is correct. | MEDIUM |
| 6 | Single value vs array for `category` | Single value for v4.0. Array is a v4.1+ concern once the catalog grows. | HIGH |

---

## Sources

- [Material Design 3 — Chips guidelines](https://m3.material.io/components/chips/guidelines) — FilterChip / ChoiceChip single-select and deselect behavior
- [MDN — History: replaceState() method](https://developer.mozilla.org/en-US/docs/Web/API/History/replaceState) — replaceState vs pushState back-button implications
- [MDN — Working with the History API](https://developer.mozilla.org/en-US/docs/Web/API/History_API/Working_with_the_History_API) — SPA history management
- [Remy Sharp — How tabs should work](https://remysharp.com/2016/12/11/how-tabs-should-work) — hash-based tab state, hashchange-driven architecture, back button for free
- [DEV — Query Strings vs. Hash Fragments](https://dev.to/zahra_mirkazemi/query-strings-vs-hash-fragments-whats-the-real-difference-597n) — when hash is correct vs. query string; caching implications
- [W3C TAG — Hash in URL usage patterns](https://www.w3.org/2001/tag/doc/hash-in-url) — authoritative W3C guidance on fragment vs. query string conventions
- [LogRocket — Getting filters right: UX/UI design patterns and best practices](https://blog.logrocket.com/ux-design/filtering-ux-ui-design-patterns-best-practices/) — count badges strongly recommended; empty state guidance
- [Pencil & Paper — UX pattern analysis: enterprise filtering](https://www.pencilandpaper.io/articles/ux-pattern-analysis-enterprise-filtering) — "don't add advanced filters to a 10-item list"; count indicators with "(N)" format
- [GitHub Marketplace](https://github.com/marketplace?type=apps) — query string category filtering, no count badges, "Clear filter" affordance
- [Hugging Face Models](https://huggingface.co/models) — `?pipeline_tag=` query string, server-rendered, no count badges on sidebar
- [TestParty — Accessible Toggle Buttons](https://testparty.ai/blog/accessible-toggle-buttons-modern-web-apps-complete-guide) — aria-pressed, WCAG 2.2, screen reader state announcement
- [Accessibility Developer Guide — aria-pressed](https://www.accessibility-developer-guide.com/examples/sensible-aria-usage/pressed/) — "button, pressed" / "not pressed" announcement pattern
- [Smashing Magazine — UI Patterns for Mobile: Search, Sort, Filter](https://www.smashingmagazine.com/2012/04/ui-patterns-for-mobile-apps-search-sort-filter/) — anti-pattern: sort controls for small datasets
- [VS Code Marketplace — category search tips](https://devblogs.microsoft.com/devops/tips-and-tricks-for-search-on-visual-studio-marketplace/) — category URL convention `?category=`

---

*Feature research for: v4.0 App Categories on WEKA App Store Home Screen*
*Researched: 2026-04-21*
