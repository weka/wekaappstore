---
phase: 18-operator-wiring-and-docs
plan: 02
subsystem: docs
tags: [docs, readme, variable-substitution, user-facing]

dependency_graph:
  requires:
    - "Phase 16 render() helper (consumed implicitly by README error semantics)"
    - "Phase 17 CRD admission (referenced in syntax table for invalid-key behavior)"
  provides:
    - "Canonical README section: ## Variable substitution in AppStack manifests"
    - "Phase 20 AIDP migration template (worked example is copy-paste-adapt)"
  affects:
    - "README.md TOC ordering: Common configuration -> Variable substitution -> Upgrading"

tech_stack:
  added: []
  patterns:
    - "GitHub-Flavored Markdown table (2 columns)"
    - "GitHub > **Note:** block-quote callout convention"
    - "Fenced ```yaml code blocks for syntax-highlighted CR examples"

key_files:
  created: []
  modified:
    - "README.md"

decisions:
  - "Section placed between Common configuration and Upgrading (D-06) for natural reading flow when configuring an AppStack CR."
  - "Worked example is AIDP-style multi-component (helm + valuesFiles, kubernetesManifest) using fully-resolved values; the broken PRD pattern milvus.${namespace}.svc.cluster.local appears ONLY in the WRONG snippet (D-08, L-04)."
  - "DOC-05 no-recursion presented as > **Note:** callout immediately followed by side-by-side WRONG/CORRECT fenced YAML pair (D-08); highest reader-signal placement."
  - "DOC-06 hard recommendation rendered as **Recommendation:** Omit targetNamespace ... (D-09); case-insensitive grep matches per W-5 fix."
  - "Error semantics paragraph names BOTH kopf.PermanentError (undefined / malformed / invalid-key) AND kopf.TemporaryError(delay=30) (missing CM/Secret)."

metrics:
  duration_minutes: 4
  files_changed: 1
  insertions: 121
  deletions: 0
  completed_date: "2026-05-08"
  tasks_completed: 1
---

# Phase 18 Plan 02: README Variable Substitution Section Summary

Added a complete user-facing `## Variable substitution in AppStack manifests` section to README.md (DOC-01..06), placed in the natural TOC slot between `## Common configuration` and `## Upgrading`. Authors landing on the repo can now discover, understand, copy, and adapt the feature without reading the operator source.

## What Was Built

One file modified: `README.md`. One new top-level Markdown section inserted; surrounding sections byte-identical except for one separating blank line.

## Insertion Location

- **README.md line 80** — start of the new `## Variable substitution in AppStack manifests` heading.
- **README.md line 200** — end of the section (last bullet of the `### Errors` list).
- **README.md line 201** — `## Upgrading` heading (pushed down from previous line 80; content unchanged).
- **README.md line 39** — `## Common configuration` heading (unchanged).

Final section ordering:

```
39  ## Common configuration
80  ## Variable substitution in AppStack manifests
201 ## Upgrading
```

## Section Structure (D-10 ordering)

1. One-paragraph "why" — portability across namespaces / environments.
2. `### Syntax` reference table (5 rows): `${VAR}`, `$$`, `${namespace}`, undefined `${VAR}`, invalid key.
3. `### Worked example` — full AIDP-style `WekaAppStore` CR (`vector-db` helm component with `valuesFiles` + ConfigMap-loaded `${milvusHost}`; `ingress` `kubernetesManifest` component with `${namespace}`). Followed by a same-blueprint-different-namespace example.
4. `### Variable values are NOT recursively resolved` — `> **Note:**` callout immediately followed by separate `# WRONG` / `# CORRECT` fenced YAML blocks. WRONG shows the broken `milvus.${namespace}.svc.cluster.local` nested-`${}` pattern; CORRECT shows fully-resolved `milvus.aidp-prod.svc.cluster.local`.
5. `### Operator-control fields are NOT templated` — bullet list of fields (`helmChart.repository`, `helmChart.name`, `helmChart.version`, `releaseName`, `targetNamespace`, `readinessCheck.*`) plus the hard `**Recommendation:** Omit \`targetNamespace\`` paragraph.
6. `### Errors` — bullets covering: undefined variable → `kopf.PermanentError`; missing referenced ConfigMap/Secret → `kopf.TemporaryError(delay=30)`; malformed `${...}` → `kopf.PermanentError`; invalid variable key → `kopf.PermanentError`.

## Acceptance Anchors (downstream verification)

The six DOC-01..06 anchor families that `gsd-verify-work` and Phase 18 success-criterion 5 will check. All passed at commit time:

| Requirement | Anchor grep | Threshold | Actual |
|-------------|-------------|-----------|--------|
| DOC-01 | `grep -c '\${milvusHost}' README.md` | >= 2 | 2 |
| DOC-02 | `grep -c '\`\$\$\`' README.md` | >= 1 | 1 |
| DOC-03 | `grep -c '\${namespace}' README.md` | >= 3 | 4 |
| DOC-04 (Permanent) | `grep -c 'kopf.PermanentError' README.md` | >= 2 | 4 |
| DOC-04 (Temporary) | `grep -c 'kopf.TemporaryError' README.md` | >= 1 | 1 |
| DOC-05 (Note callout) | `grep -c '> \*\*Note:\*\* Variable values are taken literally' README.md` | == 1 | 1 |
| DOC-05 (WRONG snippet) | `grep -c '# WRONG' README.md` | >= 1 | 1 |
| DOC-05 (CORRECT snippet) | `grep -c '# CORRECT' README.md` | >= 1 | 1 |
| DOC-05 (broken nested-${} in WRONG) | `grep -c 'milvus\.\${namespace}\.svc\.cluster\.local' README.md` | == 1 | 1 |
| DOC-05 (fully-resolved in CORRECT) | `grep -c 'milvus\.aidp-prod\.svc\.cluster\.local' README.md` | >= 1 | 3 |
| DOC-06 (helmChart.repository) | `grep -c 'helmChart\.repository' README.md` | >= 1 | 1 |
| DOC-06 (releaseName) | `grep -c 'releaseName' README.md` | >= 1 | 1 |
| DOC-06 (readinessCheck) | `grep -c 'readinessCheck' README.md` | >= 1 | 4 |
| DOC-06 (Omit targetNamespace, case-insensitive per W-5) | `grep -ci 'omit \`targetNamespace\`' README.md` | >= 1 | 1 |
| Section heading exact | `grep -c '^## Variable substitution in AppStack manifests$' README.md` | == 1 | 1 |
| TOC ordering | `awk` line-numbers | Common < Variable < Upgrading | 39 < 80 < 201 |

## Confirmation: No Out-of-Section Modifications

`git diff README.md` review (1 file, 121 insertions, 0 deletions):

- The only change is the insertion of the new section between the trailing blank line of `## Common configuration` and the `## Upgrading` heading.
- No `-` lines deleting `## Common configuration` or `## Upgrading` (`git diff README.md | grep -c '^-## Common configuration'` == 0; same for `^-## Upgrading`).
- Surrounding sections (Quick start, Common configuration, Upgrading, Uninstalling, Troubleshooting, Readiness checks, Publishing) are byte-identical to pre-commit content.

## Threat Mitigation (T-18-04)

Worked example uses placeholder variable names (`${milvusHost}`, `${namespace}`, `${unset}`) only. No realistic-looking secret-bearing names (no `${apiKey}`, `${dbPassword}`, etc.). The `$$` table row references a database-password-starting-with-`$` use case abstractly without modeling a credential format that authors would copy.

## Deviations from Plan

None — plan executed exactly as written. Section text in the plan's `<action>` block was used verbatim, including:

- The `> **Note:**` block-quote on its own paragraph (blank line above and below) so GitHub renders the callout.
- Separate fenced YAML blocks for WRONG and CORRECT snippets, each with the comment line as the first line of the rendered code (NOT a Markdown heading).
- Capital-O "**Recommendation:** Omit `targetNamespace`" — case-insensitive grep gate per W-5 accepts either capitalization; capital-O matches the surrounding bold-header convention.

## Commits

- `70a0977` — `docs(18-02): add Variable substitution in AppStack manifests README section` (Task 1; README.md +121/-0)

## Self-Check: PASSED

- README.md: FOUND (modified, 1 file changed, +121/-0)
- Commit `70a0977`: FOUND in `git log --oneline`
- All 16 acceptance grep anchors green (verified above)
- No STATE.md or ROADMAP.md modifications (parallel-executor contract honored)
- No operator code modifications (Phase 18-02 is README-only)
