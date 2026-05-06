# Phase 17: CRD Schema Additive Update - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Add an optional `spec.appStack.variables` map to the `WekaAppStore` CRD at `weka-app-store-operator-chart/templates/crd.yaml` (sibling of `components` under `appStack`), with `additionalProperties: { type: string }` for value-type enforcement, `propertyNames: { pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$" }` for key-name enforcement at admission, and a multi-line description covering `${VAR}` syntax, `$$` escape, `${namespace}` auto-default, and identifier-name requirement. Bump chart version `0.1.61 → 0.1.62`. Ship a verification script (`weka-app-store-operator-chart/scripts/verify-crd.sh`) that proves admission rules work via `helm template` + `kubectl --dry-run=server` + `kubectl explain` keyword grep.

**No operator code changes.** Phase 17 is independently deployable in parallel with Phase 16. Phase 18 wires `render()` into the reconcile flow once both are deployed. Phase 17 does NOT publish the new chart `.tgz` to `docs/` — publishing is deferred to end-of-v5.0.

Requirements covered: CRD-01, CRD-02, CRD-03.

</domain>

<decisions>
## Implementation Decisions

### Variable Key Name Enforcement (defense in depth)
- **D-01:** CRD enforces variable key names at admission via `propertyNames: { pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$" }`. Hyphenated keys like `my-host` are rejected by `kubectl apply` immediately with a schema error.
- **D-02:** Pattern is `^[_a-zA-Z][_a-zA-Z0-9]*$` (Python identifier — exact match for `string.Template`'s internal rule). Keys passing admission are guaranteed to render in Phase 18.
- **D-03:** Phase 18's OP-10 operator-level key check is KEPT unchanged (belt + suspenders). Catches the rare case where admission is bypassed (apiserver misconfig, raw API writes, etc.). Cost: ~3 lines of operator code; trivial.

### Description Format
- **D-04:** Multi-line block scalar (`description: |`) covering all four CRD-02 keywords: `${VAR}` syntax, `$$` escape, `${namespace}` auto-default, identifier-name requirement.
- **D-05:** Locked description text (verifier reads this verbatim):
  ```
  description: |
    Map of variable name → string value. Substituted as ${VAR} into:
      - kubernetesManifest strings
      - The raw content of ConfigMap/Secret referenced by valuesFiles
    The variable ${namespace} auto-defaults to the CR's metadata.namespace
    if not explicitly set. Use $$ to escape a literal dollar sign.
    Undefined references raise a permanent error.
    Variable names must match Python identifier syntax: [_a-zA-Z][_a-zA-Z0-9]*.
  ```
  PRD's proposed text + one explicit identifier-name line. Satisfies CRD-02 SC#4 (`kubectl explain` keyword grep) word-for-word.

### Schema Structure
- **D-06:** `variables:` lives at `spec.versions[0].schema.openAPIV3Schema.properties.spec.properties.appStack.properties.variables` (sibling of `components`). Confirmed insertion point: line ~89 of current crd.yaml, immediately after `components:` block ends.
- **D-07:** Schema is `type: object` + `description: |` (above) + `propertyNames: { pattern: ... }` (D-02) + `additionalProperties: { type: string }` (CRD-03 — string-only values, no `x-kubernetes-preserve-unknown-fields`).
- **D-08:** `variables` is NOT in `required:` — existing CRs without it continue to pass admission (CRD-01 SC#5).

### Validation Methodology
- **D-09:** New verification script at `weka-app-store-operator-chart/scripts/verify-crd.sh`. Renders the chart via `helm template`, then runs `kubectl apply --dry-run=server` against an existing live cluster (the user's EKS from v3.0).
- **D-10:** Script tests 4 inline CR fixtures (heredocs, no separate fixture files):
  1. **Valid:** `variables: {namespace: foo, milvusHost: milvus.foo.svc.cluster.local}` — expect PASS
  2. **Integer value:** `variables: {count: 42}` — expect FAIL with type error (assert stderr contains substring like `Invalid value` and `string`)
  3. **Hyphenated key:** `variables: {my-host: foo}` — expect FAIL with propertyNames pattern error (assert stderr contains substring like `propertyNames` or `pattern`)
  4. **No variables block:** existing-style CR with just `components:` — expect PASS (backward-compat; CRD-01 SC#5)
- **D-11:** Script defaults to `kubectl --dry-run=server` (safe to re-run); `--apply` flag opts in to actually installing the CRD on the live cluster (required for `kubectl explain` since explain queries the live apiserver).
- **D-12:** Script also runs `kubectl explain wekaappstores.spec.appStack.variables` (in `--apply` mode only) and greps the output for required keywords: `${VAR}`, `$$`, `${namespace}`, `identifier`. Asserts CRD-02 SC#4 in code, not just visually.
- **D-13:** Script exits non-zero on any unexpected outcome (PASS where FAIL expected, FAIL where PASS expected, missing keyword in explain output). All four cases run before exit so output captures all failures, not just the first.

### Chart Versioning
- **D-14:** Bump `Chart.yaml` `version: 0.1.61` → `0.1.62` in this phase. Patch bump signals additive backward-compat change. Phase 18 will bump again (likely 0.1.63) when render() wiring lands.
- **D-15:** Do NOT publish the new chart `.tgz` to `docs/` in this phase. No `helm package`, no `helm repo index`, no `docs/` mutations. Publishing batches to end-of-v5.0 (after Phase 18 wiring) or whenever the user runs the existing publish workflow manually.
- **D-16:** Do NOT add a CHANGELOG.md or modify `Chart.yaml description:` field. Release notes deferred to a future docs/release-prep phase. The git commit message + version bump is the historical record.

### Claude's Discretion
- Exact placement of `variables:` block within the rendered schema (alphabetical or grouped with related fields — Claude picks; the block goes right after `components:` for proximity).
- Whitespace/indent style matching the rest of `crd.yaml` (existing file uses 2-space indent — match).
- Script exit-code conventions (any non-zero is fail; specific code-per-case is Claude's choice).
- The exact stderr-substring matches for the dry-run failure cases — Kubernetes' admission error messages can vary slightly between server versions; pick robust substrings (`Invalid value` / `string` for type, `propertyNames` / `pattern` for key).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Specs
- `.planning/PRD-appstack-variable-substitution.md` §"CRD Change" — proposed schema verbatim; description text source.
- `.planning/REQUIREMENTS.md` — CRD-01, CRD-02, CRD-03 (Phase 17 mappings).
- `.planning/ROADMAP.md` §"Phase 17: CRD Schema Additive Update" — 5 success criteria.
- `.planning/STATE.md` §"Key Architectural Decisions (v5.0)" — single-pass invariants, Phase 18 wiring contract.

### Code to Touch
- `weka-app-store-operator-chart/templates/crd.yaml` (258 lines today; Helm template gated on `{{- if .Values.customResourceDefinition.create }}`; insertion point for `variables:` is sibling of `components:` at appStack property indent — line ~89 area).
- `weka-app-store-operator-chart/Chart.yaml` (line 18: `version: 0.1.61` → `0.1.62`).
- `weka-app-store-operator-chart/scripts/verify-crd.sh` (NEW — verification script).

### Code to Reference (do NOT modify)
- `cluster_init/app-store-cluster-init.yaml` — sample CR currently in production; verify it still applies cleanly with new CRD (backward-compat).
- `mcp-server/tests/fixtures/sample_blueprints/*.yaml` — additional reference CRs the verifier may want to dry-run.
- `operator_module/main.py:render()` (Phase 16) — the function whose syntax this CRD locks (`${VAR}`, `$$`, identifier).

### Phase 16 Deliverable (already shipped)
- `.planning/phases/16-render-helper-and-test-scaffolding/16-VERIFICATION.md` — confirms render() exists and the contract Phase 17 must align with.

### Codebase Maps
- `.planning/codebase/STRUCTURE.md` — repo layout; chart lives at `weka-app-store-operator-chart/`.
- `.planning/codebase/TESTING.md` — confirms no CI exists; explains why Phase 17 ships a manual-verification script rather than relying on CI.

### Operational Reference
- `README.md` §"Publishing (maintainers)" — publish workflow (`helm package` + `helm repo index`) — INTENTIONALLY NOT exercised in this phase per D-15.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `weka-app-store-operator-chart/templates/crd.yaml` lines 85–192 — current `appStack` schema; the new `variables:` block follows the same indent and description-style convention used by `components:`, `helmChart:`, `valuesFiles:` siblings.
- `weka-app-store-operator-chart/Chart.yaml` — semver convention already established (currently 0.1.61); patch bumps are how prior incremental changes shipped.
- README's "Publishing (maintainers)" section documents `helm package` + `helm repo index` — Phase 17 does NOT use these; reserved for end-of-milestone.

### Established Patterns
- Multi-component CRDs in this repo use `description: |` (block scalar) for any description longer than ~50 chars (see line 79's `targetNamespace` description style).
- `additionalProperties: { type: string }` is already used elsewhere in the schema (line 184: `matchLabels.additionalProperties.type: string`).
- `propertyNames` is NOT yet used in this repo — Phase 17 introduces it. Kubernetes 1.16+ supports it; EKS clusters are well past this.

### Integration Points
- The new `variables:` block must NOT interfere with the existing `components:` array shape — sibling at the same `appStack.properties` indent.
- After Phase 17 ships, Phase 18 will read `spec.appStack.variables` in `handle_appstack_deployment` (existing function at line 551 of `operator_module/main.py`) and pass it to render(). No CRD work needed in Phase 18.
- Chart users running v0.1.61 with no `variables:` field will continue working unchanged after upgrading to 0.1.62 (CRD-01 SC#5).

### Codebase Constraints
- Helm templates use Go-template syntax (`{{- if ... }}`); the CRD itself is templated only at the file boundary. The schema body is pure YAML once rendered.
- No CRD migration tooling in the repo. Backward-compat is enforced solely by the additive (non-required) nature of the change.

</code_context>

<specifics>
## Specific Ideas

- **Exact `variables:` block to add (locked):**
  ```yaml
  variables:
    type: object
    description: |
      Map of variable name → string value. Substituted as ${VAR} into:
        - kubernetesManifest strings
        - The raw content of ConfigMap/Secret referenced by valuesFiles
      The variable ${namespace} auto-defaults to the CR's metadata.namespace
      if not explicitly set. Use $$ to escape a literal dollar sign.
      Undefined references raise a permanent error.
      Variable names must match Python identifier syntax: [_a-zA-Z][_a-zA-Z0-9]*.
    propertyNames:
      pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$"
    additionalProperties:
      type: string
  ```
  Inserted as the new last sibling under `appStack.properties` (after `components:` ends).

- **Verify-crd.sh outline (Claude fills in details):**
  - Usage: `./weka-app-store-operator-chart/scripts/verify-crd.sh [--apply]`
  - Dry-run mode (default): renders CRD, runs 4 fixture dry-runs, reports per-case PASS/FAIL with a final exit code
  - `--apply` mode: also applies the CRD to the cluster, then runs `kubectl explain wekaappstores.spec.appStack.variables` and greps for required keywords
  - Output format: one line per case (`✓ valid CR accepted`, `✗ integer-value CR unexpectedly accepted`, etc.) — easy to paste into PR description as evidence

- **Kubernetes API server requirement:** `propertyNames` works on K8s 1.16+; EKS clusters are well past this — no version gate needed.

</specifics>

<deferred>
## Deferred Ideas

- **Publishing the chart `.tgz` to `docs/`** (D-15) — runs at end of v5.0 milestone or via the existing manual publish workflow.
- **CHANGELOG.md** (D-16) — future docs/release-prep phase; current commit messages are the historical record.
- **README user-facing variable substitution docs** — locked to Phase 18 (DOC-01..06); explicitly out of Phase 17 scope.
- **CI/automated CRD schema tests** (e.g., `openapi-schema-validator`-based unit tests) — repo has no CI today; future test-infra phase.
- **`status.appStackVariables` observability field** — already deferred to v51-02 (PRD Open Q3) per `.planning/REQUIREMENTS.md`.
- **Templating `targetNamespace` / other operator-control fields** — already deferred to v51-01.
- **Default-value syntax (`${VAR:-default}`)** — already deferred to v51-03.

### Reviewed Todos (not folded)
None — discussion stayed within phase scope.

</deferred>

---

*Phase: 17-crd-schema-additive-update*
*Context gathered: 2026-05-06*
