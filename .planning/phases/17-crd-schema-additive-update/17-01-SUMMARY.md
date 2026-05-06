---
phase: 17-crd-schema-additive-update
plan: 01
subsystem: weka-app-store-operator-chart
tags:
  - crd
  - schema
  - helm
  - admission
  - verification
requirements:
  - CRD-01
  - CRD-02
  - CRD-03
dependency_graph:
  requires:
    - "weka-app-store-operator-chart/templates/crd.yaml (existing appStack.properties insertion site at line 192)"
    - "weka-app-store-operator-chart/Chart.yaml (existing version: 0.1.61)"
    - "scripts/validate-phase13.sh (analog scaffold for shell verifier)"
  provides:
    - "spec.appStack.variables OpenAPI v3 schema with type:object + propertyNames pattern + additionalProperties:string"
    - "Chart version 0.1.62 (patch bump signaling additive backward-compat change)"
    - "weka-app-store-operator-chart/scripts/verify-crd.sh (executable admission/explain verifier)"
  affects:
    - "Phase 18 render() wiring (handle_appstack_deployment will consume spec.appStack.variables once Phase 18 lands; OP-10 unchanged per D-03)"
tech-stack:
  added: []
  patterns:
    - "OpenAPI v3 propertyNames pattern (first use in this repo's CRD)"
    - "OpenAPI v3 additionalProperties:string value-type enforcement (already used at matchLabels lines 184-185)"
    - "Block-scalar description (description: |) — first use in crd.yaml; standard YAML 1.2 indent contract"
    - "Single-quoted heredoc fixtures (<<'YAML') to preserve ${VAR} / $$ literals through shell expansion"
key-files:
  created:
    - "weka-app-store-operator-chart/scripts/verify-crd.sh (202 lines, 0755)"
  modified:
    - "weka-app-store-operator-chart/templates/crd.yaml (+14 lines, -0; strictly additive)"
    - "weka-app-store-operator-chart/Chart.yaml (+1 line, -1; version bump only)"
decisions:
  - "Strictly additive crd.yaml diff (14/0). Restored the original 10-space-trailing blank line on line 193 after the editor normalized it, so git diff is insertions-only."
  - "Precheck added to verify-crd.sh: refuses to run (exit 2) if cluster CRD lacks spec.appStack.variables, preventing misleading 'unknown field' errors when fixtures would otherwise execute against the old schema."
  - "kubectl explain keyword check uses grep -qF (fixed-string) so ${VAR} and $$ are not regex-interpreted (D-12)."
  - "Single-quoted heredocs (<<'YAML') used for all four fixtures so ${VAR} and $$ literals reach the apiserver unmodified (PATTERNS.md Pattern 6)."
metrics:
  duration: "~3 minutes wall-clock"
  completed: "2026-05-06T13:44:08Z"
  tasks_total: 3
  tasks_completed: 3
  files_created: 1
  files_modified: 2
  commits: 3
---

# Phase 17 Plan 01: CRD Schema Additive Update Summary

**One-liner:** Adds optional `spec.appStack.variables` map to the WekaAppStore CRD with admission-time key/value enforcement, bumps chart to 0.1.62, and ships a self-contained `verify-crd.sh` that proves admission rules and `kubectl explain` documentation are correct — all in a strictly additive, backward-compatible, schema-only change with no operator code, no chart publishing, and no docs mutation.

## Files Modified

### `weka-app-store-operator-chart/templates/crd.yaml` (+14 / -0)

Inserted the new `variables:` block at line 193 (immediately after `default: 300` closes the `readinessCheck.timeout` field), at 18-space indent — same column as `components:` on line 89. Diff hunk:

```diff
@@ -190,6 +190,20 @@ spec:
                               type: integer
                               description: "Timeout in seconds (default: 300)"
                               default: 300
+                  variables:
+                    type: object
+                    description: |
+                      Map of variable name → string value. Substituted as ${VAR} into:
+                        - kubernetesManifest strings
+                        - The raw content of ConfigMap/Secret referenced by valuesFiles
+                      The variable ${namespace} auto-defaults to the CR's metadata.namespace
+                      if not explicitly set. Use $$ to escape a literal dollar sign.
+                      Undefined references raise a permanent error.
+                      Variable names must match Python identifier syntax: [_a-zA-Z][_a-zA-Z0-9]*.
+                    propertyNames:
+                      pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$"
+                    additionalProperties:
+                      type: string
           
           status:
             type: object
```

- D-04, D-05: block-scalar description containing all four CRD-02 keywords (`${VAR}`, `$$`, `${namespace}`, `identifier`).
- D-01, D-02: `propertyNames.pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$"` enforces Python identifier syntax for keys at admission.
- CRD-03, D-07: `additionalProperties.type: string` enforces string-only values; no `x-kubernetes-preserve-unknown-fields` in the new block (file-wide count remains 2, both pre-existing).
- D-08, CRD-01 SC#5: `variables` is NOT added to any `required:` array; existing CRs continue to validate identically.
- D-06: block lives at `spec.versions[0].schema.openAPIV3Schema.properties.spec.properties.appStack.properties.variables` (sibling of `components:`).

**Commit:** `ea44d6d` — `feat(17-01): add spec.appStack.variables to CRD schema`

### `weka-app-store-operator-chart/Chart.yaml` (+1 / -1)

Diff hunk:

```diff
@@ -15,7 +15,7 @@ type: application
 # This is the chart version. This version number should be incremented each time you make changes
 # to the chart and its templates, including the app version.
 # Versions are expected to follow Semantic Versioning (https://semver.org/)
-version: 0.1.61
+version: 0.1.62
 
 # This is the version number of the application being deployed.
```

- D-14: patch bump signaling additive backward-compatible change.
- D-16: no `description:` edit (line 3 byte-identical), no `appVersion:` bump (line 24 byte-identical), no CHANGELOG.md.
- D-15: no `helm package`, no `helm repo index`, no `docs/` mutations.
- Bare semver (no quotes) preserves the file's existing `version:` convention.

**Commit:** `81d86ed` — `chore(17-01): bump chart version 0.1.61 -> 0.1.62`

## Files Created

### `weka-app-store-operator-chart/scripts/verify-crd.sh` (NEW — 202 lines, mode 0755)

New executable shell script. Created the `weka-app-store-operator-chart/scripts/` directory (did not previously exist).

Structure (mirroring `scripts/validate-phase13.sh` per PATTERNS.md):
- Shebang `#!/usr/bin/env bash` + header comment + `set -euo pipefail`.
- Argument parser accepts only `--apply`; any other flag prints "Unknown flag: …" and exits 1.
- `SCRIPT_DIR` / `CHART_DIR` resolution from `BASH_SOURCE[0]`.
- `--apply` step 0: `helm template "${CHART_DIR}" --show-only templates/crd.yaml | kubectl apply -f -`.
- Precheck: refuses to run (exit 2) if cluster CRD lacks `spec.appStack.variables`. Prevents fixtures producing misleading "unknown field" errors against the old schema.
- `run_dry_run_case` helper accumulates PASS/FAIL across cases.
- Four single-quoted heredoc fixtures (D-10):
  1. `variables: {namespace: foo, milvusHost: milvus.foo.svc.cluster.local}` → expect PASS (CRD-03 SC#2).
  2. `variables: {count: 42}` → expect FAIL with `Invalid value`/`string`/`type` error (CRD-03 SC#3).
  3. `variables: {my-host: foo}` → expect FAIL with `propertyNames`/`pattern` error (D-01, D-02; D-10 case 3).
  4. Existing-style CR with `components:` only → expect PASS (CRD-01 SC#5; D-08).
- All four cases run before exit (D-13).
- `--apply` mode also runs `kubectl explain wekaappstores.spec.appStack.variables` and asserts each of the four CRD-02 keywords (`${VAR}`, `$$`, `${namespace}`, `identifier`) is present using `grep -qF` (D-12).
- Final summary block exits 0 only when `FAIL=0`.

**Commit:** `d2c7a61` — `feat(17-01): add verify-crd.sh phase-17 admission verifier`

## Verification Commands Run and Results

### Task 1 — `crd.yaml` insertion

| Check | Command | Result |
|-------|---------|--------|
| Indent parity (variables vs components) | `grep -nE '^( *)variables:$\|^( *)components:$' weka-app-store-operator-chart/templates/crd.yaml` | line 89 (components) and line 193 (variables) both at 18-space indent — PASS |
| variables: 18-space indent | `grep -c '^                  variables:$'` | 1 — PASS |
| propertyNames: 20-space indent | `grep -c '^                    propertyNames:$'` | 1 — PASS |
| pattern: 22-space indent | `grep -c '^                      pattern: "\^\[_a-zA-Z\]\[_a-zA-Z0-9\]\*\$"$'` | 1 — PASS |
| `description: |` count | `grep -c 'description: \|'` | 1 — PASS (only the new block uses block-scalar) |
| D-05 keyword: `${VAR}` | `grep -c 'Substituted as \${VAR} into:'` | 1 — PASS |
| D-05 keyword: `$$` | `grep -c 'Use \$\$ to escape a literal dollar sign\.'` | 1 — PASS |
| D-05 keyword: `${namespace}` | `grep -c '\${namespace} auto-defaults'` | 1 — PASS |
| D-05 keyword: `identifier` | `grep -c 'Python identifier syntax'` | 1 — PASS |
| `x-kubernetes-preserve-unknown-fields` total (D-07) | `grep -c 'x-kubernetes-preserve-unknown-fields'` | 2 — PASS (unchanged) |
| New block has none | `grep -A12 '^                  variables:$' \| grep -c 'x-kubernetes-preserve-unknown-fields'` | 0 — PASS |
| No new top-level `required:` under appStack | `grep -E '^                required:'` | (no output) — PASS |
| `helm template weka-app-store-operator-chart >/dev/null` | helm v4.1.4 | exit 0 — PASS |
| Strictly additive diff | `git diff --numstat` | `14 0 weka-app-store-operator-chart/templates/crd.yaml` — PASS |
| `operator_module/main.py` untouched | `git status --porcelain operator_module/main.py` | empty — PASS |

### Task 2 — Chart.yaml version bump

| Check | Command | Result |
|-------|---------|--------|
| New version present | `grep -c '^version: 0\.1\.62$'` | 1 — PASS |
| Old version absent | `grep -c '^version: 0\.1\.61$'` | 0 — PASS |
| `appVersion:` unchanged | `grep -c '^appVersion: "1\.16\.0"$'` | 1 — PASS |
| `description:` unchanged | `grep -c '^description: A Helm chart to install the Weka App Store Operator and its web GUI$'` | 1 — PASS |
| Diff numstat | `git diff --numstat` | `1 1 …Chart.yaml` — PASS |
| Diff content | `git diff` | exactly two lines: `-version: 0.1.61` / `+version: 0.1.62` — PASS |
| `docs/` untouched | `git status --porcelain docs/` | empty — PASS |
| `CHANGELOG.md` untouched | `git status --porcelain CHANGELOG.md` | empty — PASS |

### Task 3 — `verify-crd.sh`

| Check | Command | Result |
|-------|---------|--------|
| File exists | `test -f …/verify-crd.sh` | OK |
| Executable bit set | `test -x …/verify-crd.sh` | OK |
| Shebang | `head -1` | `#!/usr/bin/env bash` — PASS |
| `set -euo pipefail` count | grep | 1 — PASS |
| `--apply) APPLY_MODE=true` count | grep | 1 — PASS |
| `<<'YAML'` heredoc count | grep | 4 — PASS |
| `kubectl apply --dry-run=server` count | grep | 1 — PASS (≥1 required) |
| `kubectl explain wekaappstores` count | grep | 2 — see Deviations §1 |
| All four explain keywords present | grep -F per keyword | each = 1 — PASS |
| `Invalid value` substring (case 2) | grep | 1 — PASS |
| `propertyNames` substring (case 3) | grep | 2 (one in case 3 match pattern, one in precheck error msg) — PASS (≥1 required) |
| Fixture `milvus.foo.svc.cluster.local` (case 1) | grep | 1 — PASS |
| Fixture `count: 42` (case 2) | grep | 1 — PASS |
| Fixture `my-host: foo` (case 3) | grep | 1 — PASS |
| Forbidden: `helm package` | grep | 0 — PASS |
| Forbidden: `helm repo index` | grep | 0 — PASS |
| Forbidden: `docs/` | grep | 0 — PASS |
| `bash -n` syntax check | `bash -n …/verify-crd.sh` | exit 0 — PASS |
| Bogus flag rejection | `bash …/verify-crd.sh --bogus 2>&1; echo "EXIT=$?"` | "Unknown flag: --bogus" + "EXIT=1" — PASS |
| Line count | `wc -l` | 202 — PASS (≥80 required) |

### Phase boundary verification (CLAUDE.md rule "no operator/Phase 16/18/19 territory touched")

| Path | `git diff main..HEAD` | Result |
|------|------------------------|--------|
| `operator_module/main.py` | empty | OK |
| `operator_module/tests/` | empty | OK |
| `mcp-server/tools/validate_yaml.py` | empty | OK |
| `docs/` | empty | OK |
| `CHANGELOG.md` | empty | OK (file does not exist) |

## Deviations from Plan

### 1. `kubectl explain wekaappstores` count returned 2, plan acceptance criterion expected 1 — kept the locked script body as-is

**Rule:** Plan-internal contradiction; preserved the substantive intent (script invokes `kubectl explain wekaappstores...`) at the cost of the literal grep-counter equality check.

**Found during:** Task 3 acceptance check.

**Issue:** The plan's `<action>` block locked the script body verbatim, including a header comment line `# kubectl explain wekaappstores.spec.appStack.variables`. That comment plus the actual command line at `EXPLAIN_OUTPUT="$(kubectl explain wekaappstores.spec.appStack.variables 2>&1 || true)"` produces 2 matches for `grep -c 'kubectl explain wekaappstores'`. The plan's automated verify (line 558) and acceptance criterion (line 580) both expect exactly 1.

**Fix:** None — the substantive contract ("the script invokes `kubectl explain wekaappstores`") is satisfied. The header comment is part of the locked script body the plan required. Removing the comment would deviate from the locked body. Removing the substantive command would break D-12. The mismatched grep arithmetic in the acceptance criterion is a planning-time error, not an executor error.

**Files modified:** none (script kept as-locked).

**Commit:** N/A.

### 2. Restored a 10-space-trailing-whitespace blank line in crd.yaml after editor normalization

**Rule:** Rule 1 (cosmetic — preserve strictly-additive diff invariant).

**Found during:** Task 1 post-edit diff review.

**Issue:** When the Edit tool inserted the new block, it normalized the original line 193 (a blank line containing only 10 trailing spaces) to a fully-empty blank line. While functionally identical YAML, the change violated the acceptance criterion "git diff shows ONLY additive lines (no removals, no modifications outside the inserted block)".

**Fix:** Used Python to re-insert the 10-space-trailing blank line so the resulting diff is `14 0` (14 added, 0 removed). Verified via `git diff --numstat`.

**Files modified:** `weka-app-store-operator-chart/templates/crd.yaml`.

**Commit:** `ea44d6d` (rolled into Task 1 commit).

## Authentication Gates

None encountered. The script's `--apply` mode (which would require `kubectl` cluster access and is the only externally-authenticated step) was NOT exercised at execution time per the plan note that "live-cluster kubectl access is not guaranteed at execution time" — only the script's structural integrity was verified via `bash -n` and `--bogus` flag rejection. Recommend running `bash weka-app-store-operator-chart/scripts/verify-crd.sh --apply` against the user's EKS cluster post-merge to satisfy ROADMAP Phase 17 success criteria 1, 2, 3, and 4 end-to-end.

## --apply mode exercised at execution time?

No. Live-cluster kubectl access not assumed during executor run. The structural-integrity checks (`bash -n` syntax check, `--bogus` flag rejection, 4 heredoc count, all-four explain-keyword call sites, no forbidden tokens) were exercised and passed. The behavioral assertions (Cases 1–4 dry-run results and the explain-keyword grep) require a live cluster and are deferred to manual run by the user post-merge.

## Decision Reference Map

| Decision | Where applied |
|----------|---------------|
| D-01 (CRD admission key-name enforcement) | crd.yaml `propertyNames.pattern`; verify-crd.sh case 3 expected substring |
| D-02 (exact `^[_a-zA-Z][_a-zA-Z0-9]*$` pattern) | crd.yaml `propertyNames.pattern` quoted regex |
| D-03 (Phase 18 OP-10 unchanged) | `operator_module/main.py` untouched (boundary check confirms) |
| D-04 (block-scalar description) | crd.yaml `description: \|` |
| D-05 (locked description text) | crd.yaml verbatim 7-line description body |
| D-06 (insertion location) | crd.yaml `variables:` at 18-space indent under `appStack.properties` |
| D-07 (string-only values, no preserve-unknown) | crd.yaml `additionalProperties.type: string`; no `x-kubernetes-preserve-unknown-fields` in new block |
| D-08 (variables not required) | No `required:` field added under `appStack` |
| D-09 (helm template + kubectl --dry-run=server) | verify-crd.sh `kubectl apply --dry-run=server -f -` per case |
| D-10 (4 fixtures via heredocs) | verify-crd.sh four `<<'YAML'` heredoc fixtures |
| D-11 (default = dry-run; --apply opts in) | verify-crd.sh `APPLY_MODE` toggle |
| D-12 (explain keyword grep for ${VAR}, $$, ${namespace}, identifier) | verify-crd.sh `check_explain_keyword` calls |
| D-13 (all cases run before exit) | verify-crd.sh `PASS`/`FAIL` accumulators + final summary block |
| D-14 (Chart.yaml 0.1.61 → 0.1.62) | Chart.yaml line 18 |
| D-15 (no helm package / repo index / docs/) | verify-crd.sh contains none; Chart.yaml line bump only; `docs/` untouched |
| D-16 (no CHANGELOG, no description: edit, no appVersion: bump) | only `version:` line touched in Chart.yaml |

## Self-Check: PASSED

- `weka-app-store-operator-chart/templates/crd.yaml`: FOUND (modified, +14/-0)
- `weka-app-store-operator-chart/Chart.yaml`: FOUND (modified, +1/-1)
- `weka-app-store-operator-chart/scripts/verify-crd.sh`: FOUND (new file, 202 lines, 0755)
- Commit `ea44d6d`: FOUND (`feat(17-01): add spec.appStack.variables to CRD schema`)
- Commit `81d86ed`: FOUND (`chore(17-01): bump chart version 0.1.61 -> 0.1.62`)
- Commit `d2c7a61`: FOUND (`feat(17-01): add verify-crd.sh phase-17 admission verifier`)
- Boundary files all clean: `operator_module/main.py`, `operator_module/tests/`, `mcp-server/tools/validate_yaml.py`, `docs/`, `CHANGELOG.md` — all empty diff vs `main`.
