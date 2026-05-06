# Phase 17: CRD Schema Additive Update - Pattern Map

**Mapped:** 2026-05-06
**Files analyzed:** 3 (2 modified, 1 new)
**Analogs found:** 3 / 3 (1 partial — `propertyNames` has no in-repo analog; canonical source documented)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `weka-app-store-operator-chart/templates/crd.yaml` (ADD `variables:` block) | config (CRD schema, Helm template) | n/a (declarative schema) | Same file: `appStack.properties.components.items` siblings (lines 89-192); `matchLabels` (lines 181-185) | exact (same file, sibling pattern, same indent) |
| `weka-app-store-operator-chart/Chart.yaml` (version bump 0.1.61 → 0.1.62) | config (Helm chart metadata) | n/a | git history of prior `version:` bumps (commit `f9635fa` lineage; current `version: 0.1.61` line 18) | exact (single-line edit, no quoting) |
| `weka-app-store-operator-chart/scripts/verify-crd.sh` (NEW) | utility (verification shell script) | request-response (`kubectl --dry-run=server`); event-driven (per-fixture pass/fail accounting) | `scripts/validate-phase13.sh` (full file, esp. lines 16-56, 65-113, 119-128, 275-297) | role-match (different domain — CRD admission vs sandbox-pod readiness — but same scaffold, helper, summary, exit-code conventions) |

**Directory creation:** `weka-app-store-operator-chart/scripts/` does NOT exist (`ls -la weka-app-store-operator-chart/` shows only `charts/`, `templates/`, `Chart.yaml`, `values.yaml`, `.helmignore`). Planner must create the directory as part of writing `verify-crd.sh`. No `__init__.py`-style marker file is needed for shell scripts.

---

## Pattern Assignments

### `weka-app-store-operator-chart/templates/crd.yaml` — ADD `variables:` block under `appStack.properties`

**Analog:** Same file. Three concrete sibling patterns to mirror.

**Insertion point:** Line 192 (end of `components:` block — closing of `readinessCheck.timeout.default: 300`). New block becomes the second sibling under `appStack.properties` after `components`. Indentation depth: `variables:` at 18-space indent (same column as `components:` on line 89).

#### Pattern 1 — `appStack.properties` sibling indent and shape (crd.yaml lines 84-91)

Defines the column where `variables:` keyword lands and how its `description:` should sit relative to `type:`.

```yaml
              # AppStack configuration for multi-component deployments with dependencies
              appStack:
                type: object
                description: "Multi-component application stack with dependency management"
                properties:
                  components:
                    type: array
                    description: "List of components to deploy in order with dependencies"
```

**What to copy:** `variables:` lives at the same indent as `components:` on line 89 (24-space indent for the key, 26-space for child fields like `type:`, `description:`, `propertyNames:`, `additionalProperties:`).

#### Pattern 2 — `additionalProperties: { type: string }` already used in this CRD (crd.yaml lines 181-185)

Confirms the value-type-enforcement pattern is already established in-repo and CRD-03 introduces no new convention.

```yaml
                            matchLabels:
                              type: object
                              description: "Kubernetes-style matchLabels map (alternative to selector)"
                              additionalProperties:
                                type: string
```

**What to copy:** `additionalProperties` block-form (key on its own line, `type: string` indented two spaces underneath). The new `variables:` block uses the same nested shape — NOT the inline flow-mapping form `{ type: string }`. Match the existing block style.

#### Pattern 3 — Multi-line `description:` block-scalar — NO existing in-repo analog

**Search result:** Every `description:` in `crd.yaml` is a single-line double-quoted scalar (e.g., line 24: `description: "Container image for pod-based deployment (legacy)"`; line 87: `description: "Multi-component application stack with dependency management"`; line 130: `"How to handle CRDs contained in the chart: Auto (default) skips if CRDs..."`). Block-scalar `description: |` is NOT used anywhere in this file today.

CONTEXT.md `<code_context>` line 111 references "line 79's `targetNamespace` description style" as a long-description example, but the file itself shows that line 81 is `description: "Target namespace for Helm installation (optional, defaults to CR namespace)"` — also single-line double-quoted. There is no existing block-scalar to mirror.

**Recommendation for planner:** The block-scalar text is locked verbatim by CONTEXT.md D-05 (the 7-line description). Use the standard YAML 1.2 indent rule: `description: |` is followed by a body indented two spaces deeper than the `description:` key. Verifier in `verify-crd.sh` (Check D-12) greps the rendered output for the four required keywords (`${VAR}`, `$$`, `${namespace}`, `identifier`) — this is the binding contract, not stylistic conformance to a non-existent in-repo block-scalar pattern.

**What to copy:** None — write D-05 verbatim with two-space body indent under `description: |`. This is the one stylistic departure from the existing file convention; it is justified by the locked text length (>50 chars across multiple sentences) and the explicit `kubectl explain` keyword-grep contract.

#### Pattern 4 — `propertyNames: { pattern: ... }` — NO in-repo analog

**Search result:** `propertyNames` does not appear anywhere in `weka-app-store-operator-chart/templates/crd.yaml`. CONTEXT.md `<code_context>` line 113 explicitly notes Phase 17 introduces it.

**Recommendation for planner:** No codebase pattern to copy. Use the canonical Kubernetes OpenAPI v3 schema spec — `propertyNames` is a standard JSON Schema Draft 4+ keyword wired into Kubernetes 1.16+ validation (CRD `apiextensions.k8s.io/v1` already in use on line 2 of `crd.yaml`). EKS clusters used for v3.0 / v5.0 are well past 1.16, so no version gate. Use the block form (matching Pattern 2 style):

```yaml
                    propertyNames:
                      pattern: "^[_a-zA-Z][_a-zA-Z0-9]*$"
```

**Verification:** `verify-crd.sh` Check #3 (hyphenated-key fixture `my-host`) is the contract — if the cluster rejects `my-host` at admission with a `propertyNames` or `pattern` error, the schema is correctly authored. CONTEXT.md `<decisions>` D-10 case 3 locks this expectation.

#### Final block to insert (locked verbatim from CONTEXT.md `<specifics>`)

Place at line 192+ (immediately after `default: 300` closes the `readinessCheck.timeout` field — note the indentation must align with `components:` on line 89, NOT with `readinessCheck` on line 167):

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

(Note: the indentation prefix above represents the appStack.properties sibling depth in the actual file. Planner verifies the exact column count by reading line 89 first, then matching it.)

---

### `weka-app-store-operator-chart/Chart.yaml` — version bump

**Analog:** The current file itself. Single-line edit — no whitespace, comment, or formatting change.

**Pattern to preserve** (Chart.yaml line 18, current state):

```yaml
version: 0.1.61
```

**Target state:**

```yaml
version: 0.1.62
```

**What to copy:**
- No quotes (the existing line uses bare semver, NOT a quoted string — contrast `appVersion: "1.16.0"` on line 24 which IS quoted).
- No trailing whitespace.
- No CHANGELOG.md, no `description:` field edit, no `appVersion:` bump (CONTEXT.md D-16 explicitly defers all of these).
- Do NOT touch the comment block on lines 15-17 above the version line.

**Operational note:** README §"Publishing (maintainers)" lines 151-158 describes `helm package` + `helm repo index` to publish the bumped chart. Phase 17 explicitly does NOT run these commands per CONTEXT.md D-15. The version bump exists in source but is not packaged-and-shipped to `docs/` until end-of-v5.0 milestone. Planner must NOT include `helm package` or `helm repo index` invocations in any plan action.

---

### `weka-app-store-operator-chart/scripts/verify-crd.sh` (NEW)

**Analog:** `scripts/validate-phase13.sh` (full file, 297 lines). The Phase 17 verifier and the Phase 13 validator share the same problem shape: render manifests, dry-run apply against a live cluster, run per-fixture checks with a final summary and non-zero exit on failure. The Phase 17 script is smaller in scope (4 fixtures vs 10 checks, no live-pod inspection) but adopts the same scaffold.

#### Pattern 1 — Shebang and shell-options preamble (validate-phase13.sh lines 1-16)

```bash
#!/usr/bin/env bash
# scripts/validate-phase13.sh
# Validates Phase 13 manifests: Sandbox CR sidecar wiring, MCP sidecar health,
# openclaw.json generation, SKILL.md mount, and RBAC permissions.
#
# Usage: bash scripts/validate-phase13.sh [NAMESPACE] [--live]
#   NAMESPACE: target namespace (default: $NAMESPACE env var, fallback: wekaappstore)
#   --live:    run live cluster checks (requires kubectl cluster access)
#              dry-run structural checks always run first
#
# Exit codes:
#   0 — all checks PASS (WARNs do not cause failure)
#   1 — one or more FAIL
#
# Requires: kubectl configured with appropriate cluster access (for --live mode)
set -euo pipefail
```

**What to copy:**
- Exact shebang `#!/usr/bin/env bash` (NOT `#!/bin/bash` — portability convention in this repo).
- Header comment naming the script with relative path, summarizing purpose, listing usage, exit codes, and prerequisites.
- `set -euo pipefail` immediately after the header (before any executable line).
- Convert `--live` to `--apply` per CONTEXT.md D-11 (default = dry-run; `--apply` opts in to actually installing the CRD on the live cluster, required for the `kubectl explain` keyword check in D-12).

#### Pattern 2 — Argument parsing for boolean flag (validate-phase13.sh lines 18-36)

```bash
# ─── Argument parsing ─────────────────────────────────────────────────────────
NAMESPACE="wekaappstore"
LIVE_MODE=false

for arg in "$@"; do
  case "${arg}" in
    --live)
      LIVE_MODE=true
      ;;
    --*)
      echo "Unknown flag: ${arg}"
      echo "Usage: $0 [NAMESPACE] [--live]"
      exit 1
      ;;
    *)
      NAMESPACE="${arg}"
      ;;
  esac
done
```

**What to copy:** The `for arg in "$@"; do case ...` loop pattern. For Phase 17 only `--apply` is needed (no positional NAMESPACE — the CRD is cluster-scoped, see crd.yaml line 258 `scope: Namespaced` applies to instances, not the CRD itself). Reduce to:

```bash
APPLY_MODE=false
for arg in "$@"; do
  case "${arg}" in
    --apply) APPLY_MODE=true ;;
    *)
      echo "Unknown flag: ${arg}"
      echo "Usage: $0 [--apply]"
      exit 1
      ;;
  esac
done
```

#### Pattern 3 — Repo-root resolution from script location (validate-phase13.sh lines 41-46)

```bash
# ─── Resolve manifest paths relative to repo root ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RBAC_MANIFEST="${REPO_ROOT}/k8s/agent-sandbox/mcp-rbac.yaml"
```

**What to copy:** The `SCRIPT_DIR` / `REPO_ROOT` resolution idiom. Phase 17's script lives one directory deeper (`weka-app-store-operator-chart/scripts/` not `scripts/`), so the planner must use `cd "${SCRIPT_DIR}/../.." && pwd` for repo root, OR resolve `CHART_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"` and use `helm template "${CHART_DIR}"` directly without needing repo root at all. The chart-dir-only form is cleaner for this script.

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
```

#### Pattern 4 — PASS/FAIL/WARNS counters and per-check accounting (validate-phase13.sh lines 48-50, 65-79)

```bash
PASS=0
FAIL=0
WARNS=()

# ─── Check 1: RBAC YAML syntax ────────────────────────────────────────────────
echo "[1] kubectl dry-run: mcp-rbac.yaml..."
if [[ -f "${RBAC_MANIFEST}" ]]; then
  RBAC_OUT=$(kubectl apply --dry-run=client -f "${RBAC_MANIFEST}" -n "${NAMESPACE}" 2>&1 || true)
  if echo "${RBAC_OUT}" | grep -qE "created|configured|unchanged"; then
    echo "  PASS: mcp-rbac.yaml is valid YAML and accepted by API server (dry-run)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: mcp-rbac.yaml dry-run failed"
    echo "        ${RBAC_OUT}"
    FAIL=$((FAIL + 1))
  fi
```

**What to copy:**
- `PASS=0`, `FAIL=0`, `WARNS=()` integer/array accumulators initialized before any check.
- `[N] description...` numbered check headers with two-space indented `PASS:` / `FAIL:` lines below.
- `2>&1 || true` capture-stderr-and-don't-let-pipe-fail idiom (essential because `kubectl apply --dry-run` exits non-zero on validation failure but we WANT to capture and assert that failure for cases 2 and 3).
- `echo "${OUTPUT}" | grep -qE "pattern1|pattern2"` for asserting expected substrings in stderr.
- `PASS=$((PASS + 1))` / `FAIL=$((FAIL + 1))` arithmetic accumulation.

**Phase 17 specialization — invert success on failure-expected fixtures:** validate-phase13.sh Checks 1-3 expect SUCCESS from `kubectl apply --dry-run`. Phase 17's case 2 (integer value) and case 3 (hyphenated key) expect FAILURE. The pattern inverts: `if grep -qE "Invalid value|string"` (positive substring assertion on stderr) → PASS; otherwise → FAIL. CONTEXT.md `<decisions>` D-10 cases 2 and 3 lock the expected stderr substrings (loose match per Claude's discretion D-09 since admission error wording varies between K8s minor versions).

#### Pattern 5 — Reusable grep-based assertion helper (validate-phase13.sh lines 119-128)

```bash
check_grep() {
  local pattern="$1"
  local desc="$2"
  if grep -q "${pattern}" "${SANDBOX_MANIFEST}" 2>/dev/null; then
    echo "  PASS: ${desc}"
  else
    echo "  FAIL: ${desc} — pattern not found: '${pattern}'"
    STRUCT_FAIL=$((STRUCT_FAIL + 1))
  fi
}
```

**What to copy:** The function-with-local-pattern-and-description pattern. Phase 17's `kubectl explain` keyword check (D-12, four keywords: `${VAR}`, `$$`, `${namespace}`, `identifier`) is a great fit. Adapt:

```bash
EXPLAIN_OUTPUT=""  # populated once via kubectl explain in --apply mode
check_explain_keyword() {
  local keyword="$1"
  local desc="$2"
  if echo "${EXPLAIN_OUTPUT}" | grep -qF "${keyword}"; then
    echo "  PASS: ${desc} (keyword '${keyword}' present in explain output)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: ${desc} (keyword '${keyword}' missing from explain output)"
    FAIL=$((FAIL + 1))
  fi
}
```

Use `grep -qF` (fixed-string, not regex) for `${VAR}` and `$$` so the dollar-sign metacharacters don't get interpreted.

#### Pattern 6 — Inline heredoc fixtures (no analog in validate-phase13.sh, but standard bash)

CONTEXT.md D-10 specifies four fixtures as heredocs (no separate fixture files). validate-phase13.sh does NOT use heredocs because its inputs are pre-existing manifest files. Phase 17 DOES need them. No in-repo analog — use standard bash heredoc syntax piped into `kubectl apply --dry-run=server -f -`:

```bash
FIXTURE_VALID=$(cat <<'YAML'
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: verify-valid
  namespace: default
spec:
  appStack:
    variables:
      namespace: foo
      milvusHost: milvus.foo.svc.cluster.local
    components:
      - name: dummy
        kubernetesManifest: ""
YAML
)
echo "${FIXTURE_VALID}" | kubectl apply --dry-run=server -f - 2>&1
```

**What to copy:** Single-quoted heredoc delimiter (`<<'YAML'`) to suppress shell variable expansion inside the fixture YAML — critical because the fixture itself contains `${VAR}` and `$$` literals that must survive to the apiserver. Do NOT use unquoted `<<YAML` (which would expand `${namespace}` shell-side).

**`kubectl --dry-run=server` rationale (CONTEXT.md D-09):** Server-side dry-run runs the full admission chain (including `propertyNames` and `additionalProperties` validation), unlike `--dry-run=client` which only validates locally-resolved schema. Cases 2 (integer) and 3 (hyphenated key) require server-side dry-run to exercise the new schema rules. The Phase 13 validator uses `--dry-run=client` for YAML-syntax checks; that is NOT sufficient for Phase 17.

**Reference fixtures for backward-compat (D-10 case 4):** `cluster_init/app-store-cluster-init.yaml` is the canonical "no variables block" production CR (CONTEXT.md `<canonical_refs>` "Code to Reference"). Verifier can either inline a minimal version as a heredoc OR `kubectl apply --dry-run=server -f cluster_init/app-store-cluster-init.yaml` directly using the resolved `${REPO_ROOT}` path. Inline-heredoc is simpler and self-contained per D-10 ("heredocs, no separate fixture files").

#### Pattern 7 — Final summary block and exit code (validate-phase13.sh lines 275-297)

```bash
# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 13 Validation Summary ==="
echo "Checks passed: ${PASS}"
echo "Checks failed: ${FAIL}"

if [[ ${#WARNS[@]} -gt 0 ]]; then
  echo ""
  echo "Warnings:"
  for warn in "${WARNS[@]}"; do
    echo "  - ${warn}"
  done
fi

if [[ "${FAIL}" -eq 0 ]]; then
  echo ""
  echo "=== Phase 13 Validation PASSED (${PASS} checks passed, ${#WARNS[@]} warnings) ==="
  exit 0
else
  echo ""
  echo "=== Phase 13 Validation FAILED (${FAIL} failures) ==="
  exit 1
fi
```

**What to copy verbatim (with title swap):** the final summary block. Adapt the title to "Phase 17 CRD Verification". Critical: `exit 0` only when `FAIL=0`. CONTEXT.md D-13 requires that all four cases run BEFORE exit (do not `exit 1` mid-script on first failure) — this matches the validate-phase13.sh pattern where each check increments counters and only the final block exits. Already conforming.

#### Pattern 8 — Section dividers for readability (validate-phase13.sh throughout)

```bash
# ─── Check 1: RBAC YAML syntax ────────────────────────────────────────────────
```

**What to copy:** The `# ─── ... ─────` Unicode-box-drawing divider style. Used throughout validate-phase13.sh for section headers. Not load-bearing, but matches established convention. Optional but recommended for parity.

---

## Shared Patterns

### Indentation: 2-space throughout

**Source:** `weka-app-store-operator-chart/templates/crd.yaml` (entire file uses 2-space indent — confirm by counting between `apiVersion:` line 2 and any nested key).
**Apply to:** The new `variables:` block in `crd.yaml`.

CONTEXT.md `<decisions>` "Claude's Discretion" item 2 explicitly locks this: "Whitespace/indent style matching the rest of `crd.yaml` (existing file uses 2-space indent — match)."

### Helm-template gating (do NOT extend)

**Source:** `weka-app-store-operator-chart/templates/crd.yaml` lines 1 and 259:

```yaml
{{- if .Values.customResourceDefinition.create }}
apiVersion: apiextensions.k8s.io/v1
...
{{- end }}
```

**Apply to:** Phase 17 changes the schema BODY only (lines 17-253 zone). Do NOT touch the `{{- if ... }}` / `{{- end }}` template guards. The new `variables:` block sits inside the existing gated zone — same gating applies automatically.

### Backward-compat invariant (CRD-01 SC#5)

**Source:** CONTEXT.md `<decisions>` D-08 — "`variables` is NOT in `required:` — existing CRs without it continue to pass admission."
**Apply to:** Both crd.yaml (no `required:` field added under `appStack` mentioning `variables`) AND verify-crd.sh case 4 (existing-style CR with no `variables:` block must dry-run-apply cleanly, asserting backward-compat).

Reference for case 4 fixture: `cluster_init/app-store-cluster-init.yaml` (current production CR, no `variables:` block) and `mcp-server/tests/fixtures/sample_blueprints/ai-research.yaml` / `data-pipeline.yaml` (additional reference CRs the verifier MAY dry-run as a bonus check, optional per D-10).

### Exit-code convention (any non-zero is fail)

**Source:** `scripts/validate-phase13.sh` lines 16, 289-297; `scripts/validate-phase14-prereqs.sh` lines 14, 39-46.
**Apply to:** `verify-crd.sh`. CONTEXT.md "Claude's Discretion" item 3 grants the planner discretion on specific code-per-case (e.g., 2 = type-mismatch, 3 = pattern-mismatch) but the established repo convention is `exit 0` on full pass and `exit 1` on any failure. Stick with this — agents and humans grepping CI logs expect the binary signal.

---

## No Analog Found

| File / Pattern | Reason | Recommendation |
|----------------|--------|----------------|
| `propertyNames` keyword in CRD schema | Phase 17 introduces it; no existing usage in `crd.yaml` or any other in-repo CRD | Use canonical Kubernetes OpenAPI v3 schema (JSON Schema Draft 4+) — `propertyNames` is supported on Kubernetes 1.16+ via `apiextensions.k8s.io/v1`. No version gate needed (EKS clusters are well past 1.16 per CONTEXT.md `<specifics>`). Block form: key on its own line, `pattern:` indented two spaces underneath. |
| Multi-line `description: \|` block-scalar in `crd.yaml` | All existing descriptions are single-line double-quoted | Use YAML 1.2 standard: body indented two spaces deeper than the `description:` key. Locked text from CONTEXT.md D-05 is binding; verifier's keyword-grep (D-12) is the real contract. |
| Inline `kubectl --dry-run=server` heredoc fixtures in shell | Existing scripts (`validate-phase13.sh`, etc.) consume pre-existing manifest files, not inline fixtures | Use single-quoted heredoc delimiter (`<<'YAML'`) to prevent shell variable expansion of `${namespace}` / `${milvusHost}` inside the fixture YAML. Pipe to `kubectl apply --dry-run=server -f -`. |
| `helm template` rendering of operator chart inside a verification script | No prior verifier runs `helm template` against `weka-app-store-operator-chart/`; operator-chart deployment historically uses `helm install` from `docs/` index | Standard helm CLI: `helm template "${CHART_DIR}"` (relies on `Chart.yaml` and `values.yaml` already in chart dir). Filter to just the CRD: pipe through `kubectl apply --dry-run=server -f -` to render+validate in one step, OR use `--show-only templates/crd.yaml` for a focused render. The latter is cleaner for assertion-grepping. |

---

## Metadata

**Analog search scope:**
- `weka-app-store-operator-chart/` (entire directory, recursive)
- `scripts/*.sh` (5 shell scripts: `validate-phase13.sh`, `validate-phase14-prereqs.sh`, `install-agent-sandbox.sh`, `validate-topology.sh`, `capture-e2e-evidence.sh`)
- `weka-csi-config/weka-operator/resources/*.sh` (excluded — third-party install scripts, no shared style)
- `cluster_init/app-store-cluster-init.yaml` (reference CR for backward-compat fixture)
- `mcp-server/tests/fixtures/sample_blueprints/*.yaml` (additional reference CRs)
- `.planning/phases/16-render-helper-and-test-scaffolding/16-PATTERNS.md` (style precedent for this PATTERNS.md document)

**Files scanned:** 12 (1 CRD, 1 Chart.yaml, 5 shell scripts, 3 sample CRs, 1 README, 1 prior PATTERNS.md, 1 cluster_init CR)

**Pattern extraction date:** 2026-05-06
