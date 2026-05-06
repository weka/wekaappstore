#!/usr/bin/env bash
# weka-app-store-operator-chart/scripts/verify-crd.sh
# Verifies Phase 17 additive CRD update: spec.appStack.variables admission
# rules and kubectl explain documentation.
#
# Usage: bash weka-app-store-operator-chart/scripts/verify-crd.sh [--apply]
#   (no flags) — dry-run only: helm template | kubectl --dry-run=server (4 fixtures)
#   --apply    — also installs the CRD on the live cluster, then runs
#                kubectl explain wekaappstores.spec.appStack.variables
#                and asserts all four CRD-02 keywords are present
#
# Exit codes:
#   0 — all checks PASS
#   1 — one or more FAIL (all checks run before exit; D-13)
#   2 — precondition failure (cluster CRD does not yet expose spec.appStack.variables;
#       re-run with --apply to install the new CRD first)
#
# Requires: kubectl + helm configured with cluster access for --dry-run=server
set -euo pipefail

# ─── Argument parsing ─────────────────────────────────────────────────────
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

# ─── Path resolution ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PASS=0
FAIL=0

# ─── Step 0: install/update CRD only in --apply mode ──────────────────────
if [[ "${APPLY_MODE}" == true ]]; then
  echo "[--apply mode] Installing CRD on live cluster..."
  helm template "${CHART_DIR}" --show-only templates/crd.yaml \
    | kubectl apply -f -
fi

# ─── Precheck: confirm new schema is installed before running fixtures ───
# Fixtures validate against whatever CRD is currently in the cluster. If the
# new variables/propertyNames schema is NOT in the cluster, the fixtures
# produce misleading results (Case 1 fails as "unknown field" instead of
# passing; Cases 2/3 fail with the wrong error). Detect this and refuse to
# run.
INSTALLED_HAS_VARIABLES=$(kubectl get crd wekaappstores.warp.io -o yaml 2>/dev/null \
  | grep -c '^                  variables:$' || true)
if [[ "${INSTALLED_HAS_VARIABLES}" -eq 0 ]]; then
  echo "ERROR: cluster CRD does not yet contain spec.appStack.variables." >&2
  echo "       Re-run with --apply (will install the new CRD), then re-run" >&2
  echo "       this script in default mode." >&2
  exit 2
fi

run_dry_run_case() {
  local label="$1"
  local expectation="$2"   # PASS or FAIL
  local match_pattern="$3" # required stderr substring (regex) when expectation=FAIL; ignored when PASS
  local fixture_yaml="$4"

  echo ""
  echo "─── ${label} (expect ${expectation}) ────────────────────────────"
  local out
  out=$(printf '%s' "${fixture_yaml}" | kubectl apply --dry-run=server -f - 2>&1 || true)

  if [[ "${expectation}" == "PASS" ]]; then
    if echo "${out}" | grep -qE "created|configured|unchanged"; then
      echo "  PASS: ${label} accepted by admission"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: ${label} expected PASS but admission did not accept it"
      echo "        ${out}"
      FAIL=$((FAIL + 1))
    fi
  else  # FAIL expected
    if echo "${out}" | grep -qE "${match_pattern}"; then
      echo "  PASS: ${label} rejected by admission with expected substring (${match_pattern})"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: ${label} expected FAIL with substring (${match_pattern}) but got:"
      echo "        ${out}"
      FAIL=$((FAIL + 1))
    fi
  fi
}

# ─── Case 1: valid variables map (D-10 case 1) ────────────────────────────
FIXTURE_VALID=$(cat <<'YAML'
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: verify-crd-valid
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
run_dry_run_case "Case 1: valid string variables" "PASS" "" "${FIXTURE_VALID}"

# ─── Case 2: integer value rejected (D-10 case 2; CRD-03) ─────────────────
FIXTURE_INT=$(cat <<'YAML'
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: verify-crd-int
  namespace: default
spec:
  appStack:
    variables:
      count: 42
    components:
      - name: dummy
        kubernetesManifest: ""
YAML
)
run_dry_run_case "Case 2: integer value rejected" "FAIL" "spec\.appStack\.variables\.count.*(must be of type string|expected string)|Invalid value.*: \"integer\"" "${FIXTURE_INT}"

# ─── Case 3: hyphenated key rejected (D-10 case 3; D-01, D-02) ────────────
FIXTURE_HYPHEN=$(cat <<'YAML'
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: verify-crd-hyphen
  namespace: default
spec:
  appStack:
    variables:
      my-host: foo
    components:
      - name: dummy
        kubernetesManifest: ""
YAML
)
run_dry_run_case "Case 3: hyphenated key rejected" "FAIL" "propertyNames|pattern" "${FIXTURE_HYPHEN}"

# ─── Case 4: backward-compat — no variables block (D-10 case 4; CRD-01 SC#5) ─
FIXTURE_NO_VARS=$(cat <<'YAML'
apiVersion: warp.io/v1alpha1
kind: WekaAppStore
metadata:
  name: verify-crd-no-vars
  namespace: default
spec:
  appStack:
    components:
      - name: dummy
        kubernetesManifest: ""
YAML
)
run_dry_run_case "Case 4: backward-compat (no variables block)" "PASS" "" "${FIXTURE_NO_VARS}"

# ─── kubectl explain keyword check (D-12) — only in --apply mode ──────────
if [[ "${APPLY_MODE}" == true ]]; then
  echo ""
  echo "─── kubectl explain keyword check ────────────────────────────────"
  EXPLAIN_OUTPUT="$(kubectl explain wekaappstores.spec.appStack.variables 2>&1 || true)"
  check_explain_keyword() {
    local keyword="$1"
    if echo "${EXPLAIN_OUTPUT}" | grep -qF "${keyword}"; then
      echo "  PASS: kubectl explain mentions '${keyword}'"
      PASS=$((PASS + 1))
    else
      echo "  FAIL: kubectl explain missing keyword '${keyword}'"
      echo "        explain output was:"
      echo "${EXPLAIN_OUTPUT}" | sed 's/^/          /'
      FAIL=$((FAIL + 1))
    fi
  }
  check_explain_keyword '${VAR}'
  check_explain_keyword '$$'
  check_explain_keyword '${namespace}'
  check_explain_keyword 'identifier'
fi

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 17 CRD Verification Summary ==="
echo "Checks passed: ${PASS}"
echo "Checks failed: ${FAIL}"
if [[ "${FAIL}" -eq 0 ]]; then
  echo ""
  echo "=== Phase 17 CRD Verification PASSED (${PASS} checks) ==="
  exit 0
else
  echo ""
  echo "=== Phase 17 CRD Verification FAILED (${FAIL} failures) ==="
  exit 1
fi
