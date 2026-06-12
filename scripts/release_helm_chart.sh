#!/usr/bin/env bash
# Package and publish the weka-app-store-operator Helm chart.
# Publishes to docs/ (served as GitHub Pages at https://weka.github.io/wekaappstore).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHART_DIR="${REPO_ROOT}/weka-app-store-operator-chart"
DOCS_DIR="${REPO_ROOT}/docs"
GUI_TEMPLATE="${CHART_DIR}/templates/deploy-app-store-gui.yaml"
HELM_REPO_URL="${HELM_REPO_URL:-https://weka.github.io/wekaappstore}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--version X.Y.Z] [--gui-tag TAG] [--dry-run] [-h]

Packages weka-app-store-operator-chart and publishes it to docs/ (GitHub Pages).

Options:
  --version X.Y.Z   Bump Chart.yaml version before packaging
  --gui-tag TAG      Update the GUI container image tag in the Helm template (e.g. v0.42)
  --dry-run          Lint and package only; do not commit or push
  -h, --help         Show this help

Environment overrides:
  HELM_REPO_URL      Public chart repo URL (default: https://weka.github.io/wekaappstore)

Examples:
  # Bump chart version and update GUI image tag, then publish
  $(basename "$0") --version 0.1.65 --gui-tag v0.42

  # Only bump chart version (GUI tag unchanged)
  $(basename "$0") --version 0.1.65

  # Lint and package without committing
  $(basename "$0") --version 0.1.65 --dry-run
EOF
}

VERSION=""
GUI_TAG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:-}"; shift 2 ;;
    --gui-tag) GUI_TAG="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }
need helm
need git

semver_ok() { [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+].+)?$ ]]; }

# Validate args before touching any files
if [[ -n "$VERSION" ]]; then
  semver_ok "$VERSION" || { echo "invalid --version: $VERSION" >&2; exit 2; }
fi

if $DRY_RUN; then
  [[ -n "$VERSION" ]] && echo "==> dry-run: would bump Chart.yaml version to ${VERSION}"
  [[ -n "$GUI_TAG" ]] && echo "==> dry-run: would update GUI image tag to ${GUI_TAG}"
else
  # --- optional version bump ---
  if [[ -n "$VERSION" ]]; then
    echo "==> Bumping Chart.yaml version to ${VERSION}"
    sed -i.bak "s/^version: .*/version: ${VERSION}/" "${CHART_DIR}/Chart.yaml"
    rm -f "${CHART_DIR}/Chart.yaml.bak"
  fi

  # --- optional GUI image tag update ---
  if [[ -n "$GUI_TAG" ]]; then
    echo "==> Updating GUI image tag to ${GUI_TAG}"
    sed -i.bak "s|wekachrisjen/weka-app-store-gui:[^ ]*|wekachrisjen/weka-app-store-gui:${GUI_TAG}|" \
      "${GUI_TEMPLATE}"
    rm -f "${GUI_TEMPLATE}.bak"
  fi
fi

CHART_NAME="$(grep '^name:' "${CHART_DIR}/Chart.yaml" | awk '{print $2}')"
CHART_VERSION="$(grep '^version:' "${CHART_DIR}/Chart.yaml" | awk '{print $2}')"
TGZ_NAME="${CHART_NAME}-${CHART_VERSION}.tgz"

echo
echo "==> ${CHART_NAME} ${CHART_VERSION}"

echo "    helm lint"
helm lint "${CHART_DIR}"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "    helm package"
helm package "${CHART_DIR}" -d "${STAGING}"

[[ -f "${STAGING}/${TGZ_NAME}" ]] || { echo "expected package not found: ${TGZ_NAME}" >&2; exit 1; }

if $DRY_RUN; then
  cp "${STAGING}/${TGZ_NAME}" "${REPO_ROOT}/${TGZ_NAME}"
  echo
  echo "==> dry-run: package built at ${REPO_ROOT}/${TGZ_NAME}"
  echo "    (not committed — remove --dry-run to publish)"
  trap - EXIT
  exit 0
fi

# --- guard against duplicate ---
if [[ -f "${DOCS_DIR}/${TGZ_NAME}" ]]; then
  echo "error: ${TGZ_NAME} already exists in docs/ — bump the chart version first" >&2
  exit 1
fi

# --- copy into docs/ and rebuild index ---
cp "${STAGING}/${TGZ_NAME}" "${DOCS_DIR}/"

echo "==> helm repo index --merge"
if [[ -f "${DOCS_DIR}/index.yaml" ]]; then
  helm repo index "${DOCS_DIR}" --url "${HELM_REPO_URL}" --merge "${DOCS_DIR}/index.yaml"
else
  helm repo index "${DOCS_DIR}" --url "${HELM_REPO_URL}"
fi

# --- commit + push ---
declare -a COMMIT_FILES=(
  "${DOCS_DIR}/index.yaml"
  "${DOCS_DIR}/${TGZ_NAME}"
)
[[ -n "$VERSION" ]] && COMMIT_FILES+=("${CHART_DIR}/Chart.yaml")
[[ -n "$GUI_TAG" ]] && COMMIT_FILES+=("${GUI_TEMPLATE}")

COMMIT_MSG="chore: publish ${CHART_NAME}@${CHART_VERSION}"
[[ -n "$GUI_TAG" ]] && COMMIT_MSG+=" (gui=${GUI_TAG})"

CURRENT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"

echo "==> git commit: ${COMMIT_MSG}"
git -C "${REPO_ROOT}" add "${COMMIT_FILES[@]}"
git -C "${REPO_ROOT}" commit -m "${COMMIT_MSG}"

echo "==> git push origin ${CURRENT_BRANCH}"
git -C "${REPO_ROOT}" push origin "${CURRENT_BRANCH}"

echo
echo "✓ Published: ${CHART_NAME}@${CHART_VERSION}"
echo "  helm repo add weka-app-store ${HELM_REPO_URL}"
echo "  helm repo update"
echo "  helm install weka-app-store weka-app-store/${CHART_NAME} --version ${CHART_VERSION}"
