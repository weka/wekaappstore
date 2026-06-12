#!/usr/bin/env bash
# Build and optionally push the weka-app-store-gui Docker image to Docker Hub.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-wekachrisjen/weka-app-store-gui}"
DOCKERFILE="${REPO_ROOT}/app-store-gui/Dockerfile"
GIT_SHA="${GIT_SHA:-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"

usage() {
  cat <<EOF
Usage: $(basename "$0") (--version X.Y.Z | --branch NAME) [--push] [--dry-run] [-h]

Builds the weka-app-store-gui Docker image.

Options:
  --version X.Y.Z   Release build: tags as vX.Y.Z and latest
  --branch NAME     Branch build: tags as branch-<slug>-<sha> and branch-<slug>
                    (also emits latest if NAME matches the default branch)
  --push            Push all tags to Docker Hub after building
  --dry-run         Show what would run without executing docker commands
  -h, --help        Show this help

Environment overrides:
  IMAGE             Full image name  (default: wekachrisjen/weka-app-store-gui)
  GIT_SHA           Git SHA for branch tags (default: git rev-parse --short HEAD)
  DEFAULT_BRANCH    Default branch name for latest tag logic (default: main)
EOF
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/(^-|-$)//g'
}

version_ok() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)+([-+].+)?$ ]]; }

VERSION=""
BRANCH=""
PUSH=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:-}"; shift 2 ;;
    --branch)  BRANCH="${2:-}";  shift 2 ;;
    --push)    PUSH=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "$VERSION" && -n "$BRANCH" ]]; then
  echo "error: choose either --version or --branch, not both" >&2; exit 2
fi
if [[ -z "$VERSION" && -z "$BRANCH" ]]; then
  echo "error: --version or --branch is required" >&2; usage; exit 2
fi

# Derive tags
declare -a TAGS=()
IMMUTABLE_TAG=""

if [[ -n "$VERSION" ]]; then
  version_ok "$VERSION" || { echo "invalid --version: $VERSION (want e.g. 0.42 or 1.2.3)" >&2; exit 2; }
  IMMUTABLE_TAG="v${VERSION#v}"
  TAGS=("$IMMUTABLE_TAG" "latest")
else
  SLUG="$(slugify "$BRANCH")"
  [[ -n "$SLUG" ]] || { echo "error: branch '$BRANCH' produced empty slug" >&2; exit 2; }
  IMMUTABLE_TAG="branch-${SLUG}-${GIT_SHA}"
  TAGS=("$IMMUTABLE_TAG" "branch-${SLUG}")
  [[ "$(slugify "$BRANCH")" == "$(slugify "$DEFAULT_BRANCH")" ]] && TAGS+=("latest")
fi

IMMUTABLE_REF="${IMAGE}:${IMMUTABLE_TAG}"

echo "==> Image:  ${IMAGE}"
echo "    Tags:   ${TAGS[*]}"
echo "    SHA:    ${GIT_SHA}"
echo

if $DRY_RUN; then
  echo "dry-run: docker build -f ${DOCKERFILE} -t ${IMMUTABLE_REF} ${REPO_ROOT}"
  for tag in "${TAGS[@]}"; do
    [[ "$tag" != "$IMMUTABLE_TAG" ]] && echo "dry-run: docker tag ${IMMUTABLE_REF} ${IMAGE}:${tag}"
  done
  $PUSH && echo "dry-run: docker push (all tags)" || echo "dry-run: push skipped (pass --push)"
  exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 1; }

echo "==> docker build"
docker build -f "${DOCKERFILE}" -t "${IMMUTABLE_REF}" "${REPO_ROOT}"

for tag in "${TAGS[@]}"; do
  [[ "$tag" != "$IMMUTABLE_TAG" ]] && docker tag "${IMMUTABLE_REF}" "${IMAGE}:${tag}"
done

if $PUSH; then
  for tag in "${TAGS[@]}"; do
    echo "==> docker push ${IMAGE}:${tag}"
    docker push "${IMAGE}:${tag}"
  done
fi

echo
echo "✓ Built: ${IMMUTABLE_REF}"
$PUSH && echo "  Pushed: ${TAGS[*]/#/${IMAGE}:}" || echo "  (not pushed — pass --push to publish)"
