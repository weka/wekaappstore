#!/usr/bin/env bash
# Build and optionally push the weka-app-store-operator Docker image to Docker Hub.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-wekachrisjen/weka-app-store-multi-arch}"
DOCKERFILE="${REPO_ROOT}/docker/operator.Dockerfile"
GIT_SHA="${GIT_SHA:-$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"

usage() {
  cat <<EOF
Usage: $(basename "$0") (--version X.Y.Z | --branch NAME) [--push] [--dry-run] [-h]

Builds the weka-app-store-operator Docker image.

Options:
  --version X.Y.Z   Release build: tags as vX.Y.Z and latest
  --branch NAME     Branch build: tags as branch-<slug>-<sha> and branch-<slug>
                    (also emits latest if NAME matches the default branch)
  --push            Push all tags to Docker Hub after building
  --dry-run         Show what would run without executing docker commands
  -h, --help        Show this help

Environment overrides:
  IMAGE             Full image name  (default: wekachrisjen/weka-app-store-multi-arch)
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
  version_ok "$VERSION" || { echo "invalid --version: $VERSION (want e.g. 0.10 or 1.2.3)" >&2; exit 2; }
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

# Build -t flags for all tags
declare -a TAG_ARGS=()
for tag in "${TAGS[@]}"; do
  TAG_ARGS+=("-t" "${IMAGE}:${tag}")
done

if $DRY_RUN; then
  if $PUSH; then
    echo "dry-run: docker buildx build -f ${DOCKERFILE} --platform linux/amd64 ${TAG_ARGS[*]} --push ${REPO_ROOT}"
  else
    echo "dry-run: docker buildx build -f ${DOCKERFILE} --platform linux/amd64 -t ${IMMUTABLE_REF} --load ${REPO_ROOT}"
  fi
  exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 1; }

if $PUSH; then
  echo "==> docker buildx build --platform linux/amd64 --push"
  docker buildx build \
    -f "${DOCKERFILE}" \
    --platform linux/amd64 \
    "${TAG_ARGS[@]}" \
    --push \
    "${REPO_ROOT}"
else
  echo "==> docker buildx build --platform linux/amd64 --load"
  docker buildx build \
    -f "${DOCKERFILE}" \
    --platform linux/amd64 \
    -t "${IMMUTABLE_REF}" \
    --load \
    "${REPO_ROOT}"
fi

echo
echo "✓ Built: ${IMMUTABLE_REF}"
$PUSH && echo "  Pushed: ${TAGS[*]}" || echo "  Loaded locally (not pushed — pass --push to publish)"
