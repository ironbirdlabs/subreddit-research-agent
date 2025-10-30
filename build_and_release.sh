#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_and_release.sh – self-contained agent build helper
# ---------------------------------------------------------------------------
# This script builds (and optionally pushes) an agent Docker image.
# It is completely standalone – agent developers do NOT need the
# agentsystems-build-tools repo.
# ---------------------------------------------------------------------------


# ---- usage ---------------------------------------------------------------
function usage() {
  cat <<EOF
Usage: $(basename "$0") --image <name> [--version <ver>] [--push] [--git-tag] [--dockerfile <path>] [--context <dir>] [--platform <list>]
Options:
  --image        Required. Full image name, e.g. johndoe/echo-agent
  --version      Tag to apply. Default:
                   • env VERSION
                   • git describe --tags --always
  --push         Push image to registry after build (Buildx --push)
  --git-tag      Create and push a Git tag (vX.Y.Z) after successful build
  --dockerfile   Path to Dockerfile (default: Dockerfile in context dir)
  --context      Build context (default: repo root / current dir)
  --platform     Override platforms. Default if --push: linux/amd64,linux/arm64; else host arch.
  --help         Show this help.
EOF
}

# ---- arg parsing ---------------------------------------------------------
IMAGE=""
VERSION="${VERSION:-}"
PUSH="false"
CREATE_GIT_TAG="false"
DOCKERFILE=""
CONTEXT="$(pwd)"
PLATFORM=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --image) IMAGE="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --push) PUSH="true"; shift;;
    --git-tag) CREATE_GIT_TAG="true"; shift;;
    --dockerfile) DOCKERFILE="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    --platform) PLATFORM="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -z "$IMAGE" ]] && { echo "--image is required"; usage; exit 1; }

# -------- version & semver helpers ----------------------------------------
semver_regex='^v?[0-9]+\.[0-9]+\.[0-9]+([A-Za-z0-9.-]*)?$'

if [[ -z "$VERSION" ]]; then
  if [[ -n "${GITHUB_REF_NAME:-}" && "${GITHUB_REF:-}" == refs/tags/* ]]; then
    VERSION="$GITHUB_REF_NAME"
  else
    if git rev-parse --is-inside-work-tree &>/dev/null; then
      VERSION="$(git describe --tags --always)"
    else
      echo "Not a git repo and --version not supplied."; exit 1
    fi
  fi
fi

if [[ ! "$VERSION" =~ $semver_regex ]]; then
  echo "Invalid version format: $VERSION"; exit 1
fi

if [[ "$VERSION" == v* ]]; then
  GIT_TAG="$VERSION"
  DOCKER_VERSION="${VERSION#v}"
else
  GIT_TAG="v$VERSION"
  DOCKER_VERSION="$VERSION"
fi

CORE_VER="${DOCKER_VERSION%%-*}"

if git rev-parse "$GIT_TAG" &>/dev/null; then
  echo "Git tag $GIT_TAG already exists – aborting."; exit 1
fi

LATEST_CORE="$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' | sed 's/^v//' | sort -V | tail -1 || true)"
if [[ -n "$LATEST_CORE" && "$DOCKER_VERSION" != *-* ]]; then
  if [[ "$(printf '%s\n%s' "$LATEST_CORE" "$CORE_VER" | sort -V | tail -1)" != "$CORE_VER" ]]; then
    echo "Version $CORE_VER must be greater than latest released $LATEST_CORE"; exit 1
  fi
fi

TAG_ARGS=("-t" "$IMAGE:$DOCKER_VERSION")
if [[ "$DOCKER_VERSION" != *-* ]]; then
  TAG_ARGS+=("-t" "$IMAGE:latest")
fi

if [[ -z "$PLATFORM" ]]; then
  if [[ "$PUSH" == "true" ]]; then
    PLATFORM="linux/amd64,linux/arm64"
  else
    PLATFORM="$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
  fi
fi

if [[ -z "$DOCKERFILE" ]]; then
  DOCKERFILE="$CONTEXT/Dockerfile"
fi

export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}

BUILD_CMD=(docker buildx build "${TAG_ARGS[@]}" --platform "$PLATFORM" -f "$DOCKERFILE" "$CONTEXT")
if [[ "$PUSH" == "true" ]]; then
  BUILD_CMD+=(--push)
else
  BUILD_CMD+=(--load)
fi

echo "# ------------------------------------------------------------"
echo "# Building image: $IMAGE"
echo "# Git tag        : $GIT_TAG (create: $CREATE_GIT_TAG)"
echo "# Docker version : $DOCKER_VERSION"
echo "# Dockerfile     : $DOCKERFILE"
echo "# Context dir    : $CONTEXT"
echo "# Platforms      : $PLATFORM"
echo "# Push after build: $PUSH"
echo "# ------------------------------------------------------------"

# ---- confirmation ---------------------------------------------------------
read -r -p "Proceed with build and release? Type 'y' or 'yes' to continue: " CONFIRM
case "${CONFIRM}" in
  y|Y|yes|YES)
    ;;
  *)
    echo "Aborted.";
    exit 1;
    ;;
esac

"${BUILD_CMD[@]}"

echo "Image $IMAGE:$DOCKER_VERSION built successfully."
if [[ "$PUSH" == "true" ]]; then
  echo "Pushed to registry as $IMAGE:$DOCKER_VERSION";
fi

if [[ "$CREATE_GIT_TAG" == "true" ]]; then
  echo "Creating git tag $GIT_TAG and pushing to origin…"
  git tag -a "$GIT_TAG" -m "Release $GIT_TAG"
  git push origin "$GIT_TAG"
fi
