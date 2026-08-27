#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 post-slug" >&2
  exit 1
fi

post_slug=$1
case "$post_slug" in
  *[!a-z0-9-]* | -* | *-)
    echo "Use lowercase letters, numbers, and internal hyphens only." >&2
    exit 1
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
hugo_image="ghcr.io/gohugoio/hugo:v0.165.0@sha256:608a19e34f86de36773503adbaab174fc28a6e338dc7904e03c70320b003a153"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required. Install or start Docker Desktop, then retry." >&2
  exit 1
}

git -C "$project_dir" submodule update --init --recursive

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HUGO_CACHEDIR=/tmp/hugo_cache \
  --volume "$project_dir:/project" \
  --workdir /project \
  "$hugo_image" \
  new content "posts/$post_slug.md"

echo "Created content/posts/$post_slug.md"
