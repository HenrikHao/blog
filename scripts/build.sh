#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
hugo_image="ghcr.io/gohugoio/hugo:v0.165.0@sha256:608a19e34f86de36773503adbaab174fc28a6e338dc7904e03c70320b003a153"
blog_base_url=${BLOG_BASE_URL:-https://blog.mhserver.heiyu.space/}

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required. Install or start Docker Desktop, then retry." >&2
  exit 1
}

git -C "$project_dir" submodule update --init --recursive
mkdir -p "$project_dir/dist"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HUGO_CACHEDIR=/tmp/hugo_cache \
  --volume "$project_dir:/project" \
  --workdir /project \
  "$hugo_image" \
  --gc \
  --minify \
  --cleanDestinationDir \
  --baseURL "$blog_base_url"
