#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

command -v lzc-cli >/dev/null 2>&1 || {
  echo "lzc-cli is required. Install it, select a box, then retry." >&2
  exit 1
}

cd "$project_dir"
lzc-cli project lint
lzc-cli project deploy --release
lzc-cli project info --release
