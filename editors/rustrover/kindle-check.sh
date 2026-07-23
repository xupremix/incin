#!/usr/bin/env bash
# Runs `cargo kindle check` (humanized typenum diagnostics) against the
# current package and prints its output. Meant to be wired up as a RustRover
# "External Tool" (see README.md in this directory) — RustRover's own Rust
# plugin doesn't support kindle-lsp the way an LSP-based editor does, so this
# script is the fallback integration path.
set -euo pipefail

if ! command -v cargo-kindle >/dev/null 2>&1; then
  echo "kindle-check: 'cargo-kindle' not found on PATH." >&2
  echo "Install it with: cargo install --path crates/kindle (from the Kindle repo)" >&2
  exit 1
fi

exec cargo kindle check "$@"
