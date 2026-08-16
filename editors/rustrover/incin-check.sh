#!/usr/bin/env bash
# Runs `cargo incin check` (humanized typenum diagnostics) against the
# current package and prints its output. Meant to be wired up as a RustRover
# "External Tool" (see README.md in this directory), RustRover's own Rust
# plugin doesn't support incin-lsp the way an LSP-based editor does, so this
# script is the fallback integration path.
set -euo pipefail

if ! command -v cargo-incin >/dev/null 2>&1; then
  echo "incin-check: 'cargo-incin' not found on PATH." >&2
  echo "Install it with: cargo install --path crates/incin (from the Incin repo)" >&2
  exit 1
fi

exec cargo incin check "$@"
