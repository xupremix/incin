#!/usr/bin/env bash
set -euo pipefail

# Fixed-budget cargo-fuzz campaign for the three adversarial parsers (#48).
#
# Mirrors .github/workflows/fuzz.yml so a local run exercises exactly what the
# scheduled job runs: same targets, same nightly, same input cap, same default
# time budget per target.
#
# Usage: tools/fuzz-budget.sh [seconds-per-target]   (default: 60)
#
# Requires cargo-fuzz and the nightly named in fuzz/rust-toolchain.toml:
#   cargo install cargo-fuzz --version 0.13.2 --locked
#   rustup toolchain install nightly-2026-07-28 --profile minimal --component rust-src

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${BOLD}=== [FUZZ] $1 ===${NC}"; }
success() { echo -e "${GREEN}OK: $1${NC}"; }
fail() { echo -e "${RED}FAILED: $1${NC}"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUDGET="${1:-60}"

# cargo-fuzz shells out to `cargo` in the *current* directory, so a bare
# `cargo fuzz` from the repo root picks up the root rust-toolchain.toml (a
# stable pin) and dies on `-Zsanitizer`. The nightly is named explicitly, the
# same way the workflow names it, instead of relying on rustup's cwd lookup.
NIGHTLY="$(sed -n 's/^channel = "\(.*\)"/\1/p' fuzz/rust-toolchain.toml)"
[ -n "$NIGHTLY" ] || fail "could not read the pinned nightly from fuzz/rust-toolchain.toml"
TARGETS=(onnx_parser state_envelope gguf_reader)

command -v cargo-fuzz >/dev/null 2>&1 ||
    fail "cargo-fuzz is not installed (cargo install cargo-fuzz --version 0.13.2 --locked)"

for target in "${TARGETS[@]}"; do
    step "$target (${BUDGET}s budget, max_len 65536)"
    cargo "+${NIGHTLY}" fuzz run "$target" -- "-max_total_time=${BUDGET}" "-max_len=65536" ||
        fail "$target"
    success "$target"
done

echo -e "\n${BOLD}${GREEN}All fuzz budgets passed.${NC}\n"
