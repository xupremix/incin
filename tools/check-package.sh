#!/usr/bin/env bash
set -euo pipefail

# Verify the two failure modes that previously made exported snapshots
# irreproducible: a required source file missing from git, or omitted from the
# Cargo package archive.  Keep this check shell-only so it can run before Rust
# compilation in CI and from a clean checkout.

required_files=(
  crates/incin-core/src/lib.rs
  crates/incin-core/src/dist/mod.rs
  crates/incin-backends/src/dist/mod.rs
)

for path in "${required_files[@]}"; do
  if ! git ls-files --error-unmatch "$path" >/dev/null; then
    printf 'required source is not tracked: %s\n' "$path" >&2
    exit 1
  fi
done

package_list=$(mktemp)
trap 'rm -f "$package_list"' EXIT

cargo package -p incin-core --allow-dirty --no-verify --list >"$package_list"

for path in src/lib.rs src/dist/mod.rs; do
  if ! grep -Fxq "$path" "$package_list"; then
    printf 'required source is not in the incin-core package: %s\n' "$path" >&2
    exit 1
  fi
done

printf 'package reproducibility check passed\n'
