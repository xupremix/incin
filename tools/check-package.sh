#!/usr/bin/env bash
set -euo pipefail

# Verify the failure modes that make exported artifacts irreproducible: a
# required source file missing from git, or omitted from a Cargo package
# archive.  Keep this check shell-only so it can run before Rust compilation in
# CI and from a clean checkout.

packages=(
  incin
  incin-backends
  incin-core
  incin-data
  incin-diagnostics
  incin-lsp
  incin-macros
  incin-telemetry
  incin-viz-plugin-api
  incin-viz
)

package_list=$(mktemp)
trap 'rm -f "$package_list"' EXIT

for package in "${packages[@]}"; do
  : >"$package_list"
  cargo package -p "$package" --allow-dirty --no-verify --list >"$package_list"

  # Cargo uses the workspace's SPDX license expression for these crates. The
  # root license files are not copied into each package archive, so validate
  # the manifest metadata separately below instead of requiring duplicate
  # files in every crate directory.
  for path in Cargo.toml README.md src/lib.rs; do
    if ! grep -Fxq "$path" "$package_list"; then
      printf 'required file is not in the %s package: %s\n' "$package" "$path" >&2
      exit 1
    fi
  done

  manifest="crates/$package/Cargo.toml"
  if [[ "$package" == "incin" ]]; then
    binary_paths=(src/bin/cargo-incin.rs)
  elif [[ "$package" == "incin-lsp" ]]; then
    binary_paths=(src/main.rs src/bin/mock_rust_analyzer.rs)
  elif [[ "$package" == "incin-viz" ]]; then
    binary_paths=(src/main.rs)
  else
    binary_paths=()
  fi
  for path in "${binary_paths[@]}"; do
    if ! grep -Fxq "$path" "$package_list"; then
      printf 'binary entry point is not in the %s package: %s\n' "$package" "$path" >&2
      exit 1
    fi
  done

  if ! grep -Eq '^(license[[:space:]]*=|license\.workspace[[:space:]]*=)' "$manifest"; then
    printf 'package has no declared license metadata: %s\n' "$manifest" >&2
    exit 1
  fi

  if ! git ls-files --error-unmatch "$manifest" >/dev/null; then
    printf 'package manifest is not tracked: %s\n' "$manifest" >&2
    exit 1
  fi
done

for license in LICENSE_APACHE LICENSE_MIT; do
  git ls-files --error-unmatch "$license" >/dev/null || {
    printf 'workspace license file is not tracked: %s\n' "$license" >&2
    exit 1
  }
done

printf 'package reproducibility check passed: %s publishable crates\n' "${#packages[@]}"
