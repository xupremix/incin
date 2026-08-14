#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

failures=0
check_absent() {
    local label="$1"
    local pattern="$2"
    shift 2
    if rg -n "$pattern" "$@"; then
        printf 'architecture check failed: %s\n' "$label" >&2
        failures=$((failures + 1))
    fi
}

check_absent "internal wildcard preludes" \
    '^[[:space:]]*use[[:space:]]+(crate|incin_core)::prelude::\*;' \
    crates/incin-core/src crates/incin-backends/src crates/incin-data/src \
    --glob '*.rs' --glob '!**/tests.rs'
check_absent "tensor depends upward on nn" \
    'crate::nn::|incin_core::nn::' \
    crates/incin-core/src/tensor
check_absent "shape proof depends upward on execution" \
    'crate::exec::ProofLevel' \
    crates/incin-core/src/shapes
check_absent "autoref module traversal" \
    'Autoref|&&self\.|&mut &mut' \
    crates/incin-core/src/nn/module.rs crates/incin-core/src/nn/module_optional.rs \
    crates/incin-macros/src/module.rs
check_absent "crate-wide core warning suppressions" \
    '^#!\[allow\((dead_code|unused_imports)\)\]' crates/incin-core/src/lib.rs

if [[ ! -x tools/check-package.sh ]]; then
    echo "architecture check failed: tools/check-package.sh is missing or not executable" >&2
    failures=$((failures + 1))
fi
if [[ ! -x tools/check-public-api.sh ]]; then
    echo "architecture check failed: tools/check-public-api.sh is missing or not executable" >&2
    failures=$((failures + 1))
fi
if [[ ! -x tools/check-large-files.sh ]]; then
    echo "architecture check failed: tools/check-large-files.sh is missing or not executable" >&2
    failures=$((failures + 1))
elif ! tools/check-large-files.sh; then
    failures=$((failures + 1))
fi
if ! rg -q 'FOUNDATION|OPERATION SEMANTICS|TENSOR RUNTIME|NN and state' docs/HANDOFF.md; then
    echo "architecture check failed: docs/HANDOFF.md has no layer contract" >&2
    failures=$((failures + 1))
fi

if (( failures != 0 )); then
    exit 1
fi
echo "architecture checks passed"
