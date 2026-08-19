#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

failures=0
# Portable stand-in for `rg -n PATTERN PATHS... [--glob '*.rs'] [--glob '!**/tests.rs']`:
# splits the trailing `--glob` flags this file used to forward straight to
# ripgrep into `grep -r`'s `--include`/`--exclude` (both match by basename, so
# `!**/tests.rs` and `--exclude=tests.rs` agree here), so the check has no
# dependency on ripgrep being installed.
check_absent() {
    local label="$1"
    local pattern="$2"
    shift 2
    local paths=()
    local grep_args=()
    while (($#)); do
        case "$1" in
            --glob)
                local glob="$2"
                shift 2
                if [[ "$glob" == '!'* ]]; then
                    grep_args+=(--exclude="${glob##*/}")
                else
                    grep_args+=(--include="$glob")
                fi
                ;;
            *)
                paths+=("$1")
                shift
                ;;
        esac
    done
    if grep -rnE "${grep_args[@]}" "$pattern" "${paths[@]}"; then
        printf 'architecture check failed: %s\n' "$label" >&2
        failures=$((failures + 1))
    fi
}

check_absent "internal wildcard preludes" \
    '^[[:space:]]*use[[:space:]]+(crate|incin_core)::prelude::\*;' \
    crates/incin-core/src crates/incin-backends/src crates/incin-data/src \
    --glob '*.rs' --glob '!**/tests.rs'
check_absent "named public-prelude dependencies" \
    '(^|[^[:alnum:]_])(crate|incin_core)::prelude::[A-Za-z_][A-Za-z0-9_]*' \
    crates/incin-core/src crates/incin-backends/src crates/incin-data/src crates/incin/src \
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
check_absent "legacy erased variable contract" \
    '\bRawVar\b' crates/incin-core/src crates/incin-backends/src crates/incin-macros/src \
    --glob '*.rs'
check_absent "legacy module parameter traversal" \
    '\bParameters[[:space:]]*<|\bnamed_parameters[[:space:]]*\(' \
    crates/incin-core/src/nn crates/incin-macros/src --glob '*.rs'
check_absent "crate-wide core warning suppressions" \
    '^#!\[allow\((dead_code|unused_imports)\)\]' crates/incin-core/src/lib.rs
if sed -n '/^pub trait Backend:/,/^}/p' crates/incin-core/src/tensor/backend.rs | grep -nE 'HostInterop|AutogradBackend'; then
    echo "architecture check failed: Backend requires optional host/autograd capabilities" >&2
    failures=$((failures + 1))
fi

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
if ! grep -qE 'FOUNDATION|OPERATION SEMANTICS|TENSOR RUNTIME|NN and state' docs/HANDOFF.md; then
    echo "architecture check failed: docs/HANDOFF.md has no layer contract" >&2
    failures=$((failures + 1))
fi

if (( failures != 0 )); then
    exit 1
fi
echo "architecture checks passed"
