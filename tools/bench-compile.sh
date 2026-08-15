#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

printf 'case\tseconds\tstatus\n'
printf 'rustc\t%s\n' "$(rustc -Vv | sed -n '1p')"
printf 'cargo\t%s\n' "$(cargo -V)"
printf 'host\t%s\n' "$(uname -srmo)"

if [[ "${CLEAN:-0}" == 1 && "${CLEAN_EACH:-0}" != 1 ]]; then
    cargo clean -p incin
fi

run_case() {
    local name="$1"
    shift
    local start end
    if [[ "${CLEAN_EACH:-0}" == 1 ]]; then
        cargo clean -p incin >/tmp/incin-compile-bench-clean.log 2>&1
    fi
    start="$(date +%s%N)"
    if "$@" >/tmp/incin-compile-bench.log 2>&1; then
        end="$(date +%s%N)"
        awk -v n="$name" -v s="$((end - start))" 'BEGIN { printf "%s\t%.3f\tpassed\n", n, s / 1000000000 }'
    else
        end="$(date +%s%N)"
        awk -v n="$name" -v s="$((end - start))" 'BEGIN { printf "%s\t%.3f\tfailed\n", n, s / 1000000000 }'
        sed -n '1,80p' /tmp/incin-compile-bench.log >&2
        return 1
    fi
}

run_case tiny cargo check -p incin --example compile_fixture_tiny
run_case mlp cargo check -p incin --example compile_fixture_mlp
run_case cnn cargo check -p incin --example compile_fixture_cnn
run_case transformer-static cargo check -p incin --example compile_fixture_transformer_static
run_case transformer-mixed cargo check -p incin --example compile_fixture_transformer_mixed
run_case transformer-dyn cargo check -p incin --example compile_fixture_transformer_dyn
run_case transformer-proof cargo test -p incin --test transformer_block --no-run

# The second invocation measures the practical incremental path for the same
# representative portfolio without rebuilding the dependency graph.
run_case incremental-portfolio cargo check -p incin --examples
