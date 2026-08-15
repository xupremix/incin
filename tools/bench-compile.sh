#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

printf 'case\tseconds\tstatus\n'
run_case() {
    local name="$1"
    shift
    local start end
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

run_case default-lib cargo check -p incin --lib
run_case examples cargo check -p incin --examples
run_case book-feature-surface cargo check -p incin --lib --features 'target-api backend-authoring'
run_case transformer-proof cargo test -p incin --test transformer_block --no-run
