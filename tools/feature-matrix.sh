#!/usr/bin/env bash
set -euo pipefail

# Supported feature contract matrix. This is the single source used by CI and
# tools/ci-local.sh. It deliberately names supported combinations instead of
# pretending that the Cartesian product of every opt-in is a product contract.
# These rows compile only; hardware-dependent runtime coverage belongs to the
# hardware workflow and the focused preview tests.

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
TIER="${1:-all}"

if [[ "$TIER" != "all" && "$TIER" != "stable" ]]; then
    echo "usage: tools/feature-matrix.sh [stable]" >&2
    exit 2
fi

run_row() {
    local name="$1"
    shift
    printf '\n%b=== [FEATURE CONTRACT] %s ===%b\n' "$BOLD$YELLOW" "$name" "$NC"
    "$@"
    printf '%b✓ %s%b\n' "$GREEN" "$name" "$NC"
}

echo "Supported feature contract matrix"

# The compatibility promise has its own deterministic MSRV row. CI runs this
# mode under Rust 1.88; keeping it executable also makes a locally installed
# MSRV prove the exact same union rather than a hand-maintained approximation.
if [[ "$TIER" == "stable" ]]; then
    exec cargo xtask feature-msrv
fi

# Core: no_std, normal std, and representative independent opt-ins. The full
# core powerset remains in the cargo-hack job because it is small and useful.
run_row "core-no-default" cargo check -p incin-core --no-default-features
run_row "core-default" cargo check -p incin-core
run_row "core-compiled-distributed" cargo check -p incin-core --no-default-features --features std,compiled,distributed
run_row "core-serialization" cargo check -p incin-core --no-default-features --features std,postcard,safetensors,serde_json

# Backends: minimal/default CPU, each compiled backend family, external
# Candle, authoring-adjacent contracts, telemetry, and distributed transports.
run_row "backend-cpu-minimal" cargo check -p incin-backends --no-default-features --features std,cpu
run_row "backend-default" cargo check -p incin-backends
run_row "backend-cpu-blas" cargo check -p incin-backends --no-default-features --features std,cpu,cpu-blas
run_row "backend-wgpu-compile" cargo check -p incin-backends --no-default-features --features std,wgpu
run_row "backend-cuda-compile" cargo check -p incin-backends --no-default-features --features std,cuda
# `cuda-vendor` is a supported compatibility feature layered on the native
# CUDA backend. It has no separate implementation until vendor-kernel dispatch
# is added, but it must remain a compiling public configuration.
run_row "backend-cuda-vendor-compile" cargo check -p incin-backends --no-default-features --features std,cuda-vendor
run_row "backend-metal-compile" cargo check -p incin-backends --no-default-features --features std,metal
run_row "backend-external-candle" cargo check -p incin-backends --no-default-features --features std,external-candle
run_row "backend-telemetry" cargo check -p incin-backends --no-default-features --features std,cpu,telemetry
run_row "backend-distributed-reference" cargo check -p incin-backends --no-default-features --features std,cpu,distributed-reference
run_row "backend-distributed-nccl-compile" cargo check -p incin-backends --no-default-features --features std,distributed-nccl
run_row "backend-cpu-wgpu-telemetry" cargo check -p incin-backends --no-default-features --features std,cpu,wgpu,telemetry

# Facade: default/no-default, public preview surfaces, each backend family,
# extension contracts, and representative orthogonal interactions.
run_row "facade-default" cargo check -p incin
run_row "facade-std-no-backend" cargo check -p incin --no-default-features --features std
run_row "facade-training" cargo check -p incin --no-default-features --features std,cpu,train
run_row "facade-compiled" cargo check -p incin --no-default-features --features std,cpu,compiled
run_row "facade-telemetry" cargo check -p incin --no-default-features --features std,cpu,telemetry
run_row "facade-backend-authoring" cargo check -p incin --no-default-features --features std,cpu,backend-authoring
run_row "facade-wgpu-compile" cargo check -p incin --no-default-features --features std,wgpu
run_row "facade-cuda-compile" cargo check -p incin --no-default-features --features std,cuda
run_row "facade-metal-compile" cargo check -p incin --no-default-features --features std,metal
run_row "facade-external-candle" cargo check -p incin --no-default-features --features std,external-candle
run_row "facade-distributed-reference" cargo check -p incin --no-default-features --features std,cpu,distributed-reference
run_row "facade-distributed-nccl-compile" cargo check -p incin --no-default-features --features std,distributed-nccl
run_row "facade-telemetry" cargo check -p incin --no-default-features --features std,cpu,telemetry
run_row "facade-compiled-telemetry" cargo check -p incin --no-default-features --features std,cpu,compiled,telemetry

printf '\n%b=== Supported feature contract matrix: PASS ===%b\n' "$BOLD$GREEN" "$NC"
