#!/usr/bin/env bash
set -e

# Local CI Check Script for Incin
# Replicates GitHub Actions CI pipeline locally.

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

step() {
    echo -e "\n${BOLD}${YELLOW}=== [CI STEP] $1 ===${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

SKIP_POWERSET=1
if [ "$1" == "--powerset" ] || [ "$1" == "--full" ]; then
    SKIP_POWERSET=0
fi

step "1. Formatting Check"
cargo fmt --all -- --check || fail "Formatting check failed! Run 'cargo fmt --all' to fix."
success "Formatting OK"

step "2. Ledger & Governance Validation"
cargo xtask ledger || fail "Ledger validation failed!"
cargo test -p xtask || fail "xtask unit tests failed!"
cargo xtask budgets || fail "Budgets enforcement failed!"
cargo xtask docs --check || fail "README feature table check failed!"
cargo xtask feature-matrix || fail "Backend feature matrix failed!"
tools/audit-shapes.sh --check || fail "Shape audit check failed!"
success "Ledger & Governance OK"

step "3. CPU Clippy Lints"
cargo clippy --all-targets --no-default-features --features incin-backends/cpu,incin/cpu -- -D warnings || fail "Clippy failed!"
success "Clippy OK"

step "4. CPU Unit & Integration Tests"
cargo test --all-targets --no-default-features --features incin-backends/cpu,incin/cpu || fail "CPU unit/integration tests failed!"
success "CPU Tests OK"

step "5. Build Examples"
cargo build --examples --no-default-features --features incin-backends/cpu,incin/cpu || fail "Examples build failed!"
success "Examples Build OK"

step "6. Preview-Feature Tests"
cargo test -p incin --features train --test trainer
cargo test -p incin-core --features distributed --test mesh_compile
cargo test -p incin-core --features distributed --test mesh_bind
cargo test -p incin-core --features distributed --test placement_rules
cargo test -p incin-core --features distributed --test placement_tensor
cargo test -p incin-core --features distributed --test collective_plan
cargo test -p incin-core --features distributed --test data_parallel
cargo test -p incin-core --features distributed --test tensor_parallel
cargo test -p incin-core --features distributed --test pipeline
cargo test -p incin-core --features distributed --test hybrid_plan
cargo test -p incin-backends --features distributed-reference --test reference_collectives
cargo test -p incin-backends --features distributed-reference --test data_parallel_reference
cargo test -p incin-backends --features distributed-reference --test tensor_parallel_reference
cargo test -p incin-backends --features distributed-reference --test pipeline_reference
cargo test -p incin-backends --features distributed-reference,autotune --test collective_tuning
cargo test -p incin-backends --features autotune --test tuning_identity
cargo test -p incin-backends --features autotune --test tuning_cache
cargo test -p incin-backends --features autotune --test tuning_service
cargo test -p incin-backends --features distributed-nccl --test nccl_contract
cargo test -p incin --features distributed-nccl --test rendezvous
cargo test -p incin --features distributed-nccl --test dp2_network --no-run
cargo test -p incin --features distributed-nccl --test tp2_network --no-run
cargo test -p incin --features distributed-nccl --test pp2_network --no-run
success "Preview-Feature Tests OK"

step "7. Isolated Bare-CPU & CPU-BLAS Tests"
cargo test -p incin-backends --no-default-features --features std,cpu || fail "Bare-CPU tests failed!"
cargo test -p incin-backends --no-default-features --features std,cpu,cpu-blas || fail "CPU-BLAS tests failed!"
success "Isolated CPU Tests OK"

step "8. Core no_std Check"
cargo check -p incin-core --no-default-features || fail "core no_std check failed!"
success "Core no_std OK"

step "9. WGPU Software Adapter Tests & Lint"
cargo test -p incin-backends --all-targets --no-default-features --features std,cpu,wgpu || fail "WGPU tests failed!"
cargo clippy --workspace --all-targets --no-default-features --features incin-backends/cpu,incin-backends/wgpu,incin/cpu,incin/wgpu -- -D warnings || fail "WGPU clippy failed!"
success "WGPU OK"

step "10. CUDA Check & Tests"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected via nvidia-smi. Running CUDA compile check and hardware tests..."
    cargo check -p incin-backends --all-targets --no-default-features --features std,cpu,cuda || fail "CUDA compile check failed!"
    cargo test -p incin-backends --no-default-features --features std,cpu,cuda || fail "CUDA hardware runtime tests failed!"
    success "CUDA Hardware & Runtime Tests OK"
else
    echo -e "${YELLOW}⚠ WARNING: No CUDA hardware / nvidia-smi detected on this system. Running CUDA compile check only.${NC}"
    cargo check -p incin-backends --all-targets --no-default-features --features std,cpu,cuda || fail "CUDA compile check failed!"
    success "CUDA Compile Check OK (hardware runtime tests skipped)"
fi

step "11. Documentation Build & Public Doctests"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features --features incin-backends/cpu,incin/cpu || fail "Default doc build failed!"
RUSTDOCFLAGS="-D warnings" cargo doc -p incin -p incin-backends -p incin-core --no-deps --no-default-features --features incin/cpu,incin/cuda,incin/wgpu,incin/telemetry,incin/external-candle || fail "Full doc build failed!"
cargo test --workspace --doc --features incin-core/distributed || fail "Public doctests failed!"
success "Documentation OK"

if [ $SKIP_POWERSET -eq 0 ]; then
    step "12. Feature Powerset (cargo hack)"
    if command -v cargo-hack &> /dev/null; then
        cargo hack check -p incin-core --feature-powerset --all-targets --exclude-features nightly
        cargo hack check -p incin-macros --feature-powerset --no-dev-deps --exclude-features nightly
        cargo hack check -p incin-diagnostics --feature-powerset --all-targets
        cargo hack check -p incin-backends --feature-powerset --all-targets --exclude-features candle
        cargo hack check -p incin-exclude-features nightly,candle
        success "Powerset OK"
    else
        echo -e "${YELLOW}cargo-hack not found, skipping powerset check. Install with: cargo install cargo-hack${NC}"
    fi
fi

echo -e "\n${BOLD}${GREEN}========================================${NC}"
echo -e "${BOLD}${GREEN} ALL LOCAL CI CHECKS PASSED SUCCESSFULLY! ${NC}"
echo -e "${BOLD}${GREEN}========================================${NC}\n"
