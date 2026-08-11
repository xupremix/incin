#!/usr/bin/env bash
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

step() {
    echo -e "\n${BOLD}${YELLOW}=== [FEATURE MATRIX] $1 ===${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

step "1. incin-backends: CPU"
cargo check -p incin-backends --no-default-features --features std,cpu,target-api || fail "incin-backends CPU check failed"
success "incin-backends CPU OK"

step "2. incin-backends: CUDA"
cargo check -p incin-backends --no-default-features --features std,cuda,target-api || fail "incin-backends CUDA check failed"
success "incin-backends CUDA OK"

step "3. incin-backends: WGPU"
cargo check -p incin-backends --no-default-features --features std,wgpu,target-api || fail "incin-backends WGPU check failed"
success "incin-backends WGPU OK"

step "4. incin-backends: Metal"
cargo check -p incin-backends --no-default-features --features std,metal,target-api || fail "incin-backends Metal check failed"
success "incin-backends Metal OK"

step "5. incin-backends: External Candle"
cargo check -p incin-backends --no-default-features --features std,external-candle,target-api || fail "incin-backends External Candle check failed"
success "incin-backends External Candle OK"

step "6. incin: CPU facade"
cargo check -p incin --no-default-features --features std,cpu,target-api || fail "incin CPU check failed"
success "incin CPU OK"

step "7. incin: CUDA facade"
cargo check -p incin --no-default-features --features std,cuda,target-api || fail "incin CUDA check failed"
success "incin CUDA OK"

step "8. incin: WGPU facade"
cargo check -p incin --no-default-features --features std,wgpu,target-api || fail "incin WGPU check failed"
success "incin WGPU OK"

step "9. incin: Metal facade"
cargo check -p incin --no-default-features --features std,metal,target-api || fail "incin Metal check failed"
success "incin Metal OK"

step "10. incin: External Candle facade"
cargo check -p incin --no-default-features --features std,external-candle,target-api || fail "incin External Candle check failed"
success "incin External Candle OK"

echo -e "\n${BOLD}${GREEN}=== ALL FEATURE MATRIX CHECKS PASSED ===${NC}"
