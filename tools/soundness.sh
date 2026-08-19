#!/usr/bin/env bash
set -euo pipefail

# Dynamic soundness gate for Incin.
#
# The workspace carries roughly 170 `unsafe` sites, almost all of them in the
# CPU backend's SIMD kernels. Compiling them proves nothing about them, and the
# ordinary test suite does not exercise several of them at all. This script is
# the gate that does.
#
# Usage: tools/soundness.sh [miri|asan|tsan|all]   (default: all)
#
# Requires a nightly toolchain with the `miri` and `rust-src` components:
#   rustup toolchain install nightly --component miri,rust-src

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

step() { echo -e "\n${BOLD}${YELLOW}=== [SOUNDNESS] $1 ===${NC}"; }
success() { echo -e "${GREEN}OK: $1${NC}"; }
fail() { echo -e "${RED}FAILED: $1${NC}"; exit 1; }

WHICH="${1:-all}"
TARGET="$(rustc -vV | sed -n 's/^host: //p')"

# Miri runs under Tree Borrows rather than Stacked Borrows. This is not a
# loosening to make our own code pass: under Stacked Borrows the run aborts
# inside crossbeam-epoch 0.9.18, which rayon pulls in, at
# `internal.rs:549 &*local_ptr` in `Local::element_of`. That is a known
# violation in a dependency we do not control. Tree Borrows accepts it and
# still rejects the aliasing mistakes this gate exists to catch. If rayon ever
# ships a crossbeam-epoch that satisfies Stacked Borrows, drop the flag.
# The core test-only DummyBackend has a deliberately shape-only storage type
# (`Vec<usize>`), while StorageBackend::metadata returns a borrowed TensorMeta.
# It therefore uses Box::leak to manufacture the borrowed view. Ignore only
# those process-lifetime fixture allocations so Miri can still check aliasing
# and undefined behavior in the full test run. The real CPU backend remains
# leak-checked by the ASan backend leg below.
MIRIFLAGS_COMMON="-Zmiri-tree-borrows -Zmiri-disable-isolation -Zmiri-ignore-leaks"

# Miri's software floating-point implementation is not a suitable oracle for
# the backend's finite-difference tolerances: these numerical tests fail only
# because interpreted `exp`/transcendental results differ in the last few
# digits (or amplify that difference through a gradcheck). Keep those tests in
# the ordinary and ASan suites, while excluding only these exact numerical
# cases from the aliasing/UB run below.
MIRI_BACKEND_NUMERIC_SKIPS=(
    --skip cpu::ops::elementwise::tests::softmax_gradcheck
    --skip cpu::ops::elementwise::tests::log_softmax_gradcheck
    --skip cpu::ops::elementwise::tests::swish_forward_and_gradcheck
    --skip cpu::ops::elementwise::tests::tanh_gradcheck
    --skip cpu::ops::elementwise::tests::log_forward_gradcheck_and_domain_propagation
    --skip cpu::ops::elementwise::tests::sigmoid_forward_and_gradcheck
    --skip cpu::ops::elementwise::tests::acosh_gradcheck
    --skip cpu::ops::elementwise::tests::trig_and_hyperbolic_gradchecks
    --skip cpu::ops::elementwise_kernel::tests::unary_family_uses_native_float_compute
    --skip cpu::ops::loss::tests::cross_entropy_loss_gradcheck
    --skip cpu::ops::loss::tests::bce_with_logits_gradcheck
)

run_miri() {
    step "Miri: incin-core (interpreter, Tree Borrows)"
    MIRIFLAGS="$MIRIFLAGS_COMMON" cargo +nightly miri test \
        -p incin-core --no-default-features --features std --lib \
        || fail "miri: incin-core"
    success "miri: incin-core"

    step "Miri: incin-backends CPU (interpreter, Tree Borrows)"
    MIRIFLAGS="$MIRIFLAGS_COMMON" cargo +nightly miri test \
        -p incin-backends --no-default-features --features std,cpu --lib \
        -- "${MIRI_BACKEND_NUMERIC_SKIPS[@]}" \
        || fail "miri: incin-backends"
    success "miri: incin-backends"
}

# AddressSanitizer stands in for valgrind here. It catches the same class of
# error (out-of-bounds, use-after-free, leaks), it understands Rust's allocator
# rather than guessing at it, and it runs at roughly test speed instead of
# roughly 50x slower. Nothing below needs valgrind to be installed.
#
# `--all-targets` is used for incin-backends because every one of its test
# targets is a runtime test. For incin-core and incin only the library tests
# run: their integration targets include trybuild and facade-contract cases
# that spawn a nested `cargo`, and RUSTFLAGS propagates into that build, which
# links proc-macro `.so` files against an ASan runtime the host cargo does not
# load (`undefined symbol: __asan_option_detect_stack_use_after_return`).
# Excluding them costs no signal: they assert on compiler diagnostics, and
# never execute a kernel.
run_asan() {
    # The core DummyBackend has the same intentional process-lifetime metadata
    # fixture allocation described above. Keep leak detection enabled for the
    # real backend target, but disable it for the core fixture target.
    export ASAN_OPTIONS=detect_leaks=1

    step "AddressSanitizer + LeakSanitizer (baseline target features)"
    CARGO_TARGET_DIR=target/asan RUSTFLAGS="-Zsanitizer=address" \
        cargo +nightly test -p incin-backends --no-default-features \
        --features std,cpu --all-targets --target "$TARGET" \
        || fail "asan: incin-backends"
    ASAN_OPTIONS=detect_leaks=0 CARGO_TARGET_DIR=target/asan RUSTFLAGS="-Zsanitizer=address" \
        cargo +nightly test -p incin-core --no-default-features \
        --features std --lib --target "$TARGET" \
        || fail "asan: incin-core"
    CARGO_TARGET_DIR=target/asan RUSTFLAGS="-Zsanitizer=address" \
        cargo +nightly test -p incin --no-default-features \
        --features cpu --lib --target "$TARGET" \
        || fail "asan: incin"
    success "asan: baseline"

    # `simd_lanes()` is a `const fn` over `target_feature`, so on a default
    # x86_64 build `simd_lanes::<f32>()` is 4 and every `if LANES >= 8` branch
    # into the dense AVX2 kernels is const-false and eliminated. Those kernels
    # are therefore unreachable in the build CI otherwise tests, and this leg
    # is the only thing that executes them. The runtime-detected iteration
    # kernels are reached either way; the dense ones are not.
    step "AddressSanitizer with +avx2 (reaches the dense SIMD kernels)"
    if ! grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then
        echo -e "${YELLOW}Host does not report avx2; skipping.${NC}"
        return
    fi
    CARGO_TARGET_DIR=target/asan-avx2 \
        RUSTFLAGS="-Zsanitizer=address -C target-feature=+avx2,+fma" \
        cargo +nightly test -p incin-backends --no-default-features \
        --features std,cpu --all-targets --target "$TARGET" \
        || fail "asan+avx2: incin-backends"
    success "asan: +avx2"
}

# ThreadSanitizer is what would check the claim the SIMD kernels rest on: that
# the chunks handed to rayon workers by `spare_capacity_mut().par_chunks_mut(..)`
# are disjoint, so `Vec::set_len` afterwards observes fully initialized memory.
# It is not wired up. Two attempts, both dead ends:
#
#   1. `-Zbuild-std` (needed so TSan instruments std, since an uninstrumented
#      std ABI-mismatches against a TSan-built crate): incin-backends depends
#      on `hashbrown` directly, and build-std vendors its own copy for `alloc`,
#      so linking fails with E0464 "multiple candidates for rmeta dependency
#      alloc found". `-Cunsafe-allow-abi-mismatch=sanitizer` sidesteps the
#      build but not honestly: it allows racing on std's own primitives, so
#      TSan reported 151 warnings, all inside `rayon_core::sleep::Sleep`,
#      `LockLatch`, the std test-harness mpmc channel, and `__tsan_memcpy` /
#      `free` — every legitimate rayon handoff flagged because TSan cannot see
#      synchronization inside the std it wasn't told to instrument. 0 of the
#      151 were in incin code.
#   2. Without `-Zbuild-std`: `error: mixing -Zsanitizer will cause an ABI
#      mismatch ... -Zsanitizer=thread in this crate is incompatible with
#      -Zsanitizer being unset in dependency core`. Does not build at all.
#
# Fixing this needs either hashbrown removed as a direct incin-backends
# dependency (so build-std's copy is the only one) or a resolution for
# rust-lang/wg-cargo-std-aware's known build-std + third-party-hashbrown
# conflict. Until one of those lands, the race-freedom claim above is
# unverified by an automated gate; it rests on code review of the
# `par_chunks_mut` call sites instead.
run_tsan() {
    fail "tsan: not wired up (hashbrown vs -Zbuild-std conflict; see comment above run_tsan in this file)"
}

# `tsan` is deliberately excluded from `all`: it cannot currently pass (see
# the comment above run_tsan), so including it would make every default run
# fail on a leg nothing here can fix yet. Run it explicitly if you want the
# failure demonstrated.
case "$WHICH" in
    miri) run_miri ;;
    asan) run_asan ;;
    tsan) run_tsan ;;
    all)  run_miri; run_asan ;;
    *)    fail "unknown selector '$WHICH' (expected miri, asan, tsan or all)" ;;
esac

echo -e "\n${BOLD}${GREEN}Soundness gate passed.${NC}"
