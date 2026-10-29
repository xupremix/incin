//! Every CUDA source the retained `codegen` modules render must actually compile.
//!
//! This file exists because rendering returns a `String`, and a `String` is
//! easy to assert about without ever asking a compiler whether it is valid
//! CUDA C. The module tests check that emitted text contains the substrings
//! they expect, and it does — which is how an entire generation of emitters
//! carried an `#include <math.h>` that NVRTC rejects outright (it compiles a
//! translation unit with no host headers on the include path) while every
//! text assertion passed. Nothing executed them, so nothing held them to a
//! compiler. Per #111 those emitters are gone; what remains is the IR/DSL
//! kernel renderer behind `dsl` and `jit`, and it earns the same check.
//!
//! A rendered kernel is only worth anything if NVRTC accepts it, so that is
//! what these check — through `cuda::testing::compile_for_device`, which
//! delegates to the production `compile_ptx_for_arch` path (same include
//! resolution, same architecture selection), for the same reason
//! `kernel::tests` does: certifying a source under options nothing builds it
//! with is not certifying it.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test codegen_nvrtc_smoke -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::codegen::{define_binary_custom_op, define_unary_custom_op};
use incin_core::tensor::dtype::DTypeId;

/// Aborts unless a CUDA device is present.
///
/// # Panics
///
/// If no CUDA device can be opened on ordinal 0.
fn require_cuda() {
    assert!(
        cudarc::driver::CudaContext::new(0).is_ok(),
        "no CUDA device, but this test is #[ignore]d -- running it is an explicit request \
         for hardware. Skipping here would report `ok` for a test that compiled nothing."
    );
}

/// Compiles `source` with NVRTC for the running device's architecture.
///
/// Panics with the compiler log on failure, because the log is the entire
/// value: "it did not compile" is not actionable, and the `math.h` failure was
/// diagnosed from exactly this text.
fn must_compile(label: &str, source: &str) {
    match incin_backends::cuda::testing::compile_for_device(source) {
        Ok(_) => {}
        Err(error) => panic!(
            "{label} rendered CUDA that NVRTC refused.\n\
             A module whose output does not compile cannot have a consumer, which is how \
             every removed codegen module carried an `#include <math.h>` NVRTC cannot resolve.\n\n\
             {error:?}\n\n--- source ---\n{source}"
        ),
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn every_retained_codegen_emitter_renders_compilable_cuda() {
    require_cuda();

    let unary = define_unary_custom_op("smoke_swish", DTypeId::F32, |x| {
        let s = incin_backends::codegen::sigmoid(x.clone());
        x * s
    });
    must_compile(
        "KernelDefinition::render_forward_cuda",
        &unary.render_forward_cuda(),
    );
    must_compile(
        "KernelDefinition::render_backward_cuda",
        &unary
            .render_backward_cuda(0)
            .expect("unary definition carries a derivative"),
    );

    let binary = define_binary_custom_op("smoke_gated_linear_unit", DTypeId::F32, |a, b| {
        a * incin_backends::codegen::sigmoid(b)
    });
    must_compile("binary forward", &binary.render_forward_cuda());
    for input_idx in 0..binary.input_arity {
        must_compile(
            &format!("binary backward {input_idx}"),
            &binary
                .render_backward_cuda(input_idx)
                .expect("each input carries a derivative"),
        );
    }
}
