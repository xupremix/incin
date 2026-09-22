//! Every CUDA source a `codegen` module renders must actually compile.
//!
//! This file exists because all 21 modules emitted `#include <math.h>`, which
//! NVRTC rejects outright -- it compiles a translation unit with no host
//! headers on the include path:
//!
//! ```text
//! catastrophic error: cannot open source file "math.h"
//! ```
//!
//! So not one of them could produce a usable kernel. That is the shared reason
//! behind #111's "21 modules with no consumer": they could not have had one.
//! Nothing caught it because rendering returns a `String`, and a `String` is
//! easy to assert about without ever asking a compiler whether it is valid CUDA
//! C. The module tests checked that the text contained the substrings they
//! expected, and it did.
//!
//! A rendered kernel is only worth anything if NVRTC accepts it, so that is
//! what these check -- against the real device's architecture, for the same
//! reason `kernel::tests` does: certifying a source for a target nothing runs
//! is not certifying it.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test codegen_nvrtc_smoke -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::codegen::CompositeFusionSpec;
use incin_backends::codegen::{normalization, pointwise, reduction, vectorized};
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
             every codegen module carried an `#include <math.h>` NVRTC cannot resolve.\n\n\
             {error:?}\n\n--- source ---\n{source}"
        ),
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn every_codegen_module_renders_compilable_cuda() {
    require_cuda();

    must_compile(
        "CompositeFusionSpec",
        &CompositeFusionSpec::swiglu_residual("smoke_fusion", DTypeId::F32).render_cuda(),
    );

    for kind in [
        normalization::NormKind::LayerNorm,
        normalization::NormKind::RmsNorm,
    ] {
        let spec = normalization::NormalizationSpec::new(
            "smoke_norm",
            kind,
            DTypeId::F32,
            256,
            1e-5,
            true,
        );
        must_compile("NormalizationSpec::forward", &spec.render_cuda_forward());
        must_compile("NormalizationSpec::backward", &spec.render_cuda_backward());
    }

    must_compile(
        "PointwiseOpSpec",
        &pointwise::PointwiseOpSpec {
            name: "smoke_pointwise".to_string(),
            inputs: vec![DTypeId::F32],
            output: DTypeId::F32,
            expr: pointwise::PointwiseExpr::Arg(0),
            layout: pointwise::LayoutKind::Contiguous,
            work_group_size: 256,
        }
        .render_cuda(),
    );

    must_compile(
        "ReductionOpSpec",
        &reduction::ReductionOpSpec {
            name: "smoke_reduce".to_string(),
            dtype: DTypeId::F32,
            op: reduction::ReductionOpKind::Sum,
            layout: reduction::ReductionLayout::RowWise,
            reduction_size: Some(256),
            work_group_size: 256,
        }
        .render_cuda(),
    );

    must_compile(
        "VectorizedOpSpec",
        &vectorized::VectorizedOpSpec {
            name: "smoke_vectorized".to_string(),
            inputs: vec![DTypeId::F32],
            output: DTypeId::F32,
            expr: pointwise::PointwiseExpr::Arg(0),
            vector_width: vectorized::VectorWidth::Vec4,
            work_group_size: 256,
        }
        .render_cuda(),
    );
}
