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

use incin_backends::codegen::{
    AttentionSpec, CompositeFusionSpec, CrossEntropySpec, FusedEpilogueKind, FusedEpilogueSpec,
    GemmSpec, MoeGatingSpec, TensorCoreMmaSpec,
};
use incin_backends::codegen::{conv, normalization, optim, pointwise, quant_gemm};
use incin_backends::codegen::{reduction, rope, scan, sota_gemm, vectorized};
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
        "AttentionSpec",
        &AttentionSpec::new("smoke_attention", DTypeId::F32, 64, true).render_cuda(),
    );
    must_compile(
        "ImplicitConv2dSpec",
        &conv::ImplicitConv2dSpec::new("smoke_conv", DTypeId::F32, (3, 3), (1, 1), (1, 1))
            .render_cuda(),
    );

    let cross_entropy = CrossEntropySpec::new("smoke_xent", DTypeId::F32, 128, 0.0);
    must_compile(
        "CrossEntropySpec::forward",
        &cross_entropy.render_cuda_forward(),
    );
    must_compile(
        "CrossEntropySpec::backward",
        &cross_entropy.render_cuda_backward(),
    );

    must_compile(
        "FusedEpilogueSpec",
        &FusedEpilogueSpec::new("smoke_epilogue", DTypeId::F32, FusedEpilogueKind::BiasRelu)
            .render_cuda(),
    );
    must_compile(
        "CompositeFusionSpec",
        &CompositeFusionSpec::swiglu_residual("smoke_fusion", DTypeId::F32).render_cuda(),
    );
    must_compile(
        "GemmSpec",
        &GemmSpec::new("smoke_gemm", DTypeId::F32).render_cuda(),
    );
    must_compile(
        "TensorCoreMmaSpec",
        &TensorCoreMmaSpec::new("smoke_mma", DTypeId::F16, DTypeId::F32, 2, 2).render_cuda(),
    );
    must_compile(
        "MoeGatingSpec",
        &MoeGatingSpec::new("smoke_moe", DTypeId::F32, 8, 2).render_cuda_gating(),
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

    let rope = rope::RopeSpec::new("smoke_rope", DTypeId::F32, 64, 10000.0, false);
    must_compile("RopeSpec::forward", &rope.render_cuda_forward());
    must_compile("RopeSpec::backward", &rope.render_cuda_backward());

    must_compile(
        "PrefixScanSpec",
        &scan::PrefixScanSpec::new("smoke_scan", scan::ScanOpKind::Sum, DTypeId::F32, true)
            .render_cuda(),
    );

    must_compile(
        "FusedOptimizerSpec",
        &optim::FusedOptimizerSpec {
            name: "smoke_optim".to_string(),
            kind: optim::OptimizerKind::AdamW,
            dtype: DTypeId::F32,
            grad_dtype: DTypeId::F32,
            lr: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
        .render_cuda(),
    );

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
        "QuantGemmSpec",
        &quant_gemm::QuantGemmSpec {
            name: "smoke_quant".to_string(),
            act_dtype: DTypeId::F32,
            weight_dtype: DTypeId::U8,
            block_size: 32,
        }
        .render_cuda_gemv(),
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

    must_compile(
        "SotaGemmSpec",
        &sota_gemm::SotaGemmSpec {
            name: "smoke_sota".to_string(),
            engine: sota_gemm::GemmComputeEngine::SimdFmaTiled,
            dtype_a: DTypeId::F32,
            dtype_b: DTypeId::F32,
            dtype_c: DTypeId::F32,
            dtype_accum: DTypeId::F32,
            block_m: 64,
            block_n: 64,
            block_k: 16,
            thread_m: 4,
            thread_n: 4,
            pipeline_stages: 2,
            use_cp_async: false,
            use_l2_swizzle: false,
            swizzle_factor: 1,
            has_bias: false,
            has_residual: false,
            activation: sota_gemm::EpilogueActivation::None,
        }
        .render_cuda(),
    );
}
