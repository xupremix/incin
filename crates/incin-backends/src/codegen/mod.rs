//! Unified cross-backend kernel emitter module (PRF-007).
//!
//! Provides single-source shader and kernel code generation for CUDA C++, WebGPU
//! Shading Language (WGSL), and Metal Shading Language (MSL):
//! - Pointwise expressions with automatic AST emission
//! - Shape-informed 128-bit vectorized kernels (`float4`, `__half2`, `vec4<f32>`)
//! - Fused GEMM / Linear epilogues (`BiasAdd`, `ResidualAdd`, `Gelu`, `Relu`, `Silu`)

pub mod attention;
pub mod autotune_config;
pub mod cross_entropy;
pub mod dsl;
pub mod fused_epilogue;
pub mod gemm;
pub mod ir;
pub mod jit;
pub mod mma;
pub mod normalization;
pub mod optim;
pub mod pointwise;
pub mod quant_gemm;
pub mod reduction;
pub mod rope;
pub mod scan;
pub mod scheduler;
pub mod strided;
pub mod vectorized;

pub use attention::AttentionSpec;
pub use autotune_config::{AutotuneCandidate, AutotuneSpace, GpuArchProfile};
pub use cross_entropy::CrossEntropySpec;
pub use dsl::{define_binary_custom_op, define_ternary_custom_op, define_unary_custom_op};
pub use fused_epilogue::{FusedEpilogueKind, FusedEpilogueSpec};
pub use gemm::{GemmSpec, GemmTileConfig};
pub use ir::{
    IrBinaryOp, IrExpr, IrTernaryOp, IrUnaryOp, KernelDefinition, exp, fma, gelu, log, relu, rsqrt,
    sigmoid, silu, sqrt, tanh,
};
pub use jit::CpuJitKernel;
#[cfg(feature = "cuda")]
pub use jit::CudaJitKernel;
pub use mma::{TensorCoreMmaLayout, TensorCoreMmaSpec};
pub use normalization::{NormKind, NormalizationSpec};
pub use optim::{FusedOptimizerSpec, OptimizerKind};
pub use pointwise::{
    BinaryOp, LayoutKind, PointwiseExpr, PointwiseOpSpec, TernaryOp, UnaryOp, render_cuda,
    render_msl, render_wgsl,
};
pub use quant_gemm::QuantGemmSpec;
pub use reduction::{ReductionLayout, ReductionOpKind, ReductionOpSpec};
pub use rope::RopeSpec;
pub use scan::{PrefixScanSpec, ScanOpKind};
pub use scheduler::{BlockTensorPtr, KernelScheduler, LoopScheduleKind, MemorySpace};
pub use strided::{FastDivisor, StridedIndexSpec};
pub use vectorized::{VectorWidth, VectorizedOpSpec};
