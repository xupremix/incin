//! Unified cross-backend kernel emitter module (PRF-007).
//!
//! Provides single-source shader and kernel code generation for CUDA C++, WebGPU
//! Shading Language (WGSL), and Metal Shading Language (MSL).
//!
//! Live lowering path: capture → plan → `FusionPass` + `PointwiseChain`
//! (`fuser`) → `fragment::lower_scalar` → `kernel::scalar` templates.
//!
//! Modules under this tree that do not participate in that path were removed
//! per issue #111; `dsl` and `jit` remain as the deferred "decide separately"
//! surface, and `fusion`/`scheduler`/`pointwise`/`reduction`/`normalization`/
//! `vectorized`/`strided` are adoption candidates for the live fuser.

pub mod catalog;
pub mod dsl;
pub mod fragment;
pub(crate) mod fuser;
pub mod fusion;
pub mod ir;
pub mod jit;
pub mod normalization;
pub mod pointwise;
pub mod reduction;
pub mod scheduler;
pub mod strided;
pub mod vectorized;

pub use catalog::{binary_forward, unary_forward, unary_fused_backward};
pub use dsl::{define_binary_custom_op, define_ternary_custom_op, define_unary_custom_op};
pub use fragment::{ScalarFragment, lower_scalar};
pub use fusion::{CompositeFusionSpec, FusedNode};
pub use ir::{
    IrBinaryOp, IrExpr, IrTernaryOp, IrUnaryOp, KernelDefinition, exp, fma, gelu, log, relu, rsqrt,
    sigmoid, silu, sqrt, tanh,
};
pub use jit::CpuJitKernel;
#[cfg(feature = "cuda")]
pub use jit::CudaJitKernel;
pub use normalization::{NormKind, NormalizationSpec};
pub use pointwise::{
    BinaryOp, LayoutKind, PointwiseExpr, PointwiseOpSpec, TernaryOp, UnaryOp, render_cuda,
    render_msl, render_wgsl,
};
pub use reduction::{ReductionLayout, ReductionOpKind, ReductionOpSpec};
pub use scheduler::{BlockTensorPtr, KernelScheduler, LoopScheduleKind, MemorySpace};
pub use strided::{FastDivisor, StridedIndexSpec};
pub use vectorized::{VectorWidth, VectorizedOpSpec};
