//! Expression-level kernel code generation (PRF-007).
//!
//! The body seam that ships today: an operation arrives as an [`ir::IrExpr`],
//! [`fragment::lower_scalar`] renders it against caller-named operands into a
//! [`fragment::ScalarFragment`], and the `kernel::scalar` templates place that
//! fragment where the arithmetic belongs — so a hand-written literal and an IR
//! expression travel the identical rendering, keying, tuning and launch path.
//! `catalog` declares the pointwise vocabulary as `IrExpr`s; its
//! `unary_fused_backward` differentiates them symbolically so one unary
//! backward becomes a single binary kernel instead of two launches and a
//! temporary.
//!
//! `dsl` and `jit` are the experimental custom-operation story: `dsl` builds a
//! [`ir::KernelDefinition`] from a closure over [`ir::IrExpr`] with symbolically
//! derived backward passes, and `jit` executes it — `CpuJitKernel` as a host
//! `f64` tree-walking reference (not a compiler), `CudaJitKernel` through the
//! production NVRTC dispatcher. See `docs/book/src/custom_operations.md`.
//!
//! `fuser` (crate-private) rebuilds a proven CMP-005 pointwise group through
//! this same seam for the preview `compiled` pipeline; its final wiring into
//! executable lowering is tracked with that workstream (#112), and its
//! legality, lowering and CPU-JIT numerical gates are exercised in-module.
//!
//! Whole-kernel emitters with no consumer under `crates/*/src/` were removed
//! per issue #111: every module here either executes or is gone.

pub mod catalog;
pub mod dsl;
pub mod fragment;
pub(crate) mod fuser;
pub mod ir;
pub mod jit;

pub use catalog::{binary_forward, unary_forward, unary_fused_backward};
pub use dsl::{define_binary_custom_op, define_unary_custom_op};
pub use fragment::{ScalarFragment, lower_scalar};
pub use ir::{
    IrBinaryOp, IrExpr, IrTernaryOp, IrUnaryOp, KernelDefinition, exp, fma, gelu, log, relu, rsqrt,
    sigmoid, silu, sqrt, tanh,
};
pub use jit::CpuJitKernel;
#[cfg(feature = "cuda")]
pub use jit::CudaJitKernel;
