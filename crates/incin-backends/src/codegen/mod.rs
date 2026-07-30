//! Unified cross-backend kernel emitter module (PRF-007).
//!
//! Provides single-source shader and kernel code generation for CUDA C++, WebGPU
//! Shading Language (WGSL), and Metal Shading Language (MSL).

pub mod pointwise;

pub use pointwise::{
    BinaryOp, LayoutKind, PointwiseExpr, PointwiseOpSpec, TernaryOp, UnaryOp, render_cuda,
    render_msl, render_wgsl,
};
