/// Connects `Tensor`'s type parameters to the runtime arguments needed to construct one.
pub mod arg;
/// Converts user-facing constructor arguments into each `TensorArgs` field.
pub mod arg_into;
/// Compile-time device selection from enabled features.
pub mod auto_device;
/// The `Backend` trait family and the test-only `DummyBackend` stand-in.
pub mod backend;
/// The `Tensor` type itself and its core inherent methods.
pub mod base;
/// 2D convolution output-shape and parameter validation shared across backends.
pub mod conv2d;
/// The `Device` trait family (`Cpu`, `Cuda<N>`, `Wgpu<N>`, `Dyn`) and `DeviceId`.
pub mod device;
/// The `DType` trait family (`f32`/`f16`/`bf16`/.../`Dyn`) and `DTypeId`.
pub mod dtype;
/// The `RequiresGrad` marker trait (`Grad`/`NoGrad`) controlling autodiff tracking.
pub mod grad;
/// Matrix multiplication shape validation shared across backends.
pub mod matmul;
/// Operator-trait implementations (`Add`, `Index`, etc.) for `Tensor`.
pub mod ops;
/// The ONNX-tracing backend wrapper used to record ops into a `Graph`.
pub mod tracing;

/// Re-exports the public tensor-layer API: `Tensor`, `Backend`, `Device`, `DType`, and their supporting traits.
pub mod prelude {
    pub use super::arg::*;
    pub use super::arg_into::*;
    pub use super::auto_device::*;
    pub use super::backend::*;
    pub use super::base::*;
    pub use super::conv2d::*;
    pub use super::device::*;
    pub use super::dtype::*;
    pub use super::grad::*;
    pub use super::matmul::*;
    pub use super::tracing::*;
}
