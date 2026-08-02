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
    pub use super::arg::TensorArgs;
    pub use super::arg_into::{ArgInto, TensorArgsData};
    pub use super::auto_device::{BestDevice, BestDeviceAt};
    pub use super::backend::{Backend, StorageBackend, SupportsDType, TransferTo};
    pub use super::base::{Dyn, Tensor};
    #[cfg(feature = "distributed")]
    pub use super::base::PlacedTensorError;
    pub use super::device::{ConstDevice, Cpu, Device, DeviceId, DeviceKind};
    #[cfg(feature = "cuda")]
    pub use super::device::{Cuda, CudaN};
    #[cfg(feature = "wgpu")]
    pub use super::device::{Wgpu, WgpuN};
    #[cfg(feature = "metal")]
    pub use super::device::{Metal, MetalN};

    pub use super::dtype::{
        BoolDType, ConstDType, DType, DTypeId, FloatDType, IntDType, PlainDType, Q8_0, QuantDType,
        TensorElement,
    };
    pub use super::grad::{Grad, NoGrad, RequiresGrad};
    pub use super::matmul::MatMulShape;
    pub use super::tracing::{
        TracingBackend, extract_graph, tracing_mark_input, tracing_mark_output,
    };
}
