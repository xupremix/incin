/// Checked byte lengths for tensor storage allocation.
pub mod allocation;
/// Connects `Tensor`'s type parameters to the runtime arguments needed to construct one.
pub mod arg;
/// Converts user-facing constructor arguments into each `TensorArgs` field.
pub mod arg_into;
/// Compile-time device selection from enabled features.
pub mod auto_device;
/// The `Backend` trait family.
pub(crate) mod backend;
/// The `Tensor` type itself and its core inherent methods.
pub mod base;
/// 2D convolution output-shape and parameter validation shared across backends.
pub mod conv2d;
/// The `Device` trait family (`Cpu`, `Cuda<N>`, `Wgpu<N>`, `Dyn`) and `DeviceId`.
pub mod device;
/// PyTorch-style rendering of a tensor's values, shared by every backend's
/// `HostInterop::host_format_display`/`host_format_debug` default bodies.
pub(crate) mod display;
/// The `DType` trait family (`f32`/`f16`/`bf16`/.../`Dyn`) and `DTypeId`.
pub mod dtype;
/// The `RequiresGrad` marker trait (`Grad`/`NoGrad`) controlling autodiff tracking.
pub mod grad;
/// Backend-owned gradient handles produced by backward passes.
/// Matrix multiplication shape validation shared across backends.
pub mod matmul;
/// Operator-trait implementations (`Add`, `Index`, etc.) for `Tensor`.
pub mod ops;
/// Runtime reduction semantics shared by tensor and neural-network operations.
pub mod reduction;
/// The ONNX-tracing backend wrapper used to record ops into a `Graph`.
pub mod tracing;
/// Ownership-preserving transfer contracts for tensors and module state.
pub mod transfer;

/// Re-exports the public tensor-layer API: `Tensor`, `Backend`, `Device`, `DType`, and their supporting traits.
pub mod prelude {
    pub use super::allocation::{CheckedByteLen, checked_byte_len_from_dims};
    pub use super::arg::TensorArgs;
    pub use super::arg_into::{ArgInto, TensorArgsData};
    pub use super::auto_device::{BestDevice, BestDeviceAt};
    pub use super::backend::{
        AutogradBackend, Backend, HostInterop, StorageBackend, StorageTransfer, SupportsDType,
        TransferBackend, TransferTo, VariableBackend,
    };
    #[cfg(feature = "distributed")]
    pub use super::base::PlacedTensorError;
    pub use super::base::Tensor;
    pub use super::device::{
        ConstDevice, Cpu, Device, DeviceId, DeviceKind, DevicePreference, DeviceSet, DeviceSetError,
    };
    #[cfg(feature = "cuda")]
    pub use super::device::{Cuda, CudaN};
    #[cfg(feature = "metal")]
    pub use super::device::{Metal, MetalN};
    #[cfg(feature = "wgpu")]
    pub use super::device::{Wgpu, WgpuN};
    pub use crate::shapes::Dyn;

    pub use super::dtype::{
        BoolDType, BuiltinDType, ConstDType, DType, DTypeDescriptor, DTypeId, DTypeKey, DTypeKind,
        FloatDType, IntDType, PlainDType, Q8_0, QuantDType, StorageEncoding, TensorElement,
    };
    pub use super::grad::{Grad, GradJoin, JoinedGrad, NoGrad, RequiresGrad};
    pub use super::matmul::MatMulShape;
    pub use super::reduction::Reduction;
    pub use super::tracing::{
        TracingBackend, extract_graph, tracing_mark_input, tracing_mark_input_typed,
        tracing_mark_output,
    };
    pub use super::transfer::ToDevice;
    pub use crate::autograd::Gradients;
}
