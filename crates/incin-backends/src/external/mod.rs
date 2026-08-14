//! Third-party backend integrations.
//!
//! These adapters delegate execution to external tensor ecosystems and are
//! intentionally separate from Incin native CPU, CUDA, and WGPU backends.

pub use incin_core::backend_authoring::{
    Alignment, AttributeContract, AutogradBackend, Backend, CanonicalOperation, Capabilities, CapabilityQuery,
    CapabilityRegistry, Descriptor, DescriptorError, Execute, ExecuteOutput, ExecutionContext,
    ExecutionDescriptor, ExecutionRequest, LogicalTensorMeta, Operation, OperationCatalogEntry,
    OperationIdentity, OperationKey, ShapeBuf, StorageBackend, StorageOutput, SupportLevel,
    SupportsDType, TensorBackend, TensorMeta, TransferTo, UnsupportedReason, Validated, VariableBackend, execute,
    execute_shaped, execute_shaped_with_payload, execute_with_payload,
};
pub use incin_core::prelude::{
    BackendError, DType, DTypeDescriptor, DTypeId, Device, DeviceId, Error, OperationKind, Result,
};

/// The conformance suite an external backend runs against itself (`EXE-010`).
///
/// Not gated on any particular integration: it is the authoring surface
/// PROPOSALS.md sec. 2.9 describes, and an author writing a backend for some
/// ecosystem this repository has never heard of should not have to enable the
/// Candle adapter to test it.
#[cfg(feature = "std")]
pub mod conformance;

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------

/// Wraps the `candle_core` crate, providing `CandleBackend` as a `Backend`
/// implementation backed by Candle's own tensor type.
#[cfg(feature = "external-candle")]
pub mod candle;
