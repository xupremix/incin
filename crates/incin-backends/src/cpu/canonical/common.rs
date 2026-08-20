//! Common helper functions and operand binders for CPU canonical executors.

use incin_core::backend_authoring::StorageBackend;
use incin_core::error::BackendError;
use incin_core::exec::catalog::Descriptor;
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, MathMode, SupportLevel, TensorHandle,
};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::{Device, DeviceKind};

use crate::cpu::CpuBackendImpl;
use crate::cpu::capability::CPU_NAME;
use crate::cpu::storage::CpuStorage;
use crate::descriptor_bind::invalid;

/// Recover CPU storage from a checked handle.
pub(crate) fn operand<'a>(
    handle: &'a TensorHandle<'a>,
    operation: OperationKind,
) -> Result<&'a CpuStorage, BackendError> {
    let storage = handle
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| invalid(operation, "operand is not CPU storage"))?;
    let metadata = storage.metadata();
    if metadata.device().kind() != DeviceKind::Cpu {
        return Err(invalid(operation, "operand is not on a CPU device"));
    }
    Ok(storage)
}

/// Whether the request is in module training mode.
pub(crate) fn training_mode<B: StorageBackend>(context: &ExecutionContext<B>) -> bool {
    context.training()
}

/// Re-check the exact capability row from inside the executor.
pub(crate) fn admitted<D: Device>(
    backend: &CpuBackendImpl<D>,
    operation: OperationKind,
    storage: &CpuStorage,
    training: bool,
) -> Result<(), BackendError> {
    let metadata = storage.metadata();
    let query = CapabilityQuery {
        operation: incin_core::exec::OperationIdentity::Builtin(operation),
        dtype: metadata.dtype(),
        layout: metadata.layout(),
        rank: metadata.shape().rank(),
        training,
        math_mode: MathMode::Precise,
    };
    match backend.support(&query) {
        SupportLevel::Unsupported(reason) => Err(BackendError::unsupported(CPU_NAME, reason)),
        _ => Ok(()),
    }
}

/// Bind the single operand a reduction consumes.
pub(crate) fn reduction_operand<'a, D: Device>(
    backend: &CpuBackendImpl<D>,
    inputs: &'a [TensorHandle<'a>],
    operation: OperationKind,
    training: bool,
) -> Result<&'a CpuStorage, BackendError> {
    let [input] = inputs else {
        return Err(invalid(
            operation,
            "a reduction expects exactly one operand",
        ));
    };
    let input = operand(input, operation)?;
    admitted(backend, operation, input, training)?;
    Ok(input)
}

/// The output shape the descriptor already resolved, if it has one.
pub(crate) fn resolved_output_shape<O: incin_core::exec::CanonicalOperation>(
    operation: &incin_core::exec::Validated<Descriptor<O>>,
) -> Option<&[usize]> {
    let [single] = operation.descriptor().outputs() else {
        return None;
    };
    single.shape.as_deref()
}
