pub(crate) mod cast;
pub(crate) mod compare;
pub(crate) mod conv;
// The vendor path is what the `cuda-vendor` feature exists to gate (issue
// #85): without the feature the module is absent, so no cuBLASLt symbol can
// be reached from a build that did not ask for vendor libraries.
#[cfg(feature = "cuda-vendor")]
pub(crate) mod cublaslt;
pub(crate) mod dropout;
pub(crate) mod elementwise;
#[cfg(test)]
mod ir_conformance_tests;
pub(crate) mod kernels;
pub(crate) mod logical;
pub(crate) mod matmul;
pub(crate) mod norm;
pub(crate) mod optimizer;
pub(crate) mod pool;
pub(crate) mod quant;
pub(crate) mod reduce;
pub(crate) mod select;
pub(crate) mod shape;
#[cfg(all(test, feature = "std"))]
mod view_cost_bench;

use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeDescriptor;

use crate::cuda::storage::CudaStorage;

/// Host-side metadata the pure fit policies decide on, borrowed from a
/// storage without touching the device. Shared between the native batched
/// plan (`matmul::batched_gemm_plan`) and, behind `cuda-vendor`, the
/// cuBLASLt policies in `cublaslt`, so both read the same fields through
/// one constructor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OperandMeta<'a> {
    pub(crate) dtype: DTypeDescriptor,
    pub(crate) device_id: usize,
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) offset: usize,
}

impl<'a> OperandMeta<'a> {
    pub(crate) fn of(storage: &'a CudaStorage) -> Self {
        Self {
            dtype: storage.buffer.dtype,
            device_id: storage.buffer.device_id,
            shape: storage.shape.dims(),
            strides: storage.strides.strides(),
            offset: storage.offset_elements(),
        }
    }
}

/// Allocate a zeroed device buffer sized for `elements` values of `dtype`.
///
/// Before `EXE-008` these allocations multiplied the element count by a literal
/// `4` and unwrapped the driver result, so an `F64` or `I64` output - both of
/// which the CUDA capability registry accepts for storage and shape work - was
/// given half the bytes its own recorded dtype requires, and an allocation
/// failure aborted the process. The dtype now decides the width, the
/// multiplication is checked, and the driver's failure is reported.
pub(crate) fn alloc_zeroed_bytes(
    stream: &Arc<cudarc::driver::CudaStream>,
    dtype: DTypeDescriptor,
    elements: usize,
    operation: OperationKind,
) -> Result<cudarc::driver::CudaSlice<u8>> {
    let byte_len = crate::bytes::byte_len(dtype, elements, operation)?;
    stream.alloc_zeros::<u8>(byte_len).map_err(|error| {
        Error::Msg(format!(
            "CUDA {operation} allocation of {byte_len} bytes failed: {error:?}"
        ))
    })
}
