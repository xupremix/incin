pub(crate) mod conv;
pub(crate) mod elementwise;
pub(crate) mod embedding;
pub(crate) mod kernels;
pub(crate) mod loss;
pub(crate) mod matmul;
pub(crate) mod norm;
pub(crate) mod pool;
pub(crate) mod quant;
pub(crate) mod reduce;
pub(crate) mod shape;

use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeDescriptor;

/// Allocate a zeroed device buffer sized for `elements` values of `dtype`.
///
/// Before `EXE-008` these allocations multiplied the element count by a literal
/// `4` and unwrapped the driver result, so an `F64` or `I64` output — both of
/// which the CUDA capability registry accepts for storage and shape work — was
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
