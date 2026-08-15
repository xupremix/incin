// The implementation type and associated types must be public because they
// appear when the public `IncinBackend` alias is normalized. They are not
// re-exported from the public prelude.
pub(crate) mod backend;
pub(crate) mod capability;
pub(crate) mod executor;
pub(crate) mod gpu;
pub(crate) mod ops;
pub(crate) mod storage;
pub(crate) mod tape;

pub use backend::{CudaBackendImpl, CudaGrads, CudaVar};
/// Number of entries currently on this backend's autograd tape.
///
/// Re-exported since `GRD-002`: the row claims a `NoGrad` chain records
/// nothing, and its evidence test lives outside this crate. A guarantee
/// nothing outside can observe is not a guarantee.
pub use tape::depth as tape_depth;

pub(crate) fn checked_u32(
    value: usize,
    expression: &'static str,
) -> incin_core::error::Result<u32> {
    u32::try_from(value).map_err(|_| {
        incin_core::shapes::ShapeError::ArithmeticOverflow {
            operation: incin_core::shapes::error::OperationKind::Storage,
            expression,
        }
        .into()
    })
}

pub(crate) fn checked_i32(
    value: usize,
    expression: &'static str,
) -> incin_core::error::Result<i32> {
    i32::try_from(value).map_err(|_| {
        incin_core::shapes::ShapeError::ArithmeticOverflow {
            operation: incin_core::shapes::OperationKind::Storage,
            expression,
        }
        .into()
    })
}
