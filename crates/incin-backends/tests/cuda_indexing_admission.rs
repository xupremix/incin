//! Issue #88 capability admission probe: the CUDA indexing rows must admit
//! the *integer index operand* of `gather`/`scatter`/`index_select`, not just
//! the `f32` data operand.
//!
//! `dispatch::execute`'s `admit_invocation` checks every input handle against
//! the one resolved capability row in turn, and these three operations carry
//! an integer index as operand 1 (operand 2 for `scatter`). A row that lists
//! only `f32` makes the operation unreachable: the index operand fails dtype
//! admission before any kernel launches. This is the exact bug class
//! `F32_AND_BOOL` and `INDEX_AND_F32_DTYPES` document, and it is why
//! `capabilities.md` can advertise a CUDA row that real invocations never
//! reach.
//!
//! These tests need no hardware: `capability::support` answers from the
//! compiled table alone.
#![cfg(feature = "cuda")]

use incin_backends::capability::{CUDA_CAPABILITIES, support};
use incin_core::exec::{CapabilityQuery, OperationIdentity, SupportLevel};
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

/// Finds the single CUDA rule for `operation`, or explains its absence.
fn cuda_rule(operation: K) -> &'static incin_core::exec::CapabilityRule {
    CUDA_CAPABILITIES
        .iter()
        .find(|rule| rule.operation == operation)
        .unwrap_or_else(|| panic!("CUDA declares no capability row for {operation:?}"))
}

/// The data and index operand dtypes a real invocation of `operation` carries,
/// mirroring the descriptor's own contract (`catalog/inference.rs` requires
/// operand 1 to be integer for these three).
fn operand_dtypes(operation: K) -> &'static [DTypeId] {
    match operation {
        // data=f32, index=integer
        K::Gather | K::IndexSelect => &[DTypeId::F32, DTypeId::I64],
        // target=f32, index=integer, source=f32
        K::Scatter => &[DTypeId::F32, DTypeId::I64, DTypeId::F32],
        _ => panic!("{operation:?} is not an indexing op under test"),
    }
}

fn admits(operation: K, dtype: DTypeId) -> bool {
    let rule = cuda_rule(operation);
    let query = CapabilityQuery {
        operation: OperationIdentity::Builtin(operation),
        dtype: dtype.descriptor(),
        layout: rule.layouts[0],
        rank: rule.min_rank.max(1),
        training: rule.training,
        math_mode: rule.math_modes[0],
    };
    !matches!(
        support(DeviceKind::Cuda, &query),
        SupportLevel::Unsupported(_)
    )
}

/// Every operand a real `gather`/`scatter`/`index_select` invocation carries
/// must clear CUDA dtype admission, or the operation is unreachable through
/// `dispatch::execute` however good its kernel is.
#[test]
fn cuda_indexing_rows_admit_every_operand_dtype_a_real_invocation_carries() {
    for operation in [K::Gather, K::Scatter, K::IndexSelect] {
        for dtype in operand_dtypes(operation) {
            assert!(
                admits(operation, *dtype),
                "CUDA: {operation:?} refuses a {dtype:?} operand, so a real \
                 invocation fails admission on whichever operand this row does \
                 not list -- the kernel is unreachable. Row declares {:?}",
                cuda_rule(operation).dtypes,
            );
        }
    }
}

/// The index operand specifically: this is the one `F32_ONLY` misses.
#[test]
fn cuda_indexing_rows_admit_the_integer_index_operand() {
    for operation in [K::Gather, K::Scatter, K::IndexSelect] {
        for dtype in [DTypeId::U8, DTypeId::U32, DTypeId::I64] {
            assert!(
                admits(operation, dtype),
                "CUDA: {operation:?} refuses a {dtype:?} index operand",
            );
        }
    }
}
