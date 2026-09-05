//! Operands and typed execution shims, keyed by the declaration group.
//!
//! The harness drives runtime data: an `AdvertisedTuple` names an operation
//! as an `OperationKind`. Execution is typed: `dispatch::execute` is generic
//! over the `op::X` marker and over that marker's attribute type. Something has
//! to cross between the two, and this module is it.
//!
//! The crossing is keyed on operand contract, not on attribute type and not on
//! semantic profile. Those two look like the obvious keys and are both wrong.
//! Seventy-eight catalog rows declare `NoAttributes`, and that set contains
//! unary floats, binary broadcasts, comparisons that return `bool` whatever
//! they read, logical operations that are `bool` on both sides, and a quantized
//! matmul that reads `Q8_0` blocks. One key cannot serve all of those.
//!
//! [`contracts`] holds the vocabulary that keying is written in, [`families`]
//! holds the tables, and this file holds the lookup that walks them.
//!
//! An operation with no fixture is not a failure and not a pass. It is
//! [`Coverage::Unfixtured`] with a reason, counted, and held against a floor by
//! `crates/incin-backends/tests/conformance_oracle.rs` so the number can only
//! go up.

mod contracts;
mod families;

use incin_core::shapes::error::OperationKind;

use crate::conformance::shaped;

pub use contracts::Coverage;
pub(crate) use contracts::{Fixture, Operands, Role, Route, Subject};
// Visible to `shaped`, which holds the families whose attributes name extents
// and so has to build shims of its own. `macro_rules!` is textual and scoped to
// the rest of its own file without this.
pub(crate) use families::{
    constant_attribute_shim, derived_attribute_shim, family, on_axis_zero, typed_family,
    with_epsilon,
};

use families::*;

/// The fixture for `operation`, or the reason there is not one yet.
///
/// Order is arbitrary because the family lists are disjoint. Nothing enforces
/// that beyond their being written that way, which is worth knowing: naming an
/// operation twice would silently give whichever family is consulted first.
pub(crate) fn fixture(operation: OperationKind) -> Result<Fixture, &'static str> {
    unary_float(operation)
        .or_else(|| binary_elementwise(operation))
        .or_else(|| unary_logical(operation))
        .or_else(|| scalar_elementwise(operation))
        .or_else(|| reduce_all(operation))
        .or_else(|| reduce_axis(operation))
        .or_else(|| index_reduce_axis(operation))
        .or_else(|| readback(operation))
        .or_else(|| readback_scalar(operation))
        .or_else(|| clamping(operation))
        .or_else(|| interpolating(operation))
        .or_else(|| transposing(operation))
        .or_else(|| diagonal_shape(operation))
        .or_else(|| norm_reduce(operation))
        .or_else(|| variance_all(operation))
        .or_else(|| variance_axis(operation))
        .or_else(|| epsilon_unary(operation))
        .or_else(|| dropping(operation))
        .or_else(|| same_dtype_loss(operation))
        .or_else(|| selecting(operation))
        .or_else(|| masking(operation))
        .or_else(|| gathering(operation))
        .or_else(|| selecting_rows(operation))
        .or_else(|| embedding_lookup(operation))
        .or_else(|| matrix_product(operation))
        .or_else(|| vector_product(operation))
        .or_else(|| compressing(operation))
        .or_else(|| expanding(operation))
        .or_else(|| converting_dtype(operation))
        .or_else(|| unsqueezing(operation))
        .or_else(|| joining(operation))
        .or_else(|| reshaping(operation))
        .or_else(|| narrowing(operation))
        .or_else(|| flattening(operation))
        .or_else(|| slicing(operation))
        .or_else(|| padding(operation))
        .or_else(|| repeating(operation))
        .or_else(|| chunking(operation))
        .or_else(|| splitting(operation))
        .or_else(|| order_statistic(operation))
        .or_else(|| sorting(operation))
        .or_else(|| grouped_norm(operation))
        .or_else(|| shaped::creating(operation))
        .or_else(|| shaped::creating_from_host(operation))
        .or_else(|| shaped::creating_full(operation))
        .or_else(|| shaped::creating_arange(operation))
        .or_else(|| shaped::creating_linspace(operation))
        .or_else(|| shaped::squeezing(operation))
        .or_else(|| shaped::pooling_max(operation))
        .or_else(|| shaped::pooling_average(operation))
        .or_else(|| shaped::pooling_adaptive(operation))
        .or_else(|| shaped::sliding(operation))
        .or_else(|| shaped::shuffling(operation))
        .or_else(|| shaped::convolving_1d(operation))
        .or_else(|| shaped::convolving_2d(operation))
        .or_else(|| shaped::convolving_transposed(operation))
        .or_else(|| shaped::normalizing_layer(operation))
        .or_else(|| shaped::normalizing_rms(operation))
        .or_else(|| shaped::normalizing_batch(operation))
        .or_else(|| fused_product_sum(operation))
        .or_else(|| attending(operation))
        .or_else(|| projecting(operation))
        .or_else(|| class_loss(operation))
        .or_else(|| scattering(operation))
        .or_else(|| accumulating(operation))
        .or_else(|| encoding(operation))
        .ok_or_else(|| unfixtured_reason(operation))
}

/// Whether the tuple's dtype reaches any operand of `operation`'s fixture.
///
/// False for a fixture whose every operand has a fixed role, as `embedding`'s
/// index vector and float table both do. The row's dtype set for such an
/// operation is a union describing operands the fixture pins, so posing a
/// different dtype changes nothing about the invocation and the
/// unadvertised-dtype probe would report the row executing something it never
/// advertised when in fact the dtype was never used.
pub(crate) fn varies_with_tuple_dtype(operation: OperationKind) -> bool {
    fixture(operation).is_ok_and(|fixture| {
        fixture.roles.is_empty() || fixture.roles.iter().any(|role| role.carries_tuple_dtype())
    })
}

/// Why an operation has no fixture, in the terms a contributor closing the gap
/// would need.
fn unfixtured_reason(operation: OperationKind) -> &'static str {
    use OperationKind::*;
    match operation {
        QuantizedMatMul => {
            "the descriptor and the kernel disagree about which operand axis is             contracted: `OutputRule::MatMul` requires `lhs[-1] == rhs[-2]`,             while the CPU kernel reads the right operand as `[n, k]` because             thirty-two logical values share a scale only along the contiguous             axis. The two orientations coincide only on a square operand, and             posing a square one would report an agreement that does not exist"
        }
        coarse if !coarse.is_exact() => {
            "a coarse family row rather than an exact identity: it has no              descriptor and nothing to execute, and the exact rows beneath it              carry the coverage"
        }
        // Every exact identity the CPU registry advertises today reaches one
        // of the families above it, so this arm is unreachable in practice. It
        // stays because removing a fixture must produce a counted gap with a
        // reason rather than an empty string, which
        // `every_uncovered_operation_carries_a_reason` asserts.
        _ => "no fixture yet",
    }
}
