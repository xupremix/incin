//! Worked examples of the target API's layout surface.
//!
//! These are tests rather than prose so the examples in the book and the design
//! note cannot drift from what compiles. Each one is a case from
//! `docs/plan/research/0.2.0/layout-at-construction.md`.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::prelude::*;
use incin_core::shapes::{ChannelsLast, Dense, FreshLayout, RowMajor, ShapeArgs, dense_strides};
use incin_core::tensor::device::Cpu;
use incin_macros::s;
use std::vec::Vec;

/// The array constructors and `s![..]` now agree, so this is just `s![2, 2]`.
/// It was a separate `ConstDim` spelling until #116.
type Arr2x2 = s![2, 2];

/// The static 2x3 tensor cases share this, which also keeps clippy's
/// `type_complexity` quiet about repeating it.
type Dense2x3 = Dense<s![2, 3], CpuBackendImpl>;

/// 1. The plain constructor claims nothing, exactly as before.
#[test]
fn a_plain_constructor_claims_nothing() {
    let x = Cpu.tensor([[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(x.dims().as_ref(), &[2, 2]);
    // Layout slot takes its default: this annotation is what proves it.
    let _: incin_core::prelude::Tensor<Arr2x2, CpuBackendImpl, f32> = x;
}

/// 2. `tensor_in` names the layout, and the result carries it.
#[test]
fn tensor_in_carries_the_layout_it_was_asked_for() {
    // No turbofish: `L` appears only in the return type, so the annotation
    // that names the proof is what chooses the layout.
    let x: Dense<Arr2x2, CpuBackendImpl> = Cpu.tensor_in([[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(x.dims().as_ref(), &[2, 2]);
    assert_eq!(x.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// 3. A layout no backend can allocate is refused, not silently satisfied.
#[test]
fn an_unallocatable_layout_is_refused_at_construction() {
    // Rank four, because channels-last is defined against NCHW. Built through
    // `zeros_in` rather than `tensor_in` only because `TensorData` covers
    // rank one and two arrays.
    type Nchw = s![1, 2, 2, 2];
    let refused = Cpu.zeros_in::<ChannelsLast, ShapeArgs<Nchw>>(ShapeArgs::new(Default::default()));
    assert!(
        refused.is_err(),
        "no backend uploads host bytes in channels-last order yet, so the \
         request must be refused rather than satisfied with a dense buffer \
         wearing a channels-last type"
    );
    let message = alloc_string(&refused.unwrap_err());
    assert!(
        message.contains("layout") && message.contains("unsupported"),
        "the refusal must name the layout as the reason, got: {message}"
    );
}

fn alloc_string(e: &incin_core::error::Error) -> String {
    format!("{e}")
}

/// 4. The refusal is decided by the strides the layout asks for, not by a list
///    of known layouts -- which is why a layout added later needs no change.
#[test]
fn the_refusal_is_decided_by_strides() {
    let dims = [1usize, 2, 2, 2];
    type Nchw = s![1, 2, 2, 2];

    assert_eq!(
        <RowMajor as FreshLayout<Nchw>>::strides(&dims).as_ref(),
        dense_strides(&dims).as_ref(),
        "row-major asks for the dense strides, so it is allocatable"
    );
    assert_ne!(
        <ChannelsLast as FreshLayout<Nchw>>::strides(&dims).as_ref(),
        dense_strides(&dims).as_ref(),
        "channels-last does not, which is exactly why case 3 fails"
    );
}

/// 5. `zeros_in` is the same idea where the backend allocates rather than the
///    host uploading, and its shape spelling composes with `reshape_view`.
#[test]
fn zeros_in_yields_a_usable_proof() {
    let x: Dense2x3 = Cpu
        .zeros_in::<RowMajor, _>(ShapeArgs::new(Default::default()))
        .unwrap();

    // The proof is usable with no runtime stride scan.
    let flat = x.reshape_view::<s![6]>().unwrap();
    assert_eq!(flat.dims().as_ref(), &[6]);
    assert_eq!(flat.to_vec1::<f32>().unwrap(), vec![0.0; 6]);
}

/// 6. A proof from the target API composes with the operations that state one.
#[test]
fn a_target_proof_composes_with_operations() {
    let x: Dense2x3 = Cpu
        .zeros_in::<RowMajor, _>(ShapeArgs::new(Default::default()))
        .unwrap();

    // Pointwise states `RowMajor` of its own result rather than carrying the
    // operand's, so the chain keeps a proof without ever propagating one.
    let flat = x.relu().unwrap().reshape_view::<s![6]>().unwrap();
    assert_eq!(flat.dims().as_ref(), &[6]);
}

/// 7. `into_layout` is the checked promotion, and it refuses a false claim.
#[test]
fn into_layout_checks_rather_than_assumes() {
    type Nchw = s![1, 2, 2, 2];
    let t: incin_core::prelude::Tensor<Nchw, CpuBackendImpl> =
        incin_core::prelude::Tensor::zeros(()).unwrap();

    // A fresh allocation really is dense, so the claim is granted.
    assert!(t.clone().into_layout::<RowMajor>().is_ok());

    // The same buffer is not channels-last, and saying so is refused.
    let refused = t.into_layout::<ChannelsLast>();
    assert!(
        refused.is_err(),
        "into_layout compares strides; a dense buffer is not channels-last"
    );
}

/// The array constructors produce the same shape type `s![..]` does.
///
/// They used to build from `ConstDim<N>`, which is a different type and does
/// not implement `ConcreteStaticExtent` -- so nothing `Cpu.tensor(..)` returned
/// could reach `ElementCount`, and `reshape`/`reshape_view` were unavailable on
/// the most ergonomic constructor in the crate. Issue #116.
///
/// Both halves are asserted, because either alone would pass for the wrong
/// reason: the annotation pins the *type*, and the reshape pins that the type
/// is one the shape arithmetic can actually use.
#[test]
fn an_array_constructor_produces_a_reshapable_shape() {
    type Plain2x2 = incin_core::prelude::Tensor<s![2, 2], CpuBackendImpl, f32>;
    let x: Plain2x2 = Cpu.tensor([[1.0f32, 2.0], [3.0, 4.0]]).unwrap();

    let flat = x.into_row_major().unwrap().reshape_view::<s![4]>().unwrap();
    assert_eq!(flat.dims().as_ref(), &[4]);
    assert_eq!(flat.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Rank-three and rank-four literals reach the target API, in the right order.
///
/// `TensorData` stopped at rank two, so an NCHW literal could not be written
/// at all -- the earlier version of this file had to build its rank-four case
/// through `zeros_in` for that reason. Issue #116.
///
/// The assertion that matters is the *order*, not the rank. A flatten that
/// walked the nesting wrongly would still produce a buffer of the right length
/// and the right shape, and only the values would be wrong -- the same trap the
/// strided-GEMM test was written to avoid.
#[test]
fn rank_three_and_four_literals_flatten_row_major() {
    // Distinct values, so a wrong walk produces wrong numbers rather than a
    // plausible buffer.
    let three = Cpu
        .tensor([[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        .unwrap();
    assert_eq!(three.dims().as_ref(), &[2, 2, 2]);
    assert_eq!(
        three.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "the outer array indexes the slowest axis, so declaration order is row-major"
    );

    // NCHW: one image, two channels, 2x2 spatial. Channel 0 is 1..4, channel 1
    // is 5..8, which is what makes a channels-last permutation visible.
    let nchw = Cpu
        .tensor([[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        .unwrap();
    assert_eq!(nchw.dims().as_ref(), &[1, 2, 2, 2]);
    assert_eq!(
        nchw.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    // And the shape is the `s![..]` spelling, so it composes with the shape
    // arithmetic rather than being a second, dead-end vocabulary.
    type PlainNchw = incin_core::prelude::Tensor<s![1, 2, 2, 2], CpuBackendImpl, f32>;
    let _: PlainNchw = nchw;
}

/// A channels-last tensor can now be built, and holds the right numbers.
///
/// This is the case the design note flagged as most likely to be got wrong.
/// Host data arrives in NCHW order and the buffer has to be in NHWC order, so
/// a permutation that walks the nesting wrongly produces a buffer of exactly
/// the right length, the right shape and the right strides -- holding the
/// wrong values. Nothing structural catches it, so this asserts the values.
///
/// The input is one image, two channels, 2x2 spatial. Channel 0 is 1..4 and
/// channel 1 is 5..8, chosen so an NHWC interleave is unmistakable: read
/// physically, the buffer must alternate channels rather than run 1,2,3,4.
#[test]
fn a_channels_last_tensor_is_built_with_its_elements_interleaved() {
    use incin_core::backend_authoring::StorageBackend;

    let x = Cpu
        .tensor_in::<ChannelsLast, _>([[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        .unwrap();

    // The type claims channels-last, and the metadata agrees: stride[C] == 1.
    let meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(x.inner());
    assert_eq!(meta.shape().as_ref(), &[1, 2, 2, 2]);
    assert_eq!(
        meta.strides().as_ref(),
        &[8, 1, 4, 2],
        "N=C*H*W, C=1, H=C*W, W=C -- channels varies fastest"
    );

    // Reading it back recovers the input exactly, and that is the whole proof.
    //
    // Both accessors read *by the strides*, so neither exposes raw memory --
    // `to_bytes` is canonical on purpose, or a saved tensor could not be
    // reloaded. The round-trip is still discriminating, because the strides
    // and the permutation have to agree for it to hold: a row-major walk of
    // [1, 2, 2, 2] visits offsets 0, 2, 4, 6, 1, 3, 5, 7 under strides
    // [8, 1, 4, 2], so a buffer that was *not* permuted would read back as
    // [1, 3, 5, 7, 2, 4, 6, 8]. Getting the input back is only possible if the
    // upload scattered the values to exactly the offsets the layout names.
    assert_eq!(
        x.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "the layout changed and the values did not"
    );

    // The negative control, so the assertion above is not merely plausible.
    // Same numbers, same strides, no permutation -- which is what a broken
    // upload would produce, and it reads back differently.
    let unpermuted =
        <CpuBackendImpl as incin_core::backend_authoring::HostInterop>::from_bytes_strided::<f32>(
            bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            &[1, 2, 2, 2],
            &[8, 1, 4, 2],
            <f32 as incin_core::tensor::dtype::BuiltinDType>::DTYPE.descriptor(),
            &incin_core::tensor::device::DeviceId::cpu(),
        )
        .unwrap();
    let wrong =
        <CpuBackendImpl as incin_core::backend_authoring::HostReadback>::float_to_vec1::<f32>(
            &unpermuted,
        )
        .unwrap();
    assert_eq!(
        wrong,
        vec![1.0f64, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
        "an unpermuted upload under channels-last strides reads back scrambled, \
         which is exactly what the assertion above rules out"
    );

    // And the proof is not forgeable in the other direction: the same numbers
    // uploaded densely are not channels-last, and saying so is refused.
    let dense = Cpu
        .tensor([[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        .unwrap();
    assert!(
        dense.into_layout::<ChannelsLast>().is_err(),
        "into_layout compares strides; a dense buffer is not channels-last"
    );
}

/// The scatter map is a permutation, and the dense case is the identity.
#[test]
fn scatter_positions_is_a_permutation() {
    use incin_core::shapes::{dense_strides, scatter_positions};

    let dims = [1usize, 2, 2, 2];
    let dense = scatter_positions(&dims, dense_strides(&dims).as_ref()).unwrap();
    assert_eq!(
        dense,
        (0..8).collect::<Vec<_>>(),
        "dense order is the identity"
    );

    let cl = scatter_positions(&dims, &[8, 1, 4, 2]).unwrap();
    assert_eq!(cl, vec![0, 2, 4, 6, 1, 3, 5, 7]);
    let mut sorted = cl.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..8).collect::<Vec<_>>(),
        "every slot filled exactly once"
    );

    // A stride of zero on an extent-1 axis is *not* overlapping: that index is
    // always zero, so the offsets are unchanged. Worth pinning, because it is
    // the case a naive "no zero strides" check would reject wrongly.
    assert_eq!(scatter_positions(&dims, &[0, 1, 4, 2]).unwrap(), cl);

    // Genuinely overlapping strides cannot be filled from a dense source, and
    // refusing beats writing part of the buffer.
    assert!(
        scatter_positions(&[2, 2], &[1, 1]).is_none(),
        "offsets 0,1,1,2 collide and overrun"
    );
    assert!(
        scatter_positions(&[2, 2], &[1]).is_none(),
        "rank disagreement"
    );
}

/// Naming the layout never costs a placeholder; naming the data type never
/// needs one.
///
/// `tensor_in` has two type parameters and Rust's turbofish is all-or-nothing,
/// so whichever comes second is the one you have to hold a place for. The
/// order is not arbitrary: the two parameters differ in whether they can be
/// inferred at all.
///
/// `D` is the argument's own type, so it is always fixable *at the argument* --
/// bind the value, or suffix the literal. `L` appears only in the return type,
/// so nothing about the call determines it and it is the one that sometimes
/// has to be said. Putting `L` first means the parameter that occasionally
/// needs naming is the one you can name alone.
#[test]
fn the_layout_comes_first_because_the_data_type_is_always_inferable() {
    // The layout, from the annotation. No turbofish.
    let a: Dense<Arr2x2, CpuBackendImpl> = Cpu.tensor_in([[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(a.dims().as_ref(), &[2, 2]);

    // Pinning the data type instead: bind it. This is the "reverse problem"
    // that reordering might have created, and it does not arise -- the
    // argument's type is the parameter, so annotating the binding settles it.
    let data: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let b: Dense<Arr2x2, CpuBackendImpl, f64> = Cpu.tensor_in(data).unwrap();
    assert_eq!(b.to_vec1::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    // Or suffix the literal, which does the same job inline.
    let c: Dense<Arr2x2, CpuBackendImpl, f64> = Cpu.tensor_in([[1.0f64, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(c.dims().as_ref(), &[2, 2]);

    // The explicit form still works, and this is where the placeholder lands:
    // trailing, on the parameter that never needed naming.
    let d = Cpu
        .tensor_in::<RowMajor, _>([[1.0f32, 2.0], [3.0, 4.0]])
        .unwrap();
    assert_eq!(d.dims().as_ref(), &[2, 2]);
}
