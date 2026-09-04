//! Worked examples of the target API's layout surface.
//!
//! These are tests rather than prose so the examples in the book and the design
//! note cannot drift from what compiles. Each one is a case from
//! `docs/plan/research/0.2.0/layout-at-construction.md`.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::prelude::*;
use incin_core::shapes::dim::ConstDim;
use incin_core::shapes::{
    ChannelsLast, Dense, DimCons, FreshLayout, Nil, RowMajor, ShapeArgs, dense_strides,
};
use incin_core::tensor::device::Cpu;
use incin_macros::s;

/// The array constructors infer a `ConstDim`-based shape, which is a different
/// spelling from the `s![..]` typenum one. Named once here rather than repeated.
type Arr2x2 = DimCons<ConstDim<2>, DimCons<ConstDim<2>, Nil>>;

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
    let x: Dense<Arr2x2, CpuBackendImpl> = Cpu
        .tensor_in::<_, RowMajor<Arr2x2>>([[1.0f32, 2.0], [3.0, 4.0]])
        .unwrap();
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

    let refused =
        Cpu.zeros_in::<ShapeArgs<Nchw>, ChannelsLast<Nchw>>(ShapeArgs::new(Default::default()));
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
        <RowMajor<Nchw> as FreshLayout<Nchw>>::strides(&dims).as_ref(),
        dense_strides(&dims).as_ref(),
        "row-major asks for the dense strides, so it is allocatable"
    );
    assert_ne!(
        <ChannelsLast<Nchw> as FreshLayout<Nchw>>::strides(&dims).as_ref(),
        dense_strides(&dims).as_ref(),
        "channels-last does not, which is exactly why case 3 fails"
    );
}

/// 5. `zeros_in` is the same idea where the backend allocates rather than the
///    host uploading, and its shape spelling composes with `reshape_view`.
#[test]
fn zeros_in_yields_a_usable_proof() {
    let x: Dense2x3 = Cpu
        .zeros_in::<ShapeArgs<s![2, 3]>, RowMajor<s![2, 3]>>(ShapeArgs::new(Default::default()))
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
        .zeros_in::<ShapeArgs<s![2, 3]>, RowMajor<s![2, 3]>>(ShapeArgs::new(Default::default()))
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
    assert!(t.clone().into_layout::<RowMajor<Nchw>>().is_ok());

    // The same buffer is not channels-last, and saying so is refused.
    let refused = t.into_layout::<ChannelsLast<Nchw>>();
    assert!(
        refused.is_err(),
        "into_layout compares strides; a dense buffer is not channels-last"
    );
}
