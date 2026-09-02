//! A layout proof must be derived correctly and must gate what needs it.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::shapes::{Contiguous, Layout, LayoutOf, RowMajor, Shape, Unknown};
use incin_macros::s;

incin_core::dim!(Batch);

/// A tensor carrying a real layout must keep the API a plain one has.
///
/// This is the check the parameter's default would otherwise hide. An
/// `impl<S, B, K, G, P> Tensor<S, B, K, G, P>` binds `L` to its default, so it
/// silently stops applying to a tensor that has proven something -- the crate
/// still compiles and the loss is invisible until someone holds a proven
/// tensor and finds it has no methods.
///
/// Every accessor exercised here is one that used to be unreachable that way.
/// Extend this as further modules are converted; a method that stops compiling
/// here is a module that still pins `L`.
#[test]
fn a_proven_tensor_keeps_the_ordinary_api() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .expect("a freshly created tensor is dense row-major");

    assert_eq!(t.dims().as_ref(), &[3, 4]);
    assert_eq!(t.shape_buf().as_ref(), &[3, 4]);
    assert_eq!(t.dtype(), incin_core::prelude::DTypeId::F32.into());
    assert_eq!(t.numel(), 12);
    let _ = t.inner();
}

/// The same, for the operation surface rather than the accessors.
///
/// Kept separate because these are the conversions still outstanding: an
/// operation has to decide what layout its *output* carries, which is a
/// contract question rather than a rename, so the modules behind these calls
/// are converted deliberately and one at a time.
#[test]
fn a_proven_tensor_can_still_be_operated_on() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    let negated = t.neg().expect("a unary op applies to a proven tensor");
    assert_eq!(negated.dims().as_ref(), &[3, 4]);

    // Binary ops take a second operand with its own, independent layout: the
    // two need not agree, and neither is required to have proven anything.
    let unproven = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();
    let summed = negated
        .add_exact(&unproven)
        .expect("a proven operand and an unproven one combine");
    assert_eq!(summed.dims().as_ref(), &[3, 4]);
}

/// A pointwise op preserves the operand's layout rather than upgrading or
/// discarding it.
///
/// This is the propagation rule, and it is what keeps the parameter usable: a
/// proven tensor stays proven through a chain, so `reshape_view` is still
/// reachable at the end of one. Asserting `RowMajor` on every output instead
/// would be equally truthful -- the buffer is dense either way -- but it forces
/// every downstream signature that says `Tensor<S, B, K, G>` to be rewritten,
/// because `Unknown` and `RowMajor` are different types.
#[test]
fn a_proof_survives_a_pointwise_chain() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    // Still contiguous after two ops, so the `Contiguous` bound is satisfied.
    let flat = t
        .neg()
        .unwrap()
        .neg()
        .unwrap()
        .reshape_view::<s![12]>()
        .expect("contiguity survives a pointwise chain");
    assert_eq!(flat.dims().as_ref(), &[12]);
}

/// The `Dense` alias names the common case without repeating the shape.
///
/// `Tensor<S, B, K, G, P, RowMajor<S>>` mentions the shape twice, which is
/// noise: `RowMajor` is congruent with the shape it describes by construction.
#[test]
fn the_dense_alias_is_the_ergonomic_spelling() {
    // Aliased once locally, which is how a caller would actually write it and
    // what keeps `clippy::type_complexity` quiet: `s![3, 4]` still expands to a
    // typenum chain, so `Dense` shortens the spelling rather than the type.
    type Batch34 = incin_core::prelude::Dense<s![3, 4], CpuBackendImpl>;

    fn takes_dense(t: &Batch34) -> usize {
        t.numel()
    }

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();
    assert_eq!(takes_dense(&t), 12);
}

/// Row-major strides are the suffix products of the shape's extents.
#[test]
fn a_row_major_layout_derives_its_strides_from_the_shape() {
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)][..]
    );
    assert_eq!(
        <RowMajor<s![2, 3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(12), Some(4), Some(1)][..]
    );
    assert_eq!(<RowMajor<s![3, 4]> as Layout>::STATIC_OFFSET, Some(0));
}

/// Only a proven layout satisfies `Contiguous`.
///
/// This is the gate the whole parameter exists for: `reshape_view` is bounded
/// on it, so a tensor that has established nothing cannot reinterpret its
/// buffer. The negative case is a compile-fail fixture rather than an
/// assertion, because "does not implement" is not observable at runtime.
#[test]
fn only_a_proven_layout_satisfies_contiguous() {
    fn needs_contiguous<L: Contiguous>() {}
    needs_contiguous::<RowMajor<s![3, 4]>>();
    // needs_contiguous::<Unknown>() does not compile.

    // The two carry different evidence, so the distinction is real rather than
    // a marker that everything happens to satisfy.
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)][..]
    );
    assert_eq!(<Unknown as Layout>::STATIC_STRIDES, &[][..]);
    assert_eq!(<Unknown as Layout>::STATIC_OFFSET, None);
}

/// Congruence: a layout describes a shape of the same rank.
#[test]
fn congruence_relates_a_layout_to_its_shape() {
    fn describes<S: Shape, L: LayoutOf<S>>() {}

    describes::<s![3, 4], RowMajor<s![3, 4]>>();
    // `Unknown` describes anything, because it claims nothing about it.
    describes::<s![3, 4], Unknown>();
}

/// A dynamic axis voids the strides outside it and spares those inside.
///
/// The asymmetry that justifies reporting per axis: each stride is a product
/// of the extents inner to it, so a dynamic *outermost* axis voids nothing at
/// all, and a dynamic inner axis voids only what encloses it.
#[test]
fn a_dynamic_axis_voids_only_the_strides_that_enclose_it() {
    assert_eq!(
        <RowMajor<s![Batch, 3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(12), Some(4), Some(1)][..]
    );
    assert_eq!(
        <RowMajor<s![2, Batch, 4]> as Layout>::STATIC_STRIDES,
        &[None, Some(4), Some(1)][..]
    );
}
