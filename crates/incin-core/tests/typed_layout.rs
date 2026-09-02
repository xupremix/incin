//! A layout proof must be derived correctly and must gate what needs it.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_core::shapes::{Contiguous, Layout, LayoutOf, RowMajor, Shape, Unknown};
use incin_macros::s;

incin_core::dim!(Batch);

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
