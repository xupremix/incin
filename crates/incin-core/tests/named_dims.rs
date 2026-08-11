//! Semantic named-axis checks for the canonical structural shape engine.

extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_core::shapes::SwapAt;
use incin_core::shapes::reshape::{ElementCount, ReshapeShape};
use incin_core::test_utils::DummyBackend;
use incin_macros::{axis, s};
use typenum::Unsigned;

incin_core::dim!(Batch, Channels, Height, Width, Features);

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
{
}

#[test]
fn named_runtime_axes_are_stored_in_shape_buf() {
    type S = s![Batch, Channels, 224, 224];
    let field = S::from_dyn(&[2, 3, 224, 224]).unwrap();
    assert_eq!(field.as_ref(), &[2, 3, 224, 224]);
}

#[test]
fn named_selector_reaches_the_canonical_reduction_descriptor() {
    type B = DummyBackend<Cpu>;
    type S = s![Batch, Channels];
    let tensor: Tensor<S, B> = Tensor::ones((2usize, 3usize)).unwrap();

    let reduced = tensor.sum_named(Channels::selector()).unwrap();
    assert_eq!(reduced.shape_buf().as_ref(), &[2]);

    let kept = tensor.sum_keepdim_named(Channels::selector()).unwrap();
    assert_eq!(kept.shape_buf().as_ref(), &[2, 1]);
}

#[test]
fn named_tags_are_zero_sized_semantic_metadata() {
    assert_eq!(<Batch as AxisTag>::NAME, "Batch");
    assert_eq!(core::mem::size_of::<Batch>(), 0);
    assert_eq!(core::mem::size_of::<NamedDim<Batch, usize>>(), 0);
}

#[test]
fn named_tags_have_schema_local_identity_ids() {
    assert_eq!(<Batch as AxisIdentity>::Id::USIZE, 0);
    assert_eq!(<Channels as AxisIdentity>::Id::USIZE, 1);
    assert_eq!(<Height as AxisIdentity>::Id::USIZE, 2);
    assert_eq!(<Width as AxisIdentity>::Id::USIZE, 3);
    assert_eq!(
        core::any::TypeId::of::<<Batch as AxisIdentity>::Schema>(),
        core::any::TypeId::of::<<Channels as AxisIdentity>::Schema>()
    );
}

#[test]
fn named_lookup_resolves_current_position_without_storing_one() {
    type S = s![Batch, Channels, Height, Width];
    assert_eq!(Channels::selector().resolve::<S>().unwrap(), 1);
    type T = s![Width, Height, Batch, Channels];
    assert_eq!(Channels::selector().resolve::<T>().unwrap(), 3);
}

#[test]
fn named_axis_macro_expands_to_the_runtime_lookup_selector() {
    type S = s![Batch, Channels, Height, Width];
    let selector = axis!(named Channels);
    assert_eq!(selector.resolve::<S>().unwrap(), 1);
}

#[test]
fn named_lookup_rejects_missing_and_duplicate_names() {
    type S = s![Batch, Channels];
    assert!(matches!(
        NamedAxisSelector::<Width>::default().resolve::<S>(),
        Err(Error::Shape(ShapeError::MissingNamedAxis { name: "Width" }))
    ));
    type Duplicate = s![Channels, Channels];
    assert!(matches!(
        NamedAxisSelector::<Channels>::default().resolve::<Duplicate>(),
        Err(Error::Shape(ShapeError::AmbiguousNamedAxis {
            name: "Channels"
        }))
    ));
}

#[test]
fn transpose_preserves_the_complete_named_axis_type() {
    type S = s![Batch, Channels, Height, Width];
    type T = <S as SwapAt<Here, Next<Next<Next<Here>>>>>::Output;
    type Expected = s![Width, Channels, Height, Batch];
    assert_same::<T, Expected>();
}

#[test]
fn named_broadcast_output_preserves_the_semantic_axis() {
    type L = s![Channels];
    type R = s![Channels];
    type Out = <L as BroadcastShape<R>>::Output;
    type Expected = DimCons<
        NamedDim<Channels, BroadcastExtent<usize, usize>>,
        Nil,
    >;
    assert_same::<Out, Expected>();
}

#[test]
fn keepdim_rebinds_a_named_axis_to_static_one() {
    type S = s![Batch, Channels, 224, 224];
    type Out = <S as ReduceKeepAt<Next<Here>>>::Output;
    type Expected = DimCons<
        NamedDim<Batch, usize>,
        DimCons<
            NamedDim<Channels, typenum::U1>,
            DimCons<typenum::U224, DimCons<typenum::U224, Nil>>,
        >,
    >;
    assert_same::<Out, Expected>();
}

#[test]
fn named_static_extents_are_distinct_semantic_axes() {
    type S = s![Batch: 25, Features: 25];
    type Out = <S as SwapAt<Here, Next<Here>>>::Output;
    type Expected = s![Features: 25, Batch: 25];

    assert_same::<Out, Expected>();
    let dims = <Out as Shape>::from_dyn(&[25, 25]).unwrap();
    assert_eq!(dims.as_ref(), &[25, 25]);
}

#[test]
fn named_static_keepdim_rebinds_only_extent() {
    type S = s![Batch: 25, Channels: 64];
    type Out = <S as ReduceKeepAt<Next<Here>>>::Output;
    type Expected = s![Batch: 25, Channels: 1];

    assert_same::<Out, Expected>();
}

#[test]
fn named_runtime_keepdim_rebinds_runtime_extent_to_one() {
    type S = s![Batch: dyn, Channels: dyn];
    type Out = <S as ReduceKeepAt<Next<Here>>>::Output;
    type Expected = s![Batch: dyn, Channels: 1];

    assert_same::<Out, Expected>();
}

#[test]
fn concrete_static_named_extents_participate_in_element_count_arithmetic() {
    type Source = s![Batch: 2, Channels: 3];
    type Target = s![Height: 1, Width: 6];

    fn assert_reshape<S, T>()
    where
        S: Shape + ElementCount + ReshapeShape<T>,
        T: Shape + ElementCount,
    {
    }

    assert_reshape::<Source, Target>();
    assert_eq!(<Source as ElementCount>::Count::to_usize(), 6);
}
