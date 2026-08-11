#![cfg(feature = "std")]
#![recursion_limit = "512"]

extern crate incin_core as incin;

use incin_core::axis;
use incin_core::exec::{AxisSet, RankSupport, ReduceOp, ReductionSpec};
use incin_core::prelude::*;
use incin_core::shapes::dim::{
    AddDim, CheckedSubDim, ConstDim, Dim, ExactDivDim, MulDim, NamedDim, StaticExtent,
};
use incin_core::shapes::idx::{Here, Next};
use incin_core::shapes::shape::{
    AddOneRank, DimCons, FlattenAt, Nil, PreserveRank, Ranked, RemoveOneRank, SwapAt,
};

incin_core::dim!(BatchTag, ChannelsTag, HeightTag, WidthTag);

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
{
}

#[test]
fn test_const_dim_and_extent_classification() {
    let d32 = ConstDim::<32>;
    assert_eq!(d32.size(), 32);
    assert_eq!(d32.static_extent(), StaticExtent::Value(32));

    let dyn_dim = 16usize;
    assert_eq!(dyn_dim.size(), 16);
    assert_eq!(dyn_dim.static_extent(), StaticExtent::RuntimeUnknown);
}

#[test]
fn test_derived_symbolic_dimensions() {
    let m = MulDim::<ConstDim<32>, ConstDim<4>>::default();
    assert_eq!(m.static_extent(), StaticExtent::Value(128));

    // Semantic equality vs ConstDim<128>
    assert_eq!(m.size(), ConstDim::<128>.size());
    assert_eq!(m.static_extent(), ConstDim::<128>.static_extent());

    let a = AddDim::<ConstDim<10>, ConstDim<5>>::default();
    assert_eq!(a.static_extent(), StaticExtent::Value(15));

    let s = CheckedSubDim::<ConstDim<10>, ConstDim<3>>::default();
    assert_eq!(s.static_extent(), StaticExtent::Value(7));

    let s_under = CheckedSubDim::<ConstDim<3>, ConstDim<10>>::default();
    assert_eq!(s_under.static_extent(), StaticExtent::Invalid);

    let d = ExactDivDim::<ConstDim<12>, ConstDim<4>>::default();
    assert_eq!(d.static_extent(), StaticExtent::Value(3));

    let d_inv = ExactDivDim::<ConstDim<10>, ConstDim<3>>::default();
    assert_eq!(d_inv.static_extent(), StaticExtent::Invalid);
}

#[test]
fn invalid_static_arithmetic_is_not_reported_as_static_knowledge() {
    type Overflow = MulDim<ConstDim<{ usize::MAX }>, ConstDim<2>>;
    assert_eq!(<Overflow as Dim>::STATIC, StaticExtent::Invalid);
    const { assert!(!<Overflow as Dim>::STATIC_SIZE) };
}

#[test]
fn structural_operations_compile_at_rank_16_64_and_200() {
    use incin_core::shapes::idx::{FromEnd, Here};
    use incin_core::shapes::{BroadcastShape, ReduceAt, SwapAxes};

    type S16 = s![1; 16];
    type S64 = s![1; 64];
    type S200 = s![1; 200];

    type T16 = <S16 as SwapAxes<Here, Next<Here>>>::Output;
    type R64 = <S64 as ReduceAt<FromEnd<Here>>>::Output;
    type B200 = <S200 as BroadcastShape<S200>>::Output;

    assert_eq!(<T16 as Shape>::RANK, Some(16));
    assert_eq!(<R64 as Shape>::RANK, Some(63));
    assert_eq!(<B200 as Shape>::RANK, Some(200));

    // Exercise the runtime side of the structural rule as well as checking
    // the resulting type.  These calls intentionally use ShapeBuf, the sole
    // runtime shape representation.
    let s64 = <S64 as Shape>::try_from_dims(&[1; 64]).unwrap();
    let _r64 = <R64 as Shape>::try_from_dims(&[1; 63]).unwrap();
    let _t16 = <T16 as Shape>::try_from_dims(&[1; 16]).unwrap();
    let s200 = <S200 as Shape>::try_from_dims(&[1; 200]).unwrap();
    let b200 = <S200 as BroadcastShape<S200>>::output_shape(&s200, &s200).unwrap();
    let b64 = <S64 as BroadcastShape<S64>>::output_shape(&s64, &s64).unwrap();
    assert_eq!(b200.len(), 200);
    assert_eq!(b64.len(), 64);
}

#[test]
fn test_named_dim_tag_extent_orthogonality() {
    let named_runtime = NamedDim::<ChannelsTag, usize>::new();
    assert_eq!(
        <NamedDim<ChannelsTag, usize> as Dim>::resolve_arg(64),
        Ok(64)
    );
    assert_eq!(core::mem::size_of_val(&named_runtime), 0);
    assert_eq!(named_runtime.static_extent(), StaticExtent::RuntimeUnknown);

    let named_static = NamedDim::<ChannelsTag, ConstDim<64>>::new();
    assert_eq!(named_static.size(), 64);
    assert_eq!(named_static.static_extent(), StaticExtent::Value(64));

    // keepdim preserves ChannelsTag while replacing extent with ConstDim<1>
    let keepdim_named = NamedDim::<ChannelsTag, ConstDim<1>>::new();
    assert_eq!(keepdim_named.size(), 1);
    assert_eq!(keepdim_named.static_extent(), StaticExtent::Value(1));
}

#[test]
fn test_generic_array_shapes_unlimited_rank() {
    let r0 = [];
    assert_eq!(ShapeBuf::from_slice(&r0).as_ref(), &[] as &[usize]);

    let r1 = [16];
    assert_eq!(ShapeBuf::from_slice(&r1).as_ref(), &[16]);

    let r8 = [1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(
        ShapeBuf::from_slice(&r8).as_ref(),
        &[1, 2, 3, 4, 5, 6, 7, 8]
    );

    let r16 = [1usize; 16];
    assert_eq!(ShapeBuf::from_slice(&r16).len(), 16);

    let r32 = [2usize; 32];
    assert_eq!(ShapeBuf::from_slice(&r32).len(), 32);

    let r64 = [3usize; 64];
    assert_eq!(ShapeBuf::from_slice(&r64).len(), 64);

    let r200 = [4usize; 200];
    assert_eq!(ShapeBuf::from_slice(&r200).len(), 200);
}

#[test]
fn test_recursive_fixed_rank_dim_cons() {
    type TestCons = DimCons<ConstDim<32>, DimCons<usize, Nil>>;

    let field = TestCons::resolve(((), (16, ()))).unwrap();
    assert_eq!(field.as_ref(), &[32, 16]);

    let resolved = TestCons::try_from_dims(&[32, 16]).unwrap();
    assert_eq!(resolved.as_ref(), &[32, 16]);

    assert!(TestCons::try_from_dims(&[31, 16]).is_err());
}

#[test]
fn generic_typenum_ranked_shape_preserves_rank_without_a_rank_ladder() {
    type S = Ranked<typenum::U64>;
    let field = <S as Shape>::try_from_dims(&[7; 64]).unwrap();
    assert_eq!(<S as Shape>::RANK, Some(64));
    assert_eq!(<S as DynShape>::rank(&field), 64);
}

#[test]
fn generic_typenum_rank_arithmetic_is_structural() {
    type Base = Ranked<typenum::U64>;
    type Same = <Base as PreserveRank>::Output;
    type Reduced = <Base as RemoveOneRank>::Output;
    type Added = <Base as AddOneRank>::Output;

    assert_eq!(<Same as Shape>::RANK, Some(64));
    assert_eq!(<Reduced as Shape>::RANK, Some(63));
    assert_eq!(<Added as Shape>::RANK, Some(65));
}

#[test]
fn structural_swap_moves_complete_dimension_metadata_at_rank_three() {
    use incin_core::shapes::idx::{Here, Next};

    type Source = DimCons<typenum::U1, DimCons<typenum::U2, DimCons<typenum::U3, Nil>>>;
    type Swapped = <Source as SwapAt<Next<Here>, Next<Next<Here>>>>::Output;
    type Expected = DimCons<typenum::U1, DimCons<typenum::U3, DimCons<typenum::U2, Nil>>>;

    assert_same::<Swapped, Expected>();

    let field = <Swapped as Shape>::try_from_dims(&[1, 3, 2]).unwrap();
    assert_eq!(field.as_ref(), &[1, 3, 2]);
}

#[test]
fn structural_flatten_collapses_an_arbitrary_cursor_range() {
    use incin_core::shapes::idx::{Here, Next};

    type Source =
        DimCons<typenum::U2, DimCons<typenum::U3, DimCons<typenum::U4, DimCons<typenum::U5, Nil>>>>;
    type Flat = <Source as FlattenAt<Next<Here>, Next<Next<Here>>>>::Output;
    type Expected = DimCons<
        typenum::U2,
        DimCons<
            incin_core::shapes::dim::MulDim<typenum::U3, typenum::U4>,
            DimCons<typenum::U5, Nil>,
        >,
    >;

    assert_same::<Flat, Expected>();
}

#[test]
fn test_axis_selector_macro_and_cursors() {
    let sel = axis!(0, 2, -1);
    assert_eq!(sel.raw_axes, vec![0, 2, -1]);

    let normalized = sel.normalize(4).unwrap();
    assert_eq!(normalized, vec![0, 2, 3]);

    let sel = axis!(1, -2);
    assert_eq!(sel.normalize(4).unwrap(), vec![1, 2]);

    // Out of range check
    assert!(axis!(5).normalize(4).is_err());

    // Duplicate check
    assert!(axis!(2, -2).normalize(4).is_err());

    // The most-negative signed selector must be rejected as data, not panic
    // while taking its magnitude.
    assert!(axis!(isize::MIN).normalize(4).is_err());

    // Structural cursors compile check
    let _here = Here;
}

#[test]
fn test_static_cursor_types() {
    use incin_core::shapes::idx::{FromEnd, Here, Next};

    let here: Here = Here;
    assert_eq!(ToAxisIndex::to_axis_index(&here), 0);

    let next1: Next<Here> = Next(core::marker::PhantomData);
    assert_eq!(ToAxisIndex::to_axis_index(&next1), 1);

    let end1: FromEnd<Here> = FromEnd(core::marker::PhantomData);
    assert_eq!(ToAxisIndex::to_axis_index(&end1), -1);
}

#[test]
fn test_descriptor_axis_mask_and_axis_set_above_31() {
    let set = AxisSet::EMPTY.insert(32).insert(45);
    assert!(set.contains(32));
    assert!(set.contains(45));
    assert!(!set.contains(0));
    assert_eq!(set.count(), 2);

    let set = AxisSet::EMPTY.insert(32).insert(70);
    assert!(set.contains(32));
    assert!(set.contains(70));
    assert_eq!(set.count(), 2);
}

#[test]
fn descriptor_reduction_accepts_axis_seventy() {
    let input = ShapeBuf::from_slice(&[1; 71]);
    let axes = AxisSet::EMPTY.insert(70);
    let spec = ReductionSpec::new(&input, axes, false, ReduceOp::Sum).unwrap();
    assert_eq!(spec.output.rank(), 70);
}

#[test]
fn test_backend_rank_support() {
    let any = RankSupport::Any;
    let upto = RankSupport::UpTo(8);
    let range = RankSupport::Range { min: 2, max: 4 };

    assert_eq!(any, RankSupport::Any);
    assert_eq!(upto, RankSupport::UpTo(8));
    assert_eq!(range, RankSupport::Range { min: 2, max: 4 });
}

use incin_core::shapes::idx::ToAxisIndex;
use incin_core::test_utils::DummyBackend;

#[test]
fn test_derived_dimension_argument_validation_is_fallible() {
    let valid = MulDim::<ConstDim<32>, ConstDim<4>>::resolve_arg(128).unwrap();
    assert_eq!(valid, 128);
    assert!(MulDim::<ConstDim<32>, ConstDim<4>>::resolve_arg(999).is_err());
}

#[test]
fn test_static_axis_macro_type_preservation() {
    let a0 = axis!(0);
    assert_eq!(ToAxisIndex::to_axis_index(&a0), 0);

    let a1 = axis!(1);
    assert_eq!(ToAxisIndex::to_axis_index(&a1), 1);

    let a_end = axis!(-1);
    assert_eq!(ToAxisIndex::to_axis_index(&a_end), -1);
}

#[test]
fn test_high_rank_tensor_creation() {
    let t16 = Tensor::<Ranked<typenum::consts::U16>, DummyBackend<Cpu>, f32>::zeros(
        incin_core::shapes::ShapeBuf::from_slice(&[1usize; 16]),
    )
    .unwrap();
    assert_eq!(t16.dims().len(), 16);

    let t32 = Tensor::<Ranked<typenum::consts::U32>, DummyBackend<Cpu>, f32>::zeros(
        incin_core::shapes::ShapeBuf::from_slice(&[1usize; 32]),
    )
    .unwrap();
    assert_eq!(t32.dims().len(), 32);

    let t64 = Tensor::<Ranked<typenum::consts::U64>, DummyBackend<Cpu>, f32>::zeros(
        incin_core::shapes::ShapeBuf::from_slice(&[1usize; 64]),
    )
    .unwrap();
    assert_eq!(t64.dims().len(), 64);
}
