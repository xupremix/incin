#![cfg(feature = "std")]
#![recursion_limit = "512"]

extern crate incin_core as incin;

use incin_core::axis;
use incin_core::exec::catalog::{AxisAttributes, Descriptor, LogicalTensorMeta, op};
use incin_core::exec::{AxisSet, ExecutionDescriptor, OperationIdentity, RankSupport};
use incin_core::prelude::*;

static TRACE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
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
    B: Same<A>,
{
}

#[test]
fn test_const_dim_and_extent_classification() {
    assert_eq!(ConstDim::<32>::static_size(), Ok(32));
    assert_eq!(<ConstDim<32> as Dim>::STATIC, StaticExtent::Value(32));

    assert!(usize::static_size().is_err());
    assert_eq!(<usize as Dim>::STATIC, StaticExtent::RuntimeUnknown);
}

#[test]
fn repeat_rejects_a_repeat_vector_with_the_wrong_rank() {
    type B = incin_core::test_utils::DummyBackend<Cpu>;
    let tensor: Tensor<s![2, 3], B> = Tensor::ones(()).unwrap();

    assert!(matches!(
        tensor.repeat(&[2]),
        Err(Error::Shape(ShapeError::RankMismatch {
            operation: OperationKind::Repeat,
            expected: RankExpectation::Exactly(2),
            actual: 1,
        }))
    ));
}

#[test]
fn test_derived_symbolic_dimensions() {
    let m = MulDim::<ConstDim<32>, ConstDim<4>>::default();
    assert_eq!(m.static_extent(), StaticExtent::Value(128));

    // Semantic equality vs ConstDim<128>
    assert_eq!(
        <MulDim<ConstDim<32>, ConstDim<4>> as Dim>::static_size(),
        ConstDim::<128>::static_size()
    );
    assert_eq!(m.static_extent(), <ConstDim<128> as Dim>::STATIC);

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
fn invalid_symbolic_dimensions_are_zst_specs_not_panic_values() {
    type InvalidProduct = MulDim<ConstDim<{ usize::MAX }>, ConstDim<2>>;
    type InvalidDifference = CheckedSubDim<ConstDim<3>, ConstDim<10>>;
    type InvalidDivision = ExactDivDim<ConstDim<10>, ConstDim<3>>;

    assert!(InvalidProduct::resolve_arg(usize::MAX).is_err());
    assert!(InvalidDifference::resolve_arg(0).is_err());
    assert!(InvalidDivision::resolve_arg(0).is_err());
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

    assert_same::<T16, S16>();
    assert_same::<R64, s![1; 63]>();

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
    assert_eq!(
        <NamedDim<ChannelsTag, ConstDim<64>> as Dim>::static_size(),
        Ok(64)
    );
    assert_eq!(named_static.static_extent(), StaticExtent::Value(64));

    // keepdim preserves ChannelsTag while replacing extent with ConstDim<1>
    let keepdim_named = NamedDim::<ChannelsTag, ConstDim<1>>::new();
    assert_eq!(
        <NamedDim<ChannelsTag, ConstDim<1>> as Dim>::static_size(),
        Ok(1)
    );
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

    assert_same::<Same, Base>();
    assert_same::<Reduced, Ranked<typenum::U63>>();
    assert_same::<Added, Ranked<typenum::U65>>();
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
    assert!(axes.contains(70));
    let descriptor = Descriptor::<op::SumDim>::infer_runtime(
        AxisAttributes { axis: 70 },
        vec![LogicalTensorMeta {
            shape: Some(input),
            dtype: None,
            device: None,
        }],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(descriptor.output_shape().unwrap().rank(), 70);
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

#[test]
fn exact_tracing_dispatch_unwraps_inner_storage_and_records_the_descriptor() {
    let _guard = TRACE_TEST_LOCK.lock().unwrap();
    type B = incin_core::prelude::TracingBackend<DummyBackend<Cpu>>;
    type S = s![2, 3];

    let _ = incin_core::prelude::extract_graph();
    let lhs: Tensor<S, B, f32> = Tensor::zeros(()).unwrap();
    let rhs: Tensor<S, B, f32> = Tensor::zeros(()).unwrap();
    let output = lhs.add(&rhs).unwrap();

    assert_eq!(output.dims(), [2, 3]);
    let graph = incin_core::prelude::extract_graph();
    assert!(graph.nodes.iter().any(|node| node.operation
        == OperationIdentity::Builtin(incin_core::prelude::OperationKind::Add)));
}

#[test]
fn exact_tracing_records_boolean_comparison_output_dtype() {
    let _guard = TRACE_TEST_LOCK.lock().unwrap();
    type B = incin_core::prelude::TracingBackend<DummyBackend<Cpu>>;
    type S = s![2, 3];

    let _ = incin_core::prelude::extract_graph();
    let lhs: Tensor<S, B, f32> = Tensor::zeros(()).unwrap();
    let rhs: Tensor<S, B, f32> = Tensor::zeros(()).unwrap();
    let _output = lhs.eq(&rhs).unwrap();

    let graph = incin_core::prelude::extract_graph();
    let node = graph
        .nodes
        .iter()
        .find(|node| {
            node.operation == OperationIdentity::Builtin(incin_core::prelude::OperationKind::CmpEq)
        })
        .expect("comparison node should be traced");
    let output = graph
        .values
        .get(&node.outputs[0])
        .expect("comparison output should be traced");
    assert_eq!(output.dtype.builtin_id(), Some(DTypeId::Bool));
}

#[test]
fn exact_tracing_records_canonical_unary_and_shape_descriptors() {
    let _guard = TRACE_TEST_LOCK.lock().unwrap();
    type B = incin_core::prelude::TracingBackend<DummyBackend<Cpu>>;
    type S = s![2, 3];

    let _ = incin_core::prelude::extract_graph();
    let input: Tensor<S, B, f32> = Tensor::zeros(()).unwrap();
    let relu = input.relu().unwrap();
    let _reshaped = relu.reshape::<s![3, 2]>(((), ((), ()))).unwrap();

    let graph = incin_core::prelude::extract_graph();
    assert!(graph.nodes.iter().any(|node| node.operation
        == OperationIdentity::Builtin(incin_core::prelude::OperationKind::Relu)));
    assert!(
        graph
            .nodes
            .iter()
            .find(|node| node.operation
                == OperationIdentity::Builtin(incin_core::prelude::OperationKind::Relu))
            .and_then(|node| node.descriptor_payload.as_ref())
            .is_some()
    );
    assert!(graph.nodes.iter().any(|node| node.operation
        == OperationIdentity::Builtin(incin_core::prelude::OperationKind::ReshapeExact)));
}

#[test]
fn typed_tracing_preserves_runtime_and_static_input_axes() {
    let _guard = TRACE_TEST_LOCK.lock().unwrap();
    type B = incin_core::prelude::TracingBackend<DummyBackend<Cpu>>;
    type S = s![usize, 768];

    let _ = incin_core::prelude::extract_graph();
    let input: Tensor<S, B, f32> = Tensor::zeros((7usize, ())).unwrap();
    incin_core::prelude::tracing_mark_input_typed::<S>(input.inner().value_id);

    let graph = incin_core::prelude::extract_graph();
    let value = graph.values.get(&input.inner().value_id).unwrap();
    assert!(matches!(
        value.shape_expr.dims[0],
        incin_core::exec::DimExpr::Symbol(_)
    ));
    assert_eq!(
        value.shape_expr.dims[1],
        incin_core::exec::DimExpr::Const(768)
    );
}
