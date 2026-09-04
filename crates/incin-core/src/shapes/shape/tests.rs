use super::*;
use crate::resource::ResourceLimits;
use crate::shapes::error::{OperationKind, ShapeError};

#[test]
fn test_scalar_shape() {
    let scalar = ShapeBuf::scalar();
    assert_eq!(<Nil as DynShape>::rank(&scalar), 0);
    assert_eq!(<Nil as DynShape>::numel(&scalar), 1);
    let empty_dims: [usize; 0] = [];
    assert_eq!(scalar, empty_dims);
    assert_eq!(<Nil as DynShape>::rank(&scalar), 0);
}

#[test]
fn test_dyn_shape() {
    let d = ShapeBuf::from_slice(&[2, 3, 4]);
    assert_eq!(<Dyn as DynShape>::rank(&d), 3);
    assert_eq!(<Dyn as DynShape>::numel(&d), 24);
    let dims = d;
    assert_eq!(dims.as_ref(), &[2, 3, 4]);
}

#[test]
fn test_array_shape() {
    let shape: [usize; 3] = [2, 3, 4];
    let field = ShapeBuf::from_slice(&shape);
    assert_eq!(<Ranked<typenum::U3> as DynShape>::rank(&field), 3);
    assert_eq!(<Ranked<typenum::U3> as DynShape>::numel(&field), 24);
    assert_eq!(field.dims(), &[2, 3, 4]);
    assert_eq!(<Ranked<typenum::U3> as Shape>::RANK, Some(3));
}

#[test]
fn dyn_is_zero_sized() {
    assert_eq!(core::mem::size_of::<Dyn>(), 0);
    let marker = Dyn::marker();
    assert_eq!(core::mem::size_of_val(&marker), 0);
}

#[test]
fn checked_allocation_lengths_cover_scalar_zero_limit_and_overflow_edges() {
    let mut limits = ResourceLimits::trusted_local_large_model();
    limits.max_rank = 8;
    #[allow(unused_assignments)]
    {
        limits.max_dimension = u64::MAX;
    }
    limits.max_tensor_bytes = u64::MAX;

    assert_eq!(
        CheckedNumel::from_dims(OperationKind::Storage, &[], &limits)
            .unwrap()
            .get(),
        1
    );
    assert_eq!(
        CheckedNumel::from_dims(
            OperationKind::Storage,
            &[usize::MAX, 0, usize::MAX],
            &limits,
        )
        .unwrap()
        .get(),
        0
    );
    assert!(matches!(
        CheckedNumel::from_dims(OperationKind::Storage, &[usize::MAX, 2], &limits),
        Err(ShapeError::ArithmeticOverflow { .. })
    ));

    limits.max_rank = 1;
    assert!(matches!(
        CheckedNumel::from_dims(OperationKind::Storage, &[2, 3], &limits),
        Err(ShapeError::RankMismatch { .. })
    ));
    limits.max_rank = 8;
    limits.max_dimension = 2;
    assert!(matches!(
        CheckedNumel::from_dims(OperationKind::Storage, &[3], &limits),
        Err(ShapeError::InvalidParameter {
            parameter: "dimension",
            value: 3,
            ..
        })
    ));
}

use crate::exec::ShapeEvidence;
use crate::shapes::{DimCons, Dyn, ExpectedShapes, Nil, ProofLevel, ShapeBuf, ShapeValue};
use typenum::{U2, U3, U4};

type Static23 = DimCons<U2, DimCons<U3, Nil>>;
type Mixed2dyn = DimCons<U2, DimCons<usize, Nil>>;
type Static4 = DimCons<U4, Nil>;

fn static_2x3() -> ShapeValue<Static23> {
    ShapeValue::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap()
}

fn mixed_2xdyn() -> ShapeValue<Mixed2dyn> {
    ShapeValue::try_new(ShapeBuf::from_slice(&[2, 5])).unwrap()
}

fn dynamic_1() -> ShapeValue<Dyn> {
    ShapeValue::try_new(ShapeBuf::from_slice(&[7])).unwrap()
}

// A single expectation lowers exactly as the old path did: the proof the
// shape type carries and the evidence `of::<S>` builds, over Static, Mixed
// and Dynamic alike. The typed single-output comparison must observe nothing
// new, which is what makes the alias below behavior-preserving.
#[test]
fn expected_shapes_single_preserves_proof_and_evidence() {
    let static_value = static_2x3();
    assert_eq!(<ShapeValue<Static23> as ExpectedShapes>::ARITY, 1);
    assert_eq!(static_value.combined_proof(), ProofLevel::Static);
    assert_eq!(
        static_value.combined_evidence(),
        ShapeEvidence::of::<Static23>()
    );
    assert_eq!(
        static_value.shape_bufs().count(),
        1,
        "arity and iteration must agree"
    );

    let mixed_value = mixed_2xdyn();
    assert_eq!(mixed_value.combined_proof(), ProofLevel::Mixed);
    assert_eq!(
        mixed_value.combined_evidence(),
        ShapeEvidence::of::<Mixed2dyn>()
    );

    let dynamic_value = dynamic_1();
    assert_eq!(dynamic_value.combined_proof(), ProofLevel::Dynamic);
    assert_eq!(
        dynamic_value.combined_evidence(),
        ShapeEvidence::of::<Dyn>()
    );
}

// Several outputs can only promise what every one of them proves, and no
// single output's static geometry may stand for the rest: the combined value
// carries the weakest proof with empty statics, the `Dynamic` posture.
#[test]
fn expected_shapes_tuple_takes_weakest_proof_and_no_statics() {
    let all_static = (static_2x3(), static_2x3(), static_2x3());
    assert_eq!(all_static.combined_proof(), ProofLevel::Static);
    assert_eq!(all_static.combined_evidence().static_rank(), None);
    assert_eq!(all_static.combined_evidence().static_numel(), None);
    assert!(
        all_static.combined_evidence().static_extents().is_empty(),
        "one output's extents must not stand for all three"
    );

    let mixed_pair = (static_2x3(), mixed_2xdyn());
    assert_eq!(mixed_pair.combined_proof(), ProofLevel::Mixed);

    let with_dynamic = (static_2x3(), mixed_2xdyn(), dynamic_1());
    assert_eq!(with_dynamic.combined_proof(), ProofLevel::Dynamic);
    assert_eq!(
        with_dynamic.combined_evidence().proof(),
        ProofLevel::Dynamic
    );
}

// Buffers come out borrowed and in output order: the comparison zips these
// against inference positionally, so order is load-bearing.
#[test]
fn expected_shapes_tuple_buffers_are_borrowed_in_order() {
    let pair = (
        ShapeValue::<Static23>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap(),
        ShapeValue::<Static4>::try_new(ShapeBuf::from_slice(&[4])).unwrap(),
    );
    assert_eq!(
        <(ShapeValue<Static23>, ShapeValue<Static4>) as ExpectedShapes>::ARITY,
        2
    );
    let bufs: Vec<&ShapeBuf> = pair.shape_bufs().collect();
    assert_eq!(bufs.len(), 2);
    assert_eq!(bufs[0].as_ref(), &[2, 3]);
    assert_eq!(bufs[1].as_ref(), &[4]);
}
