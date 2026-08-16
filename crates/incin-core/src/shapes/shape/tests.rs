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
