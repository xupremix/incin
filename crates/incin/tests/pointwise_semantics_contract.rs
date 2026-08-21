//! Pointwise Semantics Acceptance Contract Test.
//! Verifies standard pointwise numeric operations, real bool comparisons,
//! logical operations on bool tensors, target factories for bool, and exact bool extraction.

use incin::prelude::*;

#[test]
fn ordinary_numeric_pointwise_api() -> Result<()> {
    let a = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0, 3.0], vec![3])?;
    let b = Tensor::<Dyn, DefaultBackend>::from_slice(&[4.0_f32, 5.0, 6.0], vec![3])?;

    assert_eq!(a.try_add(&b)?.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
    assert_eq!(a.try_sub(&b)?.to_vec1::<f32>()?, vec![-3.0, -3.0, -3.0]);
    assert_eq!(a.try_mul(&b)?.to_vec1::<f32>()?, vec![4.0, 10.0, 18.0]);

    Ok(())
}

#[test]
fn arithmetic_operators_match_checked_methods() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([4.0_f32, 5.0, 6.0])?;

    let operator_sum = &a + &b;
    let checked_sum = a.try_add(&b)?;
    assert_eq!(
        operator_sum.to_vec1::<f32>()?,
        checked_sum.to_vec1::<f32>()?
    );

    let operator_product = a.clone() * &b;
    let checked_product = a.try_mul(&b)?;
    assert_eq!(
        operator_product.to_vec1::<f32>()?,
        checked_product.to_vec1::<f32>()?
    );

    let operator_neg = -&b;
    let checked_neg = b.try_neg()?;
    assert_eq!(
        operator_neg.to_vec1::<f32>()?,
        checked_neg.to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn checked_arithmetic_broadcasts_and_matches_the_operator() -> Result<()> {
    let lhs = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0], vec![2, 1])?;
    let rhs = Tensor::<Dyn, DefaultBackend>::from_slice(&[10.0_f32, 20.0, 30.0], vec![1, 3])?;

    let checked = lhs.try_add(&rhs)?;
    let operator = &lhs + &rhs;

    assert_eq!(checked.dims(), [2, 3]);
    assert_eq!(checked.to_vec1::<f32>()?, operator.to_vec1::<f32>()?);
    assert_eq!(
        checked.to_vec1::<f32>()?,
        vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
    Ok(())
}

#[test]
fn exact_arithmetic_keeps_exact_shape_contract() -> Result<()> {
    let lhs = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0], vec![2, 1])?;
    let rhs = Tensor::<Dyn, DefaultBackend>::from_slice(&[10.0_f32, 20.0, 30.0], vec![1, 3])?;

    assert!(lhs.add_exact(&rhs).is_err());
    Ok(())
}

#[test]
fn scalar_operators_match_checked_methods() -> Result<()> {
    let tensor = Cpu.tensor([2.0_f32, 4.0])?;

    assert_eq!((&tensor * 2.0_f32).to_vec1::<f32>()?, vec![4.0, 8.0]);
    assert_eq!((&tensor + 1.0_f32).to_vec1::<f32>()?, vec![3.0, 5.0]);
    assert_eq!((&tensor - 1.0_f32).to_vec1::<f32>()?, vec![1.0, 3.0]);
    assert_eq!((&tensor / 2.0_f32).to_vec1::<f32>()?, vec![1.0, 2.0]);
    assert_eq!(tensor.mul_scalar(2.0)?.to_vec1::<f32>()?, vec![4.0, 8.0]);
    Ok(())
}

#[test]
fn invalid_runtime_named_method_returns_an_error_and_operator_panics_boundedly() -> Result<()> {
    let a = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0], vec![2])?;
    let b = Tensor::<Dyn, DefaultBackend>::from_slice(&[3.0_f32, 4.0, 5.0], vec![3])?;
    assert!(a.try_add(&b).is_err());

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| &a + &b));
    let panic = outcome.expect_err("operator evaluation must panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("operator panic has a string message");
    assert_eq!(message, "incin tensor operator `+` failed");
    assert!(
        !message.contains("[1.0"),
        "panic must not contain tensor data"
    );
    Ok(())
}

#[test]
fn comparisons_return_real_bool() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([1.0_f32, 0.0, 4.0])?;

    let eq = a.eq(&b)?;
    let lt = a.lt(&b)?;

    assert_eq!(eq.dtype(), <bool as ConstDType>::DESCRIPTOR);
    assert_eq!(eq.to_vec1::<bool>()?, vec![true, false, false]);
    assert_eq!(lt.to_vec1::<bool>()?, vec![false, false, true]);

    Ok(())
}

#[test]
fn logical_ops_compose_comparison_masks() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([1.0_f32, 0.0, 4.0])?;

    let eq = a.eq(&b)?;
    let lt = a.lt(&b)?;

    let mask = eq.logical_or(&lt)?;
    assert_eq!(mask.to_vec1::<bool>()?, vec![true, false, true]);

    let inverted = mask.logical_not()?;
    assert_eq!(inverted.to_vec1::<bool>()?, vec![false, true, false]);

    Ok(())
}

#[test]
fn bool_target_factories_work() -> Result<()> {
    let target = Cpu.dtype::<bool>()?;

    let zeros = target.zeros([4])?;
    let ones = target.ones([4])?;

    assert_eq!(zeros.to_vec1::<bool>()?, vec![false, false, false, false]);
    assert_eq!(ones.to_vec1::<bool>()?, vec![true, true, true, true]);

    Ok(())
}

#[test]
fn bool_extraction_requires_bool_tensor() -> Result<()> {
    let numeric = Cpu.tensor([0_u8, 1_u8, 5_u8])?;

    assert!(numeric.to_vec1::<bool>().is_err());

    let bools = Cpu.dtype::<bool>()?.ones([3])?;

    assert_eq!(bools.to_vec1::<bool>()?, vec![true, true, true]);

    Ok(())
}
