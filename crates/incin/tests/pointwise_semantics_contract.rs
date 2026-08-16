//! Pointwise Semantics Acceptance Contract Test.
//! Verifies standard pointwise numeric operations, real bool comparisons,
//! logical operations on bool tensors, target factories for bool, and exact bool extraction.

use incin::prelude::*;

#[test]
fn ordinary_numeric_pointwise_api() -> Result<()> {
    let a = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0, 3.0], vec![3])?;
    let b = Tensor::<Dyn, DefaultBackend>::from_slice(&[4.0_f32, 5.0, 6.0], vec![3])?;

    assert_eq!(a.add(&b)?.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
    assert_eq!(a.sub(&b)?.to_vec1::<f32>()?, vec![-3.0, -3.0, -3.0]);
    assert_eq!(a.mul(&b)?.to_vec1::<f32>()?, vec![4.0, 10.0, 18.0]);

    Ok(())
}

#[test]
fn arithmetic_operators_return_tensors_and_checked_methods_return_results() -> Result<()> {
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
fn checked_runtime_failure_is_a_panic_through_the_operator() -> Result<()> {
    let a = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0_f32, 2.0], vec![2])?;
    let b = Tensor::<Dyn, DefaultBackend>::from_slice(&[3.0_f32, 4.0, 5.0], vec![3])?;
    assert!(a.try_add(&b).is_err());

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = &a + &b;
    }));
    let message = panic.expect_err("an invalid runtime add must panic");
    let message = message
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| message.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(message.contains("Incin Add failed"));
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
