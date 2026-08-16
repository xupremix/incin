//! Pointwise Semantics Acceptance Contract Test.
//! Verifies standard pointwise numeric operations, real bool comparisons,
//! logical operations on bool tensors, target factories for bool, and exact bool extraction.

use incin::prelude::*;

#[test]
fn ordinary_numeric_pointwise_api() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([4.0_f32, 5.0, 6.0])?;

    assert_eq!(a.add(&b)?.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);
    assert_eq!(a.sub(&b)?.to_vec1::<f32>()?, vec![-3.0, -3.0, -3.0]);
    assert_eq!(a.mul(&b)?.to_vec1::<f32>()?, vec![4.0, 10.0, 18.0]);

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
