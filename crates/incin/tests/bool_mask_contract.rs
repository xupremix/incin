#![cfg(feature = "target-api")]

use incin::prelude::*;

#[test]
fn comparison_mask_feeds_where_cond() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([2.0_f32, 1.0, 4.0])?;
    let mask = a.lt(&b)?;

    let yes = Cpu.tensor([10.0_f32, 20.0, 30.0])?;
    let no = Cpu.tensor([-1.0_f32, -2.0, -3.0])?;

    let out = mask.where_cond(&yes, &no)?;

    assert_eq!(out.to_vec1::<f32>()?, vec![10.0, -2.0, 30.0]);

    Ok(())
}

#[test]
fn comparison_mask_feeds_masked_fill() -> Result<()> {
    let x = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let threshold = Cpu.tensor([1.5_f32, 1.5, 1.5])?;
    let mask = x.gt(&threshold)?;

    let out = x.masked_fill(&mask, 0.0)?;

    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, 0.0, 0.0]);

    Ok(())
}

#[test]
fn logical_masks_feed_selection() -> Result<()> {
    let x = Cpu.tensor([-2.0_f32, 0.5, 3.0])?;
    let lower = Cpu.tensor([0.0_f32; 3])?;
    let upper = Cpu.tensor([2.0_f32; 3])?;

    let in_range = x.gt(&lower)?.logical_and(&x.lt(&upper)?)?;
    let zeros = Cpu.zeros([3])?;

    let selected = in_range.where_cond(&x, &zeros)?;

    assert_eq!(selected.to_vec1::<f32>()?, vec![0.0, 0.5, 0.0]);

    Ok(())
}

#[test]
fn bool_where_preserves_integer_value_dtype() -> Result<()> {
    let lhs = Cpu.tensor([1_i64, 0, 1])?;
    let rhs = Cpu.tensor([1_i64, 1, 1])?;
    let mask = lhs.eq(&rhs)?;

    let yes = Cpu.tensor([10_i64, 20, 30])?;
    let no = Cpu.tensor([40_i64, 50, 60])?;

    let out = mask.where_cond(&yes, &no)?;

    assert_eq!(out.to_vec1::<i64>()?, vec![10, 50, 30]);

    Ok(())
}
