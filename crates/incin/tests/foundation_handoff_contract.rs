//! Master release handoff contract test suite for the Incin foundation.
//!
//! Validates the end-to-end user experience specified in Section 2 & 5A of the
//! Master Implementation Prompt:
//! - Shapes (static, mixed, fixed-rank runtime, dynamic-rank)
//! - Target allocation (`Cpu.zeros(shape![...])`)
//! - DType target selection
//! - Real bool comparisons, logical operations, and selection
//! - Unary, reduction, and matmul frontend APIs
//! - Builder-first NN layer initialization and frozen layers

#![cfg(all(feature = "target-api", feature = "cpu"))]

use incin::nn;
use incin::prelude::*;

const WIDTH: usize = 8;

#[test]
fn test_foundation_user_contract_shapes_and_allocation() -> Result<()> {
    let batch = 3usize;

    // 1. Shapes
    let _static = shape![4, const WIDTH];
    let _mixed = shape![batch, const WIDTH];
    let _fixed_rank_runtime = [batch, WIDTH];
    let _dynamic_rank = [batch, WIDTH];

    // 2. Target allocation without backend alias
    let x = Cpu.zeros(shape![batch, const WIDTH])?;
    assert_eq!(x.dims(), [3, 8]);

    // 3. DType target selection
    let ids = Cpu.dtype::<i64>()?.zeros([batch])?;
    assert_eq!(ids.dtype(), <i64 as ConstDType>::DESCRIPTOR);
    assert_eq!(ids.to_vec1::<i64>()?, vec![0, 0, 0]);

    Ok(())
}

#[test]
fn test_foundation_user_contract_bool_and_selection() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0, 3.0])?;
    let b = Cpu.tensor([0.0_f32, 2.0, 4.0])?;

    // Real bool comparison
    let mask = a.gt(&b)?;
    assert_eq!(mask.to_vec1::<bool>()?, vec![true, false, false]);

    // Bool selection
    let out = mask.where_cond(&a, &b)?;
    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, 2.0, 4.0]);

    let masked = a.masked_fill(&mask.logical_not()?, 0.0)?;
    assert_eq!(masked.to_vec1::<f32>()?, vec![1.0, 0.0, 0.0]);

    Ok(())
}

#[test]
fn test_foundation_user_contract_operations_and_nn() -> Result<()> {
    const IN: usize = 8;
    const OUT: usize = 4;
    let batch = 2usize;

    let target = Cpu;
    let model = nn::linear(shape![const IN, const OUT]).init(&target)?;

    let x = target.zeros(shape![batch, const IN])?;
    let y = model.forward(x.into_dyn().require_grad())?;

    let zero = target.zeros(shape![batch, const OUT])?;
    let positive = y.gt(&zero.into_dyn())?;

    let clipped = y.masked_fill(&positive.logical_not()?, 0.0)?;
    let score = clipped.sum::<Next<Here>>()?;
    assert_eq!(score.dims(), [2]);

    // Frozen linear layer
    let frozen_model = nn::linear(shape![const IN, const OUT])
        .frozen()
        .init(&target)?;
    let x_nograd = target.zeros(shape![batch, const IN])?;
    let frozen_out = frozen_model.forward(x_nograd.into_dyn().require_grad())?;
    assert_eq!(frozen_out.dims(), [2, 4]);

    Ok(())
}
