//! Integration coverage for #100: `where_cond` and `masked_fill` admit
//! masks that *broadcast into* the operands instead of demanding `ShapeEq`.
//!
//! The issue's direction is part of the contract under test: the mask may be
//! rank-deficit or carry size-1 axes, but it may never *enlarge* the data -
//! `where_cond` pins `S: BroadcastShape<S2, Output = S2>` (output keeps the
//! data's type) and `masked_fill` pins `S: BroadcastShape<S2, Output = S>`
//! (output keeps the input's type), so an enlarging mask has no matching
//! `Output` impl and a `Dyn` mask that would enlarge its data is refused at
//! run time with a named error.
#![cfg(feature = "cpu")]

use incin::prelude::*;

/// A typed mask with a size-1 axis broadcasts across each row, and the
/// result keeps the input's own shape type.
#[test]
fn masked_fill_accepts_a_typed_size_one_mask_axis() -> Result<()> {
    let x = Cpu.tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    // [2, 1] against [2, 3]: row 0's mask column is true, row 1's is false.
    let lhs = Cpu.tensor([[1.0f32], [0.0]])?;
    let mask = lhs.gt(&Cpu.tensor([[0.5f32], [0.5]])?)?;

    let out = x.masked_fill(&mask, 0.0)?;

    assert_eq!(out.dims(), [2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
    Ok(())
}

/// A typed rank-deficit mask (`[3]` against `[2, 3]`) is admitted: the
/// broadcast is right-aligned, so the same column mask selects on both rows.
#[test]
fn masked_fill_accepts_a_typed_rank_deficit_mask() -> Result<()> {
    let x = Cpu.tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let lhs = Cpu.tensor([1.0f32, 0.0, 1.0])?;
    let mask = lhs.gt(&Cpu.tensor([0.5f32, 0.5, 0.5])?)?;

    let out = x.masked_fill(&mask, 0.0)?;

    assert_eq!(out.dims(), [2, 3]);
    // Mask `[true, false, true]`: columns 0 and 2 are overwritten on both
    // rows; the middle column keeps its value.
    assert_eq!(out.to_vec1::<f32>()?, vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);
    Ok(())
}

/// The mirror pin on `where_cond`: a typed size-1 mask axis selects across
/// both rows, and the result keeps the *data's* shape type (`Output = S2`).
#[test]
fn where_cond_accepts_a_typed_size_one_mask_axis() -> Result<()> {
    let lhs = Cpu.tensor([[1.0f32, 0.0, 1.0]])?;
    let mask = lhs.gt(&Cpu.tensor([[0.5f32, 0.5, 0.5]])?)?;
    let on_true = Cpu.tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let on_false = Cpu.tensor([[-1.0f32, -1.0, -1.0], [-1.0, -1.0, -1.0]])?;

    let out = mask.where_cond(&on_true, &on_false)?;

    assert_eq!(out.dims(), [2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, -1.0, 3.0, 4.0, -1.0, 6.0]);
    Ok(())
}

/// A typed rank-deficit mask (`[3]` against `[2, 3]` data) selects over the
/// whole data - the causal-mask shape #100 exists to admit.
#[test]
fn where_cond_accepts_a_typed_rank_deficit_mask() -> Result<()> {
    let lhs = Cpu.tensor([1.0f32, 0.0, 1.0])?;
    let mask = lhs.gt(&Cpu.tensor([0.5f32, 0.5, 0.5])?)?;
    let on_true = Cpu.tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let on_false = Cpu.tensor([[-1.0f32, -1.0, -1.0], [-1.0, -1.0, -1.0]])?;

    let out = mask.where_cond(&on_true, &on_false)?;

    assert_eq!(out.dims(), [2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, -1.0, 3.0, 4.0, -1.0, 6.0]);
    Ok(())
}

/// The issue's own acceptance case: a causal `[T, T]` mask selects over
/// `[B, H, T, T]` scores without an explicit `broadcast_to`, and the result
/// keeps the scores' rank-4 shape type (`Output = S2` by rank promotion).
#[test]
fn where_cond_applies_a_rank_two_mask_to_rank_four_scores() -> Result<()> {
    let scores = Cpu.ones(shape![2, 2, 3, 3])?;
    let on_false = Cpu.zeros(shape![2, 2, 3, 3])?;
    // Row/column positions as 1-based values; `ge` yields the inclusive
    // lower-triangular (causal) mask over the `[T, T]` axes.
    let row = Cpu.tensor([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])?;
    let col = Cpu.tensor([[1.0f32, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])?;
    let mask = row.ge(&col)?;

    let out = mask.where_cond(&scores, &on_false)?;

    assert_eq!(out.dims(), [2, 2, 3, 3]);
    // One `[T, T]` plane, row-major, repeated across the `[B, H]` axes:
    // masked (future) positions take `on_false`, kept positions take the
    // scores' ones.
    let plane = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let expected: Vec<f32> = plane.iter().copied().cycle().take(36).collect();
    assert_eq!(out.to_vec1::<f32>()?, expected);
    Ok(())
}

/// The `Dyn` path the transformer layers actually take: both operands erase
/// their shape proof, the rank-deficit mask is admitted by the runtime
/// broadcast rules, and the backward pass routes the cotangent through the
/// *aligned* mask read - masked columns receive nothing on both rows.
#[test]
fn masked_fill_broadcast_mask_backward_reaches_the_input() -> Result<()> {
    let x = Cpu
        .tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?
        .into_dyn()
        .require_grad();
    let mask_src = Cpu.tensor([1.0f32, 0.0, 1.0])?;
    let mask = mask_src.gt(&Cpu.tensor([0.5f32, 0.5, 0.5])?)?.into_dyn();

    let out = x.masked_fill(&mask, 0.0)?;
    assert_eq!(out.dims(), [2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0]);

    let grads = out.sum_all()?.backward()?;
    let gradient = grads.require(&x)?;
    // Filled columns (mask true) receive nothing; the middle column passes
    // the cotangent through, identically on both rows.
    assert_eq!(
        gradient.to_vec1::<f32>()?,
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
    Ok(())
}

/// `where_cond`'s piecewise backward with a broadcast mask: `on_true` keeps
/// the cotangent where the mask fired, `on_false` where it did not, both
/// at the data's full shape.
#[test]
fn where_cond_broadcast_mask_backward_splits_both_operands() -> Result<()> {
    let mask_src = Cpu.tensor([1.0f32, 0.0, 1.0])?;
    let mask = mask_src.gt(&Cpu.tensor([0.5f32, 0.5, 0.5])?)?.into_dyn();
    let on_true = Cpu
        .tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?
        .into_dyn()
        .require_grad();
    let on_false = Cpu
        .tensor([[-1.0f32, -1.0, -1.0], [-1.0, -1.0, -1.0]])?
        .into_dyn()
        .require_grad();

    let out = mask.where_cond(&on_true, &on_false)?;
    assert_eq!(out.dims(), [2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![1.0, -1.0, 3.0, 4.0, -1.0, 6.0]);

    let grads = out.sum_all()?.backward()?;
    assert_eq!(
        grads.require(&on_true)?.to_vec1::<f32>()?,
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(
        grads.require(&on_false)?.to_vec1::<f32>()?,
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
    Ok(())
}

/// Fail-closed direction of the pin: a `Dyn` mask with an axis the input
/// cannot meet never reaches a kernel. The sentence is `validated.rs`'s
/// MaskedFill rule verbatim, which the CPU backend also repeats at its own
/// admission gate.
#[test]
fn masked_fill_rejects_a_dyn_mask_that_does_not_broadcast() -> Result<()> {
    let x = Cpu
        .tensor([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?
        .into_dyn();
    let mask_src = Cpu.tensor([[1.0f32, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])?;
    let mask = mask_src
        .gt(&Cpu.tensor([[0.5f32, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])?)?
        .into_dyn();

    let err = x.masked_fill(&mask, 0.0).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("mask must broadcast to the input shape"),
        "unexpected error: {text}"
    );
    Ok(())
}

/// The `where_cond` mirror: a `Dyn` mask larger than the data infers an
/// output that disagrees with the data shape the caller holds, so the
/// descriptor refuses it before any storage is read.
#[test]
fn where_cond_rejects_a_dyn_mask_that_enlarges_the_data() -> Result<()> {
    let mask_src = Cpu.tensor([[1.0f32, 0.0, 1.0], [0.0, 1.0, 0.0]])?;
    let mask = mask_src
        .gt(&Cpu.tensor([[0.5f32, 0.5, 0.5], [0.5, 0.5, 0.5]])?)?
        .into_dyn();
    let on_true = Cpu.tensor([[1.0f32, 2.0, 3.0]])?.into_dyn();
    let on_false = Cpu.tensor([[-1.0f32, -1.0, -1.0]])?.into_dyn();

    assert!(mask.where_cond(&on_true, &on_false).is_err());
    Ok(())
}
