//! Routing primitives on the public API.
//!
//! Issue #103 lists the operations a mixture-of-experts layer needs and the
//! catalog does not have. Four of the eight it names were already there
//! (`log_softmax`, `logsumexp`, `one_hot`, `scatter_add`); this file covers
//! the ones that arrived since. `repeat_interleave` expands a token once per
//! expert it was routed to, `bincount` counts how many landed on each, and
//! `sort` groups them. Only `nonzero` is still missing, and it waits on the
//! shape decision in #102.
//!
//! The kernels are gradchecked inside `incin-backends`. What is pinned here is
//! the public surface: the operation dispatches under an active `GradMode`,
//! records on the tape through `execute_shaped`, and delivers a gradient
//! through `Gradients`.
#![cfg(feature = "cpu")]

use incin::prelude::*;

/// B.
type B = incin_backends::cpu::CpuBackendImpl;

/// The distinction the operation exists for: `repeat` lays the axis down end
/// to end, `repeat_interleave` repeats each element in place.
///
/// Both spellings produce the same extent, so a test that only checked the
/// shape would pass on either one.
#[test]
fn it_repeats_each_element_where_repeat_tiles_the_whole_axis() -> Result<()> {
    let x = Tensor::<s![3], B>::from_slice(&[1.0, 2.0, 3.0], ())?;

    assert_eq!(
        x.repeat_interleave(2, 0)?.to_vec1::<f32>()?,
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
    assert_eq!(
        x.repeat(&[2])?.to_vec1::<f32>()?,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "the two are the same function, so the interleave assertion proves nothing"
    );
    Ok(())
}

/// Only the named axis grows, and a negative axis counts from the end.
#[test]
fn it_expands_one_axis_and_accepts_a_negative_one() -> Result<()> {
    let x = Tensor::<s![2, 3], B>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;

    let rows = x.repeat_interleave(2, 0)?;
    assert_eq!(rows.dims().as_ref(), &[4, 3]);
    assert_eq!(
        rows.to_vec1::<f32>()?,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
    );

    let columns = x.repeat_interleave(2, 1)?;
    assert_eq!(columns.dims().as_ref(), &[2, 6]);
    assert_eq!(
        columns.to_vec1::<f32>()?,
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
    );
    assert_eq!(
        x.repeat_interleave(2, -1)?.to_vec1::<f32>()?,
        columns.to_vec1::<f32>()?
    );
    Ok(())
}

/// The gradient is the sum over each element's own block.
///
/// The weights are distinct powers of ten so that a backward which broadcast
/// one position's cotangent, or summed the wrong block, lands on a different
/// number rather than on a plausible one.
#[test]
fn the_gradient_sums_each_source_elements_block() -> Result<()> {
    let x = Tensor::<s![3], B, f32, Grad>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let weights =
        Tensor::<Dyn, B>::from_slice(&[1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0], vec![6])?;

    let loss = x.repeat_interleave(2, 0)?.mul_exact(&weights)?.sum_all()?;
    assert!((loss.to_scalar::<f32>()? - 332_211.0).abs() < 1e-1);

    let grads = loss.backward()?;
    assert_eq!(
        grads.require(&x)?.to_vec1::<f32>()?,
        vec![11.0, 1100.0, 110_000.0]
    );
    Ok(())
}

/// One repeat is the identity, not a rejection: a caller computing the count
/// should not have to special-case the value one.
#[test]
fn a_single_repeat_is_the_identity() -> Result<()> {
    let x = Tensor::<s![2, 2], B>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let once = x.repeat_interleave(1, 0)?;
    assert_eq!(once.dims().as_ref(), &[2, 2]);
    assert_eq!(once.to_vec1::<f32>()?, x.to_vec1::<f32>()?);
    Ok(())
}

/// Zero repeats and an out-of-range axis are refused, rather than producing a
/// tensor that holds no values while claiming a geometry.
#[test]
fn it_refuses_a_zero_repeat_and_an_axis_it_does_not_have() -> Result<()> {
    let x = Tensor::<s![3], B>::from_slice(&[1.0, 2.0, 3.0], ())?;
    assert!(x.repeat_interleave(0, 0).is_err());
    assert!(x.repeat_interleave(2, 3).is_err());
    assert!(x.repeat_interleave(2, -2).is_err());
    Ok(())
}

/// `bincount` turns an assignment into the per-slot counts a router needs.
///
/// The geometry that addressed the indices does not survive: a `[T, K]`
/// top-k assignment counts into the same `[N]` histogram as a flat vector of
/// the same indices.
#[test]
fn bincount_counts_every_index_whatever_shape_held_it() -> Result<()> {
    let flat = Tensor::<s![6], B, i64>::from_slice(&[2, 0, 2, 1, 2, 0], ())?;
    assert_eq!(flat.bincount::<3>()?.to_vec1::<i64>()?, vec![2, 1, 3]);

    let assignment = Tensor::<s![3, 2], B, i64>::from_slice(&[2, 0, 2, 1, 2, 0], ())?;
    assert_eq!(assignment.bincount::<3>()?.to_vec1::<i64>()?, vec![2, 1, 3]);

    // Wider than the indices used: the unused slot counts zero rather than
    // being absent, which is what keeps the offsets aligned with the experts.
    assert_eq!(flat.bincount::<5>()?.to_vec1::<i64>()?, vec![2, 1, 3, 0, 0]);
    Ok(())
}

/// The counts are the offsets, once scanned.
///
/// This is the composition the operation exists for: `bincount` then `cumsum`
/// describes where each expert's rows begin in a grouped buffer, so the
/// per-expert token count never becomes a tensor extent.
#[test]
fn the_counts_scan_into_the_offsets_a_grouped_buffer_needs() -> Result<()> {
    let assignment = Tensor::<s![6], B, i64>::from_slice(&[2, 0, 2, 1, 2, 0], ())?;
    let offsets = assignment.bincount::<3>()?.cumsum(axis!(0))?;
    assert_eq!(offsets.to_vec1::<i64>()?, vec![2, 3, 6]);
    Ok(())
}

/// An index outside the range is refused, not dropped.
///
/// A dropped index leaves a count quietly low and every offset after it wrong
/// by the same amount, which no consumer can distinguish from a genuinely
/// empty bin.
#[test]
fn an_index_outside_the_range_is_refused() -> Result<()> {
    let indices = Tensor::<s![3], B, i64>::from_slice(&[0, 3, 1], ())?;
    assert!(
        indices.bincount::<3>().is_err(),
        "index 3 is outside 0..3 and was counted anyway"
    );
    let negative = Tensor::<s![2], B, i64>::from_slice(&[0, -1], ())?;
    assert!(negative.bincount::<3>().is_err());
    Ok(())
}

/// `sort` returns the values `argsort` leaves behind.
///
/// The permutation alone is enough only when the operand is still to hand to
/// gather from. The contract pinned here is that the two agree: position `i`
/// of the values is the element the permutation's position `i` names.
#[test]
fn it_returns_the_values_beside_the_permutation_that_made_them() -> Result<()> {
    let x = Tensor::<s![5], B>::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], ())?;

    let (values, order) = x.sort(0, false)?;
    assert_eq!(values.to_vec1::<f32>()?, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    assert_eq!(order.to_vec1::<u32>()?, vec![1, 3, 0, 2, 4]);

    // The permutation and the values describe the same reordering.
    let source = x.to_vec1::<f32>()?;
    let gathered: Vec<f32> = order
        .to_vec1::<u32>()?
        .into_iter()
        .map(|index| source[index as usize])
        .collect();
    assert_eq!(gathered, values.to_vec1::<f32>()?);

    // `argsort` beside it is the same permutation without the values.
    assert_eq!(
        x.argsort(0, false)?.to_vec1::<u32>()?,
        order.to_vec1::<u32>()?
    );
    Ok(())
}

/// Descending is the same sort read the other way, and it is just as stable.
#[test]
fn it_sorts_descending_without_reversing_the_ties() -> Result<()> {
    let x = Tensor::<s![5], B>::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], ())?;

    let (values, order) = x.sort(0, true)?;
    assert_eq!(values.to_vec1::<f32>()?, vec![5.0, 4.0, 3.0, 1.0, 1.0]);
    // The two ones are still 1 then 3. Reversing the ascending permutation
    // would have put 3 first, which is the defect this asserts against.
    assert_eq!(order.to_vec1::<u32>()?, vec![4, 2, 0, 1, 3]);
    Ok(())
}

/// Equal keys keep the order they arrived in.
///
/// This is what lets a router group the same assignment the same way twice.
/// An unstable sort would still return correct values, so an assertion on the
/// values alone would pass while the grouping moved between runs.
#[test]
fn equal_keys_keep_the_order_they_arrived_in() -> Result<()> {
    let assignment = Tensor::<s![6], B, i64>::from_slice(&[2, 0, 2, 1, 2, 0], ())?;

    let (experts, order) = assignment.sort(0, false)?;
    assert_eq!(experts.to_vec1::<i64>()?, vec![0, 0, 1, 2, 2, 2]);
    assert_eq!(order.to_vec1::<u32>()?, vec![1, 5, 3, 0, 2, 4]);
    Ok(())
}

/// The composition the operation exists for.
///
/// Sorting the assignment groups the tokens by expert; `bincount` scanned by
/// `cumsum` says where each expert's block ends. Together they describe a
/// grouped buffer without the per-expert token count ever being an extent.
#[test]
fn the_sorted_assignment_and_the_offsets_describe_the_same_grouping() -> Result<()> {
    let assignment = Tensor::<s![6], B, i64>::from_slice(&[2, 0, 2, 1, 2, 0], ())?;

    let (experts, order) = assignment.sort(0, false)?;
    let offsets = assignment
        .bincount::<3>()?
        .cumsum(axis!(0))?
        .to_vec1::<i64>()?;
    assert_eq!(offsets, vec![2, 3, 6]);

    // Every expert's block in the sorted keys is exactly the span its offsets
    // name, so a grouped matmul can be handed the ranges rather than a shape.
    let sorted = experts.to_vec1::<i64>()?;
    let mut start = 0usize;
    for (expert, end) in offsets.iter().enumerate() {
        let end = *end as usize;
        assert!(
            sorted[start..end].iter().all(|slot| *slot == expert as i64),
            "expert {expert} owns {start}..{end}, which holds {:?}",
            &sorted[start..end]
        );
        start = end;
    }

    // And the permutation names, for each position in that grouping, the token
    // whose row belongs there.
    assert_eq!(order.to_vec1::<u32>()?, vec![1, 5, 3, 0, 2, 4]);
    let original = assignment.to_vec1::<i64>()?;
    for (position, token) in order.to_vec1::<u32>()?.into_iter().enumerate() {
        assert_eq!(original[token as usize], sorted[position]);
    }
    Ok(())
}

/// A named axis sorts within its rows rather than across them.
#[test]
fn it_sorts_along_the_axis_it_is_given() -> Result<()> {
    let x = Tensor::<s![2, 3], B>::from_slice(&[3.0, 1.0, 2.0, 9.0, 7.0, 8.0], ())?;

    let (values, order) = x.sort(1, false)?;
    // Row-major, so the flat buffer is the two rows end to end. Each row is
    // sorted within itself; a sort that ran down the columns instead would
    // have left the rows unchanged, since they are already ordered that way.
    assert_eq!(values.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    // Each row's permutation is its own, and indexes within the row.
    assert_eq!(order.to_vec1::<u32>()?, vec![1, 2, 0, 1, 2, 0]);
    Ok(())
}

/// An axis the operand does not have is refused.
#[test]
fn it_refuses_an_axis_it_does_not_have() -> Result<()> {
    let x = Tensor::<s![4], B>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    assert!(x.sort(1, false).is_err());
    Ok(())
}
