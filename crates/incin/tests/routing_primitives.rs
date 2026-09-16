//! Routing primitives on the public API.
//!
//! Issue #103 lists the operations a mixture-of-experts layer needs and the
//! catalog does not have. Four of the eight it names have since been added
//! (`log_softmax`, `logsumexp`, `one_hot`, `scatter_add`); this file covers
//! the ones still arriving, starting with `repeat_interleave`, which is how a
//! token is expanded once per expert it was routed to.
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
