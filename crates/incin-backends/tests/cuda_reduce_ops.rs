//! CUDA reduce operations: axis reductions, argmax/argmin, cumsum, topk.
//!
//! The previous version of this file was named for topk, cumsum, argmax and
//! argmin, and asserted that `DTypeDescriptor::size_bytes` returns `Ok` for
//! three dtypes. It launched nothing. Its one green result said nothing about
//! any operation this file is named for, while counting as coverage.
//!
//! The rewrite of the sibling optimizer suite, which was vacuous in the same
//! way, uncovered that no CUDA optimizer kernel was ever launched -- so these
//! are written on the assumption that a suite asserting nothing was hiding
//! something.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_reduce_ops -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{
    argmax_argmin, cuda_available, cumsum, download_f32, download_i64, reduce, topk,
    upload_f32_shaped,
};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-5 * left.abs().max(right.abs()).max(1.0)
}

/// A 2x4 with a distinct maximum and minimum per row, so a reduction that
/// silently returned the first or last element would be caught.
fn matrix() -> (Vec<usize>, Vec<f32>) {
    (vec![2, 4], vec![3.0, 1.0, 4.0, 1.5, -2.0, 5.0, 0.5, -3.0])
}

#[test]
#[ignore = "requires CUDA hardware"]
fn axis_reductions_match_their_definitions() {
    if !cuda_available() {
        return;
    }
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);
    let rows: Vec<&[f32]> = values.chunks(4).collect();

    for (op, expect) in [
        ("sum", vec![9.5_f64, 0.5]),
        ("mean", vec![2.375, 0.125]),
        ("max", vec![4.0, 5.0]),
        ("min", vec![1.0, -3.0]),
        ("prod", vec![18.0, 15.0]),
    ] {
        let out = reduce(op, &input, 1, false).unwrap_or_else(|e| panic!("{op} must launch: {e}"));
        let got = download_f32(&out);
        assert_eq!(got.len(), rows.len(), "{op} should reduce axis 1 away");
        for (row, expected) in expect.iter().enumerate() {
            assert!(
                close(f64::from(got[row]), *expected),
                "{op} row {row}: kernel gave {}, definition gives {expected}",
                got[row]
            );
        }
    }
}

/// `keepdim` must keep the axis with extent one rather than drop it.
#[test]
#[ignore = "requires CUDA hardware"]
fn keepdim_retains_the_reduced_axis() {
    if !cuda_available() {
        return;
    }
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let dropped = reduce("sum", &input, 1, false).unwrap();
    let kept = reduce("sum", &input, 1, true).unwrap();

    assert_eq!(&dropped.shape[..], &[2]);
    assert_eq!(&kept.shape[..], &[2, 1]);
    assert_eq!(download_f32(&dropped), download_f32(&kept));
}

/// argmax and argmin must return *positions*, not values.
///
/// The rows are chosen so the extreme is neither first nor last, which a
/// kernel that returned a fixed index would otherwise pass.
#[test]
#[ignore = "requires CUDA hardware"]
fn argmax_and_argmin_return_positions() {
    if !cuda_available() {
        return;
    }
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    // row 0: [3, 1, 4, 1.5] -> max at 2, min at 1
    // row 1: [-2, 5, 0.5, -3] -> max at 1, min at 3
    let max_idx = download_i64(&argmax_argmin("argmax", &input, Some(1)).expect("argmax launches"));
    let min_idx = download_i64(&argmax_argmin("argmin", &input, Some(1)).expect("argmin launches"));

    assert_eq!(max_idx, vec![2, 1], "argmax positions");
    assert_eq!(min_idx, vec![1, 3], "argmin positions");
}

/// A prefix sum must be inclusive and per-row.
#[test]
#[ignore = "requires CUDA hardware"]
fn cumsum_accumulates_along_the_axis() {
    if !cuda_available() {
        return;
    }
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let got = download_f32(&cumsum(&input, 1).expect("cumsum launches"));

    let mut expected = Vec::new();
    for row in values.chunks(4) {
        let mut running = 0.0_f64;
        for value in row {
            running += f64::from(*value);
            expected.push(running);
        }
    }
    assert_eq!(got.len(), expected.len());
    for (index, want) in expected.iter().enumerate() {
        assert!(
            close(f64::from(got[index]), *want),
            "cumsum at {index}: kernel gave {}, definition gives {want}",
            got[index]
        );
    }
}

/// `topk` must return the k largest in descending order, with their indices.
#[test]
#[ignore = "requires CUDA hardware"]
fn topk_returns_ordered_values_and_their_indices() {
    if !cuda_available() {
        return;
    }
    let (shape, values) = matrix();
    let input = upload_f32_shaped(&shape, &values);

    let (vals, idx) = topk(&input, 2, 1, true).expect("topk launches");
    let (vals, idx) = (download_f32(&vals), download_i64(&idx));

    // row 0: [3, 1, 4, 1.5] -> 4 at 2, then 3 at 0
    // row 1: [-2, 5, 0.5, -3] -> 5 at 1, then 0.5 at 2
    assert!(close(f64::from(vals[0]), 4.0) && close(f64::from(vals[1]), 3.0));
    assert!(close(f64::from(vals[2]), 5.0) && close(f64::from(vals[3]), 0.5));
    assert_eq!(idx, vec![2, 0, 1, 2], "topk indices");

    // The indices must actually address the values returned beside them.
    for (position, index) in idx.iter().enumerate() {
        let row = position / 2;
        let source = values[row * 4 + usize::try_from(*index).unwrap()];
        assert!(
            close(f64::from(vals[position]), f64::from(source)),
            "topk index {index} does not point at the value it was returned with"
        );
    }
}
