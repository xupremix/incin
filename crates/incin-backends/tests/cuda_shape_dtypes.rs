//! CUDA shape and movement operations: transpose, broadcast, narrow.
//!
//! The previous version of this file asserted that `DTypeDescriptor::size_bytes`
//! returns 1, 2, 4 or 8 for eight dtypes. It launched nothing, opened no
//! device, and would have passed unchanged with the entire CUDA backend
//! deleted -- while standing in as coverage for the operations the file is
//! named for. Its sibling reduce and optimizer suites were vacuous in the same
//! way, and rewriting the optimizer one uncovered that no CUDA optimizer
//! kernel had ever been launched, so these are written on the assumption that
//! a suite asserting nothing is hiding something.
//!
//! The old name promised more than the backend offers, too. Among the movement
//! operations CUDA advertises `f32` alone -- for transpose, broadcast, narrow,
//! concat, stack, squeeze, unsqueeze and slice alike; only `reshape` advertises
//! more than one element type.
//!
//! Measuring that boundary turned it around. The last test here expected an
//! unadvertised element type to be refused and found instead that the kernels
//! move `i64` to exactly the right places: they move bytes by element width and
//! do no arithmetic, so nothing in them is specific to `f32`. The `f32`-only
//! rows are therefore a declaration that under-sells the kernels, in the same
//! way the elementwise rows once declared contiguous-only while a complete
//! strided kernel sat behind them. Widening the rows is left as open work,
//! because a declaration should be widened against evidence for every dtype it
//! would then claim.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_shape_dtypes -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{
    broadcast, download_f32, download_i64, narrow, require_cuda, transpose, upload_f32_shaped,
    upload_i64,
};

/// A 2x3 whose every element is distinct, so a movement that dropped, repeated
/// or transposed the wrong axis cannot land on the expected answer by accident.
fn matrix() -> (Vec<usize>, Vec<f32>) {
    (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
#[ignore = "requires CUDA hardware"]
fn transpose_permutes_the_elements_and_not_just_the_shape() {
    require_cuda();
    let (dims, values) = matrix();
    let source = upload_f32_shaped(&dims, &values);

    let out = transpose(&source, 0, 1).expect("transposing a 2x3 on device 0");

    assert_eq!(
        &out.shape[..],
        &[3, 2],
        "transpose did not swap the extents"
    );
    // Row-major 2x3 [[1,2,3],[4,5,6]] transposed is [[1,4],[2,5],[3,6]].
    assert_eq!(
        download_f32(&out),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "the extents were swapped but the elements were not moved"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn transpose_is_its_own_inverse() {
    require_cuda();
    let (dims, values) = matrix();
    let source = upload_f32_shaped(&dims, &values);

    let there = transpose(&source, 0, 1).expect("first transpose");
    let back = transpose(&there, 0, 1).expect("second transpose");

    assert_eq!(&back.shape[..], &[2, 3]);
    assert_eq!(
        download_f32(&back),
        values,
        "transposing twice did not restore the operand"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_repeats_the_operand_along_the_new_axis() {
    require_cuda();
    // A 1x3 row broadcast down to 2x3 must repeat, not tile the flat buffer.
    let source = upload_f32_shaped(&[1, 3], &[7.0, 8.0, 9.0]);

    let out = broadcast(&source, &[2, 3]).expect("broadcasting 1x3 to 2x3");

    assert_eq!(&out.shape[..], &[2, 3]);
    assert_eq!(download_f32(&out), vec![7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_repeats_along_a_trailing_axis_too() {
    require_cuda();
    // A 3x1 column to 3x2 repeats *within* each row rather than across rows,
    // which is the case a flat memcpy would get wrong while passing the test
    // above.
    let source = upload_f32_shaped(&[3, 1], &[7.0, 8.0, 9.0]);

    let out = broadcast(&source, &[3, 2]).expect("broadcasting 3x1 to 3x2");

    assert_eq!(&out.shape[..], &[3, 2]);
    assert_eq!(download_f32(&out), vec![7.0, 7.0, 8.0, 8.0, 9.0, 9.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_takes_the_window_along_the_leading_axis() {
    require_cuda();
    let source = upload_f32_shaped(&[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let out = narrow(&source, 0, 1, 2).expect("narrowing rows 1..3");

    assert_eq!(&out.shape[..], &[2, 2]);
    assert_eq!(download_f32(&out), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_takes_the_window_along_an_inner_axis() {
    require_cuda();
    // Narrowing a non-leading axis is the strided case: the window is not one
    // contiguous run, so a kernel that offset the pointer and copied `len`
    // elements would pass the leading-axis test and fail here.
    let (dims, values) = matrix();
    let source = upload_f32_shaped(&dims, &values);

    let out = narrow(&source, 1, 1, 2).expect("narrowing columns 1..3");

    assert_eq!(&out.shape[..], &[2, 2]);
    assert_eq!(download_f32(&out), vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_of_the_whole_axis_returns_the_operand_unchanged() {
    require_cuda();
    let (dims, values) = matrix();
    let source = upload_f32_shaped(&dims, &values);

    let out = narrow(&source, 0, 0, 2).expect("narrowing the whole leading axis");

    assert_eq!(&out.shape[..], &[2, 3]);
    assert_eq!(download_f32(&out), values);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn movement_kernels_are_element_type_agnostic() {
    require_cuda();
    // These kernels move bytes by element width; they do no arithmetic, so
    // nothing in them is specific to `f32`. Measured here rather than assumed:
    // every one of the three moves `i64` to exactly the right places.
    //
    // The capability registry nevertheless advertises `f32` alone for
    // transpose, broadcast and narrow. This test seam calls the kernels
    // directly and so bypasses that gate, which is why the mismatch is visible
    // here and not through the public API -- through the public API the rows
    // are what makes an `i64` transpose unavailable, not the kernel. Widening
    // the rows is tracked as open work (the narrow-CUDA-rows issue) rather
    // than done here, because a declaration should be widened against evidence
    // for every dtype it would then claim, not the one this test measures.
    let source = upload_i64(&[2, 3], &[1, 2, 3, 4, 5, 6]);

    let moved = transpose(&source, 0, 1).expect("the transpose kernel accepts i64");
    assert_eq!(&moved.shape[..], &[3, 2]);
    assert_eq!(
        download_i64(&moved),
        vec![1, 4, 2, 5, 3, 6],
        "transpose accepted i64 but moved it to the wrong places"
    );

    let window = narrow(&source, 0, 1, 1).expect("the narrow kernel accepts i64");
    assert_eq!(&window.shape[..], &[1, 3]);
    assert_eq!(download_i64(&window), vec![4, 5, 6]);

    let row = upload_i64(&[1, 3], &[7, 8, 9]);
    let spread = broadcast(&row, &[2, 3]).expect("the broadcast kernel accepts i64");
    assert_eq!(download_i64(&spread), vec![7, 8, 9, 7, 8, 9]);
}
