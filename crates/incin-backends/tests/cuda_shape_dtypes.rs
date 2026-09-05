//! CUDA shape and movement operations: transpose, broadcast, narrow, concat.
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
//! Measuring the dtype boundary turned the registry around once already: the
//! kernels move bytes by element width and do no arithmetic, so nothing in
//! them is specific to `f32`, and the `f32`-only rows were a declaration
//! under-selling working kernels. The byte-exact matrix below is the
//! evidence those rows were widened against -- every dense storage dtype,
//! every movement kernel -- and the `q8_0` refusal pins the boundary a block
//! encoding draws.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_shape_dtypes -- --ignored`.

#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{
    broadcast, concat, download_bytes, download_f32, download_i64, narrow, require_cuda, transpose,
    upload_bytes, upload_f32_shaped, upload_i64,
};
use incin_core::tensor::dtype::DTypeId;

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
    // nothing in them is specific to `f32`. This was the first measurement
    // of that fact -- every one of the three moves `i64` to exactly the
    // right places -- made when the registry still advertised `f32` alone
    // and the seam below bypassed that gate. The rows have since been
    // widened on the byte-exact matrix at the bottom of this file; this test
    // stays as the typed-seam counterpart to it.
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

/// Every dense CUDA storage dtype with its element width: the exact set the
/// widened capability rows claim. `q8_0` is deliberately absent -- a block
/// encoding has no element width to move by, and the refusal test below pins
/// that it stays refused rather than silently misreading blocks as scalars.
fn movement_dtypes() -> [(DTypeId, usize); 6] {
    [
        (DTypeId::Bool, 1),
        (DTypeId::F16, 2),
        (DTypeId::BF16, 2),
        (DTypeId::F32, 4),
        (DTypeId::F64, 8),
        (DTypeId::I64, 8),
    ]
}

/// Element `index` as `width` bytes. Strided by 16 so no two elements of one
/// tensor share a byte pattern however the kernel reorders them.
fn element_bytes(width: usize, index: usize) -> Vec<u8> {
    (0..width).map(|b| (index * 16 + b) as u8).collect()
}

fn elements_bytes(width: usize, indices: &[usize]) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|i| element_bytes(width, *i))
        .collect()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn movement_kernels_move_every_storage_dtype_byte_exact() {
    require_cuda();
    for (dtype, width) in movement_dtypes() {
        // Transpose of a 2x3: elements [0,1,2,3,4,5] land as [0,3,1,4,2,5].
        let source = upload_bytes(&[2, 3], dtype, &elements_bytes(width, &[0, 1, 2, 3, 4, 5]));
        let moved = transpose(&source, 0, 1).expect("transpose accepts the dtype");
        assert_eq!(&moved.shape[..], &[3, 2], "{dtype:?}: transpose extents");
        assert_eq!(
            download_bytes(&moved),
            elements_bytes(width, &[0, 3, 1, 4, 2, 5]),
            "{dtype:?}: transpose moved bytes to the wrong places"
        );

        // Broadcast of a 1x3 row to 2x3 repeats rather than tiling the flat
        // buffer; broadcast of a 3x1 column repeats within each row.
        let row = upload_bytes(&[1, 3], dtype, &elements_bytes(width, &[0, 1, 2]));
        let spread = broadcast(&row, &[2, 3]).expect("broadcast accepts the dtype");
        assert_eq!(
            download_bytes(&spread),
            elements_bytes(width, &[0, 1, 2, 0, 1, 2]),
            "{dtype:?}: row broadcast repeated wrong"
        );
        let column = upload_bytes(&[3, 1], dtype, &elements_bytes(width, &[0, 1, 2]));
        let widened = broadcast(&column, &[3, 2]).expect("broadcast accepts the dtype");
        assert_eq!(
            download_bytes(&widened),
            elements_bytes(width, &[0, 0, 1, 1, 2, 2]),
            "{dtype:?}: column broadcast repeated wrong"
        );

        // Narrow along the leading axis (one contiguous run) and along an
        // inner axis (strided: the window is not one run).
        let wide = upload_bytes(
            &[4, 2],
            dtype,
            &elements_bytes(width, &[0, 1, 2, 3, 4, 5, 6, 7]),
        );
        let leading = narrow(&wide, 0, 1, 2).expect("narrow accepts the dtype");
        assert_eq!(&leading.shape[..], &[2, 2]);
        assert_eq!(
            download_bytes(&leading),
            elements_bytes(width, &[2, 3, 4, 5]),
            "{dtype:?}: leading-axis narrow took the wrong window"
        );
        let matrix = upload_bytes(&[2, 3], dtype, &elements_bytes(width, &[0, 1, 2, 3, 4, 5]));
        let inner = narrow(&matrix, 1, 1, 2).expect("narrow accepts the dtype");
        assert_eq!(&inner.shape[..], &[2, 2]);
        assert_eq!(
            download_bytes(&inner),
            elements_bytes(width, &[1, 2, 4, 5]),
            "{dtype:?}: inner-axis narrow took the wrong window"
        );

        // Concat along the leading axis and along an inner axis.
        let first = upload_bytes(&[1, 3], dtype, &elements_bytes(width, &[0, 1, 2]));
        let second = upload_bytes(&[1, 3], dtype, &elements_bytes(width, &[3, 4, 5]));
        let stacked = concat(&[&first, &second], 0).expect("concat accepts the dtype");
        assert_eq!(&stacked.shape[..], &[2, 3]);
        assert_eq!(
            download_bytes(&stacked),
            elements_bytes(width, &[0, 1, 2, 3, 4, 5]),
            "{dtype:?}: leading-axis concat ordered wrong"
        );
        let left = upload_bytes(&[2, 1], dtype, &elements_bytes(width, &[0, 3]));
        let right = upload_bytes(&[2, 2], dtype, &elements_bytes(width, &[1, 2, 4, 5]));
        let joined = concat(&[&left, &right], 1).expect("concat accepts the dtype");
        assert_eq!(&joined.shape[..], &[2, 3]);
        assert_eq!(
            download_bytes(&joined),
            elements_bytes(width, &[0, 1, 2, 3, 4, 5]),
            "{dtype:?}: inner-axis concat ordered wrong"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn movement_kernels_refuse_block_quantized_storage() {
    require_cuda();
    // 64 logical values in two 34-byte blocks: there is no element width to
    // move by, so every movement kernel must refuse rather than reinterpret
    // block bytes as scalars. The refusal is the contract, not the absence
    // of a test for a dtype the rows never claimed.
    let source = upload_bytes(&[2, 32], DTypeId::Q8_0, &[0u8; 68]);
    assert!(transpose(&source, 0, 1).is_err());
    assert!(broadcast(&source, &[2, 32, 1]).is_err());
    assert!(narrow(&source, 0, 0, 1).is_err());
    let other = upload_bytes(&[2, 32], DTypeId::Q8_0, &[0u8; 68]);
    assert!(concat(&[&source, &other], 0).is_err());
}

/// The widened rows admit what the kernels do: `i64` transpose and broadcast
/// through the public dispatch path, capability admission included. The
/// kernel-level tests above bypass the registry by construction, so without
/// this one the advertised claim would be exactly as unchecked as the old
/// `f32`-only claim was unmeasured.

#[test]
#[ignore = "requires CUDA hardware"]
fn widened_rows_admit_i64_through_public_dispatch() {
    use incin_backends::cuda::CudaBackendImpl;
    use incin_core::backend_authoring::ExecutionContext;
    use incin_core::exec::catalog::{ShapeAttributes, TransposeAttributes};
    use incin_core::exec::{TensorHandle, dispatch, op};
    use incin_core::tensor::device::Cuda;

    type B = CudaBackendImpl<Cuda>;
    require_cuda();
    let context = ExecutionContext::new(B::new());

    let source = upload_i64(&[2, 3], &[1, 2, 3, 4, 5, 6]);
    let handle = TensorHandle::from_storage::<B, i64, _>(&source);
    let moved = dispatch::execute::<op::TransposeExact, _>(
        &context,
        TransposeAttributes {
            first: 0,
            second: 1,
        },
        &[handle],
    )
    .expect("the widened transpose row admits i64");
    assert_eq!(
        download_i64(&moved),
        vec![1, 4, 2, 5, 3, 6],
        "admitted but misplaced: the row promises what the kernel does"
    );

    let row = upload_i64(&[1, 3], &[7, 8, 9]);
    let row_handle = TensorHandle::from_storage::<B, i64, _>(&row);
    let spread = dispatch::execute::<op::BroadcastAs, _>(
        &context,
        ShapeAttributes { shape: vec![2, 3] },
        &[row_handle],
    )
    .expect("the widened broadcast row admits i64");
    assert_eq!(download_i64(&spread), vec![7, 8, 9, 7, 8, 9]);
}
