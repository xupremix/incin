//! Issue #88 runtime half: CUDA indexing and layout for the eleven rows the
//! issue listed — `embedding`, `gather`, `scatter`, `index_select`, `diag`,
//! `pad`, `repeat`, `tril`, `triu`, `unfold`, `pixel_shuffle`.
//!
//! The capability admission bug (integer index operands refused by the CUDA
//! rows) is covered host-side by `cuda_indexing_admission`. This file is the
//! hardware half: every test `#[ignore]`d until a CUDA runner picks it up,
//! every forward checked against a host reference with the same formulas CPU
//! uses, every training row's tape depth checked so a forward that returns
//! the right numbers but records nothing cannot pass, and real backwards
//! through each differentiable row.
//!
//! Out-of-range indices are the contract this issue most cares about. CPU's
//! `get_f64` panics on `gather`/`index_select` and `scatter` silently drops
//! the write; CUDA refuses with a typed `InvalidInput` before the kernel
//! reads the element (`error_flag` fires on the load path). Matching CPU
//! literally is impossible — a panic is not a typed error and a silent drop
//! is not fail-closed — so these tests pin the CUDA contract: a bounded
//! `reason`, the right `OperationKind`, and no device-side read of the
//! out-of-range location.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_indexing_layout -- --ignored`.
#![cfg(feature = "cuda")]

use incin_backends::cuda::testing::{download_f32, require_cuda, upload_f32_shaped, upload_i64};
use incin_backends::cuda::{CudaBackendImpl, tape_depth};
use incin_core::backend_authoring::{AutogradBackend, Execute, StorageBackend};
use incin_core::error::BackendError;
use incin_core::exec::catalog::{
    AxisAttributes, DiagonalAttributes, DuplicateIndexRule, NoAttributes, PadAttributes,
    PixelShuffleAttributes, RepeatAttributes, ScatterAttributes, UnfoldAttributes,
};
use incin_core::exec::{
    CanonicalError, CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle, op,
};
use incin_core::prelude::{CudaN, Local};
use incin_core::shapes::OperationKind;
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

fn h32(storage: &TestStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
}

fn hi64(storage: &<TestBackend as StorageBackend>::Storage<i64>) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, i64, Local>(storage)
}

fn read(storage: &TestStorage) -> Vec<f64> {
    download_f32(storage)
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol || (a.is_nan() && e.is_nan()),
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

/// Runs one op through `dispatch`, reporting the tape entries it added.
fn run<O, Attr>(
    attrs: Attr,
    handles: &[TensorHandle<'_>],
) -> Result<(TestStorage, usize), CanonicalError>
where
    O: CanonicalOperation<Attributes = Attr>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attrs, handles)?;
    Ok((out, tape_depth() - before))
}

fn run_ok<O, Attr>(attrs: Attr, handles: &[TensorHandle<'_>]) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = Attr>,
    TestBackend: Execute<O, Output = TestStorage>,
{
    run::<O, _>(attrs, handles).expect("an advertised operation must execute")
}

/// Seeds `loss = sum_all(out)` and walks backward, returning the grad for `id`.
fn backward_from_sum_all(id: incin_core::exec::TensorId, out: &TestStorage) -> Vec<f64> {
    let context = ExecutionContext::new(TestBackend::new());
    let loss =
        incin_core::exec::dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[h32(out)])
            .expect("sum_all executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad = grads.get(id).expect("the input has a gradient");
    read(grad)
}

/// Matches a typed `InvalidInput` refusal the index kernels raise when an
/// index value falls outside the axis it names.
fn expect_invalid_input(err: CanonicalError, operation: OperationKind, reason: &str) {
    match err {
        CanonicalError::Backend(BackendError::InvalidInput {
            operation: got,
            reason: got_reason,
        }) => {
            assert_eq!(
                got, operation,
                "OOB refusal names the wrong operation (expected {operation:?})"
            );
            assert_eq!(got_reason, reason, "OOB refusal has the wrong reason");
        }
        other => panic!("expected a typed InvalidInput refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Indexing (5)
// ---------------------------------------------------------------------------

/// `gather` on a 2x3 with a 2x2 index on axis 1: every output element is a
/// distinct source position, so a kernel that returned the first rows or
/// ignored the axis would be caught. Tape depth 1 (the tracked form only).
#[test]
#[ignore = "requires CUDA hardware"]
fn gather_picks_the_indexed_positions_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let index = upload_i64(&[2, 2], &[2, 0, 0, 1]);
    let (out, recorded) =
        run_ok::<op::Gather, _>(AxisAttributes { axis: 1 }, &[h32(&input), hi64(&index)]);
    assert_eq!(
        &out.shape[..],
        &[2, 2],
        "gather output follows the index shape"
    );
    assert_close(&read(&out), &[3.0, 1.0, 4.0, 5.0], 0.0, "gather forward");
    assert_eq!(
        recorded, 1,
        "gather's tracked form pushes exactly one tape entry"
    );
}

/// SumAll seeds ones; gather's backward scatter-adds them back onto the
/// positions the index named, accumulating duplicates.
#[test]
#[ignore = "requires CUDA hardware"]
fn gather_backward_scatters_onto_the_indexed_positions() {
    require_cuda();
    let input = upload_f32_shaped(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let index = upload_i64(&[2, 2], &[2, 0, 0, 1]);
    let input_id = TapeStorage::id(&input);
    let (out, _) =
        run_ok::<op::Gather, _>(AxisAttributes { axis: 1 }, &[h32(&input), hi64(&index)]);
    let grad = backward_from_sum_all(input_id, &out);
    // out[0,0]=3 -> in[0,2]; out[0,1]=1 -> in[0,0];
    // out[1,0]=4 -> in[1,0]; out[1,1]=5 -> in[1,1].
    assert_close(
        &grad,
        &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        1e-6,
        "gather backward",
    );
}

/// `scatter` overwrites four of the six target positions; the two untouched
/// positions keep their input values.
#[test]
#[ignore = "requires CUDA hardware"]
fn scatter_overwrites_the_indexed_positions_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let index = upload_i64(&[2, 2], &[2, 0, 0, 1]);
    let src = upload_f32_shaped(&[2, 2], &[7.0, 8.0, 9.0, 10.0]);
    let (out, recorded) = run_ok::<op::Scatter, _>(
        ScatterAttributes {
            axis: 1,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &[h32(&input), hi64(&index), h32(&src)],
    );
    assert_eq!(
        &out.shape[..],
        &[2, 3],
        "scatter preserves the target shape"
    );
    assert_close(
        &read(&out),
        &[8.0, 2.0, 7.0, 9.0, 10.0, 6.0],
        0.0,
        "scatter forward",
    );
    assert_eq!(recorded, 1, "scatter pushes exactly one tape entry");
}

/// The input keeps its cotangent everywhere except the overwritten
/// positions; the source receives the output cotangent only through the
/// last (here, only) write to each destination.
#[test]
#[ignore = "requires CUDA hardware"]
fn scatter_backward_splits_between_target_and_source() {
    require_cuda();
    let input = upload_f32_shaped(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let index = upload_i64(&[2, 2], &[2, 0, 0, 1]);
    let src = upload_f32_shaped(&[2, 2], &[7.0, 8.0, 9.0, 10.0]);
    let input_id = TapeStorage::id(&input);
    let src_id = TapeStorage::id(&src);
    let (out, _) = run_ok::<op::Scatter, _>(
        ScatterAttributes {
            axis: 1,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &[h32(&input), hi64(&index), h32(&src)],
    );
    let context = ExecutionContext::new(TestBackend::new());
    let loss =
        incin_core::exec::dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[h32(&out)])
            .expect("sum_all executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let grad_t = grads.get(input_id).expect("target has a gradient");
    let grad_src = grads.get(src_id).expect("source has a gradient");
    // Written positions zeroed: (0,0),(0,2),(1,0),(1,1). Unwritten: (0,1),(1,2).
    assert_close(
        &read(grad_t),
        &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-6,
        "scatter backward wrt target",
    );
    // Every write landed on a unique destination, so each source element
    // earns the full cotangent of the position it wrote.
    assert_close(
        &read(grad_src),
        &[1.0, 1.0, 1.0, 1.0],
        1e-6,
        "scatter backward wrt source",
    );
}

/// `index_select` is reshape+broadcast+gather: three tape entries, output
/// length equal to the flat index vector.
#[test]
#[ignore = "requires CUDA hardware"]
fn index_select_repeats_rows_in_index_order_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[4], &[10.0, 20.0, 30.0, 40.0]);
    let index = upload_i64(&[3], &[2, 0, 2]);
    let (out, recorded) =
        run_ok::<op::IndexSelect, _>(AxisAttributes { axis: 0 }, &[h32(&input), hi64(&index)]);
    assert_eq!(
        &out.shape[..],
        &[3],
        "index_select output length follows the index"
    );
    assert_close(
        &read(&out),
        &[30.0, 10.0, 30.0],
        0.0,
        "index_select forward",
    );
    assert_eq!(
        recorded, 3,
        "index_select = reshape + broadcast + gather, three tape entries"
    );
}

/// Duplicate selections accumulate on the backward scatter: index 2 appears
/// twice, so its source position receives both contributions.
#[test]
#[ignore = "requires CUDA hardware"]
fn index_select_backward_accumulates_duplicate_selections() {
    require_cuda();
    let input = upload_f32_shaped(&[4], &[10.0, 20.0, 30.0, 40.0]);
    let index = upload_i64(&[3], &[2, 0, 2]);
    let input_id = TapeStorage::id(&input);
    let (out, _) =
        run_ok::<op::IndexSelect, _>(AxisAttributes { axis: 0 }, &[h32(&input), hi64(&index)]);
    let grad = backward_from_sum_all(input_id, &out);
    assert_close(&grad, &[1.0, 0.0, 2.0, 0.0], 1e-6, "index_select backward");
}

/// `diag` has two forms: extract a matrix's diagonal, construct a matrix
/// from a vector. Both push one tape entry.
#[test]
#[ignore = "requires CUDA hardware"]
fn diag_extract_and_construct_match_the_host_reference_and_record() {
    require_cuda();

    // Extract: [2,2] -> [2].
    let matrix = upload_f32_shaped(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let (extracted, recorded) =
        run_ok::<op::Diag, _>(DiagonalAttributes { offset: 0 }, &[h32(&matrix)]);
    assert_eq!(
        &extracted.shape[..],
        &[2],
        "extract yields one element per diagonal step"
    );
    assert_close(&read(&extracted), &[1.0, 4.0], 0.0, "diag extract");
    assert_eq!(recorded, 1, "diag extract records one tape entry");

    // Construct: [2] -> [2,2].
    let vector = upload_f32_shaped(&[2], &[5.0, 6.0]);
    let (constructed, recorded) =
        run_ok::<op::Diag, _>(DiagonalAttributes { offset: 0 }, &[h32(&vector)]);
    assert_eq!(
        &constructed.shape[..],
        &[2, 2],
        "construct squares the vector"
    );
    assert_close(
        &read(&constructed),
        &[5.0, 0.0, 0.0, 6.0],
        0.0,
        "diag construct",
    );
    assert_eq!(recorded, 1, "diag construct records one tape entry");
}

/// Extract's adjoint scatters the cotangent back onto the diagonal;
/// construct's adjoint re-extracts the diagonal of the cotangent square.
#[test]
#[ignore = "requires CUDA hardware"]
fn diag_backwards_cover_both_forms() {
    require_cuda();

    let matrix = upload_f32_shaped(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let matrix_id = TapeStorage::id(&matrix);
    let (extracted, _) = run_ok::<op::Diag, _>(DiagonalAttributes { offset: 0 }, &[h32(&matrix)]);
    assert_close(
        &backward_from_sum_all(matrix_id, &extracted),
        &[1.0, 0.0, 0.0, 1.0],
        1e-6,
        "diag extract backward",
    );

    let vector = upload_f32_shaped(&[2], &[5.0, 6.0]);
    let vector_id = TapeStorage::id(&vector);
    let (constructed, _) = run_ok::<op::Diag, _>(DiagonalAttributes { offset: 0 }, &[h32(&vector)]);
    assert_close(
        &backward_from_sum_all(vector_id, &constructed),
        &[1.0, 1.0],
        1e-6,
        "diag construct backward",
    );
}

// ---------------------------------------------------------------------------
// Shape/layout (6)
// ---------------------------------------------------------------------------

/// CPU fixture: [1,2] padded with (1,1) and fill 9 -> [9,1,2,9].
#[test]
#[ignore = "requires CUDA hardware"]
fn pad_fills_the_requested_margins_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[2], &[1.0, 2.0]);
    let (out, recorded) = run_ok::<op::Pad, _>(
        PadAttributes {
            padding: vec![(1, 1)],
            value: 9.0,
        },
        &[h32(&input)],
    );
    assert_eq!(&out.shape[..], &[4], "pad grows each axis by before+after");
    assert_close(&read(&out), &[9.0, 1.0, 2.0, 9.0], 0.0, "pad forward");
    assert_eq!(recorded, 1, "pad records one tape entry");
}

/// Backward narrows the cotangent back to the interior, dropping the
/// padding's contribution.
#[test]
#[ignore = "requires CUDA hardware"]
fn pad_backward_shifts_the_cotangent_by_the_offsets() {
    require_cuda();
    let input = upload_f32_shaped(&[2], &[1.0, 2.0]);
    let input_id = TapeStorage::id(&input);
    let (out, _) = run_ok::<op::Pad, _>(
        PadAttributes {
            padding: vec![(1, 1)],
            value: 9.0,
        },
        &[h32(&input)],
    );
    assert_close(
        &backward_from_sum_all(input_id, &out),
        &[1.0, 1.0],
        1e-6,
        "pad backward",
    );
}

/// Rank-1 and rank-2 repeat: each source element appears the product of
/// its axis repeats times.
#[test]
#[ignore = "requires CUDA hardware"]
fn repeat_tiles_every_axis_and_records() {
    require_cuda();

    let v = upload_f32_shaped(&[2], &[1.0, 2.0]);
    let (out, recorded) =
        run_ok::<op::Repeat, _>(RepeatAttributes { repeats: vec![2] }, &[h32(&v)]);
    assert_eq!(&out.shape[..], &[4], "rank-1 repeat doubles the length");
    assert_close(&read(&out), &[1.0, 2.0, 1.0, 2.0], 0.0, "repeat rank-1");
    assert_eq!(recorded, 1, "repeat records one tape entry");

    let m = upload_f32_shaped(&[2, 1], &[1.0, 2.0]);
    let (out, recorded) = run_ok::<op::Repeat, _>(
        RepeatAttributes {
            repeats: vec![2, 2],
        },
        &[h32(&m)],
    );
    assert_eq!(
        &out.shape[..],
        &[4, 2],
        "rank-2 repeat multiplies both extents"
    );
    assert_close(
        &read(&out),
        &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
        0.0,
        "repeat rank-2",
    );
    assert_eq!(recorded, 1, "repeat records one tape entry");
}

/// Backward sums each source element's tiles back onto itself.
#[test]
#[ignore = "requires CUDA hardware"]
fn repeat_backward_sums_the_tiles_onto_the_source() {
    require_cuda();

    let v = upload_f32_shaped(&[2], &[1.0, 2.0]);
    let v_id = TapeStorage::id(&v);
    let (out, _) = run_ok::<op::Repeat, _>(RepeatAttributes { repeats: vec![2] }, &[h32(&v)]);
    assert_close(
        &backward_from_sum_all(v_id, &out),
        &[2.0, 2.0],
        1e-6,
        "repeat rank-1 backward",
    );

    let m = upload_f32_shaped(&[2, 1], &[1.0, 2.0]);
    let m_id = TapeStorage::id(&m);
    let (out, _) = run_ok::<op::Repeat, _>(
        RepeatAttributes {
            repeats: vec![2, 2],
        },
        &[h32(&m)],
    );
    assert_close(
        &backward_from_sum_all(m_id, &out),
        &[4.0, 4.0],
        1e-6,
        "repeat rank-2 backward",
    );
}

/// Rank-1 triangular: the kernel runs as a 1xC matrix, so element `(0, i)`
/// is the only position — matching CPU's `(0, idx[0])` rule. Four offsets
/// exercise both the k=0 diagonal and an off-diagonal shift.
#[test]
#[ignore = "requires CUDA hardware"]
fn triangular_rank_one_respects_every_offset_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[3], &[1.0, 2.0, 3.0]);

    let (tril0, recorded) = run_ok::<op::Tril, _>(DiagonalAttributes { offset: 0 }, &[h32(&input)]);
    assert_close(&read(&tril0), &[1.0, 0.0, 0.0], 0.0, "tril k0 rank-1");
    assert_eq!(recorded, 1, "tril records one tape entry");

    let (triu0, recorded) = run_ok::<op::Triu, _>(DiagonalAttributes { offset: 0 }, &[h32(&input)]);
    assert_close(&read(&triu0), &[1.0, 2.0, 3.0], 0.0, "triu k0 rank-1");
    assert_eq!(recorded, 1, "triu records one tape entry");

    let (tril1, _) = run_ok::<op::Tril, _>(DiagonalAttributes { offset: 1 }, &[h32(&input)]);
    assert_close(&read(&tril1), &[1.0, 2.0, 0.0], 0.0, "tril k1 rank-1");

    let (triu1, _) = run_ok::<op::Triu, _>(DiagonalAttributes { offset: 1 }, &[h32(&input)]);
    assert_close(&read(&triu1), &[0.0, 2.0, 3.0], 0.0, "triu k1 rank-1");
}

/// Rank-2 sanity: the same mask logic on a matrix, plus a real backward
/// (the mask passes the cotangent through on kept positions).
#[test]
#[ignore = "requires CUDA hardware"]
fn triangular_rank_two_matches_and_trains() {
    require_cuda();
    let input = upload_f32_shaped(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

    let (lower, _) = run_ok::<op::Tril, _>(DiagonalAttributes { offset: 0 }, &[h32(&input)]);
    assert_close(&read(&lower), &[1.0, 0.0, 3.0, 4.0], 0.0, "tril k0 rank-2");

    let input_id = TapeStorage::id(&input);
    assert_close(
        &backward_from_sum_all(input_id, &lower),
        &[1.0, 0.0, 1.0, 1.0],
        1e-6,
        "tril backward",
    );

    let (upper, _) = run_ok::<op::Triu, _>(DiagonalAttributes { offset: 0 }, &[h32(&input)]);
    assert_close(&read(&upper), &[1.0, 2.0, 0.0, 4.0], 0.0, "triu k0 rank-2");
}

/// CPU fixture: three overlapping windows of size 2, step 1 over a length-4
/// vector. The composite is narrow+unsqueeze per window then concat; tape
/// depth is the sum of those recorded views.
#[test]
#[ignore = "requires CUDA hardware"]
fn unfold_windows_the_vector_and_records() {
    require_cuda();
    let input = upload_f32_shaped(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let (out, recorded) = run_ok::<op::Unfold, _>(
        UnfoldAttributes {
            axis: 0,
            size: 2,
            step: 1,
        },
        &[h32(&input)],
    );
    assert_eq!(&out.shape[..], &[3, 2], "three windows of width 2");
    assert_close(
        &read(&out),
        &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
        0.0,
        "unfold forward",
    );
    assert!(
        recorded >= 1,
        "unfold's composite (narrow + reshape per window, then concat) records tape"
    );
}

/// Overlapping windows accumulate: interior elements belong to two windows.
#[test]
#[ignore = "requires CUDA hardware"]
fn unfold_backward_accumulates_through_overlapping_windows() {
    require_cuda();
    let input = upload_f32_shaped(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let input_id = TapeStorage::id(&input);
    let (out, _) = run_ok::<op::Unfold, _>(
        UnfoldAttributes {
            axis: 0,
            size: 2,
            step: 1,
        },
        &[h32(&input)],
    );
    assert_close(
        &backward_from_sum_all(input_id, &out),
        &[1.0, 2.0, 2.0, 1.0],
        1e-6,
        "unfold backward",
    );
}

/// CPU fixture: 1x4 channels of a 1x1 image shuffled to 1 channel of a 2x2.
/// The composite is reshape + three transposes + reshape; the permutation is
/// bijective, so the ones-seed backward stays ones.
#[test]
#[ignore = "requires CUDA hardware"]
fn pixel_shuffle_permutes_channels_and_inverts() {
    require_cuda();
    let input = upload_f32_shaped(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) =
        run_ok::<op::PixelShuffle, _>(PixelShuffleAttributes { upscale_factor: 2 }, &[h32(&input)]);
    assert_eq!(
        &out.shape[..],
        &[1, 1, 2, 2],
        "channels collapse by r^2, spatial extents grow by r"
    );
    assert_close(
        &read(&out),
        &[1.0, 2.0, 3.0, 4.0],
        0.0,
        "pixel_shuffle forward",
    );
    assert!(
        recorded >= 1,
        "pixel_shuffle's composite (reshape + 3 transposes + reshape) records tape"
    );
    assert_close(
        &backward_from_sum_all(input_id, &out),
        &[1.0, 1.0, 1.0, 1.0],
        1e-6,
        "pixel_shuffle backward",
    );
}

/// The dispatch path for `embedding` (not just the testing seam): rows are
/// gathered in index order, and backward accumulates repeated indices onto
/// the weight table.
#[test]
#[ignore = "requires CUDA hardware"]
fn embedding_dispatch_gathers_rows_and_accumulates_backward() {
    require_cuda();
    let weight = upload_f32_shaped(
        &[4, 3],
        &[
            0.0, 0.1, 0.2, // row 0
            1.0, 1.1, 1.2, // row 1
            2.0, 2.1, 2.2, // row 2
            3.0, 3.1, 3.2, // row 3
        ],
    );
    let indices = upload_i64(&[4], &[2, 0, 3, 2]);
    let weight_id = TapeStorage::id(&weight);
    let (out, recorded) =
        run_ok::<op::EmbeddingExact, _>(NoAttributes, &[hi64(&indices), h32(&weight)]);
    assert_eq!(&out.shape[..], &[4, 3], "one row of width 3 per index");
    assert_close(
        &read(&out),
        &[
            2.0, 2.1, 2.2, // index 2
            0.0, 0.1, 0.2, // index 0
            3.0, 3.1, 3.2, // index 3
            2.0, 2.1, 2.2, // index 2 again
        ],
        1e-6,
        "embedding forward",
    );
    assert_eq!(recorded, 1, "embedding records one tape entry");

    // SumAll seeds ones; row 2 receives both of its appearances.
    let grad = backward_from_sum_all(weight_id, &out);
    assert_close(
        &grad,
        &[
            1.0, 1.0, 1.0, // row 0: once
            0.0, 0.0, 0.0, // row 1: never
            2.0, 2.0, 2.0, // row 2: twice
            1.0, 1.0, 1.0, // row 3: once
        ],
        1e-6,
        "embedding backward accumulates repeated indices",
    );
}

// ---------------------------------------------------------------------------
// Out-of-range indices: typed refusals (CPU divergence documented per test)
// ---------------------------------------------------------------------------

/// CPU's `gather_storage` panics here (`index out of bounds: the len is 2
/// but the index is 5`); CUDA refuses with a typed `InvalidInput` before
/// the kernel reads the element. The two cannot match literally — a panic
/// is not a typed error — so the CUDA contract is what this pins.
#[test]
#[ignore = "requires CUDA hardware"]
fn out_of_range_gather_is_refused_typed() {
    require_cuda();
    let input = upload_f32_shaped(&[2], &[10.0, 20.0]);
    let index = upload_i64(&[1], &[5]);
    let err = run::<op::Gather, _>(AxisAttributes { axis: 0 }, &[h32(&input), hi64(&index)])
        .expect_err("an out-of-range gather index must be refused");
    expect_invalid_input(err, OperationKind::Gather, "index out of bounds");
}

/// Routes through `launch_gather`, so the refusal carries `OperationKind::
/// Gather` rather than `IndexSelect`. CPU panics the same way gather does;
/// CUDA refuses typed before the read.
#[test]
#[ignore = "requires CUDA hardware"]
fn out_of_range_index_select_is_refused_typed() {
    require_cuda();
    let input = upload_f32_shaped(&[2], &[10.0, 20.0]);
    let index = upload_i64(&[1], &[5]);
    let err = run::<op::IndexSelect, _>(AxisAttributes { axis: 0 }, &[h32(&input), hi64(&index)])
        .expect_err("an out-of-range index_select index must be refused");
    // The shared gather launcher names itself, not the composite.
    expect_invalid_input(err, OperationKind::Gather, "index out of bounds");
}

/// CPU's `scatter_storage` silently drops an out-of-range write (the kernel
/// would too if it lacked the flag); CUDA refuses typed before any element
/// is read. Dropping is not fail-closed, so matching it would be wrong.
#[test]
#[ignore = "requires CUDA hardware"]
fn out_of_range_scatter_is_refused_typed() {
    require_cuda();
    let input = upload_f32_shaped(&[2], &[10.0, 20.0]);
    let index = upload_i64(&[1], &[5]);
    let src = upload_f32_shaped(&[1], &[7.0]);
    let err = run::<op::Scatter, _>(
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &[h32(&input), hi64(&index), h32(&src)],
    )
    .expect_err("an out-of-range scatter index must be refused");
    expect_invalid_input(err, OperationKind::Scatter, "index out of bounds");
}

/// The embedding launcher raises `OperationKind::Embedding` (the family
/// identity the kernel already used) rather than `EmbeddingExact`; the
/// pass-through keeps that typed refusal intact. Index 4 is outside a
/// vocabulary of 4 rows.
#[test]
#[ignore = "requires CUDA hardware"]
fn out_of_range_embedding_is_refused_typed() {
    require_cuda();
    let weight = upload_f32_shaped(&[4, 2], &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let indices = upload_i64(&[1], &[4]);
    let err = run::<op::EmbeddingExact, _>(NoAttributes, &[hi64(&indices), h32(&weight)])
        .expect_err("an out-of-range embedding index must be refused");
    expect_invalid_input(
        err,
        OperationKind::Embedding,
        "embedding index out of bounds",
    );
}
