//! WGPU order statistics: `argmax`/`argmin`/`argsort`/`sort`/`topk` (#91).
//!
//! All five are host-side walks over downloaded `f32` in
//! `wgpu/backend/reduce.rs`, mirroring CPU's `cpu::ops::reduce::select` walk
//! for walk (strict `>`/`<` first-wins extremum, stable `sort_by` so ties
//! keep arrival order). Values stay `f32`; index outputs honor the requested
//! `u8`/`u32`/`i64` width with a checked fit.
//!
//! `dispatch::execute` runs `admit_invocation` (the capability check) before
//! `Execute`, so every test below proves its row is both advertised *and*
//! executable on the real request path — the runtime twin of the
//! compile-time `assert_every_advertised_wgpu_row_executes` in
//! `wgpu/executor.rs`. All five resolve `training = false`, so each case
//! also asserts the call recorded nothing on the tape.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --no-default-features --features std,cpu,wgpu --test wgpu_sort_ops`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{ArgsortAttributes, IndexReductionAttributes, TopKAttributes};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Aborts unless a WGPU adapter is present.
///
/// Failing is right here because this suite is `#![cfg(feature = "wgpu")]`:
/// compiling it at all is an explicit request for the backend, and both CI
/// jobs that enable it install a software adapter first. Skipping would
/// report `ok` for a test that ran nothing.
fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request for this backend. Skipping here would report `ok` for a test that ran nothing."
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

fn read_i64(storage: &TestStorage) -> Vec<i64> {
    <TestBackend as HostReadback>::int_to_vec1::<i64>(storage)
        .expect("reading an i64 index buffer back must succeed")
}

fn read_u32(storage: &TestStorage) -> Vec<i64> {
    <TestBackend as HostReadback>::int_to_vec1::<u32>(storage)
        .expect("reading a u32 index buffer back must succeed")
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

/// Single-output dispatch: `argmax`/`argmin`/`argsort`.
fn run1<O, A>(input: &TestStorage, attributes: A) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised WGPU operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

/// Multi-output dispatch: `sort`/`topk` return `(values, indices)`.
fn run_pair_outputs<O, A>(input: &TestStorage, attributes: A) -> ((TestStorage, TestStorage), usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = (TestStorage, TestStorage)>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised WGPU operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

#[test]
fn argmax_returns_positions_and_records_nothing() {
    require_wgpu();
    // row 0: [3, 1, 4] -> max at 2; row 1: [-2, 5, 0] -> max at 1
    let input = upload(&[3.0, 1.0, 4.0, -2.0, 5.0, 0.0], &[2, 3]);
    let (out, recorded) = run1::<op::ArgMax, _>(
        &input,
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "argmax is forward-only: no tape entry");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&out).dims(),
        &[2],
        "argmax squeezes the reduced axis"
    );
    assert_eq!(read_i64(&out), vec![2, 1], "argmax positions");

    // Global argmax: flat index of the max (5.0 at flat 4).
    let (flat, recorded) = run1::<op::ArgMax, _>(
        &input,
        IndexReductionAttributes {
            axis: None,
            dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "global argmax is forward-only");
    let dims = <TestBackend as StorageBackend>::shape::<f32>(&flat);
    assert!(
        dims.is_empty(),
        "axis=None yields a scalar, got shape {dims:?}"
    );
    assert_eq!(read_i64(&flat), vec![4], "global argmax flat index");
}

#[test]
fn argmax_hands_ties_to_the_first_winner() {
    require_wgpu();
    let input = upload(&[2.0, 5.0, 5.0], &[3]);
    let (out, recorded) = run1::<op::ArgMax, _>(
        &input,
        IndexReductionAttributes {
            axis: Some(0),
            dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "argmax is forward-only: no tape entry");
    assert_eq!(
        read_i64(&out),
        vec![1],
        "strict `>` keeps the first maximum on ties, matching CPU"
    );
}

#[test]
fn argmin_mirrors_argmax_and_records_nothing() {
    require_wgpu();
    let input = upload(&[3.0, 1.0, 4.0, -2.0, 5.0, 0.0], &[2, 3]);
    let (out, recorded) = run1::<op::ArgMin, _>(
        &input,
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "argmin is forward-only: no tape entry");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&out).dims(),
        &[2],
        "argmin squeezes the reduced axis"
    );
    assert_eq!(read_i64(&out), vec![1, 0], "argmin positions");

    let (flat, _) = run1::<op::ArgMin, _>(
        &input,
        IndexReductionAttributes {
            axis: None,
            dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(read_i64(&flat), vec![3], "global argmin flat index");
}

#[test]
fn argsort_returns_an_addressing_permutation_in_the_requested_width() {
    require_wgpu();
    // row 0: [3, 1, 4, 1.5]; row 1: [-2, 5, 0.5, -3]
    let values = [3.0f32, 1.0, 4.0, 1.5, -2.0, 5.0, 0.5, -3.0];
    let input = upload(&values, &[2, 4]);

    let (idx, recorded) = run1::<op::Argsort, _>(
        &input,
        ArgsortAttributes {
            axis: 1,
            descending: false,
            index_dtype: DTypeId::U32.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "argsort is forward-only: no tape entry");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&idx).dims(),
        &[2, 4],
        "argsort keeps the operand geometry"
    );
    // row 0: 1, 1.5, 3, 4 at [1, 3, 0, 2]; row 1: -3, -2, 0.5, 5 at [3, 0, 2, 1]
    let idx = read_u32(&idx);
    assert_eq!(idx, vec![1, 3, 0, 2, 3, 0, 2, 1], "argsort ascending");
    for (flat, &position) in idx.iter().enumerate() {
        let row = flat / 4;
        let source = values[row * 4 + usize::try_from(position).unwrap()];
        let below = if flat % 4 == 0 {
            None
        } else {
            Some(values[row * 4 + usize::try_from(idx[flat - 1]).unwrap()])
        };
        if let Some(below) = below {
            assert!(
                f64::from(below) <= f64::from(source) + 1e-6,
                "ascending order broken at {flat}: {below} !<= {source}"
            );
        }
    }

    let (idx, _) = run1::<op::Argsort, _>(
        &input,
        ArgsortAttributes {
            axis: 1,
            descending: true,
            index_dtype: DTypeId::U32.descriptor(),
        },
    );
    assert_eq!(
        read_u32(&idx),
        vec![2, 0, 3, 1, 1, 2, 0, 3],
        "argsort descending"
    );
}

#[test]
fn sort_orders_each_axis_slice_and_returns_a_replaying_permutation() {
    require_wgpu();
    let flat = [3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0];
    let input = upload(&flat, &[2, 3]);
    let ((values, indices), recorded) = run_pair_outputs::<op::Sort, _>(
        &input,
        ArgsortAttributes {
            axis: 1,
            descending: false,
            index_dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "sort is forward-only: no tape entry");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&values).dims(),
        &[2, 3],
        "sort keeps the operand geometry"
    );
    assert_eq!(
        read(&values),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "ascending sort values"
    );
    let idx = read_i64(&indices);
    assert_eq!(idx, vec![1, 2, 0, 1, 2, 0], "sort permutation");
    // The permutation replays to the sorted values.
    let sorted = read(&values);
    for (flat_i, &position) in idx.iter().enumerate() {
        assert_eq!(
            f64::from(flat[flat_i / 3 * 3 + position as usize]),
            sorted[flat_i],
            "indices must reorder the input into the sorted values at {flat_i}"
        );
    }

    // Descending flips both outputs.
    let ((values, indices), _) = run_pair_outputs::<op::Sort, _>(
        &input,
        ArgsortAttributes {
            axis: 1,
            descending: true,
            index_dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(
        read(&values),
        &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
        "descending sort values"
    );
    assert_eq!(
        read_i64(&indices),
        vec![0, 2, 1, 0, 2, 1],
        "descending permutation"
    );
}

#[test]
fn sort_keeps_tie_order_stable() {
    require_wgpu();
    let input = upload(&[2.0, 1.0, 2.0], &[3]);
    let ((values, indices), _) = run_pair_outputs::<op::Sort, _>(
        &input,
        ArgsortAttributes {
            axis: 0,
            descending: false,
            index_dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_eq!(read(&values), vec![1.0, 2.0, 2.0]);
    assert_eq!(
        read_i64(&indices),
        vec![1, 0, 2],
        "tied values keep arrival order, matching CPU's stable sort"
    );
}

#[test]
fn topk_returns_ordered_values_and_their_indices() {
    require_wgpu();
    // row 0: [3, 1, 4, 1.5]; row 1: [-2, 5, 0.5, -3]
    let flat = [3.0f32, 1.0, 4.0, 1.5, -2.0, 5.0, 0.5, -3.0];
    let input = upload(&flat, &[2, 4]);
    let ((values, indices), recorded) = run_pair_outputs::<op::TopK, _>(
        &input,
        TopKAttributes {
            k: 2,
            axis: 1,
            largest: true,
            index_dtype: DTypeId::U32.descriptor(),
        },
    );
    assert_eq!(recorded, 0, "topk is forward-only: no tape entry");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&values).dims(),
        &[2, 2],
        "topk shrinks the axis to k"
    );
    assert_close(&read(&values), &[4.0, 3.0, 5.0, 0.5], 1e-6, "topk values");
    let idx = read_u32(&indices);
    assert_eq!(
        idx,
        vec![2, 0, 1, 2],
        "topk indices address the values beside them"
    );
    let vals = read(&values);
    for (position, &index) in idx.iter().enumerate() {
        let row = position / 2;
        let source = flat[row * 4 + index as usize];
        assert!(
            (vals[position] - f64::from(source)).abs() < 1e-6,
            "topk index {index} does not point at the value it was returned with"
        );
    }

    // Smallest two ride the same walk with the comparator flipped.
    let ((values, indices), _) = run_pair_outputs::<op::TopK, _>(
        &input,
        TopKAttributes {
            k: 2,
            axis: 1,
            largest: false,
            index_dtype: DTypeId::I64.descriptor(),
        },
    );
    assert_close(&read(&values), &[1.0, 1.5, -3.0, -2.0], 1e-6, "smallest values");
    assert_eq!(
        read_i64(&indices),
        vec![1, 3, 3, 0],
        "smallest indices"
    );
}

#[test]
fn index_reductions_refuse_a_non_f32_operand_at_admission() {
    require_wgpu();
    // The rows are `F32_ONLY`: an integer operand must be refused by the
    // registry before any kernel or host walk is reached.
    let bytes: Vec<u8> = [1i64, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    let ints = <TestBackend as HostInterop>::from_bytes::<i64>(
        &bytes,
        &[3],
        DTypeId::I64.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading i64 storage must succeed");
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [TensorHandle::from_storage::<TestBackend, i64, _>(&ints)];
    let error = match incin_core::exec::dispatch::execute::<op::ArgMax, _>(
        &context,
        IndexReductionAttributes {
            axis: Some(0),
            dtype: DTypeId::I64.descriptor(),
        },
        &handles,
    ) {
        Ok(_) => panic!("argmax over i64 storage must be refused"),
        Err(error) => error,
    };
    let message = format!("{error}");
    assert!(
        message.contains("argmax") || message.contains("ArgMax"),
        "the refusal must name the operation: {message}"
    );
}

#[test]
fn index_reductions_refuse_an_out_of_range_axis() {
    require_wgpu();
    let input = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];
    assert!(
        incin_core::exec::dispatch::execute::<op::Argsort, _>(
            &context,
            ArgsortAttributes {
                axis: 5,
                descending: false,
                index_dtype: DTypeId::I64.descriptor(),
            },
            &handles,
        )
        .is_err(),
        "argsort past the operand rank must fail closed"
    );
}
