//! WGPU indexing ops: `embedding`/`gather`/`index_select` host-walks and the
//! `masked_fill`/`where_cond` GPU selection kernel (#91 Tier-1 remaining).
//!
//! Fail-closed contract: every capability row this suite exercises has an
//! `Execute` impl *and* a real runtime CPU-twin test here — forward values
//! against the same recipe CPU runs, backward against CPU's scatter-add, and
//! a broadcast-mask case that would only pass if the pre-broadcast path
//! actually launches. Bool/u8/u32/i64 round-trips go through
//! `from_bytes`/`to_bytes`/`int_to_vec1` so the widened storage rows are
//! proven rather than merely declared.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --features wgpu --test wgpu_indexing`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, TapeStorage, op,
};
use incin_core::error::BackendError;
use incin_core::exec::catalog::{
    AxisAttributes, BincountAttributes, DiagonalAttributes, DuplicateIndexRule, NoAttributes,
    OneHotAttributes, ScalarAttributes, ScatterAttributes,
};
use incin_core::exec::{CanonicalError, ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::shapes::error::OperationKind;
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;
type IntStorage = <TestBackend as StorageBackend>::Storage<i64>;
type BoolStorage = <TestBackend as StorageBackend>::Storage<bool>;

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

fn upload_f32(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading f32 values must succeed")
}

fn upload_i64(values: &[i64], shape: &[usize]) -> IntStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<i64>(
        &bytes,
        shape,
        DTypeId::I64.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading i64 indices must succeed")
}

type U8Storage = <TestBackend as StorageBackend>::Storage<u8>;

fn upload_u8(values: &[u8], shape: &[usize]) -> U8Storage {
    <TestBackend as HostInterop>::from_bytes::<u8>(
        values,
        shape,
        DTypeId::U8.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading u8 indices must succeed")
}

/// Bool mask uploaded as one byte per element (0/1); physical `f32` expansion
/// is `from_bytes`'s job.
fn upload_bool(values: &[bool], shape: &[usize]) -> BoolStorage {
    let bytes: Vec<u8> = values.iter().map(|&b| u8::from(b)).collect();
    <TestBackend as HostInterop>::from_bytes::<bool>(
        &bytes,
        shape,
        DTypeId::Bool.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading a bool mask must succeed")
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading an f32 buffer back must succeed")
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

/// Run a mixed-dtype request (f32 / i64 / bool operands in any slot) and
/// return the output storage plus how many tape entries the call pushed.
fn run_mixed<O, Attr>(
    f32_inputs: &[&TestStorage],
    int_inputs: &[&IntStorage],
    bool_inputs: &[&BoolStorage],
    attrs: Attr,
) -> Result<
    (
        <TestBackend as incin_core::backend_authoring::Execute<O>>::Output,
        usize,
    ),
    incin_core::exec::CanonicalError,
>
where
    O: incin_core::exec::CanonicalOperation<Attributes = Attr>,
    TestBackend: incin_core::backend_authoring::Execute<O>,
    Attr: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let mut handles: Vec<TensorHandle<'_>> = Vec::new();
    for s in f32_inputs {
        handles.push(TensorHandle::from_storage::<TestBackend, f32, _>(s));
    }
    for s in int_inputs {
        handles.push(TensorHandle::from_storage::<TestBackend, i64, _>(s));
    }
    for s in bool_inputs {
        handles.push(TensorHandle::from_storage::<TestBackend, bool, _>(s));
    }
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attrs, &handles)?;
    Ok((out, incin_backends::wgpu::tape_depth() - before))
}

fn sum_loss(out: &TestStorage) -> TestStorage {
    let ctx = ExecutionContext::new(TestBackend::default());
    let handle = TensorHandle::from_storage::<TestBackend, f32, _>(out);
    incin_core::exec::dispatch::execute::<op::SumAll, _>(&ctx, NoAttributes, &[handle])
        .expect("sum_all must run to seed the backward")
}

/// Run a ternary index request in catalog order (input, index, source) and
/// return the output storage plus how many tape entries the call pushed.
///
/// `run_mixed` cannot serve these: it groups `f32` operands before integer
/// ones, which would order the handles (input, source, index) against the
/// catalog's (input, index, source).
fn run_ternary<O, Attr>(
    input: &TestStorage,
    index: &IntStorage,
    source: &TestStorage,
    attrs: Attr,
) -> Result<(TestStorage, usize), CanonicalError>
where
    O: incin_core::exec::CanonicalOperation<Attributes = Attr>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    Attr: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(input),
        TensorHandle::from_storage::<TestBackend, i64, _>(index),
        TensorHandle::from_storage::<TestBackend, f32, _>(source),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attrs, &handles)?;
    Ok((out, incin_backends::wgpu::tape_depth() - before))
}

// ── embedding ───────────────────────────────────────────────────────────────

#[test]
fn embedding_gathers_rows_and_accumulates_repeated_indices() {
    require_wgpu();
    // weight [3,2]: row0=[1,2], row1=[3,4], row2=[5,6]; indices [0,2,0]
    let w = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let idx = upload_i64(&[0, 2, 0], &[3]);

    // Catalog input order is (indices, weight), matching CPU and CUDA.
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, i64, _>(&idx),
        TensorHandle::from_storage::<TestBackend, f32, _>(&w),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::EmbeddingExact, _>(
        &context,
        NoAttributes,
        &handles,
    )
    .expect("embedding must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;

    assert_eq!(
        read_f32(&out),
        vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0],
        "row 0, row 2, row 0 again — CPU's own fixture"
    );
    assert_eq!(
        recorded, 1,
        "embedding advertises training and pushes ONE TapeEntry (weight only, \
         index off-tape), matching cpu::ops::embedding exactly"
    );

    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("embedding backward must run");
    let gw = grads
        .get(TapeStorage::id(&w))
        .expect("weight table must receive a gradient");
    // row 0 addressed twice -> 2.0; row 1 never -> 0.0; row 2 once -> 1.0
    assert_eq!(
        read_f32(gw),
        vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0],
        "repeated index must ACCUMULATE, not overwrite (CPU's scatter-add rule)"
    );
}

// ── gather ──────────────────────────────────────────────────────────────────

#[test]
fn gather_selects_along_axis_and_scatters_grad_back() {
    require_wgpu();
    // input [2,3] row-major; gather dim=1 with index [[2,0],[0,1]] shape [2,2]
    // out[0][*] = input[0][idx[0][*]] = [3, 1]; out[1][*] = [4, 5]
    let input = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = upload_i64(&[2, 0, 0, 1], &[2, 2]);
    let (out, recorded) =
        run_mixed::<op::Gather, _>(&[&input], &[&index], &[], AxisAttributes { axis: 1 })
            .expect("gather must execute");

    assert_close(
        &read_f32(&out),
        &[3.0, 1.0, 4.0, 5.0],
        0.0,
        "gather forward",
    );
    assert_eq!(recorded, 1, "gather pushes ONE TapeEntry (input only)");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("gather backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    // cotangent 1 at out(0,0)->in(0,2); out(0,1)->in(0,0);
    // out(1,0)->in(1,0); out(1,1)->in(1,1)
    assert_eq!(
        read_f32(gin),
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        "cotangent lands only on the selected positions"
    );
}

// ── index_select ────────────────────────────────────────────────────────────

#[test]
fn index_select_replaces_an_axis_and_accumulates_dups() {
    require_wgpu();
    // input [4]; index [2,0,2] replaces axis 0 -> [v2, v0, v2]
    let input = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let index = upload_i64(&[2, 0, 2], &[3]);
    let (out, recorded) =
        run_mixed::<op::IndexSelect, _>(&[&input], &[&index], &[], AxisAttributes { axis: 0 })
            .expect("index_select must execute");

    assert_eq!(read_f32(&out), vec![30.0, 10.0, 30.0]);
    assert_eq!(recorded, 1);

    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("index_select backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    // index 2 appears twice -> 2.0; index 0 once -> 1.0; 1 and 3 untouched
    assert_eq!(
        read_f32(gin),
        vec![1.0, 0.0, 2.0, 0.0],
        "duplicate selections accumulate"
    );
}

// ── one_hot ───────────────────────────────────────────────────────────────────

#[test]
fn one_hot_encodes_indices_as_a_bool_mask_without_a_tape_entry() {
    require_wgpu();
    // indices [0, 2, 1] with depth 3 -> rows [1,0,0], [0,0,1], [0,1,0].
    let indices = upload_i64(&[0, 2, 1], &[3]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [TensorHandle::from_storage::<TestBackend, i64, _>(&indices)];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::OneHot, _>(
        &context,
        OneHotAttributes { depth: 3 },
        &handles,
    )
    .expect("one_hot is advertised and must execute");
    assert_eq!(
        incin_backends::wgpu::tape_depth() - before,
        0,
        "one_hot is forward-only: no tape entry"
    );
    assert_eq!(
        out.metadata().dtype(),
        DTypeId::Bool.descriptor(),
        "one_hot result is bool storage over the physical f32 encoding"
    );
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&out).dims(),
        &[3, 3],
        "one_hot appends a depth axis"
    );
    let packed = <TestBackend as HostInterop>::to_bytes::<bool>(&out)
        .expect("packing the 0.0/1.0 mask as bool bytes must succeed");
    assert_eq!(
        packed,
        vec![1, 0, 0, 0, 0, 1, 0, 1, 0],
        "one_hot mask"
    );

    // An index outside 0..depth writes no 1 (CPU's ONNX skip, not an error).
    let oob = upload_i64(&[5], &[1]);
    let handles = [TensorHandle::from_storage::<TestBackend, i64, _>(&oob)];
    let out = incin_core::exec::dispatch::execute::<op::OneHot, _>(
        &context,
        OneHotAttributes { depth: 3 },
        &handles,
    )
    .expect("out-of-range one_hot index is a silent skip, not an error");
    let packed = <TestBackend as HostInterop>::to_bytes::<bool>(&out)
        .expect("packing the all-zero mask as bool bytes must succeed");
    assert_eq!(packed, vec![0, 0, 0], "oob one_hot is all zero");

    // The u8 index width rides the same walk (the capability row's union).
    let narrow = upload_u8(&[1, 0], &[2]);
    let handles = [TensorHandle::from_storage::<TestBackend, u8, _>(&narrow)];
    let out = incin_core::exec::dispatch::execute::<op::OneHot, _>(
        &context,
        OneHotAttributes { depth: 2 },
        &handles,
    )
    .expect("u8 one_hot indices must execute");
    let packed = <TestBackend as HostInterop>::to_bytes::<bool>(&out)
        .expect("packing the u8-indexed mask as bool bytes must succeed");
    assert_eq!(packed, vec![0, 1, 1, 0], "u8 one_hot mask");
}

// ── bincount ────────────────────────────────────────────────────────────────

fn read_histogram(out: &IntStorage) -> Vec<i64> {
    <TestBackend as HostReadback>::int_to_vec1::<i64>(out)
        .expect("reading an i64 histogram back must succeed")
}

#[test]
fn bincount_counts_indices_into_a_fixed_histogram_without_a_tape_entry() {
    require_wgpu();
    // [0, 2, 0, 1, 2, 2] with 3 bins -> [2, 1, 3], the routing fixture CPU proves.
    let indices = upload_i64(&[0, 2, 0, 1, 2, 2], &[6]);
    let (out, recorded) = run_mixed::<op::Bincount, _>(
        &[],
        &[&indices],
        &[],
        BincountAttributes { bins: 3 },
    )
    .expect("bincount is advertised and must execute");
    assert_eq!(recorded, 0, "bincount is forward-only: no tape entry");
    assert_eq!(
        out.metadata().dtype(),
        DTypeId::I64.descriptor(),
        "bincount result is i64 storage"
    );
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<i64>(&out).dims(),
        &[3],
        "bincount output is the static [bins] histogram"
    );
    assert_eq!(read_histogram(&out), vec![2, 1, 3], "bincount histogram");

    // Geometry does not survive: the same values as [2, 3] count identically.
    let grid = upload_i64(&[0, 2, 0, 1, 2, 2], &[2, 3]);
    let (out, _) = run_mixed::<op::Bincount, _>(&[], &[&grid], &[], BincountAttributes { bins: 3 })
        .expect("bincount over a rank-2 index tensor must execute");
    assert_eq!(
        read_histogram(&out),
        vec![2, 1, 3],
        "bincount ignores the operand's own shape"
    );

    // Wider than the data: trailing bins stay zero.
    let (out, _) = run_mixed::<op::Bincount, _>(
        &[],
        &[&indices],
        &[],
        BincountAttributes { bins: 5 },
    )
    .expect("bincount with spare bins must execute");
    assert_eq!(
        read_histogram(&out),
        vec![2, 1, 3, 0, 0],
        "empty bins count zero"
    );

    // The u8 index width rides the same walk (the capability row's union).
    let narrow = upload_u8(&[1, 0], &[2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [TensorHandle::from_storage::<TestBackend, u8, _>(&narrow)];
    let out = incin_core::exec::dispatch::execute::<op::Bincount, _>(
        &context,
        BincountAttributes { bins: 2 },
        &handles,
    )
    .expect("u8 bincount indices must execute");
    assert_eq!(read_histogram(&out), vec![1, 1], "u8 bincount histogram");
}

#[test]
fn bincount_refuses_out_of_range_indices_and_zero_bins_like_cpu() {
    require_wgpu();
    // Slot 5 does not exist in a 3-bin histogram: CPU's `bincount_storage`
    // raises `InvalidInput` here, and this backend mirrors that rather than
    // `one_hot`'s silent ONNX skip — a quietly-low count would corrupt every
    // offset derived from it.
    let bad = upload_i64(&[5], &[1]);
    let error = match run_mixed::<op::Bincount, _>(&[], &[&bad], &[], BincountAttributes { bins: 3 })
    {
        Ok(_) => panic!("an out-of-range bincount index must be refused"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            CanonicalError::Backend(BackendError::InvalidInput {
                operation: OperationKind::Bincount,
                ..
            })
        ),
        "the refusal must stay the typed InvalidInput CPU raises, got {error:?}"
    );
    assert!(
        format!("{error:?}").contains("bincount index is not a whole number inside the bin range"),
        "the refusal must carry CPU's reason, got {error:?}"
    );

    // Zero bins names no histogram at all.
    let indices = upload_i64(&[0], &[1]);
    let error =
        match run_mixed::<op::Bincount, _>(&[], &[&indices], &[], BincountAttributes { bins: 0 })
        {
            Ok(_) => panic!("a zero-bin bincount must be refused"),
            Err(error) => error,
        };
    assert!(
        format!("{error:?}").contains("bincount needs at least one bin to count into"),
        "the refusal must carry CPU's reason, got {error:?}"
    );

    // A float operand is not an index: `index_values` refuses it by name
    // (or the descriptor's own per-operand contract does first).
    let floats = upload_f32(&[0.0], &[1]);
    match run_mixed::<op::Bincount, _>(&[&floats], &[], &[], BincountAttributes { bins: 1 }) {
        Ok(_) => panic!("a float bincount operand must be refused"),
        Err(_) => {}
    }
}

// ── diag ────────────────────────────────────────────────────────────────────

#[test]
fn diag_extracts_the_offset_diagonal_and_scatters_grad_back() {
    require_wgpu();
    // input [3,3] row-major 1..=9; offset 1 names [2, 6].
    let input = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let (out, recorded) =
        run_mixed::<op::Diag, _>(&[&input], &[], &[], DiagonalAttributes { offset: 1 })
            .expect("diag must execute");

    assert_close(&read_f32(&out), &[2.0, 6.0], 0.0, "diag offset-1 forward");
    assert_eq!(recorded, 1, "diag pushes ONE TapeEntry (input only)");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("diag backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    // Cotangent 1 lands on (0,1) and (1,2) only — CPU's own fixture.
    assert_eq!(
        read_f32(gin),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "extract backward scatters onto the named diagonal"
    );
}

#[test]
fn diag_builds_the_offset_matrix_and_reads_its_grad_back() {
    require_wgpu();
    // vector [1, 2] with offset -1: entry i lands at (i + 1, i).
    let input = upload_f32(&[1.0, 2.0], &[2]);
    let (out, recorded) =
        run_mixed::<op::Diag, _>(&[&input], &[], &[], DiagonalAttributes { offset: -1 })
            .expect("diag must execute");

    assert_eq!(
        read_f32(&out),
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        "construct forward writes the sub-diagonal of a 3x3"
    );
    assert_eq!(recorded, 1);

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("diag backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    assert_eq!(
        read_f32(gin),
        vec![1.0, 1.0],
        "construct backward reads the cotangent's own diagonal"
    );
}

// ── scatter ─────────────────────────────────────────────────────────────────

#[test]
fn scatter_last_write_wins_and_splits_the_cotangent() {
    require_wgpu();
    // input [10, 20, 30, 40]; index [1, 2, 1], source [7, 8, 9] along dim 0:
    // slot 1 takes 7 then 9 (last wins), slot 2 takes 8.
    let input = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let index = upload_i64(&[1, 2, 1], &[3]);
    let source = upload_f32(&[7.0, 8.0, 9.0], &[3]);
    let (out, recorded) = run_ternary::<op::Scatter, _>(
        &input,
        &index,
        &source,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
    )
    .expect("scatter must execute");

    assert_eq!(read_f32(&out), vec![10.0, 9.0, 8.0, 40.0]);
    assert_eq!(
        recorded, 1,
        "scatter pushes ONE TapeEntry over both value operands"
    );

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("scatter backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    assert_eq!(
        read_f32(gin),
        vec![1.0, 0.0, 0.0, 1.0],
        "overwritten positions keep no input cotangent"
    );
    let gsrc = grads
        .get(TapeStorage::id(&source))
        .expect("source must receive a gradient");
    assert_eq!(
        read_f32(gsrc),
        vec![0.0, 1.0, 1.0],
        "only the last write per destination earns the cotangent"
    );
    assert!(
        grads.get(TapeStorage::id(&index)).is_none(),
        "the integer index operand must stay off the tape, as on CPU"
    );
}

#[test]
fn scatter_refuses_an_out_of_range_index_like_cuda() {
    require_wgpu();
    // Slot 4 does not exist in a length-4 axis: CUDA's `launch_scatter`
    // raises `InvalidInput` here, and this backend mirrors that rather
    // than CPU/Metal's silent drop.
    let input = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let index = upload_i64(&[4], &[1]);
    let source = upload_f32(&[7.0], &[1]);
    let error = match run_ternary::<op::Scatter, _>(
        &input,
        &index,
        &source,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
    ) {
        Ok(_) => panic!("an out-of-range scatter index must be refused"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            CanonicalError::Backend(BackendError::InvalidInput {
                operation: OperationKind::Scatter,
                ..
            })
        ),
        "the refusal must stay the typed InvalidInput CUDA raises, got {error:?}"
    );
    assert!(
        format!("{error:?}").contains("index out of bounds"),
        "the refusal must carry CUDA's reason, got {error:?}"
    );
}

#[test]
fn scatter_refuses_reject_duplicates_like_cpu() {
    require_wgpu();
    let input = upload_f32(&[10.0, 20.0], &[2]);
    let index = upload_i64(&[0, 0], &[2]);
    let source = upload_f32(&[1.0, 2.0], &[2]);
    let error = match run_ternary::<op::Scatter, _>(
        &input,
        &index,
        &source,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::Reject,
        },
    ) {
        Ok(_) => panic!("Reject is not a last-write-wins rule"),
        Err(error) => error,
    };
    assert!(
        format!("{error:?}").contains("last-write-wins and cannot reject duplicate indices"),
        "refusal must carry CPU's wording, got {error:?}"
    );
}

// ── scatter_add ─────────────────────────────────────────────────────────────

#[test]
fn scatter_add_accumulates_in_order_and_drops_out_of_range_targets() {
    require_wgpu();
    // CUDA's own fixture: duplicate destinations sum (1 + 2 at slot 0);
    // target 9 is outside a length-3 axis and drops, so the 100 lands
    // nowhere — the ordered kernel and CPU agree here.
    let input = upload_f32(&[0.0, 0.0, 0.0], &[3]);
    let index = upload_i64(&[0, 0, 2, 9], &[4]);
    let source = upload_f32(&[1.0, 2.0, 4.0, 100.0], &[4]);
    let (out, recorded) = run_ternary::<op::ScatterAdd, _>(
        &input,
        &index,
        &source,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::Accumulate,
        },
    )
    .expect("scatter_add must execute");

    assert_eq!(read_f32(&out), vec![3.0, 0.0, 4.0]);
    assert_eq!(recorded, 1);

    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("scatter_add backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    assert_eq!(
        read_f32(gin),
        vec![1.0, 1.0, 1.0],
        "addition displaces nothing, so the input cotangent passes through"
    );
    let gsrc = grads
        .get(TapeStorage::id(&source))
        .expect("source must receive a gradient");
    assert_eq!(
        read_f32(gsrc),
        vec![1.0, 1.0, 1.0, 0.0],
        "every surviving write takes the cotangent; the dropped one takes zero"
    );
    assert!(
        grads.get(TapeStorage::id(&index)).is_none(),
        "the integer index operand must stay off the tape, as on CPU"
    );
}

#[test]
fn scatter_add_refuses_last_write_wins_with_the_cpus_wording() {
    require_wgpu();
    let input = upload_f32(&[0.0, 0.0], &[2]);
    let index = upload_i64(&[0, 0], &[2]);
    let source = upload_f32(&[1.0, 2.0], &[2]);
    let error = match run_ternary::<op::ScatterAdd, _>(
        &input,
        &index,
        &source,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
    ) {
        Ok(_) => panic!("last-write-wins is not a scatter_add rule"),
        Err(error) => error,
    };
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("scatter_add accumulates duplicate indices and implements no other rule")
            && rendered.contains("use scatter for last-write-wins"),
        "refusal must carry CPU's wording, got {rendered}"
    );
}

// ── masked_fill ─────────────────────────────────────────────────────────────

#[test]
fn masked_fill_writes_the_constant_under_the_mask_and_zeros_its_grad() {
    require_wgpu();
    let input = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    let mask = upload_bool(&[true, false, true, false, true, false], &[6]);

    let (out, recorded) =
        run_mixed::<op::MaskedFill, _>(&[&input], &[], &[&mask], ScalarAttributes { value: -1.0 })
            .expect("masked_fill must execute");

    assert_eq!(read_f32(&out), vec![-1.0, 2.0, -1.0, 4.0, -1.0, 6.0]);
    assert_eq!(recorded, 1, "masked_fill pushes ONE TapeEntry (input only)");

    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("masked_fill backward must run");
    let gin = grads
        .get(TapeStorage::id(&input))
        .expect("input must receive a gradient");
    assert_eq!(
        read_f32(gin),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "positions under a true mask receive zero cotangent"
    );
}

#[test]
fn masked_fill_broadcasts_a_rank_deficit_mask_without_enlarging() {
    require_wgpu();
    // mask [1,3] right-aligned into input [2,3] (#100 broadcast-into rule).
    let input = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mask = upload_bool(&[true, false, true], &[1, 3]);

    let (out, _) =
        run_mixed::<op::MaskedFill, _>(&[&input], &[], &[&mask], ScalarAttributes { value: 9.0 })
            .expect("a broadcast mask must execute");

    // Both rows take the same mask (broadcast along axis 0).
    assert_eq!(
        read_f32(&out),
        vec![9.0, 2.0, 9.0, 9.0, 5.0, 9.0],
        "broadcast must stretch the mask, not refuse it or enlarge the output"
    );
}

// ── where_cond ──────────────────────────────────────────────────────────────

#[test]
fn where_cond_selects_by_mask_and_splits_the_cotangent() {
    require_wgpu();
    let mask = upload_bool(&[true, false, true, false], &[4]);
    let on_true = upload_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let on_false = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[4]);

    let (out, recorded) =
        run_mixed::<op::WhereCond, _>(&[], &[], &[&mask, &on_true, &on_false], NoAttributes)
            .expect("where_cond must execute");

    assert_eq!(read_f32(&out), vec![1.0, 20.0, 3.0, 40.0]);
    assert_eq!(
        recorded, 1,
        "where_cond pushes ONE entry naming the broadcasted value ids"
    );

    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("where_cond backward must run");
    let gt = grads
        .get(TapeStorage::id(&on_true))
        .expect("on_true must receive a gradient");
    let gf = grads
        .get(TapeStorage::id(&on_false))
        .expect("on_false must receive a gradient");
    assert_eq!(read_f32(gt), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(read_f32(gf), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn where_cond_broadcasts_a_small_mask_over_bigger_values() {
    require_wgpu();
    // mask [1,2] right-aligned over values [2,2]: both rows take [T, F].
    // out[0] = [on_true[0], on_false[1]] = [1, 20]
    // out[1] = [on_true[2], on_false[3]] = [3, 40]
    let mask = upload_bool(&[true, false], &[1, 2]);
    let on_true = upload_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let on_false = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    let (out, _) =
        run_mixed::<op::WhereCond, _>(&[], &[], &[&mask, &on_true, &on_false], NoAttributes)
            .expect("where_cond with a broadcast mask must execute");

    assert_eq!(
        read_f32(&out),
        vec![1.0, 20.0, 3.0, 40.0],
        "broadcast mask stretches along axis 0 without enlarging the values"
    );
}

// ── storage dtype round-trips (widened Storage / TensorFromBytes rows) ─────

#[test]
fn bool_bytes_round_trip_through_physical_f32_storage() {
    require_wgpu();
    let flags = [true, false, true, true, false];
    let mask = upload_bool(&flags, &[5]);
    let packed = <TestBackend as HostInterop>::to_bytes::<bool>(&mask)
        .expect("bool to_bytes must pack 0/1 bytes");
    assert_eq!(packed, vec![1, 0, 1, 1, 0]);

    let ints = <TestBackend as HostReadback>::int_to_vec1::<bool>(&mask)
        .expect("bool int readback must succeed");
    assert_eq!(ints, vec![1, 0, 1, 1, 0]);

    // Non-0/1 physical values must be refused rather than coerced.
    let bad_bytes = [2u8; 5];
    assert!(
        <TestBackend as HostInterop>::from_bytes::<bool>(
            &bad_bytes,
            &[5],
            DTypeId::Bool.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_err(),
        "from_bytes must refuse a byte that is neither 0 nor 1"
    );
}

#[test]
fn integer_index_bytes_round_trip_at_their_own_widths() {
    require_wgpu();
    for (dtype, bytes, shape) in [
        (
            DTypeId::I64.descriptor(),
            vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            vec![2usize],
        ),
        (
            DTypeId::U32.descriptor(),
            vec![1, 0, 0, 0, 2, 0, 0, 0],
            vec![2],
        ),
        (DTypeId::U8.descriptor(), vec![1, 2], vec![2]),
    ] {
        let storage = <TestBackend as HostInterop>::from_bytes::<i64>(
            &bytes,
            &shape,
            dtype,
            &DeviceId::wgpu(0),
        )
        .unwrap_or_else(|e| panic!("from_bytes({dtype:?}) must succeed: {e}"));
        let round = <TestBackend as HostInterop>::to_bytes::<i64>(&storage)
            .unwrap_or_else(|e| panic!("to_bytes({dtype:?}) must succeed: {e}"));
        assert_eq!(
            round, bytes,
            "round-trip must be byte-identical for {dtype:?}"
        );
    }
}
