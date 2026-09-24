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
use incin_core::exec::catalog::{AxisAttributes, NoAttributes, ScalarAttributes};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
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
