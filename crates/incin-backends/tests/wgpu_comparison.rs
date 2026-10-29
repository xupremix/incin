//! WGPU comparison and logical ops: the nine bool-output identities (#91).
//!
//! Fail-closed contract: every capability row this suite exercises has an
//! `Execute` impl *and* a real runtime CPU-twin test here — the six
//! numeric comparisons and the three logicals against the same recipes
//! `cpu::ops::shape_ops::elementwise_cmp` and the CPU logical executors
//! run, a broadcast case that would only pass if the raw (non-recording)
//! path actually launches, and a `cmp → masked_fill` integration that
//! proves the `Bool` label survives into the one consumer that demands
//! it. Bool outputs round-trip through `to_bytes`/`int_to_vec1` so the
//! physical-f32-under-a-Bool-label representation is proven rather than
//! merely declared, and the tape-depth assertions pin the "no entry a
//! backward walk could never reach" rule from `wgpu/backend/compare.rs`.
//!
//! Requires a WGPU adapter for the runtime half:
//! `cargo test -p incin-backends --features wgpu --test wgpu_comparison`.
//! The capability-table half answers from the compiled registry alone.
#![cfg(feature = "wgpu")]

use incin_backends::capability::{WGPU_CAPABILITIES, support};
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{NoAttributes, ScalarAttributes};
use incin_core::exec::{
    CapabilityQuery, OperationIdentity, SupportLevel, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
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

/// Bool operand uploaded as one byte per element (0/1); physical `f32`
/// expansion is `from_bytes`'s job.
fn upload_bool(values: &[bool], shape: &[usize]) -> BoolStorage {
    let bytes: Vec<u8> = values.iter().map(|&b| u8::from(b)).collect();
    <TestBackend as HostInterop>::from_bytes::<bool>(
        &bytes,
        shape,
        DTypeId::Bool.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading a bool operand must succeed")
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading an f32 buffer back must succeed")
}

/// Packed 0/1 bytes of a `Bool`-labeled result: proves both the values
/// and the physical-f32 encoding round-trip through `to_bytes`.
fn read_bool(storage: &BoolStorage) -> Vec<u8> {
    <TestBackend as HostInterop>::to_bytes::<bool>(storage)
        .expect("bool to_bytes must pack 0/1 bytes")
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
    let context = incin_core::exec::ExecutionContext::new(TestBackend::default());
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

// ── the six numeric comparisons ─────────────────────────────────────────────

#[test]
fn comparisons_produce_bool_storage_matching_cpu_and_push_no_tape() {
    require_wgpu();
    // CPU's own fixture: enough disagreement that every mode has both a
    // true and a false answer. a=[1,2,3,4] vs b=[1,4,3,0], evaluated the
    // way `cpu::ops::shape_ops::elementwise_cmp`'s closures do.
    let a = upload_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = upload_f32(&[1.0, 4.0, 3.0, 0.0], &[4]);

    macro_rules! cmp_case {
        ($op:ident, $name:literal, $want:expr) => {{
            let (out, recorded) = run_mixed::<op::$op, _>(&[&a, &b], &[], &[], NoAttributes)
                .expect(concat!($name, " must execute"));
            assert_eq!(
                out.metadata().dtype(),
                DTypeId::Bool.descriptor(),
                concat!(
                    $name,
                    ": the catalog types the result Bool, so the storage label must \
                     say Bool over the physical f32 encoding"
                )
            );
            assert_eq!(
                read_bool(&out),
                $want.to_vec(),
                concat!($name, ": CPU parity on [1,2,3,4] vs [1,4,3,0]")
            );
            assert_eq!(
                recorded, 0,
                concat!(
                    $name,
                    " pushes NO tape entry: descriptor_training resolves it false on \
                     every backend, so a dead entry would only be a lie the tape tells"
                )
            );
        }};
    }

    cmp_case!(CmpEq, "cmp_eq", [1u8, 0, 1, 0]);
    cmp_case!(CmpNe, "cmp_ne", [0u8, 1, 0, 1]);
    cmp_case!(CmpLt, "cmp_lt", [0u8, 1, 0, 0]);
    cmp_case!(CmpLe, "cmp_le", [1u8, 1, 1, 0]);
    cmp_case!(CmpGt, "cmp_gt", [0u8, 0, 0, 1]);
    // 4 >= 0 is the true half this fixture exists for.
    cmp_case!(CmpGe, "cmp_ge", [1u8, 0, 1, 1]);
}

#[test]
fn comparisons_broadcast_rank_deficit_operands_without_recording() {
    require_wgpu();
    // a [2,3] against b [3]: b stretches along axis 0. If the stretch
    // rode the tape-recording `broadcast_storage` instead of the raw
    // form, `recorded` below would be 1, not 0.
    let a = upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = upload_f32(&[6.0, 5.0, 4.0], &[3]);

    let (lt, recorded_lt) =
        run_mixed::<op::CmpLt, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_lt must broadcast");
    // [1<6, 2<5, 3<4, 4<6, 5<5, 6<4]
    assert_eq!(read_bool(&lt), vec![1, 1, 1, 1, 0, 0]);
    assert_eq!(
        recorded_lt, 0,
        "the broadcast behind a comparison must ride the RAW path -- a recording \
         entry here would be unreachable from any backward walk"
    );

    let (ge, recorded_ge) =
        run_mixed::<op::CmpGe, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_ge must broadcast");
    // [1>=6, 2>=5, 3>=4, 4>=6, 5>=5, 6>=4]
    assert_eq!(read_bool(&ge), vec![0, 0, 0, 0, 1, 1]);
    assert_eq!(recorded_ge, 0);
}

#[test]
fn nan_comparisons_agree_with_cpu_f64_semantics() {
    require_wgpu();
    // CPU evaluates through its f64 accessor; WGSL compares f32. Both
    // IEEE-754 answers agree on every NaN question: == is false, != is
    // true, and every ordered compare is false.
    let a = upload_f32(&[f32::NAN, 1.0], &[2]);
    let b = upload_f32(&[f32::NAN, 1.0], &[2]);

    let (eq, _) =
        run_mixed::<op::CmpEq, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_eq must execute");
    assert_eq!(
        read_bool(&eq),
        vec![0, 1],
        "NaN == NaN is false, 1.0 == 1.0 is true"
    );

    let (ne, _) =
        run_mixed::<op::CmpNe, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_ne must execute");
    assert_eq!(read_bool(&ne), vec![1, 0], "NaN != NaN is true");

    let (lt, _) =
        run_mixed::<op::CmpLt, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_lt must execute");
    assert_eq!(
        read_bool(&lt),
        vec![0, 0],
        "NaN < NaN and 1.0 < 1.0 are both false"
    );

    let (gt, _) =
        run_mixed::<op::CmpGt, _>(&[&a], &[&b], &[], NoAttributes).expect("cmp_gt must execute");
    assert_eq!(
        read_bool(&gt),
        vec![0, 0],
        "NaN > NaN and 1.0 > 1.0 are both false"
    );
}

// ── the three logicals ──────────────────────────────────────────────────────

#[test]
fn logical_ops_consume_and_produce_bool_storage_without_recording() {
    require_wgpu();
    let x = upload_bool(&[true, false, true, true], &[4]);
    let y = upload_bool(&[true, true, false, false], &[4]);

    let (and, recorded_and) = run_mixed::<op::LogicalAnd, _>(&[], &[], &[&x, &y], NoAttributes)
        .expect("logical_and must execute");
    assert_eq!(and.metadata().dtype(), DTypeId::Bool.descriptor());
    assert_eq!(read_bool(&and), vec![1, 0, 0, 0]);
    assert_eq!(
        recorded_and, 0,
        "logical ops push no tape entry, like CPU/CUDA"
    );
    let (or, recorded_or) = run_mixed::<op::LogicalOr, _>(&[], &[], &[&x, &y], NoAttributes)
        .expect("logical_or must execute");
    // x=[T,F,T,T] or y=[T,T,F,F]: every position is true (position 1 is
    // saved by y, positions 2/3 by x).
    assert_eq!(read_bool(&or), vec![1, 1, 1, 1]);
    assert_eq!(recorded_or, 0);

    let (not, recorded_not) = run_mixed::<op::LogicalNot, _>(&[], &[], &[&x], NoAttributes)
        .expect("logical_not must execute");
    assert_eq!(not.metadata().dtype(), DTypeId::Bool.descriptor());
    assert_eq!(read_bool(&not), vec![0, 1, 0, 0]);
    assert_eq!(recorded_not, 0);
}

// ── integration: the Bool label into the one consumer that demands it ──────

#[test]
fn a_comparison_mask_feeds_masked_fill_end_to_end() {
    require_wgpu();
    // The result of `cmp_lt` must carry a `Bool` label dense enough for
    // `masked_fill`'s descriptor contract (mask slot must be bool) and
    // for the row's `F32_AND_BOOL` admission -- otherwise the pipeline
    // that motivates comparisons on an accelerator is unreachable.
    let input = upload_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let bound = upload_f32(&[2.0, 2.0, 2.0, 2.0], &[4]);

    let (mask, cmp_recorded) = run_mixed::<op::CmpLt, _>(&[&input], &[&bound], &[], NoAttributes)
        .expect("the comparison must execute");
    assert_eq!(cmp_recorded, 0);
    assert_eq!(read_bool(&mask), vec![1, 0, 0, 0]);

    let (out, fill_recorded) =
        run_mixed::<op::MaskedFill, _>(&[&input], &[], &[&mask], ScalarAttributes { value: -9.0 })
            .expect("masked_fill must accept the comparison's Bool-labeled result");
    assert_eq!(
        read_f32(&out),
        vec![-9.0, 2.0, 3.0, 4.0],
        "positions under a true comparison mask take the constant"
    );
    assert_eq!(
        fill_recorded, 1,
        "only masked_fill pushes an entry; the comparison feeding it never does"
    );
}

// ── named refusals: wrong operand dtypes end to end ────────────────────────

#[test]
fn wrong_operand_dtypes_are_refused_rather_than_reinterpreted() {
    require_wgpu();
    // A bool pair through a row that declares F32_ONLY: admission must
    // refuse it by dtype before any kernel is reached, rather than let
    // the shader read the encoding as raw floats it was never given.
    let x = upload_bool(&[true, false], &[2]);
    let y = upload_bool(&[false, true], &[2]);
    let refused = run_mixed::<op::CmpEq, _>(&[], &[], &[&x, &y], NoAttributes);
    assert!(
        refused.is_err(),
        "cmp_eq over bool operands must be refused by the F32_ONLY row, not executed"
    );

    // An f32 pair through the logical contract: the descriptor itself
    // refuses a non-bool operand (`inference.rs`: "logical operations
    // require boolean inputs") before the BOOL_ONLY row is even asked.
    let a = upload_f32(&[1.0, 0.0], &[2]);
    let b = upload_f32(&[0.0, 1.0], &[2]);
    let refused = run_mixed::<op::LogicalAnd, _>(&[&a], &[&b], &[], NoAttributes);
    assert!(
        refused.is_err(),
        "logical_and over f32 operands must be refused, not executed"
    );
}

// ── capability-table half: answers from the compiled registry alone ───────

fn query(operation: K, dtype: DTypeId, rank: usize, training: bool) -> CapabilityQuery {
    CapabilityQuery {
        operation: OperationIdentity::Builtin(operation),
        dtype: dtype.descriptor(),
        layout: incin_core::exec::LayoutClass::Contiguous,
        rank,
        training,
        math_mode: incin_core::exec::MathMode::Precise,
    }
}

fn wgpu_rule(operation: K) -> &'static incin_core::exec::CapabilityRule {
    WGPU_CAPABILITIES
        .iter()
        .find(|rule| rule.operation == operation)
        .unwrap_or_else(|| panic!("WGPU declares no capability row for {operation:?}"))
}

#[test]
fn comparison_rows_admit_f32_operands_and_refuse_everything_else() {
    for operation in [K::CmpEq, K::CmpNe, K::CmpLt, K::CmpLe, K::CmpGt, K::CmpGe] {
        let rule = wgpu_rule(operation);
        assert!(
            !matches!(
                support(
                    DeviceKind::Wgpu,
                    &query(operation, DTypeId::F32, rule.min_rank, rule.training)
                ),
                SupportLevel::Unsupported(_)
            ),
            "WGPU: {operation:?} must admit its f32 operand"
        );
        for dtype in [
            DTypeId::Bool,
            DTypeId::I64,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::F64,
        ] {
            assert!(
                matches!(
                    support(
                        DeviceKind::Wgpu,
                        &query(operation, dtype, rule.min_rank, rule.training)
                    ),
                    SupportLevel::Unsupported(UnsupportedReason::DType { .. })
                ),
                "WGPU: {operation:?} must refuse a {dtype:?} operand by dtype"
            );
        }
        // The row's own training claim: comparisons resolve `false`, so
        // a training-mode invocation must fail loudly at admission
        // rather than run and deliver no gradient.
        assert!(
            matches!(
                support(
                    DeviceKind::Wgpu,
                    &query(operation, DTypeId::F32, rule.min_rank, true)
                ),
                SupportLevel::Unsupported(UnsupportedReason::Training { .. })
            ),
            "WGPU: {operation:?} must refuse a training-mode invocation"
        );
        // Rank ceiling: `accelerator_max_rank` caps the shape-kernel
        // broadcast at 6 on this backend.
        assert!(
            !matches!(
                support(DeviceKind::Wgpu, &query(operation, DTypeId::F32, 6, false)),
                SupportLevel::Unsupported(_)
            ),
            "WGPU: {operation:?} must admit rank 6"
        );
        assert!(
            matches!(
                support(DeviceKind::Wgpu, &query(operation, DTypeId::F32, 7, false)),
                SupportLevel::Unsupported(UnsupportedReason::Rank { .. })
            ),
            "WGPU: {operation:?} must refuse rank 7 (shape.wgsl packs rank 6)"
        );
    }
}

#[test]
fn logical_rows_admit_bool_operands_and_refuse_the_rest() {
    for operation in [K::LogicalAnd, K::LogicalOr, K::LogicalNot] {
        let rule = wgpu_rule(operation);
        assert!(
            !matches!(
                support(
                    DeviceKind::Wgpu,
                    &query(operation, DTypeId::Bool, rule.min_rank, rule.training)
                ),
                SupportLevel::Unsupported(_)
            ),
            "WGPU: {operation:?} must admit its bool operand"
        );
        for dtype in [
            DTypeId::F32,
            DTypeId::I64,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::F64,
        ] {
            assert!(
                matches!(
                    support(
                        DeviceKind::Wgpu,
                        &query(operation, dtype, rule.min_rank, rule.training)
                    ),
                    SupportLevel::Unsupported(UnsupportedReason::DType { .. })
                ),
                "WGPU: {operation:?} must refuse a {dtype:?} operand by dtype"
            );
        }
        assert!(
            matches!(
                support(
                    DeviceKind::Wgpu,
                    &query(operation, DTypeId::Bool, rule.min_rank, true)
                ),
                SupportLevel::Unsupported(UnsupportedReason::Training { .. })
            ),
            "WGPU: {operation:?} must refuse a training-mode invocation"
        );
    }
}
