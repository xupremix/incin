//! Closing the remaining narrow CUDA dtype capability rows: admission pins
//! plus hardware execution of the newly widened identities.
//!
//! Two halves:
//! 1. Compile-time admission tests (no hardware) that `capability::support`
//!    answers from the compiled table alone — the widened rows must admit
//!    every dtype they claim, and the honest-f32 rows this session left
//!    narrow must still refuse half precision by dtype (not by a missing
//!    row, and not by a training flag the dtype check would shadow).
//! 2. `#[ignore]`d hardware tests that drive each widened identity through
//!    `dispatch::execute` at a dtype the row newly admits, asserting exact
//!    values against hand-computed references and at least one tape entry
//!    where the row claims `training = true`.
//!
//! Requires a GPU for the second half:
//! `cargo test -p incin-backends --features cuda --test cuda_dtype_rows -- --ignored`.
#![cfg(feature = "cuda")]

use half::{bf16, f16};
use incin_backends::capability::{CUDA_CAPABILITIES, support};
use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{download_bytes, require_cuda, upload_bytes},
};
use incin_core::backend_authoring::{Execute, StorageBackend};
use incin_core::exec::catalog::{
    AxisAttributes, FlattenAttributes, LerpAttributes, NoAttributes, PixelShuffleAttributes,
    SliceAttributes,
};
use incin_core::exec::{
    CanonicalError, CanonicalOperation, CapabilityQuery, ExecutionContext, LayoutClass,
    OperationIdentity, SupportLevel, TensorHandle, UnsupportedReason, dispatch, op,
};
use incin_core::prelude::CudaN;
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Finds the first CUDA rule for `operation` (legacy rows render first, so
/// this is the standalone wide row where one exists, else the group row).
fn cuda_rule(operation: K) -> &'static incin_core::exec::CapabilityRule {
    CUDA_CAPABILITIES
        .iter()
        .find(|rule| rule.operation == operation)
        .unwrap_or_else(|| panic!("CUDA declares no capability row for {operation:?}"))
}

/// Builds a query from the first matching rule's own bounds, with an
/// explicit dtype/layout/training so each axis can be pinned independently.
fn query(
    operation: K,
    dtype: DTypeId,
    layout: LayoutClass,
    rank: usize,
    training: bool,
) -> SupportLevel {
    let rule = cuda_rule(operation);
    support(
        DeviceKind::Cuda,
        &CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout,
            rank,
            training,
            math_mode: rule.math_modes[0],
        },
    )
}

fn admits(operation: K, dtype: DTypeId, layout: LayoutClass, training: bool) -> bool {
    let rule = cuda_rule(operation);
    let rank = rule.min_rank.max(1);
    !matches!(
        query(operation, dtype, layout, rank, training),
        SupportLevel::Unsupported(_)
    )
}

// ---------------------------------------------------------------------------
// Compile-time admission tests (no hardware).
// ---------------------------------------------------------------------------

/// The four elementwise identities moved from the honest-f32 `native_tensor`
/// group into `elementwise`: FLOAT_DTYPES on CUDA_LAYOUTS, training true.
#[test]
fn elementwise_widened_rows_admit_half_and_double_on_both_layouts() {
    for operation in [K::Maximum, K::Minimum, K::AbsDiff, K::Lerp] {
        for dtype in [DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64] {
            for layout in [LayoutClass::Contiguous, LayoutClass::Strided] {
                assert!(
                    admits(operation, dtype, layout, true),
                    "CUDA: {operation:?} must admit {dtype:?} at {layout:?} \
                     with training=true (elementwise group), got {:?}",
                    query(
                        operation,
                        dtype,
                        layout,
                        cuda_rule(operation).min_rank.max(1),
                        true
                    ),
                );
            }
        }
    }
}

/// The eight composed shape identities plus PixelShuffle/Unfold/TensorToBytes:
/// CUDA_BOOL_SAFE_STORAGE_DTYPES (i64, bf16, f16, f32, f64, bool) contiguous,
/// training true for the composed/native training rows and false for readback.
#[test]
fn composed_and_movement_rows_admit_the_dense_storage_set() {
    for operation in [
        K::FlattenExact,
        K::SqueezeExact,
        K::UnsqueezeExact,
        K::StackExact,
        K::SliceExact,
        K::BroadcastLeft,
        K::Chunk,
        K::Split,
        K::PixelShuffle,
        K::Unfold,
    ] {
        for dtype in [
            DTypeId::I64,
            DTypeId::BF16,
            DTypeId::F16,
            DTypeId::F32,
            DTypeId::F64,
            DTypeId::Bool,
        ] {
            assert!(
                admits(operation, dtype, LayoutClass::Contiguous, true),
                "CUDA: {operation:?} must admit {dtype:?} contiguous with \
                 training=true, got {:?}",
                query(
                    operation,
                    dtype,
                    LayoutClass::Contiguous,
                    cuda_rule(operation).min_rank.max(1),
                    true
                ),
            );
        }
    }
}

/// `TensorToBytes` widens through its standalone row; the four `ToHost*`
/// identities stay honest-f32 through the group row.
#[test]
fn tensor_to_bytes_admits_the_dense_set_while_tohost_refuses_half() {
    for dtype in [
        DTypeId::I64,
        DTypeId::BF16,
        DTypeId::F16,
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::Bool,
    ] {
        assert!(
            admits(K::TensorToBytes, dtype, LayoutClass::Contiguous, false),
            "CUDA: TensorToBytes must admit {dtype:?} contiguous (standalone \
             wide row), got {:?}",
            query(
                K::TensorToBytes,
                dtype,
                LayoutClass::Contiguous,
                cuda_rule(K::TensorToBytes).min_rank.max(1),
                false
            ),
        );
    }
    for operation in [
        K::ToHostFloatVec,
        K::ToHostFloatScalar,
        K::ToHostIntVec,
        K::ToHostIntScalar,
    ] {
        assert!(
            !admits(operation, DTypeId::F16, LayoutClass::Contiguous, false),
            "CUDA: {operation:?} must still refuse f16 by dtype (cuda_require_f32 \
             in contract.rs), got {:?}",
            query(
                operation,
                DTypeId::F16,
                LayoutClass::Contiguous,
                cuda_rule(operation).min_rank.max(1),
                false
            ),
        );
        assert!(
            admits(operation, DTypeId::F32, LayoutClass::Contiguous, false),
            "CUDA: {operation:?} must still admit f32 so the f16 refusal is a \
             dtype answer, not a missing row",
        );
    }
}

/// Refusal pins: the honest-f32 native_tensor members this session left
/// narrow must refuse `f16` by `UnsupportedReason::DType`. The dtype flag is
/// set before the training flag in `CapabilityRegistry::support`, so the
/// reason is DType regardless of the training bit; all eight are single-rule
/// honest-f32 rows, so the first-match `cuda_rule` is the whole story.
#[test]
fn honest_f32_rows_refuse_f16_by_dtype() {
    for operation in [
        K::CmpEq,
        K::Cumsum,
        K::ArgMax,
        K::TopK,
        K::Triu,
        K::Pad,
        K::RepeatInterleave,
        K::VarianceAll,
    ] {
        let rule = cuda_rule(operation);
        let level = query(
            operation,
            DTypeId::F16,
            LayoutClass::Contiguous,
            rule.min_rank.max(1),
            false,
        );
        assert!(
            matches!(
                level,
                SupportLevel::Unsupported(UnsupportedReason::DType { .. })
            ),
            "CUDA: {operation:?} must refuse f16 by dtype (honest-f32 kernel), \
             got {level:?}",
        );
        assert!(
            admits(operation, DTypeId::F32, LayoutClass::Contiguous, false),
            "CUDA: {operation:?} must still admit f32 so the f16 refusal is a \
             dtype answer, not a missing row",
        );
    }
}

// ---------------------------------------------------------------------------
// Hardware runtime tests (ignored without a GPU).
// ---------------------------------------------------------------------------

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn decode_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect()
}

/// Two inputs through `dispatch`, returning the result alongside tape depth
/// delta. `K` is the operand dtype the handles present to admission.
fn run2<O, A, K>(
    context: &ExecutionContext<TestBackend>,
    first: &TestStorage,
    second: &TestStorage,
    attributes: A,
) -> (Result<TestStorage, CanonicalError>, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
    K: DType,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[
            TensorHandle::from_storage::<TestBackend, K, _>(first),
            TensorHandle::from_storage::<TestBackend, K, _>(second),
        ],
    );
    (out, tape_depth() - before)
}

/// One input through `dispatch`.
fn run1<O, A, K>(
    context: &ExecutionContext<TestBackend>,
    input: &TestStorage,
    attributes: A,
) -> (Result<TestStorage, CanonicalError>, usize)
where
    O: CanonicalOperation<Attributes = A>,
    TestBackend: Execute<O, Output = TestStorage>,
    K: DType,
{
    let before = tape_depth();
    let out = dispatch::execute::<O, TestBackend>(
        context,
        attributes,
        &[TensorHandle::from_storage::<TestBackend, K, _>(input)],
    );
    (out, tape_depth() - before)
}

#[test]
#[ignore = "requires CUDA hardware"]
fn maximum_takes_elementwise_max_at_f16() {
    // [1,5,3] vs [4,2,7] → [4,5,7]. Every value is a small integer exactly
    // representable in f16, so the assertion is equality — a promotion to f32
    // or a wrong entry point would change the output dtype or the bytes.
    require_cuda();
    let lhs = upload_bytes(&[3], DTypeId::F16, &f16_bytes(&[1.0, 5.0, 3.0]));
    let rhs = upload_bytes(&[3], DTypeId::F16, &f16_bytes(&[4.0, 2.0, 7.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::Maximum, _, f16>(&context, &lhs, &rhs, NoAttributes);
    let out = out.expect("an f16 maximum pair is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode Maximum must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![3], "f16 maximum output shape");
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![4.0, 5.0, 7.0],
        "maximum must take the elementwise max at f16"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn minimum_bf16_and_abs_diff_f64_execute_exactly() {
    // Minimum at bf16: [1,5,3] vs [4,2,7] → [1,2,3] (bf16 keeps integers to
    // 256 exact). AbsDiff at f64: |[1,5,3] - [4,2,7]| → [3,3,4]. Two dtypes
    // the honest-f32 native_tensor group used to refuse, now on the
    // elementwise row.
    require_cuda();
    let lhs_b = upload_bytes(&[3], DTypeId::BF16, &bf16_bytes(&[1.0, 5.0, 3.0]));
    let rhs_b = upload_bytes(&[3], DTypeId::BF16, &bf16_bytes(&[4.0, 2.0, 7.0]));
    let lhs_d = upload_bytes(&[3], DTypeId::F64, &f64_bytes(&[1.0, 5.0, 3.0]));
    let rhs_d = upload_bytes(&[3], DTypeId::F64, &f64_bytes(&[4.0, 2.0, 7.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::Minimum, _, bf16>(&context, &lhs_b, &rhs_b, NoAttributes);
    let out = out.expect("a bf16 minimum pair is advertised and must launch");
    assert!(recorded >= 1, "training-mode Minimum must record tape");
    assert_eq!(out.dtype, DTypeId::BF16.descriptor());
    assert_eq!(
        decode_bf16(&download_bytes(&out)),
        vec![1.0, 2.0, 3.0],
        "minimum must take the elementwise min at bf16"
    );

    let (out, recorded) = run2::<op::AbsDiff, _, f64>(&context, &lhs_d, &rhs_d, NoAttributes);
    let out = out.expect("an f64 abs_diff pair is advertised and must launch");
    assert!(recorded >= 1, "training-mode AbsDiff must record tape");
    assert_eq!(out.dtype, DTypeId::F64.descriptor());
    assert_eq!(
        decode_f64(&download_bytes(&out)),
        vec![3.0, 3.0, 4.0],
        "abs_diff must take the absolute difference at f64"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn lerp_interpolates_at_f16() {
    // start=[0,4,8], end=[4,8,12], weight=0.25 → [1,5,9]. Small integers
    // stay exact in f16 through the sub/mul/add composition.
    require_cuda();
    let start = upload_bytes(&[3], DTypeId::F16, &f16_bytes(&[0.0, 4.0, 8.0]));
    let end = upload_bytes(&[3], DTypeId::F16, &f16_bytes(&[4.0, 8.0, 12.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) =
        run2::<op::Lerp, _, f16>(&context, &start, &end, LerpAttributes { weight: 0.25 });
    let out = out.expect("an f16 lerp pair is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode Lerp must record tape, recorded {recorded}"
    );
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![1.0, 5.0, 9.0],
        "lerp must interpolate at f16"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn flatten_and_slice_execute_at_f16() {
    // Flatten: [2,6] of 1..12 → flat [12], byte-identical (metadata-only
    // reshape pushes one tape entry). Slice: [4] of 1..4 take [1,3) → [2,3].
    // Both are composed identities whose rewrite targets are the measured
    // wide movement kernels.
    require_cuda();
    let flat_in = upload_bytes(
        &[2, 6],
        DTypeId::F16,
        &f16_bytes(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]),
    );
    let slice_in = upload_bytes(&[4], DTypeId::F16, &f16_bytes(&[1.0, 2.0, 3.0, 4.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run1::<op::FlattenExact, _, f16>(
        &context,
        &flat_in,
        FlattenAttributes {
            start_axis: 0,
            end_axis: 1,
        },
    );
    let out = out.expect("an f16 flatten is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode FlattenExact must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![12], "flatten must collapse [2,6] to [12]");
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        (1..=12).map(|v| v as f32).collect::<Vec<_>>(),
        "flatten is a view: the elements must be untouched"
    );

    let (out, recorded) = run1::<op::SliceExact, _, f16>(
        &context,
        &slice_in,
        SliceAttributes {
            ranges: vec![(1, 3)],
        },
    );
    let out = out.expect("an f16 slice is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode SliceExact must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2], "slice [1,3) of a [4] must be length 2");
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![2.0, 3.0],
        "slice must take the half-open window"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn stack_exact_stacks_two_f16_vectors() {
    // Two [2] inputs stacked on axis 0 → [2,2] row-major: [[1,2],[3,4]].
    // Composed from unsqueeze + concat, both of which push tape.
    require_cuda();
    let first = upload_bytes(&[2], DTypeId::F16, &f16_bytes(&[1.0, 2.0]));
    let second = upload_bytes(&[2], DTypeId::F16, &f16_bytes(&[3.0, 4.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) =
        run2::<op::StackExact, _, f16>(&context, &first, &second, AxisAttributes { axis: 0 });
    let out = out.expect("an f16 stack pair is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode StackExact must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2, 2], "stack of two [2] on axis 0 is [2,2]");
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![1.0, 2.0, 3.0, 4.0],
        "stack must join along the new leading axis"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn pixel_shuffle_permutes_channels_into_spatial_at_f16() {
    // Standard depth-to-space, r=2: input [1,4,1,2] (C=4=r², H=1, W=2),
    // values 1..8. out[c, h*r+i, w*r+j] = in[c*r²+i*r+j, h, w], so the
    // output is [1,1,2,4] with flat order [1,3,2,4,5,7,6,8] — a real
    // permutation, not the identity an [1,8,1,1] input would produce.
    // Executor is reshape + three transposes over the measured kernels.
    require_cuda();
    let input = upload_bytes(
        &[1, 4, 1, 2],
        DTypeId::F16,
        &f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run1::<op::PixelShuffle, _, f16>(
        &context,
        &input,
        PixelShuffleAttributes { upscale_factor: 2 },
    );
    let out = out.expect("an f16 pixel_shuffle is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode PixelShuffle must record tape, recorded {recorded}"
    );
    assert_eq!(
        out.shape,
        vec![1, 1, 2, 4],
        "pixel_shuffle r=2 of [1,4,1,2] must be [1,1,2,4]"
    );
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        "pixel_shuffle must permute channels into the spatial axes"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn tensor_to_bytes_round_trips_f16_through_dispatch() {
    // Upload f16 bytes, run op::TensorToBytes through dispatch, and require
    // the returned Vec<u8> to equal the input — a raw device-to-host copy
    // that never reinterprets elements.
    require_cuda();
    let payload = f16_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let storage = upload_bytes(&[4], DTypeId::F16, &payload);

    let context = ExecutionContext::new(TestBackend::new());
    let before = tape_depth();
    let out = dispatch::execute::<op::TensorToBytes, TestBackend>(
        &context,
        NoAttributes,
        &[TensorHandle::from_storage::<TestBackend, f16, _>(&storage)],
    );
    let recorded = tape_depth() - before;
    let bytes = out.expect("an f16 tensor_to_bytes is advertised and must launch");
    assert_eq!(
        recorded, 0,
        "a device-to-host copy records nothing to differentiate"
    );
    assert_eq!(
        bytes, payload,
        "tensor_to_bytes must round-trip f16 bytes unchanged"
    );
}
