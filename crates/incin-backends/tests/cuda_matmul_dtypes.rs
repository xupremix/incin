//! Issue #90: the composed CUDA matmul-family rows at non-`f32` dtypes.
//!
//! `BatchedMatMul`, `Addmm` and `Linear` advertise `bf16`/`f16`/`f32`/`f64`
//! on CUDA via the `composed_reduction` group, and `backend/tests.rs` pins
//! that admission host-side. What no hardware run covered was the composed
//! *executor* path below `f32`: `op::Addmm` had never been launched on CUDA
//! at any dtype, and `op::BatchedMatMul`/`op::Linear` only at `f32`
//! (`cuda_gemm_batched`). Each forward here drives one row at a dtype whose
//! small integers stay exactly representable, so every assertion is equality
//! against hand-computed values - a tolerance window would hide the
//! accumulation and entry-point choice this file exists to observe.
//!
//! The mixed-dtype refusal keeps the other half of the contract: a row
//! states the union of what its operands may carry (one `FLOAT_DTYPES` set
//! for all of them), so the same-dtype equality the row cannot express must
//! fail host-side instead of silently promoting. The default execution
//! policy is `fp32`, under which dispatch's autocast casts nothing, so the
//! mixed pair reaches the executor as given.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_matmul_dtypes -- --ignored`.
#![cfg(feature = "cuda")]

use half::{bf16, f16};
use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{download_bytes, require_cuda, upload_bytes},
};
use incin_core::backend_authoring::{Execute, StorageBackend};
use incin_core::exec::catalog::{AddmmAttributes, LinearAttributes, NoAttributes};
use incin_core::exec::{
    CanonicalError, CanonicalOperation, ExecutionContext, TensorHandle, dispatch, op,
};
use incin_core::prelude::CudaN;
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

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

/// Two inputs through `dispatch`, returning the result alongside how many
/// tape entries the call recorded. `K` is the operand dtype the handles
/// present to admission; the storage itself is the same `CudaStorage` for
/// every `K`.
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

/// Three inputs through `dispatch` (`linear`'s input, weight, bias).
fn run3<O, A, K>(
    context: &ExecutionContext<TestBackend>,
    first: &TestStorage,
    second: &TestStorage,
    third: &TestStorage,
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
            TensorHandle::from_storage::<TestBackend, K, _>(third),
        ],
    );
    (out, tape_depth() - before)
}

#[test]
#[ignore = "requires CUDA hardware"]
fn batched_matmul_multiplies_f16_operands() {
    // [2,2,2] x [2,2,2]: batch 0 multiplies by the identity, batch 1 by
    // 2I, so a batch loop that ran the wrong entry point, dropped a batch
    // or mixed operand batches cannot land on the expected answer. Every
    // value is an integer below 2048, exactly representable in f16, so the
    // assertion is equality against the hand product - the f16 kernel
    // accumulates in f32 under the hood, which these magnitudes cannot
    // distinguish, but a wrong entry point or batch count they can.
    require_cuda();
    let lhs = upload_bytes(
        &[2, 2, 2],
        DTypeId::F16,
        &f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );
    let rhs = upload_bytes(
        &[2, 2, 2],
        DTypeId::F16,
        &f16_bytes(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
    );

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::BatchedMatMul, _, f16>(&context, &lhs, &rhs, NoAttributes);
    let out = out.expect("a homogeneous f16 rank-3 pair is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode BatchedMatMul must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2, 2, 2], "batched f16 product shape");
    assert_eq!(
        out.dtype,
        DTypeId::F16.descriptor(),
        "an f16 pair must produce f16 storage, not a promoted dtype"
    );
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0],
        "batch 0 must hold lhs, batch 1 must hold lhs scaled by 2"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn batched_matmul_multiplies_f64_operands() {
    // Same shapes and values as the f16 test: batch 0 by the identity,
    // batch 1 by 2I. This is the `matmul_batched_f64` entry point, which
    // no prior hardware test had launched; f64 makes any narrowing in the
    // composed path visible as itself.
    require_cuda();
    let lhs = upload_bytes(
        &[2, 2, 2],
        DTypeId::F64,
        &f64_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );
    let rhs = upload_bytes(
        &[2, 2, 2],
        DTypeId::F64,
        &f64_bytes(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
    );

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run2::<op::BatchedMatMul, _, f64>(&context, &lhs, &rhs, NoAttributes);
    let out = out.expect("a homogeneous f64 rank-3 pair is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode BatchedMatMul must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2, 2, 2], "batched f64 product shape");
    assert_eq!(out.dtype, DTypeId::F64.descriptor());
    assert_eq!(
        decode_f64(&download_bytes(&out)),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0],
        "f64 batched product must match the hand computation exactly"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn addmm_fuses_scale_and_addend_in_f16() {
    // The first launch of `op::Addmm` on CUDA at any dtype. Operand order
    // is [mat, lhs, rhs] and the executor computes
    // `beta * mat + alpha * (lhs @ rhs)`; with alpha = beta = 1 that is
    // `[[1,2],[3,4]] + [[58,64],[139,154]]`. Every value is an integer
    // below 2048, exactly representable in f16, so the equality pins the
    // fusion rather than a tolerance window - and exercises mul_scalar and
    // the elementwise add at f16 on the way, which no matmul-family test
    // had done before.
    require_cuda();
    let mat = upload_bytes(&[2, 2], DTypeId::F16, &f16_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let lhs = upload_bytes(
        &[2, 3],
        DTypeId::F16,
        &f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let rhs = upload_bytes(
        &[3, 2],
        DTypeId::F16,
        &f16_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    );

    let context = ExecutionContext::new(TestBackend::new());
    let before = tape_depth();
    let out = dispatch::execute::<op::Addmm, TestBackend>(
        &context,
        AddmmAttributes {
            alpha: 1.0,
            beta: 1.0,
        },
        &[
            TensorHandle::from_storage::<TestBackend, f16, _>(&mat),
            TensorHandle::from_storage::<TestBackend, f16, _>(&lhs),
            TensorHandle::from_storage::<TestBackend, f16, _>(&rhs),
        ],
    );
    let recorded = tape_depth() - before;
    let out = out.expect("a homogeneous f16 addmm is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode Addmm must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2, 2], "addmm f16 result shape");
    assert_eq!(out.dtype, DTypeId::F16.descriptor());
    assert_eq!(
        decode_f16(&download_bytes(&out)),
        vec![59.0, 66.0, 142.0, 158.0],
        "beta * mat + alpha * (lhs @ rhs) must fuse at f16"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn linear_projects_bf16_operands_with_a_bias() {
    // input [2,3] = 1..6, weight [2,3] = [1,0,1,0,1,0], bias = [1,1].
    // CUDA computes `input @ transpose(weight) + bias`, so row [1,2,3]
    // projects to [1*1+3*1, 2*1] = [4,2] and row [4,5,6] to [10,5]; with
    // the bias the result is [5,3,11,6]. bf16 keeps integers up to 256
    // exact, so this is equality, not a window - and bf16 had never
    // travelled the Linear executor on CUDA before.
    require_cuda();
    let input = upload_bytes(
        &[2, 3],
        DTypeId::BF16,
        &bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let weight = upload_bytes(
        &[2, 3],
        DTypeId::BF16,
        &bf16_bytes(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    );
    let bias = upload_bytes(&[2], DTypeId::BF16, &bf16_bytes(&[1.0, 1.0]));

    let context = ExecutionContext::new(TestBackend::new());
    let (out, recorded) = run3::<op::Linear, _, bf16>(
        &context,
        &input,
        &weight,
        &bias,
        LinearAttributes { has_bias: true },
    );
    let out = out.expect("a homogeneous bf16 linear with bias is advertised and must launch");
    assert!(
        recorded >= 1,
        "training-mode Linear must record tape, recorded {recorded}"
    );
    assert_eq!(out.shape, vec![2, 2], "bf16 linear output shape");
    assert_eq!(out.dtype, DTypeId::BF16.descriptor());
    assert_eq!(
        decode_bf16(&download_bytes(&out)),
        vec![5.0, 3.0, 11.0, 6.0],
        "input @ transpose(weight) + bias must project at bf16"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn addmm_refuses_a_mixed_dtype_addend() {
    // The row advertises FLOAT_DTYPES for every operand, which states the
    // union of what the operands may carry but not that they must agree;
    // `dispatch` applies the same set to each operand in turn, so a mixed
    // f32 addend with an f16 product passes admission. The equality the
    // row cannot express is the executor's to enforce: the final add must
    // refuse host-side rather than promote one operand. Every stage before
    // it is homogeneous and succeeds, so the refusal pinpoints the add.
    require_cuda();
    let mat = upload_bytes(
        &[2, 2],
        DTypeId::F32,
        &1.0f32
            .to_bits()
            .to_le_bytes()
            .iter()
            .chain(2.0f32.to_le_bytes().iter())
            .chain(3.0f32.to_le_bytes().iter())
            .chain(4.0f32.to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>(),
    );
    let lhs = upload_bytes(
        &[2, 3],
        DTypeId::F16,
        &f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    let rhs = upload_bytes(
        &[3, 2],
        DTypeId::F16,
        &f16_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    );

    let context = ExecutionContext::new(TestBackend::new());
    let out = dispatch::execute::<op::Addmm, TestBackend>(
        &context,
        AddmmAttributes {
            alpha: 1.0,
            beta: 1.0,
        },
        &[
            TensorHandle::from_storage::<TestBackend, f32, _>(&mat),
            TensorHandle::from_storage::<TestBackend, f16, _>(&lhs),
            TensorHandle::from_storage::<TestBackend, f16, _>(&rhs),
        ],
    );
    let err = out.expect_err("a mixed f32/f16 addmm must refuse, not promote");
    let text = format!("{err:?}").to_ascii_lowercase();
    assert!(
        text.contains("dtype") || text.contains("mismatch") || text.contains("mix"),
        "the refusal must name the dtype problem it found, got: {text}"
    );
}
