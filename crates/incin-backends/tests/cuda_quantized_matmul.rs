//! Issue #3, Option A: CUDA `quantized_matmul` matches the canonical
//! Q8_0xQ8_0 contract.
//!
//! Both operands are packed Q8_0, `lhs` is `[..., K]`, `rhs` is exactly
//! `[N, K]` (`k == k2`, `k % 32 == 0`), and the output is `[..., N]` f32
//! holding the row-dot the CPU kernel computes
//! (`cpu/ops/quant.rs::quantized_matmul_storage`). Every parity test below
//! quantizes the same host values on both backends and compares against the
//! CPU `quantized_matmul` output — not against an f32 matmul, which would
//! hide a kernel that decoded the wrong way but landed near the
//! unquantized product.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_quantized_matmul -- --ignored`.
#![cfg(all(feature = "cuda", feature = "cpu"))]

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_backends::cuda::{
    CudaBackendImpl,
    testing::{download_f32, require_cuda, upload_f32_shaped},
};
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::catalog::{NoAttributes, QuantizationAttributes, op};
use incin_core::exec::{CanonicalError, ExecutionContext, TensorHandle, dispatch};
use incin_core::prelude::{CudaN, DTypeId, Local};
use incin_core::tensor::dtype::Q8_0;
use incin_core::typenum::U0;

type CudaBackend = CudaBackendImpl<CudaN<U0>>;
type CudaStorage = <CudaBackend as StorageBackend>::Storage<f32>;
type CpuBackend = CpuBackendImpl;

fn cuda_context() -> ExecutionContext<CudaBackend> {
    ExecutionContext::new(CudaBackend::new())
}

fn cpu_context() -> ExecutionContext<CpuBackend> {
    ExecutionContext::new(CpuBackend::new())
}

fn cuda_f32_handle(storage: &CudaStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<CudaBackend, f32, Local>(storage)
}

fn cuda_q8_handle(storage: &CudaStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<CudaBackend, Q8_0, Local>(storage)
}

fn cpu_f32_handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<CpuBackend, f32, _>(storage)
}

fn cpu_q8_handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<CpuBackend, Q8_0, _>(storage)
}

fn quantize_cuda(context: &ExecutionContext<CudaBackend>, input: &CudaStorage) -> CudaStorage {
    dispatch::execute::<op::Quantize, _>(
        context,
        QuantizationAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[cuda_f32_handle(input)],
    )
    .expect("CUDA quantize executes")
}

fn quantized_matmul_cuda(
    context: &ExecutionContext<CudaBackend>,
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<CudaStorage, CanonicalError> {
    dispatch::execute::<op::QuantizedMatMul, _>(
        context,
        NoAttributes,
        &[cuda_q8_handle(lhs), cuda_q8_handle(rhs)],
    )
}

fn cpu_storage(values: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), shape)
        .expect("well-formed f32 CPU storage")
}

fn quantize_cpu(context: &ExecutionContext<CpuBackend>, input: &CpuStorage) -> CpuStorage {
    dispatch::execute::<op::Quantize, _>(
        context,
        QuantizationAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[cpu_f32_handle(input)],
    )
    .expect("CPU quantize executes")
}

fn quantized_matmul_cpu(
    context: &ExecutionContext<CpuBackend>,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
) -> CpuStorage {
    dispatch::execute::<op::QuantizedMatMul, _>(
        context,
        NoAttributes,
        &[cpu_q8_handle(lhs), cpu_q8_handle(rhs)],
    )
    .expect("CPU quantized_matmul executes")
}

/// Reads an f32 `CpuStorage` back on the host, element by element.
fn read_cpu_f32(storage: &CpuStorage) -> Vec<f32> {
    let shape = storage.shape.to_vec();
    let numel: usize = shape.iter().product();
    // `get` takes a multi-index; walk the flat index through the shape.
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for flat in 0..numel {
        let mut remainder = flat;
        for (axis, extent) in shape.iter().enumerate().rev() {
            index[axis] = remainder % extent;
            remainder /= extent;
        }
        out.push(storage.get(&index) as f32);
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

/// Deterministic values with mixed per-row magnitudes, so every Q8_0 block
/// gets its own scale — a kernel that shared one scale across rows, or
/// dropped a scale, cannot land on the reference.
fn mixed_scale_values(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * k);
    for row in 0..rows {
        let magnitude = match (row as u64 + seed) % 4 {
            0 => 0.13,
            1 => 1.0,
            2 => 4.25,
            _ => 0.02,
        };
        for col in 0..k {
            let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
            let wobble = ((row * 37 + col * 11 + seed as usize) % 17) as f32 / 16.0;
            out.push(sign * magnitude * (0.25 + wobble));
        }
    }
    out
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_matches_cpu_reference_on_ragged_shapes() {
    // M = 3, N = 5: neither is a multiple of the 32-wide launch tile, so
    // the div_ceil fringe threads must guard out rather than write past
    // the output or accumulate a partial tile twice.
    require_cuda();
    let cuda_ctx = cuda_context();
    let cpu_ctx = cpu_context();
    let (m, n, k) = (3usize, 5usize, 64usize);

    let lhs_values = mixed_scale_values(m, k, 1);
    let rhs_values = mixed_scale_values(n, k, 7);

    let cuda_lhs = quantize_cuda(&cuda_ctx, &upload_f32_shaped(&[m, k], &lhs_values));
    let cuda_rhs = quantize_cuda(&cuda_ctx, &upload_f32_shaped(&[n, k], &rhs_values));
    let out = quantized_matmul_cuda(&cuda_ctx, &cuda_lhs, &cuda_rhs)
        .expect("a Q8_0 [M,K] x Q8_0 [N,K] pair must launch");
    assert_eq!(out.shape.to_vec(), vec![m, n], "output geometry");
    assert_eq!(out.dtype(), DTypeId::F32.descriptor(), "output dtype");

    let cpu_lhs = quantize_cpu(&cpu_ctx, &cpu_storage(lhs_values, vec![m, k]));
    let cpu_rhs = quantize_cpu(&cpu_ctx, &cpu_storage(rhs_values, vec![n, k]));
    let reference = quantized_matmul_cpu(&cpu_ctx, &cpu_lhs, &cpu_rhs);
    assert_eq!(reference.shape.to_vec(), vec![m, n]);

    // Same blocks in, same row-dot math: the kernels must agree bit-near.
    // 1e-4 covers f16-scale rounding plus float accumulation order, and
    // nothing else — a dropped scale or a transposed rhs misses by orders
    // of magnitude more.
    assert_close(
        &download_f32(&out),
        &read_cpu_f32(&reference),
        1e-4,
        "CUDA Q8_0xQ8_0 vs CPU reference",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_matches_cpu_reference_on_batched_lhs() {
    // lhs [2, 3, 64]: the launcher flattens the leading axes into one row
    // count and the kernel indexes the output flat, so batch 1 must read
    // lhs rows 3..6, not repeat batch 0.
    require_cuda();
    let cuda_ctx = cuda_context();
    let cpu_ctx = cpu_context();
    let (b, m, n, k) = (2usize, 3usize, 4usize, 64usize);

    let lhs_values = mixed_scale_values(b * m, k, 3);
    let rhs_values = mixed_scale_values(n, k, 11);

    let cuda_lhs = quantize_cuda(&cuda_ctx, &upload_f32_shaped(&[b, m, k], &lhs_values));
    let cuda_rhs = quantize_cuda(&cuda_ctx, &upload_f32_shaped(&[n, k], &rhs_values));
    let out = quantized_matmul_cuda(&cuda_ctx, &cuda_lhs, &cuda_rhs)
        .expect("a batched Q8_0 lhs must launch");
    assert_eq!(out.shape.to_vec(), vec![b, m, n], "batched output geometry");
    assert_eq!(out.dtype(), DTypeId::F32.descriptor(), "output dtype");

    let cpu_lhs = quantize_cpu(&cpu_ctx, &cpu_storage(lhs_values, vec![b, m, k]));
    let cpu_rhs = quantize_cpu(&cpu_ctx, &cpu_storage(rhs_values, vec![n, k]));
    let reference = quantized_matmul_cpu(&cpu_ctx, &cpu_lhs, &cpu_rhs);
    assert_eq!(reference.shape.to_vec(), vec![b, m, n]);

    assert_close(
        &download_f32(&out),
        &read_cpu_f32(&reference),
        1e-4,
        "batched CUDA Q8_0xQ8_0 vs CPU reference",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_refuses_k_not_a_multiple_of_32() {
    // [2, 48]: 96 elements quantize fine (a whole number of blocks), but
    // K = 48 is not block-aligned, so the product must refuse rather than
    // read a partial block.
    require_cuda();
    let ctx = cuda_context();
    let lhs_values = mixed_scale_values(2, 48, 5);
    let rhs_values = mixed_scale_values(2, 48, 9);
    let lhs = quantize_cuda(&ctx, &upload_f32_shaped(&[2, 48], &lhs_values));
    let rhs = quantize_cuda(&ctx, &upload_f32_shaped(&[2, 48], &rhs_values));
    let error =
        quantized_matmul_cuda(&ctx, &lhs, &rhs).expect_err("K = 48 must be refused before launch");
    let message = format!("{error}");
    assert!(
        message.contains("multiple of 32"),
        "the refusal must name the block rule, got: {message}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_refuses_k_mismatch() {
    require_cuda();
    let ctx = cuda_context();
    let lhs = quantize_cuda(
        &ctx,
        &upload_f32_shaped(&[2, 64], &mixed_scale_values(2, 64, 13)),
    );
    let rhs = quantize_cuda(
        &ctx,
        &upload_f32_shaped(&[3, 32], &mixed_scale_values(3, 32, 17)),
    );
    let error = quantized_matmul_cuda(&ctx, &lhs, &rhs)
        .expect_err("K 64 vs 32 must be refused before launch");
    let message = format!("{error}");
    assert!(
        message.contains("mismatch") || message.contains("contracting"),
        "the refusal must name the contracting dimensions, got: {message}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_refuses_a_float_lhs() {
    // The pre-#3 launcher read the lhs as `const float*`; under the
    // canonical contract an f32 lhs is a typed refusal, not a launch.
    require_cuda();
    let ctx = cuda_context();
    let lhs = upload_f32_shaped(&[2, 32], &mixed_scale_values(2, 32, 19));
    let rhs = quantize_cuda(
        &ctx,
        &upload_f32_shaped(&[2, 32], &mixed_scale_values(2, 32, 23)),
    );
    let error = dispatch::execute::<op::QuantizedMatMul, _>(
        &ctx,
        NoAttributes,
        &[cuda_f32_handle(&lhs), cuda_q8_handle(&rhs)],
    )
    .expect_err("an f32 lhs must be refused, not launched");
    let message = format!("{error}").to_ascii_lowercase();
    assert!(
        message.contains("dtype") || message.contains("q8_0"),
        "the refusal must name the dtype problem, got: {message}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantized_matmul_refuses_a_float_rhs() {
    require_cuda();
    let ctx = cuda_context();
    let lhs = quantize_cuda(
        &ctx,
        &upload_f32_shaped(&[2, 32], &mixed_scale_values(2, 32, 29)),
    );
    let rhs = upload_f32_shaped(&[2, 32], &mixed_scale_values(2, 32, 31));
    let error = dispatch::execute::<op::QuantizedMatMul, _>(
        &ctx,
        NoAttributes,
        &[cuda_q8_handle(&lhs), cuda_f32_handle(&rhs)],
    )
    .expect_err("an f32 rhs must be refused, not launched");
    let message = format!("{error}").to_ascii_lowercase();
    assert!(
        message.contains("dtype") || message.contains("q8_0"),
        "the refusal must name the dtype problem, got: {message}"
    );
}
