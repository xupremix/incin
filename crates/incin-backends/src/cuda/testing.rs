//! Handles for integration tests that need to reach CUDA kernels directly.
//!
//! The launch functions are `pub(crate)`, which is right: they are not API. But
//! the suites under `tests/` are separate crates, so without a deliberate seam
//! they can only test what the public surface exposes -- and the four CUDA
//! suites that existed before this module were written around that limit by
//! asserting things that needed no kernel at all. One of them recomputed Adam in
//! its own body and checked its own arithmetic.
//!
//! This is that seam, kept as narrow as the tests require: upload, download, and
//! the optimizer entry points. It is `#[doc(hidden)]` because it is a test
//! affordance rather than API, and feature-gated with `cuda` so it cannot exist
//! on a build that has no kernels to reach.

use crate::cuda::storage::CudaStorage;
use incin_core::error::Result;
use incin_core::exec::catalog::{AdamAttributes, AdamWAttributes, SgdAttributes};
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeId;

/// Whether a CUDA device is present, so a suite can skip rather than fail.
#[must_use]
pub fn cuda_available() -> bool {
    cudarc::driver::CudaContext::new(0).is_ok()
}

/// Uploads `values` as a rank-1 `f32` tensor on device 0.
#[must_use]
pub fn upload_f32(values: &[f32]) -> CudaStorage {
    crate::cuda::backend::cuda_from_f32(
        &[values.len()],
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values.to_vec(),
        "testing::upload_f32",
    )
    .expect("uploading an f32 slice to device 0")
}

/// Reads a storage back to the host as `f32`.
#[must_use]
pub fn download_f32(storage: &CudaStorage) -> alloc::vec::Vec<f32> {
    crate::cuda::backend::download_f32_host(storage).expect("reading f32 storage back to the host")
}

/// Uploads `values` as a tensor of `shape` on device 0.
#[must_use]
pub fn upload_f32_shaped(shape: &[usize], values: &[f32]) -> CudaStorage {
    crate::cuda::backend::cuda_from_f32(
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values.to_vec(),
        "testing::upload_f32_shaped",
    )
    .expect("uploading an f32 slice to device 0")
}

/// Reads an `i64`-typed storage back to the host.
///
/// `StorageBackend::int_to_vec1` cannot serve here: it downloads as `f32` and
/// converts, so it rejects genuinely integer storage -- which is exactly what
/// `argmax` and `topk` return.
///
/// # Panics
///
/// Panics if the storage is not `i64`, or if the device transfer fails.
#[must_use]
pub fn download_i64(storage: &CudaStorage) -> alloc::vec::Vec<i64> {
    assert_eq!(
        storage.buffer.dtype.builtin_id(),
        Some(DTypeId::I64),
        "download_i64 expects i64 storage"
    );
    let bytes = storage
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*storage.buffer.data)
        .expect("reading i64 storage back to the host");
    bytemuck::cast_slice::<u8, i64>(&bytes).to_vec()
}

/// Runs an axis reduction (`sum`, `mean`, `max`, `min`, `prod`).
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn reduce(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<CudaStorage> {
    crate::cuda::ops::reduce::launch_reduce_op(op_name, storage, axis, keepdim)
}

/// Runs `argmax` or `argmin`, returning the index tensor.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn argmax_argmin(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: Option<usize>,
) -> Result<CudaStorage> {
    crate::cuda::ops::reduce::launch_argmax_argmin_op(op_name, storage, axis, DTypeId::I64)
}

/// Runs a prefix sum along `axis`.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn cumsum(storage: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    crate::cuda::ops::reduce::launch_cumsum_op(storage, axis)
}

/// Runs `topk`, returning values and their indices.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn topk(
    storage: &CudaStorage,
    k: usize,
    axis: usize,
    largest: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    crate::cuda::ops::reduce::launch_topk_op(storage, k, axis, largest, DTypeId::I64)
}

/// Uploads `values` as an `i64` tensor on device 0.
///
/// Index operands are genuinely integer, so they cannot go through the `f32`
/// upload path.
///
/// # Panics
///
/// Panics if the allocation or transfer fails.
#[must_use]
pub fn upload_i64(shape: &[usize], values: &[i64]) -> CudaStorage {
    let bytes = bytemuck::cast_slice::<i64, u8>(values);
    crate::cuda::backend::cuda_from_bytes(shape, DTypeId::I64.into(), 0, bytes)
        .expect("uploading an i64 slice to device 0")
}

/// Gathers embedding rows named by `indices`.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn embedding(weight: &CudaStorage, indices: &CudaStorage) -> Result<CudaStorage> {
    crate::cuda::ops::shape::launch_embedding(weight, indices)
}

/// Scatters an embedding gradient back into a vocabulary-sized buffer.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn embedding_backward(
    grad_output: &CudaStorage,
    indices: &CudaStorage,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<CudaStorage> {
    crate::cuda::ops::shape::launch_embedding_backward(
        grad_output,
        indices,
        vocab_size,
        hidden_size,
    )
}

/// Transposes two axes by materialising a fresh contiguous buffer.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn transpose(t: &CudaStorage, dim1: usize, dim2: usize) -> Result<CudaStorage> {
    crate::cuda::ops::shape::launch_transpose(t, dim1, dim2)
}

/// Broadcasts `t` to `target_shape`, materialising the result.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn broadcast(t: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
    crate::cuda::ops::shape::launch_broadcast(t, target_shape)
}

/// Narrows `t` along `dim`, materialising the result.
///
/// # Errors
///
/// Propagates a launch or validation failure from the kernel.
pub fn narrow(t: &CudaStorage, dim: usize, start: usize, len: usize) -> Result<CudaStorage> {
    crate::cuda::ops::shape::launch_narrow(t, dim, start, len)
}

/// Runs the fused SGD step kernel.
///
/// # Errors
///
/// Propagates a launch or allocation failure from the kernel.
pub fn sgd_step(
    params: &CudaStorage,
    grad: &CudaStorage,
    attrs: &SgdAttributes,
) -> Result<CudaStorage> {
    crate::cuda::ops::optimizer::launch_sgd_step(params, grad, attrs)
}

/// Runs the fused Adam step kernel, returning the new parameters and moments.
///
/// # Errors
///
/// Propagates a launch or allocation failure from the kernel.
pub fn adam_step(
    params: &CudaStorage,
    grad: &CudaStorage,
    first_moment: Option<&CudaStorage>,
    second_moment: Option<&CudaStorage>,
    attrs: &AdamAttributes,
) -> Result<(CudaStorage, CudaStorage, CudaStorage)> {
    crate::cuda::ops::optimizer::launch_adam_step(params, grad, first_moment, second_moment, attrs)
}

/// Runs the fused AdamW step kernel, returning the new parameters and moments.
///
/// # Errors
///
/// Propagates a launch or allocation failure from the kernel.
pub fn adamw_step(
    params: &CudaStorage,
    grad: &CudaStorage,
    first_moment: Option<&CudaStorage>,
    second_moment: Option<&CudaStorage>,
    attrs: &AdamWAttributes,
) -> Result<(CudaStorage, CudaStorage, CudaStorage)> {
    crate::cuda::ops::optimizer::launch_adamw_step(params, grad, first_moment, second_moment, attrs)
}
