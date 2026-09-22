//! Wires `kernels/matmul.cu`'s tiled shared-memory GEMM (`BM=128, BN=128,
//! BK=8, TM=8, TN=8`, 16x16 thread blocks) into the CUDA backend. Unbatched
//! 2D operands only - batched matmul is composed in
//! `cuda/backend/shape_ops.rs` - and, since issue #90, every float storage
//! dtype the kernel exports an entry point for: `f32`/`f64` accumulate in
//! their own type, `f16`/`bf16` hold half-precision operands and
//! accumulate in `f32`.

use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf};
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

const BM: u32 = 128;
const BN: u32 = 128;

#[cfg(feature = "cuda")]
fn ensure_matmul_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "matmul").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "matmul",
            crate::cuda::ops::kernels::MATMUL_KERNEL,
            "matmul",
        )?;
    }
    Ok(())
}

/// The kernel entry point for a storage dtype, or a typed refusal for
/// anything `matmul.cu` does not export (`i64`/`bool`/`q8_0`/`u8`/`u32`).
/// Fail-closed by construction: the launcher launches exactly this name, so
/// a dtype without an entry can never reach a buffer reinterpreted as the
/// wrong type.
#[cfg(feature = "cuda")]
fn matmul_entry_point(dtype: DTypeDescriptor) -> Result<&'static str> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok("matmul"),
        Some(DTypeId::F64) => Ok("matmul_f64"),
        Some(DTypeId::F16) => Ok("matmul_f16"),
        Some(DTypeId::BF16) => Ok("matmul_bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "matmul",
        }),
    }
}

/// `lhs`: `[M, K]`, `rhs`: `[K, N]` -> `[M, N]`. Caller (the `Execute` /
/// trait method) is responsible for the `lhs.shape[1] == rhs.shape[0]`
/// shape check - this function assumes it already holds. Both operands
/// must carry the same float dtype: one typed kernel reads them both, so a
/// mixed pair is refused rather than reinterpreted.
#[cfg(feature = "cuda")]
pub(crate) fn launch_matmul(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let (lhs_buf, rhs_buf) = (&*lhs.buffer, &*rhs.buffer);
    if lhs_buf.dtype != rhs_buf.dtype {
        return Err(Error::DTypeMismatch {
            operation: "matmul",
            expected: lhs_buf.dtype,
            actual: rhs_buf.dtype,
        });
    }
    // Selects the entry point and refuses every dtype without one, before
    // any buffer is reinterpreted.
    let entry_point = matmul_entry_point(lhs_buf.dtype)?;
    let device_id = lhs_buf.device_id;
    ensure_matmul_loaded(device_id)?;

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("matmul", entry_point)?;
    let stream = lhs_buf.device.default_stream();

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];
    let out_shape = alloc::vec![m, n];
    let total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;

    let mut out_b = CudaBuffer {
        len: total,
        dtype: lhs_buf.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            lhs_buf.dtype,
            total,
            OperationKind::MatMul,
        )?),
        device: lhs_buf.device.clone(),
        device_id,
    };

    let m_u32 = crate::cuda::checked_u32(m, "CUDA matmul row grid dimension")?;
    let n_u32 = crate::cuda::checked_u32(n, "CUDA matmul column grid dimension")?;
    let m_i32 = crate::cuda::checked_i32(m, "CUDA matmul row count")?;
    let k_i32 = crate::cuda::checked_i32(k, "CUDA matmul inner dimension")?;
    let n_i32 = crate::cuda::checked_i32(n, "CUDA matmul column count")?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_u32.div_ceil(BN), m_u32.div_ceil(BM), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: checked matrix dimensions establish each slice length and the
    // launch extent; out_b was just allocated and is uniquely owned. The
    // views pass through as byte buffers, the convention `quant.rs` and
    // `embedding.rs` already use for multi-dtype entries: the driver only
    // needs the device address, the kernel is typed by the entry point, and
    // `matmul_entry_point` above established that the entry's type really
    // is this buffer's dtype.
    unsafe {
        // out_b.data was allocated immediately above and never cloned, so
        // it stays uniquely owned (refcount 1) here - Arc::get_mut succeeds
        // without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&*lhs_buf.data)
            .arg(&*rhs_buf.data)
            .arg(&mut *out_u8)
            .arg(&m_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| Error::Msg(format!("matmul launch failed: {e:?}")))?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}
