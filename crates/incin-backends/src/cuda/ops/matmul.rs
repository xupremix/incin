//! Wires `kernels/matmul.cu`'s tiled shared-memory GEMM (`BM=128, BN=128,
//! BK=8, TM=8, TN=8`, 16x16 thread blocks) into the CUDA backend. Unbatched
//! 2D operands only, matching `TensorOps::matmul`'s currently-wired scope on
//! this backend — batched matmul is not
//! implemented here.

use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::prelude::Result;
use incin_core::prelude::{OperationKind, ShapeBuf};

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

/// `lhs`: `[M, K]`, `rhs`: `[K, N]` -> `[M, N]`. Caller (the `TensorOps`
/// trait method) is responsible for the `lhs.shape[1] == rhs.shape[0]`
/// shape check — this function assumes it already holds.
#[cfg(feature = "cuda")]
pub(crate) fn launch_matmul(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    let (lhs_buf, rhs_buf) = (&*lhs.buffer, &*rhs.buffer);
    let device_id = lhs_buf.device_id;
    ensure_matmul_loaded(device_id)?;

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("matmul", "matmul")?;
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

    unsafe {
        let lhs_f32 = lhs_buf.data.transmute::<f32>(lhs_buf.len).unwrap();
        let rhs_f32 = rhs_buf.data.transmute::<f32>(rhs_buf.len).unwrap();
        // out_b.data was allocated immediately above and never cloned, so
        // it stays uniquely owned (refcount 1) here — Arc::get_mut succeeds
        // without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        let mut out_f32 = out_u8.transmute_mut::<f32>(total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&lhs_f32)
            .arg(&rhs_f32)
            .arg(&mut out_f32)
            .arg(&m_i32)
            .arg(&k_i32)
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| incin_core::prelude::Error::Msg(format!("matmul launch failed: {e:?}")))?;
    }

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}
