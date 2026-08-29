//! Wires `kernels/pool.cu`'s max/avg/adaptive-avg 2D pooling kernels
//! (forward + backward, all present already, none previously called from
//! anywhere) into the CUDA backend. Unlike WGPU's pooling backward (which
//! reads back to a host `Vec` and computes with plain Rust loops); CUDA’s backward kernels
//! already exist as real device code, so the round trip stays entirely on
//! the GPU.

#![allow(dead_code)]

use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::Result;
use incin_core::shapes::error::OperationKind;
use incin_core::shapes::{ShapeBuf, ShapeError};
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

/// `[N, C, H, W]`-style output spatial size, matching
/// Uses checked arithmetic and rejects invalid zero parameters.
fn out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| len.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "CUDA pooling padded input dimension",
        })?;
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Pool2d,
            expression: "CUDA pooling effective kernel",
        })?;
    if effective_kernel > padded {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Pool2d,
            parameter: "effective kernel exceeds padded input",
            value: effective_kernel,
        }
        .into());
    }
    Ok((padded - effective_kernel) / stride + 1)
}

#[cfg(feature = "cuda")]
fn ensure_pool_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "pool").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "pool",
            crate::cuda::ops::kernels::POOL_KERNEL,
            "pool",
        )?;
    }
    Ok(())
}

fn alloc_zeroed(
    stream: &Arc<cudarc::driver::CudaStream>,
    device: &Arc<cudarc::driver::CudaContext>,
    device_id: usize,
    dtype: DTypeDescriptor,
    numel: usize,
) -> Result<CudaBuffer> {
    Ok(CudaBuffer {
        len: numel,
        dtype,
        data: Arc::new(alloc_zeroed_bytes(
            stream,
            dtype,
            numel,
            OperationKind::Pool2d,
        )?),
        device: device.clone(),
        device_id,
    })
}

fn launch_cfg(n: usize) -> Result<cudarc::driver::LaunchConfig> {
    let block_size: u32 = 256;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (
            crate::cuda::checked_u32(n, "CUDA pooling grid dimension")?.div_ceil(block_size),
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Forward max_pool2d. Returns `(output, max_indices)` - `max_indices` is a
/// same-length `U32`-dtype `CudaStorage` (one winning flat source index per
/// output position) captured by the backward closure and replayed through
/// `launch_scatter_pool_grad_2d`, exactly like CPU's `max_window_2d`/
/// `scatter_pool_grad_2d` pairing (`cpu/ops/pool.rs`).
#[cfg(feature = "cuda")]
pub(crate) fn launch_max_pool2d_forward(
    t: &CudaStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<(CudaStorage, CudaStorage)> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(t_buf.dtype, "max_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "max_pool2d_forward")?;
    let stream = t_buf.device.default_stream();

    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;
    let oh = out_size(h, kh, sh, ph, dh)?;
    let ow = out_size(w, kw, sw, pw, dw)?;
    let out_shape = alloc::vec![n, c, oh, ow];
    let out_total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Pool2d)?;

    let mut out_b = alloc_zeroed(&stream, &t_buf.device, device_id, t_buf.dtype, out_total)?;
    let mut idx_b = alloc_zeroed(
        &stream,
        &t_buf.device,
        device_id,
        DTypeId::U32.descriptor(),
        out_total,
    )?;

    let cfg = launch_cfg(out_total)?;
    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let out_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut out_b.data).expect("out_b freshly allocated, uniquely owned");
        let mut out_f32 = out_u8.transmute_mut::<f32>(out_total).unwrap();
        let idx_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut idx_b.data).expect("idx_b freshly allocated, uniquely owned");
        let mut idx_u32 = idx_u8.transmute_mut::<u32>(out_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&in_f32)
            .arg(&mut out_f32)
            .arg(&mut idx_u32)
            .arg(&n)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&oh)
            .arg(&ow)
            .arg(&kh)
            .arg(&kw)
            .arg(&sh)
            .arg(&sw)
            .arg(&ph)
            .arg(&pw)
            .arg(&dh)
            .arg(&dw)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!("max_pool2d_forward launch failed: {e:?}"))
            })?;
    }

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    let output =
        CudaStorage::try_from_parts(Arc::new(out_b), out_shape.clone(), out_strides.clone(), 0)?;
    let max_indices = CudaStorage::try_from_parts(Arc::new(idx_b), out_shape, out_strides, 0)?;
    Ok((output, max_indices))
}

/// Backward max_pool2d: scatters `grad_out` to each winning source position
/// (captured in `max_indices`) via `atomicAdd`, into a fresh zeroed buffer
/// shaped like the original input.
#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter_pool_grad_2d(
    grad_out: &CudaStorage,
    max_indices: &CudaStorage,
    input_shape: &[usize],
) -> Result<CudaStorage> {
    let go_buf = &*grad_out.buffer;
    let device_id = go_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(go_buf.dtype, "max_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "scatter_pool_grad_2d")?;
    let stream = go_buf.device.default_stream();

    let out_total: usize = incin_core::shapes::ShapeBuf::from_slice(&grad_out.shape)
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let in_total: usize = incin_core::shapes::ShapeBuf::from_slice(input_shape)
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    let mut grad_in_b = alloc_zeroed(&stream, &go_buf.device, device_id, go_buf.dtype, in_total)?;

    let cfg = launch_cfg(out_total)?;
    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let go_f32 = go_buf.data.transmute::<f32>(go_buf.len).unwrap();
        let idx_buf = &*max_indices.buffer;
        let idx_u32 = idx_buf.data.transmute::<u32>(idx_buf.len).unwrap();
        let gi_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut grad_in_b.data).expect("grad_in_b freshly allocated, uniquely owned");
        let mut gi_f32 = gi_u8.transmute_mut::<f32>(in_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&go_f32)
            .arg(&idx_u32)
            .arg(&mut gi_f32)
            .arg(&out_total)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!("scatter_pool_grad_2d launch failed: {e:?}"))
            })?;
    }

    let strides = crate::layout::contiguous_strides(input_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(grad_in_b), input_shape.to_vec(), strides, 0)
}

/// Forward avg_pool2d (no dilation - matches `::avg_pool2d`'s
/// signature, which has no dilation parameter either).
#[cfg(feature = "cuda")]
pub(crate) fn launch_avg_pool2d_forward(
    t: &CudaStorage,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(t_buf.dtype, "avg_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "avg_pool2d_forward")?;
    let stream = t_buf.device.default_stream();

    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let oh = out_size(h, kh, sh, ph, 1)?;
    let ow = out_size(w, kw, sw, pw, 1)?;
    let out_shape = alloc::vec![n, c, oh, ow];
    let out_total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Pool2d)?;

    let mut out_b = alloc_zeroed(&stream, &t_buf.device, device_id, t_buf.dtype, out_total)?;
    let cfg = launch_cfg(out_total)?;
    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let out_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut out_b.data).expect("out_b freshly allocated, uniquely owned");
        let mut out_f32 = out_u8.transmute_mut::<f32>(out_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&in_f32)
            .arg(&mut out_f32)
            .arg(&n)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&oh)
            .arg(&ow)
            .arg(&kh)
            .arg(&kw)
            .arg(&sh)
            .arg(&sw)
            .arg(&ph)
            .arg(&pw)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!("avg_pool2d_forward launch failed: {e:?}"))
            })?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_avg_pool2d_backward(
    grad_out: &CudaStorage,
    input_shape: &[usize],
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<CudaStorage> {
    let go_buf = &*grad_out.buffer;
    let device_id = go_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(go_buf.dtype, "avg_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "avg_pool2d_backward")?;
    let stream = go_buf.device.default_stream();

    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (oh, ow) = (grad_out.shape[2], grad_out.shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let in_total = ShapeBuf::from_slice(&[n, c, h, w]).checked_numel(OperationKind::Pool2d)?;
    let out_total = ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;

    let mut grad_in_b = alloc_zeroed(&stream, &go_buf.device, device_id, go_buf.dtype, in_total)?;
    let cfg = launch_cfg(out_total)?;
    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let go_f32 = go_buf.data.transmute::<f32>(go_buf.len).unwrap();
        let gi_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut grad_in_b.data).expect("grad_in_b freshly allocated, uniquely owned");
        let mut gi_f32 = gi_u8.transmute_mut::<f32>(in_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&go_f32)
            .arg(&mut gi_f32)
            .arg(&n)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&oh)
            .arg(&ow)
            .arg(&kh)
            .arg(&kw)
            .arg(&sh)
            .arg(&sw)
            .arg(&ph)
            .arg(&pw)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!("avg_pool2d_backward launch failed: {e:?}"))
            })?;
    }

    let strides = crate::layout::contiguous_strides(input_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(grad_in_b), input_shape.to_vec(), strides, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_adaptive_avg_pool2d(
    t: &CudaStorage,
    out_size: (usize, usize),
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(t_buf.dtype, "adaptive_avg_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "adaptive_avg_pool2d_forward")?;
    let stream = t_buf.device.default_stream();

    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let (oh, ow) = out_size;
    let out_shape = alloc::vec![n, c, oh, ow];
    let out_total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Pool2d)?;

    let mut out_b = alloc_zeroed(&stream, &t_buf.device, device_id, t_buf.dtype, out_total)?;
    let cfg = launch_cfg(out_total)?;

    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let out_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut out_b.data).expect("out_b freshly allocated, uniquely owned");
        let mut out_f32 = out_u8.transmute_mut::<f32>(out_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&in_f32)
            .arg(&mut out_f32)
            .arg(&n)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&oh)
            .arg(&ow)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!(
                    "adaptive_avg_pool2d_forward launch failed: {e:?}"
                ))
            })?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_adaptive_avg_pool2d_backward(
    grad_out: &CudaStorage,
    input_shape: &[usize],
) -> Result<CudaStorage> {
    let go_buf = &*grad_out.buffer;
    let device_id = go_buf.device_id;
    crate::cuda::capability::validate_cuda_f32_kernel(go_buf.dtype, "adaptive_avg_pool2d")?;
    ensure_pool_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("pool", "adaptive_avg_pool2d_backward")?;
    let stream = go_buf.device.default_stream();

    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let (oh, ow) = (grad_out.shape[2], grad_out.shape[3]);
    let in_total = ShapeBuf::from_slice(&[n, c, h, w]).checked_numel(OperationKind::Pool2d)?;
    let out_total = ShapeBuf::from_slice(&[n, c, oh, ow]).checked_numel(OperationKind::Pool2d)?;

    let mut grad_in_b = alloc_zeroed(&stream, &go_buf.device, device_id, go_buf.dtype, in_total)?;
    let cfg = launch_cfg(out_total)?;

    // SAFETY: shape validation fixes the element counts, allocation sizes,
    // dtypes, and launch dimensions before these device-buffer reinterprets.
    unsafe {
        let go_f32 = go_buf.data.transmute::<f32>(go_buf.len).unwrap();
        let gi_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut grad_in_b.data).expect("grad_in_b freshly allocated, uniquely owned");
        let mut gi_f32 = gi_u8.transmute_mut::<f32>(in_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&go_f32)
            .arg(&mut gi_f32)
            .arg(&n)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&oh)
            .arg(&ow)
            .launch(cfg)
            .map_err(|e| {
                incin_core::error::Error::Msg(format!(
                    "adaptive_avg_pool2d_backward launch failed: {e:?}"
                ))
            })?;
    }

    let strides = crate::layout::contiguous_strides(input_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(grad_in_b), input_shape.to_vec(), strides, 0)
}
