//! Wires `kernels/conv.cu`'s `im2col_2d`/`col2im_2d`/`im2col_1d`/`col2im_1d`
//! kernels into the CUDA backend. Unlike CPU's `im2col_2d`/`col2im_2d`
//! (`cpu/ops/conv.rs`), which produce a `[B, H_out*W_out, Cin*Kh*Kw]`
//! (spatial-major) column matrix, this kernel's own layout is
//! channel-major: `[B, Cin*Kh*Kw, H_out*W_out]` — so `conv1d`/`conv2d` in
//! `cuda/backend.rs` compute `weight_mat @ cols` directly (no transpose of
//! either operand needed), rather than CPU/WGPU's `cols @ weight_mat^T`.
//!
//! These are raw launchers only (no tape wiring — matches `ops::shape`'s
//! convention: `cuda/backend.rs` wires the actual `TapeEntry`s, pairing
//! `launch_im2col_2d`/`launch_col2im_2d` as each other's forward/backward,
//! same for the 1D pair).

use crate::cuda::storage::{CudaBuffer, CudaStorage, TensorId};
use alloc::sync::Arc;
use alloc::vec::Vec;
use kindle_core::prelude::Result;

/// `L_out`/`H_out`/`W_out` output-size formula, matching
/// `cpu/ops/conv.rs::out_size` exactly (same saturating arithmetic).
pub(crate) fn out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let padded = len + 2 * padding;
    let effective_kernel = dilation * kernel_size.saturating_sub(1) + 1;
    padded.saturating_sub(effective_kernel) / stride + 1
}

/// The "natural" (no `output_padding`) `conv_transpose2d` output size,
/// matching `cpu/ops/conv.rs::natural_transpose_out_size` exactly.
pub(crate) fn natural_transpose_out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let unstrided = len.saturating_sub(1) * stride;
    let effective_kernel = dilation * kernel_size.saturating_sub(1);
    (unstrided + effective_kernel + 1).saturating_sub(2 * padding)
}

#[cfg(feature = "cuda")]
fn ensure_conv_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "conv").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel(
            "conv",
            crate::cuda::ops::kernels::CONV_KERNEL,
            "conv",
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn alloc_zeroed(
    stream: &Arc<cudarc::driver::CudaStream>,
    device: &Arc<cudarc::driver::CudaContext>,
    device_id: usize,
    dtype: kindle_core::prelude::DTypeId,
    numel: usize,
) -> CudaBuffer {
    CudaBuffer {
        len: numel,
        dtype,
        data: Arc::new(stream.alloc_zeros::<u8>(numel * 4).unwrap()),
        device: device.clone(),
        device_id,
    }
}

#[cfg(feature = "cuda")]
fn launch_cfg(n: usize) -> cudarc::driver::LaunchConfig {
    let block_size: u32 = 256;
    cudarc::driver::LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Unfolds `t: [B, C, H, W]` into `[B, C*Kh*Kw, H_out*W_out]` (channel-major
/// — see module doc). Out-of-bounds (padded-region) source positions are
/// written as `0.0` by the kernel itself.
#[cfg(feature = "cuda")]
pub(crate) fn launch_im2col_2d(
    t: &CudaStorage,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    ensure_conv_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("conv", "im2col_2d")?;
    let stream = t_buf.device.default_stream();

    let (b, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let h_out = out_size(h, kh, stride, padding, dilation);
    let w_out = out_size(w, kw, stride, padding, dilation);
    let out_shape: Vec<usize> = alloc::vec![b, c * kh * kw, h_out * w_out];
    let out_total: usize = out_shape.iter().product();
    let thread_total = b * c * h_out * w_out;

    let mut out_b = alloc_zeroed(&stream, &t_buf.device, device_id, t_buf.dtype, out_total);
    let cfg = launch_cfg(thread_total);
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
            .arg(&b)
            .arg(&c)
            .arg(&h)
            .arg(&w)
            .arg(&h_out)
            .arg(&w_out)
            .arg(&kh)
            .arg(&kw)
            .arg(&stride)
            .arg(&stride)
            .arg(&padding)
            .arg(&padding)
            .arg(&dilation)
            .arg(&dilation)
            .launch(cfg)
            .map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("im2col_2d launch failed: {e:?}"))
            })?;
    }

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    Ok(CudaStorage {
        buffer: Arc::new(out_b),
        shape: out_shape,
        strides,
        offset: 0,
        id: TensorId::next(),
    })
}

/// Exact inverse of `launch_im2col_2d`: scatter-ADDs (`atomicAdd`) `cols:
/// [B, C*Kh*Kw, H_out*W_out]` into a fresh zero-initialized
/// `target_shape = [B, C, H_in, W_in]` buffer. `h_out`/`w_out` are passed
/// explicitly (the caller already knows them from the forward context) since
/// `cols.shape[2]` only carries their flattened product.
#[cfg(feature = "cuda")]
pub(crate) fn launch_col2im_2d(
    cols: &CudaStorage,
    target_shape: &[usize],
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let cols_buf = &*cols.buffer;
    let device_id = cols_buf.device_id;
    ensure_conv_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("conv", "col2im_2d")?;
    let stream = cols_buf.device.default_stream();

    let (b, c, h_in, w_in) = (
        target_shape[0],
        target_shape[1],
        target_shape[2],
        target_shape[3],
    );
    let out_total = b * c * h_in * w_in;
    let thread_total = b * c * h_out * w_out;

    let mut out_b = alloc_zeroed(
        &stream,
        &cols_buf.device,
        device_id,
        cols_buf.dtype,
        out_total,
    );
    let cfg = launch_cfg(thread_total);
    unsafe {
        let col_f32 = cols_buf.data.transmute::<f32>(cols_buf.len).unwrap();
        let out_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut out_b.data).expect("out_b freshly allocated, uniquely owned");
        let mut out_f32 = out_u8.transmute_mut::<f32>(out_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&col_f32)
            .arg(&mut out_f32)
            .arg(&b)
            .arg(&c)
            .arg(&h_in)
            .arg(&w_in)
            .arg(&h_out)
            .arg(&w_out)
            .arg(&kh)
            .arg(&kw)
            .arg(&stride)
            .arg(&stride)
            .arg(&padding)
            .arg(&padding)
            .arg(&dilation)
            .arg(&dilation)
            .launch(cfg)
            .map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("col2im_2d launch failed: {e:?}"))
            })?;
    }

    let strides = crate::cpu::stride::contiguous_strides(target_shape);
    Ok(CudaStorage {
        buffer: Arc::new(out_b),
        shape: target_shape.to_vec(),
        strides,
        offset: 0,
        id: TensorId::next(),
    })
}

/// 1D analogue of `launch_im2col_2d`: unfolds `t: [B, C, L]` into
/// `[B, C*K, L_out]`.
#[cfg(feature = "cuda")]
pub(crate) fn launch_im2col_1d(
    t: &CudaStorage,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    ensure_conv_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("conv", "im2col_1d")?;
    let stream = t_buf.device.default_stream();

    let (b, c, l) = (t.shape[0], t.shape[1], t.shape[2]);
    let l_out = out_size(l, k, stride, padding, dilation);
    let out_shape: Vec<usize> = alloc::vec![b, c * k, l_out];
    let out_total: usize = out_shape.iter().product();
    let thread_total = b * c * l_out;

    let mut out_b = alloc_zeroed(&stream, &t_buf.device, device_id, t_buf.dtype, out_total);
    let cfg = launch_cfg(thread_total);
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
            .arg(&b)
            .arg(&c)
            .arg(&l)
            .arg(&l_out)
            .arg(&k)
            .arg(&stride)
            .arg(&padding)
            .arg(&dilation)
            .launch(cfg)
            .map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("im2col_1d launch failed: {e:?}"))
            })?;
    }

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    Ok(CudaStorage {
        buffer: Arc::new(out_b),
        shape: out_shape,
        strides,
        offset: 0,
        id: TensorId::next(),
    })
}

/// Exact inverse of `launch_im2col_1d`. `l_out` (`cols.shape[2]`) is the
/// col's own (smaller) length; `target_shape = [B, C, L_in]` is the larger
/// buffer being scattered into.
#[cfg(feature = "cuda")]
pub(crate) fn launch_col2im_1d(
    cols: &CudaStorage,
    target_shape: &[usize],
    l_out: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let cols_buf = &*cols.buffer;
    let device_id = cols_buf.device_id;
    ensure_conv_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("conv", "col2im_1d")?;
    let stream = cols_buf.device.default_stream();

    let (b, c, l_in) = (target_shape[0], target_shape[1], target_shape[2]);
    let out_total = b * c * l_in;
    let thread_total = b * c * l_out;

    let mut out_b = alloc_zeroed(
        &stream,
        &cols_buf.device,
        device_id,
        cols_buf.dtype,
        out_total,
    );
    let cfg = launch_cfg(thread_total);
    unsafe {
        let col_f32 = cols_buf.data.transmute::<f32>(cols_buf.len).unwrap();
        let out_u8: &mut cudarc::driver::CudaSlice<u8> =
            Arc::get_mut(&mut out_b.data).expect("out_b freshly allocated, uniquely owned");
        let mut out_f32 = out_u8.transmute_mut::<f32>(out_total).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&col_f32)
            .arg(&mut out_f32)
            .arg(&b)
            .arg(&c)
            .arg(&l_in)
            .arg(&l_out)
            .arg(&k)
            .arg(&stride)
            .arg(&padding)
            .arg(&dilation)
            .launch(cfg)
            .map_err(|e| {
                kindle_core::prelude::Error::Msg(format!("col2im_1d launch failed: {e:?}"))
            })?;
    }

    let strides = crate::cpu::stride::contiguous_strides(target_shape);
    Ok(CudaStorage {
        buffer: Arc::new(out_b),
        shape: target_shape.to_vec(),
        strides,
        offset: 0,
        id: TensorId::next(),
    })
}
