//! Native single-pass online-softmax attention kernels (issue #104).
//!
//! Thin launchers over `kernels/attention.cu`: forward over query rows,
//! then three row-parallel backward kernels (`delta`, `dq`, `dkv`) driven
//! by the per-row `(m, l)` the forward stores. See the `.cu` header for
//! the algorithm, the GQA/causal rules (mirroring the CPU kernel exactly)
//! and why the loops carry no axis conditionals.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::cuda::{checked_i32, checked_u32};
use alloc::sync::Arc;
use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeId;

/// Geometry shared by every attention launch, validated once.
pub(crate) struct Geometry {
    batch: usize,
    heads_q: usize,
    heads_kv: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    groups: usize,
    kv_lead: usize,
    scale: f64,
    dtype: DTypeId,
}

pub(crate) fn geometry(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    scale: Option<f64>,
) -> Result<Geometry> {
    const OP: &str = "fused_attention";
    if let Some(s) = scale
        && (!s.is_finite() || s <= 0.0)
    {
        return Err(Error::Msg(alloc::format!(
            "{OP} scale must be positive and finite, got {s}"
        )));
    }
    for (name, storage) in [("query", q), ("key", k), ("value", v)] {
        if storage.shape.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "fused_attention",
                expected: alloc::vec![0, 0, 0, 0],
                got: storage.shape.to_vec(),
                msg: alloc::format!(
                    "{OP} needs rank-4 {name} [batch, heads, seq, head_dim], matching the \
                     descriptor contract"
                ),
            });
        }
        // The capability row admits contiguous layouts only; a strided
        // operand (transpose views, narrowed axes) is refused here rather
        // than misread by the row-major kernels below.
        let expected = crate::layout::contiguous_strides(&storage.shape);
        if storage.strides.strides() != expected.strides() {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::FusedAttention,
                reason: "fused_attention needs contiguous operands on CUDA",
            }));
        }
    }
    let (qs, ks, vs) = (q.shape.to_vec(), k.shape.to_vec(), v.shape.to_vec());
    if qs[0] != ks[0] {
        return Err(Error::ShapeMismatch {
            op: "fused_attention",
            expected: qs.clone(),
            got: ks.clone(),
            msg: alloc::format!(
                "{OP} query batch {} differs from the key/value batch {}",
                qs[0],
                ks[0]
            ),
        });
    }
    if ks != vs {
        return Err(Error::ShapeMismatch {
            op: "fused_attention",
            expected: ks.clone(),
            got: vs.clone(),
            msg: alloc::format!(
                "{OP} key and value must share [batch, kv_heads, seq_kv, head_dim]"
            ),
        });
    }
    if qs[3] != ks[3] || qs[3] == 0 {
        return Err(Error::ShapeMismatch {
            op: "fused_attention",
            expected: qs.clone(),
            got: ks.clone(),
            msg: alloc::format!("{OP} query/key head widths differ or are zero"),
        });
    }
    if qs[1] == 0 || ks[1] == 0 || qs[1] % ks[1] != 0 {
        return Err(Error::ShapeMismatch {
            op: "fused_attention",
            expected: qs.clone(),
            got: ks.clone(),
            msg: alloc::format!(
                "{OP} query heads {} must be a non-zero multiple of the key/value heads {}",
                qs[1],
                ks[1]
            ),
        });
    }
    // One thread per head-dim element: blockDim.x covers head_dim, so an
    // absurd width is refused instead of truncating the row. The ceiling
    // also bounds the tiled forward's reduction scratch at
    // QBLOCK * 1024 * 8 = 32 KiB, inside the 48 KiB available without an
    // opt-in carveout.
    if qs[3] > 1024 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::FusedAttention,
            reason: "fused_attention head_dim above 1024 exceeds one thread block",
        }));
    }
    let q_dtype = q.buffer.dtype.builtin_id();
    for storage in [k, v] {
        if storage.buffer.dtype.builtin_id() != q_dtype {
            return Err(Error::DTypeMismatch {
                operation: OP,
                expected: q.buffer.dtype,
                actual: storage.buffer.dtype,
            });
        }
    }
    let dtype = match q_dtype {
        Some(DTypeId::F32) | Some(DTypeId::F64) | Some(DTypeId::F16) | Some(DTypeId::BF16) => {
            q_dtype.expect("matched arm is always Some")
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: q.buffer.dtype,
                backend: "Cuda",
                op: OP,
            });
        }
    };
    Ok(Geometry {
        batch: qs[0],
        heads_q: qs[1],
        heads_kv: ks[1],
        seq_q: qs[2],
        seq_kv: ks[2],
        head_dim: qs[3],
        groups: qs[1] / ks[1],
        kv_lead: ks[2].saturating_sub(qs[2]),
        scale: scale.unwrap_or_else(|| 1.0 / (qs[3] as f64).sqrt()),
        dtype,
    })
}

/// Entry-point name for one kernel family and storage dtype: the `.cu`
/// file exports `<family>_f32/f64/f16/bf16`, selected by the validated
/// operand dtype so a buffer can never reach a mistyped kernel.
fn attn_entry(family: &'static str, dtype: DTypeId) -> Result<&'static str> {
    let suffix = match dtype {
        DTypeId::F32 => "f32",
        DTypeId::F64 => "f64",
        DTypeId::F16 => "f16",
        DTypeId::BF16 => "bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: dtype.descriptor(),
                backend: "Cuda",
                op: "fused_attention",
            });
        }
    };
    Ok(match (family, suffix) {
        ("attn_fwd", "f32") => "attn_fwd_f32",
        ("attn_fwd", "f64") => "attn_fwd_f64",
        ("attn_fwd", "f16") => "attn_fwd_f16",
        ("attn_fwd", "bf16") => "attn_fwd_bf16",
        ("attn_fwd_tiled", "f32") => "attn_fwd_tiled_f32",
        ("attn_fwd_tiled", "f64") => "attn_fwd_tiled_f64",
        ("attn_fwd_tiled", "f16") => "attn_fwd_tiled_f16",
        ("attn_fwd_tiled", "bf16") => "attn_fwd_tiled_bf16",
        ("attn_delta", "f32") => "attn_delta_f32",
        ("attn_delta", "f64") => "attn_delta_f64",
        ("attn_delta", "f16") => "attn_delta_f16",
        ("attn_delta", "bf16") => "attn_delta_bf16",
        ("attn_dq", "f32") => "attn_dq_f32",
        ("attn_dq", "f64") => "attn_dq_f64",
        ("attn_dq", "f16") => "attn_dq_f16",
        ("attn_dq", "bf16") => "attn_dq_bf16",
        ("attn_dkv", "f32") => "attn_dkv_f32",
        ("attn_dkv", "f64") => "attn_dkv_f64",
        ("attn_dkv", "f16") => "attn_dkv_f16",
        ("attn_dkv", "bf16") => "attn_dkv_bf16",
        _ => {
            return Err(Error::Msg(alloc::format!(
                "unknown attention kernel family {family}"
            )));
        }
    })
}

#[cfg(feature = "cuda")]
fn ensure_attention_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "attention").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "attention",
            crate::cuda::ops::kernels::ATTENTION_KERNEL,
            "attention",
        )?;
    }
    Ok(())
}

fn alloc_buffer(
    stream: &Arc<cudarc::driver::CudaStream>,
    dtype: DTypeId,
    numel: usize,
    device: &Arc<cudarc::driver::CudaContext>,
    device_id: usize,
    what: &'static str,
) -> Result<CudaBuffer> {
    let byte_len = crate::bytes::byte_len(dtype, numel, OperationKind::Storage)?;
    Ok(CudaBuffer {
        len: numel,
        dtype: dtype.descriptor(),
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|error| {
            Error::Msg(alloc::format!("CUDA {what} allocation failed: {error:?}"))
        })?),
        device: device.clone(),
        device_id,
    })
}

fn as_i64(value: usize, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        incin_core::shapes::ShapeError::ArithmeticOverflow {
            operation: OperationKind::Storage,
            expression: field,
        }
        .into()
    })
}

/// Query rows per block in the tiled forward, mirroring `ATTN_QBLOCK` in
/// `kernels/attention.cu`. The host has to know the tile size to size the
/// grid and the kernel has to know it to decompose `blockIdx`, so the two
/// definitions must stay in step; `cuda_fused_attention.rs` covers the
/// tiled path against the CPU twin.
const QBLOCK: usize = 4;

/// Below this many query rows the forward stays on the one-row-per-block
/// kernel. Tiling trades blocks for traffic, so a short sequence (and the
/// `Sq == 1` decode step in particular) would lose more parallelism than
/// it saves bytes.
const QBLOCK_MIN_ROWS: usize = 2 * QBLOCK;

/// True when the tiled forward is the right kernel for this geometry.
///
/// Only the sequence length decides it. The tiled reduction scratch is
/// `QBLOCK * head_dim` doubles, which the `head_dim <= 1024` ceiling in
/// `geometry` already bounds at 32 KiB - under the 48 KiB every CUDA
/// architecture offers - so there is no launch that needs to fall back
/// for shared memory.
fn tiled_forward(seq_q: usize) -> bool {
    seq_q >= QBLOCK_MIN_ROWS
}

fn launch_config(
    rows: u64,
    head_dim: usize,
    scratch_rows: usize,
) -> Result<cudarc::driver::LaunchConfig> {
    let grid =
        u32::try_from(rows).map_err(|_| incin_core::shapes::ShapeError::ArithmeticOverflow {
            operation: OperationKind::FusedAttention,
            expression: "attention row count exceeds u32",
        })?;
    let block = checked_u32(head_dim, "attention head_dim")?;
    let shared = checked_u32(
        head_dim
            .checked_mul(scratch_rows)
            .and_then(|lanes| lanes.checked_mul(8))
            .ok_or(incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::FusedAttention,
                expression: "attention reduction scratch overflows u32",
            })?,
        "attention reduction scratch",
    )?;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shared,
    })
}

/// Forward: `[B, Hq, Sq, D]` output plus the per-row `(m, l)` doubles the
/// backward kernels recompute weights from. No score matrix is allocated.
#[cfg(feature = "cuda")]
pub(crate) fn launch_attention_forward(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    causal: bool,
    scale: Option<f64>,
) -> Result<(CudaStorage, CudaStorage)> {
    let g = geometry(q, k, v, scale)?;
    let dtype = g.dtype;
    let device_id = q.buffer.device_id;
    ensure_attention_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let tiled = tiled_forward(g.seq_q);
    let family = if tiled { "attn_fwd_tiled" } else { "attn_fwd" };
    let entry = attn_entry(family, g.dtype)?;
    let function = dispatcher.get_function("attention", entry)?;
    let stream = q.buffer.device.default_stream();

    let out_numel = g.batch * g.heads_q * g.seq_q * g.head_dim;
    let ml_numel = g.batch * g.heads_q * g.seq_q * 2;
    let out_buffer = alloc_buffer(
        &stream,
        dtype,
        out_numel,
        &q.buffer.device,
        device_id,
        "attention output",
    )?;
    let ml_buffer = alloc_buffer(
        &stream,
        DTypeId::F64,
        ml_numel,
        &q.buffer.device,
        device_id,
        "attention row statistics",
    )?;
    let rows = (g.batch as u64) * (g.heads_q as u64) * (g.seq_q as u64);
    if rows == 0 {
        return Ok((
            CudaStorage::new(
                Arc::new(out_buffer),
                alloc::vec![g.batch, g.heads_q, g.seq_q, g.head_dim],
            ),
            CudaStorage::new(
                Arc::new(ml_buffer),
                alloc::vec![g.batch, g.heads_q, g.seq_q, 2],
            ),
        ));
    }
    // The tiled kernel covers QBLOCK query rows per block, so the grid is
    // the per-head tile count rounded up, not the row count.
    let tiles = if tiled {
        g.seq_q.div_ceil(QBLOCK)
    } else {
        g.seq_q
    };
    let blocks = (g.batch as u64) * (g.heads_q as u64) * (tiles as u64);
    let config = launch_config(blocks, g.head_dim, if tiled { QBLOCK } else { 1 })?;
    let (b, hq, hkv, sq, skv, d) = (
        checked_i32(g.batch, "batch")?,
        checked_i32(g.heads_q, "query heads")?,
        checked_i32(g.heads_kv, "kv heads")?,
        checked_i32(g.seq_q, "query seq")?,
        checked_i32(g.seq_kv, "kv seq")?,
        checked_i32(g.head_dim, "head dim")?,
    );
    let groups = checked_i32(g.groups, "groups")?;
    let kv_lead = checked_i32(g.kv_lead, "kv lead")?;
    let causal_i32 = i32::from(causal);
    let (q_off, k_off, v_off, o_off) = (
        as_i64(q.offset_elements, "query offset")?,
        as_i64(k.offset_elements, "key offset")?,
        as_i64(v.offset_elements, "value offset")?,
        0i64,
    );
    let scale_f64 = g.scale;
    let mut out_buffer = out_buffer;
    let mut ml_buffer = ml_buffer;
    // SAFETY: Launches the attention forward over fresh output and
    // statistics allocations with validated geometry.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention output unexpectedly shared".into()))?;
        let ml_u8 = Arc::get_mut(&mut ml_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention statistics unexpectedly shared".into()))?;
        stream
            .launch_builder(&function)
            .arg(&*q.buffer.data)
            .arg(&*k.buffer.data)
            .arg(&*v.buffer.data)
            .arg(&mut *out_u8)
            .arg(&mut *ml_u8)
            .arg(&b)
            .arg(&hq)
            .arg(&hkv)
            .arg(&sq)
            .arg(&skv)
            .arg(&d)
            .arg(&groups)
            .arg(&kv_lead)
            .arg(&scale_f64)
            .arg(&causal_i32)
            .arg(&q_off)
            .arg(&k_off)
            .arg(&v_off)
            .arg(&o_off)
            .launch(config)
            .map_err(|e| {
                Error::Msg(alloc::format!(
                    "CUDA attention forward launch failed: {e:?}"
                ))
            })?;
    }
    Ok((
        CudaStorage::new(
            Arc::new(out_buffer),
            alloc::vec![g.batch, g.heads_q, g.seq_q, g.head_dim],
        ),
        CudaStorage::new(
            Arc::new(ml_buffer),
            alloc::vec![g.batch, g.heads_q, g.seq_q, 2],
        ),
    ))
}

/// Per-row `dot(dO, O)` over `[B, Hq, Sq]`, hoisted out of the dk/dv kernel.
#[cfg(feature = "cuda")]
pub(crate) fn launch_attention_delta(grad: &CudaStorage, out: &CudaStorage) -> Result<CudaStorage> {
    let shape = grad.shape.to_vec();
    let (b, hq, sq, d) = (shape[0], shape[1], shape[2], shape[3]);
    let grad_dtype = grad.buffer.dtype.builtin_id().ok_or(Error::Msg(
        "attention delta needs a builtin grad dtype".into(),
    ))?;
    if !matches!(
        grad_dtype,
        DTypeId::F32 | DTypeId::F64 | DTypeId::F16 | DTypeId::BF16
    ) {
        return Err(Error::UnsupportedDType {
            dtype: grad.buffer.dtype,
            backend: "Cuda",
            op: "fused_attention",
        });
    }
    let device_id = grad.buffer.device_id;
    ensure_attention_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let entry = attn_entry("attn_delta", grad_dtype)?;
    let function = dispatcher.get_function("attention", entry)?;
    let stream = grad.buffer.device.default_stream();
    let rows = (b as u64) * (hq as u64) * (sq as u64);
    let delta_buffer = alloc_buffer(
        &stream,
        DTypeId::F64,
        b * hq * sq,
        &grad.buffer.device,
        device_id,
        "attention delta",
    )?;
    if rows == 0 {
        return Ok(CudaStorage::new(
            Arc::new(delta_buffer),
            alloc::vec![b, hq, sq],
        ));
    }
    let config = launch_config(rows, d, 1)?;
    let (b32, hq32, sq32, d32) = (
        checked_i32(b, "batch")?,
        checked_i32(hq, "query heads")?,
        checked_i32(sq, "query seq")?,
        checked_i32(d, "head dim")?,
    );
    let (g_off, o_off) = (
        as_i64(grad.offset_elements, "grad offset")?,
        as_i64(out.offset_elements, "output offset")?,
    );
    let mut delta_buffer = delta_buffer;
    // SAFETY: Launches the delta reduction over fresh statistics storage.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let delta_u8 = Arc::get_mut(&mut delta_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention delta unexpectedly shared".into()))?;
        stream
            .launch_builder(&function)
            .arg(&*grad.buffer.data)
            .arg(&*out.buffer.data)
            .arg(&mut *delta_u8)
            .arg(&b32)
            .arg(&hq32)
            .arg(&sq32)
            .arg(&d32)
            .arg(&g_off)
            .arg(&o_off)
            .launch(config)
            .map_err(|e| Error::Msg(alloc::format!("CUDA attention delta launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(
        Arc::new(delta_buffer),
        alloc::vec![b, hq, sq],
    ))
}

/// dq rows, one block per query row: exclusive writes, no atomics.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_dq(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    grad: &CudaStorage,
    ml: &CudaStorage,
    delta: &CudaStorage,
    g: &Geometry,
    causal: bool,
) -> Result<CudaStorage> {
    let dtype = g.dtype;
    let device_id = q.buffer.device_id;
    ensure_attention_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let entry = attn_entry("attn_dq", g.dtype)?;
    let function = dispatcher.get_function("attention", entry)?;
    let stream = q.buffer.device.default_stream();
    let numel = g.batch * g.heads_q * g.seq_q * g.head_dim;
    let dq_buffer = alloc_buffer(
        &stream,
        dtype,
        numel,
        &q.buffer.device,
        device_id,
        "attention dq",
    )?;
    let rows = (g.batch as u64) * (g.heads_q as u64) * (g.seq_q as u64);
    if rows == 0 {
        return Ok(CudaStorage::new(
            Arc::new(dq_buffer),
            alloc::vec![g.batch, g.heads_q, g.seq_q, g.head_dim],
        ));
    }
    let config = launch_config(rows, g.head_dim, 1)?;
    let (b, hq, hkv, sq, skv, d) = (
        checked_i32(g.batch, "batch")?,
        checked_i32(g.heads_q, "query heads")?,
        checked_i32(g.heads_kv, "kv heads")?,
        checked_i32(g.seq_q, "query seq")?,
        checked_i32(g.seq_kv, "kv seq")?,
        checked_i32(g.head_dim, "head dim")?,
    );
    let groups = checked_i32(g.groups, "groups")?;
    let kv_lead = checked_i32(g.kv_lead, "kv lead")?;
    let causal_i32 = i32::from(causal);
    let (q_off, k_off, v_off, g_off, dq_off) = (
        as_i64(q.offset_elements, "query offset")?,
        as_i64(k.offset_elements, "key offset")?,
        as_i64(v.offset_elements, "value offset")?,
        as_i64(grad.offset_elements, "grad offset")?,
        0i64,
    );
    let scale_f64 = g.scale;
    let mut dq_buffer = dq_buffer;
    // SAFETY: Launches the dq kernel over a fresh gradient allocation.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let dq_u8 = Arc::get_mut(&mut dq_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention dq unexpectedly shared".into()))?;
        stream
            .launch_builder(&function)
            .arg(&*q.buffer.data)
            .arg(&*k.buffer.data)
            .arg(&*v.buffer.data)
            .arg(&*grad.buffer.data)
            .arg(&*ml.buffer.data)
            .arg(&*delta.buffer.data)
            .arg(&mut *dq_u8)
            .arg(&b)
            .arg(&hq)
            .arg(&hkv)
            .arg(&sq)
            .arg(&skv)
            .arg(&d)
            .arg(&groups)
            .arg(&kv_lead)
            .arg(&scale_f64)
            .arg(&causal_i32)
            .arg(&q_off)
            .arg(&k_off)
            .arg(&v_off)
            .arg(&g_off)
            .arg(&dq_off)
            .launch(config)
            .map_err(|e| Error::Msg(alloc::format!("CUDA attention dq launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(
        Arc::new(dq_buffer),
        alloc::vec![g.batch, g.heads_q, g.seq_q, g.head_dim],
    ))
}

/// dk/dv rows, one block per key/value row: exclusive writes, no atomics.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_attention_dkv(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    grad: &CudaStorage,
    ml: &CudaStorage,
    delta: &CudaStorage,
    g: &Geometry,
    causal: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    let dtype = g.dtype;
    let device_id = q.buffer.device_id;
    ensure_attention_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let entry = attn_entry("attn_dkv", g.dtype)?;
    let function = dispatcher.get_function("attention", entry)?;
    let stream = q.buffer.device.default_stream();
    let numel = g.batch * g.heads_kv * g.seq_kv * g.head_dim;
    let dk_buffer = alloc_buffer(
        &stream,
        dtype,
        numel,
        &q.buffer.device,
        device_id,
        "attention dk",
    )?;
    let dv_buffer = alloc_buffer(
        &stream,
        dtype,
        numel,
        &q.buffer.device,
        device_id,
        "attention dv",
    )?;
    let rows = (g.batch as u64) * (g.heads_kv as u64) * (g.seq_kv as u64);
    if rows == 0 {
        return Ok((
            CudaStorage::new(
                Arc::new(dk_buffer),
                alloc::vec![g.batch, g.heads_kv, g.seq_kv, g.head_dim],
            ),
            CudaStorage::new(
                Arc::new(dv_buffer),
                alloc::vec![g.batch, g.heads_kv, g.seq_kv, g.head_dim],
            ),
        ));
    }
    let config = launch_config(rows, g.head_dim, 1)?;
    let (b, hq, hkv, sq, skv, d) = (
        checked_i32(g.batch, "batch")?,
        checked_i32(g.heads_q, "query heads")?,
        checked_i32(g.heads_kv, "kv heads")?,
        checked_i32(g.seq_q, "query seq")?,
        checked_i32(g.seq_kv, "kv seq")?,
        checked_i32(g.head_dim, "head dim")?,
    );
    let groups = checked_i32(g.groups, "groups")?;
    let kv_lead = checked_i32(g.kv_lead, "kv lead")?;
    let causal_i32 = i32::from(causal);
    let (q_off, k_off, v_off, g_off, dk_off, dv_off) = (
        as_i64(q.offset_elements, "query offset")?,
        as_i64(k.offset_elements, "key offset")?,
        as_i64(v.offset_elements, "value offset")?,
        as_i64(grad.offset_elements, "grad offset")?,
        0i64,
        0i64,
    );
    let scale_f64 = g.scale;
    let mut dk_buffer = dk_buffer;
    let mut dv_buffer = dv_buffer;
    // SAFETY: Launches the dk/dv kernel over fresh gradient allocations.
    unsafe {
        use cudarc::driver::PushKernelArg;
        let dk_u8 = Arc::get_mut(&mut dk_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention dk unexpectedly shared".into()))?;
        let dv_u8 = Arc::get_mut(&mut dv_buffer.data)
            .ok_or_else(|| Error::Msg("CUDA attention dv unexpectedly shared".into()))?;
        stream
            .launch_builder(&function)
            .arg(&*q.buffer.data)
            .arg(&*k.buffer.data)
            .arg(&*v.buffer.data)
            .arg(&*grad.buffer.data)
            .arg(&*ml.buffer.data)
            .arg(&*delta.buffer.data)
            .arg(&mut *dk_u8)
            .arg(&mut *dv_u8)
            .arg(&b)
            .arg(&hq)
            .arg(&hkv)
            .arg(&sq)
            .arg(&skv)
            .arg(&d)
            .arg(&groups)
            .arg(&kv_lead)
            .arg(&scale_f64)
            .arg(&causal_i32)
            .arg(&q_off)
            .arg(&k_off)
            .arg(&v_off)
            .arg(&g_off)
            .arg(&dk_off)
            .arg(&dv_off)
            .launch(config)
            .map_err(|e| Error::Msg(alloc::format!("CUDA attention dkv launch failed: {e:?}")))?;
    }
    Ok((
        CudaStorage::new(
            Arc::new(dk_buffer),
            alloc::vec![g.batch, g.heads_kv, g.seq_kv, g.head_dim],
        ),
        CudaStorage::new(
            Arc::new(dv_buffer),
            alloc::vec![g.batch, g.heads_kv, g.seq_kv, g.head_dim],
        ),
    ))
}
