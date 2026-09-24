//! Wires `kernels/matmul.cu`'s tiled shared-memory GEMM (`BM=128, BN=128,
//! BK=8, TM=8, TN=8`, 16x16 thread blocks) into the CUDA backend: the
//! unbatched 2D launcher below and, since issue #85, a native batched
//! launcher that serves the whole batch as one grid-z launch with explicit
//! per-slice element strides (a stride of 0 reads one matrix for every
//! slice - the batch-broadcast case), instead of composing a per-slice
//! launch loop. Every float storage dtype the kernel exports an entry point
//! for: `f32`/`f64` accumulate in their own type, `f16`/`bf16` hold
//! half-precision operands and accumulate in `f32`. Also since issue #85,
//! plain `f32` requests are offered to the cuBLASLt path in
//! `cuda/ops/cublaslt.rs` first - that module exists only under the
//! `cuda-vendor` feature, which gates both the plain and the batched
//! vendor attempt - and it either serves the request or reports that it
//! does not fit; every other request, and every cuBLASLt failure, reaches
//! the kernels below unchanged. Requests neither path admits fall to the
//! composed per-slice loop in `cuda/backend/shape_ops.rs`, which computes
//! the same product from tape-tracked reshapes around the 2D launcher.

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
///
/// Issue #85 dispatch: under `cuda-vendor`, a plain request that fits
/// `cuda/ops/cublaslt.rs`'s policy is served by cuBLASLt; anything
/// else - a non-`f32` dtype, a strided or offset view, a zero extent, or
/// any cuBLASLt failure - falls through to `kernels/matmul.cu` below,
/// which computes the identical product. The fallback is what keeps the
/// fast path advisory: no request can regress by cuBLASLt being absent or
/// refusing a shape, it merely misses the fusion opportunity. This
/// function carries no epilogue parameters; bias/activation requests reach
/// `cublaslt::try_launch_matmul` only from its own callers.
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
    // Issue #85: canonical f32 products go to cuBLASLt first when the
    // `cuda-vendor` feature is on. The dtype check above already ran, so a
    // mixed pair never reaches it, and a non-f32 dtype is refused by
    // `gemm_request_fits` rather than by an error - `Ok(None)` and `Err`
    // alike fall through to the kernel path, which is why the vendor
    // result collapses to an `Option` here. Without the feature the module
    // does not exist and every request takes the kernel path directly.
    #[cfg(feature = "cuda-vendor")]
    let vendor = super::cublaslt::try_launch_matmul(lhs, rhs, None, None)
        .ok()
        .flatten();
    #[cfg(not(feature = "cuda-vendor"))]
    let vendor: Option<CudaStorage> = None;
    if let Some(product) = vendor {
        return Ok(product);
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

/// The batched kernel entry point for a storage dtype, or a typed refusal
/// for anything `matmul.cu` does not export a batched name for. Same
/// fail-closed contract as [`matmul_entry_point`]: the launcher launches
/// exactly this name, and the batched plan refuses (rather than errors) on
/// a dtype without one so the composed loop reaches the 2D launcher's own
/// refusal with its established wording.
#[cfg(feature = "cuda")]
fn matmul_batched_entry_point(dtype: DTypeDescriptor) -> Result<&'static str> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok("matmul_batched"),
        Some(DTypeId::F64) => Ok("matmul_batched_f64"),
        Some(DTypeId::F16) => Ok("matmul_batched_f16"),
        Some(DTypeId::BF16) => Ok("matmul_batched_bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "matmul",
        }),
    }
}

/// CUDA's grid-z limit, and therefore the largest batch the batched kernel
/// can address: `blockIdx.z` names the slice, so a larger batch would wrap
/// rather than launch.
#[cfg(feature = "cuda")]
const MAX_BATCHED_GRID_Z: usize = 65_535;

/// Host-side admission and geometry for one native batched launch, decided
/// entirely without touching the device. `None` from the plan is not an
/// error: it means "not this path", and the caller keeps the composed
/// per-slice loop, which computes the same product from tape-tracked
/// reshapes around the 2D kernel. Private to this module - the launch
/// consumes a plan the tests build directly from [`super::OperandMeta`].
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchedGemmPlan {
    /// The kernel name `matmul.cu` exports for `dtype`, selected by the
    /// same fail-closed map the 2D launcher uses.
    entry_point: &'static str,
    dtype: DTypeDescriptor,
    device_id: usize,
    /// Broadcast batch dims with `[m, n]` appended - the product's shape.
    out_shape: alloc::vec::Vec<usize>,
    /// Total flattened slices (the product of the batch dims), which is
    /// also the grid's z extent.
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    /// Elements between consecutive flat slices of `lhs` / `rhs`: the
    /// affine constant `c` the check below derives from the operand's own
    /// batch strides, or 0 when the operand reads one matrix for the whole
    /// batch (rank deficit or size-1 axes).
    stride_a: usize,
    stride_b: usize,
}

/// Row-major trailing matrix: for some `rows >= 1`, the last two axes of
/// `shape` are a dense `[rows, cols]` block (`strides[-1] == 1`,
/// `strides[-2] == cols`) and `strides` has one entry per axis. The
/// batched kernel indexes every slice as exactly such a flat block, so a
/// padded or transposed matrix view is refused rather than re-strided.
#[cfg(feature = "cuda")]
fn trailing_matrix_row_major(operand: &super::OperandMeta<'_>) -> bool {
    let rank = operand.shape.len();
    rank >= 2
        && operand.strides.len() == rank
        && operand.strides[rank - 1] == 1
        && operand.strides[rank - 2] == operand.shape[rank - 1]
}

/// The shared flat-slice stride `c` (in elements) if every out-batch axis
/// with extent greater than 1 is affine in the flat batch index: for each
/// such axis `d`, the operand's stride there must equal `c * F_d`, where
/// `F_d` is the product of the out extents after `d` (the odometer
/// multiplier of the row-major flat index). Axes the operand does not vary
/// along - it is shorter than the output (padding, right-aligned) or its
/// own extent is 1 - read nothing, so their stride counts as 0 and must
/// satisfy the same equation, which is what refuses a broadcast the flat
/// kernel cannot express (it would force `c = 0` while a later varying
/// axis needs `c > 0`, or vice versa). The first constrained axis whose
/// stride is divisible by its `F_d` fixes `c = stride / F_d`; every later
/// one must produce the same `c`. No constrained axis leaves `c = 0`.
/// Returns `None` on any disagreement - a launch there would mis-stride,
/// so the composed loop takes over instead.
#[cfg(feature = "cuda")]
fn affine_batch_stride(
    out_batch: &[usize],
    operand_batch: &[usize],
    operand_strides: &[usize],
) -> Option<usize> {
    debug_assert!(operand_batch.len() <= out_batch.len());
    debug_assert_eq!(operand_batch.len(), operand_strides.len());
    let deficit = out_batch.len() - operand_batch.len();
    let mut shared: Option<usize> = None;
    for d in 0..out_batch.len() {
        if out_batch[d] == 1 {
            // Vacuous: the coordinate is fixed at 0 for every operand, so
            // no stride is observed here and no constraint applies.
            continue;
        }
        // `F_d`: the flat odometer multiplier for axis `d`. Nonzero by
        // admission - the plan refuses a zero batch before reaching here,
        // so every trailing product is a product of extents >= 1.
        let factor: usize = out_batch[d + 1..].iter().product();
        let stride = if d < deficit || operand_batch[d - deficit] == 1 {
            // Padding (the operand is shorter than the output) and
            // size-1 axes both read nothing, so their stride is 0. `||`
            // short-circuits before the subtraction, keeping the index
            // in range on the padding side.
            0
        } else {
            operand_strides[d - deficit]
        };
        if stride % factor != 0 {
            return None;
        }
        let candidate = stride / factor;
        match shared {
            None => shared = Some(candidate),
            Some(expected) if expected == candidate => {}
            Some(_) => return None,
        }
    }
    Some(shared.unwrap_or(0))
}

/// The batched admission policy and launch geometry, pure and host-side.
///
/// Admits a request only when every one of the following holds - anything
/// else returns `None` and the composed loop serves the request exactly as
/// it did before issue #85, so a refusal is never a regression:
///
/// - both operands share one dtype, and that dtype has a batched kernel
///   entry (`f32`/`f64`/`f16`/`bf16`; an integer or quantized pair falls
///   through to the 2D launcher's typed refusal via the composed loop),
/// - both operands live on the same device at offset zero,
/// - both have rank at least 2 with matching inner dimensions, no zero
///   extent, and row-major trailing matrix strides (see
///   [`trailing_matrix_row_major`]),
/// - the batch dims broadcast to a nonzero total that fits CUDA's grid-z
///   limit,
/// - each operand's batch strides are affine in the flat batch index with
///   one shared constant per operand (see [`affine_batch_stride`]) - this
///   admits contiguous batches, non-contiguous batch views with a constant
///   slice stride, and broadcast axes the flat kernel expresses as stride
///   0, while refusing any stride pattern a single `stride * slice` cannot
///   reproduce,
/// - every launch parameter the kernel and grid take (`m`/`k`/`n` as
///   `int`, the strides as `long long`, the grid extents as `u32`)
///   converts without overflow.
#[cfg(feature = "cuda")]
fn batched_gemm_plan_meta(
    lhs: &super::OperandMeta<'_>,
    rhs: &super::OperandMeta<'_>,
) -> Option<BatchedGemmPlan> {
    if lhs.dtype != rhs.dtype {
        return None;
    }
    let entry_point = matmul_batched_entry_point(lhs.dtype).ok()?;
    if lhs.device_id != rhs.device_id {
        return None;
    }
    if lhs.offset != 0 || rhs.offset != 0 {
        return None;
    }
    if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
        return None;
    }
    if !trailing_matrix_row_major(lhs) || !trailing_matrix_row_major(rhs) {
        return None;
    }
    let m = lhs.shape[lhs.shape.len() - 2];
    let k = lhs.shape[lhs.shape.len() - 1];
    let rhs_k = rhs.shape[rhs.shape.len() - 2];
    let n = rhs.shape[rhs.shape.len() - 1];
    if k != rhs_k || m == 0 || k == 0 || n == 0 {
        return None;
    }

    let lhs_batch = &lhs.shape[..lhs.shape.len() - 2];
    let rhs_batch = &rhs.shape[..rhs.shape.len() - 2];
    // A broadcast mismatch is not this path's error to report: `None`
    // hands the request to the composed loop, whose own `broadcast_shape`
    // raises the framework's established `ShapeMismatch`.
    let out_batch = crate::layout::broadcast_shape(lhs_batch, rhs_batch).ok()?;
    let batch = out_batch
        .iter()
        .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))?;
    if batch == 0 || batch > MAX_BATCHED_GRID_Z {
        return None;
    }

    let stride_a = affine_batch_stride(&out_batch, lhs_batch, &lhs.strides[..lhs_batch.len()])?;
    let stride_b = affine_batch_stride(&out_batch, rhs_batch, &rhs.strides[..rhs_batch.len()])?;

    // Launch representability, kept in the plan so the launch's own
    // checked conversions cannot fail on an admitted request.
    crate::cuda::checked_u32(m, "CUDA batched matmul row grid dimension").ok()?;
    crate::cuda::checked_u32(n, "CUDA batched matmul column grid dimension").ok()?;
    crate::cuda::checked_i32(m, "CUDA batched matmul row count").ok()?;
    crate::cuda::checked_i32(k, "CUDA batched matmul inner dimension").ok()?;
    crate::cuda::checked_i32(n, "CUDA batched matmul column count").ok()?;
    let stride_c = m.checked_mul(n)?;
    i64::try_from(stride_a).ok()?;
    i64::try_from(stride_b).ok()?;
    i64::try_from(stride_c).ok()?;

    let mut out_shape = out_batch;
    out_shape.extend_from_slice(&[m, n]);
    Some(BatchedGemmPlan {
        entry_point,
        dtype: lhs.dtype,
        device_id: lhs.device_id,
        out_shape,
        batch,
        m,
        k,
        n,
        stride_a,
        stride_b,
    })
}

/// [`batched_gemm_plan_meta`] over real storages - the device-facing half
/// of the pure policy above, so tests can exercise the plan without a GPU
/// while the launch still reads one checked record.
#[cfg(feature = "cuda")]
fn batched_gemm_plan(lhs: &CudaStorage, rhs: &CudaStorage) -> Option<BatchedGemmPlan> {
    batched_gemm_plan_meta(&super::OperandMeta::of(lhs), &super::OperandMeta::of(rhs))
}

/// One grid-z launch of `matmul.cu`'s batched kernel for an admitted
/// [`BatchedGemmPlan`]. The grid covers `(ceil(n/BN), ceil(m/BM), batch)`
/// blocks - each `(x, y)` block computes one `[m, n]` tile of one slice,
/// and `blockIdx.z` names the slice - with the plan's per-slice element
/// strides as the remaining arguments. The output is a fresh contiguous
/// allocation, so its slice stride is exactly `m * n`.
///
/// # Errors
///
/// Propagates an allocation or launch failure. Admission already
/// guaranteed every checked conversion here succeeds, so an `Err` means
/// the device itself failed - there is no silent partial batch.
#[cfg(feature = "cuda")]
fn launch_batched_native(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    plan: &BatchedGemmPlan,
) -> Result<CudaStorage> {
    let (lhs_buf, rhs_buf) = (&*lhs.buffer, &*rhs.buffer);
    debug_assert_eq!(lhs_buf.dtype, plan.dtype);
    debug_assert_eq!(rhs_buf.dtype, plan.dtype);
    debug_assert_eq!(lhs_buf.device_id, plan.device_id);
    let device_id = plan.device_id;
    ensure_matmul_loaded(device_id)?;

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("matmul", plan.entry_point)?;
    let stream = lhs_buf.device.default_stream();

    let (batch, m, k, n) = (plan.batch, plan.m, plan.k, plan.n);
    let out_shape = plan.out_shape.clone();
    let total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;

    let mut out_b = CudaBuffer {
        len: total,
        dtype: plan.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            plan.dtype,
            total,
            OperationKind::MatMul,
        )?),
        device: lhs_buf.device.clone(),
        device_id,
    };

    let m_u32 = crate::cuda::checked_u32(m, "CUDA batched matmul row grid dimension")?;
    let n_u32 = crate::cuda::checked_u32(n, "CUDA batched matmul column grid dimension")?;
    let batch_u32 = crate::cuda::checked_u32(batch, "CUDA batched matmul batch grid dimension")?;
    let m_i32 = crate::cuda::checked_i32(m, "CUDA batched matmul row count")?;
    let k_i32 = crate::cuda::checked_i32(k, "CUDA batched matmul inner dimension")?;
    let n_i32 = crate::cuda::checked_i32(n, "CUDA batched matmul column count")?;
    let stride_c = m.checked_mul(n).ok_or_else(|| {
        Error::Msg(format!(
            "CUDA batched matmul output slice stride overflows usize: {m} * {n}"
        ))
    })?;
    let stride_a = i64::try_from(plan.stride_a).map_err(|_| {
        Error::Msg(format!(
            "CUDA batched matmul lhs slice stride out of i64 range: {}",
            plan.stride_a
        ))
    })?;
    let stride_b = i64::try_from(plan.stride_b).map_err(|_| {
        Error::Msg(format!(
            "CUDA batched matmul rhs slice stride out of i64 range: {}",
            plan.stride_b
        ))
    })?;
    let stride_c = i64::try_from(stride_c).map_err(|_| {
        Error::Msg(format!(
            "CUDA batched matmul output slice stride out of i64 range: {m} * {n}"
        ))
    })?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_u32.div_ceil(BN), m_u32.div_ceil(BM), batch_u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: the plan's admission established that each operand's trailing
    // matrix is a dense row-major block and that its batch strides are
    // affine in the flat index, so `A + b * strideA` (likewise B, C) is
    // exactly slice `b` of each operand for every `b < batch`, and the
    // launch extent covers every output element once. out_b was just
    // allocated and is uniquely owned. The views pass through as byte
    // buffers with the entry point selected from the same dtype the plan
    // admitted, following the byte-buffer convention documented on
    // `launch_matmul`.
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
            .arg(&stride_a)
            .arg(&stride_b)
            .arg(&stride_c)
            .launch(cfg)
            .map_err(|e| Error::Msg(format!("batched matmul launch failed: {e:?}")))?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

/// The native batched path alone: plan, then one kernel launch.
///
/// # Return contract
///
/// - `Ok(Some(product))` - one grid-z launch computed the whole batch.
/// - `Ok(None)` - the plan refused (see [`batched_gemm_plan_meta`]); the
///   caller keeps the composed loop, which computes the same product.
/// - `Err(..)` - the plan admitted the request but the device failed.
///   Propagates: the native path is the one this module owns, so a
///   hardware failure is reported rather than papered over.
#[cfg(feature = "cuda")]
pub(crate) fn try_launch_batched_native(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<Option<CudaStorage>> {
    let Some(plan) = batched_gemm_plan(lhs, rhs) else {
        return Ok(None);
    };
    launch_batched_native(lhs, rhs, &plan).map(Some)
}

/// Issue #85's batched orchestrator: the `cuda-vendor` cuBLASLt attempt
/// first (a strided-batched call with no launch loop of its own), then the
/// native batched kernel, then `Ok(None)` for the composed per-slice loop
/// in `shape_ops`. Vendor refusals *and* vendor failures both fall
/// through: on an epilogue-free request the native kernels compute the
/// identical product, so falling through can only change which
/// implementation runs, never the values - which is what makes the vendor
/// path advisory. Without `cuda-vendor` the vendor module does not exist
/// and the `None` binding takes its place, keeping the native-first order
/// intact.
///
/// # Errors
///
/// Only from the native launch (see [`try_launch_batched_native`]): a plan
/// refusal is `Ok(None)`, not an error.
#[cfg(feature = "cuda")]
pub(crate) fn try_launch_batched_matmul(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<Option<CudaStorage>> {
    #[cfg(feature = "cuda-vendor")]
    let vendor = super::cublaslt::try_launch_batched_matmul(lhs, rhs)
        .ok()
        .flatten();
    #[cfg(not(feature = "cuda-vendor"))]
    let vendor: Option<CudaStorage> = None;
    if let Some(product) = vendor {
        return Ok(Some(product));
    }
    try_launch_batched_native(lhs, rhs)
}

/// Pure entry-point selection: runs wherever the `cuda` feature is on,
/// with no device required. The launcher launches exactly the name
/// `matmul_entry_point` returns, so proving the mapping here is proving a
/// buffer can never reach a kernel typed for a different dtype.
#[cfg(all(test, feature = "cuda"))]
mod entry_point_tests {
    use super::matmul_entry_point;
    use incin_core::error::Error;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn matmul_entry_point_maps_each_float_and_refuses_the_rest() {
        assert_eq!(
            matmul_entry_point(DTypeId::F32.descriptor()).unwrap(),
            "matmul"
        );
        assert_eq!(
            matmul_entry_point(DTypeId::F64.descriptor()).unwrap(),
            "matmul_f64"
        );
        assert_eq!(
            matmul_entry_point(DTypeId::F16.descriptor()).unwrap(),
            "matmul_f16"
        );
        assert_eq!(
            matmul_entry_point(DTypeId::BF16.descriptor()).unwrap(),
            "matmul_bf16"
        );

        for id in [
            DTypeId::I64,
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::Q8_0,
        ] {
            let error = match matmul_entry_point(id.descriptor()) {
                Ok(name) => panic!("{id:?} must be refused, got entry point {name:?}"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    error,
                    Error::UnsupportedDType {
                        backend: "Cuda",
                        op: "matmul",
                        ..
                    }
                ),
                "{id:?}: expected UnsupportedDType naming Cuda/matmul, got {error:?}"
            );
        }
    }
}

/// The batched half of the entry-point contract: the launcher launches
/// exactly the name `matmul_batched_entry_point` returns, so proving the
/// mapping here is proving a buffer can never reach a batched kernel typed
/// for a different dtype. Runs wherever `cuda` is on, with no device.
#[cfg(all(test, feature = "cuda"))]
mod batched_entry_point_tests {
    use super::matmul_batched_entry_point;
    use incin_core::error::Error;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn batched_entry_point_maps_each_float_and_refuses_the_rest() {
        assert_eq!(
            matmul_batched_entry_point(DTypeId::F32.descriptor()).unwrap(),
            "matmul_batched"
        );
        assert_eq!(
            matmul_batched_entry_point(DTypeId::F64.descriptor()).unwrap(),
            "matmul_batched_f64"
        );
        assert_eq!(
            matmul_batched_entry_point(DTypeId::F16.descriptor()).unwrap(),
            "matmul_batched_f16"
        );
        assert_eq!(
            matmul_batched_entry_point(DTypeId::BF16.descriptor()).unwrap(),
            "matmul_batched_bf16"
        );

        for id in [
            DTypeId::I64,
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::Q8_0,
        ] {
            let error = match matmul_batched_entry_point(id.descriptor()) {
                Ok(name) => panic!("{id:?} must be refused, got entry point {name:?}"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    error,
                    Error::UnsupportedDType {
                        backend: "Cuda",
                        op: "matmul",
                        ..
                    }
                ),
                "{id:?}: expected UnsupportedDType naming Cuda/matmul, got {error:?}"
            );
        }
    }
}

/// The pure batched admission policy (issue #85): every shape the plan
/// admits launches one correct grid-z batch, and every shape it refuses is
/// one the composed loop already serves. No device required - the tests
/// build [`super::OperandMeta`] records directly, which is why the plan
/// takes metas rather than storages.
#[cfg(all(test, feature = "cuda"))]
mod batched_plan_tests {
    use super::{BatchedGemmPlan, affine_batch_stride, batched_gemm_plan_meta};
    use crate::cuda::ops::OperandMeta;
    use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

    fn meta<'a>(
        dtype: DTypeDescriptor,
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> OperandMeta<'a> {
        OperandMeta {
            dtype,
            device_id: 0,
            shape,
            strides,
            offset,
        }
    }

    fn f32() -> DTypeDescriptor {
        DTypeId::F32.descriptor()
    }

    fn plan(lhs: &OperandMeta<'_>, rhs: &OperandMeta<'_>) -> Option<BatchedGemmPlan> {
        batched_gemm_plan_meta(lhs, rhs)
    }

    #[test]
    fn plan_admits_a_contiguous_rank_three_pair_and_derives_slice_strides() {
        // [2,3,4] x [2,4,5] -> [2,3,5]; the flat slice strides are the
        // contiguous ones: 3*4 = 12 and 4*5 = 20.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        let plan = plan(&lhs, &rhs).expect("the canonical pair must be admitted");
        assert_eq!(plan.entry_point, "matmul_batched");
        assert_eq!(plan.dtype, f32());
        assert_eq!(plan.device_id, 0);
        assert_eq!(plan.out_shape, alloc::vec![2, 3, 5]);
        assert_eq!(plan.batch, 2);
        assert_eq!((plan.m, plan.k, plan.n), (3, 4, 5));
        assert_eq!(plan.stride_a, 12, "lhs slice stride is m * k");
        assert_eq!(plan.stride_b, 20, "rhs slice stride is k * n");
    }

    #[test]
    fn plan_admits_a_contiguous_rank_four_pair_with_a_shared_affine_constant() {
        // [2,2,3,4] x [2,2,4,5] -> [2,2,3,5]. Flat slice index is
        // i * 2 + j; the lhs offset i*24 + j*12 is (i*2 + j) * 12, so one
        // constant serves both batch axes: stride 12, not 24.
        let lhs = meta(f32(), &[2, 2, 3, 4], &[24, 12, 4, 1], 0);
        let rhs = meta(f32(), &[2, 2, 4, 5], &[40, 20, 5, 1], 0);
        let plan = plan(&lhs, &rhs).expect("the rank-four pair must be admitted");
        assert_eq!(plan.out_shape, alloc::vec![2, 2, 3, 5]);
        assert_eq!(plan.batch, 4);
        assert_eq!(plan.stride_a, 12);
        assert_eq!(plan.stride_b, 20);
    }

    #[test]
    fn plan_admits_a_rank_two_operand_as_a_stride_zero_broadcast() {
        // [3,4] x [2,4,5] -> [2,3,5]: the lhs has no batch axis, so it
        // reads the same matrix for every slice - stride 0, no materialized
        // copy.
        let lhs = meta(f32(), &[3, 4], &[4, 1], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        let plan = plan(&lhs, &rhs).expect("a rank-two lhs must broadcast");
        assert_eq!(plan.out_shape, alloc::vec![2, 3, 5]);
        assert_eq!(plan.stride_a, 0, "the broadcast operand reads one matrix");
        assert_eq!(plan.stride_b, 20);
    }

    #[test]
    fn plan_admits_a_size_one_leading_batch_axis_as_a_vacuous_axis() {
        // [1,3,4] x [1,4,5]: out batch [1] - the only batch axis has
        // extent 1, so no stride is observed and any slice stride works.
        let lhs = meta(f32(), &[1, 3, 4], &[999, 4, 1], 0);
        let rhs = meta(f32(), &[1, 4, 5], &[999, 5, 1], 0);
        let plan = plan(&lhs, &rhs).expect("a vacuous batch axis constrains nothing");
        assert_eq!(plan.batch, 1);
        assert_eq!(plan.stride_a, 0, "no constrained axis leaves c = 0");
        assert_eq!(plan.stride_b, 0);
    }

    #[test]
    fn plan_admits_a_non_contiguous_batch_view_via_its_actual_stride() {
        // Same shapes as the rank-three case, but the lhs batch stride is
        // 100 rather than the contiguous 12 - a view into a wider
        // allocation. The plan reads the operand's actual stride, so the
        // kernel jumps 100 elements per slice instead of assuming 12.
        let lhs = meta(f32(), &[2, 3, 4], &[100, 4, 1], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        let plan = plan(&lhs, &rhs).expect("a strided batch view must be admitted");
        assert_eq!(plan.stride_a, 100);
    }

    #[test]
    fn plan_refuses_a_non_affine_batch_broadcast_the_flat_kernel_cannot_express() {
        // [2,1,3,4] x [2,4,4,5] -> out batch [2,4]. The lhs varies along
        // axis 0 with stride 12 while axis 1 must read stride 0 (extent
        // 1); one shared c cannot satisfy both, so the flat-stride kernel
        // would mis-stride. The composed loop keeps serving this shape.
        let lhs = meta(f32(), &[2, 1, 3, 4], &[12, 12, 4, 1], 0);
        let rhs = meta(f32(), &[2, 4, 4, 5], &[80, 20, 5, 1], 0);
        assert!(
            plan(&lhs, &rhs).is_none(),
            "a non-affine batch broadcast must fall to the composed loop"
        );
    }

    #[test]
    fn plan_refuses_a_right_aligned_batch_the_flat_kernel_cannot_express() {
        // [2,3,4] x [2,2,4,5]: broadcast right-aligns the lhs batch [2]
        // against [2,2], so the lhs varies on the axis where the flat
        // index needs c = 0 (padding) while its own stride is 12.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 0);
        let rhs = meta(f32(), &[2, 2, 4, 5], &[40, 20, 5, 1], 0);
        assert!(
            plan(&lhs, &rhs).is_none(),
            "a right-aligned non-affine batch must fall to the composed loop"
        );
    }

    #[test]
    fn plan_refuses_dtype_device_offset_rank_inner_and_zero_admission_misses() {
        // Mismatched dtypes: the composed loop reports the established
        // DTypeMismatch through the 2D launcher instead.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 0);
        let rhs = meta(DTypeId::F64.descriptor(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "dtype mismatch refused");

        // A dtype with no batched kernel entry refuses without an error -
        // the composed loop reaches the 2D launcher's typed refusal.
        let lhs = meta(DTypeId::I64.descriptor(), &[2, 3, 4], &[12, 4, 1], 0);
        let rhs = meta(DTypeId::I64.descriptor(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "i64 refused");

        // Different devices must never share one launch.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 0);
        let mut rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        rhs.device_id = 1;
        assert!(plan(&lhs, &rhs).is_none(), "device mismatch refused");

        // Offset views address into another allocation's element stream;
        // the plan only launches from offset zero.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 3);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "offset lhs refused");

        // Rank below 2 has no matrix axes to multiply.
        let lhs = meta(f32(), &[4], &[1], 0);
        let rhs = meta(f32(), &[4, 5], &[5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "rank below 2 refused");

        // Inner dimensions disagree.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 1], 0);
        let rhs = meta(f32(), &[2, 5, 5], &[25, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "k mismatch refused");

        // A zero extent has no slice to launch (the composed path's own
        // zero-batch reshape stays the answer).
        let lhs = meta(f32(), &[2, 0, 4], &[0, 4, 1], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "zero extent refused");
    }

    #[test]
    fn plan_refuses_a_batch_larger_than_the_grid_z_limit() {
        // 65536 slices: one past CUDA's grid-z maximum. The plan refuses
        // rather than letting blockIdx.z wrap onto earlier slices.
        const OVER: usize = super::MAX_BATCHED_GRID_Z + 1;
        let lhs = meta(f32(), &[OVER, 1, 1], &[1, 1, 1], 0);
        let rhs = meta(f32(), &[OVER, 1, 1], &[1, 1, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "batch over grid z refused");

        // Exactly the limit still fits.
        const AT: usize = super::MAX_BATCHED_GRID_Z;
        let lhs = meta(f32(), &[AT, 1, 1], &[1, 1, 1], 0);
        let rhs = meta(f32(), &[AT, 1, 1], &[1, 1, 1], 0);
        let plan = plan(&lhs, &rhs).expect("a batch of exactly 65535 must fit");
        assert_eq!(plan.batch, AT);
    }

    #[test]
    fn plan_refuses_a_matrix_view_that_is_not_row_major_contiguous() {
        // The batched kernel indexes each slice as a dense row-major
        // block: strides must end in [cols, 1]. A padded interior stride
        // (here [12, 1, 1] for a [.., 3, 4] matrix) would read the wrong
        // elements, so it refuses rather than re-striding.
        let lhs = meta(f32(), &[2, 3, 4], &[12, 1, 1], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(plan(&lhs, &rhs).is_none(), "padded matrix strides refused");

        let lhs = meta(f32(), &[2, 3, 4], &[12, 4, 2], 0);
        let rhs = meta(f32(), &[2, 4, 5], &[20, 5, 1], 0);
        assert!(
            plan(&lhs, &rhs).is_none(),
            "non-unit trailing stride refused"
        );
    }

    #[test]
    fn affine_stride_admits_broadcasts_vacuous_axes_and_refuses_disagreement() {
        // One shared constant across every constrained axis: contiguous
        // [B] slices of size 12.
        assert_eq!(
            affine_batch_stride(&[3], &[3], &[12]),
            Some(12),
            "a single axis sets c from its own stride"
        );

        // Rank-four contiguity: axis 0 contributes 36 with F = 3, axis 1
        // contributes 12 with F = 1 - both give c = 12.
        assert_eq!(
            affine_batch_stride(&[2, 3], &[2, 3], &[36, 12]),
            Some(12),
            "adjacent axes must agree on one c"
        );

        // Disagreement: axis 0 wants c = 4 (12 / 3) while axis 1 wants
        // c = 7 - no single flat stride reproduces both.
        assert_eq!(
            affine_batch_stride(&[3, 4], &[3, 4], &[12, 7]),
            None,
            "axes that disagree on c refuse"
        );

        // A padding axis (operand shorter than the output) reads nothing:
        // stride 0 there forces c = 0, which a later varying axis then
        // violates.
        assert_eq!(
            affine_batch_stride(&[2, 3], &[3], &[12]),
            None,
            "a varying axis under a forced c = 0 refuses"
        );

        // The same shape with an operand that varies on no axis at all
        // (rank-two) is the c = 0 broadcast the kernel does express.
        assert_eq!(
            affine_batch_stride(&[2, 3], &[], &[]),
            Some(0),
            "no constrained axis is the stride-zero broadcast"
        );

        // Size-1 out axes are vacuous: no constraint, whatever the stride.
        assert_eq!(
            affine_batch_stride(&[1, 4], &[1, 4], &[999, 12]),
            Some(12),
            "extent-1 out axes impose nothing"
        );
        assert_eq!(
            affine_batch_stride(&[1], &[], &[]),
            Some(0),
            "an all-vacuous batch is c = 0"
        );

        // An operand size-1 axis under a varying out axis reads nothing
        // (stride 0 -> c = 0) and must be consistent with the rest.
        assert_eq!(
            affine_batch_stride(&[3, 4], &[3, 1], &[12, 6]),
            None,
            "size-1 operand axis needing c = 0 under c = 3 refuses"
        );
        assert_eq!(
            affine_batch_stride(&[3], &[1], &[6]),
            Some(0),
            "an all-broadcast operand is c = 0"
        );
    }
}
