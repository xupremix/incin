//! The cuBLASLt GEMM path for CUDA matrix products (issue #85).
//!
//! `kernels/matmul.cu`'s hand-written tiled kernel computes `f32` products
//! but has no epilogue: a biased linear layer materializes the product and
//! then pays a second full read-modify-write of the output to add the bias.
//! cuBLASLt selects a hardware-tuned algorithm for the shape and can fuse
//! the bias plus an optional `relu`/`gelu` activation into that single
//! kernel. This module owns that path and the policy that decides when it
//! may serve a request.
//!
//! # Dispatch policy (the acceptance contract from #85)
//!
//! [`crate::cuda::ops::matmul::launch_matmul`] tries this path first for
//! every plain (epilogue-free) request. A request *fits* - checked
//! host-side by [`gemm_request_fits`] - only when all of the following
//! hold:
//!
//! - both operands are `f32`, the only dtype wired here; `f16`/`bf16`/`f64`
//!   keep their `matmul.cu` entry points,
//! - both operands live on the same device,
//! - both operands are rank 2 with matching inner dimensions and no zero
//!   extent,
//! - both operands are contiguous at offset zero.
//!
//! Anything that does not fit falls through to `kernels/matmul.cu`.
//! Anything that *does* fit but fails inside cuBLASLt (handle missing from
//! the process, heuristic refusal, launch error) also falls through: the
//! NVRTC kernel computes the identical product, so on an epilogue-free
//! request the fallback is never a numeric divergence - only a different
//! implementation of the same math. Epilogue requests (a bias or an
//! activation passed to [`try_launch_matmul`]) have no such fallback and
//! fail closed with an error rather than silently computing the product
//! *without* the epilogue.
//!
//! # Batched requests (issue #85, `cuda-vendor` only)
//!
//! [`try_launch_batched_matmul`] serves plain rank-3 x rank-3 products with
//! an equal contiguous batch as one strided-batched cuBLASLt call: the
//! matrix layouts carry `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` and
//! `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`, so cuBLASLt iterates the
//! batch itself. Its policy ([`batched_gemm_request_fits`]) is stricter
//! than the native plan's - no batch broadcast (a stride-0 read), no rank
//! deficit - because the native path in `matmul.rs` covers those; this
//! module only claims the shape where the vendor library is the straight
//! win. `Ok(None)` means the request does not fit, and the caller keeps
//! its existing path (the orchestrator then offers the native batched
//! launch, falling back to the composed loop only if that refuses too).
//!
//! # Compute type
//!
//! The matmul descriptor forces `CUBLAS_COMPUTE_32F`. cudarc's own
//! `Matmul<f32>` impl uses `CUBLAS_COMPUTE_32F_FAST_TF32`, which would
//! quietly relax precision below what the CUDA capability rows advertise
//! for matmul (`MathMode::Precise`; see `capability/tables.rs`), so this
//! module drives `cudarc::cublaslt::result` directly instead of the safe
//! wrapper.
//!
//! # Row-major layout
//!
//! cuBLASLt is column-major. Producing row-major `C[M,N] = A[M,K] · B[K,N]`
//! from row-major operand buffers uses the standard `Cᵀ = Bᵀ · Aᵀ` identity:
//! the col-major call runs with `m = N`, `n = M`, `k = K`, first operand =
//! the `rhs` buffer (leading dimension `N`), second operand = the `lhs`
//! buffer (leading dimension `K`), output at leading dimension `N`, all
//! transposes off - so every element lands exactly where row-major indexing
//! expects it. The bias vector of the column-major call has length `m = N`,
//! which broadcasts down the columns of the row-major output: bias per
//! output feature, the linear-layer semantics.
//!
//! # Handle and workspace cache
//!
//! One cudarc [`CudaBlasLT`] handle (which supplies the raw
//! `cublasLtHandle_t` plus cudarc's own `Send`/`Sync` story - this module
//! writes no `unsafe impl`s) and one workspace are cached per device for
//! the lifetime of the process, in the same shape as
//! [`crate::cuda::gpu::cuda_cache`]. cudarc's `Workspace` has no public
//! accessor, so the workspace passed to `cublasLtMatmul` is allocated here
//! alongside cudarc's internal one: two workspaces exist per handle
//! (4 MiB, or 32 MiB on compute capability >= 9, each) rather than either
//! churning a handle per call or hand-rolling `Send` for a raw pointer.
//!
//! # Verification status
//!
//! This module is compile-verified and unit-tests its host-side policy
//! without hardware. The end-to-end tests live as `#[ignore = "requires
//! CUDA hardware"]` cases in [`crate::cuda::backend::tests`] and have not
//! been executed in an environment without a GPU.

use alloc::sync::Arc;
use core::ffi::c_void;
use core::mem;

use cudarc::cublaslt::{
    Activation, CudaBlasLT, MatmulShared,
    result::{self, CublasError},
    sys,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::{OperandMeta, alloc_zeroed_bytes};
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf};
use incin_core::tensor::dtype::DTypeId;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

/// cuBLASLt state per device: the cudarc handle wrapper and this module's
/// own workspace (see the module doc for why two workspaces exist).
struct CublasLtState {
    blas: CudaBlasLT,
    workspace: CudaSlice<u8>,
    workspace_size: usize,
}

/// Process-wide per-device state, mirroring `gpu::cuda_cache`'s cache of
/// contexts and modules: created once, never evicted, because recreating a
/// handle is the expensive event.
static STATES: OnceLock<Mutex<BTreeMap<usize, Arc<CublasLtState>>>> = OnceLock::new();

/// Set once handle creation has failed for a reason that will not change
/// within the process (library not loadable, driver refusing the handle),
/// so later calls stop paying the probe. Shape-specific heuristic or launch
/// failures deliberately do not set it: another shape may still succeed.
static UNAVAILABLE: AtomicBool = AtomicBool::new(false);

// Host-side metadata for the fit policies lives next to their other
// consumer (`matmul`'s native batch plan) in `ops/mod.rs`: one
// `OperandMeta` type, one `of` constructor, both policies reading the
// same borrowed fields.

/// Whether a plain product request fits the cuBLASLt path of issue #85.
///
/// The conjunction is the dispatch policy documented on the module: `f32`
/// on both operands of the same device, rank 2 with matching inner
/// dimensions and no zero extent, contiguous at offset zero. Every request
/// this function refuses keeps the behaviour it had before the path
/// existed - `kernels/matmul.cu` computes it - so refusing is never a
/// regression, only a missed fusion opportunity. Pure and host-side so the
/// policy is unit-testable without a GPU.
pub(crate) fn gemm_request_fits(lhs: &OperandMeta<'_>, rhs: &OperandMeta<'_>) -> bool {
    let f32 = DTypeId::F32.descriptor();
    if lhs.dtype != f32 || rhs.dtype != f32 {
        return false;
    }
    if lhs.device_id != rhs.device_id {
        return false;
    }
    if lhs.offset != 0 || rhs.offset != 0 {
        return false;
    }
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return false;
    }
    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];
    if m == 0 || k == 0 || n == 0 || k != rhs.shape[0] {
        return false;
    }
    is_contiguous_rank_two(lhs) && is_contiguous_rank_two(rhs)
}

/// Row-major contiguity for a rank-2 operand: strides `[n, 1]`, which is
/// exactly what `StrideBuf::contiguous_for` produces for a `[m, n]` shape
/// (reverse cumulative product of the dimensions).
fn is_contiguous_rank_two(operand: &OperandMeta<'_>) -> bool {
    operand.strides.len() == 2 && operand.strides[0] == operand.shape[1] && operand.strides[1] == 1
}

/// Whether a bias operand can be handed to the fused epilogue of issue #85.
///
/// cuBLASLt's bias attribute takes a device pointer to a vector whose
/// length equals the columns of the row-major output (the `m` dimension of
/// the column-major call - see the module doc), read as `f32`, contiguous,
/// at offset zero. Anything else is refused rather than reinterpreted.
fn bias_request_fits(bias: &OperandMeta<'_>, device_id: usize, columns: usize) -> bool {
    bias.dtype == DTypeId::F32.descriptor()
        && bias.device_id == device_id
        && bias.offset == 0
        && bias.shape.len() == 1
        && bias.shape[0] == columns
        && bias.strides.len() == 1
        && bias.strides[0] == 1
}

/// Maps a cuBLASLt status into the framework error type.
fn lt_error(error: CublasError) -> Error {
    Error::Msg(format!("cuBLASLt operation failed: {error:?}"))
}

/// Checked conversion of a host dimension into cuBLASLt's `u64` extents.
fn extent(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| Error::Msg(format!("cuBLASLt {field} out of u64 range: {value}")))
}

/// Checked conversion of a host dimension into cuBLASLt's `i64` leading
/// dimensions.
fn leading(value: usize, field: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::Msg(format!("cuBLASLt {field} out of i64 range: {value}")))
}

/// Returns the cached cuBLASLt state for a device, creating it on first use.
///
/// Handle creation is wrapped in `catch_unwind` for the same reason
/// `cuda_cache::try_get_cuda_device` wraps `CudaContext::new`: cudarc's
/// `CudaBlasLT::new` unwraps its workspace allocation, and a driver-level
/// allocation failure must surface as an error here, not as a panic on some
/// caller's matmul. A failure that means "this process will never have a
/// usable cuBLASLt" (creation error or panic) flips [`UNAVAILABLE`] so the
/// probe is paid exactly once; the caller then decides whether to fall back
/// (plain requests) or report the refusal (epilogue requests).
fn cublaslt_state(
    device_id: usize,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<Arc<CublasLtState>> {
    if UNAVAILABLE.load(Ordering::Acquire) {
        return Err(Error::Msg(
            "cuBLASLt is unavailable in this process: handle creation failed earlier".into(),
        ));
    }
    let states = STATES.get_or_init(|| Mutex::new(BTreeMap::new()));
    {
        let map = states.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(existing) = map.get(&device_id) {
            return Ok(existing.clone());
        }
    }
    // Creation runs outside the lock - it is the slow path - and the insert
    // below re-checks under the lock so a lost race keeps one handle, not
    // two, and never leaks the loser (dropping a `CudaBlasLT` destroys its
    // handle).
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| Error::Msg(format!("CUDA context bind failed: {error:?}")))?;
    let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CudaBlasLT::new(stream.clone())
    }));
    let blas = match created {
        Ok(Ok(blas)) => blas,
        Ok(Err(error)) => {
            UNAVAILABLE.store(true, Ordering::Release);
            return Err(Error::Msg(format!(
                "cuBLASLt handle creation failed: {error:?}"
            )));
        }
        Err(_) => {
            UNAVAILABLE.store(true, Ordering::Release);
            return Err(Error::Msg(
                "cuBLASLt handle creation panicked (workspace allocation or driver failure)".into(),
            ));
        }
    };
    // Mirrors cudarc's own workspace sizing (Hopper-class devices get
    // 32 MiB, everything else 4 MiB) because this module's workspace is
    // the one `cublasLtMatmul` actually receives.
    let (major, _minor) = stream
        .context()
        .compute_capability()
        .map_err(|error| Error::Msg(format!("CUDA compute capability query failed: {error:?}")))?;
    let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };
    let workspace = stream.alloc_zeros::<u8>(workspace_size).map_err(|error| {
        Error::Msg(format!(
            "cuBLASLt workspace allocation of {workspace_size} bytes failed: {error:?}"
        ))
    })?;
    let state = Arc::new(CublasLtState {
        blas,
        workspace,
        workspace_size,
    });
    let mut map = states.lock().unwrap_or_else(PoisonError::into_inner);
    Ok(map.entry(device_id).or_insert(state).clone())
}

/// RAII wrapper around a `cublasLtMatrixLayout_t` created in this module.
struct MatrixLayout(sys::cublasLtMatrixLayout_t);

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // SAFETY: the handle was produced by `result::create_matrix_layout`
        // in this same call, ownership never moves out of this guard, and
        // `Drop` runs exactly once, so the destroy call sees a live handle
        // it has not seen before.
        let _ = unsafe { result::destroy_matrix_layout(self.0) };
    }
}

/// RAII wrapper around a `cublasLtMatmulDesc_t` created in this module.
struct MatmulDesc(sys::cublasLtMatmulDesc_t);

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        // SAFETY: the handle was produced by `result::create_matmul_desc`
        // in this same call, ownership never moves out of this guard, and
        // `Drop` runs exactly once, so the destroy call sees a live handle
        // it has not seen before.
        let _ = unsafe { result::destroy_matmul_desc(self.0) };
    }
}

/// RAII wrapper around a `cublasLtMatmulPreference_t` created in this
/// module.
struct MatmulPref(sys::cublasLtMatmulPreference_t);

impl Drop for MatmulPref {
    fn drop(&mut self) {
        // SAFETY: the handle was produced by `result::create_matmul_pref`
        // in this same call, ownership never moves out of this guard, and
        // `Drop` runs exactly once, so the destroy call sees a live handle
        // it has not seen before.
        let _ = unsafe { result::destroy_matmul_pref(self.0) };
    }
}

/// Runs one `f32` GEMM through cuBLASLt, optionally with a fused bias and
/// activation epilogue (issue #85).
///
/// `lhs` is `[M, K]`, `rhs` is `[K, N]`, the product is `[M, N]`. The
/// caller owns the shape contract - inner dimensions already match - and
/// this function additionally enforces the dispatch policy of
/// [`gemm_request_fits`] plus, when present, [`bias_request_fits`] on the
/// bias.
///
/// # Return contract
///
/// - `Ok(Some(product))` - cuBLASLt computed the product, with the bias
///   and activation folded in when they were requested.
/// - `Ok(None)` - the request does not fit the policy, and *only* when no
///   epilogue was requested; the caller may fall back to
///   `kernels/matmul.cu`, which computes the same values.
/// - `Err(..)` - an epilogue was requested but refused (out-of-policy
///   operands or an invalid bias), or cuBLASLt itself failed. Callers with
///   an epilogue must propagate the error; callers without one may fall
///   back, because the NVRTC kernel's product is identical.
///
/// The epilogue half of the contract is what makes the path fail closed:
/// there is no configuration in which a bias or activation request is
/// answered by a kernel that omits it.
pub(crate) fn try_launch_matmul(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    bias: Option<&CudaStorage>,
    activation: Option<Activation>,
) -> Result<Option<CudaStorage>> {
    let lhs_meta = OperandMeta::of(lhs);
    let rhs_meta = OperandMeta::of(rhs);
    let epilogue_requested = bias.is_some() || activation.is_some();
    if !gemm_request_fits(&lhs_meta, &rhs_meta) {
        return if epilogue_requested {
            Err(Error::Msg(format!(
                "cuBLASLt epilogue request does not fit the #85 dispatch policy \
                 (f32, same device, rank 2, contiguous, offset 0, nonzero extents); \
                 got lhs shape {:?} strides {:?} offset {}, rhs shape {:?} strides {:?} offset {}",
                lhs_meta.shape,
                lhs_meta.strides,
                lhs_meta.offset,
                rhs_meta.shape,
                rhs_meta.strides,
                rhs_meta.offset
            )))
        } else {
            Ok(None)
        };
    }
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let n = rhs.shape[1];
    if let Some(bias_storage) = bias {
        let bias_meta = OperandMeta::of(bias_storage);
        if !bias_request_fits(&bias_meta, lhs_meta.device_id, n) {
            return Err(Error::Msg(format!(
                "cuBLASLt matmul bias must be a contiguous offset-0 f32 vector of \
                 length {n} (the output columns) on the operand device; got dtype {:?}, \
                 shape {:?}, strides {:?}, offset {}, device {}",
                bias_meta.dtype,
                bias_meta.shape,
                bias_meta.strides,
                bias_meta.offset,
                bias_meta.device_id
            )));
        }
    }

    let state = cublaslt_state(lhs_meta.device_id, &lhs.buffer.device.default_stream())?;
    let stream = lhs.buffer.device.default_stream();

    let m_u64 = extent(m, "row count")?;
    let k_u64 = extent(k, "inner dimension")?;
    let n_u64 = extent(n, "column count")?;
    let k_i64 = leading(k, "inner dimension")?;
    let n_i64 = leading(n, "column count")?;

    let out_shape = alloc::vec![m, n];
    let total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;
    let mut out_b = CudaBuffer {
        len: total,
        dtype: lhs.buffer.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            lhs.buffer.dtype,
            total,
            OperationKind::MatMul,
        )?),
        device: lhs.buffer.device.clone(),
        device_id: lhs_meta.device_id,
    };

    // Row-major derivation (module doc): the col-major call computes
    // C̃[N,M] = rhs[N,K] · lhs[K,M] with C̃ the same buffer as the
    // row-major product, leading dimension N on both rhs and the output,
    // leading dimension K on lhs, and no transposes.
    let a_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, n_u64, k_u64, n_i64)
            .map_err(lt_error)?,
    );
    let b_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, k_u64, m_u64, k_i64)
            .map_err(lt_error)?,
    );
    let c_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, n_u64, m_u64, n_i64)
            .map_err(lt_error)?,
    );
    let desc = MatmulDesc(
        result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )
        .map_err(lt_error)?,
    );
    let pref = MatmulPref(result::create_matmul_pref().map_err(lt_error)?);

    let (lhs_ptr, _lhs_sync) = lhs.buffer.data.device_ptr(&stream);
    let (rhs_ptr, _rhs_sync) = rhs.buffer.data.device_ptr(&stream);
    let (out_ptr, _out_sync) = {
        // out_b.data was allocated immediately above and never cloned, so
        // it stays uniquely owned (refcount 1) here - Arc::get_mut succeeds
        // without cloning first.
        let out_u8: &mut CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        out_u8.device_ptr_mut(&stream)
    };
    let bias_ptr = bias.map(|bias_storage| bias_storage.buffer.data.device_ptr(&stream));
    let (workspace_ptr, _workspace_sync) = state.workspace.device_ptr(&stream);

    let epilogue = match (&bias_ptr, &activation) {
        (Some(_), Some(Activation::Relu)) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
        (Some(_), Some(Activation::Gelu)) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        (Some(_), None) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (None, Some(Activation::Relu)) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        (None, Some(Activation::Gelu)) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        (None, None) => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
    };
    let transpose_off = 0i32;
    // SAFETY: every handle passed here (`desc`, the layouts, `pref`) was
    // created a few lines above by the matching `result::create_*` call and
    // is kept alive by its RAII guard until the end of this function; the
    // attribute values are plain host-side integers whose addresses and
    // sizes match what `set_matmul_desc_attribute` / the preference setter
    // read for these attributes; `bias_ptr`'s device address and
    // `SyncOnDrop` record are held in `bias_ptr` for the whole call, so the
    // bias buffer stays allocated and synchronized for the kernel that the
    // attribute now points at.
    unsafe {
        for (attr, value) in [
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                transpose_off,
            ),
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                transpose_off,
            ),
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC,
                transpose_off,
            ),
        ] {
            result::set_matmul_desc_attribute(
                desc.0,
                attr,
                (&value as *const i32).cast::<c_void>(),
                mem::size_of::<i32>(),
            )
            .map_err(lt_error)?;
        }
        result::set_matmul_desc_attribute(
            desc.0,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&epilogue as *const sys::cublasLtEpilogue_t).cast::<c_void>(),
            mem::size_of::<sys::cublasLtEpilogue_t>(),
        )
        .map_err(lt_error)?;
        if let Some((bias_address, _bias_sync)) = bias_ptr.as_ref() {
            result::set_matmul_desc_attribute(
                desc.0,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                (bias_address as *const cudarc::driver::sys::CUdeviceptr).cast::<c_void>(),
                mem::size_of::<cudarc::driver::sys::CUdeviceptr>(),
            )
            .map_err(lt_error)?;
        }
        result::set_matmul_pref_attribute(
            pref.0,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&state.workspace_size as *const usize).cast::<c_void>(),
            mem::size_of::<usize>(),
        )
        .map_err(lt_error)?;
    }

    // SAFETY: `desc`, all three layouts, and `pref` were created above and
    // are still owned by their guards, so none of the seven handles has
    // been freed; the layouts describe live allocations of exactly the
    // shapes passed in (`rhs`, `lhs`, and the output buffer each hold at
    // least `rows * cols` f32 elements by the caller's shape contract and
    // the checked output allocation), which is what "valid layouts for
    // allocations" requires of the heuristic query.
    let heuristic = unsafe {
        result::get_matmul_algo_heuristic(
            *state.blas.handle(),
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            c_layout.0,
            pref.0,
        )
    }
    .map_err(lt_error)?;

    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: all descriptor handles are the same live guards as above.
    // The operand, output, and workspace addresses come from `device_ptr` /
    // `device_ptr_mut` calls made on this stream a few lines earlier, so
    // the driver has synchronized their prior writes and the `SyncOnDrop`
    // records (`_lhs_sync`, `_rhs_sync`, `_out_sync`, `bias_ptr`,
    // `_workspace_sync`) are still in scope, keeping that synchronization
    // in force for the duration of this enqueue. `alpha` and `beta` are
    // host-side `f32` values whose addresses cuBLASLt reads synchronously.
    // Dimensions and leading dimensions match the layouts handed to the
    // heuristic, and `out_ptr` aliases the same buffer for `c` (read,
    // `beta = 0`) and `d` (written) exactly as `cublasLtMatmul` requires.
    unsafe {
        result::matmul(
            *state.blas.handle(),
            desc.0,
            (&alpha as *const f32).cast::<c_void>(),
            (&beta as *const f32).cast::<c_void>(),
            rhs_ptr as *const c_void,
            a_layout.0,
            lhs_ptr as *const c_void,
            b_layout.0,
            out_ptr as *const c_void,
            c_layout.0,
            out_ptr as *mut c_void,
            c_layout.0,
            &heuristic.algo as *const _,
            workspace_ptr as *mut c_void,
            state.workspace_size,
            stream.cu_stream() as *mut _,
        )
        .map_err(lt_error)?;
    }

    // The sync record returned by `device_ptr_mut` holds the exclusive
    // borrow of `out_b.data` for as long as it lives; dropping it here -
    // after the enqueue, the same point cudarc's own path reaches when its
    // records go out of scope at end of call - releases the borrow so the
    // buffer can be moved into the storage below.
    drop(_out_sync);

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0).map(Some)
}

/// Whether a plain *batched* product request fits the cuBLASLt path of
/// issue #85: `f32` on both operands of the same device, rank 3 each, an
/// equal nonzero batch, matching inner dimensions, no zero extent, fully
/// row-major contiguous at offset zero. Anything else - a batch broadcast
/// (stride-0 read), a rank deficit, a strided or offset view - returns
/// false so the orchestrator keeps the native batched plan or the composed
/// loop; refusing is never a regression, only a missed vendor opportunity.
/// Pure and host-side so the policy is unit-testable without a GPU.
pub(crate) fn batched_gemm_request_fits(lhs: &OperandMeta<'_>, rhs: &OperandMeta<'_>) -> bool {
    let f32 = DTypeId::F32.descriptor();
    if lhs.dtype != f32 || rhs.dtype != f32 {
        return false;
    }
    if lhs.device_id != rhs.device_id {
        return false;
    }
    if lhs.offset != 0 || rhs.offset != 0 {
        return false;
    }
    if lhs.shape.len() != 3 || rhs.shape.len() != 3 {
        return false;
    }
    let batch = lhs.shape[0];
    if batch == 0 || rhs.shape[0] != batch {
        return false;
    }
    let m = lhs.shape[1];
    let k = lhs.shape[2];
    let n = rhs.shape[2];
    if m == 0 || k == 0 || n == 0 || k != rhs.shape[1] {
        return false;
    }
    is_contiguous_rank_three(lhs) && is_contiguous_rank_three(rhs)
}

/// Row-major contiguity for a rank-3 operand: strides `[m*k, k, 1]`,
/// exactly what `StrideBuf::contiguous_for` produces for the shape.
fn is_contiguous_rank_three(operand: &OperandMeta<'_>) -> bool {
    operand.strides.len() == 3
        && operand.strides[2] == 1
        && operand.strides[1] == operand.shape[2]
        && operand.shape[1].checked_mul(operand.shape[2]) == Some(operand.strides[0])
}

/// Checked element count of one batch slice, for the strided-batch offsets.
fn slice_elements(a: usize, b: usize, field: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Msg(format!("cuBLASLt {field} slice overflows usize: {a} * {b}")))
}

/// Runs one strided-batched `f32` GEMM through cuBLASLt (issue #85).
///
/// `lhs` is `[B, M, K]`, `rhs` is `[B, K, N]`, the product is `[B, M, N]`.
/// The caller owns the shape contract - this function additionally
/// enforces [`batched_gemm_request_fits`]. The row-major derivation is the
/// same `Cᵀ = Bᵀ · Aᵀ` identity the plain launcher documents; the batch
/// rides the layouts as a count plus an element stride per operand
/// (`k*n` on the rhs-turned-A, `m*k` on the lhs-turned-B, `m*n` on the
/// output), so cuBLASLt walks the batch itself - one host-side call, no
/// launch loop.
///
/// # Return contract
///
/// - `Ok(Some(product))` - cuBLASLt computed every slice of the batch.
/// - `Ok(None)` - the request does not fit [`batched_gemm_request_fits`];
///   the caller keeps its native batched plan or composed loop, which
///   compute the same values.
/// - `Err(..)` - the request fit but cuBLASLt failed (handle, heuristic,
///   launch) or a host-side extent did not convert. Plain callers may fall
///   back: the batched kernels compute the identical product. The
///   orchestrator in `matmul.rs` treats `Err` exactly like `Ok(None)` for
///   that reason; a direct caller (the hardware tests) sees the real error
///   instead of a silent miss.
pub(crate) fn try_launch_batched_matmul(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<Option<CudaStorage>> {
    let lhs_meta = OperandMeta::of(lhs);
    let rhs_meta = OperandMeta::of(rhs);
    if !batched_gemm_request_fits(&lhs_meta, &rhs_meta) {
        return Ok(None);
    }
    let (batch, m, k, n) = (lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[2]);

    let state = cublaslt_state(lhs_meta.device_id, &lhs.buffer.device.default_stream())?;
    let stream = lhs.buffer.device.default_stream();

    let m_u64 = extent(m, "row count")?;
    let k_u64 = extent(k, "inner dimension")?;
    let n_u64 = extent(n, "column count")?;
    let k_i64 = leading(k, "inner dimension")?;
    let n_i64 = leading(n, "column count")?;
    let batch_i32 = i32::try_from(batch)
        .map_err(|_| Error::Msg(format!("cuBLASLt batch count out of i32 range: {batch}")))?;
    // Element strides between consecutive batch slices, in the col-major
    // call's operand order (A = rhs, B = lhs, C = output - see the module
    // doc's row-major derivation).
    let a_stride = leading(slice_elements(k, n, "rhs")?, "rhs batch stride")?;
    let b_stride = leading(slice_elements(m, k, "lhs")?, "lhs batch stride")?;
    let c_stride = leading(slice_elements(m, n, "output")?, "output batch stride")?;

    let out_shape = alloc::vec![batch, m, n];
    let total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;
    let mut out_b = CudaBuffer {
        len: total,
        dtype: lhs.buffer.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            lhs.buffer.dtype,
            total,
            OperationKind::MatMul,
        )?),
        device: lhs.buffer.device.clone(),
        device_id: lhs_meta.device_id,
    };

    // Same row-major derivation as the plain launcher, three times over:
    // A[batch, N, K] = rhs, B[batch, K, M] = lhs, C[batch, N, M] = output.
    let a_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, n_u64, k_u64, n_i64)
            .map_err(lt_error)?,
    );
    let b_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, k_u64, m_u64, k_i64)
            .map_err(lt_error)?,
    );
    let c_layout = MatrixLayout(
        result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, n_u64, m_u64, n_i64)
            .map_err(lt_error)?,
    );
    // The batch itself: count plus element stride on each layout, the
    // strided-batched configuration cuBLASLt walks without the host
    // launching per slice.
    for (layout, stride) in [
        (a_layout.0, a_stride),
        (b_layout.0, b_stride),
        (c_layout.0, c_stride),
    ] {
        // SAFETY: the handle was created a few lines above by
        // `result::create_matrix_layout` and is kept alive by its RAII
        // guard; `batch_i32` and `stride` are plain host-side integers
        // whose addresses and sizes (`i32`, `i64`) match what
        // `cublasLtMatrixLayoutSetAttribute` reads for `BATCH_COUNT` and
        // `STRIDED_BATCH_OFFSET`.
        unsafe {
            result::set_matrix_layout_attribute(
                layout,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&batch_i32 as *const i32).cast::<c_void>(),
                mem::size_of::<i32>(),
            )
            .map_err(lt_error)?;
            result::set_matrix_layout_attribute(
                layout,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride as *const i64).cast::<c_void>(),
                mem::size_of::<i64>(),
            )
            .map_err(lt_error)?;
        }
    }
    let desc = MatmulDesc(
        result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )
        .map_err(lt_error)?,
    );
    let pref = MatmulPref(result::create_matmul_pref().map_err(lt_error)?);

    let (lhs_ptr, _lhs_sync) = lhs.buffer.data.device_ptr(&stream);
    let (rhs_ptr, _rhs_sync) = rhs.buffer.data.device_ptr(&stream);
    let (out_ptr, _out_sync) = {
        // out_b.data was allocated immediately above and never cloned, so
        // it stays uniquely owned (refcount 1) here - Arc::get_mut succeeds
        // without cloning first.
        let out_u8: &mut CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        out_u8.device_ptr_mut(&stream)
    };
    let (workspace_ptr, _workspace_sync) = state.workspace.device_ptr(&stream);

    let transpose_off = 0i32;
    // SAFETY: same handle-liveness and host-value argument as the plain
    // launcher's attribute block: every descriptor was created above and is
    // owned by its guard, the attribute values are plain integers read
    // synchronously, and no epilogue pointer is involved on this path.
    unsafe {
        for (attr, value) in [
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                transpose_off,
            ),
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                transpose_off,
            ),
            (
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC,
                transpose_off,
            ),
        ] {
            result::set_matmul_desc_attribute(
                desc.0,
                attr,
                (&value as *const i32).cast::<c_void>(),
                mem::size_of::<i32>(),
            )
            .map_err(lt_error)?;
        }
        result::set_matmul_desc_attribute(
            desc.0,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            (&sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT as *const sys::cublasLtEpilogue_t)
                .cast::<c_void>(),
            mem::size_of::<sys::cublasLtEpilogue_t>(),
        )
        .map_err(lt_error)?;
        result::set_matmul_pref_attribute(
            pref.0,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&state.workspace_size as *const usize).cast::<c_void>(),
            mem::size_of::<usize>(),
        )
        .map_err(lt_error)?;
    }

    // SAFETY: `desc`, all three (now batched) layouts, and `pref` were
    // created above and are still owned by their guards; the layouts
    // describe live allocations of exactly the shapes passed in (the
    // fits policy proved rank-3 contiguity and the output allocation
    // covers `batch * m * n` elements).
    let heuristic = unsafe {
        result::get_matmul_algo_heuristic(
            *state.blas.handle(),
            desc.0,
            a_layout.0,
            b_layout.0,
            c_layout.0,
            c_layout.0,
            pref.0,
        )
    }
    .map_err(lt_error)?;

    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: all descriptor handles are the same live guards as above;
    // the operand, output, and workspace addresses come from `device_ptr` /
    // `device_ptr_mut` calls made on this stream a few lines earlier, so
    // the driver has synchronized their prior writes and the `SyncOnDrop`
    // records are still in scope; `alpha`/`beta` are host-side values read
    // synchronously; the batch count and strides travel inside the layouts,
    // so this call itself is the same shape as the plain one.
    unsafe {
        result::matmul(
            *state.blas.handle(),
            desc.0,
            (&alpha as *const f32).cast::<c_void>(),
            (&beta as *const f32).cast::<c_void>(),
            rhs_ptr as *const c_void,
            a_layout.0,
            lhs_ptr as *const c_void,
            b_layout.0,
            out_ptr as *const c_void,
            c_layout.0,
            out_ptr as *mut c_void,
            c_layout.0,
            &heuristic.algo as *const _,
            workspace_ptr as *mut c_void,
            state.workspace_size,
            stream.cu_stream() as *mut _,
        )
        .map_err(lt_error)?;
    }

    // Release the exclusive borrow of `out_b.data` (same point cudarc's
    // own path reaches when its records go out of scope) so the buffer can
    // move into the storage below.
    drop(_out_sync);

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::tensor::dtype::DTypeDescriptor;

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

    #[test]
    fn gemm_fits_admits_the_canonical_f32_rank_two_pair() {
        let lhs = meta(f32(), &[2, 3], &[3, 1], 0);
        let rhs = meta(f32(), &[3, 2], &[2, 1], 0);
        assert!(gemm_request_fits(&lhs, &rhs));
    }

    #[test]
    fn gemm_fits_refuses_every_non_f32_dtype() {
        for dtype in [
            DTypeId::F16.descriptor(),
            DTypeId::BF16.descriptor(),
            DTypeId::F64.descriptor(),
            DTypeId::I64.descriptor(),
        ] {
            let lhs = meta(dtype, &[2, 3], &[3, 1], 0);
            let rhs = meta(f32(), &[3, 2], &[2, 1], 0);
            assert!(!gemm_request_fits(&lhs, &rhs), "lhs {dtype:?} refused");
            let lhs = meta(f32(), &[2, 3], &[3, 1], 0);
            let rhs = meta(dtype, &[3, 2], &[2, 1], 0);
            assert!(!gemm_request_fits(&lhs, &rhs), "rhs {dtype:?} refused");
        }
    }

    #[test]
    fn gemm_fits_refuses_mismatched_devices() {
        let lhs = meta(f32(), &[2, 3], &[3, 1], 0);
        let mut rhs = meta(f32(), &[3, 2], &[2, 1], 0);
        rhs.device_id = 1;
        assert!(!gemm_request_fits(&lhs, &rhs));
    }

    #[test]
    fn gemm_fits_refuses_rank_above_two_mismatched_inner_dims_and_zero_extents() {
        let lhs = meta(f32(), &[2, 2, 3], &[6, 3, 1], 0);
        let rhs = meta(f32(), &[2, 3, 2], &[6, 2, 1], 0);
        assert!(!gemm_request_fits(&lhs, &rhs), "rank 3 refused");

        let lhs = meta(f32(), &[2, 3], &[3, 1], 0);
        let rhs = meta(f32(), &[4, 2], &[2, 1], 0);
        assert!(!gemm_request_fits(&lhs, &rhs), "k mismatch refused");

        let lhs = meta(f32(), &[0, 3], &[3, 1], 0);
        let rhs = meta(f32(), &[3, 2], &[2, 1], 0);
        assert!(!gemm_request_fits(&lhs, &rhs), "zero extent refused");
    }

    #[test]
    fn gemm_fits_refuses_strided_or_offset_views() {
        let lhs = meta(f32(), &[2, 3], &[6, 1], 0);
        let rhs = meta(f32(), &[3, 2], &[2, 1], 0);
        assert!(!gemm_request_fits(&lhs, &rhs), "padded lhs refused");

        let lhs = meta(f32(), &[2, 3], &[3, 1], 1);
        let rhs = meta(f32(), &[3, 2], &[2, 1], 0);
        assert!(!gemm_request_fits(&lhs, &rhs), "offset lhs refused");

        let lhs = meta(f32(), &[2, 3], &[3, 1], 0);
        let rhs = meta(f32(), &[3, 2], &[4, 1], 1);
        assert!(!gemm_request_fits(&lhs, &rhs), "offset rhs refused");
    }

    #[test]
    fn bias_fits_only_a_contiguous_f32_vector_of_column_length() {
        assert!(bias_request_fits(&meta(f32(), &[2], &[1], 0), 0, 2));
        assert!(
            !bias_request_fits(&meta(f32(), &[3], &[1], 0), 0, 2),
            "wrong length refused"
        );
        assert!(
            !bias_request_fits(&meta(f32(), &[2, 1], &[1, 1], 0), 0, 2),
            "rank 2 refused"
        );
        assert!(
            !bias_request_fits(&meta(DTypeId::F64.descriptor(), &[2], &[1], 0), 0, 2),
            "f64 refused"
        );
        assert!(
            !bias_request_fits(&meta(f32(), &[2], &[1], 4), 0, 2),
            "offset refused"
        );
        let mut wrong_device = meta(f32(), &[2], &[1], 0);
        wrong_device.device_id = 1;
        assert!(!bias_request_fits(&wrong_device, 0, 2));
    }
}
