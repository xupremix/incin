use core::cell::RefCell;

use incin_core::error::{Error, Result};
use incin_core::exec::tape;
use incin_core::exec::{GradientMap, Tape, TapeNode, TapeStorage};
use incin_core::shapes::{OperationKind, ShapeBuf};

use crate::cuda::storage::{CudaBuffer, CudaStorage, TensorId};

/// One recorded operation, as the core defines it.
pub(crate) type TapeEntry = TapeNode<CudaStorage>;

thread_local! {
    static TAPE: RefCell<Tape<CudaStorage>> = const { RefCell::new(Tape::new()) };
}

/// The three things a reverse walk needs of CUDA storage.
impl TapeStorage for CudaStorage {
    fn id(&self) -> TensorId {
        self.id
    }

    fn ones_like(&self) -> Result<Self> {
        use incin_core::tensor::dtype::DTypeId;
        let numel = ShapeBuf::from_slice(&self.shape).checked_numel(OperationKind::Storage)?;
        let device_id = self.buffer.device_id;
        let stream = self.buffer.device.default_stream();
        // The seed must match the loss storage's own dtype: a hardcoded f32
        // ones vector under an f64 descriptor allocates numel*4 bytes for a
        // numel*8 claim, which `CudaStorage::new` refuses (and f16/bf16
        // would silently mis-shape the same way).
        let data_u8: Vec<u8> = match self.buffer.dtype.builtin_id() {
            Some(DTypeId::F64) => bytemuck::cast_slice(&vec![1.0f64; numel]).to_vec(),
            Some(DTypeId::F16) => {
                bytemuck::cast_slice(&vec![half::f16::from_f32(1.0); numel]).to_vec()
            }
            Some(DTypeId::BF16) => {
                bytemuck::cast_slice(&vec![half::bf16::from_f32(1.0); numel]).to_vec()
            }
            _ => bytemuck::cast_slice(&vec![1.0f32; numel]).to_vec(),
        };
        let u8_slice = stream
            .clone_htod(&data_u8)
            .map_err(|e| Error::Msg(alloc::format!("CUDA HTOD failed: {e:?}")))?;

        let buf = CudaBuffer {
            len: numel,
            dtype: self.buffer.dtype,
            data: alloc::sync::Arc::new(u8_slice),
            device: self.buffer.device.clone(),
            device_id,
        };
        Ok(CudaStorage::new(
            alloc::sync::Arc::new(buf),
            self.shape.to_vec(),
        ))
    }

    fn accumulate(&self, contribution: &Self) -> Result<Self> {
        add_cuda_storage(self, contribution)
    }

    fn has_non_finite(&self) -> Result<bool> {
        use incin_core::tensor::dtype::DTypeId;
        let bytes = self
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*self.buffer.data)
            .map_err(|error| incin_core::error::BackendError::Execution {
                operation: incin_core::shapes::error::OperationKind::Storage,
                message: alloc::format!("CUDA gradient readback failed: {error:?}").into(),
            })?;
        Ok(match self.buffer.dtype.builtin_id() {
            Some(DTypeId::F64) => bytemuck::cast_slice::<u8, f64>(&bytes)
                .iter()
                .any(|x| x.is_nan() || x.is_infinite()),
            Some(DTypeId::F16) => bytemuck::cast_slice::<u8, half::f16>(&bytes)
                .iter()
                .any(|x| x.is_nan() || x.is_infinite()),
            Some(DTypeId::BF16) => bytemuck::cast_slice::<u8, half::bf16>(&bytes)
                .iter()
                .any(|x| x.is_nan() || x.is_infinite()),
            _ => bytemuck::cast_slice::<u8, f32>(&bytes)
                .iter()
                .any(|x| x.is_nan() || x.is_infinite()),
        })
    }
}

/// Push a `TapeEntry` unless `GradMode` forbids recording.
pub(crate) fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
}

/// Record a custom operation's backward recipe on this thread's tape.
///
/// The downstream half of the custom-training contract, mirroring
/// `cpu::tape_record`: a foreign `Execute` implementation runs its forward
/// kernel, then calls this with a `TapeNode` whose recipe maps one output
/// gradient to one gradient per input. The node joins the same tape the
/// built-in kernels record on, under the same `GradMode` gate, so mixed
/// graphs walk as one graph. Recipes should stay in-kernel (broadcast,
/// scale, elementwise launches): every host value access is a readback.
/// Hardware-executed coverage arrives with the GPU execution runner (#82).
pub fn record(entry: TapeNode<CudaStorage>) {
    push(entry);
}

/// Record a custom operation's backward recipe, building it only if kept.
///
/// The lazy form of [`record`](self::record): the entry closure runs only
/// when the ambient `GradMode` records.
pub fn record_with(entry: impl FnOnce() -> TapeNode<CudaStorage>) {
    if !incin_core::exec::GradMode::current().records() {
        return;
    }
    push(entry());
}

impl<D, K> incin_core::backend_authoring::RecordingBackend<K> for super::CudaBackendImpl<D>
where
    D: incin_core::tensor::device::Device,
    K: incin_core::tensor::dtype::DType,
{
    fn record_custom(node: TapeNode<CudaStorage>) {
        push(node);
    }
}

/// Number of entries currently on the tape.
#[must_use]
pub fn depth() -> usize {
    TAPE.with(|t| t.borrow().depth())
}

/// Drain every entry off the tape, returning them in push order.
///
/// The narrow primitive behind op-level collapsing: a composed forward
/// (`conv1d` via `conv2d`) runs its pieces normally, then the caller
/// re-pushes the entries that predate its own (`split_off` at a
/// previously recorded [`depth`]) and folds the remainder into the one
/// entry it records. Entries keep their identities, so the backward
/// walk is unaffected; only the entry count changes.
pub(crate) fn drain_all() -> alloc::vec::Vec<TapeEntry> {
    TAPE.with(|t| t.borrow_mut().drain())
}

/// The CUDA backend's gradient container (`Backend::Grads`).
pub struct CudaGrads {
    pub(crate) grads: GradientMap<CudaStorage>,
}

impl CudaGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    /// Replace the gradient recorded for `id`.
    ///
    /// A replacement rather than an accumulation, which is why it is spelled
    /// differently from anything the reverse walk calls. See
    /// `AutogradBackend::set_grad`.
    pub fn set(&mut self, id: TensorId, value: CudaStorage) {
        self.grads.insert(id, value);
    }

    pub fn get(&self, id: TensorId) -> Option<&CudaStorage> {
        self.grads.get(id)
    }

    /// How many tensors the backward pass reached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether it reached none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

pub(crate) fn backward(loss: &CudaStorage) -> Result<CudaGrads> {
    let nodes = TAPE.with(|t| t.borrow_mut().drain_reachable(loss.id()));
    let grads = incin_core::exec::GradMode::Disabled.scope(|| tape::backward(nodes, loss))?;
    Ok(CudaGrads { grads })
}

/// Walk the reachable graph with an explicit output cotangent.
pub(crate) fn backward_with(loss: &CudaStorage, seed: &CudaStorage) -> Result<CudaGrads> {
    let nodes = TAPE.with(|t| t.borrow_mut().drain_reachable(loss.id()));
    let grads = incin_core::exec::GradMode::Disabled
        .scope(|| tape::backward_with_seed(nodes, loss, seed))?;
    Ok(CudaGrads { grads })
}

fn add_cuda_storage(a: &CudaStorage, b: &CudaStorage) -> Result<CudaStorage> {
    crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", a, b, &a.shape)
}

/// Un-broadcast `grad` to `target_shape` for backward.
///
/// Semantics (fail-closed, shared with CPU and WGPU): on success the
/// result's shape equals `target_shape`, so it can always be accumulated
/// into the target. A compatible grad of lower rank than the target - a
/// scalar seed, e.g. `[] -> [1]` - expands (a scalar broadcasts to any
/// shape); a grad that does not broadcast into the target refuses with a
/// named `ShapeMismatch`. Never panics on rank-deficient input.
pub(crate) fn unbroadcast(grad: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
    if grad.shape == target_shape {
        return Ok(grad.clone());
    }

    let ndim_diff = grad.shape.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    // Reduce leading dims
    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0)?;
    }

    // Reduce keepdim dims. Only when the ranks agree after the leading
    // squeeze (grad rank >= target rank): that squeeze right-aligns the
    // axes, so indexing `result.shape[i]` against `target_shape[i]` is
    // sound only at equal rank. When the target outranks the grad - the
    // scalar-seed case, e.g. `[] -> [1]` - there is no aligned axis to
    // reduce; the tail expands the smaller grad instead of indexing past
    // it (the latent cross-backend panic this guards).
    if result.shape.len() == target_shape.len() {
        for (i, &t_dim) in target_shape.iter().enumerate() {
            if t_dim == 1 && result.shape[i] != 1 {
                result = sum_dim_keepdim(&result, i)?;
            }
        }
    }

    if result.shape == target_shape {
        return Ok(result);
    }

    // Expand what reduction left smaller: a reduced-all-the-way scalar seed
    // reaches here with fewer elements than its target, and the kernels
    // downstream do not broadcast scalars implicitly the way the CPU ones
    // do, so handing the scalar on produces a binary launch the iteration
    // plan refuses. `broadcast_shape` must resolve *to* the target, not
    // merely be mutual: `launch_broadcast` assumes a legal right-aligned
    // target and would otherwise read out of bounds on a shape like
    // `[4] -> [2,1]`. Mirrors the WGPU tail (`wgpu/tape.rs`) and the CPU
    // tail (`cpu/tape.rs`).
    let resolved = crate::layout::broadcast_shape(&result.shape, target_shape)?;
    if resolved.as_slice() != target_shape {
        return Err(Error::ShapeMismatch {
            op: "autograd unbroadcast",
            expected: target_shape.to_vec(),
            got: result.shape.to_vec(),
            msg: "the reduced gradient does not broadcast into the target shape".into(),
        });
    }
    crate::cuda::ops::shape::launch_broadcast(&result, target_shape)
}

fn sum_dim_squeeze(storage: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    let reduced = sum_dim_keepdim(storage, axis)?;
    let mut new_shape = reduced.shape.to_vec();
    new_shape.remove(axis);
    Ok(CudaStorage::new(reduced.buffer.clone(), new_shape))
}

/// Rewrap a gradient that a squeeze-reducing forward (`sum_dim`,
/// `mean_dim`, `prod_dim`) produced back to the keepdim form of the
/// forward input's shape, so [`unbroadcast`] can expand it.
///
/// A squeeze reduction drops axis `dim` from the output shape (`[2, 3]`
/// over dim 1 arrives as `[2]`), so the cotangent is one rank short with
/// no marker of which axis went missing. The generic [`unbroadcast`]
/// right-aligns, which misreads `[2]` against `[2, 3]` (`2 != 3`) and
/// refuses a gradient that is perfectly defined. The axis is reinserted
/// here, positionally, as size 1 (`[2, 1]`), and the tail broadcast
/// expands it to the target. Metadata-only: same numel, same
/// offset-zero element order - the same assumption the Welford walk in
/// `backend/reduce.rs` already makes when it rewraps squeezed grads
/// inline, and the same reinsertion the CPU `sum_dim` backward performs
/// before broadcasting.
///
/// Fail-closed: when the shapes do not line up exactly (wrong numel,
/// axis outside the target rank), the gradient passes through untouched
/// and [`unbroadcast`] refuses as before, rather than a mis-shaped
/// rewrap reaching accumulation.
pub(crate) fn unsqueeze_squeezed_grad(
    grad_out: &CudaStorage,
    t_shape: &[usize],
    dim: usize,
) -> CudaStorage {
    let mut keepdim_shape = t_shape.to_vec();
    if dim >= keepdim_shape.len() {
        return grad_out.clone();
    }
    keepdim_shape[dim] = 1;
    let keepdim_numel: usize = keepdim_shape.iter().product();
    let grad_numel: usize = grad_out.shape.iter().product();
    if grad_out.shape != keepdim_shape && grad_numel == keepdim_numel {
        CudaStorage::new(grad_out.buffer.clone(), keepdim_shape)
    } else {
        grad_out.clone()
    }
}

/// Sum one axis, keeping it.
///
/// Fallible rather than unwrapping the launch. Every CUDA recipe with a
/// broadcast operand reaches this through `unbroadcast`, so a panic here is a
/// panic inside a backward pass - the exact shape `GRD-005` made `BackwardFn`
/// fallible to remove, and the one site that kept it.
fn sum_dim_keepdim(storage: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    crate::cuda::ops::reduce::launch_reduce_op("sum", storage, axis, true)
}
