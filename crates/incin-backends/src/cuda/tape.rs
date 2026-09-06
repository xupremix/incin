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
        let numel = ShapeBuf::from_slice(&self.shape).checked_numel(OperationKind::Storage)?;
        let device_id = self.buffer.device_id;
        let stream = self.buffer.device.default_stream();
        let data_ones = vec![1.0f32; numel];
        let data_u8: &[u8] = bytemuck::cast_slice(&data_ones);
        let u8_slice = stream
            .clone_htod(data_u8)
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
        let bytes = self
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*self.buffer.data)
            .map_err(|error| incin_core::error::BackendError::Execution {
                operation: incin_core::shapes::error::OperationKind::Storage,
                message: alloc::format!("CUDA gradient readback failed: {error:?}").into(),
            })?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .any(|x| x.is_nan() || x.is_infinite()))
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

pub(crate) fn unbroadcast(grad: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
    if grad.shape == target_shape {
        return Ok(grad.clone());
    }

    let ndim_diff = grad.shape.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    // Reduce leading dims
    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0);
    }

    // Reduce keepdim dims
    for (i, &t_dim) in target_shape.iter().enumerate() {
        if t_dim == 1 && result.shape[i] != 1 {
            result = sum_dim_keepdim(&result, i);
        }
    }

    if result.shape == target_shape {
        return Ok(result);
    }

    // Expand what reduction left smaller: a reduced-all-the-way scalar seed
    // reaches here with fewer elements than its target, and the kernels
    // downstream do not broadcast scalars implicitly the way the CPU ones
    // do, so handing the scalar on produces a binary launch the iteration
    // plan refuses. `broadcast_shape` checks compatibility first, because
    // `launch_broadcast` assumes a legal target and would otherwise read out
    // of bounds on a genuinely incompatible shape.
    crate::layout::broadcast_shape(&result.shape, target_shape)?;
    crate::cuda::ops::shape::launch_broadcast(&result, target_shape)
}

fn sum_dim_squeeze(storage: &CudaStorage, axis: usize) -> CudaStorage {
    let reduced = sum_dim_keepdim(storage, axis);
    let mut new_shape = reduced.shape.to_vec();
    new_shape.remove(axis);
    CudaStorage::new(reduced.buffer.clone(), new_shape)
}

fn sum_dim_keepdim(storage: &CudaStorage, axis: usize) -> CudaStorage {
    crate::cuda::ops::reduce::launch_reduce_op("sum", storage, axis, true).unwrap()
}
