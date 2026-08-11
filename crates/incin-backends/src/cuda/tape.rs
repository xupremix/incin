use core::cell::RefCell;

use incin_core::exec::tape;
use incin_core::exec::{GradientMap, Tape, TapeNode, TapeStorage};
use incin_core::prelude::{Error, OperationKind, Result, ShapeBuf};

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
            .map_err(|error| incin_core::prelude::BackendError::Execution {
                operation: incin_core::prelude::OperationKind::Storage,
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
    #[must_use]
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

    Ok(result)
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
