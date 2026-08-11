use alloc::vec::Vec;
use core::cell::RefCell;

use incin_core::exec::tape;
use incin_core::exec::{GradientMap, Tape, TapeNode, TapeStorage};
use incin_core::prelude::{DTypeId, OperationKind, Result};

use crate::wgpu::dispatch;
use crate::wgpu::storage::{TensorId, WgpuBuffer, WgpuStorage};

/// One recorded operation, as the core defines it.
pub(crate) type TapeEntry = TapeNode<WgpuStorage>;

thread_local! {
    static TAPE: RefCell<Tape<WgpuStorage>> = const { RefCell::new(Tape::new()) };
}

/// The three things a reverse walk needs of WGPU storage.
impl TapeStorage for WgpuStorage {
    fn id(&self) -> TensorId {
        self.id
    }

    fn ones_like(&self) -> Result<Self> {
        let n = crate::wgpu::backend::num_elements(&self.shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, self.shape.to_vec()))
    }

    fn accumulate(&self, contribution: &Self) -> Result<Self> {
        add_wgpu_storage(self, contribution)
    }

    fn has_non_finite(&self) -> Result<bool> {
        let data: Vec<f32> = self.buffer.to_vec()?;
        Ok(data.iter().any(|x| x.is_nan() || x.is_infinite()))
    }
}

/// Number of entries currently on the tape.
#[must_use]
pub fn depth() -> usize {
    TAPE.with(|t| t.borrow().depth())
}

#[cfg(feature = "telemetry")]
thread_local! {
    static BACKWARD_STEP: RefCell<usize> = const { RefCell::new(0) };
}

/// Push a `TapeEntry` unless `GradMode` forbids recording.
pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
    #[cfg(feature = "telemetry")]
    {
        let depth = depth() as f64;
        let step = BACKWARD_STEP.with(|s| *s.borrow());
        crate::telemetry::emit_scalar(step, "tape/depth", depth);
    }
}

pub struct WgpuGrads {
    pub(crate) grads: GradientMap<WgpuStorage>,
}

impl WgpuGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    #[must_use]
    pub fn get(&self, id: TensorId) -> Option<&WgpuStorage> {
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

pub fn backward(loss: &WgpuStorage) -> Result<WgpuGrads> {
    #[cfg(feature = "telemetry")]
    let n_ops = depth();

    let nodes = TAPE.with(|t| t.borrow_mut().drain_reachable(loss.id()));
    let grads = incin_core::exec::GradMode::Disabled.scope(|| tape::backward(nodes, loss))?;

    #[cfg(feature = "telemetry")]
    {
        let step = BACKWARD_STEP.with(|s| {
            let cur = *s.borrow();
            *s.borrow_mut() += 1;
            cur
        });
        emit_backward_telemetry(step, n_ops);
    }

    Ok(WgpuGrads { grads })
}

#[cfg(feature = "telemetry")]
fn emit_backward_telemetry(step: usize, n_ops: usize) {
    crate::telemetry::emit_scalar(step, "tape/ops", n_ops as f64);
    #[cfg(feature = "std")]
    {
        if let Some(g) = incin_core::backend_authoring::tracing_graph_snapshot() {
            crate::telemetry::emit_graph_snapshot(g);
        }
    }
}

fn add_wgpu_storage(a: &WgpuStorage, b: &WgpuStorage) -> Result<WgpuStorage> {
    debug_assert_eq!(
        a.shape, b.shape,
        "tape accumulation requires matching shapes"
    );
    let n = crate::wgpu::backend::num_elements(&a.shape)?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n, OperationKind::Storage)?;
    let params = [
        0,
        u32::try_from(n).map_err(|_| {
            incin_core::prelude::Error::Msg("WGPU launch element count exceeds u32".into())
        })?,
    ]; // op_mode 0=add
    dispatch::dispatch_binary(&a.buffer, &b.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, a.shape.to_vec()))
}

pub fn unbroadcast(grad: &WgpuStorage, target_shape: &[usize]) -> Result<WgpuStorage> {
    if grad.shape == target_shape {
        return Ok(grad.clone());
    }

    let ndim_diff = grad.shape.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    // Reduce leading dims
    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0)?;
    }

    // Reduce keepdim dims
    for (i, &t_dim) in target_shape.iter().enumerate() {
        if t_dim == 1 && result.shape[i] != 1 {
            result = sum_dim_keepdim(&result, i)?;
        }
    }

    Ok(result)
}

fn sum_dim_squeeze(storage: &WgpuStorage, axis: usize) -> Result<WgpuStorage> {
    let reduced = sum_dim_keepdim(storage, axis)?;
    let mut new_shape = reduced.shape.to_vec();
    new_shape.remove(axis);
    Ok(WgpuStorage::new(reduced.buffer.clone(), new_shape))
}

fn sum_dim_keepdim(storage: &WgpuStorage, axis: usize) -> Result<WgpuStorage> {
    let mut out_shape = storage.shape.to_vec();
    out_shape[axis] = 1;
    let total: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, total, OperationKind::Storage)?;

    let inner_stride: usize =
        incin_core::prelude::ShapeBuf::from_slice(&(storage.shape[axis + 1..]))
            .checked_numel(incin_core::prelude::OperationKind::Storage)?;

    let axis_len = u32::try_from(storage.shape[axis]).map_err(|_| {
        incin_core::prelude::Error::Msg("WGPU reduction axis length exceeds u32".into())
    })?;
    let inner_stride = u32::try_from(inner_stride)
        .map_err(|_| incin_core::prelude::Error::Msg("WGPU reduction stride exceeds u32".into()))?;
    let total = u32::try_from(total).map_err(|_| {
        incin_core::prelude::Error::Msg("WGPU reduction output length exceeds u32".into())
    })?;
    dispatch::dispatch_reduce_dim(
        &storage.buffer,
        &out_buf,
        0, // sum
        axis_len,
        inner_stride,
        total,
    );
    Ok(WgpuStorage::new(out_buf, out_shape))
}
