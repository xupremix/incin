use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::cell::RefCell;

use incin_core::prelude::{DTypeId, OperationKind, Result};

use crate::wgpu::dispatch;
use crate::wgpu::storage::{TensorId, WgpuBuffer, WgpuStorage};

pub struct TapeEntry {
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub backward: Box<dyn Fn(&WgpuStorage) -> Vec<WgpuStorage> + Send + Sync>,
}

thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "telemetry")]
thread_local! {
    static BACKWARD_STEP: RefCell<usize> = const { RefCell::new(0) };
}

pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
    #[cfg(feature = "telemetry")]
    {
        let depth = TAPE.with(|t| t.borrow().len()) as f64;
        let step = BACKWARD_STEP.with(|s| *s.borrow());
        crate::telemetry::emit_scalar(step, "tape/depth", depth);
    }
}

pub struct WgpuGrads {
    // Private per B-3 (.agents/API_DESIGN.md "pub(crate) is default"): use
    // `.get(id)` — downstream crates shouldn't inspect/mutate the internal
    // BTreeMap beyond the intended query API.
    pub(crate) grads: BTreeMap<TensorId, WgpuStorage>,
}

impl WgpuGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    pub fn get(&self, id: TensorId) -> Option<&WgpuStorage> {
        self.grads.get(&id)
    }
}

pub fn backward(loss: &WgpuStorage) -> Result<WgpuGrads> {
    let mut grads: BTreeMap<TensorId, WgpuStorage> = BTreeMap::new();

    // Seed with ones
    let n = loss.shape.iter().product::<usize>();
    let data: Vec<f32> = vec![1.0; n];
    let buf = WgpuBuffer::from_slice(&data);
    grads.insert(loss.id, WgpuStorage::new(buf, loss.shape.to_vec()));

    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));
    #[cfg(feature = "telemetry")]
    let n_ops = entries.len();

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            match grads.entry(input_id) {
                Entry::Occupied(mut slot) => {
                    let accumulated = add_wgpu_storage(slot.get(), &g)?;
                    slot.insert(accumulated);
                }
                Entry::Vacant(slot) => {
                    slot.insert(g);
                }
            }
        }
    }
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
        if let Some(g) = incin_core::prelude::tracing_graph_snapshot() {
            crate::telemetry::emit_graph_snapshot(g);
        }
    }
}

fn check_nan(storage: &WgpuStorage, id: TensorId) {
    let data: Vec<f32> = storage.buffer.to_vec();
    if data.iter().any(|x| x.is_nan() || x.is_infinite()) {
        panic!("NaN or Infinity detected in gradient for TensorId {:?}", id);
    }
}

pub fn backward_with_nan_check(loss: &WgpuStorage) -> Result<WgpuGrads> {
    let mut grads: BTreeMap<TensorId, WgpuStorage> = BTreeMap::new();

    // Seed with ones
    let n = loss.shape.iter().product::<usize>();
    let data: Vec<f32> = vec![1.0; n];
    let buf = WgpuBuffer::from_slice(&data);
    grads.insert(loss.id, WgpuStorage::new(buf, loss.shape.to_vec()));

    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            check_nan(&g, input_id);
            match grads.entry(input_id) {
                Entry::Occupied(mut slot) => {
                    let accumulated = add_wgpu_storage(slot.get(), &g)?;
                    check_nan(&accumulated, input_id);
                    slot.insert(accumulated);
                }
                Entry::Vacant(slot) => {
                    slot.insert(g);
                }
            }
        }
    }

    Ok(WgpuGrads { grads })
}

fn add_wgpu_storage(a: &WgpuStorage, b: &WgpuStorage) -> Result<WgpuStorage> {
    debug_assert_eq!(
        a.shape, b.shape,
        "tape accumulation requires matching shapes"
    );
    let n = a.shape.iter().product::<usize>();
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n, OperationKind::Storage)?;
    let params = [0, n as u32]; // op_mode 0=add
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
    let total: usize = out_shape.iter().product();
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, total, OperationKind::Storage)?;

    let inner_stride: usize = storage.shape[axis + 1..].iter().product();

    dispatch::dispatch_reduce_dim(
        &storage.buffer,
        &out_buf,
        0, // sum
        storage.shape[axis] as u32,
        inner_stride as u32,
        total as u32,
    );
    Ok(WgpuStorage::new(out_buf, out_shape))
}
