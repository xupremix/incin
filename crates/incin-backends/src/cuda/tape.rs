use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::cell::RefCell;

use incin_core::exec::{GradMode, NanPolicy};
use incin_core::prelude::Result;
use incin_core::prelude::{BackwardError, NonFiniteSite};

use crate::cuda::storage::{CudaBuffer, CudaStorage, TensorId};

pub(crate) struct TapeEntry {
    pub(crate) output_id: TensorId,
    pub(crate) input_ids: Vec<TensorId>,
    /// Fallible since `GRD-005`: a recipe that cannot produce a gradient
    /// reports it. `GRD-006` replaces this backend-local type with the core's
    /// `TapeNode`, which already has this signature.
    pub(crate) backward: Box<dyn Fn(&CudaStorage) -> Result<Vec<CudaStorage>> + Send + Sync>,
}

thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = const { RefCell::new(Vec::new()) };
}

/// Push a `TapeEntry`, unless the ambient [`GradMode`] forbids recording
/// (`GRD-002`). One gate per tape rather than one per call site: a `NoGrad`
/// operation must record nothing whichever of the backend's kernels ran it.
pub(crate) fn push(entry: TapeEntry) {
    if !GradMode::current().records() {
        return;
    }
    TAPE.with(|t| t.borrow_mut().push(entry));
}

/// Number of entries currently on the tape, for tests outside this crate that
/// have to observe the `GRD-002` guarantee rather than take it on faith.
#[must_use]
pub fn depth() -> usize {
    TAPE.with(|t| t.borrow().len())
}

/// The CUDA backend's gradient container (`Backend::Grads`). The `grads`
/// map itself is private — use `.get(id)` — so downstream crates can't
/// inspect/mutate the internal `BTreeMap` beyond the intended query API.
pub struct CudaGrads {
    pub(crate) grads: BTreeMap<TensorId, CudaStorage>,
}

impl CudaGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    pub fn get(&self, id: TensorId) -> Option<&CudaStorage> {
        self.grads.get(&id)
    }
}

pub(crate) fn backward(loss: &CudaStorage) -> Result<CudaGrads> {
    let mut grads: BTreeMap<TensorId, CudaStorage> = BTreeMap::new();

    let numel = loss.shape.iter().product::<usize>();
    let device_id = loss.buffer.device_id;
    let stream = loss.buffer.device.default_stream();
    let data_ones = vec![1.0f32; numel];
    let data_u8: &[u8] = bytemuck::cast_slice(&data_ones);
    let u8_slice = stream.clone_htod(data_u8).unwrap();

    let buf = CudaBuffer {
        len: numel,
        dtype: loss.buffer.dtype,
        data: alloc::sync::Arc::new(u8_slice),
        device: loss.buffer.device.clone(),
        device_id,
    };
    grads.insert(
        loss.id,
        CudaStorage::new(alloc::sync::Arc::new(buf), loss.shape.to_vec()),
    );

    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));

    // Read once for the whole pass, exactly as the core and WGPU walks do:
    // the point of `GRD-005` making this a policy is that every backend
    // answers the same question the same way. This backend had no check at
    // all — its `backward_with_nan_check` delegated to `backward` — so a CUDA
    // user asking where a `NaN` came from was told nothing.
    let checked = NanPolicy::current().checks();

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out)?;
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            if checked {
                check_finite(&g, input_id, NonFiniteSite::Contribution)?;
            }
            match grads.entry(input_id) {
                Entry::Occupied(mut slot) => {
                    // Fallible since `GRD-005`. `and_modify` could not carry a
                    // failure, so the accumulating kernel unwrapped: a launch
                    // failure during backward aborted the process.
                    let accumulated = add_cuda_storage(slot.get(), &g)?;
                    if checked {
                        check_finite(&accumulated, input_id, NonFiniteSite::Accumulation)?;
                    }
                    slot.insert(accumulated);
                }
                Entry::Vacant(slot) => {
                    slot.insert(g);
                }
            }
        }
    }

    Ok(CudaGrads { grads })
}

fn add_cuda_storage(a: &CudaStorage, b: &CudaStorage) -> Result<CudaStorage> {
    crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", a, b, &a.shape)
}

/// Report the tensor whose gradient went non-finite.
///
/// The readback is why the policy defaults to permitting: on this backend the
/// check is a full device-to-host copy per gradient.
fn check_finite(storage: &CudaStorage, id: TensorId, site: NonFiniteSite) -> Result<()> {
    let bytes = storage
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*storage.buffer.data)
        .map_err(|error| {
            incin_core::prelude::Error::Msg(alloc::format!("CUDA download failed: {error:?}"))
        })?;
    let non_finite = bytemuck::cast_slice::<u8, f32>(&bytes)
        .iter()
        .any(|x| x.is_nan() || x.is_infinite());
    if non_finite {
        return Err(BackwardError::NonFinite {
            tensor: id.get(),
            operation: site,
        }
        .into());
    }
    Ok(())
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
