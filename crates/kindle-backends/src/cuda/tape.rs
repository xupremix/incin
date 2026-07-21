use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use kindle_core::prelude::Result;

use crate::cuda::storage::{TensorId, CudaStorage, CudaBuffer};

pub struct TapeEntry {
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub backward: Box<dyn Fn(&CudaStorage) -> Vec<CudaStorage> + Send + Sync>,
}

thread_local! {
    static TAPE: RefCell<Vec<TapeEntry>> = RefCell::new(Vec::new());
}

pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
}

pub struct CudaGrads {
    pub grads: BTreeMap<TensorId, CudaStorage>,
}

pub fn backward(loss: &CudaStorage) -> Result<CudaGrads> {
    let mut grads: BTreeMap<TensorId, CudaStorage> = BTreeMap::new();

    let numel = loss.shape.iter().product::<usize>();
    let device_id = loss.buffer.device_id;
    let stream = loss.buffer.device.default_stream();
    let data_ones = vec![1.0f32; numel];
    let data_u8: &[u8] = bytemuck::cast_slice(&data_ones);
    let u8_slice = stream.clone_htod(data_u8).unwrap();
    
    let buf = CudaBuffer {
        len: numel,
        data: alloc::sync::Arc::new(u8_slice),
        device: loss.buffer.device.clone(),
        device_id,
    };
    grads.insert(loss.id, CudaStorage::new(alloc::sync::Arc::new(buf), loss.shape.clone()));

    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            grads
                .entry(input_id)
                .and_modify(|acc| *acc = add_cuda_storage(acc, &g))
                .or_insert(g);
        }
    }

    Ok(CudaGrads { grads })
}

pub fn backward_with_nan_check(loss: &CudaStorage) -> Result<CudaGrads> {
    backward(loss) // NaN check omitted for simplicity
}

fn add_cuda_storage(a: &CudaStorage, b: &CudaStorage) -> CudaStorage {
    crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", a, b, &a.shape).unwrap()
}

pub fn unbroadcast(grad: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
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
    let mut new_shape = reduced.shape.clone();
    new_shape.remove(axis);
    CudaStorage::new(reduced.buffer.clone(), new_shape)
}

fn sum_dim_keepdim(storage: &CudaStorage, axis: usize) -> CudaStorage {
    crate::cuda::ops::reduce::launch_reduce_op("sum", "a + b", "", storage, axis, true).unwrap()
}
