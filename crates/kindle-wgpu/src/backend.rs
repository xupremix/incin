use crate::dispatch;
use crate::storage::{WgpuBuffer, WgpuStorage};
use kindle_core::prelude::*;

/// WebGPU compute backend for Kindle.
/// This backend evaluates tensor operations by compiling WGSL compute shaders
/// and dispatching them to the user's primary GPU adapter via `wgpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuBackend<T, D>(core::marker::PhantomData<(T, D)>);

#[derive(Clone)]
/// Auto-generated documentation for WgpuVar.
pub struct WgpuVar {
    /// Auto-generated documentation for storage.
    pub storage: WgpuStorage,
}

/// Auto-generated documentation for WgpuGrads.
pub struct WgpuGrads {}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compute flat element count from shape
// ─────────────────────────────────────────────────────────────────────────────
/// Auto-generated documentation for num_elements.
fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend core trait
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> Backend for WgpuBackend<T, D> {
    /// Auto-generated documentation for Device.
    type Device = D;
    /// Auto-generated documentation for FloatElem.
    type FloatElem = T;
    /// Auto-generated documentation for IntElem.
    type IntElem = i64;
    /// Auto-generated documentation for BackendWithDevice.
    type BackendWithDevice<NewD: Device> = WgpuBackend<T, NewD>;

    /// Auto-generated documentation for Storage.
    type Storage<K: DType> = WgpuStorage;
    /// Auto-generated documentation for RawVar.
    type RawVar = WgpuVar;
    /// Auto-generated documentation for Grads.
    type Grads = WgpuGrads;
    /// Auto-generated documentation for InnerBackend.
    type InnerBackend = Self;

    /// Auto-generated documentation for shape.
    fn shape<K: DType>(t: &Self::Storage<K>) -> Vec<usize> {
        t.shape.clone()
    }

    /// Auto-generated documentation for format_tensor_display.
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> String {
        "WgpuTensor(...)".to_string()
    }

    /// Auto-generated documentation for format_tensor_debug.
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> String {
        format!("WgpuTensor(shape={:?})", t.shape)
    }

    /// Auto-generated documentation for var_as_tensor.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }

    /// Auto-generated documentation for var_from_tensor.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(WgpuVar { storage: t.clone() })
    }

    /// Auto-generated documentation for var_to_device.
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        Ok(WgpuVar {
            storage: var.storage.clone(),
        })
    }

    /// Auto-generated documentation for assign_var.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }

    /// Auto-generated documentation for backward.
    fn backward<K: DType>(_loss: &Self::Storage<K>) -> Result<Self::Grads> {
        unimplemented!("Backward pass not yet implemented for WgpuBackend")
    }

    /// Auto-generated documentation for backward_with_nan_check.
    fn backward_with_nan_check<K: DType>(_loss: &Self::Storage<K>) -> Result<Self::Grads> {
        unimplemented!("Backward pass not yet implemented for WgpuBackend")
    }

    /// Auto-generated documentation for get_grad.
    fn get_grad<K: DType>(
        _t: &Self::Storage<K>,
        _grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        unimplemented!("Grads not yet implemented for WgpuBackend")
    }

    /// Auto-generated documentation for to_bytes.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        Ok(t.buffer.to_vec::<u8>())
    }

    /// Auto-generated documentation for from_bytes.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<Self::Storage<K>> {
        let buffer = WgpuBuffer::from_slice(bytes);
        Ok(WgpuStorage::new(buffer, shape.to_vec()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CreationOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> CreationOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for zeros.
    fn zeros<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let n = num_elements(shape);
        let data: Vec<f32> = vec![0.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Auto-generated documentation for ones.
    fn ones<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let n = num_elements(shape);
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Auto-generated documentation for rand.
    fn rand<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape);
        // Simple LCG for now – GPU-side random generation would need more infrastructure
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32)
            })
            .collect();
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Auto-generated documentation for randn.
    fn randn<K: DType>(
        shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape);
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let lcg = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f32) / (u32::MAX as f32)
        };
        // Box-Muller transform
        let data: Vec<f32> = (0..((n + 1) / 2))
            .flat_map(|_| {
                let u1 = lcg(&mut state).max(1e-7);
                let u2 = lcg(&mut state);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                [r * theta.cos(), r * theta.sin()]
            })
            .take(n)
            .collect();
        let buf = WgpuBuffer::from_slice(&data);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Auto-generated documentation for var_zeros.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Auto-generated documentation for var_ones.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Auto-generated documentation for var_rand.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Auto-generated documentation for var_randn.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(WgpuVar { storage: s })
    }

    /// Auto-generated documentation for tensor_to_device.
    fn tensor_to_device<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        _device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // WGPU buffers are already on the GPU
        Ok(t.clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NumericOps  (add, sub, mul, div)
// ─────────────────────────────────────────────────────────────────────────────
/// Auto-generated documentation for binary_op.
fn binary_op<T: DType, D: Device>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    op_mode: u32,
    op_name: &'static str,
) -> Result<WgpuStorage> {
    if lhs.shape != rhs.shape {
        return Err(Error::ShapeMismatch {
            op: op_name,
            expected: lhs.shape.clone(),
            got: rhs.shape.clone(),
            msg: "shapes must match for elementwise op".to_string(),
        });
    }
    let n = num_elements(&lhs.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(lhs.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_binary(&lhs.buffer, &rhs.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, lhs.shape.clone()))
}

impl<T: DType, D: Device> NumericOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for add.
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T, D>(lhs, rhs, 0, "add")
    }
    /// Auto-generated documentation for sub.
    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T, D>(lhs, rhs, 1, "sub")
    }
    /// Auto-generated documentation for mul.
    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T, D>(lhs, rhs, 2, "mul")
    }
    /// Auto-generated documentation for div.
    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        binary_op::<T, D>(lhs, rhs, 3, "div")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FloatOps  (scalar + unary activations)
// ─────────────────────────────────────────────────────────────────────────────
/// Auto-generated documentation for unary_op.
fn unary_op<T: DType, D: Device>(t: &WgpuStorage, op_mode: u32) -> Result<WgpuStorage> {
    let n = num_elements(&t.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let params = [op_mode, n];
    dispatch::dispatch_unary(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.clone()))
}

/// Auto-generated documentation for scalar_op.
fn scalar_op<T: DType, D: Device>(
    t: &WgpuStorage,
    scalar: f64,
    op_mode: u32,
) -> Result<WgpuStorage> {
    let n = num_elements(&t.shape) as u32;
    let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
    let scalar_bits = (scalar as f32).to_bits();
    let params = [op_mode, n, scalar_bits];
    dispatch::dispatch_scalar(&t.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, t.shape.clone()))
}

impl<T: DType, D: Device> FloatOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for add_scalar_float.
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        scalar_op::<T, D>(t, scalar, 0)
    }
    /// Auto-generated documentation for mul_scalar_float.
    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        scalar_op::<T, D>(t, scalar, 1)
    }
    /// Auto-generated documentation for relu.
    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 0)
    }
    /// Auto-generated documentation for step.
    fn step<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 10)
    }
    /// Auto-generated documentation for mish.
    fn mish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 11)
    }
    /// Auto-generated documentation for elu.
    fn elu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 12)
    }
    /// Auto-generated documentation for gelu.
    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 1)
    }
    /// Auto-generated documentation for tanh.
    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 2)
    }
    /// Auto-generated documentation for sigmoid.
    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 3)
    }
    /// Auto-generated documentation for abs.
    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 4)
    }
    /// Auto-generated documentation for neg.
    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 5)
    }
    /// Auto-generated documentation for sqrt.
    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 6)
    }
    /// Auto-generated documentation for exp.
    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 7)
    }
    /// Auto-generated documentation for log.
    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 8)
    }
    /// Auto-generated documentation for swish.
    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        unary_op::<T, D>(t, 9)
    }

    /// Auto-generated documentation for softmax.
    fn softmax<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        // Flatten to [batch, n] where n = shape[dim..] product
        let n: usize = shape[dim..].iter().product();
        let batch: usize = shape[..dim].iter().product::<usize>().max(1);
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
        dispatch::dispatch_softmax(&t.buffer, &out_buf, batch as u32, n as u32);
        Ok(WgpuStorage::new(out_buf, shape.clone()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorOps  (reshape, transpose, matmul, narrow, flatten, squeeze, stack, concat, etc.)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> TensorOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for matmul.
    fn matmul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: vec![2],
                got: vec![lhs.shape.len(), rhs.shape.len()],
                msg: "matmul requires at least 2D inputs".to_string(),
            });
        }
        
        let lhs_rank = lhs.shape.len();
        let rhs_rank = rhs.shape.len();
        
        let m = lhs.shape[lhs_rank - 2] as u32;
        let k = lhs.shape[lhs_rank - 1] as u32;
        let n = rhs.shape[rhs_rank - 1] as u32;

        if k as usize != rhs.shape[rhs_rank - 2] {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.clone(),
                got: rhs.shape.clone(),
                msg: "matmul inner dims must match".to_string(),
            });
        }

        // Compute batch dims
        let mut lhs_batch = 1;
        for i in 0..lhs_rank - 2 { lhs_batch *= lhs.shape[i]; }
        let mut rhs_batch = 1;
        for i in 0..rhs_rank - 2 { rhs_batch *= rhs.shape[i]; }
        
        let batch = core::cmp::max(lhs_batch, rhs_batch);
        if lhs_batch != 1 && rhs_batch != 1 && lhs_batch != rhs_batch {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.clone(),
                got: rhs.shape.clone(),
                msg: "matmul batch dims incompatible".to_string(),
            });
        }
        
        let lhs_stride_b = if lhs_batch == 1 { 0 } else { m * k };
        let rhs_stride_b = if rhs_batch == 1 { 0 } else { k * n };

        // Output shape matches the larger batched input
        let mut out_shape = if lhs_batch > 1 { lhs.shape[..lhs_rank - 2].to_vec() } else { rhs.shape[..rhs_rank - 2].to_vec() };
        if out_shape.is_empty() && batch > 1 { out_shape.push(batch); }
        out_shape.push(m as usize);
        out_shape.push(n as usize);

        let state = crate::device::get_device_state();
        let shader = include_str!("shaders/matmul.wgsl");
        let pipeline = crate::pipeline::get_or_create_pipeline("matmul", shader, "main");

        let out_buf = WgpuBuffer::new_zeros((batch as u32 * m * n) as usize * core::mem::size_of::<f32>());
        let shape_data = [m, k, n, batch as u32, lhs_stride_b, rhs_stride_b];
        let shape_buf = WgpuBuffer::from_slice(&shape_data);

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shape_buf.buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Matmul"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, batch as u32);
        }
        state.queue.submit(core::iter::once(encoder.finish()));
        Ok(WgpuStorage::new(out_buf, out_shape))
    }

    /// Auto-generated documentation for reshape.
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        if num_elements(&t.shape) != num_elements(shape) {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: t.shape.clone(),
                got: shape.to_vec(),
                msg: "total elements must match".to_string(),
            });
        }
        // Reshape is metadata-only (contiguous buffer reuse)
        Ok(WgpuStorage {
            buffer: t.buffer.clone(),
            shape: shape.to_vec(),
            strides: vec![],
        })
    }

    /// Auto-generated documentation for transpose.
    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        if shape.len() != 2 {
            // Only 2D transpose via shader; for N-D just swap dims in metadata (no copy, contiguous assumed)
            let mut new_shape = shape.clone();
            new_shape.swap(dim1, dim2);
            return Ok(WgpuStorage {
                buffer: t.buffer.clone(),
                shape: new_shape,
                strides: vec![],
            });
        }
        let rows = shape[0] as u32;
        let cols = shape[1] as u32;
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);
        dispatch::dispatch_transpose(&t.buffer, &out_buf, rows, cols);
        Ok(WgpuStorage::new(
            out_buf,
            vec![cols as usize, rows as usize],
        ))
    }

    /// Auto-generated documentation for flatten.
    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let flat_size: usize = shape[start_dim..=end_dim].iter().product();
        let mut new_shape: Vec<usize> = shape[..start_dim].to_vec();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&shape[end_dim + 1..]);
        Ok(WgpuStorage {
            buffer: t.buffer.clone(),
            shape: new_shape,
            strides: vec![],
        })
    }

    /// Auto-generated documentation for squeeze.
    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut new_shape = t.shape.clone();
        if new_shape[dim] == 1 {
            new_shape.remove(dim);
        }
        Ok(WgpuStorage {
            buffer: t.buffer.clone(),
            shape: new_shape,
            strides: vec![],
        })
    }

    /// Auto-generated documentation for narrow.
    fn narrow<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Copy the slice for dim. For now: CPU-round-trip for generality.
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let shape = &t.shape;

        // Compute strides
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut new_shape = shape.clone();
        new_shape[dim] = len;
        let out_n = num_elements(&new_shape);
        let mut out_data = vec![0.0f32; out_n];

        // Iterate over output indices
        /// Auto-generated documentation for fill.
        fn fill(
            data: &[f32],
            out: &mut [f32],
            shape: &[usize],
            new_shape: &[usize],
            strides: &[usize],
            dim: usize,
            start: usize,
            idx: &mut Vec<usize>,
            out_idx: &mut Vec<usize>,
            depth: usize,
        ) {
            if depth == shape.len() {
                let in_flat: usize = idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
                let out_flat: usize = out_idx
                    .iter()
                    .zip(
                        {
                            let mut s = vec![1usize; out_idx.len()];
                            for i in (0..out_idx.len() - 1).rev() {
                                s[i] = s[i + 1] * new_shape[i + 1];
                            }
                            s
                        }
                        .iter(),
                    )
                    .map(|(i, s)| i * s)
                    .sum();
                out[out_flat] = data[in_flat];
                return;
            }
            let range = if depth == dim {
                start..(start + new_shape[dim])
            } else {
                0..shape[depth]
            };
            for i in range {
                idx.push(i);
                let out_i = if depth == dim { i - start } else { i };
                out_idx.push(out_i);
                fill(
                    data,
                    out,
                    shape,
                    new_shape,
                    strides,
                    dim,
                    start,
                    idx,
                    out_idx,
                    depth + 1,
                );
                idx.pop();
                out_idx.pop();
            }
        }

        fill(
            &data,
            &mut out_data,
            shape,
            &new_shape,
            &strides,
            dim,
            start,
            &mut vec![],
            &mut vec![],
            0,
        );

        let buf = WgpuBuffer::from_slice(&out_data);
        Ok(WgpuStorage::new(buf, new_shape))
    }

    /// Auto-generated documentation for broadcast_as.
    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        // CPU-side for now
        let src: Vec<f32> = t.buffer.to_vec::<f32>();
        let src_shape = &t.shape;
        let out_n = num_elements(shape);
        let mut out = vec![0.0f32; out_n];
        let ndim = shape.len();
        let pad = ndim - src_shape.len();

        for i in 0..out_n {
            let mut rem = i;
            let mut src_i = 0usize;
            let mut src_stride = 1usize;
            for d in (0..ndim).rev() {
                let dim_size = shape[d];
                let coord = rem % dim_size;
                rem /= dim_size;
                if d >= pad {
                    let sd = d - pad;
                    let src_dim = src_shape[sd];
                    let src_coord = if src_dim == 1 { 0 } else { coord };
                    src_i += src_coord * src_stride;
                    src_stride *= src_shape[sd];
                }
            }
            out[i] = src[src_i];
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// Auto-generated documentation for broadcast_left.
    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Self::broadcast_as::<K>(t, shape)
    }

    /// Auto-generated documentation for slice.
    fn slice<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Start with narrow on each dim
        let mut cur = t.clone();
        for (d, &(start, end)) in ranges.iter().enumerate() {
            cur = Self::narrow::<K>(&cur, d, start, end - start)?;
        }
        Ok(cur)
    }

    /// Auto-generated documentation for stack.
    fn stack<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::Msg("stack: empty tensor list".to_string()));
        }
        // Unsqueeze each tensor at `dim` then concat
        let unsqueezed: Vec<WgpuStorage> = tensors
            .iter()
            .map(|t| {
                let mut new_shape = t.shape.clone();
                new_shape.insert(dim, 1);
                WgpuStorage {
                    buffer: t.buffer.clone(),
                    shape: new_shape,
                    strides: vec![],
                }
            })
            .collect();
        let refs: Vec<&WgpuStorage> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// Auto-generated documentation for concat.
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::Msg("concat: empty tensor list".to_string()));
        }
        // CPU-side concat for generality
        let first_shape = &tensors[0].shape;
        let ndim = first_shape.len();
        let total_dim_size: usize = tensors.iter().map(|t| t.shape[dim]).sum();
        let mut out_shape = first_shape.clone();
        out_shape[dim] = total_dim_size;

        // Simple approach: serialize each tensor and reconstruct
        // For contiguous tensors along dim, we can just concatenate with stride awareness
        let all_data: Vec<Vec<f32>> = tensors.iter().map(|t| t.buffer.to_vec::<f32>()).collect();

        let out_n = num_elements(&out_shape);
        let mut out = vec![0.0f32; out_n];

        // Compute out_strides
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        for i in 0..out_n {
            // Decompose flat index to coords in out_shape
            let mut coords = vec![0usize; ndim];
            let mut rem = i;
            for d in (0..ndim).rev() {
                coords[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            // Find which tensor owns this index along dim
            let dim_offset = coords[dim];
            let (tensor_idx, local_dim) = {
                let mut t_idx = 0;
                let mut local = dim_offset;
                for (ti, t) in tensors.iter().enumerate() {
                    if local < t.shape[dim] {
                        t_idx = ti;
                        break;
                    }
                    local -= t.shape[dim];
                }
                (t_idx, local)
            };
            let mut src_coords = coords.clone();
            src_coords[dim] = local_dim;
            let mut src_strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                src_strides[d] = src_strides[d + 1] * tensors[tensor_idx].shape[d + 1];
            }
            let src_flat: usize = src_coords
                .iter()
                .zip(src_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            out[i] = all_data[tensor_idx][src_flat];
        }

        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, out_shape))
    }

    /// Auto-generated documentation for float_to_scalar.
    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.first().copied().unwrap_or(0.0) as f64)
    }

    /// Auto-generated documentation for float_to_vec1.
    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<f64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    /// Auto-generated documentation for int_to_scalar.
    fn int_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.first().copied().unwrap_or(0.0) as i64)
    }

    /// Auto-generated documentation for int_to_vec1.
    fn int_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<i64>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        Ok(data.iter().map(|&x| x as i64).collect())
    }

    /// Auto-generated documentation for tensor_to_dtype.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as Backend>::Storage<K>,
        _dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        // Simple passthrough (all stored as f32 internally)
        Ok(WgpuStorage {
            buffer: t.buffer.clone(),
            shape: t.shape.clone(),
            strides: t.strides.clone(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReductionOps
// ─────────────────────────────────────────────────────────────────────────────
/// Auto-generated documentation for reduce_all_to_storage.
fn reduce_all_to_storage(t: &WgpuStorage, mode: u32) -> WgpuStorage {
    let n = num_elements(&t.shape) as u32;
    let out = dispatch::dispatch_reduce_all(&t.buffer, n, mode);
    WgpuStorage::new(out, vec![1])
}

/// Auto-generated documentation for reduce_dim_to_storage.
fn reduce_dim_to_storage(t: &WgpuStorage, dim: usize, mode: u32, keepdim: bool) -> WgpuStorage {
    // Approach: fold over the specified dim via CPU for correctness.
    // GPU version would use strided reduce; this is correct for all cases.
    let data: Vec<f32> = t.buffer.to_vec::<f32>();
    let shape = &t.shape;
    let ndim = shape.len();

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut out_shape = shape.clone();
    out_shape[dim] = 1;
    let out_n = num_elements(&out_shape);
    let init = if mode == 0 {
        0.0f32
    } else if mode == 1 {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut out = vec![init; out_n];

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    for in_i in 0..data.len() {
        let mut rem = in_i;
        let mut out_i = 0usize;
        for d in (0..ndim).rev() {
            let coord = rem % shape[d];
            rem /= shape[d];
            let out_coord = if d == dim { 0 } else { coord };
            out_i += out_coord * out_strides[d];
        }
        if mode == 0 {
            out[out_i] += data[in_i];
        } else if mode == 1 {
            out[out_i] = out[out_i].max(data[in_i]);
        } else {
            out[out_i] = out[out_i].min(data[in_i]);
        }
    }

    let final_shape = if keepdim {
        out_shape
    } else {
        let mut s = shape.clone();
        s.remove(dim);
        s
    };
    let buf = WgpuBuffer::from_slice(&out);
    WgpuStorage::new(buf, final_shape)
}

impl<T: DType, D: Device> ReductionOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for sum_all.
    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_all_to_storage(t, 0))
    }
    /// Auto-generated documentation for mean_all.
    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_all_to_storage(t, 0);
        let n = num_elements(&t.shape) as f64;
        scalar_op::<T, D>(&sum, 1.0 / n, 1)
    }
    /// Auto-generated documentation for max_all.
    fn max_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_all_to_storage(t, 1))
    }
    /// Auto-generated documentation for min_all.
    fn min_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_all_to_storage(t, 2))
    }

    /// Auto-generated documentation for sum_dim.
    fn sum_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 0, false))
    }
    /// Auto-generated documentation for sum_keepdim.
    fn sum_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 0, true))
    }
    /// Auto-generated documentation for mean_dim.
    fn mean_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, false);
        let n = t.shape[dim] as f64;
        scalar_op::<T, D>(&sum, 1.0 / n, 1)
    }
    /// Auto-generated documentation for mean_keepdim.
    fn mean_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, true);
        let n = t.shape[dim] as f64;
        scalar_op::<T, D>(&sum, 1.0 / n, 1)
    }
    /// Auto-generated documentation for max_dim.
    fn max_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 1, false))
    }
    /// Auto-generated documentation for max_keepdim.
    fn max_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 1, true))
    }
    /// Auto-generated documentation for min_dim.
    fn min_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 2, false))
    }
    /// Auto-generated documentation for min_keepdim.
    fn min_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Ok(reduce_dim_to_storage(t, dim, 2, true))
    }

    /// Auto-generated documentation for argmax.
    fn argmax<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let buf = WgpuBuffer::from_slice(&[idx as u32]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                let mut out_shape = shape.clone();
                out_shape[d] = 1;
                // Simple cpu argmax per slice
                let n_out = num_elements(&out_shape);
                let mut out = vec![0u32; n_out];
                let mut strides = vec![1usize; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let mut out_strides = vec![1usize; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                }

                for i in 0..n_out {
                    let mut rem = i;
                    let mut coords = vec![0usize; shape.len()];
                    for dd in (0..shape.len()).rev() {
                        coords[dd] = rem % out_shape[dd];
                        rem /= out_shape[dd];
                    }
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0;
                    for k in 0..shape[d] {
                        coords[d] = k;
                        let flat: usize =
                            coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
                        if data[flat] > best_val {
                            best_val = data[flat];
                            best_idx = k;
                        }
                    }
                    out[i] = best_idx as u32;
                }
                let mut final_shape = shape.clone();
                final_shape.remove(d);
                let buf = WgpuBuffer::from_slice(&out);
                Ok(WgpuStorage::new(buf, final_shape))
            }
        }
    }

    /// Auto-generated documentation for argmin.
    fn argmin<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        match dim {
            None => {
                let idx = data
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let buf = WgpuBuffer::from_slice(&[idx as u32]);
                Ok(WgpuStorage::new(buf, vec![1]))
            }
            Some(d) => {
                let shape = &t.shape;
                let mut out_shape = shape.clone();
                out_shape[d] = 1;
                let n_out = num_elements(&out_shape);
                let mut out = vec![0u32; n_out];
                let mut strides = vec![1usize; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                for i in 0..n_out {
                    let mut rem = i;
                    let mut coords = vec![0usize; shape.len()];
                    for dd in (0..shape.len()).rev() {
                        coords[dd] = rem % out_shape[dd];
                        rem /= out_shape[dd];
                    }
                    let mut best_val = f32::INFINITY;
                    let mut best_idx = 0;
                    for k in 0..shape[d] {
                        coords[d] = k;
                        let flat: usize =
                            coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
                        if data[flat] < best_val {
                            best_val = data[flat];
                            best_idx = k;
                        }
                    }
                    out[i] = best_idx as u32;
                }
                let mut final_shape = shape.clone();
                final_shape.remove(d);
                let buf = WgpuBuffer::from_slice(&out);
                Ok(WgpuStorage::new(buf, final_shape))
            }
        }
    }

    /// Auto-generated documentation for topk.
    fn topk<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(<Self as Backend>::Storage<K>, <Self as Backend>::Storage<KInt>)> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "topk",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("topk: axis {} out of range", dim),
            });
        }
        let k = k.min(shape[dim]);
        let data: Vec<f32> = t.buffer.to_vec::<f32>();

        let mut out_shape = shape.clone();
        out_shape[dim] = k;
        let mut base_shape = shape.clone();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape);
        let mut out_vals = vec![0.0f32; num_elements(&out_shape)];
        let mut out_indices = vec![0u32; num_elements(&out_shape)];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += coords[dd] * stride;
                    stride *= shape[dd];
                }
                slice_vals.push((data[flat], j as u32));
            }

            if largest {
                slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }

            let mut out_coords = coords.clone();
            for j in 0..k {
                out_coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..out_shape.len()).rev() {
                    flat += out_coords[dd] * stride;
                    stride *= out_shape[dd];
                }
                out_vals[flat] = slice_vals[j].0;
                out_indices[flat] = slice_vals[j].1;
            }
        }
        let buf_vals = WgpuBuffer::from_slice(&out_vals);
        let buf_indices = WgpuBuffer::from_slice(&out_indices);
        Ok((
            WgpuStorage::new(buf_vals, out_shape.clone()),
            WgpuStorage::new(buf_indices, out_shape),
        ))
    }

    /// Auto-generated documentation for argsort.
    fn argsort<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "argsort",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("argsort: axis {} out of range", dim),
            });
        }
        let data: Vec<f32> = t.buffer.to_vec::<f32>();

        let mut base_shape = shape.clone();
        base_shape[dim] = 1;

        let n_slices = num_elements(&base_shape);
        let mut out = vec![0u32; num_elements(shape)];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += coords[dd] * stride;
                    stride *= shape[dd];
                }
                slice_vals.push((data[flat], j as u32));
            }

            if descending {
                slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }

            let mut out_coords = coords.clone();
            for j in 0..shape[dim] {
                out_coords[dim] = j;
                let mut flat = 0usize;
                let mut stride = 1usize;
                for dd in (0..shape.len()).rev() {
                    flat += out_coords[dd] * stride;
                    stride *= shape[dd];
                }
                out[flat] = slice_vals[j].1;
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.clone()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModuleOps
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> ModuleOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for embedding.
    fn embedding<K: DType, KInt: DType>(
        indices: &<Self as Backend>::Storage<KInt>,
        weight: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let idx_data: Vec<f32> = indices.buffer.to_vec::<f32>();
        let weight_data: Vec<f32> = weight.buffer.to_vec::<f32>();
        let embed_dim = weight.shape[1];
        let seq_len = num_elements(&indices.shape);
        let mut out = vec![0.0f32; seq_len * embed_dim];
        for (i, &idx_f) in idx_data.iter().enumerate() {
            let idx = idx_f as usize;
            let src = &weight_data[idx * embed_dim..(idx + 1) * embed_dim];
            let dst = &mut out[i * embed_dim..(i + 1) * embed_dim];
            dst.copy_from_slice(src);
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, vec![seq_len, embed_dim]))
    }

    /// Auto-generated documentation for layer_norm.
    fn layer_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let gamma: Vec<f32> = weight.buffer.to_vec::<f32>();
        let beta_data: Option<Vec<f32>> = bias.map(|b| b.buffer.to_vec::<f32>());
        let shape = &t.shape;
        let norm_size = shape.last().copied().unwrap_or(1);
        let batch = num_elements(shape) / norm_size;
        let mut out = data.clone();
        for b in 0..batch {
            let slice = &data[b * norm_size..(b + 1) * norm_size];
            let mean = slice.iter().sum::<f32>() / norm_size as f32;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;
            let std = (var + eps).sqrt();
            for i in 0..norm_size {
                let norm = (data[b * norm_size + i] - mean) / std;
                out[b * norm_size + i] =
                    norm * gamma[i] + beta_data.as_ref().map_or(0.0, |bv| bv[i]);
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.clone()))
    }

    /// Auto-generated documentation for batch_norm.
    fn batch_norm<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: Option<&<Self as Backend>::Storage<K>>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        running_mean: Option<&<Self as Backend>::Storage<K>>,
        running_var: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let shape = &t.shape; // [N, C, H, W] or [N, C]
        let n_total = num_elements(shape);
        let c = shape.get(1).copied().unwrap_or(1);
        let spatial = n_total / (shape[0] * c);
        let gamma: Option<Vec<f32>> = weight.map(|w| w.buffer.to_vec::<f32>());
        let beta: Option<Vec<f32>> = bias.map(|b| b.buffer.to_vec::<f32>());
        let rm: Option<Vec<f32>> = running_mean.map(|m| m.buffer.to_vec::<f32>());
        let rv: Option<Vec<f32>> = running_var.map(|v| v.buffer.to_vec::<f32>());
        let mut out = data.clone();
        for ch in 0..c {
            let mean = rm.as_ref().map_or_else(
                || {
                    let mut sum = 0.0f32;
                    for n in 0..shape[0] {
                        for s in 0..spatial {
                            sum += data[n * c * spatial + ch * spatial + s];
                        }
                    }
                    sum / (shape[0] * spatial) as f32
                },
                |m| m[ch],
            );
            let var = rv.as_ref().map_or_else(
                || {
                    let mut sum = 0.0f32;
                    for n in 0..shape[0] {
                        for s in 0..spatial {
                            let x = data[n * c * spatial + ch * spatial + s] - mean;
                            sum += x * x;
                        }
                    }
                    sum / (shape[0] * spatial) as f32
                },
                |v| v[ch],
            );
            let std = (var + eps).sqrt();
            let g = gamma.as_ref().map_or(1.0, |gv| gv[ch]);
            let b = beta.as_ref().map_or(0.0, |bv| bv[ch]);
            for n in 0..shape[0] {
                for s in 0..spatial {
                    let idx = n * c * spatial + ch * spatial + s;
                    out[idx] = (data[idx] - mean) / std * g + b;
                }
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape.clone()))
    }

    /// Auto-generated documentation for adaptive_avg_pool2d.
    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape; // [N, C, H, W]
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh, ow) = output_size;
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let mut out = vec![0.0f32; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let h_start = oi * h / oh;
                        let h_end = ((oi + 1) * h + oh - 1) / oh;
                        let w_start = oj * w / ow;
                        let w_end = ((oj + 1) * w + ow - 1) / ow;
                        let mut sum = 0.0f32;
                        let mut cnt = 0;
                        for hi in h_start..h_end {
                            for wi in w_start..w_end {
                                sum += data[bi * c * h * w + ci * h * w + hi * w + wi];
                                cnt += 1;
                            }
                        }
                        out[bi * c * oh * ow + ci * oh * ow + oi * ow + oj] = sum / cnt as f32;
                    }
                }
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, vec![n, c, oh, ow]))
    }

    /// Auto-generated documentation for avg_pool2d.
    fn avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let mut out = vec![0.0f32; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let mut sum = 0.0f32;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = (oi * sh + ki) as isize - ph as isize;
                                let wi = (oj * sw + kj) as isize - pw as isize;
                                if hi >= 0 && hi < h as isize && wi >= 0 && wi < w as isize {
                                    sum += data[bi * c * h * w
                                        + ci * h * w
                                        + hi as usize * w
                                        + wi as usize];
                                }
                            }
                        }
                        out[bi * c * oh * ow + ci * oh * ow + oi * ow + oj] =
                            sum / (kh * kw) as f32;
                    }
                }
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, vec![n, c, oh, ow]))
    }

    /// Auto-generated documentation for max_pool2d.
    fn max_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape;
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;
        let oh = (h + 2 * ph - eff_kh) / sh + 1;
        let ow = (w + 2 * pw - eff_kw) / sw + 1;
        let data: Vec<f32> = t.buffer.to_vec::<f32>();
        let mut out = vec![f32::NEG_INFINITY; n * c * oh * ow];
        for bi in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = (oi * sh + ki * dh) as isize - ph as isize;
                                let wi = (oj * sw + kj * dw) as isize - pw as isize;
                                if hi >= 0 && hi < h as isize && wi >= 0 && wi < w as isize {
                                    let v = data[bi * c * h * w
                                        + ci * h * w
                                        + hi as usize * w
                                        + wi as usize];
                                    let o =
                                        &mut out[bi * c * oh * ow + ci * oh * ow + oi * ow + oj];
                                    if v > *o {
                                        *o = v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, vec![n, c, oh, ow]))
    }

    /// Auto-generated documentation for conv1d.
    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Implement as conv2d over a fake spatial H=1 dimension
        // Input:  [N, C_in, L]       -> [N, C_in, 1, L]
        // Weight: [C_out, C_in, Kl]  -> [C_out, C_in, 1, Kl]
        let inp_shape = &t.shape; // [N, C_in, L]
        let w_shape = &weight.shape; // [C_out, C_in/groups, Kl]
        let (n, c_in, l_in) = (inp_shape[0], inp_shape[1], inp_shape[2]);
        let (c_out, _, kl) = (w_shape[0], w_shape[1], w_shape[2]);

        let inp4d = WgpuStorage {
            buffer: t.buffer.clone(),
            shape: vec![n, c_in, 1, l_in],
            strides: vec![],
        };
        let w4d = WgpuStorage {
            buffer: weight.buffer.clone(),
            shape: vec![c_out, w_shape[1], 1, kl],
            strides: vec![],
        };
        let bias4d = bias;

        let out = Self::conv2d::<K>(&inp4d, &w4d, bias4d, stride, padding, dilation, groups)?;
        // out: [N, C_out, 1, L_out]  -> [N, C_out, L_out]
        let l_out = out.shape[3];
        Ok(WgpuStorage {
            buffer: out.buffer,
            shape: vec![n, c_out, l_out],
            strides: vec![],
        })
    }

    /// Auto-generated documentation for conv2d.
    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // im2col + batched matmul (groups=1 fast path; groups>1 loop)
        let shape = &t.shape; // [N, C_in, H, W]
        let ws = &weight.shape; // [C_out, C_in/groups, Kh, Kw]
        if shape.len() != 4 || ws.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![4],
                got: vec![shape.len()],
                msg: "expected 4D input and weight".into(),
            });
        }
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (c_out, c_in_per_g, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let g = groups;
        let c_in_g = c_in / g;
        assert_eq!(c_in_g, c_in_per_g, "groups mismatch");

        let h_out = (h_in + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;

        // ── im2col ────────────────────────────────────────────────────────────
        // col: [N, C_in * Kh * Kw, H_out * W_out]
        let col_channels = c_in * kh * kw;
        let col_spatial = h_out * w_out;
        let col_buf = WgpuBuffer::new_zeros(batch * col_channels * col_spatial * 4);

        let params: [u32; 14] = [
            batch as u32,
            c_in as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            padding as u32,
            padding as u32,
            dilation as u32,
            dilation as u32,
        ];
        dispatch::dispatch_im2col(&t.buffer, &col_buf, &params);

        // ── matmul per batch: weight [C_out/g, C_in/g * Kh * Kw] x col_slice -> out_slice ──
        // For g=1 this is a single batched matmul.
        // For g>1 we slice and loop.
        let _w_data: Vec<f32> = weight.buffer.to_vec::<f32>();
        let _col_data: Vec<f32> = col_buf.to_vec::<f32>();
        let c_out_g = c_out / g;
        let k_size = c_in_g * kh * kw;

        if g == 1 {
            // GPU batched matmul fast path
            let w_storage = WgpuStorage {
                buffer: weight.buffer.clone(),
                shape: vec![c_out, k_size],
                strides: vec![],
            };
            let col_storage = WgpuStorage {
                buffer: col_buf,
                shape: vec![batch, k_size, col_spatial],
                strides: vec![],
            };
            let out_storage = Self::matmul::<K>(&w_storage, &col_storage)?;

            // Apply bias on GPU (if present)
            let out_buf = if let Some(b_storage) = bias {
                // Bias addition can be done by a broadcast add shader, or here on CPU if lazy.
                // For now, doing it on CPU as a bridge, but ideally dispatch_binary should be used.
                let mut out_data = out_storage.buffer.to_vec::<f32>();
                let b_data = b_storage.buffer.to_vec::<f32>();
                for b in 0..batch {
                    for oc in 0..c_out {
                        let off = b * c_out * col_spatial + oc * col_spatial;
                        let bias_val = b_data[oc];
                        for s in 0..col_spatial {
                            out_data[off + s] += bias_val;
                        }
                    }
                }
                WgpuBuffer::from_slice(&out_data)
            } else {
                out_storage.buffer
            };

            return Ok(WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]));
        }

        // ── matmul per batch for groups > 1 ──
        let w_data: Vec<f32> = weight.buffer.to_vec::<f32>();
        let col_data: Vec<f32> = col_buf.to_vec::<f32>();

        let mut out_data = vec![0.0f32; batch * c_out * h_out * w_out];

        for b in 0..batch {
            for gi in 0..g {
                let w_off = gi * c_out_g * k_size;
                let col_off = b * col_channels * col_spatial + gi * k_size * col_spatial;
                let out_off = b * c_out * col_spatial + gi * c_out_g * col_spatial;

                for oc in 0..c_out_g {
                    for sp in 0..col_spatial {
                        let mut acc = 0.0f32;
                        for k in 0..k_size {
                            acc += w_data[w_off + oc * k_size + k]
                                * col_data[col_off + k * col_spatial + sp];
                        }
                        out_data[out_off + oc * col_spatial + sp] = acc;
                    }
                }
            }
        }

        if let Some(b_storage) = bias {
            let b_data: Vec<f32> = b_storage.buffer.to_vec::<f32>();
            for b in 0..batch {
                for oc in 0..c_out {
                    let off = b * c_out * h_out * w_out + oc * h_out * w_out;
                    for s in 0..h_out * w_out {
                        out_data[off + s] += b_data[oc];
                    }
                }
            }
        }

        let buf = WgpuBuffer::from_slice(&out_data);
        Ok(WgpuStorage::new(buf, vec![batch, c_out, h_out, w_out]))
    }

    /// Auto-generated documentation for conv_transpose2d.
    fn conv_transpose2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        dilation: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let shape = &t.shape; // [N, C_in, H_in, W_in]
        let ws = &weight.shape; // [C_in, C_out/groups, kH, kW]
        
        if shape.len() != 4 || ws.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![4],
                got: vec![shape.len()],
                msg: "expected 4D input and weight".into(),
            });
        }
        
        let batch = shape[0];
        let c_in = shape[1];
        let h_in = shape[2];
        let w_in = shape[3];
        
        let w_c_in = ws[0];
        let c_out_per_group = ws[1];
        let kh = ws[2];
        let kw = ws[3];
        
        let c_out = c_out_per_group * groups;
        assert_eq!(c_in, w_c_in, "Input channels must match weight in_channels");

        let h_out = (h_in - 1) * stride + dilation * (kh - 1) + output_padding + 1;
        let h_out = h_out.saturating_sub(2 * padding);
        let w_out = (w_in - 1) * stride + dilation * (kw - 1) + output_padding + 1;
        let w_out = w_out.saturating_sub(2 * padding);

        let out_buf = WgpuBuffer::new_zeros(batch * c_out * h_out * w_out * 4);
        
        let params: [u32; 16] = [
            batch as u32,
            c_in as u32,
            c_out as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            kh as u32,
            kw as u32,
            stride as u32,
            stride as u32,
            padding as u32,
            padding as u32,
            dilation as u32,
            dilation as u32,
            groups as u32,
        ];
        
        dispatch::dispatch_conv_transpose2d(&t.buffer, &weight.buffer, &out_buf, &params);
        
        let out_storage = WgpuStorage::new(out_buf, vec![batch, c_out, h_out, w_out]);
        
        if let Some(b_storage) = bias {
            let mut out_data = out_storage.buffer.to_vec::<f32>();
            let b_data = b_storage.buffer.to_vec::<f32>();
            for b in 0..batch {
                for oc in 0..c_out {
                    let off = b * c_out * h_out * w_out + oc * h_out * w_out;
                    let bias_val = b_data[oc];
                    for s in 0..h_out * w_out {
                        out_data[off + s] += bias_val;
                    }
                }
            }
            let buf = WgpuBuffer::from_slice(&out_data);
            return Ok(WgpuStorage::new(buf, vec![batch, c_out, h_out, w_out]));
        }
        
        Ok(out_storage)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LossOps (cross_entropy delegated to base trait which composes from float/reduce ops)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> LossOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for cross_entropy_loss.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<KInt>,
        reduction: kindle_core::prelude::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Compute softmax then nll
        let softmax = <Self as FloatOps<Self>>::softmax::<K>(pred, pred.shape.len() - 1)?;
        let log_sm = <Self as FloatOps<Self>>::log::<K>(&softmax)?;
        let log_data: Vec<f32> = log_sm.buffer.to_vec::<f32>();
        let idx_data: Vec<f32> = target.buffer.to_vec::<f32>();
        let batch = idx_data.len();
        let n_classes = pred.shape.last().copied().unwrap_or(1);
        let nlls: Vec<f32> = idx_data
            .iter()
            .enumerate()
            .map(|(i, &idx)| -log_data[i * n_classes + idx as usize])
            .collect();
        let out = match reduction {
            kindle_core::prelude::Reduction::None => nlls,
            kindle_core::prelude::Reduction::Mean => vec![nlls.iter().sum::<f32>() / batch as f32],
            kindle_core::prelude::Reduction::Sum => vec![nlls.iter().sum::<f32>()],
        };
        let shape = match reduction {
            kindle_core::prelude::Reduction::None => vec![batch],
            _ => vec![1],
        };
        let buf = WgpuBuffer::from_slice(&out);
        Ok(WgpuStorage::new(buf, shape))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedOps (stub)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> QuantizedOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for quantize.
    fn quantize<K: FloatDType, Q: QuantDType>(
        _t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<Q>> {
        unimplemented!()
    }
    /// Auto-generated documentation for dequantize.
    fn dequantize<Q: QuantDType, K: FloatDType>(
        _t: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!()
    }
    /// Auto-generated documentation for quantized_matmul.
    fn quantized_matmul<Q: QuantDType>(
        _lhs: &<Self as Backend>::Storage<Q>,
        _rhs: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<f32>> {
        unimplemented!()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OptimizerOps (AdamW)
// ─────────────────────────────────────────────────────────────────────────────
impl<T: DType, D: Device> OptimizerOps<Self> for WgpuBackend<T, D> {
    /// Auto-generated documentation for adamw_step.
    fn adamw_step<K: DType>(
        var: &mut <Self as Backend>::RawVar,
        grad: &<Self as Backend>::Storage<K>,
        m: &mut <Self as Backend>::Storage<K>,
        v: &mut <Self as Backend>::Storage<K>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<()> {
        let n = num_elements(&var.storage.shape) as u32;
        let bc1 = (1.0 - beta1.powi(step as i32)) as f32;
        let bc2 = (1.0 - beta2.powi(step as i32)) as f32;

        // Pack all hyperparams as f32 bits in a u32 metadata buffer
        let meta: [u32; 8] = [
            n,
            (lr as f32).to_bits(),
            (beta1 as f32).to_bits(),
            (beta2 as f32).to_bits(),
            (eps as f32).to_bits(),
            (weight_decay as f32).to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ];

        dispatch::dispatch_adamw(
            &var.storage.buffer,
            &grad.buffer,
            &m.buffer,
            &v.buffer,
            &meta,
        );
        Ok(())
    }
}
