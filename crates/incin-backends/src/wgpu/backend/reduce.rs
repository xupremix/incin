//! Reduction WGPU operations: full and per-dimension
//! sum/mean/max/min/prod, with and without keeping the reduced dimension.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
//
// ─────────────────────────────────────────────────────────────────────────────
/// `reduce_all_to_storage`.
pub(crate) fn reduce_all_to_storage(t: &WgpuStorage, mode: u32) -> Result<WgpuStorage> {
    let n = checked_u32(num_elements(&t.shape)?, "WGPU reduction element count")?;
    let out = dispatch::dispatch_reduce_all(&t.buffer, n, mode)?;
    Ok(WgpuStorage::new(out, vec![]))
}

/// `reduce_dim_to_storage`.
pub(crate) fn reduce_dim_to_storage(
    t: &WgpuStorage,
    dim: usize,
    mode: u32,
    keepdim: bool,
) -> Result<WgpuStorage> {
    let shape = &t.shape;
    if dim >= shape.len() {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        }
        .into());
    }
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out_n = num_elements(&out_shape)?;

    let dim_size = checked_u32(shape[dim], "WGPU reduction dimension")?;
    let inner_stride =
        ShapeBuf::from_slice(&shape[dim + 1..]).checked_numel(OperationKind::Reduction)?;

    // mode mapping: CPU reduce_dim mode (0=sum, 1=max, 2=min, 3=product) maps
    // to my shader ops (0=sum, 2=max, 3=min, 6=product).
    let op_mode = match mode {
        0 => 0u32, // sum
        1 => 2u32, // max
        2 => 3u32, // min
        3 => 6u32, // product
        _ => {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Reduction,
                reason: "unknown WGPU reduction mode",
            }));
        }
    };

    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
    dispatch::dispatch_reduce_dim(
        &t.buffer,
        &out_buf,
        op_mode,
        dim_size,
        checked_u32(inner_stride, "WGPU reduction inner stride")?,
        checked_u32(out_n, "WGPU reduction output element count")?,
    )?;

    let final_shape = if keepdim {
        out_shape
    } else {
        let mut s = shape.to_vec();
        s.remove(dim);
        s
    };
    Ok(WgpuStorage::new(out_buf, final_shape))
}

/// Zero-aware cotangent of a full product, mirroring
/// `cpu::ops::reduce::helpers::prod_all_grad` in `f32`: no zeros means
/// `g * prod / x[i]`; exactly one zero means `g * (product of the rest)` at
/// the zero position and zero elsewhere; two or more zeros mean the whole
/// gradient is zero.
fn prod_all_backward(input: &[f32], scalar_grad: f32) -> Vec<f32> {
    let mut nonzero_product = 1.0f32;
    let mut zero_count = 0usize;
    let mut zero_flat_idx = 0usize;
    for (flat, &value) in input.iter().enumerate() {
        if value == 0.0 {
            zero_count += 1;
            zero_flat_idx = flat;
        } else {
            nonzero_product *= value;
        }
    }
    let mut grads = vec![0.0f32; input.len()];
    match zero_count {
        0 => {
            for (grad, &value) in grads.iter_mut().zip(input.iter()) {
                *grad = scalar_grad * nonzero_product / value;
            }
        }
        1 => grads[zero_flat_idx] = scalar_grad * nonzero_product,
        _ => {}
    }
    grads
}

/// Zero-aware cotangent of [`prod_dim`], one independent product per
/// `dim`-slice of `input_shape`, with the same zero rule as
/// [`prod_all_backward`] applied per slice. Mirrors
/// `cpu::ops::reduce::helpers::prod_dim_grad`.
fn prod_dim_backward(
    input_shape: &[usize],
    dim: usize,
    input: &[f32],
    grad_out: &[f32],
) -> Vec<f32> {
    let rank = input_shape.len();
    // Map an input flat index to its slice's flat index in `grad_out`: drop
    // the `dim` coordinate and pack the survivors row-major.
    let slice_of = |flat: usize| -> usize {
        let mut key = 0usize;
        let mut mult = 1usize;
        let mut rest = flat;
        for axis in (0..rank).rev() {
            let coord = rest % input_shape[axis];
            rest /= input_shape[axis];
            if axis != dim {
                key += coord * mult;
                mult *= input_shape[axis];
            }
        }
        key
    };
    let out_len = grad_out.len();
    let mut nonzero_product = vec![1.0f32; out_len];
    let mut zero_count = vec![0usize; out_len];
    let mut zero_flat_idx = vec![0usize; out_len];
    for (flat, &value) in input.iter().enumerate() {
        let slice = slice_of(flat);
        if value == 0.0 {
            zero_count[slice] += 1;
            zero_flat_idx[slice] = flat;
        } else {
            nonzero_product[slice] *= value;
        }
    }
    let mut grads = vec![0.0f32; input.len()];
    for (flat, grad) in grads.iter_mut().enumerate() {
        let slice = slice_of(flat);
        let g = grad_out[slice];
        match zero_count[slice] {
            0 => *grad = g * nonzero_product[slice] / input[flat],
            1 if flat == zero_flat_idx[slice] => *grad = g * nonzero_product[slice],
            _ => {}
        }
    }
    grads
}
/// Splits a contiguous shape into `(outer, axis, inner)` element counts
/// around `dim`, i.e. `shape[..dim]`, `shape[dim]`, `shape[dim+1..]`
/// products. For a contiguous row-major tensor this is enough to address
/// any element directly (`outer_idx*(axis*inner) + axis_idx*inner + inner_idx`)
/// without a general N-dimensional odometer - WGPU storage has no
/// non-contiguous view support, so this always applies.
pub(crate) fn axis_reduce_dims(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    let outer: usize = incin_core::shapes::ShapeBuf::from_slice(&(shape[..dim]))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let axis = shape[dim];
    let inner: usize = incin_core::shapes::ShapeBuf::from_slice(&(shape[dim + 1..]))
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    Ok((outer, axis, inner))
}

/// Backward for `max_dim`/`min_dim`: recomputes each output position's
/// winning (first-encountered, strict `>`/`<`) source position from the
/// captured input, then scatters `grad_out`'s value there with a bare `=`
/// (never `+=` - unlike pooling, a plain axis reduction never has two output
/// positions sharing the same winning source element). Mirrors the CPU
/// backend's `max_axis_with_indices`/`min_axis_with_indices` +
/// `scatter_axis_grad` (`cpu/ops/reduce/helpers.rs`) exactly. Not used for
/// `max_keepdim`/`min_keepdim` - see their doc comments.
pub(crate) fn push_extremum_dim_tape_entry(
    t: &WgpuStorage,
    out: &WgpuStorage,
    dim: usize,
    is_max: bool,
) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
            let (outer, axis, inner) = axis_reduce_dims(&input_shape, dim)?;
            let input_data = t_capture.buffer.to_vec::<f32>()?;
            let grad_data = grad_out.buffer.to_vec::<f32>()?;
            let grad_elements = ShapeBuf::from_slice(&[outer, axis, inner])
                .checked_numel(OperationKind::Reduction)?;
            let mut grad_input = vec![0.0f32; grad_elements];
            for o in 0..outer {
                for i in 0..inner {
                    let mut best_val = if is_max {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    let mut best_flat = o * (axis * inner) + i;
                    for a in 0..axis {
                        let flat = o * (axis * inner) + a * inner + i;
                        let v = input_data[flat];
                        if (is_max && v > best_val) || (!is_max && v < best_val) {
                            best_val = v;
                            best_flat = flat;
                        }
                    }
                    let flat_out = o * inner + i;
                    grad_input[best_flat] = grad_data[flat_out];
                }
            }
            Ok(vec![WgpuStorage::new(
                WgpuBuffer::from_slice(&grad_input),
                input_shape.clone(),
            )])
        }),
    });
}

/// Backward for `max_all`/`min_all`: the whole-tensor special case of
/// `push_extremum_dim_tape_entry` (`outer = inner = 1`, `axis = numel`).
pub(crate) fn push_extremum_all_tape_entry(t: &WgpuStorage, out: &WgpuStorage, is_max: bool) {
    let input_shape = t.shape.to_vec();
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
            let input_data = t_capture.buffer.to_vec::<f32>()?;
            let grad_values = grad_out.buffer.to_vec::<f32>()?;
            let grad_val = *grad_values.first().ok_or(Error::InternalInvariant {
                operation: "wgpu_extremum_backward",
                reason: "scalar extremum gradient read back no value",
            })?;
            let mut best_val = if is_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            let mut best_flat = 0usize;
            for (flat, &v) in input_data.iter().enumerate() {
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_flat = flat;
                }
            }
            let mut grad_input = vec![0.0f32; input_data.len()];
            grad_input[best_flat] = grad_val;
            Ok(vec![WgpuStorage::new(
                WgpuBuffer::from_slice(&grad_input),
                input_shape.clone(),
            )])
        }),
    });
}

impl<D: Device> WgpuBackendImpl<D> {
    /// `prod_all`. Not autograd-wired, matching CPU.
    pub(crate) fn prod_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 3)?;
        // This row is advertised with `training = true`, so the reduction
        // records its backward: the zero-aware product rule CPU proves in
        // `cpu::ops::reduce::helpers::prod_all_grad`.
        let t_capture = t.clone();
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = t_capture.buffer.to_vec::<f32>()?;
                let scalar = grad_out
                    .buffer
                    .to_vec::<f32>()?
                    .into_iter()
                    .next()
                    .unwrap_or(0.0);
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&prod_all_backward(&input_data, scalar)),
                    original_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }
    /// `prod_dim`, with the per-slice zero-aware product rule CPU proves in
    /// `cpu::ops::reduce::helpers::prod_dim_grad`.
    pub(crate) fn prod_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 3, false)?;
        let t_capture = t.clone();
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let input_data = t_capture.buffer.to_vec::<f32>()?;
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&prod_dim_backward(
                        &input_shape,
                        dim,
                        &input_data,
                        &grad_data,
                    )),
                    input_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }

    /// `sum_all`.
    pub(crate) fn sum_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 0)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_all`.
    pub(crate) fn mean_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_all_to_storage(t, 0)?;
        let n = num_elements(&t.shape)? as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let scaled = scalar_op::<K>(grad_out, 1.0 / n, 1)?;
                Ok(vec![Self::broadcast_as::<K>(&scaled, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `max_all`.
    pub(crate) fn max_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 1)?;
        push_extremum_all_tape_entry(t, &out, true);
        Ok(out)
    }
    /// `min_all`.
    pub(crate) fn min_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_all_to_storage(t, 2)?;
        push_extremum_all_tape_entry(t, &out, false);
        Ok(out)
    }

    /// `sum_dim`.
    pub(crate) fn sum_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, false)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.to_vec();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape)?;
                Ok(vec![Self::broadcast_as::<K>(&keepdim, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `sum_keepdim`.
    pub(crate) fn sum_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 0, true)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::broadcast_as::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }
    /// `mean_dim`.
    pub(crate) fn mean_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, false)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut keepdim_shape = grad_out.shape.to_vec();
                keepdim_shape.insert(dim, 1);
                let keepdim = Self::reshape::<K>(grad_out, &keepdim_shape)?;
                let expanded = Self::broadcast_as::<K>(&keepdim, &original_shape)?;
                Ok(vec![scalar_op::<K>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }
    /// `mean_keepdim`.
    pub(crate) fn mean_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let sum = reduce_dim_to_storage(t, dim, 0, true)?;
        let n = t.shape[dim] as f64;
        let out = scalar_op::<K>(&sum, 1.0 / n, 1)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let expanded = Self::broadcast_as::<K>(grad_out, &original_shape)?;
                Ok(vec![scalar_op::<K>(&expanded, 1.0 / n, 1)?])
            }),
        });
        Ok(out)
    }

    /// Bessel (or population) scale for a variance over `count` samples:
    /// `1 / (count - 1)` when `unbiased` and `count > 1`, else `1 / count`,
    /// and `0` when the divisor would be non-positive. Same table CPU's
    /// `variance_scale` uses.
    fn variance_scale(count: usize, unbiased: bool) -> f64 {
        let count = count as f64;
        let divisor = if unbiased {
            if count <= 1.0 { 0.0 } else { count - 1.0 }
        } else {
            count
        };
        if divisor > 0.0 { 1.0 / divisor } else { 0.0 }
    }

    /// Scale a pre-reduced squared-deviation sum by
    /// [`variance_scale`](Self::variance_scale), optionally square-rooting.
    /// The mean/center/square/reduce steps live in each wrapper so all six
    /// stay readable against CPU's `variance_executors!` line by line.
    fn variance_scale_sum<K: DType>(
        summed: &<Self as StorageBackend>::Storage<K>,
        count: usize,
        unbiased: bool,
        square_root: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let scaled = Self::mul_scalar_float::<K>(summed, Self::variance_scale(count, unbiased))?;
        if square_root {
            Self::sqrt::<K>(&scaled)
        } else {
            Ok(scaled)
        }
    }

    /// `var_all`: population/Bessel variance over every element.
    pub(crate) fn variance_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_all::<K>(t)?;
        let count = num_elements(&t.shape)?;
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_all::<K>(&squared)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_all`: [`variance_all`](Self::variance_all) then a square root.
    pub(crate) fn std_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_all::<K>(t)?;
        let count = num_elements(&t.shape)?;
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_all::<K>(&squared)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `var_dim`: variance along `dim`, dropping the axis.
    pub(crate) fn variance_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape.get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_dim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_dim`: [`variance_dim`](Self::variance_dim) then a square root.
    pub(crate) fn std_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape.get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_dim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `var_keepdim`: variance along `dim`, keeping a unit axis.
    pub(crate) fn variance_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape.get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_keepdim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_keepdim`: [`variance_keepdim`](Self::variance_keepdim) then a
    /// square root.
    pub(crate) fn std_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape.get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_keepdim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `norm(order)`: the p-norm over every element. Order 1 and 2 take the
    /// dedicated abs-sum / Euclidean paths CPU takes (same tolerance-free
    /// special cases); any other positive order is `sum(|x|^p)^(1/p)`.
    pub(crate) fn norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        order: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        const NORM_ORDER_TOLERANCE: f64 = 1e-6;
        if (order - 1.0).abs() < NORM_ORDER_TOLERANCE {
            let magnitude = Self::abs::<K>(t)?;
            return Self::sum_all::<K>(&magnitude);
        }
        if (order - 2.0).abs() < NORM_ORDER_TOLERANCE {
            let squared = Self::mul::<K>(t, t)?;
            let summed = Self::sum_all::<K>(&squared)?;
            return Self::sqrt::<K>(&summed);
        }
        let magnitude = Self::abs::<K>(t)?;
        let raised = Self::powf::<K>(&magnitude, order)?;
        let summed = Self::sum_all::<K>(&raised)?;
        Self::powf::<K>(&summed, 1.0 / order)
    }

    /// `max_dim`.
    pub(crate) fn max_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 1, false)?;
        push_extremum_dim_tape_entry(t, &out, dim, true);
        Ok(out)
    }
    /// `max_keepdim`.
    ///
    /// Autograd-wired the same way as `max_dim`. `log_softmax` calls this to
    /// subtract a per-row max `M` for numerical stability:
    /// `log_softmax(x) = (x - M) - log(sum(exp(x - M)))`. This is not merely
    /// numerically close but ALGEBRAICALLY IDENTICAL to `x - log(sum(exp(x)))`
    /// for any `M` (the `M` terms cancel exactly:
    /// `-M - log(exp(-M)*sum(exp(x))) = -M - (-M + log(sum(exp(x)))) =
    /// -log(sum(exp(x)))`), so the two expressions are literally the same
    /// function of `x` on their whole domain, not just equal at one point --
    /// their gradients must therefore be identical too, whether or not `M`
    /// is treated as differentiable. Wiring a real gradient here does NOT
    /// need `log_softmax` to detach `M`. Matches the CPU backend, whose
    /// `max_keepdim` is fully wired and whose composed `log_softmax` (same
    /// formula) passes `softmax_gradcheck`/`log_softmax_gradcheck`.
    pub(crate) fn max_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 1, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, true);
        Ok(out)
    }
    /// `min_dim`.
    pub(crate) fn min_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, false)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
    }
    /// `min_keepdim`.
    ///
    /// Autograd-wired the same way as `min_dim` -- see `max_keepdim`'s doc
    /// comment above.
    pub(crate) fn min_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = reduce_dim_to_storage(t, dim, 2, true)?;
        push_extremum_dim_tape_entry(t, &out, dim, false);
        Ok(out)
    }

    /// `logsumexp(x, dim) = max + log(sum_keepdim(exp(x - max), dim))`.
    ///
    /// CPU's recipe (`cpu::ops::reduce::logsumexp`): shift by the axis
    /// maximum so every exponential lands in `(0, 1]`, sum, log, shift back.
    /// Each step is already a taped primitive, so this writes no backward of
    /// its own — the composite's gradient is the replay of those entries,
    /// which is what makes `training = true` on the capability row honest.
    pub(crate) fn logsumexp_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "logsumexp_keepdim",
                expected: t.shape.to_vec(),
                got: alloc::vec![dim],
                msg: alloc::format!(
                    "logsumexp_keepdim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let max = Self::max_keepdim::<K>(t, dim)?;
        let shifted = Self::sub::<K>(t, &max)?;
        let exp_shifted = Self::exp::<K>(&shifted)?;
        let sum_exp = Self::sum_keepdim::<K>(&exp_shifted, dim)?;
        let log_sum = Self::log::<K>(&sum_exp)?;
        Self::add::<K>(&max, &log_sum)
    }

    /// [`logsumexp_keepdim`](Self::logsumexp_keepdim) with the reduced axis
    /// removed, through `squeeze` so the chain keeps its tape continuity.
    pub(crate) fn logsumexp_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "logsumexp_dim",
                expected: t.shape.to_vec(),
                got: alloc::vec![dim],
                msg: alloc::format!(
                    "logsumexp_dim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        Self::squeeze::<K>(&Self::logsumexp_keepdim::<K>(t, dim)?, dim)
    }

    /// `cumsum(x, dim)`: host-side prefix scan along `dim`, mirrored from
    /// `cpu::ops::reduce::cumsum`. The Jacobian is lower-triangular ones, so
    /// the backward is the suffix sum (`reverse_cumsum`) of the cotangent —
    /// every input position receives `sum_{k >= d} grad_out[k]`.
    pub(crate) fn cumsum<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "cumsum",
                expected: t.shape.to_vec(),
                got: alloc::vec![dim],
                msg: alloc::format!("cumsum: axis {dim} out of range for shape {:?}", t.shape),
            });
        }
        let shape = t.shape.to_vec();
        let total = num_elements(&shape)?;
        let dim_len = shape[dim];
        let strides = contiguous_strides(&shape);
        let data = t.buffer.to_vec::<f32>()?;
        let mut out_data = alloc::vec![0.0f32; total];
        let mut idx = alloc::vec![0usize; shape.len()];
        for _ in 0..total {
            if idx[dim] == 0 {
                let mut current = 0.0f32;
                for step in 0..dim_len {
                    let mut step_idx = idx.clone();
                    step_idx[dim] = step;
                    let flat: usize = step_idx
                        .iter()
                        .zip(strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    current += data[flat];
                    out_data[flat] = current;
                }
            }
            increment_index(&mut idx, &shape);
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_data), shape.clone());

        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let mut rev = alloc::vec![0.0f32; total];
                let mut idx = alloc::vec![0usize; shape.len()];
                for _ in 0..total {
                    if idx[dim] == 0 {
                        let mut current = 0.0f32;
                        for step in (0..dim_len).rev() {
                            let mut step_idx = idx.clone();
                            step_idx[dim] = step;
                            let flat: usize = step_idx
                                .iter()
                                .zip(strides.iter())
                                .map(|(&i, &s)| i * s)
                                .sum();
                            current += grad_data[flat];
                            rev[flat] = current;
                        }
                    }
                    increment_index(&mut idx, &shape);
                }
                Ok(alloc::vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&rev),
                    shape.clone()
                )])
            }),
        });
        Ok(out)
    }

    /// `argmax(t, dim)`: index of the maximum along `dim` (axis removed, as
    /// `max_dim` squeezes) or, for `None`, the flat index of the global
    /// maximum as a scalar — CPU's `argmax` walk for walk, including the
    /// strict `>` that hands ties to the first winner. Forward-only (no tape,
    /// matching CPU's deliberate non-push and
    /// `descriptor_training(ArgMax) = false`); indices honor `KInt`.
    pub(crate) fn argmax<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::extremum_index::<KInt>(t, dim, true, "argmax")
    }

    /// `argmin(t, dim)`: mirror of [`argmax`](Self::argmax) with strict `<`.
    pub(crate) fn argmin<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::extremum_index::<KInt>(t, dim, false, "argmin")
    }

    /// Shared `argmax`/`argmin` walk: one pass over the downloaded `f32`
    /// values per output position, first-encountered winner on ties.
    fn extremum_index<KInt: DType>(
        t: &WgpuStorage,
        dim: Option<usize>,
        is_max: bool,
        op: &'static str,
    ) -> Result<WgpuStorage> {
        require_f32(t, op)?;
        let shape = t.shape.to_vec();
        let data = t.buffer.to_vec::<f32>()?;
        match dim {
            Some(d) => {
                if d >= shape.len() {
                    return Err(Error::ShapeMismatch {
                        op,
                        expected: shape.clone(),
                        got: alloc::vec![d],
                        msg: alloc::format!(
                            "{op}: axis {d} out of range for shape {shape:?}"
                        ),
                    });
                }
                let mut base_shape = shape.clone();
                base_shape[d] = 1;
                let n_slices = num_elements(&base_shape)?;
                let strides = contiguous_strides(&shape);
                let mut winners = alloc::vec![0i64; n_slices];
                for (i, slot) in winners.iter_mut().enumerate() {
                    let mut rem = i;
                    let mut coords = alloc::vec![0usize; shape.len()];
                    for dd in (0..shape.len()).rev() {
                        coords[dd] = rem % base_shape[dd];
                        rem /= base_shape[dd];
                    }
                    let mut best_val = if is_max {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    let mut best_axis = 0usize;
                    for j in 0..shape[d] {
                        coords[d] = j;
                        let flat: usize = coords
                            .iter()
                            .zip(strides.iter())
                            .map(|(&c, &s)| c * s)
                            .sum();
                        if (is_max && data[flat] > best_val)
                            || (!is_max && data[flat] < best_val)
                        {
                            best_val = data[flat];
                            best_axis = j;
                        }
                    }
                    *slot = best_axis as i64;
                }
                // Removing a size-1 axis preserves flat order, so the slice
                // ordinal layout above is already the squeezed layout — the
                // same bytes Metal's keepdim-plus-reshape produces.
                let mut out_shape = shape.clone();
                out_shape.remove(d);
                index_storage::<KInt>(op, &winners, &out_shape)
            }
            None => {
                let mut best_val = if is_max {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                let mut best_flat = 0i64;
                for (flat, &v) in data.iter().enumerate() {
                    if (is_max && v > best_val) || (!is_max && v < best_val) {
                        best_val = v;
                        best_flat = flat as i64;
                    }
                }
                index_storage::<KInt>(op, &[best_flat], &[])
            }
        }
    }

    /// `argsort(t, dim, descending)`: the permutation that sorts each
    /// axis-slice, with the operand's own geometry — the index half of
    /// [`sort`](Self::sort). Forward-only; indices honor `KInt`.
    pub(crate) fn argsort<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        require_f32(t, "argsort")?;
        let dim_len = *t.shape.get(dim).ok_or(Error::ShapeMismatch {
            op: "argsort",
            expected: t.shape.to_vec(),
            got: alloc::vec![dim],
            msg: alloc::format!("argsort: axis {dim} out of range for shape {:?}", t.shape),
        })?;
        let (_vals, idxs, out_shape) = sort_pairs(t, dim, descending, Some(dim_len))?;
        index_storage::<KInt>("argsort", &idxs, &out_shape)
    }

    /// `sort(t, dim, descending)`: sorted values beside the permutation that
    /// produced them, both with the operand's own geometry — the pair CUDA's
    /// `Sort` returns via full-`k` `topk`. Forward-only
    /// (`descriptor_training(Sort) = false`); values are `f32`, indices honor
    /// `KInt`.
    pub(crate) fn sort<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<K>,
    )> {
        require_f32(t, "sort")?;
        let dim_len = *t.shape.get(dim).ok_or(Error::ShapeMismatch {
            op: "sort",
            expected: t.shape.to_vec(),
            got: alloc::vec![dim],
            msg: alloc::format!("sort: axis {dim} out of range for shape {:?}", t.shape),
        })?;
        let (vals, idxs, out_shape) = sort_pairs(t, dim, descending, Some(dim_len))?;
        let values = WgpuStorage::new(WgpuBuffer::from_slice(&vals), out_shape.clone());
        let indices = index_storage::<KInt>("sort", &idxs, &out_shape)?;
        Ok((values, indices))
    }

    /// `topk(t, k, dim, largest)`: the `k` extreme values along `dim` beside
    /// their indices; the axis shrinks to `k`. `k` clamps to the axis length
    /// the way CPU's does (the descriptor already refuses `k` past it, so
    /// the clamp is belt-and-braces inside [`sort_pairs`]). Forward-only;
    /// values are `f32`, indices honor `KInt`.
    pub(crate) fn topk<K: DType, KInt: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<K>,
    )> {
        require_f32(t, "topk")?;
        if k == 0 {
            return Err(Error::Msg("topk: k must be at least one".into()));
        }
        let (vals, idxs, out_shape) = sort_pairs(t, dim, largest, Some(k))?;
        let values = WgpuStorage::new(WgpuBuffer::from_slice(&vals), out_shape.clone());
        let indices = index_storage::<KInt>("topk", &idxs, &out_shape)?;
        Ok((values, indices))
    }
}

/// Index-output storage for the order-statistic family below, in the exact
/// width the caller asked for (`u8`/`u32`/`i64`, the set CPU's `index_buffer`
/// and the descriptor's `validate_index_dtype` both admit).
///
/// Indices are non-negative and bounded by the axis they index, so a narrower
/// width is a checked fit rather than a truncation: a silently wrapped index
/// is refused with `UnsupportedDType`, the same fail-closed rule CPU uses.
/// Uploads go through `try_from_slice` rather than `from_slice` (#91): an
/// `i64` index buffer is twice the input's bytes for the same element count,
/// so an input that fit the adapter's `max_storage_buffer_binding_size` does
/// not imply its index buffer does, and the refusal must name the limit
/// instead of panicking inside an unrelated upload.
fn index_storage<KInt: DType>(
    op: &'static str,
    indices: &[i64],
    shape: &[usize],
) -> Result<WgpuStorage> {
    fn narrow<T>(op: &'static str, indices: &[i64], dtype: DTypeDescriptor) -> Result<Vec<T>>
    where
        T: TryFrom<i64>,
    {
        indices
            .iter()
            .map(|&index| {
                T::try_from(index).map_err(|_| Error::UnsupportedDType {
                    dtype,
                    backend: "Wgpu",
                    op,
                })
            })
            .collect()
    }

    let dtype = KInt::descriptor(&Default::default());
    let bytes: Vec<u8> = match dtype.builtin_id() {
        Some(DTypeId::I64) => indices.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Some(DTypeId::U32) => narrow::<u32>(op, indices, dtype)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        Some(DTypeId::U8) => narrow::<u8>(op, indices, dtype)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                backend: "Wgpu",
                op,
            });
        }
    };
    let buffer = WgpuBuffer::try_from_slice(bytes.as_slice())?;
    WgpuStorage::try_new_with_dtype(buffer, shape.to_vec(), dtype)
}

/// Shared axis-slice sort for `sort`, `topk` and `argsort`: the `base_shape`
/// walk CPU's `topk` uses (the operand shape with `dim` set to 1,
/// unflattened), stable-sorting each slice's `(value, index)` pairs so ties
/// keep arrival order (`sort_by`, matching CPU's comparator on equal keys
/// exactly), optionally truncating to `k`. Returns the flat value/index
/// vectors plus the output shape. `f32` only — callers run `require_f32`
/// first, so no `f64` accumulation ever appears (#91).
fn sort_pairs(
    t: &WgpuStorage,
    dim: usize,
    descending: bool,
    k: Option<usize>,
) -> Result<(Vec<f32>, Vec<i64>, Vec<usize>)> {
    let shape = t.shape.to_vec();
    if dim >= shape.len() {
        return Err(Error::ShapeMismatch {
            op: "sort_pairs",
            expected: shape.clone(),
            got: alloc::vec![dim],
            msg: alloc::format!("axis {dim} out of range for shape {shape:?}"),
        });
    }
    let mut base_shape = shape.clone();
    base_shape[dim] = 1;
    let n_slices = num_elements(&base_shape)?;
    let k = k.map(|k| k.min(shape[dim]));
    let mut out_shape = shape.clone();
    if let Some(k) = k {
        out_shape[dim] = k;
    }
    let out_len = num_elements(&out_shape)?;
    let mut out_vals = alloc::vec![0.0f32; out_len];
    let mut out_indices = alloc::vec![0i64; out_len];

    let data = t.buffer.to_vec::<f32>()?;
    let strides = contiguous_strides(&shape);
    let out_strides = contiguous_strides(&out_shape);

    for i in 0..n_slices {
        let mut rem = i;
        let mut coords = alloc::vec![0usize; shape.len()];
        for dd in (0..shape.len()).rev() {
            coords[dd] = rem % base_shape[dd];
            rem /= base_shape[dd];
        }
        let mut slice: Vec<(f32, i64)> = Vec::with_capacity(shape[dim]);
        for j in 0..shape[dim] {
            coords[dim] = j;
            let flat: usize = coords
                .iter()
                .zip(strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();
            slice.push((data[flat], j as i64));
        }
        if descending {
            slice.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            slice.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }
        let take = k.unwrap_or(shape[dim]);
        let mut out_coords = coords.clone();
        for (j, &(val, idx)) in slice.iter().enumerate().take(take) {
            out_coords[dim] = j;
            let flat: usize = out_coords
                .iter()
                .zip(out_strides.iter())
                .map(|(&c, &s)| c * s)
                .sum();
            out_vals[flat] = val;
            out_indices[flat] = idx;
        }
    }
    Ok((out_vals, out_indices, out_shape))
}

/// Row-major contiguous strides for `shape`.
fn contiguous_strides(shape: &[usize]) -> alloc::vec::Vec<usize> {
    let rank = shape.len();
    let mut strides = alloc::vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Odometer increment over `shape`, wrapping like CPU's `increment_index`.
fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}
