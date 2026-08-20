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
    );

    let final_shape = if keepdim {
        out_shape
    } else {
        let mut s = shape.to_vec();
        s.remove(dim);
        s
    };
    Ok(WgpuStorage::new(out_buf, final_shape))
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
        reduce_all_to_storage(t, 3)
    }
    /// `prod_dim`. Not autograd-wired, matching CPU.
    pub(crate) fn prod_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        reduce_dim_to_storage(t, dim, 3, false)
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
}
