//! Reduction CUDA operations: full and per-dimension sum/mean/max/min,
//! with and without keeping the reduced dimension.

#![allow(dead_code)]

use super::*;

impl<D: Device> CudaBackendImpl<D> {
    pub(crate) fn sum_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::sum_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    pub(crate) fn mean_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let total = checked_numel(&t.shape)? as f64;
        let sum = Self::sum_all::<K>(t)?;
        if total > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / total)
        } else {
            Ok(sum)
        }
    }

    pub(crate) fn max_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::max_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    pub(crate) fn min_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::min_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    pub(crate) fn sum_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, false)?;
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::tape::unbroadcast(grad_out, &t_shape)
        });
        Ok(out)
    }

    pub(crate) fn sum_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, true)?;
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::tape::unbroadcast(grad_out, &t_shape)
        });
        Ok(out)
    }

    pub(crate) fn mean_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let axis_len = *t.shape.get(dim).ok_or(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        })? as f64;
        let sum = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, false)?;
        let out = if axis_len > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / axis_len)?
        } else {
            sum
        };
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            if axis_len > 0.0 {
                let expr = format!("x * ({:.8}f)", (1.0 / axis_len) as f32);
                crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &unb)
            } else {
                Ok(unb)
            }
        });
        Ok(out)
    }

    pub(crate) fn mean_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let axis_len = *t.shape.get(dim).ok_or(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        })? as f64;
        let sum = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, true)?;
        let out = if axis_len > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / axis_len)?
        } else {
            sum
        };
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            if axis_len > 0.0 {
                let expr = format!("x * ({:.8}f)", (1.0 / axis_len) as f32);
                crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &unb)
            } else {
                Ok(unb)
            }
        });
        Ok(out)
    }

    pub(crate) fn max_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("max", t, dim, false)
    }

    pub(crate) fn max_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("max", t, dim, true)
    }

    pub(crate) fn min_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("min", t, dim, false)
    }

    pub(crate) fn min_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("min", t, dim, true)
    }

    pub(crate) fn prod_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::prod_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    pub(crate) fn prod_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("prod", t, dim, false)?;
        let t_capture = t.clone();
        let out_capture = out.clone();
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb_grad = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            let unb_prod = crate::cuda::tape::unbroadcast(&out_capture, &t_shape)?;
            let grad_scaled = crate::cuda::backend::cuda_mul_storage(
                &unb_grad,
                &unb_prod,
                crate::kernel::KernelSpecialization::NONE,
            )?;
            crate::cuda::backend::cuda_div_storage(
                &grad_scaled,
                &t_capture,
                crate::kernel::KernelSpecialization::NONE,
            )
        });
        Ok(out)
    }

    pub(crate) fn prod_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("prod", t, dim, true)?;
        let t_capture = t.clone();
        let out_capture = out.clone();
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb_grad = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            let unb_prod = crate::cuda::tape::unbroadcast(&out_capture, &t_shape)?;
            let grad_scaled = crate::cuda::backend::cuda_mul_storage(
                &unb_grad,
                &unb_prod,
                crate::kernel::KernelSpecialization::NONE,
            )?;
            crate::cuda::backend::cuda_div_storage(
                &grad_scaled,
                &t_capture,
                crate::kernel::KernelSpecialization::NONE,
            )
        });
        Ok(out)
    }

    pub(crate) fn argmax<KInt: DType>(t: &CudaStorage, dim: Option<usize>) -> Result<CudaStorage> {
        let dtype_id = KInt::descriptor(&Default::default())
            .builtin_id()
            .unwrap_or(DTypeId::I64);
        crate::cuda::ops::reduce::launch_argmax_argmin_op("argmax", t, dim, dtype_id)
    }

    pub(crate) fn argmin<KInt: DType>(t: &CudaStorage, dim: Option<usize>) -> Result<CudaStorage> {
        let dtype_id = KInt::descriptor(&Default::default())
            .builtin_id()
            .unwrap_or(DTypeId::I64);
        crate::cuda::ops::reduce::launch_argmax_argmin_op("argmin", t, dim, dtype_id)
    }

    pub(crate) fn argmax_dim<KInt: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let dtype_id = KInt::descriptor(&Default::default())
            .builtin_id()
            .unwrap_or(DTypeId::I64);
        crate::cuda::ops::reduce::launch_argmax_argmin_op("argmax", t, Some(dim), dtype_id)
    }

    pub(crate) fn argmin_dim<KInt: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let dtype_id = KInt::descriptor(&Default::default())
            .builtin_id()
            .unwrap_or(DTypeId::I64);
        crate::cuda::ops::reduce::launch_argmax_argmin_op("argmin", t, Some(dim), dtype_id)
    }

    pub(crate) fn topk<KInt: DType>(
        t: &CudaStorage,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(CudaStorage, CudaStorage)> {
        let dtype_id = KInt::descriptor(&Default::default())
            .builtin_id()
            .unwrap_or(DTypeId::I64);
        crate::cuda::ops::reduce::launch_topk_op(t, k, dim, largest, dtype_id)
    }

    pub(crate) fn argsort<KInt: DType>(
        t: &CudaStorage,
        dim: usize,
        descending: bool,
    ) -> Result<CudaStorage> {
        let dim_len = *t.shape.get(dim).ok_or(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        })?;
        let (_vals, indices) = Self::topk::<KInt>(t, dim_len, dim, descending)?;
        Ok(indices)
    }

    pub(crate) fn var_all<K: DType>(t: &CudaStorage, unbiased: bool) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::reduce::launch_welford_var_std(t, None, false, unbiased, false)?;
        push_var_std_tape_entry(t, &out, None, false, unbiased, false);
        Ok(out)
    }

    pub(crate) fn var_dim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), false, unbiased, false)?;
        push_var_std_tape_entry(t, &out, Some(dim), false, unbiased, false);
        Ok(out)
    }

    pub(crate) fn var_keepdim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), true, unbiased, false)?;
        push_var_std_tape_entry(t, &out, Some(dim), true, unbiased, false);
        Ok(out)
    }

    pub(crate) fn std_all<K: DType>(t: &CudaStorage, unbiased: bool) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_welford_var_std(t, None, false, unbiased, true)?;
        push_var_std_tape_entry(t, &out, None, false, unbiased, true);
        Ok(out)
    }

    pub(crate) fn std_dim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), false, unbiased, true)?;
        push_var_std_tape_entry(t, &out, Some(dim), false, unbiased, true);
        Ok(out)
    }

    pub(crate) fn std_keepdim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), true, unbiased, true)?;
        push_var_std_tape_entry(t, &out, Some(dim), true, unbiased, true);
        Ok(out)
    }

    pub(crate) fn cumsum<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_cumsum_op(t, dim)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::ops::reduce::launch_reverse_cumsum_op(grad_out, dim)
        });
        Ok(out)
    }
}

/// `1 / (count - 1)` when `unbiased` and `count > 1`, `1 / count` otherwise,
/// and `0` when the divisor would be non-positive. Same table CPU's and
/// WGPU's `variance_scale` use, so the three backends agree on the Bessel
/// correction - including the `count <= 1` case, where the forward Welford
/// kernel also reports `0`.
fn variance_scale(count: usize, unbiased: bool) -> f64 {
    let count = count as f64;
    let divisor = if unbiased {
        if count <= 1.0 { 0.0 } else { count - 1.0 }
    } else {
        count
    };
    if divisor > 0.0 { 1.0 / divisor } else { 0.0 }
}

/// Records the backward recipe for one of the six Welford `var`/`std` rows.
///
/// The capability rows advertise `training = true`, but the forward is a
/// dedicated kernel rather than a composition of taped primitives, so the
/// tape entry is written here. The closure recomputes what the gradient
/// needs - the mean, and for `std` the forward standard deviation - through
/// raw launches that record nothing (the walk also runs under
/// `GradMode::Disabled`, so any incidental push would be refused anyway).
///
/// With `scale = variance_scale(count, unbiased)`:
/// - `d var / d x_i = 2 * scale * (x_i - mean)`, so the variance gradient is
///   `grad * centered * 2 * scale`;
/// - `std = sqrt(var)` chains to `d std / d x_i = scale * (x_i - mean) / std`,
///   the same product scaled by `scale` once and divided by a recomputed
///   keepdim `std` so the division broadcasts back to the input's shape.
///
/// The incoming gradient arrives in the forward's output shape. For the
/// non-`keepdim` axis rows that shape has the reduced axis missing, and a
/// right-aligned broadcast of e.g. `[2]` against `[2, 4]` is not defined
/// (`2 != 4`), so it is rewrapped to the `keepdim` form first - same numel,
/// same offset-zero assumption `tape::sum_dim_squeeze` makes for the grads
/// flowing through this walk. A scalar seed keeps a lower rank and skips the
/// rewrap; it broadcasts directly.
fn push_var_std_tape_entry(
    t: &CudaStorage,
    out: &CudaStorage,
    axis: Option<usize>,
    keepdim: bool,
    unbiased: bool,
    is_std: bool,
) {
    let t_capture = t.clone();
    let t_shape = t.shape.to_vec();
    // The forward launch above already validated the axis and the storage, so
    // the two products below cannot overflow a shape that reached here.
    let total_numel = t_shape.iter().product::<usize>();
    let count = match axis {
        Some(d) => t_shape[d],
        None => total_numel,
    };
    let scale = variance_scale(count, unbiased);
    push_unary_tape_entry(t.id, out.id, move |grad_out| {
        // Mean in keepdim (axis rows) or scalar (all-row) form. The axis rows
        // use a raw per-axis `mean`; the all-row mirrors `mean_all` exactly -
        // successive raw `sum`s to a scalar, then one `mul_scalar`.
        let mean = match axis {
            Some(d) => crate::cuda::ops::reduce::launch_reduce_op("mean", &t_capture, d, true)?,
            None => {
                let mut curr = t_capture.clone();
                for d in (0..t_shape.len()).rev() {
                    curr = crate::cuda::ops::reduce::launch_reduce_op("sum", &curr, d, false)?;
                }
                if total_numel > 0 {
                    let expr = format!("x * ({:.17})", 1.0 / total_numel as f64);
                    crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &curr)?
                } else {
                    curr
                }
            }
        };
        let centered = crate::cuda::ops::elementwise::launch_binary_op(
            "sub", "a - b", &t_capture, &mean, &t_shape,
        )?;
        let grad_in = match axis {
            Some(d) if !keepdim => {
                let mut keepdim_shape = t_shape.clone();
                keepdim_shape[d] = 1;
                let keepdim_numel: usize = keepdim_shape.iter().product();
                let grad_numel: usize = grad_out.shape.iter().product();
                if grad_out.shape != keepdim_shape && grad_numel == keepdim_numel {
                    CudaStorage::new(grad_out.buffer.clone(), keepdim_shape)
                } else {
                    grad_out.clone()
                }
            }
            _ => grad_out.clone(),
        };
        let product = crate::cuda::ops::elementwise::launch_binary_op(
            "mul", "a * b", &grad_in, &centered, &t_shape,
        )?;
        let factor = if is_std { scale } else { 2.0 * scale };
        let expr = format!("x * ({:.17})", factor);
        let scaled = crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &product)?;
        if is_std {
            // Recomputed keepdim-along-axis (or scalar) so it broadcasts
            // against `scaled` regardless of the forward's `keepdim`.
            let std_bt = crate::cuda::ops::reduce::launch_welford_var_std(
                &t_capture,
                axis,
                axis.is_some(),
                unbiased,
                true,
            )?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "div", "a / b", &scaled, &std_bt, &t_shape,
            )
        } else {
            Ok(scaled)
        }
    });
}

pub(crate) fn cuda_sum_all_storage(t: &CudaStorage) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::sum_all::<f32>(t)
}

pub(crate) fn cuda_mean_all_storage(t: &CudaStorage) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::mean_all::<f32>(t)
}

pub(crate) fn cuda_sum_dim_keepdim(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::sum_keepdim::<f32>(t, dim)
}

pub(crate) fn cuda_mean_dim_keepdim(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::mean_keepdim::<f32>(t, dim)
}
