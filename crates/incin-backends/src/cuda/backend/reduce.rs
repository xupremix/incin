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
            let grad_scaled = crate::cuda::backend::cuda_mul_storage(&unb_grad, &unb_prod)?;
            crate::cuda::backend::cuda_div_storage(&grad_scaled, &t_capture)
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
            let grad_scaled = crate::cuda::backend::cuda_mul_storage(&unb_grad, &unb_prod)?;
            crate::cuda::backend::cuda_div_storage(&grad_scaled, &t_capture)
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
        crate::cuda::ops::reduce::launch_welford_var_std(t, None, false, unbiased, false)
    }

    pub(crate) fn var_dim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), false, unbiased, false)
    }

    pub(crate) fn var_keepdim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), true, unbiased, false)
    }

    pub(crate) fn std_all<K: DType>(t: &CudaStorage, unbiased: bool) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_welford_var_std(t, None, false, unbiased, true)
    }

    pub(crate) fn std_dim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), false, unbiased, true)
    }

    pub(crate) fn std_keepdim<K: DType>(
        t: &CudaStorage,
        dim: usize,
        unbiased: bool,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_welford_var_std(t, Some(dim), true, unbiased, true)
    }

    pub(crate) fn cumsum<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_cumsum_op(t, dim)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::ops::reduce::launch_reverse_cumsum_op(grad_out, dim)
        });
        Ok(out)
    }
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
