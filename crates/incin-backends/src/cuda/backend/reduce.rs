//! Reduction CUDA operations: full and per-dimension sum/mean/max/min,
//! with and without keeping the reduced dimension.

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
}
