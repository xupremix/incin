//! Shape and layout operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::convert::{from_candle_dtype, to_candle_dtype};
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

pub fn candle_readback_error(error: candle_core::Error) -> Error {
    BackendError::Execution {
        operation: OperationKind::Storage,
        message: alloc::format!("{error}").into(),
    }
    .into()
}

impl<D: incin_core::prelude::Device> CandleBackend<D> {
    // Candle has native equivalents for most of these, but this adapter does
    // not route them yet. Declaring the gap here keeps it visible instead of
    // leaving it to a trait default that reads as full coverage.
    crate::unsupported::unsupported_tensor_ops! {
        where_cond, gather, scatter, index_select, masked_fill, unsqueeze,
        repeat, pad, triu, tril, diag,
        cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge,
        logical_and, logical_or, logical_not,
        sub_scalar, div_scalar, maximum, minimum, abs_diff, lerp,
        addmm, bmm, scaled_dot_product_attention,
        unfold, pixel_shuffle, group_norm, instance_norm,
    }

    /// Matrix-multiplies `lhs` and `rhs`. For operands with more than 3
    /// dimensions (which candle's `broadcast_matmul` can't handle directly),
    /// manually broadcasts the leading batch dimensions, flattens them into
    /// a single batch axis, multiplies, and reshapes back.
    pub fn matmul<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let lhs_contig = lhs
            .tensor()
            .contiguous()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let rhs_contig = rhs
            .tensor()
            .contiguous()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

        let l_shape = lhs_contig.dims();
        let r_shape = rhs_contig.dims();

        if l_shape.len() > 3 || r_shape.len() > 3 {
            let max_len = core::cmp::max(l_shape.len(), r_shape.len());
            let mut out_shape = vec![];

            for i in 0..max_len - 2 {
                let l = if i < max_len - l_shape.len() {
                    1
                } else {
                    l_shape[i - (max_len - l_shape.len())]
                };
                let r = if i < max_len - r_shape.len() {
                    1
                } else {
                    r_shape[i - (max_len - r_shape.len())]
                };
                out_shape.push(core::cmp::max(l, r));
            }

            let m = l_shape[l_shape.len() - 2];
            let k = l_shape[l_shape.len() - 1];
            let n = r_shape[r_shape.len() - 1];

            let mut lhs_b_shape = out_shape.clone();
            lhs_b_shape.push(m);
            lhs_b_shape.push(k);

            let mut rhs_b_shape = out_shape.clone();
            rhs_b_shape.push(k);
            rhs_b_shape.push(n);

            let lhs_b = lhs_contig
                .broadcast_as(lhs_b_shape.as_slice())
                .map_err(|e| anyhow::anyhow!(e))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!(e))?;
            let rhs_b = rhs_contig
                .broadcast_as(rhs_b_shape.as_slice())
                .map_err(|e| anyhow::anyhow!(e))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!(e))?;

            let batch_size: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
                .checked_numel(incin_core::prelude::OperationKind::Storage)?;
            let lhs_flat = lhs_b
                .reshape((batch_size, m, k))
                .map_err(|e| anyhow::anyhow!(e))?;
            let rhs_flat = rhs_b
                .reshape((batch_size, k, n))
                .map_err(|e| anyhow::anyhow!(e))?;

            let res_flat = lhs_flat.matmul(&rhs_flat).map_err(|e| anyhow::anyhow!(e))?;

            let mut res_shape = out_shape;
            res_shape.push(m);
            res_shape.push(n);
            let raw = res_flat
                .reshape(res_shape.as_slice())
                .map_err(|e| anyhow::anyhow!(e))?;
            return CandleStorage::try_new(raw);
        }

        let raw = lhs_contig
            .broadcast_matmul(&rhs_contig)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Stacks `tensors` along a new dimension `dim`.
    pub fn stack<K: incin_core::prelude::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw_tensors: Vec<&candle_core::Tensor> = tensors.iter().map(|s| s.tensor()).collect();
        let raw = candle_core::Tensor::stack(&raw_tensors, dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Concatenates `tensors` along an existing dimension `dim`.
    pub fn concat<K: incin_core::prelude::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw_tensors: Vec<&candle_core::Tensor> = tensors.iter().map(|s| s.tensor()).collect();
        let raw = candle_core::Tensor::cat(&raw_tensors, dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Broadcasts `t` to `shape`.
    pub fn broadcast_as<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .broadcast_as(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }
    /// Broadcasts `t` by prepending dimensions from `shape` on the left.
    pub fn broadcast_left<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .broadcast_left(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Reshapes `t` to `shape` without changing its underlying data.
    pub fn reshape<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .reshape(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }
    /// Swaps dimensions `dim1` and `dim2`.
    pub fn transpose<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .transpose(dim1, dim2)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }
    /// Applies a per-dimension `[start, end)` narrow for each entry in
    /// `ranges`, sequentially, one dimension at a time.
    pub fn slice<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut out = t.tensor().clone();
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            out = out
                .narrow(dim, start, end - start)
                .map_err(|e| Error::Msg(format!("Candle narrow failed for slice: {}", e)))?;
        }
        CandleStorage::try_new(out)
    }

    /// Flattens dimensions `start_dim..=end_dim` into a single dimension.
    pub fn flatten<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .flatten(start_dim, end_dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Takes a contiguous sub-range of length `len` starting at `start`
    /// along `dim`.
    pub fn narrow<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .narrow(dim, start, len)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Removes dimension `dim` if it has size 1.
    pub fn squeeze<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let raw = t
            .tensor()
            .squeeze(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }

    /// Casts `t` to `f32` and extracts its single element as an `f64`
    /// scalar.
    pub fn float_to_scalar<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<f64> {
        let v = t
            .tensor()
            .to_dtype(candle_core::DType::F32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let s: f32 = v
            .to_scalar()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(s as f64)
    }
    /// Casts `t` to `f32` and collects it into a flat `Vec<f64>`.
    pub fn float_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Vec<f64>> {
        let v = t
            .tensor()
            .to_dtype(candle_core::DType::F32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let vec: Vec<f32> = v
            .to_vec1()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(vec.into_iter().map(|x| x as f64).collect())
    }
    /// Extracts one integer without implicit float truncation or saturation.
    pub fn int_to_scalar<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<i64> {
        let operation = "candle_int_to_scalar";
        let c_tensor = t.tensor();
        let value = match c_tensor.dtype() {
            candle_core::DType::U8 => {
                return Ok(i64::from(
                    c_tensor.to_scalar::<u8>().map_err(candle_readback_error)?,
                ));
            }
            candle_core::DType::U32 => {
                return Ok(i64::from(
                    c_tensor.to_scalar::<u32>().map_err(candle_readback_error)?,
                ));
            }
            candle_core::DType::I64 => {
                return c_tensor.to_scalar::<i64>().map_err(candle_readback_error);
            }
            candle_core::DType::BF16 => c_tensor
                .to_scalar::<half::bf16>()
                .map_err(candle_readback_error)?
                .to_f64(),
            candle_core::DType::F16 => c_tensor
                .to_scalar::<half::f16>()
                .map_err(candle_readback_error)?
                .to_f64(),
            candle_core::DType::F32 => {
                f64::from(c_tensor.to_scalar::<f32>().map_err(candle_readback_error)?)
            }
            candle_core::DType::F64 => {
                c_tensor.to_scalar::<f64>().map_err(candle_readback_error)?
            }
        };
        incin_core::prelude::convert_f64_to_i64(
            operation,
            from_candle_dtype(c_tensor.dtype()),
            value,
            incin_core::prelude::FloatToIntPolicy::Exact,
        )
    }
    /// Collects integers without implicit float truncation or saturation.
    pub fn int_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Vec<i64>> {
        let operation = "candle_int_to_vec1";
        let c_tensor = t.tensor();
        let dtype = from_candle_dtype(c_tensor.dtype());
        let values = match c_tensor.dtype() {
            candle_core::DType::U8 => {
                return Ok(c_tensor
                    .to_vec1::<u8>()
                    .map_err(candle_readback_error)?
                    .into_iter()
                    .map(i64::from)
                    .collect());
            }
            candle_core::DType::U32 => {
                return Ok(c_tensor
                    .to_vec1::<u32>()
                    .map_err(candle_readback_error)?
                    .into_iter()
                    .map(i64::from)
                    .collect());
            }
            candle_core::DType::I64 => {
                return c_tensor.to_vec1::<i64>().map_err(candle_readback_error);
            }
            candle_core::DType::BF16 => c_tensor
                .to_vec1::<half::bf16>()
                .map_err(candle_readback_error)?
                .into_iter()
                .map(|value| value.to_f64())
                .collect(),
            candle_core::DType::F16 => c_tensor
                .to_vec1::<half::f16>()
                .map_err(candle_readback_error)?
                .into_iter()
                .map(|value| value.to_f64())
                .collect(),
            candle_core::DType::F32 => c_tensor
                .to_vec1::<f32>()
                .map_err(candle_readback_error)?
                .into_iter()
                .map(f64::from)
                .collect(),
            candle_core::DType::F64 => c_tensor.to_vec1::<f64>().map_err(candle_readback_error)?,
        };
        values
            .into_iter()
            .map(|value| {
                incin_core::prelude::convert_f64_to_i64(
                    operation,
                    dtype,
                    value,
                    incin_core::prelude::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
    /// Casts `t` to the candle dtype corresponding to `dtype`.
    pub fn tensor_to_dtype<K: incin_core::prelude::DType, K2: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K2>> {
        let raw = t
            .tensor()
            .to_dtype(to_candle_dtype(dtype)?)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(raw)
    }
}
