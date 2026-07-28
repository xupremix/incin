//! Shape and layout operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::convert::to_candle_dtype;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::prelude::TensorOps<Self> for CandleBackend<T, D>
{
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
    fn matmul<K: incin_core::prelude::DType>(
        lhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
        rhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let lhs_contig = lhs
            .contiguous()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let rhs_contig = rhs
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

            let batch_size: usize = out_shape.iter().product();
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
            return Ok(res_flat
                .reshape(res_shape.as_slice())
                .map_err(|e| anyhow::anyhow!(e))?);
        }

        Ok(lhs_contig
            .broadcast_matmul(&rhs_contig)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Stacks `tensors` along a new dimension `dim`.
    fn stack<K: incin_core::prelude::DType>(
        tensors: &[&<Self as incin_core::prelude::Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(candle_core::Tensor::stack(tensors, dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Concatenates `tensors` along an existing dimension `dim`.
    fn concat<K: incin_core::prelude::DType>(
        tensors: &[&<Self as incin_core::prelude::Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(candle_core::Tensor::cat(tensors, dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Broadcasts `t` to `shape`.
    fn broadcast_as<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.broadcast_as(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Broadcasts `t` by prepending dimensions from `shape` on the left.
    fn broadcast_left<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.broadcast_left(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Reshapes `t` to `shape` without changing its underlying data.
    fn reshape<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.reshape(shape)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Swaps dimensions `dim1` and `dim2`.
    fn transpose<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.transpose(dim1, dim2)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies a per-dimension `[start, end)` narrow for each entry in
    /// `ranges`, sequentially, one dimension at a time.
    fn slice<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let mut out = t.clone();
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            out = out
                .narrow(dim, start, end - start)
                .map_err(|e| Error::Msg(format!("Candle narrow failed for slice: {}", e)))?;
        }
        Ok(out)
    }

    /// Flattens dimensions `start_dim..=end_dim` into a single dimension.
    fn flatten<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.flatten(start_dim, end_dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Takes a contiguous sub-range of length `len` starting at `start`
    /// along `dim`.
    fn narrow<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.narrow(dim, start, len)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Removes dimension `dim` if it has size 1.
    fn squeeze<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.squeeze(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Casts `t` to `f32` and extracts its single element as an `f64`
    /// scalar.
    fn float_to_scalar<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<f64> {
        let v = t
            .to_dtype(candle_core::DType::F32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let s: f32 = v
            .to_scalar()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(s as f64)
    }
    /// Casts `t` to `f32` and collects it into a flat `Vec<f64>`.
    fn float_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<Vec<f64>> {
        let v = t
            .to_dtype(candle_core::DType::F32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let vec: Vec<f32> = v
            .to_vec1()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(vec.into_iter().map(|x| x as f64).collect())
    }
    /// Casts `t` to `i64` and extracts its single element.
    fn int_to_scalar<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<i64> {
        let v = t
            .to_dtype(candle_core::DType::I64)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let s: i64 = v
            .to_scalar()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(s)
    }
    /// Casts `t` to `i64` and collects it into a flat `Vec<i64>`.
    fn int_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<Vec<i64>> {
        let v = t
            .to_dtype(candle_core::DType::I64)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let vec: Vec<i64> = v
            .to_vec1()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(vec)
    }
    /// Casts `t` to the candle dtype corresponding to `dtype`.
    fn tensor_to_dtype<K: incin_core::prelude::DType, K2: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dtype: DTypeId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K2>> {
        Ok(t.to_dtype(to_candle_dtype(dtype)?)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
}
