//! Neural-network module operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::prelude::ModuleOps<Self> for CandleBackend<T, D>
{
    /// Candle has no native adaptive pooling; returns an error
    /// instead of panicking so callers can handle the unsupported case.
    fn adaptive_avg_pool2d<K: incin_core::prelude::DType>(
        _t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "adaptive_avg_pool2d",
            backend: "CandleBackend",
        })
    }

    /// Applies layer normalization over the last dimension, substituting a
    /// zero bias when none is provided.
    fn layer_norm<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        weight: &<Self as incin_core::prelude::Backend>::Storage<K>,
        bias: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let zero_bias;
        let bias = match bias {
            Some(b) => b,
            None => {
                zero_bias = weight
                    .zeros_like()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                &zero_bias
            }
        };
        Ok(candle_nn::ops::layer_norm(t, weight, bias, eps)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Applies batch normalization using the running mean/variance (or
    /// defaults of 0/1 when not provided) and an optional affine
    /// weight/bias, reshaping all of them to broadcast over the channel
    /// dimension.
    fn batch_norm<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        weight: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        bias: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        running_mean: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        running_var: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let channel_dim = if t.rank() > 1 { 1 } else { 0 };
        let num_channels = t
            .dim(channel_dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

        let mut shape = vec![1; t.rank()];
        shape[channel_dim] = num_channels;

        let owned_rm;
        let r_mean = match running_mean {
            Some(rm) => rm
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_rm = candle_core::Tensor::zeros(shape.as_slice(), t.dtype(), t.device())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_rm
            }
        };
        let owned_rv;
        let r_var = match running_var {
            Some(rv) => rv
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_rv = candle_core::Tensor::ones(shape.as_slice(), t.dtype(), t.device())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_rv
            }
        };
        let owned_w;
        let w = match weight {
            Some(w) => w
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_w = candle_core::Tensor::ones(shape.as_slice(), t.dtype(), t.device())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_w
            }
        };
        let owned_b;
        let b = match bias {
            Some(b) => b
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_b = candle_core::Tensor::zeros(shape.as_slice(), t.dtype(), t.device())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_b
            }
        };

        let eps_t = candle_core::Tensor::new(&[eps], t.device())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let var_eps = r_var
            .broadcast_add(&eps_t)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let std = var_eps
            .sqrt()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let normalized = t
            .broadcast_sub(&r_mean)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
            .broadcast_div(&std)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

        let scaled = normalized
            .broadcast_mul(&w)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let out = scaled
            .broadcast_add(&b)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(out)
    }

    /// Looks up rows of embedding table `w` for each index in `t`, first
    /// casting indices to `U32` if they aren't already `U32`/`I64` (candle
    /// requires one of those two dtypes for embedding lookups).
    fn embedding<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<KInt>,
        w: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        use candle_nn::Module;
        let hidden_size = w
            .dim(1)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let emb = candle_nn::Embedding::new(w.clone(), hidden_size);

        // Candle requires U32 or I64 for embedding indices
        let t_idx = if t.dtype() != candle_core::DType::U32 && t.dtype() != candle_core::DType::I64
        {
            t.to_dtype(candle_core::DType::U32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        } else {
            t.clone()
        };

        Ok(emb
            .forward(&t_idx)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// 1-D convolution of `t` with kernel `w`; the bias argument is
    /// ignored.
    fn conv1d<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        w: &<Self as incin_core::prelude::Backend>::Storage<K>,
        _b: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.conv1d(w, padding, stride, dilation, groups)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// 2-D convolution of `t` with kernel `weight`; the bias argument is
    /// ignored.
    fn conv2d<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        weight: &<Self as incin_core::prelude::Backend>::Storage<K>,
        _bias: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.conv2d(weight, padding, stride, dilation, groups)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// 2-D transposed convolution of `t` with kernel `weight`; the bias
    /// and groups arguments are ignored.
    fn conv_transpose2d<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        weight: &<Self as incin_core::prelude::Backend>::Storage<K>,
        _bias: Option<&<Self as incin_core::prelude::Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        _groups: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(
            t.conv_transpose2d(weight, padding, output_padding, stride, dilation)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }

    /// 2-D max pooling with the given kernel size and stride; padding and
    /// dilation are ignored (not supported by candle's pooling op).
    fn max_pool2d<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(
            t.max_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }

    /// 2-D average pooling with the given kernel size and stride; padding
    /// is ignored (not supported by candle's pooling op).
    fn avg_pool2d<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(
            t.avg_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }
}
