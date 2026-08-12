//! Neural-network module operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

impl<D: incin_core::prelude::Device> incin_core::tensor::backend::ModuleOps<Self>
    for CandleBackend<D>
{
    /// Candle has no native adaptive pooling; returns an error
    /// instead of panicking so callers can handle the unsupported case.
    fn adaptive_avg_pool2d<K: incin_core::prelude::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "adaptive_avg_pool2d",
            backend: "CandleBackend",
        })
    }

    /// Applies layer normalization over the last dimension, substituting a
    /// zero bias when none is provided.
    fn layer_norm<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let zero_bias;
        let bias_tensor = match bias {
            Some(b) => b.tensor(),
            None => {
                zero_bias = weight
                    .tensor()
                    .zeros_like()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                &zero_bias
            }
        };
        let res = candle_nn::ops::layer_norm(t.tensor(), weight.tensor(), bias_tensor, eps)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Applies batch normalization using the running mean/variance (or
    /// defaults of 0/1 when not provided) and an optional affine
    /// weight/bias, reshaping all of them to broadcast over the channel
    /// dimension.
    fn batch_norm<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: Option<&<Self as StorageBackend>::Storage<K>>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        running_mean: Option<&<Self as StorageBackend>::Storage<K>>,
        running_var: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
        _momentum: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let channel_dim = if t.tensor().rank() > 1 { 1 } else { 0 };
        let num_channels = t
            .tensor()
            .dim(channel_dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

        let mut shape = vec![1; t.tensor().rank()];
        shape[channel_dim] = num_channels;

        let owned_rm;
        let r_mean = match running_mean {
            Some(rm) => rm
                .tensor()
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_rm = candle_core::Tensor::zeros(
                    shape.as_slice(),
                    t.tensor().dtype(),
                    t.tensor().device(),
                )
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_rm
            }
        };
        let owned_rv;
        let r_var = match running_var {
            Some(rv) => rv
                .tensor()
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_rv = candle_core::Tensor::ones(
                    shape.as_slice(),
                    t.tensor().dtype(),
                    t.tensor().device(),
                )
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_rv
            }
        };
        let owned_w;
        let w = match weight {
            Some(w) => w
                .tensor()
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_w = candle_core::Tensor::ones(
                    shape.as_slice(),
                    t.tensor().dtype(),
                    t.tensor().device(),
                )
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_w
            }
        };
        let owned_b;
        let b = match bias {
            Some(b) => b
                .tensor()
                .reshape(shape.as_slice())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => {
                owned_b = candle_core::Tensor::zeros(
                    shape.as_slice(),
                    t.tensor().dtype(),
                    t.tensor().device(),
                )
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                owned_b
            }
        };

        let eps_t = candle_core::Tensor::new(&[eps], t.tensor().device())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let var_eps = r_var
            .broadcast_add(&eps_t)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let std = var_eps
            .sqrt()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let normalized = t
            .tensor()
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
        CandleStorage::try_new(out)
    }

    /// Looks up rows of embedding table `w` for each index in `t`, first
    /// casting indices to `U32` if they aren't already `U32`/`I64` (candle
    /// requires one of those two dtypes for embedding lookups).
    fn embedding<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<KInt>,
        w: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        use candle_nn::Module;
        let hidden_size = w
            .tensor()
            .dim(1)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let emb = candle_nn::Embedding::new(w.tensor().clone(), hidden_size);

        // Candle requires U32 or I64 for embedding indices
        let t_idx = if t.tensor().dtype() != candle_core::DType::U32
            && t.tensor().dtype() != candle_core::DType::I64
        {
            t.tensor()
                .to_dtype(candle_core::DType::U32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        } else {
            t.tensor().clone()
        };

        let res = emb
            .forward(&t_idx)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// 1-D convolution of `t` with kernel `w`; the bias argument is
    /// ignored.
    fn conv1d<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        _b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .conv1d(w.tensor(), padding, stride, dilation, groups)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// 2-D convolution of `t` with kernel `weight`; the bias argument is
    /// ignored.
    fn conv2d<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        _bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .conv2d(weight.tensor(), padding, stride, dilation, groups)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// 2-D transposed convolution of `t` with kernel `weight`; the bias
    /// and groups arguments are ignored.
    fn conv_transpose2d<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        _bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        _groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .conv_transpose2d(weight.tensor(), padding, output_padding, stride, dilation)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// 2-D max pooling with the given kernel size and stride; padding and
    /// dilation are ignored (not supported by candle's pooling op).
    fn max_pool2d<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .max_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// 2-D average pooling with the given kernel size and stride; padding
    /// is ignored (not supported by candle's pooling op).
    fn avg_pool2d<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .avg_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
}
