//! Floating-point and activation operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

impl<D: incin_core::prelude::Device> incin_core::tensor::backend::FloatOps<Self>
    for CandleBackend<D>
{
    // Candle has native equivalents for several of these, but this adapter
    // does not route them yet. Declaring the gap here keeps it visible instead
    // of leaving it to a trait default that reads as full coverage.
    crate::unsupported::unsupported_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    /// Adds a scalar to every element of `t`.
    fn add_scalar_float<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = (t.tensor() + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Multiplies every element of `t` by a scalar.
    fn mul_scalar_float<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = (t.tensor() * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies the ReLU activation element-wise.
    fn relu<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .relu()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies GELU using candle's exact erf-based formulation
    /// (`gelu_erf`), used here as the general-purpose GELU.
    fn gelu<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .gelu_erf()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    } // using gelu_erf as fallback for general

    /// Implements Heaviside step function: H(x) = 0 if x < 0, else 1.
    /// Computed as: mask = (x >= 0), cast mask to float.
    fn step<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        // (t >= 0.0) gives a bool mask; cast to float dtype
        let zero = t
            .tensor()
            .zeros_like()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let mask = t
            .tensor()
            .ge(&zero)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let res = mask
            .to_dtype(t.tensor().dtype())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Implements Mish activation: `x * tanh(softplus(x))`
    /// where `softplus(x) = ln(1 + exp(x))`.
    fn mish<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        // softplus(x) = ln(1 + exp(x))
        let exp_x = t
            .tensor()
            .exp()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let softplus = (exp_x + 1.0f64)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
            .log()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        // mish(x) = x * tanh(softplus(x))
        let tanh_sp = softplus
            .tanh()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let res = t
            .tensor()
            .broadcast_mul(&tanh_sp)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Implements Exponential Linear Unit: ELU(x) = x if x >= 0, else 1*(e^x - 1).
    fn elu<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .elu(1.0f64)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Applies softmax along `dim`.
    fn softmax<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = candle_nn::ops::softmax(t.tensor(), dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Applies the swish/SiLU activation (`x * sigmoid(x)`) via candle's
    /// `silu` op.
    fn swish<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        // swish is x * sigmoid(x)
        let res =
            candle_nn::ops::silu(t.tensor()).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies absolute value element-wise.
    fn abs<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .abs()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Negates every element of `t`.
    fn neg<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .neg()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies element-wise square root.
    fn sqrt<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .sqrt()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies element-wise exponential.
    fn exp<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .exp()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies element-wise natural logarithm.
    fn log<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .log()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies element-wise hyperbolic tangent.
    fn tanh<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .tanh()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Applies the sigmoid activation element-wise.
    fn sigmoid<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = ::candle_nn::ops::sigmoid(t.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
}
