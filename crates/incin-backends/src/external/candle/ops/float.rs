//! Floating-point and activation operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::prelude::FloatOps<Self> for CandleBackend<T, D>
{
    /// Adds a scalar to every element of `t`.
    fn add_scalar_float<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok((t + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Multiplies every element of `t` by a scalar.
    fn mul_scalar_float<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok((t * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies the ReLU activation element-wise.
    fn relu<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.relu()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies GELU using candle's exact erf-based formulation
    /// (`gelu_erf`), used here as the general-purpose GELU.
    fn gelu<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.gelu_erf()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    } // using gelu_erf as fallback for general

    /// Implements Heaviside step function: H(x) = 0 if x < 0, else 1.
    /// Computed as: mask = (x >= 0), cast mask to float.
    fn step<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        // (t >= 0.0) gives a bool mask; cast to float dtype
        let zero = t
            .zeros_like()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let mask = t
            .ge(&zero)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(mask
            .to_dtype(t.dtype())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Implements Mish activation: `x * tanh(softplus(x))`
    /// where `softplus(x) = ln(1 + exp(x))`.
    fn mish<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        // softplus(x) = ln(1 + exp(x))
        let exp_x = t
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
        Ok(t.broadcast_mul(&tanh_sp)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Implements Exponential Linear Unit: ELU(x) = x if x >= 0, else 1*(e^x - 1).
    fn elu<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.elu(1.0f64)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Applies softmax along `dim`.
    fn softmax<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(candle_nn::ops::softmax(t, dim).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Applies the swish/SiLU activation (`x * sigmoid(x)`) via candle's
    /// `silu` op.
    fn swish<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        // swish is x * sigmoid(x)
        Ok(candle_nn::ops::silu(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies absolute value element-wise.
    fn abs<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.abs()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Negates every element of `t`.
    fn neg<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.neg()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies element-wise square root.
    fn sqrt<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.sqrt()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies element-wise exponential.
    fn exp<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.exp()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies element-wise natural logarithm.
    fn log<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.log()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies element-wise hyperbolic tangent.
    fn tanh<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.tanh()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Applies the sigmoid activation element-wise.
    fn sigmoid<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(::candle_nn::ops::sigmoid(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
}
