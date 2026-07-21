use crate::nn::{Module, Parameters};
use crate::prelude::*;

/// A Dropout layer.
///
/// Randomly zeroes some of the elements of the input tensor with probability `p` using samples
/// from a uniform distribution. The elements to zero are randomized on every forward call during training.
///
/// This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
/// Furthermore, the outputs are scaled by a factor of `1 / (1 - p)` during training. This means that during
/// evaluation the module simply computes an identity function.
#[derive(Debug, Clone)]
pub struct Dropout {
    /// The probability of an element to be zeroed.
    pub p: f32,
    /// Whether the module is in training mode.
    /// If false, dropout acts as an identity function.
    pub is_training: bool,
}

impl Dropout {
    /// Creates a new Dropout module with the specified probability.
    /// By default, `is_training` is set to true.
    pub fn new(p: f32) -> Self {
        Self {
            p,
            is_training: true,
        }
    }
}

impl<B: Backend> Parameters<B> for Dropout {
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        // No learnable parameters.
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Dropout
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        if !self.is_training || self.p <= 0.0 {
            return Ok(x);
        }

        if self.p >= 1.0 {
            return x.mul_scalar(0.0);
        }

        let scale = 1.0 / (1.0 - self.p);
        
        // Generate uniform mask in [0, 1)
        let dtype = <B::FloatElem as ConstDType>::DTYPE;
        let mask_inner = B::rand(x.dims().as_ref(), dtype, &x.device()?)?;
        let mask = Tensor::<S, B, B::FloatElem, B::Device, NoGrad>::from_parts_unchecked(
            mask_inner,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        );
        
        // mask - p is positive for (1 - p) proportion of elements
        let mask = mask.add_scalar(-self.p)?;
        
        // apply step function: 1.0 if (mask - p) > 0.0 else 0.0
        let binary_mask = mask.step()?;

        // multiply input by mask and scale
        let out = x.mul(&binary_mask.require_grad())?.mul_scalar(scale)?;
        Ok(out)
    }
}
