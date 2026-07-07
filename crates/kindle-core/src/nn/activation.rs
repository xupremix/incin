use crate::nn::module::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

/// The Rectified Linear Unit (ReLU) activation function: `f(x) = max(0, x)`.
/// 
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl<B: Backend> Parameters<B> for ReLU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for ReLU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.relu()
    }
}

/// The Gaussian Error Linear Unit (GELU) activation function.
/// 
/// GELU is a smooth approximation to ReLU commonly used in transformer architectures.
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct GELU;

impl<B: Backend> Parameters<B> for GELU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for GELU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.gelu()
    }
}

/// The Swish (SiLU) activation function: `f(x) = x * sigmoid(x)`.
/// 
/// Swish is a smooth, non-monotonic function that consistently performs better than ReLU
/// in deeper networks. This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct Swish;

impl<B: Backend> Parameters<B> for Swish {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Swish {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.swish()
    }
}

/// The Softmax activation function, applied along a specified axis.
/// 
/// Converts a vector of raw logits into a probability distribution that sums to 1.
/// 
/// ## Parameters
/// * `dim` — The axis along which the softmax normalization is applied.
#[derive(Debug, Clone)]
pub struct Softmax {
    /// The axis along which softmax is applied.
    pub dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<B: Backend> Parameters<B> for Softmax {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Softmax {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.softmax(self.dim)
    }
}

/// The Sigmoid activation function: `f(x) = 1 / (1 + exp(-x))`.
/// 
/// Squashes each element into the range `(0, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl<B: Backend> Parameters<B> for Sigmoid {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Sigmoid {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.sigmoid()
    }
}

/// The Hyperbolic Tangent (Tanh) activation function: `f(x) = tanh(x)`.
/// 
/// Squashes each element into the range `(-1, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl<B: Backend> Parameters<B> for Tanh {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Tanh {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.tanh()
    }
}

impl<B: Backend> crate::nn::module::StateDict<B> for ReLU {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for GELU {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Swish {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Softmax {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Sigmoid {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Tanh {
    fn load_state_dict(&mut self, _: &str, _: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> { Ok(()) }
    fn state_dict(&self, _: &str, _: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {}
}
