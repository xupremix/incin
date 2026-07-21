use crate::nn::module::{Module, Parameters};
use crate::prelude::*;

/// The Rectified Linear Unit (ReLU) activation function: `f(x) = max(0, x)`.
///
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl<B: Backend> Parameters<B> for ReLU {
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for ReLU {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for GELU {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Swish {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.swish()
    }
}

/// The Mish activation function: `f(x) = x * tanh(softplus(x))`.
///
/// Mish is a smooth, continuous, non-monotonic function that can improve training dynamics.
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct Mish;

impl<B: Backend> Parameters<B> for Mish {
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Mish {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.mish()
    }
}

/// The Exponential Linear Unit (ELU) activation function.
///
/// ELU approaches a negative constant as the input gets smaller.
/// This implementation hardcodes alpha to 1.0.
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct ELU;

impl<B: Backend> Parameters<B> for ELU {
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for ELU {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.elu()
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
    /// Auto-generated documentation for new.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<B: Backend> Parameters<B> for Softmax {
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Softmax {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Sigmoid {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for named_parameters.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Tanh {
    /// Auto-generated documentation for Output.
    type Output = Tensor<S, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.tanh()
    }
}

impl<B: Backend> crate::nn::module::StateDict<B> for ReLU {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for GELU {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Swish {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Softmax {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Sigmoid {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}
impl<B: Backend> crate::nn::module::StateDict<B> for Tanh {
    /// Auto-generated documentation for load_state_dict.
    fn load_state_dict(
        &mut self,
        _: &str,
        _: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        Ok(())
    }
    /// Auto-generated documentation for state_dict.
    fn state_dict(&self, _: &str, _: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>) {}
}

impl crate::nn::module::NamedLayers for ReLU {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("ReLU"),
            shape_info: "".to_string(),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for GELU {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("GELU"),
            shape_info: "".to_string(),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for Swish {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("Swish"),
            shape_info: "".to_string(),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for Sigmoid {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("Sigmoid"),
            shape_info: "".to_string(),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for Tanh {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("Tanh"),
            shape_info: "".to_string(),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for Softmax {
    /// Auto-generated documentation for layer_structure.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("Softmax"),
            shape_info: format!("dim={}", self.dim),
            children: vec![],
        }]
    }
}
