use crate::nn::module::{Module, Parameters, ShapeInfo, TrainMode};
use crate::err::{Error, Result};
use crate::shapes::{DynShape, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::backend::{Backend, StorageBackend};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use alloc::{string::{String, ToString}, vec::Vec};

/// The Rectified Linear Unit (ReLU) activation function: `f(x) = max(0, x)`.
///
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct ReLU;

macro_rules! impl_stateless_shape_info {
    ($($ty:ty),+ $(,)?) => {
        $(impl ShapeInfo for $ty {
            fn shape_info(&self) -> Option<String> {
                None
            }
        })+
    };
}

impl_stateless_shape_info!(ReLU);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for ReLU {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for ReLU {}

use crate::exec::catalog::{Descriptor, op};
use crate::tensor::backend::Execute;

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Relu>, G: RequiresGrad>
    Module<Tensor<S, B, f32, G>> for ReLU
where
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G>,
    ) -> core::result::Result<Tensor<S, B, f32, G>, Error> {
        x.relu()
    }
}

/// The Gaussian Error Linear Unit (GELU) activation function.
///
/// GELU is a smooth approximation to ReLU commonly used in transformer architectures.
/// This is a stateless module with no learnable parameters.
#[derive(Debug, Clone, Default)]
pub struct GELU;

impl_stateless_shape_info!(GELU);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for GELU {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for GELU {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Gelu>> Module<Tensor<S, B>> for GELU
where
    <B as Execute<op::Gelu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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

impl_stateless_shape_info!(Swish);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for Swish {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Swish {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Swish>> Module<Tensor<S, B>> for Swish
where
    <B as Execute<op::Swish>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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

impl_stateless_shape_info!(Mish);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for Mish {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Mish {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Mish>> Module<Tensor<S, B>> for Mish
where
    <B as Execute<op::Mish>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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
#[allow(clippy::upper_case_acronyms)]
pub struct ELU;

impl_stateless_shape_info!(ELU);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for ELU {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for ELU {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Elu>> Module<Tensor<S, B>> for ELU
where
    <B as Execute<op::Elu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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

impl ShapeInfo for Softmax {
    fn shape_info(&self) -> Option<String> {
        None
    }
}

impl Softmax {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for Softmax {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Softmax {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + crate::tensor::backend::Execute<crate::exec::catalog::op::Softmax>,
> Module<Tensor<S, B>> for Softmax
where
    <B as crate::tensor::backend::Execute<crate::exec::catalog::op::Softmax>>::Output:
        Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.softmax(self.dim)
    }
}

/// The Sigmoid activation function: `f(x) = 1 / (1 + exp(-x))`.
///
/// Squashes each element into the range `(0, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl_stateless_shape_info!(Sigmoid);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for Sigmoid {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Sigmoid {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Sigmoid>> Module<Tensor<S, B>> for Sigmoid
where
    <B as Execute<op::Sigmoid>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.sigmoid()
    }
}

/// The Hyperbolic Tangent (Tanh) activation function: `f(x) = tanh(x)`.
///
/// Squashes each element into the range `(-1, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl_stateless_shape_info!(Tanh);

impl<B: crate::tensor::backend::VariableBackend, K: DType> Parameters<B, K> for Tanh {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Tanh {}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend + Execute<op::Tanh>> Module<Tensor<S, B>> for Tanh
where
    <B as Execute<op::Tanh>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.tanh()
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for ReLU {}
impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for GELU {}
impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for Swish {}
impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for Softmax {}
impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for Sigmoid {}
impl<B: crate::tensor::backend::VariableBackend> crate::nn::module::StateDict<B> for Tanh {}

macro_rules! impl_stateless_state_visitors {
    ($($t:ty),+ $(,)?) => {$ (
        impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitState<B> for $t {
            fn visit_state<V: crate::nn::StateVisitor<B>>(
                &self,
                _: &crate::nn::StatePath,
                _: &mut V,
            ) -> crate::prelude::Result<()> { Ok(()) }
        }

        impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitStateMut<B> for $t {
            fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
                &mut self,
                _: &crate::nn::StatePath,
                _: &mut V,
            ) -> crate::prelude::Result<()> { Ok(()) }
        }
    )+ };
}

impl_stateless_state_visitors!(ReLU, GELU, Swish, Mish, ELU, Softmax, Sigmoid, Tanh);

impl crate::nn::module::NamedLayers for ReLU {
    /// Returns the layer hierarchy rooted at this module for visualization.
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
    /// Returns the layer hierarchy rooted at this module for visualization.
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
    /// Returns the layer hierarchy rooted at this module for visualization.
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
    /// Returns the layer hierarchy rooted at this module for visualization.
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
    /// Returns the layer hierarchy rooted at this module for visualization.
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
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("Softmax"),
            shape_info: format!("dim={}", self.dim),
            children: vec![],
        }]
    }
}

macro_rules! impl_unit_to_device {
    ($($t:ty),+) => {
        $(
            impl<B: crate::tensor::backend::VariableBackend, NewD: Device> crate::tensor::transfer::ToDevice<B, NewD> for $t {
                type Output = $t;
                fn to_device(self, _arg: &NewD::Arg) -> Result<Self::Output> {
                    Ok(self)
                }
            }
        )+
    };
}

impl_unit_to_device!(ReLU, GELU, Swish, Mish, ELU, Softmax, Sigmoid, Tanh);
