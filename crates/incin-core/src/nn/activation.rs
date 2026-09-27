use crate::dist::Local;
use crate::err::{Error, Result};
use crate::nn::module::{Module, ShapeInfo, TrainMode};
use crate::shapes::{DynShape, Shape};
use crate::shapes::{Layout, Restatable, RowMajor};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::grad::RequiresGrad;
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for ReLU {}

use crate::exec::catalog::op;
use crate::tensor::backend::Execute;

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Relu>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for ReLU
where
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Tensor<S, B, f32, G, Local, RowMajor>, Error> {
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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for GELU {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Gelu>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for GELU
where
    <B as Execute<op::Gelu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Swish {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Swish>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for Swish
where
    <B as Execute<op::Swish>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Mish {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Mish>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for Mish
where
    <B as Execute<op::Mish>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for ELU {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Elu>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for ELU
where
    <B as Execute<op::Elu>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        x.elu()
    }
}

/// The Softmax activation function, applied along a specified axis.
///
/// Converts a vector of raw logits into a probability distribution that sums to 1.
///
/// ## Parameters
/// * `dim` - The axis along which the softmax normalization is applied.
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

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Softmax {}

impl<
    S: Shape + DynShape + crate::shapes::RuntimeRankProjection,
    B: crate::tensor::backend::VariableBackend + Execute<crate::exec::catalog::op::Softmax>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for Softmax
where
    <B as crate::tensor::backend::Execute<crate::exec::catalog::op::Softmax>>::Output:
        Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        x.softmax(self.dim as isize)
    }
}

/// The Sigmoid activation function: `f(x) = 1 / (1 + exp(-x))`.
///
/// Squashes each element into the range `(0, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl_stateless_shape_info!(Sigmoid);

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Sigmoid {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Sigmoid>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for Sigmoid
where
    <B as Execute<op::Sigmoid>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        x.sigmoid()
    }
}

/// The Hyperbolic Tangent (Tanh) activation function: `f(x) = tanh(x)`.
///
/// Squashes each element into the range `(-1, 1)`. This is a stateless module.
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl_stateless_shape_info!(Tanh);

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for Tanh {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + Execute<op::Tanh>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for Tanh
where
    <B as Execute<op::Tanh>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        x.tanh()
    }
}

macro_rules! impl_stateless_state_visitors {
    ($($t:ty),+ $(,)?) => {$ (
        impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitState<B> for $t {
            fn visit_state<V: crate::nn::StateVisitor<B>>(
                &self,
                _: &crate::nn::StatePath,
                _: &mut V,
            ) -> crate::err::Result<()> { Ok(()) }
        }

        impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitStateMut<B> for $t {
            fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
                &mut self,
                _: &crate::nn::StatePath,
                _: &mut V,
            ) -> crate::err::Result<()> { Ok(()) }
        }

        impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitParameters<B> for $t {
            fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
                &self,
                _: &crate::nn::StatePath,
                _: &mut V,
            ) -> crate::err::Result<()> { Ok(()) }
        }
    )+ };
}

impl_stateless_state_visitors!(
    ReLU, GELU, Swish, Mish, ELU, Softmax, LogSoftmax, LeakyReLU, Sigmoid, Tanh
);

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

impl crate::nn::module::NamedLayers for LogSoftmax {
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("LogSoftmax"),
            shape_info: format!("dim={}", self.dim),
            children: vec![],
        }]
    }
}

impl crate::nn::module::NamedLayers for LeakyReLU {
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        vec![crate::nn::module::LayerNode {
            name: prefix.to_string(),
            type_name: alloc::string::String::from("LeakyReLU"),
            shape_info: format!("negative_slope={}", self.negative_slope),
            children: vec![],
        }]
    }
}
/// The Log-Softmax activation function, applied along a specified axis.
///
/// Converts raw logits into log-probabilities in one numerically stable step
/// by wrapping the existing `op::LogSoftmax` catalog op, mirroring how
/// [`Softmax`] wraps `op::Softmax`. Reading far-from-maximum entries through
/// this module keeps them finite where `softmax` then `log` would yield
/// negative infinity.
///
/// ## Parameters
/// * `dim` - The axis along which the log-softmax normalization is applied.
#[derive(Debug, Clone)]
pub struct LogSoftmax {
    /// The axis along which log-softmax is applied.
    pub dim: usize,
}

impl ShapeInfo for LogSoftmax {
    fn shape_info(&self) -> Option<String> {
        None
    }
}

impl LogSoftmax {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for LogSoftmax {}

impl<
    S: Shape + DynShape + crate::shapes::RuntimeRankProjection,
    B: crate::tensor::backend::VariableBackend + Execute<crate::exec::catalog::op::LogSoftmax>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for LogSoftmax
where
    <B as crate::tensor::backend::Execute<crate::exec::catalog::op::LogSoftmax>>::Output:
        Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::activation::LogSoftmax;
    /// use incin::nn::module::Module;
    /// use incin::prelude::*;
    /// let act = LogSoftmax::new(0);
    /// let x = Cpu.tensor([0.0f32, -200.0, 1.0]).unwrap();
    /// let log_probs = act.forward(x).unwrap().to_vec1::<f32>().unwrap();
    /// // The middle entry survives as a large finite number.
    /// assert!(log_probs.iter().all(|value| value.is_finite()));
    /// let mass: f32 = log_probs.iter().map(|value| value.exp()).sum();
    /// assert!((mass - 1.0).abs() < 1e-5);
    /// ```
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        x.log_softmax(self.dim as isize)
    }
}

/// The Leaky Rectified Linear Unit: `f(x) = x` for `x > 0`, `slope * x`
/// otherwise.
///
/// There is no LeakyReLU catalog op, so this composes existing ops the way
/// the tensor-level losses do: a strict `x > 0` comparison mask selects
/// between the identity and the `slope`-scaled branch via `where_cond`.
/// The strict comparison pins the kink subgradient at `x == 0` to `slope`,
/// matching the CPU convention used by the sibling losses.
///
/// ## Parameters
/// * `negative_slope` - The slope of the negative branch (default `0.01`).
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    /// The slope applied to negative inputs.
    pub negative_slope: f64,
}

impl LeakyReLU {
    /// Creates a new instance with the default negative slope of `0.01`.
    pub fn new() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }

    /// Creates a new instance with an explicit negative slope.
    pub fn with_slope(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    /// Default is [`LeakyReLU::new`]: negative slope `0.01`.
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeInfo for LeakyReLU {
    fn shape_info(&self) -> Option<String> {
        None
    }
}

/// Stateless - no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl TrainMode for LeakyReLU {}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::MulScalar>
        + Execute<op::CmpGt>
        + Execute<op::WhereCond>,
    G: RequiresGrad,
    L: Layout<S>,
> Module<Tensor<S, B, f32, G, Local, L>> for LeakyReLU
where
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::CmpGt>>::Output: Into<B::Storage<bool>>,
    <B as Execute<op::WhereCond>>::Output: Into<B::Storage<f32>>,
    L: Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, B, f32, G, Local, RowMajor>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::activation::LeakyReLU;
    /// use incin::nn::module::Module;
    /// use incin::prelude::*;
    /// # fn main() -> Result<()> {
    /// let act = LeakyReLU::new();
    /// let x = Cpu.tensor([-2.0f32, 0.0, 3.0])?.require_grad();
    /// let probe = x.clone();
    /// // Positive side is the identity, negative side scales by 0.01.
    /// let y = act.forward(x)?.to_vec1::<f32>()?;
    /// assert!((y[0] - -0.02).abs() < 1e-6);
    /// assert_eq!(y[1], 0.0);
    /// assert_eq!(y[2], 3.0);
    /// // Backward: 1.0 above zero, `negative_slope` at and below zero.
    /// let grads = act.forward(probe.clone())?.sum_all()?.backward()?;
    /// let g = grads.require(&probe)?.to_vec1::<f32>()?;
    /// assert!((g[0] - 0.01).abs() < 1e-6);
    /// assert!((g[1] - 0.01).abs() < 1e-6);
    /// assert_eq!(g[2], 1.0);
    /// # Ok(()) }
    /// ```
    fn forward(
        &self,
        x: Tensor<S, B, f32, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let input = x.into_dyn();
        // `mul_scalar(1.0)` restates the input as a fresh `RowMajor` buffer so
        // both `where_cond` branches share one layout; its backward pass
        // multiplies by one, leaving gradients untouched.
        let base = input.mul_scalar(1.0)?;
        // Zeros sharing the input shape for the strict `x > 0` mask. The mask
        // feeds only the grad-disabled comparison, so its grad lineage (a
        // `mul_scalar(0.0)` of the input) is inert.
        let zeros = base.mul_scalar(0.0)?;
        let positive = base.gt(&zeros)?;
        let scaled = base.mul_scalar(self.negative_slope)?;
        let out = positive.where_cond(&base, &scaled)?;
        out.into_shape::<S>()
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

impl_unit_to_device!(
    ReLU, GELU, Swish, Mish, ELU, Softmax, LogSoftmax, LeakyReLU, Sigmoid, Tanh
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaky_relu_defaults_to_slope_point_zero_one() {
        assert_eq!(LeakyReLU::new().negative_slope, 0.01);
        assert_eq!(LeakyReLU::default().negative_slope, 0.01);
    }

    #[test]
    fn leaky_relu_with_slope_stores_slope() {
        assert_eq!(LeakyReLU::with_slope(0.2).negative_slope, 0.2);
        assert_eq!(LeakyReLU::with_slope(0.0).negative_slope, 0.0);
    }

    #[test]
    fn log_softmax_stores_dim() {
        assert_eq!(LogSoftmax::new(1).dim, 1);
        assert_eq!(LogSoftmax::new(2).dim, 2);
    }

    #[test]
    fn leaky_relu_is_clone_and_debug() {
        let act = LeakyReLU::with_slope(0.3);
        assert_eq!(act.clone().negative_slope, 0.3);
        assert!(alloc::format!("{act:?}").contains("0.3"));
    }
}
