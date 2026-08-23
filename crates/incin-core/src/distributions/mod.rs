//! Probability distribution sampling with custom distribution trait support.

use crate::backend_authoring::{Capabilities, Execute};
use crate::err::{Error, Result};
use crate::exec::catalog::{CreationAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::shapes::ShapeValue;
use crate::shapes::{DynShape, Shape, ShapeBuf};
use crate::tensor::arg::TensorArgs;
use crate::tensor::backend::{Backend, SupportsDType};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use core::fmt::Debug;

fn uniform_tensor<S, B, G>(
    shape: &ShapeBuf,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<S, B, f32, G>>
where
    S: Shape + DynShape,
    B: Backend + SupportsDType<f32> + Execute<op::UniformRandom> + Capabilities,
    G: RequiresGrad,
    (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<f32>>,
{
    let device_id = B::Device::to_incin(device)?;
    let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
    let expected = ShapeValue::<S>::try_new(shape.clone()).map_err(Error::Shape)?;
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    let inner = dispatch::execute_shaped::<op::UniformRandom, B, S>(
        &context,
        CreationAttributes {
            shape: shape.as_ref().to_vec(),
            dtype,
            device: device_id,
        },
        &[],
        &expected,
    )?
    .into();
    Tensor::from_parts(
        inner,
        shape.clone(),
        Default::default(),
        device.clone(),
        Default::default(),
    )
}

fn normal_tensor<S, B, G>(
    shape: &ShapeBuf,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<S, B, f32, G>>
where
    S: Shape + DynShape,
    B: Backend + SupportsDType<f32> + Execute<op::NormalRandom> + Capabilities,
    G: RequiresGrad,
    (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    <B as Execute<op::NormalRandom>>::Output: Into<B::Storage<f32>>,
{
    let device_id = B::Device::to_incin(device)?;
    let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
    let expected = ShapeValue::<S>::try_new(shape.clone()).map_err(Error::Shape)?;
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    let inner = dispatch::execute_shaped::<op::NormalRandom, B, S>(
        &context,
        CreationAttributes {
            shape: shape.as_ref().to_vec(),
            dtype,
            device: device_id,
        },
        &[],
        &expected,
    )?
    .into();
    Tensor::from_parts(
        inner,
        shape.clone(),
        Default::default(),
        device.clone(),
        Default::default(),
    )
}

/// A trait for probability distributions capable of sampling tensors of any shape on backend `B`.
///
/// Implement this trait for custom probability distributions to allow direct sampling into
/// Incin static (`s![...]`) or dynamic (`Dyn`) tensors.
pub trait Distribution<K: DType = f32> {
    /// Samples a tensor of shape `S` on device `device`.
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<K> + DistributionExecutor<Self, K>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, K, G>>
    where
        (S, K, B::Device, G): TensorArgs<S, K, B::Device, G>;
}

/// Backend execution protocol for the special distribution sampling site.
///
/// Sampling combines a typed distribution identity with a backend-specific
/// operation sequence. The protocol keeps that resource contract explicit
/// without adding distribution operations to the core operation catalog.
pub trait DistributionExecutor<D: ?Sized, K: DType>: Backend + SupportsDType<K> {
    /// Samples a tensor from this distribution on the implementing backend.
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &D,
        shape: ShapeBuf,
        device: &<Self::Device as Device>::Field,
    ) -> Result<Tensor<S, Self, K, G>>
    where
        (S, K, Self::Device, G): TensorArgs<S, K, Self::Device, G>;
}

/// Extension trait for sampling a typed tensor from a distribution.
pub trait TensorDistributionExt<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> {
    /// Samples a tensor using the distribution's backend execution protocol.
    fn sample<D: Distribution<K>, A>(distribution: &D, args: A) -> Result<Tensor<S, B, K, G>>
    where
        A: crate::tensor::arg_into::ArgInto<
                <(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args,
            >,
        B: SupportsDType<K> + DistributionExecutor<D, K>;
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> TensorDistributionExt<S, B, K, G>
    for Tensor<S, B, K, G>
{
    fn sample<D: Distribution<K>, A>(distribution: &D, args: A) -> Result<Tensor<S, B, K, G>>
    where
        A: crate::tensor::arg_into::ArgInto<
                <(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args,
            >,
        B: SupportsDType<K> + DistributionExecutor<D, K>,
    {
        let (shape, _dtype, device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        distribution.sample::<S, B, G>(shape, &device)
    }
}

/// Uniform probability distribution over `[low, high)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform<K = f32> {
    /// Lower bound of the uniform range.
    pub low: K,
    /// Upper bound of the uniform range.
    pub high: K,
}

impl Uniform<f32> {
    /// Creates a new `Uniform` distribution over `[low, high)`.
    pub fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }
}

impl Default for Uniform<f32> {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl<B> DistributionExecutor<Uniform<f32>, f32> for B
where
    B: Backend
        + SupportsDType<f32>
        + Execute<op::UniformRandom>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + Capabilities,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<f32>>,
{
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &Uniform<f32>,
        shape: ShapeBuf,
        device: &<Self::Device as Device>::Field,
    ) -> Result<Tensor<S, Self, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        // Scale: low + (high - low) * rand
        let raw_rand = uniform_tensor::<S, B, G>(&shape, device)?;
        let range = (distribution.high - distribution.low) as f64;
        raw_rand
            .mul_scalar(range)?
            .add_scalar(distribution.low as f64)
    }
}

impl Distribution<f32> for Uniform<f32> {
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        B::sample_distribution::<S, G>(self, shape, device)
    }
}

/// Normal (Gaussian) probability distribution N(mean, std^2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal<K = f32> {
    /// Mean of the normal distribution.
    pub mean: K,
    /// Standard deviation of the normal distribution.
    pub std: K,
}

impl Normal<f32> {
    /// Creates a new `Normal` distribution with given `mean` and `std`.
    pub fn new(mean: f32, std: f32) -> Self {
        Self { mean, std }
    }
}

impl Default for Normal<f32> {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl<B> DistributionExecutor<Normal<f32>, f32> for B
where
    B: Backend
        + SupportsDType<f32>
        + Execute<op::NormalRandom>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + Capabilities,
    <B as Execute<op::NormalRandom>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<f32>>,
{
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &Normal<f32>,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        // Scale: mean + std * randn
        normal_tensor::<S, B, G>(&shape, device)?
            .mul_scalar(distribution.std as f64)?
            .add_scalar(distribution.mean as f64)
    }
}

impl Distribution<f32> for Normal<f32> {
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        B::sample_distribution::<S, G>(self, shape, device)
    }
}

/// Bernoulli probability distribution with success probability `p` in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli<K = f32> {
    /// Success probability of the Bernoulli draw.
    pub p: K,
}

impl Bernoulli<f32> {
    /// Creates a new `Bernoulli` distribution with success probability `p`.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl<B> DistributionExecutor<Bernoulli<f32>, f32> for B
where
    B: Backend
        + SupportsDType<f32>
        + Execute<op::UniformRandom>
        + Execute<op::Neg>
        + Execute<op::AddScalar>
        + Execute<op::Step>
        + Capabilities,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Neg>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Step>>::Output: Into<B::Storage<f32>>,
{
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &Bernoulli<f32>,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        // Threshold: (rand < p) -> step(p - rand)
        uniform_tensor::<S, B, G>(&shape, device)?
            .neg()?
            .add_scalar(distribution.p as f64)?
            .step()
    }
}

impl Distribution<f32> for Bernoulli<f32> {
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        B::sample_distribution::<S, G>(self, shape, device)
    }
}

/// Exponential probability distribution with rate parameter `lambda` > 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential<K = f32> {
    /// Rate parameter of the exponential draw.
    pub lambda: K,
}

impl Exponential<f32> {
    /// Creates a new `Exponential` distribution with rate parameter `lambda`.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

impl<B> DistributionExecutor<Exponential<f32>, f32> for B
where
    B: Backend
        + SupportsDType<f32>
        + Execute<op::UniformRandom>
        + Execute<op::Neg>
        + Execute<op::AddScalar>
        + Execute<op::Log>
        + Execute<op::MulScalar>
        + Capabilities,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Neg>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Log>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<f32>>,
{
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &Exponential<f32>,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        // Inverse Transform: -ln(1 - U) / lambda
        uniform_tensor::<S, B, G>(&shape, device)?
            .neg()?
            .add_scalar(1.0)?
            .log()?
            .neg()?
            .mul_scalar(1.0 / distribution.lambda as f64)
    }
}

impl Distribution<f32> for Exponential<f32> {
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        B::sample_distribution::<S, G>(self, shape, device)
    }
}

/// Gumbel probability distribution G(loc, scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gumbel<K = f32> {
    /// Location parameter of the Gumbel draw.
    pub loc: K,
    /// Scale parameter of the Gumbel draw.
    pub scale: K,
}

impl Gumbel<f32> {
    /// Creates a new `Gumbel` distribution with `loc` and `scale`.
    pub fn new(loc: f32, scale: f32) -> Self {
        Self { loc, scale }
    }
}

impl Default for Gumbel<f32> {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl<B> DistributionExecutor<Gumbel<f32>, f32> for B
where
    B: Backend
        + SupportsDType<f32>
        + Execute<op::UniformRandom>
        + Execute<op::Log>
        + Execute<op::Neg>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + Capabilities,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Log>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Neg>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<f32>>,
{
    fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
        distribution: &Gumbel<f32>,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        // Gumbel sample: loc - scale * ln(-ln(U))
        let log_u = uniform_tensor::<S, B, G>(&shape, device)?.log()?;
        let log_neg_log = log_u.neg()?.log()?;
        log_neg_log
            .mul_scalar(distribution.scale as f64)?
            .neg()?
            .add_scalar(distribution.loc as f64)
    }
}

impl Distribution<f32> for Gumbel<f32> {
    fn sample<
        S: Shape + DynShape,
        B: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
        G: RequiresGrad,
    >(
        &self,
        shape: ShapeBuf,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
    {
        B::sample_distribution::<S, G>(self, shape, device)
    }
}

// Sampling behaviour is proven in `tests/distributions.rs`, which links this
// crate the way a user does and can therefore run the samplers on the real
// CPU backend and inspect the values they produce.
