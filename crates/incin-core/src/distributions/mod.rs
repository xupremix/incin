//! Probability distribution sampling with custom distribution trait support.

use crate::prelude::*;
use core::fmt::Debug;

/// A trait for probability distributions capable of sampling tensors of any shape on backend `B`.
///
/// Implement this trait for custom probability distributions to allow direct sampling into
/// Incin static (`s![...]`) or dynamic (`Dyn`) tensors.
pub trait Distribution<K: DType = f32> {
    /// Samples a tensor of shape `S` on device `device`.
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = K>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, K, G>>
    where
        (S, K, B::Device, G): TensorArgs<S, K, B::Device, G>,
        B: SupportsDType<K>;
}

/// Uniform probability distribution over `[low, high)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform<K = f32> {
    pub low: K,
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

impl Distribution<f32> for Uniform<f32> {
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
        B: SupportsDType<f32>,
    {
        let dims = S::dims(&shape);
        let device_id = B::Device::to_incin(device)?;
        let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
        let raw_rand = B::rand(dims.as_ref(), dtype, &device_id)?;

        // Scale: low + (high - low) * rand
        let range = (self.high - self.low) as f64;
        let scaled = B::add_scalar_float(&B::mul_scalar_float(&raw_rand, range)?, self.low as f64)?;

        Tensor::from_parts(scaled, shape, Default::default(), device.clone(), Default::default())
    }
}

/// Normal (Gaussian) probability distribution N(mean, std^2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal<K = f32> {
    pub mean: K,
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

impl Distribution<f32> for Normal<f32> {
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
        B: SupportsDType<f32>,
    {
        let dims = S::dims(&shape);
        let device_id = B::Device::to_incin(device)?;
        let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
        let raw_randn = B::randn(dims.as_ref(), dtype, &device_id)?;

        // Scale: mean + std * randn
        let scaled = B::add_scalar_float(&B::mul_scalar_float(&raw_randn, self.std as f64)?, self.mean as f64)?;

        Tensor::from_parts(scaled, shape, Default::default(), device.clone(), Default::default())
    }
}

/// Bernoulli probability distribution with success probability `p` in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli<K = f32> {
    pub p: K,
}

impl Bernoulli<f32> {
    /// Creates a new `Bernoulli` distribution with success probability `p`.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Distribution<f32> for Bernoulli<f32> {
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
        B: SupportsDType<f32>,
    {
        let dims = S::dims(&shape);
        let device_id = B::Device::to_incin(device)?;
        let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
        let raw_rand = B::rand(dims.as_ref(), dtype, &device_id)?;

        // Threshold: (rand < p) -> step(p - rand)
        let sub = B::add_scalar_float(&B::neg(&raw_rand)?, self.p as f64)?;
        let step = B::step(&sub)?;

        Tensor::from_parts(step, shape, Default::default(), device.clone(), Default::default())
    }
}

/// Exponential probability distribution with rate parameter `lambda` > 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential<K = f32> {
    pub lambda: K,
}

impl Exponential<f32> {
    /// Creates a new `Exponential` distribution with rate parameter `lambda`.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

impl Distribution<f32> for Exponential<f32> {
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
        B: SupportsDType<f32>,
    {
        let dims = S::dims(&shape);
        let device_id = B::Device::to_incin(device)?;
        let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
        let raw_rand = B::rand(dims.as_ref(), dtype, &device_id)?;

        // Inverse Transform: -ln(1 - U) / lambda
        let one_minus_u = B::add_scalar_float(&B::neg(&raw_rand)?, 1.0)?;
        let log_val = B::log(&one_minus_u)?;
        let scaled = B::mul_scalar_float(&B::neg(&log_val)?, 1.0 / self.lambda as f64)?;

        Tensor::from_parts(scaled, shape, Default::default(), device.clone(), Default::default())
    }
}

/// Gumbel probability distribution G(loc, scale).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gumbel<K = f32> {
    pub loc: K,
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

impl Distribution<f32> for Gumbel<f32> {
    fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
        &self,
        shape: S::Field,
        device: &<B::Device as Device>::Field,
    ) -> Result<Tensor<S, B, f32, G>>
    where
        (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
        B: SupportsDType<f32>,
    {
        let dims = S::dims(&shape);
        let device_id = B::Device::to_incin(device)?;
        let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
        let raw_rand = B::rand(dims.as_ref(), dtype, &device_id)?;

        // Gumbel sample: loc - scale * ln(-ln(U))
        let log_u = B::log(&raw_rand)?;
        let neg_log_u = B::neg(&log_u)?;
        let log_neg_log = B::log(&neg_log_u)?;
        let scaled = B::mul_scalar_float(&log_neg_log, self.scale as f64)?;
        let gumbel = B::add_scalar_float(&B::neg(&scaled)?, self.loc as f64)?;

        Tensor::from_parts(gumbel, shape, Default::default(), device.clone(), Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>;

    #[test]
    fn test_uniform_sampling() {
        let u = Uniform::new(-5.0, 5.0);
        let t = Tensor::<Dyn, B>::sample(&u, vec![10, 10]).unwrap();
        assert_eq!(t.dims(), vec![10, 10]);
    }

    #[test]
    fn test_normal_sampling() {
        let n = Normal::new(10.0, 2.0);
        let t = Tensor::<Dyn, B>::sample(&n, vec![4, 4]).unwrap();
        assert_eq!(t.dims(), vec![4, 4]);
    }

    #[test]
    fn test_bernoulli_sampling() {
        let b_dist = Bernoulli::new(0.5);
        let t = Tensor::<Dyn, B>::sample(&b_dist, vec![100]).unwrap();
        assert_eq!(t.dims(), vec![100]);
    }

    #[test]
    fn test_exponential_sampling() {
        let exp_dist = Exponential::new(1.5);
        let t = Tensor::<Dyn, B>::sample(&exp_dist, vec![20]).unwrap();
        assert_eq!(t.dims(), vec![20]);
    }

    #[test]
    fn test_gumbel_sampling() {
        let g_dist = Gumbel::new(0.0, 1.0);
        let t = Tensor::<Dyn, B>::sample(&g_dist, vec![5, 5]).unwrap();
        assert_eq!(t.dims(), vec![5, 5]);
    }

    #[test]
    fn test_custom_distribution() {
        // Custom distribution sampling constant value + 42.0
        struct ConstantAdd(f32);
        impl Distribution<f32> for ConstantAdd {
            fn sample<S: Shape + DynShape, B: Backend<FloatElem = f32>, G: RequiresGrad>(
                &self,
                shape: S::Field,
                device: &<B::Device as Device>::Field,
            ) -> Result<Tensor<S, B, f32, G>>
            where
                (S, f32, B::Device, G): TensorArgs<S, f32, B::Device, G>,
                B: SupportsDType<f32>,
            {
                let dims = S::dims(&shape);
                let device_id = B::Device::to_incin(device)?;
                let dtype = B::resolve_dtype(&Default::default(), &device_id)?;
                let zeros = B::zeros(dims.as_ref(), dtype, &device_id)?;
                let add = B::add_scalar_float(&zeros, self.0 as f64)?;
                Tensor::from_parts(add, shape, Default::default(), device.clone(), Default::default())
            }
        }

        let custom = ConstantAdd(42.0);
        let t = Tensor::<Dyn, B>::sample(&custom, vec![3, 3]).unwrap();
        assert_eq!(t.dims(), vec![3, 3]);
    }

    #[test]
    fn test_comptime_known_device_sampling() {
        use typenum::{U2, U3};
        // When Shape and Device (and DType + Grad) are compile-time known, args is simply ()
        let dist = Normal::new(0.0, 1.0);
        let t = Tensor::<(U2, U3), B>::sample(&dist, ()).unwrap();
        assert_eq!(t.dims(), [2, 3]);
    }
}
