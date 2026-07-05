use crate::nn::module::Module;
use crate::prelude::*;

pub trait LinearShape: Shape + DynShape {
    type InF;
    type OutF;
    type WeightShape: Shape + DynShape;
    type BiasShape: Shape + DynShape;
}


impl<InF: Dim, OutF: Dim> LinearShape for (InF, OutF) {
    type InF = InF;
    type OutF = OutF;
    type WeightShape = (OutF, InF);
    type BiasShape = (OutF,);
}

impl LinearShape for Dyn {
    type InF = Dyn;
    type OutF = Dyn;
    type WeightShape = Dyn;
    type BiasShape = Dyn;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Linear<S: LinearShape, B: Backend> {
    pub weight: Param<S::WeightShape, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
}

// Implement `Module` for `Linear` when input shape is `(Batch, In)`.


impl<B: Backend> Linear<(usize, usize), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let weight = Param::<(usize, usize), B>::new_init((out_features, in_features), crate::nn::init::Init::KaimingUniform { fan_in: in_features, a: f64::sqrt(5.0) })?;
        let bias = Param::<(usize,), B>::new_init((out_features,), crate::nn::init::Init::Uniform { bound: 1.0 / (in_features as f64).sqrt() })?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

impl<InF: Dim<Arg = ()>, B: Backend> Linear<(InF, usize), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(out_features: usize) -> Result<Self> {
        let weight = Param::<(usize, InF), B>::new_init((out_features, ()), crate::nn::init::Init::KaimingUniform { fan_in: InF::from_arg(()).size(), a: f64::sqrt(5.0) })?;
        let bias = Param::<(usize,), B>::new_init((out_features,), crate::nn::init::Init::Uniform { bound: 1.0 / (InF::from_arg(()).size() as f64).sqrt() })?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

impl<OutF: Dim<Arg = ()>, B: Backend> Linear<(usize, OutF), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_features: usize) -> Result<Self> {
        let weight = Param::<(OutF, usize), B>::new_init(((), in_features), crate::nn::init::Init::KaimingUniform { fan_in: in_features, a: f64::sqrt(5.0) })?;
        let bias = Param::<(OutF,), B>::new_init((), crate::nn::init::Init::Uniform { bound: 1.0 / (in_features as f64).sqrt() })?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

impl<InF: Dim<Arg = ()>, OutF: Dim<Arg = ()>, B: Backend> Linear<(InF, OutF), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new() -> Result<Self> {
        let weight = Param::<(OutF, InF), B>::new_init((), crate::nn::init::Init::KaimingUniform { fan_in: InF::from_arg(()).size(), a: f64::sqrt(5.0) })?;
        let bias = Param::<(OutF,), B>::new_init((), crate::nn::init::Init::Uniform { bound: 1.0 / (InF::from_arg(()).size() as f64).sqrt() })?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

impl<B: Backend> Linear<Dyn, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let weight = crate::prelude::Param::<Dyn, B>::new_init([out_features, in_features], crate::nn::init::Init::KaimingUniform { fan_in: in_features, a: f64::sqrt(5.0) })?;
        let bias = crate::prelude::Param::<Dyn, B>::new_init([out_features], crate::nn::init::Init::Uniform { bound: 1.0 / (in_features as f64).sqrt() })?;
        Ok(Self { weight, bias: Some(bias) })
    }
}

// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).
// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).

// Dynamic input
impl<B: Backend> Module<Tensor<Dyn, B>> for Linear<Dyn, B> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_t = self.weight.as_tensor()?.transpose::<0, 1>()?;
        let out = x.matmul(&weight_t)?;
        if let Some(b) = &self.bias {
            out.add(&b.as_tensor()?)
        } else {
            Ok(out)
        }
    }
}

// Statically typed input utilizing ReplaceLastDim
impl<InF: Dim, OutF: Dim, InShape: Shape + DynShape + ReplaceLastDim<OutF> + crate::shapes::EndsWith<InF>, B: Backend>
    Module<Tensor<InShape, B>> for Linear<(InF, OutF), B>
where
    InShape::Output: DynShape,
{
    type Output = Tensor<InShape::Output, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<InShape, B>) -> core::result::Result<Self::Output, Error> {
        let dtype = x._dtype.clone();
        let device = x._device.clone();

        // Resolve output shape structurally before consuming x
        let mut dims = <InShape as DynShape>::dims(x.shape_field()).into();
        let last_idx = dims.len().saturating_sub(1);
        if last_idx < dims.len() {
            let w_dims = <(OutF, InF) as DynShape>::dims(self.weight.as_tensor()?.shape_field());
            dims[last_idx] = w_dims[0];
        }
        let shape = <InShape::Output as Shape>::from_dyn(&dims).unwrap();

        // Convert to dyn for computation
        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let weight_t = weight_dyn.transpose::<0, 1>()?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_dyn = x_dyn.matmul(&weight_t)?;

        let out_final = if let Some(b) = &self.bias {
            let bias_dyn = b.as_tensor()?.into_shape::<Dyn>()?;
            out_dyn.broadcast_add(&bias_dyn)?
        } else {
            out_dyn
        };

        Ok(Tensor::from_parts_unchecked(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        ))
    }
}
