use crate::prelude::*;
use crate::nn::module::{Module, Parameters};
use alloc::vec::Vec;

pub trait LinearShape: Shape + DynShape {
    type BiasShape: Shape + DynShape;
}

impl<I: Dim, O: Dim> LinearShape for (I, O) {
    type BiasShape = (O,);
}

impl LinearShape for Dyn {
    type BiasShape = Dyn;
}

#[derive(Debug, Clone)]
pub struct Linear<S: LinearShape, B: Backend<Dyn> + Backend<S, RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor> + Backend<S::BiasShape, RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor>> {
    pub weight: Param<S, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
}

impl<S: LinearShape, B: Backend<Dyn> + Backend<S, RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor> + Backend<S::BiasShape, RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor>> Parameters<B> for Linear<S, B> {
    fn parameters(&self) -> Vec<<B as Backend<Dyn>>::RawVar> {
        let mut p = self.weight.parameters();
        if let Some(b) = &self.bias {
            p.extend(b.parameters());
        }
        p
    }
}

// Implement `Module` for `Linear` when input shape is `(Batch, In)`.
// We can use the MatMul bounds for static shape verification.
// For now, let's keep it simple: input is `Tensor<IS, B>` where `IS` is the input shape.
// To do this fully statically, we need trait bounds, but since `Tensor::matmul` already requires matching shapes at runtime, we can just defer to `matmul` for now.

impl<InF: Dim<Arg = ()>, OutF: Dim<Arg = ()>, B: Backend<Dyn> + Backend<(InF, OutF), RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor> + Backend<(OutF,), RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor> > Linear<(InF, OutF), B> {
    pub fn new() -> Result<Self> {
        let weight = Param::<(InF, OutF), B>::zeros(())?;
        let bias = Param::<(OutF,), B>::zeros(())?;
        Ok(Self {
            weight,
            bias: Some(bias),
        })
    }
}

impl<B: Backend<Dyn>> Linear<Dyn, B> {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let weight = Param::<Dyn, B>::zeros([in_features, out_features])?;
        let bias = Param::<Dyn, B>::zeros([out_features])?;
        Ok(Self {
            weight,
            bias: Some(bias),
        })
    }
}

// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).
// Let's stick to `weight: (InF, OutF)` for simpler `x @ weight`.

// Dynamic input
impl<B: Backend<Dyn>> Module<Tensor<Dyn, B>> for Linear<Dyn, B> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let out = x.matmul(&self.weight.as_tensor()?)?;
        if let Some(b) = &self.bias {
            out.add(&b.as_tensor()?)
        } else {
            Ok(out)
        }
    }
}

// Dynamic input for statically sized Linear
impl<InF: Dim<Arg = ()>, OutF: Dim<Arg = ()>, B: Backend<Dyn> + Backend<(InF, OutF), RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor> + Backend<(OutF,), RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor>> Module<Tensor<Dyn, B>> for Linear<(InF, OutF), B> {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        // First convert weight to dyn for computation
        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let out = x.matmul(&weight_dyn)?;
        if let Some(b) = &self.bias {
            let bias_dyn = b.as_tensor()?.into_shape::<Dyn>()?;
            out.add(&bias_dyn)
        } else {
            Ok(out)
        }
    }
}
