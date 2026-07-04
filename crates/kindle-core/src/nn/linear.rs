use crate::nn::module::Module;
use crate::prelude::*;

pub trait LinearShape<InF: Dim, OutF: Dim>: Shape + DynShape {
    type BiasShape: Shape + DynShape;
}

impl<InF: Dim, OutF: Dim> LinearShape<InF, OutF> for (InF, OutF) {
    type BiasShape = (OutF,);
}

impl<InF: Dim, OutF: Dim> LinearShape<InF, OutF> for Dyn {
    type BiasShape = Dyn;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Linear<
    InF: Dim,
    OutF: Dim,
    S: LinearShape<InF, OutF>,
    B: Backend
        + Backend
        + Backend,
> {
    pub weight: Param<S, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
    _phantom: core::marker::PhantomData<(InF, OutF)>,
}



// Implement `Module` for `Linear` when input shape is `(Batch, In)`.
// We can use the MatMul bounds for static shape verification.
// For now, let's keep it simple: input is `Tensor<IS, B>` where `IS` is the input shape.
// To do this fully statically, we need trait bounds, but since `Tensor::matmul` already requires matching shapes at runtime, we can just defer to `matmul` for now.

impl<
    InF: Dim<Arg = ()>,
    OutF: Dim<Arg = ()>,
    B: Backend
        + Backend,
> Linear<InF, OutF, (InF, OutF), B>
{
    pub fn new() -> Result<Self>
    where B::DType: crate::prelude::ConstDType, B::Device: crate::prelude::ConstDevice
    {
        let weight = Param::<(InF, OutF), B>::zeros(())?;
        let bias = Param::<(OutF,), B>::zeros(())?;
        Ok(Self {
            weight,
            bias: Some(bias),
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<InF: Dim, OutF: Dim, B: Backend> Linear<InF, OutF, Dyn, B> where B::DType: crate::prelude::ConstDType, B::Device: crate::prelude::ConstDevice {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self>
    where B::DType: crate::prelude::ConstDType, B::Device: crate::prelude::ConstDevice
    {
        let weight = Param::<Dyn, B>::zeros([in_features, out_features])?;
        let bias = Param::<Dyn, B>::zeros([out_features])?;
        Ok(Self {
            weight,
            bias: Some(bias),
            _phantom: core::marker::PhantomData,
        })
    }
}

// In PyTorch, input is typically (*, InF) and weight is (OutF, InF).
// Let's stick to `weight: (InF, OutF)` for simpler `x @ weight`.

// Dynamic input
impl<InF: Dim, OutF: Dim, B: Backend> Module<Tensor<Dyn, B>> for Linear<InF, OutF, Dyn, B> {
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

// Statically typed input utilizing ReplaceLastDim
impl<
    InF: Dim,
    OutF: Dim,
    InShape: Shape + DynShape + ReplaceLastDim<OutF>,
    B: Backend
        + Backend
        + Backend,
> Module<Tensor<InShape, B>> for Linear<InF, OutF, (InF, OutF), B>
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
             let w_dims = <(InF, OutF) as DynShape>::dims(self.weight.as_tensor()?.shape_field());
             dims[last_idx] = w_dims[1];
        }
        let shape = <InShape::Output as Shape>::from_dyn(&dims).unwrap();

        // Convert to dyn for computation
        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let x_dyn = x.into_shape::<Dyn>()?;
        let out_dyn = x_dyn.matmul(&weight_dyn)?;
        
        let out_final = if let Some(b) = &self.bias {
            let bias_dyn = b.as_tensor()?.into_shape::<Dyn>()?;
            out_dyn.add(&bias_dyn)?
        } else {
            out_dyn
        };
        
        Ok(Tensor::from_parts(
            out_final.into_inner(),
            shape,
            dtype,
            device,
            core::marker::PhantomData,
        ))
    }
}
