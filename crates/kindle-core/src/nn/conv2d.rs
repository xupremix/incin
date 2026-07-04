use crate::nn::module::Module;
use crate::prelude::*;
use typenum::Unsigned;

pub trait Conv2dShape: Shape + DynShape {
    type BiasShape: Shape + DynShape;
}

impl<COut: Dim, CIn: Dim, KH: Dim, KW: Dim> Conv2dShape for (COut, CIn, KH, KW) {
    type BiasShape = (COut,);
}

impl Conv2dShape for Dyn {
    type BiasShape = Dyn;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv2d<
    S: Conv2dShape,
    Stride: Unsigned + Default,
    Padding: Unsigned + Default,
    B: Backend<Dyn>
        + Backend<S, RawVar = <B as Backend<Dyn>>::RawVar, RawTensor = <B as Backend<Dyn>>::RawTensor>
        + Backend<
            S::BiasShape,
            RawVar = <B as Backend<Dyn>>::RawVar,
            RawTensor = <B as Backend<Dyn>>::RawTensor,
        >,
> {
    pub weight: Param<S, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
    _stride: core::marker::PhantomData<Stride>,
    _padding: core::marker::PhantomData<Padding>,
}



// Static initialization
impl<
    COut: Dim<Arg = ()>,
    CIn: Dim<Arg = ()>,
    KH: Dim<Arg = ()>,
    KW: Dim<Arg = ()>,
    Stride: Unsigned + Default,
    Padding: Unsigned + Default,
    B: Backend<Dyn>
        + Backend<
            (COut, CIn, KH, KW),
            RawVar = <B as Backend<Dyn>>::RawVar,
            RawTensor = <B as Backend<Dyn>>::RawTensor,
        > + Backend<
            (COut,),
            RawVar = <B as Backend<Dyn>>::RawVar,
            RawTensor = <B as Backend<Dyn>>::RawTensor,
        >,
> Conv2d<(COut, CIn, KH, KW), Stride, Padding, B>
{
    pub fn new() -> core::result::Result<Self, Error> {
        let weight = Param::<(COut, CIn, KH, KW), B>::zeros(())?;
        let bias = Param::<(COut,), B>::zeros(())?;
        Ok(Self {
            weight,
            bias: Some(bias),
            _stride: core::marker::PhantomData,
            _padding: core::marker::PhantomData,
        })
    }
}

// Dynamic initialization
impl<Stride: Unsigned + Default, Padding: Unsigned + Default, B: Backend<Dyn>>
    Conv2d<Dyn, Stride, Padding, B>
{
    pub fn new(cout: usize, cin: usize, kh: usize, kw: usize) -> core::result::Result<Self, Error> {
        let weight = Param::<Dyn, B>::zeros([cout, cin, kh, kw])?;
        let bias = Param::<Dyn, B>::zeros([cout])?;
        Ok(Self {
            weight,
            bias: Some(bias),
            _stride: core::marker::PhantomData,
            _padding: core::marker::PhantomData,
        })
    }
}

// Forward Dynamic for dynamically-initialized Conv2d
impl<
    Stride: Unsigned + Default + StaticDim,
    Padding: Unsigned + Default + StaticDim,
    B: Backend<Dyn>,
> Module<Tensor<Dyn, B>> for Conv2d<Dyn, Stride, Padding, B>
{
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let bias_ref = match &self.bias {
            Some(b) => Some(b.as_tensor()?),
            None => None,
        };
        let b = bias_ref.as_ref();
        x.conv2d::<Stride, Padding, _>(&self.weight.as_tensor()?, b)
    }
}

// Forward Dynamic for statically-initialized Conv2d
impl<
    COut: Dim<Arg = ()>,
    CIn: Dim<Arg = ()>,
    KH: Dim<Arg = ()>,
    KW: Dim<Arg = ()>,
    Stride: Unsigned + Default + StaticDim,
    Padding: Unsigned + Default + StaticDim,
    B: Backend<Dyn>
        + Backend<
            (COut, CIn, KH, KW),
            RawVar = <B as Backend<Dyn>>::RawVar,
            RawTensor = <B as Backend<Dyn>>::RawTensor,
        > + Backend<
            (COut,),
            RawVar = <B as Backend<Dyn>>::RawVar,
            RawTensor = <B as Backend<Dyn>>::RawTensor,
        >,
> Module<Tensor<Dyn, B>> for Conv2d<(COut, CIn, KH, KW), Stride, Padding, B>
{
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Tensor<Dyn, B>, Error> {
        let weight_dyn = self.weight.as_tensor()?.into_shape::<Dyn>()?;
        let bias_dyn = match &self.bias {
            Some(b) => Some(b.as_tensor()?.into_shape::<Dyn>()?),
            None => None,
        };
        let b = bias_dyn.as_ref();
        x.conv2d::<Stride, Padding, _>(&weight_dyn, b)
    }
}

// Forward Static
// (Batch, CIn, HIn, WIn) -> (Batch, COut, HOut, WOut)
// We need to know HOut and WOut based on HIn and WIn.
// For now, let's keep it simple and defer to dynamic or let the user explicitly specify the output shape?
// The current tensor `conv2d` returns `Tensor<Dyn, B>` when called on static shapes and then requires `.into_shape()`.
// This is because compile-time arithmetic (HOut = (HIn + 2P - K)/S + 1) requires `typenum` math.
// Let's implement it by returning `Tensor<Dyn, B>` and letting the user cast it, OR we just implement `Module` for `Tensor<Dyn, B>` for all cases for now?
// Let's implement the math!
