use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::shapes::Conv2dShape;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv2d<K: Unsigned + crate::prelude::Dim, S: Unsigned, P: Unsigned, D: Unsigned, W: Shape, B: Backend> {
    pub weight: Param<W, B>,
    pub bias: Option<Param<Dyn, B>>,
    _phantom: core::marker::PhantomData<(K, S, P, D, W, B)>,
}



impl<K: Unsigned + crate::prelude::Dim<Arg = ()>, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend> Conv2d<K, S, P, D, (usize, usize, K, K), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_channels: usize, out_channels: usize) -> Result<Self> {
        let weight = Param::<(usize, usize, K, K), B>::zeros((out_channels, in_channels))?;
        let bias = Param::<Dyn, B>::zeros([out_channels])?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<K: Unsigned + crate::prelude::Dim<Arg = ()>, S: Unsigned, P: Unsigned, D: Unsigned, CIn: Dim<Arg = ()>, B: Backend> Conv2d<K, S, P, D, (usize, CIn, K, K), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(out_channels: usize) -> Result<Self> {
        let weight = Param::<(usize, CIn, K, K), B>::zeros(out_channels)?;
        let bias = Param::<Dyn, B>::zeros([out_channels])?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<K: Unsigned + crate::prelude::Dim<Arg = ()>, S: Unsigned, P: Unsigned, D: Unsigned, COut: Dim<Arg = ()> + Default, B: Backend> Conv2d<K, S, P, D, (COut, usize, K, K), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_channels: usize) -> Result<Self> {
        let weight = Param::<(COut, usize, K, K), B>::zeros(in_channels)?;
        let bias = Param::<Dyn, B>::zeros([COut::default().size()])?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<K: Unsigned + crate::prelude::Dim<Arg = ()>, S: Unsigned, P: Unsigned, D: Unsigned, COut: Dim<Arg = ()> + Default, CIn: Dim<Arg = ()>, B: Backend> Conv2d<K, S, P, D, (COut, CIn, K, K), B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new() -> Result<Self> {
        let weight = Param::<(COut, CIn, K, K), B>::zeros(())?;
        let bias = Param::<Dyn, B>::zeros([COut::default().size()])?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<K: Unsigned + crate::prelude::Dim<Arg = ()>, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend> Conv2d<K, S, P, D, Dyn, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(in_channels: usize, out_channels: usize) -> Result<Self> {
        let weight = Param::<Dyn, B>::zeros([out_channels, in_channels, K::USIZE, K::USIZE])?;
        let bias = Param::<Dyn, B>::zeros([out_channels])?;
        Ok(Self { weight, bias: Some(bias), _phantom: core::marker::PhantomData })
    }
}

impl<I, K, S, P, D, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv2d<K, S, P, D, (COut, CIn, K, K), B>
where
    K: Unsigned + crate::prelude::Dim,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    I: Shape + DynShape + Conv2dShape<COut, K, S, P, D> + crate::shapes::HasChannels2D<CIn>,
    B: Backend,
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?.detach()),
            None => None,
        };
        
        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 3].iter().product();
        let in_channels = x_shape[rank - 3];
        let h = x_shape[rank - 2];
        let w = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            <B as Backend>::reshape(&x.inner, &[batch_size, in_channels, h, w])?
        } else {
            x.inner.clone()
        };

        let out = <B as Backend>::conv2d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::USIZE,
            P::USIZE,
            D::USIZE,
        )?;

        let shape = <I as Conv2dShape<COut, K, S, P, D>>::compute_output_shape(
            x.shape_field(),
            weight.dims()[0],
        );
        
        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 4 {
            <B as Backend>::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            x.grad_field().clone(),
        ))
    }
}


impl<I, K, S, P, D, B> Module<Tensor<I, B>> for Conv2d<K, S, P, D, Dyn, B>
where
    K: Unsigned + crate::prelude::Dim,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    I: Shape + DynShape + Conv2dShape<Dyn, K, S, P, D>,
    B: Backend,
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?.detach()),
            None => None,
        };
        
        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 3].iter().product();
        let in_channels = x_shape[rank - 3];
        let h = x_shape[rank - 2];
        let w = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            <B as Backend>::reshape(&x.inner, &[batch_size, in_channels, h, w])?
        } else {
            x.inner.clone()
        };

        let out = <B as Backend>::conv2d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::USIZE,
            P::USIZE,
            D::USIZE,
        )?;

        let shape = <I as Conv2dShape<Dyn, K, S, P, D>>::compute_output_shape(
            x.shape_field(),
            weight.dims()[0],
        );
        
        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 4 {
            <B as Backend>::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
