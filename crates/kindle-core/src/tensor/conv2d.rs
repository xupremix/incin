//! Conv2d shape verification

use crate::prelude::*;
use typenum::{Prod, Sum, Quot, Diff, U1, U2};

// ConvOutDim already defined in arithmetic.rs and exposed via prelude

pub trait Conv2dShape<K: Shape, Stride: StaticDim, Padding: StaticDim>: Shape {
    type Output: Shape;
    fn output_shape(lhs: &<Self as Shape>::Field, kernel: &<K as Shape>::Field) -> <Self::Output as Shape>::Field;
}

// Fully static (B, C_in, H_in, W_in) with Kernel (C_out, C_in, K_h, K_w)
impl<B, C_in, H_in, W_in, C_out, K_h, K_w, Stride, Padding> Conv2dShape<(C_out, C_in, K_h, K_w), Stride, Padding> for (B, C_in, H_in, W_in)
where
    B: StaticDim,
    C_in: StaticDim,
    H_in: StaticDim + core::ops::Add<Prod<U2, Padding>>,
    Sum<H_in, Prod<U2, Padding>>: core::ops::Sub<K_h>,
    Diff<Sum<H_in, Prod<U2, Padding>>, K_h>: core::ops::Div<Stride>,
    Quot<Diff<Sum<H_in, Prod<U2, Padding>>, K_h>, Stride>: core::ops::Add<U1>,
    W_in: StaticDim + core::ops::Add<Prod<U2, Padding>>,
    Sum<W_in, Prod<U2, Padding>>: core::ops::Sub<K_w>,
    Diff<Sum<W_in, Prod<U2, Padding>>, K_w>: core::ops::Div<Stride>,
    Quot<Diff<Sum<W_in, Prod<U2, Padding>>, K_w>, Stride>: core::ops::Add<U1>,
    C_out: StaticDim,
    K_h: StaticDim,
    K_w: StaticDim,
    Stride: StaticDim,
    Padding: StaticDim,
    U2: core::ops::Mul<Padding>,
    ConvOutDim<H_in, K_h, Stride, Padding>: StaticDim,
    ConvOutDim<W_in, K_w, Stride, Padding>: StaticDim,
{
    type Output = (B, C_out, ConvOutDim<H_in, K_h, Stride, Padding>, ConvOutDim<W_in, K_w, Stride, Padding>);
    
    #[inline(always)]
    fn output_shape(_: &<Self as Shape>::Field, _: &<(C_out, C_in, K_h, K_w) as Shape>::Field) -> <Self::Output as Shape>::Field {
        (Default::default(), Default::default(), Default::default(), Default::default())
    }
}

// Dyn shapes just return Dyn
impl<Stride: StaticDim, Padding: StaticDim> Conv2dShape<Dyn, Stride, Padding> for Dyn {
    type Output = Dyn;
    fn output_shape(_: &<Dyn as Shape>::Field, _: &<Dyn as Shape>::Field) -> <Dyn as Shape>::Field {
        alloc::vec![]
    }
}

impl<S1, B: Backend<S1>, T, D, G> Tensor<S1, B, T, D, G>
where
    S1: Shape + DynShape,
    T: DType,
    D: Device,
    G: RequiresGrad,
{
    pub fn conv2d<Stride, Padding, K_Shape>(
        &self,
        weight: &Tensor<K_Shape, B, T, D, G>,
        bias: Option<&Tensor<Dyn, B, T, D, G>>, // Simplified bias for now
    ) -> Result<Tensor<S1::Output, B, T, D, G>>
    where
        Stride: StaticDim + typenum::Unsigned,
        Padding: StaticDim + typenum::Unsigned,
        K_Shape: Shape + DynShape,
        S1: Conv2dShape<K_Shape, Stride, Padding>,
        B: Backend<K_Shape, RawTensor = <B as Backend<S1>>::RawTensor>
            + Backend<Dyn, RawTensor = <B as Backend<S1>>::RawTensor>
            + Backend<S1::Output, RawTensor = <B as Backend<S1>>::RawTensor>,
    {
        let inner = <B as Backend<S1>>::conv2d(
            &self.inner, 
            &weight.inner, 
            bias.map(|b| b.inner()), 
            <Stride as typenum::Unsigned>::USIZE, 
            <Padding as typenum::Unsigned>::USIZE, 
            1 // Default dilation
        )?;
        
        let output_shape = S1::output_shape(&self._shape, &weight._shape);
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
