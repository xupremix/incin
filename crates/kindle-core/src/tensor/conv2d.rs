//! Conv2d shape verification

use crate::prelude::*;
use typenum::{Diff, Prod, Quot, Sum, U1, U2};

// ConvOutDim already defined in arithmetic.rs and exposed via prelude

pub trait KernelConv2dShape<K: Shape, Stride: StaticDim, Padding: StaticDim>: Shape {
    type Output: Shape;
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        kernel: &<K as Shape>::Field,
    ) -> <Self::Output as Shape>::Field;
}

// Fully static (B, C_in, H_in, W_in) with Kernel (C_out, C_in, K_h, K_w)
impl<B, CIn, HIn, WIn, COut, KH, KW, Stride, Padding>
    KernelConv2dShape<(COut, CIn, KH, KW), Stride, Padding> for (B, CIn, HIn, WIn)
where
    B: Dim + Default,
    CIn: StaticDim,
    HIn: StaticDim + core::ops::Add<Prod<U2, Padding>>,
    Sum<HIn, Prod<U2, Padding>>: core::ops::Sub<KH>,
    Diff<Sum<HIn, Prod<U2, Padding>>, KH>: core::ops::Div<Stride>,
    Quot<Diff<Sum<HIn, Prod<U2, Padding>>, KH>, Stride>: core::ops::Add<U1>,
    WIn: StaticDim + core::ops::Add<Prod<U2, Padding>>,
    Sum<WIn, Prod<U2, Padding>>: core::ops::Sub<KW>,
    Diff<Sum<WIn, Prod<U2, Padding>>, KW>: core::ops::Div<Stride>,
    Quot<Diff<Sum<WIn, Prod<U2, Padding>>, KW>, Stride>: core::ops::Add<U1>,
    COut: StaticDim,
    KH: StaticDim,
    KW: StaticDim,
    Stride: StaticDim,
    Padding: StaticDim,
    U2: core::ops::Mul<Padding>,
    ConvOutDim<HIn, KH, Stride, Padding>: StaticDim,
    ConvOutDim<WIn, KW, Stride, Padding>: StaticDim,
{
    type Output = (
        B,
        COut,
        ConvOutDim<HIn, KH, Stride, Padding>,
        ConvOutDim<WIn, KW, Stride, Padding>,
    );

    #[inline(always)]
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(COut, CIn, KH, KW) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}

// Dyn shapes compute output shape dynamically based on the inputs
impl<
    Stride: crate::tensor::matmul::StaticDim + typenum::Unsigned,
    Padding: crate::tensor::matmul::StaticDim + typenum::Unsigned,
> KernelConv2dShape<Dyn, Stride, Padding> for Dyn
{
    type Output = Dyn;
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        kernel: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        if lhs.len() != 4 || kernel.len() != 4 {
            return alloc::vec![];
        }
        let (n, _c_in, h_in, w_in) = (lhs[0], lhs[1], lhs[2], lhs[3]);
        let (c_out, _c_in_k, k_h, k_w) = (kernel[0], kernel[1], kernel[2], kernel[3]);

        let stride = Stride::USIZE;
        let padding = Padding::USIZE;

        let h_out = (h_in + 2 * padding - k_h) / stride + 1;
        let w_out = (w_in + 2 * padding - k_w) / stride + 1;

        alloc::vec![n, c_out, h_out, w_out]
    }
}

impl<S1: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad> Tensor<S1, B, K, D, G> {
    pub fn conv2d<Stride, Padding, KShape>(
        &self,
        weight: &Tensor<KShape, B, K, D, G>,
        bias: Option<&Tensor<Dyn, B, K, D, G>>, // Simplified bias for now
    ) -> Result<Tensor<S1::Output, B, K, D, G>>
    where
        Stride: StaticDim + typenum::Unsigned,
        Padding: StaticDim + typenum::Unsigned,
        KShape: Shape + DynShape,
        S1: KernelConv2dShape<KShape, Stride, Padding>,
    {
        let inner = B::conv2d::<K>(
            &self.inner,
            &weight.inner,
            bias.map(|b| b.inner()),
            <Stride as typenum::Unsigned>::USIZE,
            <Padding as typenum::Unsigned>::USIZE,
            1, // Default dilation
            1, // Default groups
        )?;

        let output_shape = S1::output_shape(&self._shape, &weight._shape);
        Ok(Tensor::from_parts_unchecked(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
