//! Conv2d shape verification

use crate::prelude::*;
use typenum::{Diff, Prod, Quot, Sum, U1, U2};

// ConvOutDim already defined in arithmetic.rs and exposed via prelude

/// Compile-time-checked `Tensor::conv2d` output shape rule, given
/// kernel shape `K` and compile-time-fixed `Stride`/`Padding`.
#[diagnostic::on_unimplemented(
    message = "Cannot apply a `{K}`-shaped kernel to input shape `{Self}`",
    label = "kernel/input shape mismatch",
    note = "the input's channel dimension must equal the kernel's input-channel dimension, and the input must be rank 4: (B, C, H, W)"
)]
pub trait KernelConv2dShape<K: Shape, Stride: StaticDim, Padding: StaticDim>: Shape {
    /// The convolved output shape.
    type Output: Shape;
    /// Computes the runtime `Field` of `Output` from the input and kernel fields.
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
    /// The convolved output shape: batch unchanged, channel dim
    /// replaced by `COut`, spatial dims via `ConvOutDim`.
    type Output = (
        B,
        COut,
        ConvOutDim<HIn, KH, Stride, Padding>,
        ConvOutDim<WIn, KW, Stride, Padding>,
    );

    #[inline(always)]
    /// `COut`/`HOut`/`WOut` come from `Default` — they're all `StaticDim`
    /// (typenum) here, so the default *is* the only possible value. `B`
    /// (the batch dim) is bounded only by `Dim + Default`, so a `usize` or
    /// `symbolic_dim!` batch needs its real value copied from `lhs`
    /// instead — `Default::default()` would silently produce 0.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(COut, CIn, KH, KW) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0,
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
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = Dyn;
    /// Computes the convolved output shape from the runtime input/kernel
    /// dims using the standard conv output-size formula.
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

impl<
    S1: Shape + DynShape,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S1, B, K, G>
{
    /// 2D convolution with compile-time-checked output shape (see
    /// `KernelConv2dShape`). Dilation and groups are fixed to 1.
    pub fn conv2d<Stride, Padding, KShape>(
        &self,
        weight: &Tensor<KShape, B, K, G>,
        bias: Option<&Tensor<Dyn, B, K, G>>, // Simplified bias for now
    ) -> Result<Tensor<S1::Output, B, K, G>>
    where
        Stride: StaticDim + typenum::Unsigned,
        Padding: StaticDim + typenum::Unsigned,
        KShape: Shape + DynShape,
        S1: KernelConv2dShape<KShape, Stride, Padding>,
    {
        let inner = self.under_grad_mode(|| {
            B::conv2d::<K>(
                &self.inner,
                &weight.inner,
                bias.map(|b| b.inner()),
                <Stride as typenum::Unsigned>::USIZE,
                <Padding as typenum::Unsigned>::USIZE,
                1, // Default dilation
                1, // Default groups
            )
        })?;

        let output_shape = S1::output_shape(&self._shape, &weight._shape);
        Tensor::from_parts(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
