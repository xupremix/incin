use core::ops::{Add, Div, Mul, Sub};
use typenum::{U1, U2, UInt, UTerm};

/// Compile-time formula for a single spatial dimension's conv/pool
/// output size: `(in + 2*Padding - Dilation*(Kernel-1) - 1) / Stride + 1`.
pub trait SpatialOut<Kernel, Stride, Padding, Dilation> {
    /// The output size for this input dimension.
    type Output;
}

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for UTerm
where
    Padding: Mul<U2>,
    Kernel: Sub<U1>,
    Dilation: Mul<<Kernel as Sub<U1>>::Output>,
    UTerm: Add<<Padding as Mul<U2>>::Output>,
    <UTerm as Add<<Padding as Mul<U2>>::Output>>::Output:
        Sub<<Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output>,
    <<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output: Sub<U1>,
    <<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output: Div<Stride>,
    <<<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output: Add<U1>,
{
    /// The output size, computed via the conv/pool output-size formula.
    type Output = <<<<<UTerm as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output as Add<U1>>::Output;
}

impl<U, B, Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation>
    for UInt<U, B>
where
    Padding: Mul<U2>,
    Kernel: Sub<U1>,
    Dilation: Mul<<Kernel as Sub<U1>>::Output>,
    UInt<U, B>: Add<<Padding as Mul<U2>>::Output>,
    <UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output:
        Sub<<Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output>,
    <<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output: Sub<U1>,
    <<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output: Div<Stride>,
    <<<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output: Add<U1>,
{
    /// The output size, computed via the conv/pool output-size formula.
    type Output = <<<<<UInt<U, B> as Add<<Padding as Mul<U2>>::Output>>::Output as Sub<
        <Dilation as Mul<<Kernel as Sub<U1>>::Output>>::Output,
    >>::Output as Sub<U1>>::Output as Div<Stride>>::Output as Add<U1>>::Output;
}

impl<Kernel, Stride, Padding, Dilation> SpatialOut<Kernel, Stride, Padding, Dilation> for usize {
    /// Runtime dimension in, runtime dimension out.
    type Output = usize;
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 2D pooling to shape `{Self}`",
    label = "Invalid shape for 2D pooling",
    note = "Pool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked 2D pooling (`max_pool2d`/`avg_pool2d`) output
/// shape rule, given kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait Pool2dShape<K, S, P, D>: crate::prelude::Shape {
    /// The pooled output shape (same batch/channel dims, spatial dims
    /// reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field;
}

use crate::prelude::{Dim, Dyn};
use typenum::Unsigned;

// Implement for (B, C, H, W) -> (B, C, HOut, WOut)
impl<B: Dim, C: Dim, HIn: Dim, WIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
    Pool2dShape<K, S, P, D> for (B, C, HIn, WIn)
where
    HIn: SpatialOut<K, S, P, D>,
    WIn: SpatialOut<K, S, P, D>,
    <HIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
    <WIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
{
    /// The resulting shape per this trait\'s rule.
    type Output = (
        B,
        C,
        <HIn as SpatialOut<K, S, P, D>>::Output,
        <WIn as SpatialOut<K, S, P, D>>::Output,
    );
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field {
        (input.0, input.1, Default::default(), Default::default())
    }
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> Pool2dShape<K, S, P, D> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field {
        let mut dims = input.clone();
        if dims.len() == 4 {
            dims[2] = (dims[2] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
            dims[3] = (dims[3] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
        }
        dims
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 1D convolution to shape `{Self}`",
    label = "Invalid shape for 1D convolution",
    note = "Conv1D requires a 2D or 3D tensor (C, L) or (B, C, L)"
)]
/// Compile-time-checked `conv1d` output shape rule, given output
/// channels `COut`, kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait SpatialConv1d<COut, K, S, P, D>: crate::prelude::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, length dim reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> <Self::Output as crate::prelude::Shape>::Field;
}

macro_rules! impl_conv1d_shape {
    ($($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* CIn: Dim, COut: Dim + Default, LIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
            SpatialConv1d<COut, K, S, P, D> for ($($B,)* CIn, LIn)
        where
            LIn: SpatialOut<K, S, P, D>,
            <LIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = ($($B,)* COut, <LIn as SpatialOut<K, S, P, D>>::Output);
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            fn compute_output_shape(input: &Self::Field, _out_channels: usize) -> <Self::Output as crate::prelude::Shape>::Field {
                ($(input.$idx.clone(),)* Default::default(), Default::default())
            }
        }
    };
}

impl_conv1d_shape!(B0: 0);
impl_conv1d_shape!(B0: 0, B1: 1);
impl_conv1d_shape!(B0: 0, B1: 1, B2: 2);
impl_conv1d_shape!(B0: 0, B1: 1, B2: 2, B3: 3);
impl_conv1d_shape!(B0: 0, B1: 1, B2: 2, B3: 3, B4: 4);

impl<COut, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> SpatialConv1d<COut, K, S, P, D>
    for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> <Self::Output as crate::prelude::Shape>::Field {
        let mut dims = input.clone();
        if dims.len() == 3 {
            dims[1] = out_channels;
            dims[2] = (dims[2] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
        }
        dims
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply 2D convolution to shape `{Self}`",
    label = "Invalid shape for 2D convolution",
    note = "Conv2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked `conv2d` output shape rule, given output
/// channels `COut`, kernel `K`, stride `S`, padding `P`, and dilation `D`.
pub trait SpatialConv2d<COut, K, S, P, D>: crate::prelude::Shape {
    /// The convolved output shape (batch dims unchanged, channel dim
    /// replaced by `COut`, spatial dims reduced via `SpatialOut`).
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> <Self::Output as crate::prelude::Shape>::Field;
}

macro_rules! impl_conv2d_shape {
    ($($B:ident : $idx:tt),*) => {
        impl<$($B: Dim,)* CIn: Dim, COut: Dim + Default, HIn: Dim, WIn: Dim, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned>
            SpatialConv2d<COut, K, S, P, D> for ($($B,)* CIn, HIn, WIn)
        where
            HIn: SpatialOut<K, S, P, D>,
            WIn: SpatialOut<K, S, P, D>,
            <HIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
            <WIn as SpatialOut<K, S, P, D>>::Output: Dim + Default,
        {
            /// The resulting shape per this trait\'s rule.
            type Output = ($($B,)* COut, <HIn as SpatialOut<K, S, P, D>>::Output, <WIn as SpatialOut<K, S, P, D>>::Output);
            /// Computes the runtime `Field` of `Output` from the input\'s own field.
            fn compute_output_shape(input: &Self::Field, _out_channels: usize) -> <Self::Output as crate::prelude::Shape>::Field {
                ($(input.$idx.clone(),)* Default::default(), Default::default(), Default::default())
            }
        }
    };
}

impl_conv2d_shape!(B0: 0);
impl_conv2d_shape!(B0: 0, B1: 1);
impl_conv2d_shape!(B0: 0, B1: 1, B2: 2);
impl_conv2d_shape!(B0: 0, B1: 1, B2: 2, B3: 3);

impl<COut, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> SpatialConv2d<COut, K, S, P, D>
    for Dyn
{
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(
        input: &Self::Field,
        out_channels: usize,
    ) -> <Self::Output as crate::prelude::Shape>::Field {
        let mut dims = input.clone();
        if dims.len() == 4 {
            dims[1] = out_channels;
            dims[2] = (dims[2] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
            dims[3] = (dims[3] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
        }
        dims
    }
}

#[diagnostic::on_unimplemented(
    message = "Cannot apply adaptive 2D pooling to shape `{Self}`",
    label = "Invalid shape for adaptive 2D pooling",
    note = "AdaptiveAvgPool2D requires a 3D or 4D tensor (C, H, W) or (B, C, H, W)"
)]
/// Compile-time-checked `adaptive_avg_pool2d` output shape rule: the
/// output spatial size is exactly `(HOut, WOut)` regardless of input size.
pub trait AdaptiveAvgPool2dShape<HOut, WOut>: crate::prelude::Shape {
    /// The pooled output shape: batch/channel dims unchanged, spatial dims
    /// fixed to `(HOut, WOut)`.
    type Output: crate::prelude::Shape + crate::prelude::DynShape;
    /// Computes the runtime `Field` of `Output` from the input's own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field;
}

impl<B: Dim, C: Dim, HIn: Dim, WIn: Dim, HOut: Unsigned, WOut: Unsigned>
    AdaptiveAvgPool2dShape<HOut, WOut> for (B, C, HIn, WIn)
where
    HOut: Dim + Default,
    WOut: Dim + Default,
{
    /// The resulting shape per this trait\'s rule.
    type Output = (B, C, HOut, WOut);
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field {
        (input.0, input.1, Default::default(), Default::default())
    }
}

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2dShape<HOut, WOut> for Dyn {
    /// The resulting shape per this trait\'s rule.
    type Output = Dyn;
    /// Computes the runtime `Field` of `Output` from the input\'s own field.
    fn compute_output_shape(input: &Self::Field) -> <Self::Output as crate::prelude::Shape>::Field {
        let mut dims = input.clone();
        if dims.len() == 4 {
            dims[2] = HOut::USIZE;
            dims[3] = WOut::USIZE;
        }
        dims
    }
}
