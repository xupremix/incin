//! Conv2d shape verification

use alloc::vec::Vec;

use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{Conv2dAttributes, Descriptor, op};
use crate::exec::request::TensorHandle;
use crate::shapes::Dyn;
use crate::shapes::{ConvOutDim, Dim, DimCons, DynShape, Nil, Shape, ShapeBuf, ShapeValue};
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::matmul::StaticDim;
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
    /// Computes the runtime `ShapeBuf` of `Output` from the input and kernel buffers.
    fn output_shape(lhs: &ShapeBuf, kernel: &ShapeBuf) -> ShapeBuf;
}

// Fully static (B, C_in, H_in, W_in) with Kernel (C_out, C_in, K_h, K_w)
impl<B, CIn, HIn, WIn, COut, KH, KW, Stride, Padding>
    KernelConv2dShape<DimCons<COut, DimCons<CIn, DimCons<KH, DimCons<KW, Nil>>>>, Stride, Padding>
    for DimCons<B, DimCons<CIn, DimCons<HIn, DimCons<WIn, Nil>>>>
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
    type Output = DimCons<
        B,
        DimCons<
            COut,
            DimCons<
                ConvOutDim<HIn, KH, Stride, Padding>,
                DimCons<ConvOutDim<WIn, KW, Stride, Padding>, Nil>,
            >,
        >,
    >;

    #[inline(always)]
    /// `COut`/`HOut`/`WOut` are statically proven extents. `B` is copied from
    /// `lhs` because it may be runtime-sized.
    fn output_shape(lhs: &ShapeBuf, _: &ShapeBuf) -> ShapeBuf {
        ShapeBuf::from_slice(&[
            lhs[0],
            COut::static_size().expect("StaticDim proves a concrete extent"),
            ConvOutDim::<HIn, KH, Stride, Padding>::static_size()
                .expect("StaticDim proves a concrete extent"),
            ConvOutDim::<WIn, KW, Stride, Padding>::static_size()
                .expect("StaticDim proves a concrete extent"),
        ])
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
    fn output_shape(lhs: &ShapeBuf, kernel: &ShapeBuf) -> ShapeBuf {
        if lhs.len() != 4 || kernel.len() != 4 {
            return ShapeBuf::SCALAR;
        }
        let (n, _c_in, h_in, w_in) = (lhs[0], lhs[1], lhs[2], lhs[3]);
        let (c_out, _c_in_k, k_h, k_w) = (kernel[0], kernel[1], kernel[2], kernel[3]);

        let stride = Stride::USIZE;
        let padding = Padding::USIZE;

        let h_out = (h_in + 2 * padding - k_h) / stride + 1;
        let w_out = (w_in + 2 * padding - k_w) / stride + 1;

        ShapeBuf::from_slice(&[n, c_out, h_out, w_out])
    }
}

impl<
    S1: Shape + DynShape,
    B: Backend + Execute<op::Conv2dExact>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S1, B, K, G>
where
    B: crate::exec::Capabilities,
    <B as Execute<op::Conv2dExact>>::Output: Into<B::Storage<K>>,
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
        let output_shape = S1::output_shape(&self.shape_buf_value(), &weight.shape_buf_value());
        let output_shape = ShapeValue::<S1::Output>::try_new(output_shape).map_err(Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let kernel = TensorHandle::from_storage::<B, K, Local>(&weight.inner);
        let mut inputs = Vec::with_capacity(if bias.is_some() { 3 } else { 2 });
        inputs.push(input);
        inputs.push(kernel);
        if let Some(bias) = bias {
            inputs.push(TensorHandle::from_storage::<B, K, Local>(&bias.inner));
        }
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                crate::exec::dispatch::execute_shaped::<op::Conv2dExact, B, S1::Output>(
                    &context,
                    Conv2dAttributes {
                        stride: [Stride::USIZE; 2],
                        padding: [Padding::USIZE; 2],
                        dilation: [1; 2],
                        groups: 1,
                        has_bias: bias.is_some(),
                    },
                    &inputs,
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_parts(
            inner,
            output_shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
