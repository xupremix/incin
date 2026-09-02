//! Computer vision and spatial restructuring operations.

use crate::backend_authoring::{Backend, Execute};
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::{
    EpsilonAttributes, GroupNormAttributes, PixelShuffleAttributes, Pool2dAttributes, op,
};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::error::OperationKind;
use crate::shapes::{Dyn, DynShape, Shape, ShapeBuf, ShapeValue};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use alloc::vec;

impl<
    S: Shape + DynShape,
    B: Backend + Execute<op::MaxPool2d>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    L: Layout,
> Tensor<S, B, K, G, Local, L>
where
    B: Capabilities,
    <B as Execute<op::MaxPool2d>>::Output: Into<B::Storage<K>>,
{
    /// Functional `max_pool2d` operation.
    #[allow(clippy::type_complexity)]
    pub fn max_pool2d<KShape, SShape, Pool, Dilation>(
        &self,
    ) -> Result<
        Tensor<<S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output, B, K, G>,
    >
    where
        KShape: typenum::Unsigned,
        SShape: typenum::Unsigned,
        Pool: typenum::Unsigned,
        Dilation: typenum::Unsigned,
        S: crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>,
        <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output: Shape,
    {
        let shape =
            <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::compute_output_shape(
                &self.shape_buf_value(),
            )?;
        let shape = ShapeValue::<
            <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output,
        >::try_new(shape)
        .map_err(crate::err::Error::Shape)?;
        let inputs = [TensorHandle::from_storage::<B, K, Local>(&self.inner)];
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let out = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<
                    op::MaxPool2d,
                    B,
                    <S as crate::shapes::Pool2dShape<KShape, SShape, Pool, Dilation>>::Output,
                >(
                    &context,
                    Pool2dAttributes {
                        kernel: [KShape::USIZE; 2],
                        stride: [SShape::USIZE; 2],
                        padding: [Pool::USIZE; 2],
                        dilation: [Dilation::USIZE; 2],
                    },
                    &inputs,
                    &shape,
                )
            })?
            .into();
        Tensor::from_parts(
            out,
            shape.shape_buf().clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, L: Layout>
    Tensor<S, B, K, G, Local, L>
{
    /// Rearranges elements in a 4D tensor of shape (N, C, H, W) to (N, C / r^2, H * r, W * r).
    pub fn pixel_shuffle(&self, upscale_factor: usize) -> Result<Tensor<Dyn, B, K, G>>
    where
        B: Capabilities + Execute<op::PixelShuffle>,
        <B as Execute<op::PixelShuffle>>::Output: Into<B::Storage<K>>,
    {
        let dims = self.shape_buf();
        if dims.as_ref().len() != 4 || upscale_factor == 0 {
            return Err(crate::err::Error::Msg(
                "pixel_shuffle expects a rank-4 tensor and a positive upscale factor".into(),
            ));
        }
        let factor = upscale_factor.checked_mul(upscale_factor).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::PixelShuffle,
                expression: "upscale factor squared",
            })
        })?;
        if dims.as_ref()[1] % factor != 0 {
            return Err(crate::err::Error::Msg(
                "pixel_shuffle channels must be divisible by upscale factor squared".into(),
            ));
        }
        let out_shape = vec![
            dims.as_ref()[0],
            dims.as_ref()[1] / factor,
            dims.as_ref()[2]
                .checked_mul(upscale_factor)
                .ok_or_else(|| {
                    crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                        operation: OperationKind::PixelShuffle,
                        expression: "height times upscale factor",
                    })
                })?,
            dims.as_ref()[3]
                .checked_mul(upscale_factor)
                .ok_or_else(|| {
                    crate::err::Error::Shape(crate::shapes::ShapeError::ArithmeticOverflow {
                        operation: OperationKind::PixelShuffle,
                        expression: "width times upscale factor",
                    })
                })?,
        ];
        let output_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&out_shape))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::PixelShuffle, B, Dyn>(
                    &context,
                    PixelShuffleAttributes { upscale_factor },
                    &[input],
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

    /// Group normalization across `groups`.
    pub fn group_norm(&self, groups: usize, eps: f64) -> Result<Self>
    where
        B: Execute<op::GroupNorm> + Capabilities,
        <B as Execute<op::GroupNorm>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::GroupNorm, B, S>(
                    &context,
                    GroupNormAttributes {
                        groups,
                        epsilon: eps,
                    },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Instance normalization for 4D (N, C, H, W) tensors.
    pub fn instance_norm(&self, eps: f64) -> Result<Self>
    where
        B: Execute<op::InstanceNorm> + Capabilities,
        <B as Execute<op::InstanceNorm>>::Output: Into<B::Storage<K>>,
    {
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::InstanceNorm, B, S>(
                    &context,
                    EpsilonAttributes { epsilon: eps },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
