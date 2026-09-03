//! Tensor axis permutation and transposition operations.

use crate::backend_authoring::{Backend, Execute};
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::{TransposeAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::idx::StaticCursor;
use crate::shapes::{DynShape, Shape, ShapeBuf, ShapeValue, SwapAxes};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::ops::manipulation::selectors::AxisPairSelector;

impl<
    S: Shape + DynShape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    TLayout: Layout,
> Tensor<S, B, K, G, Local, TLayout>
{
    /// Transposes two axis selectors while preserving the strongest available
    /// output shape proof.
    #[allow(clippy::type_complexity)]
    pub fn transpose<Lx, Rx>(
        &self,
        left: Lx,
        right: Rx,
    ) -> Result<Tensor<<() as AxisPairSelector<S, Lx, Rx>>::Output, B, K, G>>
    where
        (): AxisPairSelector<S, Lx, Rx>,
        B: Execute<op::TransposeExact> + Capabilities,
        <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    {
        let (first, second) =
            <() as AxisPairSelector<S, Lx, Rx>>::resolve(&(left, right), self.shape_buf().rank())?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(first, second);
        let output_shape = ShapeValue::<<() as AxisPairSelector<S, Lx, Rx>>::Output>::try_new(
            ShapeBuf::from_slice(&out_dims),
        )
        .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad).restrict(|| {
            dispatch::execute_shaped::<
                op::TransposeExact,
                B,
                <() as AxisPairSelector<S, Lx, Rx>>::Output,
            >(
                &context,
                TransposeAttributes { first, second },
                &[input],
                &output_shape,
            )
        })?;
        Tensor::from_shape_value(
            inner.into(),
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Transposes two axes without copying, when the backend can.
    ///
    /// The counterpart to [`transpose_structural`](Self::transpose_structural),
    /// which materialises: this permutes the shape and strides over the same
    /// buffer and does no work on the device.
    ///
    /// Neither is universally faster, which is why both exist. Measured on a
    /// GTX 1650 for a transpose followed by pointwise consumption, the view
    /// beats the copy by roughly 45% when the result is read once and loses by
    /// roughly 23% when it is read eight times, crossing over at about four
    /// reads. That depends on the *consumer*, which a transpose cannot know, so
    /// the choice is the caller's.
    ///
    /// The result is non-contiguous, and its layout says so: it carries
    /// [`Unknown`](crate::shapes::Unknown), so an operation bounded on
    /// `L: Contiguous` -- `reshape_view`, for instance -- will not accept it
    /// without a fresh proof.
    ///
    /// Not every backend offers this. WGPU's pointwise shaders address linearly
    /// and would read a view's elements in the wrong order, so it does not
    /// advertise the operation and this method will not resolve for it.
    ///
    /// # Errors
    ///
    /// Returns an error when the axes are out of range for the runtime rank, or
    /// when the backend refuses the operation.
    #[allow(clippy::type_complexity)]
    pub fn transpose_view<Lx, Rx>(&self) -> Result<Tensor<<S as SwapAxes<Lx, Rx>>::Output, B, K, G>>
    where
        Lx: StaticCursor,
        Rx: StaticCursor,
        S: SwapAxes<Lx, Rx>,
        <S as SwapAxes<Lx, Rx>>::Output: Shape + DynShape,
        B: Execute<op::TransposeView> + Capabilities,
        <B as Execute<op::TransposeView>>::Output: Into<B::Storage<K>>,
    {
        let axes = crate::shapes::idx::AxisSelector::new(&[Lx::INDEX, Rx::INDEX])
            .normalize(self.shape_buf().rank())?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(axes[0], axes[1]);
        let output_shape =
            ShapeValue::<<S as SwapAxes<Lx, Rx>>::Output>::try_new(ShapeBuf::from_slice(&out_dims))
                .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::TransposeView, B, <S as SwapAxes<Lx, Rx>>::Output>(
                    &context,
                    TransposeAttributes {
                        first: axes[0],
                        second: axes[1],
                    },
                    &[input],
                    &output_shape,
                )
            })
            .map_err(crate::err::Error::from)?;
        Tensor::from_shape_value(
            inner.into(),
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Advanced structural transpose retained for shape-proof internals.
    #[allow(clippy::type_complexity)]
    pub fn transpose_structural<Lx, Rx>(
        &self,
    ) -> Result<Tensor<<S as SwapAxes<Lx, Rx>>::Output, B, K, G>>
    where
        Lx: StaticCursor,
        Rx: StaticCursor,
        S: SwapAxes<Lx, Rx>,
        <S as SwapAxes<Lx, Rx>>::Output: Shape + DynShape,
        B: Execute<op::TransposeExact> + Capabilities,
        <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    {
        let axes = crate::shapes::idx::AxisSelector::new(&[Lx::INDEX, Rx::INDEX])
            .normalize(self.shape_buf().rank())?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(axes[0], axes[1]);
        let output_shape =
            ShapeValue::<<S as SwapAxes<Lx, Rx>>::Output>::try_new(ShapeBuf::from_slice(&out_dims))
                .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::TransposeExact, B, <S as SwapAxes<Lx, Rx>>::Output>(
                    &context,
                    TransposeAttributes {
                        first: axes[0],
                        second: axes[1],
                    },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::<<S as SwapAxes<Lx, Rx>>::Output, B, K, G>::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Runtime-selector transpose. Known input rank is preserved in the
    /// result; a fully dynamic input remains fully dynamic.
    #[doc(hidden)]
    pub fn transpose_runtime(&self, left: isize, right: isize) -> Result<Tensor<S::Keep, B, K, G>>
    where
        S: crate::shapes::RuntimeRankProjection,
        S::Keep: Shape + DynShape,
        B: Execute<op::TransposeExact> + Capabilities,
        <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    {
        let (first, second) = <() as AxisPairSelector<S, isize, isize>>::resolve(
            &(left, right),
            self.shape_buf().rank(),
        )?;
        let mut out_dims = self.shape_buf().as_ref().to_vec();
        out_dims.swap(first, second);
        let output_shape = ShapeValue::<S::Keep>::try_new(ShapeBuf::from_slice(&out_dims))
            .map_err(crate::err::Error::Shape)?;
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::TransposeExact, B, S::Keep>(
                    &context,
                    TransposeAttributes { first, second },
                    &[input],
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
