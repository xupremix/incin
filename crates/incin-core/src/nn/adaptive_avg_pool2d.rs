use crate::dist::placement::Local;
use crate::exec::catalog::{AdaptivePool2dAttributes, Descriptor, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::{Module, Parameters, TrainMode};
use crate::prelude::*;
use crate::tensor::backend::Execute;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// `AdaptiveAvgPool2d`.
pub struct AdaptiveAvgPool2d<HOut: Unsigned, WOut: Unsigned> {
    _phantom: core::marker::PhantomData<(HOut, WOut)>,
}

impl<HOut: Unsigned, WOut: Unsigned> Default for AdaptiveAvgPool2d<HOut, WOut> {
    fn default() -> Self {
        Self::new()
    }
}

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2d<HOut, WOut> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<HOut: Unsigned, WOut: Unsigned, B: Backend> Parameters<B> for AdaptiveAvgPool2d<HOut, WOut> {
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl<HOut: Unsigned, WOut: Unsigned> TrainMode for AdaptiveAvgPool2d<HOut, WOut> {}

impl<
    I: Shape + DynShape + crate::shapes::AdaptiveAvgPool2dShape<HOut, WOut>,
    HOut: Unsigned,
    WOut: Unsigned,
    B: Backend + crate::exec::Capabilities + Execute<op::AdaptiveAvgPool2dExact>,
> Module<Tensor<I, B>> for AdaptiveAvgPool2d<HOut, WOut>
where
    <B as Execute<op::AdaptiveAvgPool2dExact>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let input = TensorHandle::from_storage::<B, f32, Local>(x.inner());
        let context = ExecutionContext::from_scope(B::default());
        let shape = <I as crate::shapes::AdaptiveAvgPool2dShape<HOut, WOut>>::compute_output_shape(
            &x.shape_buf_value(),
        )?;
        let output_shape = crate::shapes::ShapeValue::<I::Output>::try_new(shape)
            .map_err(crate::prelude::Error::Shape)?;
        let out = dispatch::execute_shaped::<op::AdaptiveAvgPool2dExact, B, I::Output>(
            &context,
            AdaptivePool2dAttributes {
                output: [HOut::USIZE, WOut::USIZE],
            },
            &[input],
            &output_shape,
        )
        .map_err(crate::prelude::Error::from)?;

        Tensor::from_shape_value(
            out.into(),
            output_shape,
            x._dtype,
            x._device.clone(),
            core::marker::PhantomData,
        )
    }
}
