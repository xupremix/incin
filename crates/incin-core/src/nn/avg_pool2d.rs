use crate::backend_authoring::{Descriptor, Execute};
use crate::dist::placement::Local;
use crate::exec::Capabilities;
use crate::exec::catalog::{AvgPool2dAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::{Module, Parameters, TrainMode};
use crate::prelude::*;
use crate::shapes::ShapeValue;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// `AvgPool2d`.
pub struct AvgPool2d<K: Unsigned, S: Unsigned, P: Unsigned = typenum::U0, D: Unsigned = typenum::U1>
{
    _phantom: core::marker::PhantomData<(K, S, P, D)>,
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> AvgPool2d<K, S, P, D> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Result<Self> {
        Ok(Self {
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend> Parameters<B>
    for AvgPool2d<K, S, P, D>
{
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
impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> TrainMode for AvgPool2d<K, S, P, D> {}

impl<
    I: Shape + DynShape + crate::shapes::Pool2dShape<K, S, P, D>,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    B: Backend + Execute<op::AvgPool2d>,
> Module<Tensor<I, B>> for AvgPool2d<K, S, P, D>
where
    B: Capabilities,
    <B as Execute<op::AvgPool2d>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let shape = <I as crate::shapes::Pool2dShape<K, S, P, D>>::compute_output_shape(
            &x.shape_buf_value(),
        )?;
        let shape = ShapeValue::<I::Output>::try_new(shape).map_err(Error::Shape)?;
        let inputs = [TensorHandle::from_storage::<B, f32, Local>(x.inner())];
        let context = ExecutionContext::from_scope(B::default());
        let out = dispatch::execute_shaped::<op::AvgPool2d, B, I::Output>(
            &context,
            AvgPool2dAttributes {
                kernel: [K::USIZE; 2],
                stride: [S::USIZE; 2],
                padding: [P::USIZE; 2],
            },
            &inputs,
            &shape,
        )?
        .into();
        Tensor::from_parts(
            out,
            shape.shape_buf().clone(),
            x._dtype,
            x._device.clone(),
            core::marker::PhantomData,
        )
    }
}
