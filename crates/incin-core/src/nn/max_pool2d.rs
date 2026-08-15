use crate::backend_authoring::{Descriptor, Execute};
use crate::exec::Capabilities;
use crate::nn::{Module, TrainMode};
use crate::prelude::{Backend, Device, DType, DynShape, Error, Result, Shape, Tensor};
use alloc::string::String;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// `MaxPool2d`.
pub struct MaxPool2d<K: Unsigned, S: Unsigned, P: Unsigned = typenum::U0, D: Unsigned = typenum::U1>
{
    _phantom: core::marker::PhantomData<(K, S, P, D)>,
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> MaxPool2d<K, S, P, D> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Result<Self> {
        Ok(Self {
            _phantom: core::marker::PhantomData,
        })
    }
}

/// Stateless — no training-dependent behavior, opts in with the trait's
/// default no-op so it can appear inside a `Sequential` alongside layers
/// that do have one (e.g. `Dropout`).
impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> TrainMode for MaxPool2d<K, S, P, D> {}

impl<
    I: Shape + DynShape + crate::shapes::Pool2dShape<K, S, P, D>,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    B: crate::tensor::backend::VariableBackend + Execute<crate::exec::catalog::op::MaxPool2d>,
> Module<Tensor<I, B>> for MaxPool2d<K, S, P, D>
where
    B: Capabilities,
    <B as Execute<crate::exec::catalog::op::MaxPool2d>>::Output: Into<B::Storage<f32>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        x.max_pool2d::<K, S, P, D>()
    }
}
