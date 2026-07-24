use crate::nn::{Module, Parameters, TrainMode};
use crate::prelude::*;

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
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<I, B>> for AvgPool2d<K, S, P, D>
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out = B::avg_pool2d(
            x.inner(),
            (K::USIZE, K::USIZE),
            (S::USIZE, S::USIZE),
            (P::USIZE, P::USIZE),
        )?;

        let shape =
            <I as crate::shapes::Pool2dShape<K, S, P, D>>::compute_output_shape(x.shape_field());
        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
