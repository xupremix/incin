//! The position-wise feed-forward half of a transformer layer.
//!
//! [`FeedForward`] is the second sub-layer of every transformer block: two
//! projections with a nonlinearity between them, applied to each position
//! independently. It is a module rather than an entry in
//! [`activation`](crate::nn::activation) because one of the three variants
//! cannot be an activation at all.
//!
//! # Why SwiGLU forces this to be a module
//!
//! `gelu` and `relu` are pointwise functions of one tensor, so a feed-forward
//! built on either is `down(act(up(x)))` and the activation is genuinely a
//! function. SwiGLU is not: it needs a *second* projection of the input,
//! `down(swish(gate(x)) * up(x))`, so the gate is a parameter matrix and the
//! nonlinearity is a product of two different projections of `x`. That cannot
//! be expressed as a unary row in the operation catalog, which is why
//! [`FeedForwardKind`] selects between shapes of module rather than between
//! activation functions, and why `gate` is an `Option` field rather than a
//! third always-present projection.

use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::op;
use crate::nn::attention::invalid;
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Linear, Module};
use crate::shapes::{Dyn, Layout};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, RequiresGrad};

/// Which feed-forward shape a transformer layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardKind {
    /// `down(relu(up(x)))`, the original formulation.
    Relu,
    /// `down(gelu(up(x)))`, the variant most encoder stacks use.
    Gelu,
    /// `down(swish(gate(x)) * up(x))`.
    ///
    /// Costs a third parameter matrix, so a layer that wants the same
    /// parameter count as a `Gelu` layer of width `d_ff` asks for roughly
    /// `2 * d_ff / 3` instead. Nothing here scales `d_ff` on the caller's
    /// behalf: the width that was asked for is the width that is built.
    SwiGlu,
}

impl FeedForwardKind {
    /// Whether this kind needs the gate projection.
    #[must_use]
    pub const fn is_gated(self) -> bool {
        matches!(self, Self::SwiGlu)
    }
}

/// The operations a feed-forward needs beyond its projections.
///
/// Stated once as a trait alias for the same reason as
/// [`AttentionBackend`](crate::nn::AttentionBackend): the `Module` impl would
/// otherwise open with a bound list in which the reader cannot tell which
/// entries belong to the feed-forward and which to [`Linear`].
pub trait FeedForwardBackend<K: DType>:
    crate::tensor::backend::VariableBackend
    + crate::exec::Capabilities
    + Execute<op::MatMulExact>
    + Execute<op::TransposeExact>
    + Execute<op::Add>
    + Execute<op::Relu>
    + Execute<op::Gelu>
    + Execute<op::Swish>
    + Execute<op::Mul>
{
}

impl<K: DType, B> FeedForwardBackend<K> for B where
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::Add>
        + Execute<op::Relu>
        + Execute<op::Gelu>
        + Execute<op::Swish>
        + Execute<op::Mul>
{
}

/// A position-wise feed-forward network over a `[.., d_model]` input.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::nn::{FeedForward, FeedForwardKind, Module};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// let mlp = FeedForward::<Cpu>::build(64, 256, FeedForwardKind::SwiGlu, (), ())?;
/// assert!(mlp.gate.is_some());
///
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![2, 8, 64])?.require_grad();
/// assert_eq!(mlp.forward(x)?.dims().dims(), &[2, 8, 64]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_stats, no_train_mode)]
pub struct FeedForward<
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// Projection into the hidden width.
    pub up: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    /// Gate projection, present only for [`FeedForwardKind::SwiGlu`].
    pub gate: Option<Linear<Dyn, B, crate::nn::optional::True, K, Train>>,
    /// Projection back to the model width.
    pub down: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    #[module(ignore)]
    /// Which of the three shapes this module is.
    pub kind: FeedForwardKind,
}

impl<B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState> crate::nn::TrainMode
    for FeedForward<B, K, Train>
{
    /// Visits the projections although none of them has train-mode behaviour,
    /// so this stays correct if one ever gains some.
    ///
    /// Written by hand because the derive is turned off for this struct: the
    /// generated traversal cannot see through the `Option` on `gate`.
    fn set_training(&mut self, training: bool) {
        crate::nn::TrainMode::set_training(&mut self.up, training);
        if let Some(gate) = self.gate.as_mut() {
            crate::nn::TrainMode::set_training(gate, training);
        }
        crate::nn::TrainMode::set_training(&mut self.down, training);
    }
}

impl<B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    FeedForward<B, K, Train>
{
    /// Freezes every projection.
    pub fn freeze(self) -> FeedForward<B, K, Frozen> {
        FeedForward {
            up: self.up.freeze(),
            gate: self.gate.map(Linear::freeze),
            down: self.down.freeze(),
            kind: self.kind,
        }
    }

    /// Unfreezes every projection.
    pub fn unfreeze(self) -> FeedForward<B, K, Trainable> {
        FeedForward {
            up: self.up.unfreeze(),
            gate: self.gate.map(Linear::unfreeze),
            down: self.down.unfreeze(),
            kind: self.kind,
        }
    }
}

impl<B, K> FeedForward<B, K, Trainable>
where
    B: crate::tensor::backend::TensorBackend<K> + crate::nn::param::ParameterInit<K>,
    K: DType,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds the projections for `kind`.
    ///
    /// The gate is built only for a gated kind, so an ungated module carries
    /// no unused parameter matrix and `collect_state` reports exactly the
    /// tensors the forward pass reads.
    pub fn build(
        d_model: usize,
        d_ff: usize,
        kind: FeedForwardKind,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        if d_model == 0 || d_ff == 0 {
            return Err(invalid(
                "build feed-forward",
                "d_model and d_ff must both be nonzero",
            ));
        }
        let up = Linear::build_full(d_model, d_ff, dtype.clone(), device.clone(), ())?;
        let gate = if kind.is_gated() {
            Some(Linear::build_full(
                d_model,
                d_ff,
                dtype.clone(),
                device.clone(),
                (),
            )?)
        } else {
            None
        };
        let down = Linear::build_full(d_ff, d_model, dtype, device, ())?;
        Ok(Self {
            up,
            gate,
            down,
            kind,
        })
    }
}

impl<B, K, Train, G, L> Module<Tensor<Dyn, B, K, G, Local, L>> for FeedForward<B, K, Train>
where
    B: FeedForwardBackend<K> + crate::tensor::backend::SupportsDType<K>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad, Output = JoinedGrad<G, Train::TensorGrad>>
        + GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Gelu>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Swish>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
{
    /// `Dyn`: the result is the down projection's, which makes no layout
    /// claim of its own.
    type Output = Tensor<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>, Local>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let hidden = match self.kind {
            FeedForwardKind::Relu => self.up.forward(x)?.relu()?.forget_layout(),
            FeedForwardKind::Gelu => self.up.forward(x)?.gelu()?.forget_layout(),
            FeedForwardKind::SwiGlu => {
                let gate = self.gate.as_ref().ok_or_else(|| {
                    invalid(
                        "feed-forward forward",
                        "a gated feed-forward is missing its gate projection; \
                         the module was constructed inconsistently",
                    )
                })?;
                let gated = gate.forward(x.clone())?.swish()?.forget_layout();
                let projected = self.up.forward(x)?.forget_layout();
                gated.broadcast_mul(&projected)?.forget_layout()
            }
        };
        Ok(self.down.forward(hidden)?.forget_layout())
    }
}
