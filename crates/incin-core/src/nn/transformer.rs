//! Transformer layers: attention and feed-forward with residuals and norms.
//!
//! [`TransformerEncoderLayer`] and [`TransformerDecoderLayer`] are the two
//! names this module exists to provide. Both are [`TransformerLayer`] with a
//! different direction marker, which is what lets one dataflow serve both
//! without either becoming a runtime flag on the other.
//!
//! # Why the direction is a type parameter
//!
//! An encoder layer and a decoder-only layer differ in exactly one thing:
//! whether a position may attend to positions after it. Writing two structs
//! would make that one bit cost two copies of the same forward pass, which is
//! the argument [`MultiHeadAttention`] already makes for grouped-query
//! attention being a parameter rather than a module. Writing one struct with a
//! `causal: bool` field would make it a value, so nothing could state in a
//! signature that it takes a causal layer.
//!
//! A marker type gives both: [`Causal`] and [`Bidirectional`] select the mask
//! at construction, the two aliases are distinct types that cannot be
//! substituted for one another, and there is one `forward`. The direction is
//! the authority on masking, so [`build`](TransformerLayer::build) overwrites
//! [`AttentionConfig::causal`] with `D::CAUSAL` and stores the corrected
//! config, which means `layer.config.attention.causal` reads back what the
//! layer actually does rather than what was passed in.
//!
//! # What is not here
//!
//! **Cross-attention.** A full encoder-decoder layer attends to an encoder's
//! output as well as its own input, so its forward pass takes two tensors.
//! [`Module`] is parameterized by a single input, and the GPT-style stack this
//! module is for is `Sequential<[TransformerDecoderLayer<..>; N]>`, which
//! requires a single-input module to compose at all. A cross-attending layer
//! is therefore a separate module with a tuple input, not a configuration of
//! this one.
//!
//! **Static shapes.** Like [`MultiHeadAttention`], these layers are written
//! against [`Dyn`](crate::shapes::Dyn), because the causal mask needs the mask
//! and the score tensor to meet and the typed API cannot express that shape
//! pairing yet.

use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::op;
use crate::nn::attention::{AttentionBackend, apply_dropout, invalid_owned};
use crate::nn::feed_forward::FeedForwardBackend;
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{
    AttentionConfig, Dropout, FeedForward, FeedForwardKind, LayerNorm, Module, MultiHeadAttention,
};
use crate::shapes::{Dyn, Layout};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, NoGrad, RequiresGrad};

/// Whether a layer's self-attention is masked.
///
/// Implemented only by [`Causal`] and [`Bidirectional`]; it carries a constant
/// rather than behaviour, so a layer's masking is fixed when its type is.
pub trait AttentionDirection: 'static {
    /// Whether a position may attend only to itself and earlier positions.
    const CAUSAL: bool;
    /// The direction's name, for error messages and layer summaries.
    const NAME: &'static str;
}

/// Every position attends to every other: an encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bidirectional;

/// A position attends only to itself and earlier positions: a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Causal;

impl AttentionDirection for Bidirectional {
    const CAUSAL: bool = false;
    const NAME: &'static str = "bidirectional";
}

impl AttentionDirection for Causal {
    const CAUSAL: bool = true;
    const NAME: &'static str = "causal";
}

/// Where the normalization sits relative to each residual branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPlacement {
    /// `x + sublayer(norm(x))`.
    ///
    /// The residual path from input to output carries no normalization, which
    /// is what makes deep stacks trainable without a warmup schedule. This is
    /// what current decoder-only models use.
    Pre,
    /// `norm(x + sublayer(x))`.
    ///
    /// The original formulation. Kept because pre-norm and post-norm have
    /// different training dynamics, so a library that hardcodes one cannot
    /// reproduce a paper that used the other.
    Post,
}

/// Configuration for a [`TransformerLayer`].
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    /// Where each sub-layer's normalization sits.
    pub norm: NormPlacement,
    /// Which feed-forward shape the second sub-layer is.
    pub feed_forward: FeedForwardKind,
    /// Dropout applied to each sub-layer's output before the residual add.
    ///
    /// Distinct from [`AttentionConfig::dropout`], which is dropout on the
    /// attention weights. A layer can have either, both, or neither.
    pub dropout: f32,
    /// Epsilon inside both normalizations.
    pub eps: f32,
    /// How attention itself is configured.
    ///
    /// Its `causal` field is overwritten by the layer's direction marker, so
    /// setting it here has no effect.
    pub attention: AttentionConfig,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            norm: NormPlacement::Pre,
            feed_forward: FeedForwardKind::Gelu,
            dropout: 0.0,
            eps: 1e-5,
            attention: AttentionConfig::default(),
        }
    }
}

impl TransformerConfig {
    /// Selects the feed-forward shape.
    #[must_use]
    pub const fn with_feed_forward(mut self, kind: FeedForwardKind) -> Self {
        self.feed_forward = kind;
        self
    }

    /// Selects the normalization placement.
    #[must_use]
    pub const fn with_norm(mut self, placement: NormPlacement) -> Self {
        self.norm = placement;
        self
    }

    /// Sets the residual dropout probability.
    #[must_use]
    pub const fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// Replaces the attention configuration.
    ///
    /// Its `causal` field is still taken from the layer's direction.
    #[must_use]
    pub const fn with_attention(mut self, attention: AttentionConfig) -> Self {
        self.attention = attention;
        self
    }
}

/// A transformer layer: self-attention and a feed-forward, each behind a
/// residual connection and a normalization.
///
/// Reach for [`TransformerEncoderLayer`] or [`TransformerDecoderLayer`] rather
/// than naming the direction parameter by hand.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::nn::{Module, TransformerConfig, TransformerDecoderLayer};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// // Eight query heads over two key/value heads, pre-norm, GELU feed-forward.
/// let layer =
///     TransformerDecoderLayer::<Cpu>::build(64, 8, 2, 256, TransformerConfig::default(), (), ())?;
/// assert!(layer.config.attention.causal);
///
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![2, 16, 64])?.require_grad();
/// assert_eq!(layer.forward(x)?.dims().dims(), &[2, 16, 64]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[incin_macros::module(internal, no_stats, no_train_mode)]
pub struct TransformerLayer<
    D: AttentionDirection,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// Self-attention, masked or not according to `D`.
    pub attention: MultiHeadAttention<B, K, Train>,
    /// Normalization for the attention sub-layer.
    pub attention_norm: LayerNorm<Dyn, B, K, Train>,
    /// The position-wise feed-forward.
    pub feed_forward: FeedForward<B, K, Train>,
    /// Normalization for the feed-forward sub-layer.
    pub feed_forward_norm: LayerNorm<Dyn, B, K, Train>,
    #[module(ignore)]
    /// Dropout applied to each sub-layer's output before its residual add.
    ///
    /// Ignored by the derived traversal for the same reason as
    /// [`MultiHeadAttention`]'s: `Dropout` is not parameterized by the
    /// backend, so a generated `ToDevice` call could not infer one for it.
    pub dropout: Dropout,
    #[module(ignore)]
    /// The configuration the layer was built with, with `causal` corrected to
    /// the direction marker.
    pub config: TransformerConfig,
    #[module(ignore)]
    _direction: core::marker::PhantomData<D>,
}

/// A layer whose positions all see each other.
pub type TransformerEncoderLayer<B, K = f32, Train = Trainable> =
    TransformerLayer<Bidirectional, B, K, Train>;

/// A layer whose positions see only themselves and their predecessors.
pub type TransformerDecoderLayer<B, K = f32, Train = Trainable> =
    TransformerLayer<Causal, B, K, Train>;

impl<D, B, K, Train> crate::nn::TrainMode for TransformerLayer<D, B, K, Train>
where
    D: AttentionDirection,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
{
    /// Propagates the mode into both sub-layers and this layer's own dropout.
    ///
    /// Written by hand because `dropout` is skipped by the derived traversal.
    fn set_training(&mut self, training: bool) {
        crate::nn::TrainMode::set_training(&mut self.attention, training);
        crate::nn::TrainMode::set_training(&mut self.attention_norm, training);
        crate::nn::TrainMode::set_training(&mut self.feed_forward, training);
        crate::nn::TrainMode::set_training(&mut self.feed_forward_norm, training);
        self.dropout.is_training = training;
    }
}

impl<D, B, K, Train> crate::nn::ShapeInfo for TransformerLayer<D, B, K, Train>
where
    D: AttentionDirection,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
{
    /// Reports the three things that are not visible from the type.
    ///
    /// The direction is, so it is named here as well: a summary that printed
    /// twelve identical `TransformerLayer` rows would not say which of them
    /// is masked, and the alias a caller wrote is not what a layer summary
    /// prints.
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(alloc::format!(
            "{}, norm={:?}, feed_forward={:?}",
            D::NAME,
            self.config.norm,
            self.config.feed_forward
        ))
    }
}

impl<D, B, K, Train> TransformerLayer<D, B, K, Train>
where
    D: AttentionDirection,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
{
    /// The model width this layer consumes and produces.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.attention.d_model()
    }

    /// Whether this layer's attention is masked. A property of `D`, not state.
    #[must_use]
    pub const fn is_causal(&self) -> bool {
        D::CAUSAL
    }

    /// Freezes every parameter in both sub-layers.
    pub fn freeze(self) -> TransformerLayer<D, B, K, Frozen> {
        TransformerLayer {
            attention: self.attention.freeze(),
            attention_norm: self.attention_norm.freeze(),
            feed_forward: self.feed_forward.freeze(),
            feed_forward_norm: self.feed_forward_norm.freeze(),
            dropout: self.dropout,
            config: self.config,
            _direction: core::marker::PhantomData,
        }
    }

    /// Unfreezes every parameter in both sub-layers.
    pub fn unfreeze(self) -> TransformerLayer<D, B, K, Trainable> {
        TransformerLayer {
            attention: self.attention.unfreeze(),
            attention_norm: self.attention_norm.unfreeze(),
            feed_forward: self.feed_forward.unfreeze(),
            feed_forward_norm: self.feed_forward_norm.unfreeze(),
            dropout: self.dropout,
            config: self.config,
            _direction: core::marker::PhantomData,
        }
    }
}

impl<D, B, K> TransformerLayer<D, B, K, Trainable>
where
    D: AttentionDirection,
    B: crate::tensor::backend::TensorBackend<K>
        + crate::nn::param::ParameterInit<K>
        + crate::nn::attention::RotaryBackend<K>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>,
    K: DType,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
    <B as Execute<op::Arange>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Exp>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sin>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Cos>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::BroadcastAs>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
{
    /// Builds both sub-layers and their normalizations.
    ///
    /// `config.attention.causal` is ignored and replaced by `D::CAUSAL`: the
    /// direction marker owns masking, and the stored config is corrected so it
    /// cannot disagree with the type.
    pub fn build(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        d_ff: usize,
        config: TransformerConfig,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        let mut config = config;
        config.attention.causal = D::CAUSAL;

        let attention = MultiHeadAttention::build(
            d_model,
            n_heads,
            n_kv_heads,
            config.attention,
            dtype.clone(),
            device.clone(),
        )?;
        let attention_norm = LayerNorm::<Dyn, B, K, Trainable>::build_full(
            d_model,
            dtype.clone(),
            device.clone(),
            config.eps,
        )?;
        let feed_forward = FeedForward::build(
            d_model,
            d_ff,
            config.feed_forward,
            dtype.clone(),
            device.clone(),
        )?;
        let feed_forward_norm =
            LayerNorm::<Dyn, B, K, Trainable>::build_full(d_model, dtype, device, config.eps)?;

        Ok(Self {
            attention,
            attention_norm,
            feed_forward,
            feed_forward_norm,
            dropout: Dropout::new(config.dropout),
            config,
            _direction: core::marker::PhantomData,
        })
    }
}

/// Everything a transformer layer executes, in one bound.
///
/// The union of [`AttentionBackend`] and [`FeedForwardBackend`] plus the
/// normalization row. Stated as an alias because the `Module` impl below would
/// otherwise carry the two sub-layers' bound lists concatenated, and a reader
/// could not tell which entries the layer itself needs.
pub trait TransformerBackend<K: DType>:
    AttentionBackend<K> + FeedForwardBackend<K> + Execute<op::LayerNorm>
{
}

impl<K: DType, B> TransformerBackend<K> for B where
    B: AttentionBackend<K> + FeedForwardBackend<K> + Execute<op::LayerNorm>
{
}

impl<D, B, K, Train, G, L> Module<Tensor<Dyn, B, K, G, Local, L>>
    for TransformerLayer<D, B, K, Train>
where
    D: AttentionDirection,
    B: TransformerBackend<K> + crate::tensor::backend::SupportsDType<K>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad
        + GradJoin<Train::TensorGrad>
        + GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad, Output = JoinedGrad<G, Train::TensorGrad>>
        + GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>
        + GradJoin<NoGrad, Output = JoinedGrad<G, Train::TensorGrad>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::BroadcastAs>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Softmax>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Ones>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Tril>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Log>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Dropout>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Gelu>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Swish>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::LayerNorm>>::Output: Into<B::Storage<K>>,
{
    /// `Dyn`: the result comes off a residual add or a normalization
    /// depending on the placement, and neither arm's layout survives the
    /// other's, so no claim is made.
    type Output = Tensor<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>, Local>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let dims = x.shape_buf().as_ref().to_vec();
        if dims.len() != 3 || dims[2] != self.d_model() {
            return Err(invalid_owned(
                "transformer layer forward",
                alloc::format!(
                    "expected a rank-3 [batch, seq, {}] input, got {dims:?}",
                    self.d_model()
                ),
            ));
        }

        let x = x.forget_layout();
        match self.config.norm {
            // x + drop(attn(norm(x))), then the same around the feed-forward.
            // The residual path carries no normalization, which is the whole
            // point of the placement.
            NormPlacement::Pre => {
                let attended = self
                    .attention
                    .forward(self.attention_norm.forward(x.clone())?)?;
                let attended = self.residual_dropout(attended)?;
                let hidden = x.broadcast_add(&attended)?.forget_layout();

                let projected = self
                    .feed_forward
                    .forward(self.feed_forward_norm.forward(hidden.clone())?)?;
                let projected = self.residual_dropout(projected)?;
                Ok(hidden.broadcast_add(&projected)?.forget_layout())
            }
            // norm(x + drop(attn(x))), the original formulation.
            NormPlacement::Post => {
                let attended = self.residual_dropout(self.attention.forward(x.clone())?)?;
                let hidden = self
                    .attention_norm
                    .forward(x.broadcast_add(&attended)?.forget_layout())?;

                let projected =
                    self.residual_dropout(self.feed_forward.forward(hidden.clone())?)?;
                Ok(self
                    .feed_forward_norm
                    .forward(hidden.broadcast_add(&projected)?.forget_layout())?
                    .forget_layout())
            }
        }
    }
}

impl<D, B, K, Train> TransformerLayer<D, B, K, Train>
where
    D: AttentionDirection,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
{
    /// Applies the residual-branch dropout, which is the identity when the
    /// probability is zero or the layer is in evaluation mode.
    fn residual_dropout<Gr: RequiresGrad>(
        &self,
        x: Tensor<Dyn, B, K, Gr, Local>,
    ) -> Result<Tensor<Dyn, B, K, Gr, Local>>
    where
        B: crate::exec::Capabilities + Execute<op::Dropout>,
        <B as Execute<op::Dropout>>::Output: Into<B::Storage<K>>,
    {
        apply_dropout(x, self.dropout.p, self.dropout.is_training)
    }
}
