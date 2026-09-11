//! Multi-head attention, with grouped-query attention and rotary positions.
//!
//! [`MultiHeadAttention`] composes four [`Linear`] projections with the
//! catalog's matmul, softmax and pointwise rows. Nothing here is a fused
//! kernel, which is deliberate: the module runs on any backend that advertises
//! the operations it uses, rather than on the subset that has an attention
//! kernel. A fused path can be selected underneath it later without changing
//! the surface.
//!
//! # Why the head counts are runtime values
//!
//! Query heads and key/value heads are fields rather than const parameters,
//! and the module is written against [`Dyn`] rather than a static shape. The
//! reason is the causal mask: masking needs the mask and the score tensor to
//! meet, and the typed API can only express that pairing through `Dyn` today.
//! A static-shape attention module is a follow-on to the broadcast-shape work,
//! not something this module can reach on its own. Both invariants that a
//! const parameterization would have proven at compile time are checked in
//! [`MultiHeadAttention::build`] and reported as errors naming the offending
//! numbers.
//!
//! # Grouped-query attention
//!
//! `n_kv_heads` is a parameter rather than a separate module. Multi-head,
//! grouped-query and multi-query attention differ only in how many key/value
//! heads exist -- `n_heads`, some divisor of it, and `1` respectively -- so
//! three modules would be three copies of one dataflow.

use alloc::vec;

use crate::dist::Local;
use crate::err::{Error, ErrorMessage, Result};
use crate::exec::catalog::op;
use crate::nn::param::{Buffer, Frozen, TrainState, Trainable};
use crate::nn::{Dropout, Linear, Module};
use crate::shapes::{Dyn, Layout};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, NoGrad, RequiresGrad};

/// How position information enters attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionEncoding {
    /// No positional term inside attention.
    ///
    /// Positions, if wanted, are added to the input before it reaches this
    /// module -- a learned absolute embedding, for instance, which is an
    /// [`Embedding`](crate::nn::Embedding) lookup and needs nothing here.
    None,
    /// Rotary position embeddings applied to queries and keys.
    ///
    /// The rotation is applied inside attention, after the head split and
    /// before the scores are formed, so it cannot be expressed as a term added
    /// to the input. `theta` is the frequency base (10000 in the original
    /// formulation) and `max_seq_len` sizes the cached tables: a forward pass
    /// longer than it is an error rather than a silently wrong rotation.
    Rotary {
        /// Frequency base for the rotation angles.
        theta: f64,
        /// Longest sequence the cached tables cover.
        max_seq_len: usize,
    },
}

/// Configuration for [`MultiHeadAttention`].
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    /// Whether a position may attend only to itself and earlier positions.
    pub causal: bool,
    /// Dropout probability applied to the attention weights.
    ///
    /// This is dropout on the softmax output, which is where the original
    /// formulation puts it, not on the block's result.
    pub dropout: f32,
    /// How positions enter attention.
    pub position: PositionEncoding,
    /// Overrides the `1/sqrt(head_dim)` score scale when set.
    pub scale: Option<f64>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            causal: false,
            dropout: 0.0,
            position: PositionEncoding::None,
            scale: None,
        }
    }
}

impl AttentionConfig {
    /// A causal (decoder-style) configuration with no dropout.
    #[must_use]
    pub fn causal() -> Self {
        Self {
            causal: true,
            ..Self::default()
        }
    }

    /// Adds rotary positions with the given base and table extent.
    #[must_use]
    pub fn with_rotary(mut self, theta: f64, max_seq_len: usize) -> Self {
        self.position = PositionEncoding::Rotary { theta, max_seq_len };
        self
    }

    /// Sets the attention-weight dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }
}

/// Multi-head attention over a `[batch, seq, d_model]` input.
///
/// The four projections are ordinary [`Linear`] layers, so the module saves and
/// loads through the usual state traversal with no special handling. When
/// rotary positions are configured the cosine and sine tables are
/// [`Buffer`]s: they are state that must round-trip through a checkpoint but
/// must never receive gradients or be touched by an optimizer.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::nn::{AttentionConfig, Module, MultiHeadAttention};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// // Eight query heads, two key/value heads: grouped-query attention.
/// let attention = MultiHeadAttention::<Cpu>::build(64, 8, 2, AttentionConfig::causal(), (), ())?;
///
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![2, 16, 64])?.require_grad();
/// let y = attention.forward(x)?;
/// assert_eq!(y.dims().dims(), &[2, 16, 64]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_stats, no_train_mode)]
pub struct MultiHeadAttention<
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// Projection producing the query heads.
    pub query: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    /// Projection producing the key heads.
    pub key: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    /// Projection producing the value heads.
    pub value: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    /// Projection applied to the concatenated heads.
    pub output: Linear<Dyn, B, crate::nn::optional::True, K, Train>,
    /// Cosine table for rotary positions, present when rotary is configured.
    pub rotary_cos: Option<Buffer<Dyn, B, K>>,
    /// Sine table for rotary positions, present when rotary is configured.
    pub rotary_sin: Option<Buffer<Dyn, B, K>>,
    #[module(ignore)]
    /// Dropout applied to the attention weights.
    ///
    /// Ignored by the derived traversal because `Dropout` is not parameterized
    /// by the backend, so nothing in a generated `ToDevice` call could infer
    /// which backend it belongs to. Train-mode propagation is therefore
    /// written out below rather than derived.
    pub dropout: Dropout,
    #[module(ignore)]
    /// Number of query heads.
    pub n_heads: usize,
    #[module(ignore)]
    /// Number of key/value heads. Equal to `n_heads` for plain multi-head
    /// attention, `1` for multi-query attention.
    pub n_kv_heads: usize,
    #[module(ignore)]
    /// Width of one head, `d_model / n_heads`.
    pub head_dim: usize,
    #[module(ignore)]
    /// The configuration the module was built with.
    pub config: AttentionConfig,
}

impl<B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState> crate::nn::TrainMode
    for MultiHeadAttention<B, K, Train>
{
    /// Propagates the mode to the one field whose behaviour depends on it.
    ///
    /// Written by hand because `dropout` is skipped by the derived traversal;
    /// the projections have no train-mode behaviour of their own, and are
    /// still visited so that stays true if one ever gains some.
    fn set_training(&mut self, training: bool) {
        crate::nn::TrainMode::set_training(&mut self.query, training);
        crate::nn::TrainMode::set_training(&mut self.key, training);
        crate::nn::TrainMode::set_training(&mut self.value, training);
        crate::nn::TrainMode::set_training(&mut self.output, training);
        self.dropout.is_training = training;
    }
}

/// Backends able to build the rotary tables.
///
/// Stated once as a trait alias rather than repeated on every constructor: the
/// tables are built from `arange`, an exponential, a product and the two
/// trigonometric rows, and naming that list at each site obscures which bound
/// belongs to attention and which to table construction.
pub trait RotaryBackend<K: DType>:
    crate::tensor::backend::VariableBackend
    + crate::exec::Capabilities
    + Execute<op::Arange>
    + Execute<op::MulScalar>
    + Execute<op::Exp>
    + Execute<op::Sin>
    + Execute<op::Cos>
    + Execute<op::Mul>
    + Execute<op::BroadcastAs>
    + Execute<op::ReshapeExact>
    + Execute<op::ConcatExact>
{
}

impl<K: DType, B> RotaryBackend<K> for B where
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::Arange>
        + Execute<op::MulScalar>
        + Execute<op::Exp>
        + Execute<op::Sin>
        + Execute<op::Cos>
        + Execute<op::Mul>
        + Execute<op::BroadcastAs>
        + Execute<op::ReshapeExact>
        + Execute<op::ConcatExact>
{
}

impl<B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    MultiHeadAttention<B, K, Train>
{
    /// The model width this module consumes and produces.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// How many query heads share each key/value head.
    #[must_use]
    pub const fn heads_per_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Freezes every projection, leaving the tables untouched.
    ///
    /// The rotary tables are already outside the gradient path, so freezing
    /// has nothing to say about them.
    pub fn freeze(self) -> MultiHeadAttention<B, K, Frozen> {
        MultiHeadAttention {
            query: self.query.freeze(),
            key: self.key.freeze(),
            value: self.value.freeze(),
            output: self.output.freeze(),
            rotary_cos: self.rotary_cos,
            rotary_sin: self.rotary_sin,
            dropout: self.dropout,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            config: self.config,
        }
    }

    /// Unfreezes every projection.
    pub fn unfreeze(self) -> MultiHeadAttention<B, K, Trainable> {
        MultiHeadAttention {
            query: self.query.unfreeze(),
            key: self.key.unfreeze(),
            value: self.value.unfreeze(),
            output: self.output.unfreeze(),
            rotary_cos: self.rotary_cos,
            rotary_sin: self.rotary_sin,
            dropout: self.dropout,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            config: self.config,
        }
    }
}

impl<B, K> MultiHeadAttention<B, K, Trainable>
where
    B: crate::tensor::backend::TensorBackend<K>
        + crate::nn::param::ParameterInit<K>
        + RotaryBackend<K>
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
    /// Builds the module, checking the two head invariants.
    ///
    /// `d_model` must divide evenly into `n_heads`, and `n_heads` into
    /// `n_kv_heads`: each key/value head has to serve a whole number of query
    /// heads. Both are rejected here with the numbers named, which is the
    /// runtime counterpart of the compile-time check a const-parameterized
    /// module would get.
    ///
    /// Key and value project to `n_kv_heads * head_dim` rather than `d_model`,
    /// which is the whole point of grouped-query attention: with two key/value
    /// heads out of eight, those projections and the cache they feed are a
    /// quarter the size.
    pub fn build(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        config: AttentionConfig,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        if n_heads == 0 || n_kv_heads == 0 {
            return Err(invalid("build attention", "head counts must be nonzero"));
        }
        if !d_model.is_multiple_of(n_heads) {
            return Err(invalid_owned(
                "build attention",
                alloc::format!(
                    "d_model {d_model} is not divisible by n_heads {n_heads}; \
                     every head must be the same width"
                ),
            ));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(invalid_owned(
                "build attention",
                alloc::format!(
                    "n_heads {n_heads} is not divisible by n_kv_heads {n_kv_heads}; \
                     each key/value head must serve a whole number of query heads"
                ),
            ));
        }
        let head_dim = d_model / n_heads;
        if matches!(config.position, PositionEncoding::Rotary { .. }) && !head_dim.is_multiple_of(2)
        {
            return Err(invalid_owned(
                "build attention",
                alloc::format!(
                    "rotary positions need an even head_dim, got {head_dim}; \
                     the rotation pairs dimensions"
                ),
            ));
        }
        let kv_dim = n_kv_heads * head_dim;

        let query = Linear::build_full(d_model, d_model, dtype.clone(), device.clone(), ())?;
        let key = Linear::build_full(d_model, kv_dim, dtype.clone(), device.clone(), ())?;
        let value = Linear::build_full(d_model, kv_dim, dtype.clone(), device.clone(), ())?;
        let output = Linear::build_full(d_model, d_model, dtype.clone(), device.clone(), ())?;

        let (rotary_cos, rotary_sin) = match config.position {
            PositionEncoding::None => (None, None),
            PositionEncoding::Rotary { theta, max_seq_len } => {
                let (cos, sin) = rotary_tables::<B, K>(
                    max_seq_len,
                    head_dim,
                    theta,
                    &<K as DType>::init(dtype.clone()),
                    &<B::Device as Device>::init(device.clone()),
                )?;
                (Some(cos), Some(sin))
            }
        };

        Ok(Self {
            query,
            key,
            value,
            output,
            rotary_cos,
            rotary_sin,
            dropout: Dropout::new(config.dropout),
            n_heads,
            n_kv_heads,
            head_dim,
            config,
        })
    }
}

/// Resolves a dtype descriptor and device id from their field forms.
fn resolve<B, K>(
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<(
    crate::tensor::dtype::DTypeDescriptor,
    crate::tensor::device::DeviceId,
)>
where
    B: crate::tensor::backend::Backend + crate::tensor::backend::SupportsDType<K>,
    K: DType,
{
    let device_id = <B::Device as Device>::to_incin(device)?;
    let descriptor = B::resolve_dtype(dtype, &device_id)?;
    Ok((descriptor, device_id))
}

/// `arange` from field-form dtype and device, bypassing the argument tuples.
///
/// The public constructors take *arguments* and resolve them through
/// `ArgInto`, which a generic `K` and `B::Device` cannot satisfy: those tuple
/// conversions are written for concrete types. A module already holds the
/// fields, so this dispatches the same catalog row directly.
fn arange_from_fields<B, K>(
    len: usize,
    start: f64,
    step: f64,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<Dyn, B, K, NoGrad, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::tensor::backend::SupportsDType<K>
        + crate::exec::Capabilities
        + Execute<op::Arange>,
    K: DType,
    <B as Execute<op::Arange>>::Output: Into<B::Storage<K>>,
{
    let (descriptor, device_id) = resolve::<B, K>(dtype, device)?;
    let dims = crate::shapes::ShapeBuf::from_slice(&[len]);
    let expected = crate::shapes::ShapeValue::<Dyn>::try_new(dims.clone()).map_err(Error::Shape)?;
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    let inner = crate::exec::dispatch::execute_shaped::<op::Arange, B, Dyn>(
        &context,
        crate::exec::catalog::ArangeAttributes {
            shape: vec![len],
            dtype: descriptor,
            device: device_id,
            start,
            step,
        },
        &[],
        &expected,
    )?
    .into();
    Tensor::from_shape_buf(
        inner,
        dims,
        dtype.clone(),
        device.clone(),
        core::marker::PhantomData,
    )
}

/// `ones` from field-form dtype and device. See [`arange_from_fields`].
fn ones_from_fields<B, K>(
    dims: alloc::vec::Vec<usize>,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<Dyn, B, K, NoGrad, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::tensor::backend::SupportsDType<K>
        + crate::exec::Capabilities
        + Execute<op::Ones>,
    K: DType,
    <B as Execute<op::Ones>>::Output: Into<B::Storage<K>>,
{
    let (descriptor, device_id) = resolve::<B, K>(dtype, device)?;
    let shape = crate::shapes::ShapeBuf::from_slice(&dims);
    let expected =
        crate::shapes::ShapeValue::<Dyn>::try_new(shape.clone()).map_err(Error::Shape)?;
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    let inner = crate::exec::dispatch::execute_shaped::<op::Ones, B, Dyn>(
        &context,
        crate::exec::catalog::CreationAttributes {
            shape: dims,
            dtype: descriptor,
            device: device_id,
        },
        &[],
        &expected,
    )?
    .into();
    Tensor::from_shape_buf(
        inner,
        shape,
        dtype.clone(),
        device.clone(),
        core::marker::PhantomData,
    )
}

/// The cosine and sine tables, in that order.
type RotaryTables<B, K> = (Buffer<Dyn, B, K>, Buffer<Dyn, B, K>);

/// Builds the rotary cosine and sine tables as non-trainable buffers.
///
/// The tables are computed with catalog operations rather than on the host so
/// that they land in the right dtype and on the right device without a
/// transfer, and so the construction works for any float dtype the backend
/// supports rather than only the one this code could name.
///
/// The angle for position `p` and pair `i` is `p * theta^(-2i/head_dim)`,
/// evaluated as an exponential to avoid a `powf` per element. Each table is
/// the half-width angle block repeated twice, which is what pairs dimension
/// `i` with dimension `i + head_dim/2` under the rotation applied later.
fn rotary_tables<B, K>(
    max_seq_len: usize,
    head_dim: usize,
    theta: f64,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<RotaryTables<B, K>>
where
    B: RotaryBackend<K> + crate::tensor::backend::SupportsDType<K>,
    K: DType,
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
    if max_seq_len == 0 {
        return Err(invalid(
            "build rotary tables",
            "max_seq_len must be nonzero",
        ));
    }
    let half = head_dim / 2;
    let positions = arange_from_fields::<B, K>(max_seq_len, 0.0, 1.0, dtype, device)?;
    let pair_index = arange_from_fields::<B, K>(half, 0.0, 1.0, dtype, device)?;
    // theta^(-2i/head_dim) == exp(-2i/head_dim * ln(theta)).
    let inverse_frequency = pair_index
        .mul_scalar(-2.0_f64 * f64::ln(theta) / head_dim as f64)?
        .forget_layout()
        .exp()?
        .forget_layout();
    let angles = positions
        .reshape(vec![max_seq_len, 1])?
        .forget_layout()
        .broadcast_mul(&inverse_frequency.reshape(vec![1, half])?.forget_layout())?
        .forget_layout();

    let cos = angles.cos()?.forget_layout();
    let sin = angles.sin()?.forget_layout();
    let cos_table = cos.concat(&cos, 1)?.forget_layout();
    let sin_table = sin.concat(&sin, 1)?.forget_layout();

    Ok((
        into_buffer::<B, K>(cos_table, dtype, device)?,
        into_buffer::<B, K>(sin_table, dtype, device)?,
    ))
}

/// Promotes a computed table into a non-trainable [`Buffer`].
fn into_buffer<B, K>(
    tensor: Tensor<Dyn, B, K, NoGrad, Local>,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<Buffer<Dyn, B, K>>
where
    B: crate::tensor::backend::VariableBackend + crate::tensor::backend::SupportsDType<K>,
    K: DType,
{
    let shape = tensor.shape_buf().clone();
    let var = B::var_from_tensor::<K>(tensor.inner())?;
    Buffer::<Dyn, B, K>::from_parts_checked(var, shape, dtype.clone(), device.clone())
}

fn invalid(operation: &'static str, reason: &'static str) -> Error {
    Error::InvalidModuleState {
        operation,
        reason: ErrorMessage::new(reason),
    }
}

fn invalid_owned(operation: &'static str, reason: alloc::string::String) -> Error {
    Error::InvalidModuleState {
        operation,
        reason: ErrorMessage::new(reason),
    }
}

/// The operations the attention dataflow itself needs, beyond the projections.
///
/// Split out for the same reason as [`RotaryBackend`]: the `Module` impl below
/// would otherwise open with twenty lines of bounds and the reader would have
/// no way to tell which of them attention needs and which `Linear` does.
pub trait AttentionBackend<K: DType>: crate::tensor::backend::VariableBackend
    + crate::exec::Capabilities
    + Execute<op::MatMulExact>
    + Execute<op::TransposeExact>
    + Execute<op::ReshapeExact>
    + Execute<op::UnsqueezeExact>
    + Execute<op::BroadcastAs, Output = <Self as crate::tensor::backend::StorageBackend>::Storage<K>>
    + Execute<op::MulScalar>
    + Execute<op::Softmax>
    + Execute<op::Add>
    + Execute<op::Mul>
    + Execute<op::Ones>
    + Execute<op::Tril>
    + Execute<op::Log>
    + Execute<op::Neg>
    + Execute<op::Narrow>
    + Execute<op::ConcatExact>
    + Execute<op::Dropout>
{
}

impl<K: DType, B> AttentionBackend<K> for B where
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::ReshapeExact>
        + Execute<op::UnsqueezeExact>
        + Execute<op::BroadcastAs, Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>>
        + Execute<op::MulScalar>
        + Execute<op::Softmax>
        + Execute<op::Add>
        + Execute<op::Mul>
        + Execute<op::Ones>
        + Execute<op::Tril>
        + Execute<op::Log>
        + Execute<op::Neg>
        + Execute<op::Narrow>
        + Execute<op::ConcatExact>
        + Execute<op::Dropout>
{
}

impl<B, K, Train, G, L> Module<Tensor<Dyn, B, K, G, Local, L>> for MultiHeadAttention<B, K, Train>
where
    B: AttentionBackend<K> + crate::tensor::backend::SupportsDType<K>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
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
{
    /// `Dyn`: the result is the output projection's, and the chain that
    /// reaches it re-describes buffers often enough that no layout claim
    /// survives it honestly.
    type Output = Tensor<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>, Local>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let dims = x.shape_buf().as_ref().to_vec();
        let [batch, seq, model] = dims[..] else {
            return Err(invalid_owned(
                "attention forward",
                alloc::format!(
                    "expected a rank-3 [batch, seq, d_model] input, got rank {} {:?}",
                    dims.len(),
                    dims
                ),
            ));
        };
        if model != self.d_model() {
            return Err(invalid_owned(
                "attention forward",
                alloc::format!(
                    "input width {model} does not match d_model {}",
                    self.d_model()
                ),
            ));
        }

        let kv_dim = self.n_kv_heads * self.head_dim;
        let query = self.query.forward(x.clone())?.forget_layout();
        let key = self.key.forward(x.clone())?.forget_layout();
        let value = self.value.forward(x)?.forget_layout();

        // [b, t, n*hd] -> [b, n, t, hd]: split the width into heads, then move
        // the head axis in front of time so each head is a contiguous matrix
        // problem.
        let query = split_heads(&query, batch, seq, self.n_heads, self.head_dim)?;
        let key = split_heads(&key, batch, seq, self.n_kv_heads, self.head_dim)?;
        let value = split_heads(&value, batch, seq, self.n_kv_heads, self.head_dim)?;
        let _ = kv_dim;

        let (query, key) = match self.config.position {
            PositionEncoding::None => (query, key),
            PositionEncoding::Rotary { max_seq_len, .. } => {
                if seq > max_seq_len {
                    return Err(invalid_owned(
                        "attention forward",
                        alloc::format!(
                            "sequence length {seq} exceeds the rotary tables' max_seq_len \
                             {max_seq_len}; rebuild the module with a larger extent"
                        ),
                    ));
                }
                let cos = self.rotary_table(self.rotary_cos.as_ref(), seq)?;
                let sin = self.rotary_table(self.rotary_sin.as_ref(), seq)?;
                (
                    apply_rotary(&query, &cos, &sin, self.head_dim)?,
                    apply_rotary(&key, &cos, &sin, self.head_dim)?,
                )
            }
        };

        // Grouped-query attention: give every query head the key/value head of
        // its group by widening the head axis, rather than by projecting keys
        // and values at full width in the first place.
        let key = expand_kv_heads(&key, self.n_heads, self.n_kv_heads)?;
        let value = expand_kv_heads(&value, self.n_heads, self.n_kv_heads)?;

        let scale = self
            .config
            .scale
            .unwrap_or_else(|| 1.0_f64 / f64::sqrt(self.head_dim as f64));
        let scores = query
            .matmul(&key.transpose(2isize, 3isize)?.forget_layout())?
            .mul_scalar(scale)?
            .forget_layout();

        let scores = if self.config.causal {
            let mask = causal_mask::<B, K>(seq, &scores._dtype, &scores._device)?;
            scores.broadcast_add(&mask)?.forget_layout()
        } else {
            scores
        };

        let weights = scores.softmax(3)?.forget_layout();
        let weights = apply_dropout(weights, self.dropout.p, self.dropout.is_training)?;

        let attended = weights.matmul(&value)?.forget_layout();
        // [b, n, t, hd] -> [b, t, n*hd], undoing the split.
        let merged = attended
            .transpose(1isize, 2isize)?
            .forget_layout()
            .reshape(vec![batch, seq, self.n_heads * self.head_dim])?
            .forget_layout();

        Ok(self.output.forward(merged)?.forget_layout())
    }
}

impl<B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    MultiHeadAttention<B, K, Train>
{
    /// Narrows a cached rotary table to the sequence length in hand.
    fn rotary_table(
        &self,
        table: Option<&Buffer<Dyn, B, K>>,
        seq: usize,
    ) -> Result<Tensor<Dyn, B, K, NoGrad>>
    where
        B: crate::exec::Capabilities + Execute<op::Narrow>,
        <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    {
        let table = table.ok_or_else(|| {
            invalid(
                "attention forward",
                "rotary positions are configured but the tables are missing; \
                 the module was constructed without them",
            )
        })?;
        Ok(table.as_tensor()?.try_narrow(0, 0, seq)?.forget_layout())
    }
}

/// `[b, t, n*hd] -> [b, n, t, hd]`.
fn split_heads<B, K, G>(
    x: &Tensor<Dyn, B, K, G, Local>,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor<Dyn, B, K, G, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::exec::Capabilities
        + Execute<op::ReshapeExact>
        + Execute<op::TransposeExact>,
    K: DType,
    G: RequiresGrad,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
{
    Ok(x.reshape(vec![batch, seq, heads, head_dim])?
        .forget_layout()
        .transpose(1isize, 2isize)?
        .forget_layout())
}

/// Widens `[b, kv, t, hd]` to `[b, n, t, hd]` by repeating each key/value head
/// across the query heads of its group.
///
/// The widening is an unsqueeze, a broadcast and a reshape rather than a
/// concatenation, so the repeated heads are produced in one materialization
/// with the group axis adjacent to the head axis -- which is what makes the
/// following reshape a relabelling of the same order rather than a permutation.
fn expand_kv_heads<B, K, G>(
    x: &Tensor<Dyn, B, K, G, Local>,
    n_heads: usize,
    n_kv_heads: usize,
) -> Result<Tensor<Dyn, B, K, G, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::exec::Capabilities
        + Execute<op::ReshapeExact>
        + Execute<op::UnsqueezeExact>
        + Execute<op::BroadcastAs, Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>>,
    K: DType,
    G: RequiresGrad,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
{
    if n_heads == n_kv_heads {
        return Ok(x.clone());
    }
    let dims = x.shape_buf().as_ref().to_vec();
    let [batch, kv, seq, head_dim] = dims[..] else {
        return Err(invalid("expand kv heads", "expected a rank-4 operand"));
    };
    let group = n_heads / n_kv_heads;
    Ok(x.unsqueeze(2isize)?
        .forget_layout()
        .broadcast_to::<Dyn>(vec![batch, kv, group, seq, head_dim])?
        .forget_layout()
        .reshape(vec![batch, n_heads, seq, head_dim])?
        .forget_layout())
}

/// Rotates `[b, n, t, hd]` by the cached angles.
///
/// `x * cos + rotate_half(x) * sin`, where `rotate_half` pairs dimension `i`
/// with `i + hd/2` and negates the second half. The tables are `[t, hd]` and
/// broadcast across batch and heads.
fn apply_rotary<B, K, G>(
    x: &Tensor<Dyn, B, K, G, Local>,
    cos: &Tensor<Dyn, B, K, NoGrad, Local>,
    sin: &Tensor<Dyn, B, K, NoGrad, Local>,
    head_dim: usize,
) -> Result<Tensor<Dyn, B, K, G, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::exec::Capabilities
        + Execute<op::Narrow>
        + Execute<op::Neg>
        + Execute<op::ConcatExact>
        + Execute<op::Mul>
        + Execute<op::Add>
        + Execute<op::BroadcastAs>,
    K: DType,
    G: RequiresGrad + GradJoin<NoGrad, Output = G> + GradJoin<G, Output = G>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::BroadcastAs>>::Output: Into<B::Storage<K>>,
{
    let half = head_dim / 2;
    let first = x.clone().try_narrow(3, 0, half)?.forget_layout();
    let second = x.clone().try_narrow(3, half, half)?.forget_layout();
    let rotated = second
        .neg()?
        .forget_layout()
        .concat(&first, 3)?
        .forget_layout();
    let direct = x.broadcast_mul(cos)?.forget_layout();
    let turned = rotated.broadcast_mul(sin)?.forget_layout();
    Ok(direct.broadcast_add(&turned)?.forget_layout())
}

/// An additive `[t, t]` causal mask: `0` where attention is allowed and
/// negative infinity where it is not.
///
/// Built as `log(tril(ones))` rather than by filling a constant, because that
/// is exact: `log(1)` is `0` and `log(0)` is negative infinity, with no
/// sentinel value to pick and no dtype-dependent "large enough" constant that
/// silently stops being large enough in a narrower float.
fn causal_mask<B, K>(
    seq: usize,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<Dyn, B, K, NoGrad, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::tensor::backend::SupportsDType<K>
        + crate::exec::Capabilities
        + Execute<op::Ones>
        + Execute<op::Tril>
        + Execute<op::Log>,
    K: DType,
    <B as Execute<op::Ones>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Tril>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Log>>::Output: Into<B::Storage<K>>,
{
    let ones = ones_from_fields::<B, K>(vec![seq, seq], dtype, device)?;
    Ok(ones.tril(0)?.forget_layout().log()?.forget_layout())
}

/// Applies dropout without going through the [`Dropout`] module.
///
/// The module's `Module` impl is narrower than this one: it asks for
/// `K: BuiltinDType` and a `ConstDevice`, which would pull those bounds onto
/// attention as a whole for a step that is the identity whenever the
/// probability is zero. Dispatching the catalog row directly keeps the module
/// as generic as the rest of its dataflow.
fn apply_dropout<B, K, G>(
    x: Tensor<Dyn, B, K, G, Local>,
    probability: f32,
    training: bool,
) -> Result<Tensor<Dyn, B, K, G, Local>>
where
    B: crate::tensor::backend::Backend + crate::exec::Capabilities + Execute<op::Dropout>,
    K: DType,
    G: RequiresGrad,
    <B as Execute<op::Dropout>>::Output: Into<B::Storage<K>>,
{
    if !training || probability <= 0.0 {
        return Ok(x);
    }
    let input = crate::exec::TensorHandle::from_storage::<B, K, Local>(&x.inner);
    let context = crate::tensor::grad::execution_context::<B, G>(&x._grad).with_training(training);
    let output = crate::exec::dispatch::execute_shaped::<op::Dropout, B, Dyn>(
        &context,
        crate::exec::catalog::DropoutAttributes {
            probability: f64::from(probability),
            training,
        },
        &[input],
        &x._shape,
    )
    .map_err(Error::from)?
    .into();
    Tensor::from_shape_value(
        output,
        x._shape.clone(),
        x._dtype.clone(),
        x._device.clone(),
        x._grad,
    )
}
