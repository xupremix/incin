//! Multi-head attention, with grouped-query attention and rotary positions.
//!
//! [`MultiHeadAttention`] composes four [`Linear`] projections with the
//! catalog's matmul, softmax and pointwise rows, or — when attention weights
//! are not being dropped out — the catalog's fused
//! [`ScaledDotProductAttention`](crate::exec::catalog::op) row (issue #104).
//! The fused path is a single descriptor dispatch instead of the composed
//! score/softmax/attend chain; backends that only advertise the composed rows
//! still run the manual path through the same module surface.
//!
//! For incremental decoding, [`MultiHeadAttention::forward_with_cache`] takes a
//! caller-owned [`KvCache`](crate::nn::KvCache) so a generate step projects
//! only its new tokens and attends against the keys and values already stored
//! (issue #104).
//!
//! # The head counts are compile-time (issue #101)
//!
//! `D_MODEL`, `N_HEADS` and `N_KV_HEADS` are const parameters, so both head
//! invariants -- `D_MODEL` divisible by `N_HEADS`, and `N_HEADS` divisible by
//! `N_KV_HEADS` -- are `const { assert!(..) }` inside
//! [`MultiHeadAttention::build`]: a mismatched configuration fails at compile
//! time at the construction site, proved by the compile-fail fixtures
//! `attention_d_model_head_mismatch` and `attention_head_kv_mismatch`. The
//! module itself is written against [`Dyn`](crate::shapes::Dyn) rather than a
//! static shape because the causal mask needs the mask and the score tensor to
//! meet, and the typed API can only express that pairing through `Dyn` today.
//! Tensor shapes stay dynamic; the head configuration does not.
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
use crate::exec::GradMode;
use crate::exec::catalog::op;
use crate::nn::param::{Buffer, Frozen, TrainState, Trainable};
use crate::nn::{Dropout, Linear, Module};
use crate::shapes::{Dyn, DynShape, Layout, Shape};
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

/// Multi-head attention over a `[batch, seq, D_MODEL]` input.
///
/// The four projections are ordinary [`Linear`] layers, so the module saves and
/// loads through the usual state traversal with no special handling. When
/// rotary positions are configured the cosine and sine tables are
/// [`Buffer`]s: they are state that must round-trip through a checkpoint but
/// must never receive gradients or be touched by an optimizer.
///
/// The head configuration is part of the type (issue #101): `D_MODEL` must
/// divide into `N_HEADS`, and `N_HEADS` into `N_KV_HEADS`, or
/// [`build`](Self::build) fails at compile time. `N_KV_HEADS == N_HEADS` is
/// plain multi-head attention, `1` multi-query attention, and anything in
/// between grouped-query attention -- one module, three regimes.
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
/// let attention = MultiHeadAttention::<64, 8, 2, Cpu>::build(
///     AttentionConfig::causal(),
///     (),
///     (),
/// )?;
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
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
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
    /// The configuration the module was built with.
    pub config: AttentionConfig,
}

impl<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
> crate::nn::TrainMode for MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
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

impl<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
> crate::nn::ShapeInfo for MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
{
    /// Reports the head configuration, unlike the shaped layers which report
    /// nothing.
    ///
    /// The counts now live in the type, but a summary prints text, not types:
    /// twelve identical `MultiHeadAttention` rows would not say how wide they
    /// are or how many heads share a key.
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(alloc::format!(
            "d_model={D_MODEL}, heads={N_HEADS}, kv_heads={N_KV_HEADS}, head_dim={}",
            Self::head_dim()
        ))
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

impl<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
> MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
{
    /// The model width this module consumes and produces.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        D_MODEL
    }

    /// Width of one head, `D_MODEL / N_HEADS` (issue #101: a computed
    /// constant, not a configuration field).
    #[must_use]
    pub const fn head_dim() -> usize {
        D_MODEL / N_HEADS
    }

    /// Number of query heads.
    #[must_use]
    pub const fn n_heads() -> usize {
        N_HEADS
    }

    /// Number of key/value heads: `N_HEADS` for plain multi-head attention,
    /// `1` for multi-query attention, a divisor in between for grouped-query.
    #[must_use]
    pub const fn n_kv_heads() -> usize {
        N_KV_HEADS
    }

    /// How many query heads share each key/value head.
    #[must_use]
    pub const fn heads_per_group(&self) -> usize {
        N_HEADS / N_KV_HEADS
    }

    /// Freezes every projection, leaving the tables untouched.
    ///
    /// The rotary tables are already outside the gradient path, so freezing
    /// has nothing to say about them.
    pub fn freeze(self) -> MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Frozen> {
        MultiHeadAttention {
            query: self.query.freeze(),
            key: self.key.freeze(),
            value: self.value.freeze(),
            output: self.output.freeze(),
            rotary_cos: self.rotary_cos,
            rotary_sin: self.rotary_sin,
            dropout: self.dropout,
            config: self.config,
        }
    }

    /// Unfreezes every projection.
    pub fn unfreeze(self) -> MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Trainable> {
        MultiHeadAttention {
            query: self.query.unfreeze(),
            key: self.key.unfreeze(),
            value: self.value.unfreeze(),
            output: self.output.unfreeze(),
            rotary_cos: self.rotary_cos,
            rotary_sin: self.rotary_sin,
            dropout: self.dropout,
            config: self.config,
        }
    }
}

impl<const D_MODEL: usize, const N_HEADS: usize, const N_KV_HEADS: usize, B, K>
    MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Trainable>
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
    /// Builds the module. The head invariants are compile-time (issue #101).
    ///
    /// `D_MODEL` must divide evenly into `N_HEADS`, and `N_HEADS` into
    /// `N_KV_HEADS`: each key/value head has to serve a whole number of query
    /// heads, and every head the same width. Both are `const { assert!(..) }`
    /// below, so a mismatched configuration fails at compile time at this call
    /// site -- proved by the compile-fail fixtures
    /// `attention_d_model_head_mismatch` and `attention_head_kv_mismatch`
    /// rather than checked as runtime errors here.
    ///
    /// Key and value project to `N_KV_HEADS * head_dim` rather than `D_MODEL`,
    /// which is the whole point of grouped-query attention: with two key/value
    /// heads out of eight, those projections and the cache they feed are a
    /// quarter the size.
    ///
    /// # Compile-time failures
    ///
    /// `N_HEADS == 0` or `N_KV_HEADS == 0` ("head counts must be nonzero"),
    /// `D_MODEL % N_HEADS != 0` ("d_model must be divisible by n_heads"), and
    /// `N_HEADS % N_KV_HEADS != 0` ("n_heads must be divisible by
    /// n_kv_heads") all abort compilation at the construction site.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidModuleState`] when rotary positions are configured
    ///   with an odd head width (the rotation pairs dimensions), from the
    ///   table builder, or when `max_seq_len` is zero.
    pub fn build(
        config: AttentionConfig,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        // Issue #101: proven at compile time, at the construction site.
        const {
            assert!(N_HEADS > 0, "head counts must be nonzero");
        }
        const {
            assert!(N_KV_HEADS > 0, "head counts must be nonzero");
        }
        const {
            assert!(
                D_MODEL.is_multiple_of(N_HEADS),
                "d_model must be divisible by n_heads"
            );
        }
        const {
            assert!(
                N_HEADS.is_multiple_of(N_KV_HEADS),
                "n_heads must be divisible by n_kv_heads"
            );
        }
        let head_dim = D_MODEL / N_HEADS;
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
        let kv_dim = N_KV_HEADS * head_dim;

        let query = Linear::build_full(D_MODEL, D_MODEL, dtype.clone(), device.clone(), ())?;
        let key = Linear::build_full(D_MODEL, kv_dim, dtype.clone(), device.clone(), ())?;
        let value = Linear::build_full(D_MODEL, kv_dim, dtype.clone(), device.clone(), ())?;
        let output = Linear::build_full(D_MODEL, D_MODEL, dtype.clone(), device.clone(), ())?;

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

pub(crate) fn invalid(operation: &'static str, reason: &'static str) -> Error {
    Error::InvalidModuleState {
        operation,
        reason: ErrorMessage::new(reason),
    }
}

pub(crate) fn invalid_owned(operation: &'static str, reason: alloc::string::String) -> Error {
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
    + Execute<op::ScaledDotProductAttention>
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
        + Execute<op::ScaledDotProductAttention>
{
}

impl<const D_MODEL: usize, const N_HEADS: usize, const N_KV_HEADS: usize, B, K, Train, G, L>
    Module<Tensor<Dyn, B, K, G, Local, L>>
    for MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
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
    <B as Execute<op::ScaledDotProductAttention>>::Output: Into<B::Storage<K>>,
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

        let query = self.query.forward(x.clone())?.forget_layout();
        let key = self.key.forward(x.clone())?.forget_layout();
        let value = self.value.forward(x)?.forget_layout();

        // [b, t, n*hd] -> [b, n, t, hd]: split the width into heads, then move
        // the head axis in front of time so each head is a contiguous matrix
        // problem.
        let query = split_heads(&query, batch, seq, N_HEADS, Self::head_dim())?;
        let key = split_heads(&key, batch, seq, N_KV_HEADS, Self::head_dim())?;
        let value = split_heads(&value, batch, seq, N_KV_HEADS, Self::head_dim())?;

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
                    apply_rotary(&query, &cos, &sin, Self::head_dim())?,
                    apply_rotary(&key, &cos, &sin, Self::head_dim())?,
                )
            }
        };

        // Grouped-query attention: give every query head the key/value head of
        // its group by widening the head axis, rather than by projecting keys
        // and values at full width in the first place.
        let key = expand_kv_heads(&key, N_HEADS, N_KV_HEADS)?;
        let value = expand_kv_heads(&value, N_HEADS, N_KV_HEADS)?;

        let attended = if self.dropout.is_training && self.dropout.p > 0.0 {
            // Training with attention-weight dropout stays on the composed
            // path: the fused SDPA row has no dropout operand.
            let scale = self
                .config
                .scale
                .unwrap_or_else(|| 1.0_f64 / f64::sqrt(Self::head_dim() as f64));
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
            weights.matmul(&value)?.forget_layout()
        } else {
            // Eval (or zero dropout): one descriptor dispatch. The CPU row
            // composes the same math the manual path above writes out, so
            // results match to floating-point noise.
            let mask = if self.config.causal {
                Some(causal_mask::<B, K>(seq, &query._dtype, &query._device)?)
            } else {
                None
            };
            Tensor::scaled_dot_product_attention(
                &query,
                &key,
                &value,
                mask.as_ref(),
                self.config.scale,
            )?
            .forget_layout()
        };

        // [b, n, t, hd] -> [b, t, n*hd], undoing the split.
        let merged = attended
            .transpose(1isize, 2isize)?
            .forget_layout()
            .reshape(vec![batch, seq, D_MODEL])?
            .forget_layout();

        Ok(self.output.forward(merged)?.forget_layout())
    }
}

impl<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
> MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
{
    /// Narrows a cached rotary table to `[start, start+len)`.
    ///
    /// `start` is the absolute position of the first token in the window, so
    /// incremental decoding rotates a new chunk at its true offset rather than
    /// as if it began at position 0.
    fn rotary_table_range(
        &self,
        table: Option<&Buffer<Dyn, B, K>>,
        start: usize,
        len: usize,
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
        Ok(table
            .as_tensor()?
            .try_narrow(0isize, start, len)?
            .forget_layout())
    }

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
        self.rotary_table_range(table, 0, seq)
    }
}

impl<const D_MODEL: usize, const N_HEADS: usize, const N_KV_HEADS: usize, B, K, Train>
    MultiHeadAttention<D_MODEL, N_HEADS, N_KV_HEADS, B, K, Train>
where
    B: AttentionBackend<K> + crate::tensor::backend::SupportsDType<K>,
    K: DType,
    Train: TrainState,
    Train::TensorGrad: GradJoin<NoGrad, Output = Train::TensorGrad>
        + GradJoin<Train::TensorGrad, Output = Train::TensorGrad>,
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
    <B as Execute<op::ScaledDotProductAttention>>::Output: Into<B::Storage<K>>,
{
    /// One incremental decode step against a caller-owned [`KvCache`](crate::nn::KvCache).
    ///
    /// Issue #104: the input is this step's new tokens only (`[batch, seq,
    /// d_model]`, no gradient tracking). The method projects them, applies
    /// rotary positions at their **absolute** offsets (`cache.len() ..
    /// cache.len()+seq`), appends the rotated keys/values to `cache`, and
    /// runs fused scaled dot-product attention against the full stored
    /// prefix. The returned tensor is `NoGrad`: generation does not build a
    /// tape, and the ambient gradient mode is forced off for the duration of
    /// the call so trainable projections cannot record nodes either.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidModuleState`]
    ///   when `x` is not `[batch, seq, d_model]`, when the cache geometry does
    ///   not match `x`'s batch / head configuration, or when rotary tables are
    ///   configured but missing.
    /// - [`Error::CacheCapacityExceeded`]
    ///   when `cache.len() + seq` would pass the cache's capacity.
    /// - a rotary error when `cache.len() + seq` exceeds the tables'
    ///   `max_seq_len`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate incin_core as incin;
    /// # use incin::nn::{AttentionConfig, KvCache, MultiHeadAttention};
    /// # use incin::prelude::*;
    /// # type Cpu = incin_backends::cpu::CpuBackendImpl;
    /// # fn main() -> Result<()> {
    /// let attention = MultiHeadAttention::<16, 2, 2, Cpu>::build(
    ///     AttentionConfig::causal(), (), (),
    /// )?;
    /// // [batch=1, kv_heads=2, capacity=8, head_dim=8]
    /// let mut cache = KvCache::<s![1, 2, 8, 8], Cpu, f32>::new(())?;
    ///
    /// let step = Tensor::<Dyn, Cpu>::zeros(vec![1, 3, 16])?;
    /// let y = attention.forward_with_cache(step, &mut cache)?;
    /// assert_eq!(y.dims().dims(), &[1, 3, 16]);
    /// assert_eq!(cache.len(), 3);
    /// assert!(!y.requires_grad());
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_with_cache<S: Shape + DynShape>(
        &self,
        x: Tensor<Dyn, B, K, NoGrad, Local>,
        cache: &mut crate::nn::KvCache<S, B, K>,
    ) -> Result<Tensor<Dyn, B, K, NoGrad, Local>> {
        GradMode::Disabled.restrict(|| {
            let dims = x.shape_buf().as_ref().to_vec();
            let [batch, seq, model] = dims[..] else {
                return Err(invalid_owned(
                    "attention forward_with_cache",
                    alloc::format!(
                        "expected a rank-3 [batch, seq, d_model] input, got rank {} {:?}",
                        dims.len(),
                        dims
                    ),
                ));
            };
            if model != self.d_model() {
                return Err(invalid_owned(
                    "attention forward_with_cache",
                    alloc::format!(
                        "input width {model} does not match d_model {}",
                        self.d_model()
                    ),
                ));
            }
            if batch != cache.batch() {
                return Err(invalid_owned(
                    "attention forward_with_cache",
                    alloc::format!(
                        "input batch {batch} does not match the cache batch {}",
                        cache.batch()
                    ),
                ));
            }
            if N_KV_HEADS != cache.kv_heads() || Self::head_dim() != cache.head_dim() {
                return Err(invalid_owned(
                    "attention forward_with_cache",
                    alloc::format!(
                        "module kv_heads={} head_dim={} do not match the cache \
                         kv_heads={} head_dim={}",
                        N_KV_HEADS,
                        Self::head_dim(),
                        cache.kv_heads(),
                        cache.head_dim()
                    ),
                ));
            }

            let past = cache.len();
            if let PositionEncoding::Rotary { max_seq_len, .. } = self.config.position
                && past + seq > max_seq_len
            {
                return Err(invalid_owned(
                    "attention forward_with_cache",
                    alloc::format!(
                        "position {} exceeds the rotary tables' max_seq_len {max_seq_len}; \
                         rebuild the module with a larger extent",
                        past + seq
                    ),
                ));
            }

            let query = self.query.forward(x.clone())?.forget_layout();
            let key = self.key.forward(x.clone())?.forget_layout();
            let value = self.value.forward(x)?.forget_layout();

            let query = split_heads(&query, batch, seq, N_HEADS, Self::head_dim())?;
            let key = split_heads(&key, batch, seq, N_KV_HEADS, Self::head_dim())?;
            let value = split_heads(&value, batch, seq, N_KV_HEADS, Self::head_dim())?;

            // Rotate at absolute positions so a chunk that starts mid-sequence
            // sees the same angles a full forward would have used.
            let (query, key) = match self.config.position {
                PositionEncoding::None => (query, key),
                PositionEncoding::Rotary { .. } => {
                    let cos = self.rotary_table_range(self.rotary_cos.as_ref(), past, seq)?;
                    let sin = self.rotary_table_range(self.rotary_sin.as_ref(), past, seq)?;
                    (
                        apply_rotary(&query, &cos, &sin, Self::head_dim())?,
                        apply_rotary(&key, &cos, &sin, Self::head_dim())?,
                    )
                }
            };

            // Everything below is inference: retag projections to `NoGrad`
            // so SDPA's q/k/v share one gradient type and nothing can reach
            // the tape even if a caller forgot `GradMode::Disabled`.
            let query = retag_nograd(query);
            let key = retag_nograd(key);
            let value = retag_nograd(value);

            // Cache stores pre-expansion (group) heads; widen after read-back.
            cache.append(&key, &value)?;

            let (keys_all, values_all) = cache.kv()?;
            let keys_all = expand_kv_heads(&keys_all, N_HEADS, N_KV_HEADS)?;
            let values_all = expand_kv_heads(&values_all, N_HEADS, N_KV_HEADS)?;

            // Rows `past .. past+seq` of the `[past+seq, past+seq]` causal
            // mask: each new query sees the whole prefix plus itself.
            let mask = if self.config.causal {
                let full = causal_mask::<B, K>(past + seq, &query._dtype, &query._device)?;
                Some(full.try_narrow(0isize, past, seq)?.forget_layout())
            } else {
                None
            };

            let attended = Tensor::scaled_dot_product_attention(
                &query,
                &keys_all,
                &values_all,
                mask.as_ref(),
                self.config.scale,
            )?
            .forget_layout();

            let merged = attended
                .transpose(1isize, 2isize)?
                .forget_layout()
                .reshape(vec![batch, seq, D_MODEL])?
                .forget_layout();
            let out = self.output.forward(merged)?.forget_layout();

            // Inference entry point: hand back a `NoGrad` tensor even though
            // the projections are typed with trainable joins.
            Ok(retag_nograd(out))
        })
    }
}

/// Retypes a tensor as `NoGrad` without touching its storage.
///
/// Used on the inference path where the ambient gradient mode is already
/// disabled: the marker only decides whether later ops *would* record.
fn retag_nograd<S, B, K, G, L>(
    tensor: Tensor<S, B, K, G, Local, L>,
) -> Tensor<S, B, K, NoGrad, Local, L>
where
    S: crate::shapes::Shape,
    B: crate::tensor::backend::Backend,
    K: DType,
    G: RequiresGrad,
    L: Layout<S>,
{
    Tensor::from_shape_value_unchecked(
        tensor.inner,
        tensor._shape,
        tensor._dtype,
        tensor._device,
        core::marker::PhantomData,
    )
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
pub(crate) fn apply_dropout<B, K, G>(
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
