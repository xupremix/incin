//! Typed sparse mixture-of-experts routing with a dense masked forward path.
//!
//! [`MoE`] pairs a soft top-k [`Router`] with a fixed array of expert
//! submodules (issue #102). The reference path builds a per-token expert
//! weight column from one-hot assignments and runs every expert on the full
//! batch, so no token is silently dropped. Capacity-factor drop and the
//! aux load-balancing loss are deliberately out of scope for this module
//! and fail closed as unimplemented gaps in the issue notes.

use alloc::format;
use alloc::vec::Vec;

use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::op;
use crate::nn::ComputeStats;
use crate::nn::attention::{invalid, invalid_owned};
use crate::nn::linear::Linear;
use crate::nn::module::{Module, NamedLayers, ShapeInfo, TrainMode};
use crate::nn::optional::False;
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::shapes::{Dense, Dyn, Layout};
use crate::tensor::backend::{Execute, SupportsDType, TransferTo, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, RequiresGrad};
use crate::tensor::transfer::ToDevice;

/// Backend operations [`Router::forward`] itself needs beyond its gate
/// projection.
///
/// Split out so the `Module` impl's bound list stays readable: these are the
/// softmax/top-k/gather/renormalize ops the routing head performs, not the
/// projection ops [`Linear`] already requires.
pub trait RouterBackend<K: DType>: VariableBackend + crate::exec::Capabilities
where
    Self: Execute<op::Softmax>
        + Execute<op::TopK>
        + Execute<op::Gather>
        + Execute<op::SumKeepDim>
        + Execute<op::Div>
        + Execute<op::Mul>
        + Execute<op::Sub>,
{
}

impl<K: DType, B> RouterBackend<K> for B where
    B: VariableBackend
        + crate::exec::Capabilities
        + Execute<op::Softmax>
        + Execute<op::TopK>
        + Execute<op::Gather>
        + Execute<op::SumKeepDim>
        + Execute<op::Div>
        + Execute<op::Mul>
        + Execute<op::Sub>
{
}

/// Backend operations the dense masked [`MoE::forward`] path needs beyond
/// [`RouterBackend`] and whatever each expert's own `forward` requires.
pub trait MoEBackend<K: DType>: RouterBackend<K>
where
    Self: Execute<op::OneHot>
        + Execute<op::ToDType>
        + Execute<op::UnsqueezeExact>
        + Execute<op::Mul>
        + Execute<op::SumDim>
        + Execute<op::Add>
        + Execute<op::Narrow>
        + Execute<op::Bincount>
        + Execute<op::Cumsum>
        + Execute<op::Sub>
        + Execute<op::ConcatExact>
        + Execute<op::Zeros>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>,
{
}

impl<K: DType, B> MoEBackend<K> for B where
    B: RouterBackend<K>
        + Execute<op::OneHot>
        + Execute<op::ToDType>
        + Execute<op::UnsqueezeExact>
        + Execute<op::Mul>
        + Execute<op::SumDim>
        + Execute<op::Add>
        + Execute<op::Narrow>
        + Execute<op::Bincount>
        + Execute<op::Cumsum>
        + Execute<op::Sub>
        + Execute<op::ConcatExact>
        + Execute<op::Zeros>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
{
}

/// An expert leaf that can flip from trainable to frozen typestate.
///
/// Implemented for [`Linear`] and for arrays of experts, which is what
/// [`MoE::freeze`] maps over. A custom expert module outside those shapes
/// implements this trait to participate in `MoE` typestate freeze. The
/// frozen form must remain a full module so the typestate `MoE` still
/// satisfies its own trait bounds.
pub trait FreezeExpert: Sized {
    /// The frozen form of this expert.
    type Frozen: NamedLayers + ShapeInfo + TrainMode + ComputeStats;

    /// Freezes this expert's parameters by type.
    fn freeze_expert(self) -> Self::Frozen;
}

/// An expert leaf that can return from frozen to trainable typestate.
pub trait UnfreezeExpert: Sized {
    /// The trainable form of this expert.
    type Trainable: NamedLayers + ShapeInfo + TrainMode + ComputeStats;

    /// Unfreezes this expert's parameters by type.
    fn unfreeze_expert(self) -> Self::Trainable;
}

impl<S, B, Bias, K> FreezeExpert for Linear<S, B, Bias, K, Trainable>
where
    S: crate::nn::linear::LinearShape,
    B: VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
{
    type Frozen = Linear<S, B, Bias, K, Frozen>;

    fn freeze_expert(self) -> Self::Frozen {
        Linear::freeze(self)
    }
}

impl<S, B, Bias, K> UnfreezeExpert for Linear<S, B, Bias, K, Frozen>
where
    S: crate::nn::linear::LinearShape,
    B: VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
{
    type Trainable = Linear<S, B, Bias, K, Trainable>;

    fn unfreeze_expert(self) -> Self::Trainable {
        Linear::unfreeze(self)
    }
}

impl<T: FreezeExpert, const N: usize> FreezeExpert for [T; N] {
    type Frozen = [T::Frozen; N];

    fn freeze_expert(self) -> Self::Frozen {
        self.map(FreezeExpert::freeze_expert)
    }
}

impl<T: UnfreezeExpert, const N: usize> UnfreezeExpert for [T; N] {
    type Trainable = [T::Trainable; N];

    fn unfreeze_expert(self) -> Self::Trainable {
        self.map(UnfreezeExpert::unfreeze_expert)
    }
}

/// Soft top-k routing over `E` experts for tokens shaped `[..., d_model]`.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::nn::{Module, Router};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// let router = Router::<4, 2, Cpu>::build(8, (), ())?;
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![3, 8])?;
/// let routing = router.forward(x)?;
/// assert_eq!(routing.weights.dims().dims(), &[3, 2]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct Router<
    const E: usize,
    const TOPK: usize,
    B: VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// Softmax gate over the `E` experts (bias-free projection).
    pub gate: Linear<Dyn, B, False, K, Train>,
}

impl<const E: usize, const TOPK: usize, B: VariableBackend, K: DType, Train: TrainState> ShapeInfo
    for Router<E, TOPK, B, K, Train>
{
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(format!("E={E}, topk={TOPK}"))
    }
}

impl<const E: usize, const TOPK: usize, B: VariableBackend, K: DType>
    Router<E, TOPK, B, K, Trainable>
{
    /// Freezes the gate projection.
    pub fn freeze(self) -> Router<E, TOPK, B, K, Frozen> {
        Router {
            gate: self.gate.freeze(),
        }
    }
}

impl<const E: usize, const TOPK: usize, B: VariableBackend, K: DType>
    Router<E, TOPK, B, K, Frozen>
{
    /// Unfreezes the gate projection.
    pub fn unfreeze(self) -> Router<E, TOPK, B, K, Trainable> {
        Router {
            gate: self.gate.unfreeze(),
        }
    }
}

impl<const E: usize, const TOPK: usize, B, K> Router<E, TOPK, B, K, Trainable>
where
    B: crate::backend_authoring::TensorBackend<K> + crate::nn::param::ParameterInit<K>,
    K: DType,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds a bias-free gate projecting `d_model -> E`.
    pub fn build(
        d_model: usize,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        const {
            assert!(E > 0, "a router needs at least one expert");
            assert!(TOPK > 0, "top-k must be at least 1");
            assert!(TOPK <= E, "top-k cannot exceed the expert count");
        }
        if d_model == 0 {
            return Err(invalid("build router", "d_model must be nonzero"));
        }
        Ok(Self {
            gate: Linear::build_full(d_model, E, dtype, device, ())?,
        })
    }
}

/// `zeros` from field-form dtype and device, bypassing the argument tuples.
///
/// The public constructors take *arguments* and resolve them through
/// `ArgInto`, which a generic `B::Device` cannot satisfy: those tuple
/// conversions are written for concrete types. An existing tensor already
/// holds the fields, so this dispatches the same catalog row directly.
fn zeros_from_fields<B, K>(
    dims: Vec<usize>,
    dtype: &<K as DType>::Field,
    device: &<B::Device as Device>::Field,
) -> Result<Tensor<Dyn, B, K, crate::tensor::grad::NoGrad, Local>>
where
    B: crate::tensor::backend::Backend
        + crate::tensor::backend::SupportsDType<K>
        + Execute<op::Zeros>,
    K: DType,
    <B as Execute<op::Zeros>>::Output: Into<B::Storage<K>>,
{
    let device_id = <B::Device as Device>::to_incin(device)?;
    let descriptor = B::resolve_dtype(dtype, &device_id)?;
    let shape = crate::shapes::ShapeBuf::from_slice(&dims);
    let expected =
        crate::shapes::ShapeValue::<Dyn>::try_new(shape.clone()).map_err(Error::Shape)?;
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    let inner = crate::exec::dispatch::execute_shaped::<op::Zeros, B, Dyn>(
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

/// Top-k gate outputs for one forward pass.
///
/// `indices` names the experts each token selected; `weights` are the
/// (renormalized) gate probabilities at those positions, gathered from the
/// full softmax so the gate stays differentiable. `probs` is the full
/// distribution over `E`, retained for load-balancing diagnostics and for
/// [`Self::expert_offsets`].
///
/// Not `Debug`: the dense fields need `HostInterop` for `Tensor`'s `Debug`
/// impl, which a non-host backend need not implement.
#[derive(Clone)]
pub struct Routing<const E: usize, B: VariableBackend, K: DType, G: RequiresGrad> {
    /// Full softmax over the `E` experts, shape `[..., E]`.
    pub probs: Dense<Dyn, B, K, G, Local>,
    /// Renormalized top-k weights, shape `[..., TOPK]`.
    pub weights: Dense<Dyn, B, K, G, Local>,
    /// Expert ids for the top-k slots, shape `[..., TOPK]`.
    pub indices: Dense<Dyn, B, u32, crate::tensor::grad::NoGrad, Local>,
}

impl<const E: usize, B, K, G> Routing<E, B, K, G>
where
    B: VariableBackend
        + crate::exec::Capabilities
        + Execute<op::Bincount>
        + Execute<op::Cumsum>
        + Execute<op::ConcatExact>
        + Execute<op::Zeros>,
    K: DType,
    G: RequiresGrad,
    <B as Execute<op::Bincount>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::Cumsum>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::Zeros>>::Output: Into<B::Storage<i64>>,
{
    /// Exclusive expert offsets of length `E + 1` for a grouped buffer.
    ///
    /// Returned as `[0, c0, c0+c1, ..., total]` in `i64`. Built as
    /// `bincount`, an inclusive `cumsum`, then a leading zero prepended by
    /// `concat` — no host interop and no `i64` `sub` (the CPU elementwise
    /// family is float-only, so an exclusive scan via subtraction is
    /// refused). An empty router (`E == 0`) is refused rather than
    /// producing a nonsense prefix.
    pub fn expert_offsets(&self) -> Result<Dense<Dyn, B, i64, crate::tensor::grad::NoGrad, Local>>
    where
        B: crate::tensor::backend::SupportsDType<i64>,
    {
        if E == 0 {
            return Err(invalid(
                "routing expert_offsets",
                "expert count must be nonzero",
            ));
        }
        let counts = self.indices.bincount::<E>()?.into_dyn();
        let inclusive = counts.cumsum(0isize)?;
        let zero = zeros_from_fields::<B, i64>(
            alloc::vec![1],
            &<i64 as DType>::init(()),
            &inclusive._device,
        )?;
        zero.concat(&inclusive, 0isize)
    }
}

impl<const E: usize, const TOPK: usize, B, K, Train, G, L> Module<Tensor<Dyn, B, K, G, Local, L>>
    for Router<E, TOPK, B, K, Train>
where
    B: RouterBackend<K>
        + crate::tensor::backend::SupportsDType<K>
        + crate::tensor::backend::SupportsDType<u32>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad>
        + GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Softmax>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TopK>>::Output: Into<(B::Storage<K>, B::Storage<u32>)>,
    <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
{
    type Output = Routing<E, B, K, JoinedGrad<G, Train::TensorGrad>>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let dims = x.shape_buf().as_ref();
        let expected = self.gate.weight.shape_dims();
        let in_features = expected.get(1).copied();
        let Some(last) = dims.last().copied() else {
            return Err(invalid(
                "router forward",
                "expected a tensor with at least one dimension",
            ));
        };
        if let Some(expected_in) = in_features
            && last != expected_in
        {
            return Err(invalid_owned(
                "router forward",
                format!(
                    "expected last dimension {expected_in} to match the gate \
                     projection, got {last}"
                ),
            ));
        }

        let logits = self.gate.forward(x)?;
        let probs = logits.softmax(-1isize)?;
        // `topk` is only defined for the default `Dyn` layout; softmax
        // returns `RowMajor`, so drop the layout proof before selecting.
        let ranked = probs.clone().forget_layout();
        let (_top_values, indices) = ranked.topk(TOPK, -1isize, true)?;
        let weights = probs.gather(-1isize, &indices)?;
        let denom = weights.sum_keepdim(-1isize)?;
        let weights = weights.broadcast_div(&denom)?;

        Ok(Routing {
            probs,
            weights,
            indices,
        })
    }
}

/// A sparse mixture of `E` experts with soft top-k routing (issue #102).
///
/// The forward path is deliberately dense: every expert runs on the full
/// batch and each output is scaled by a gate weight that is zero for tokens
/// that did not select that expert. That keeps gradients honest and avoids
/// silent token drop until a capacity-factor path is implemented.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::nn::{Linear, Module, MoE};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// let moe = MoE::<2, 1, _, Cpu>::build(8, (), (), || {
///     Linear::<Dyn, Cpu>::build((8, 16))
/// })?;
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![4, 8])?;
/// assert_eq!(moe.forward(x)?.dims().dims(), &[4, 16]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_to_device)]
pub struct MoE<
    const E: usize,
    const TOPK: usize,
    Expert,
    B: VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> where
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    /// Soft top-k router over the expert array.
    pub router: Router<E, TOPK, B, K, Train>,
    /// The `E` expert submodules, visited as `experts.0`, `experts.1`, ...
    pub experts: [Expert; E],
}

impl<const E: usize, const TOPK: usize, Expert, B, K, Train> ShapeInfo
    for MoE<E, TOPK, Expert, B, K, Train>
where
    B: VariableBackend,
    K: DType,
    Train: TrainState,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(format!("E={E}, topk={TOPK}"))
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K> MoE<E, TOPK, Expert, B, K, Trainable>
where
    B: VariableBackend,
    K: DType,
    Expert: FreezeExpert + NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    /// Freezes the router gate and every expert.
    pub fn freeze(self) -> MoE<E, TOPK, Expert::Frozen, B, K, Frozen> {
        MoE {
            router: self.router.freeze(),
            experts: self.experts.map(FreezeExpert::freeze_expert),
        }
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K> MoE<E, TOPK, Expert, B, K, Frozen>
where
    B: VariableBackend,
    K: DType,
    Expert: UnfreezeExpert + NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    /// Unfreezes the router gate and every expert.
    pub fn unfreeze(self) -> MoE<E, TOPK, Expert::Trainable, B, K, Trainable> {
        MoE {
            router: self.router.unfreeze(),
            experts: self.experts.map(UnfreezeExpert::unfreeze_expert),
        }
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K> MoE<E, TOPK, Expert, B, K, Trainable>
where
    B: crate::backend_authoring::TensorBackend<K> + crate::nn::param::ParameterInit<K>,
    K: DType,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    /// Builds the router gate and `E` experts from `make_expert`.
    ///
    /// `make_expert` is called exactly `E` times so each slot can be built
    /// with the caller's own widths and bias choice. `d_model` is the shared
    /// input width: every expert and the gate must accept it.
    pub fn build(
        d_model: usize,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
        mut make_expert: impl FnMut() -> Result<Expert>,
    ) -> Result<Self> {
        const {
            assert!(E > 0, "MoE needs at least one expert");
            assert!(TOPK > 0, "top-k must be at least 1");
            assert!(TOPK <= E, "top-k cannot exceed the expert count");
        }
        if d_model == 0 {
            return Err(invalid("build moe", "d_model must be nonzero"));
        }
        let router = Router::build(d_model, dtype, device)?;
        let mut experts = Vec::with_capacity(E);
        for _ in 0..E {
            experts.push(make_expert()?);
        }
        let experts = experts.try_into().map_err(|_| Error::InternalInvariant {
            operation: "build moe",
            reason: "expert builder must produce exactly E experts",
        })?;
        Ok(Self { router, experts })
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K, Train, NewD> ToDevice<B, NewD>
    for MoE<E, TOPK, Expert, B, K, Train>
where
    B: TransferTo<NewD>,
    B::Output: VariableBackend,
    B::Output: SupportsDType<K>,
    K: DType,
    Train: TrainState,
    NewD: Device,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
    Router<E, TOPK, B, K, Train>: ToDevice<B, NewD, Output = Router<E, TOPK, B::Output, K, Train>>,
    Expert: ToDevice<B, NewD>,
    [Expert; E]: ToDevice<B, NewD, Output = [Expert::Output; E]>,
    Expert::Output: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    type Output = MoE<E, TOPK, Expert::Output, B::Output, K, Train>;

    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(MoE {
            router: self.router.to_device(arg)?,
            experts: self.experts.to_device(arg)?,
        })
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K, Train, G, L>
    Module<Tensor<Dyn, B, K, G, Local, L>> for MoE<E, TOPK, Expert, B, K, Train>
where
    B: MoEBackend<K>
        + crate::tensor::backend::SupportsDType<K>
        + crate::tensor::backend::SupportsDType<bool>
        + crate::tensor::backend::SupportsDType<u32>
        + crate::tensor::backend::SupportsDType<i64>,
    K: DType<Arg = ()>,
    Train: TrainState,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: GradJoin<Train::TensorGrad>
        + GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>,
    Router<E, TOPK, B, K, Train>: Module<
            Tensor<Dyn, B, K, G, Local, L>,
            Output = Routing<E, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    Expert: Module<
            Tensor<Dyn, B, K, G, Local, L>,
            Output = Dense<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>, Local>,
            Error = Error,
        >,
    <B as Execute<op::OneHot>>::Output: Into<B::Storage<bool>>,
    <B as Execute<op::ToDType>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Softmax>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TopK>>::Output: Into<(B::Storage<K>, B::Storage<u32>)>,
    <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
{
    type Output = Dense<Dyn, B, K, JoinedGrad<G, Train::TensorGrad>, Local>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        if E == 0 {
            return Err(invalid("moe forward", "expert count must be nonzero"));
        }

        let dims = x.shape_buf().as_ref().to_vec();
        let Some(last) = dims.last().copied() else {
            return Err(invalid(
                "moe forward",
                "expected a tensor with at least one dimension",
            ));
        };
        let gate_in = self.router.gate.weight.shape_dims();
        if let Some(expected) = gate_in.get(1).copied()
            && last != expected
        {
            return Err(invalid_owned(
                "moe forward",
                format!(
                    "expected last dimension {expected} to match the router \
                     gate, got {last}"
                ),
            ));
        }

        let routing = self.router.forward(x.clone())?;

        // Dense expert weight columns: one_hot -> to_dtype -> * w.unsqueeze
        // -> sum over the top-k axis yields `[..., E]`. The one_hot side is
        // NoGrad; the join with the gate weights keeps the gate gradient.
        let onehot = routing.indices.one_hot::<E>()?;
        let mask = onehot.to_dtype::<K>()?;
        let weights_col = routing.weights.unsqueeze(-1isize)?;
        let weighted = mask.broadcast_mul(&weights_col)?;
        let expert_weights = weighted.sum(-2isize)?;

        let mut acc: Option<Self::Output> = None;
        for (e, expert) in self.experts.iter().enumerate() {
            let y = expert.forward(x.clone())?;
            let col = expert_weights.clone().try_narrow(-1isize, e, 1)?;
            let contribution = y.broadcast_mul(&col)?;
            acc = Some(match acc {
                None => contribution,
                Some(a) => a.broadcast_add(&contribution)?,
            });
        }

        acc.ok_or_else(|| {
            invalid(
                "moe forward",
                "expert loop produced no contribution; E must be nonzero",
            )
        })
    }
}
