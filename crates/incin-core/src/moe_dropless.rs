//! Dropless grouped-GEMM mixture-of-experts prototype (issue #102).
//!
//! [`DroplessMoE`] is the option-C target from
//! `docs/plan/research/0.2.0/102-nameable-spans.md` §3/§7 and proposal P1 from
//! `docs/plan/research/0.2.0/moe-design-space.md`: top-k routing over `E`
//! experts with no token dropped and no padding, executed as one
//! `grouped_matmul` per projection over the static `[T*K, D]` buffer sliced by
//! the `[E+1]` [`Routing::expert_offsets`][crate::nn::Routing] tile. The
//! per-expert token count `n_e` never becomes a tensor extent: it is a row
//! range `offsets[e]..offsets[e+1]`, never a shape.
//!
//! The forward pass below is the same five stages as the worked example in
//! `crates/incin/examples/moe_dropless_grouped.rs`, in the same order, through
//! the same facade ops: router → flatten/argsort permute →
//! `repeat_interleave`/`index_select` buffer → `grouped_matmul` up, `relu`,
//! `grouped_matmul` down → gate weighting → `scatter_add` combine, with the
//! Switch-style aux loss `E * sum(f * P)` travelling beside the output as the
//! `(combined, aux)` tuple the issue decided on.
//!
//! # Prototype status and known deviations
//!
//! This is a prototype under `incin::experimental`: the public path is
//! `incin::experimental::{DroplessMoE, ExpertMlp, GroupedExpert,
//! DroplessMoEBackend}`, pending maintainer ratification (PROPOSALS.md D-111).
//!
//! Deviations from the `102-nameable-spans.md` §7 sketch, each forced by an
//! unlanded prerequisite the memo itself names:
//!
//! * **Dyn, not static `S`.** The sketch threads a caller-named static input
//!   shape through typed gather/scatter index bounds and a static-shape
//!   `grouped_matmul` overload. Neither exists: #100's index-pair bounds and
//!   the `MatMulShape`-style static rule for `grouped_matmul` (memo §8) are
//!   still open, and `grouped_matmul` returns `Dense<Dyn, …>` today. So the
//!   `Module` impl is over `Tensor<Dyn, …>` exactly like the in-tree
//!   [`Router`][crate::nn::Router]/[`MoE`][crate::nn::MoE], with the outer
//!   `[T, D]` in → `[T, D]` out geometry checked at runtime and the dynamic
//!   span kept inside as offsets (option C). `E`/`TOPK` are const generics as
//!   decided.
//! * **The combine seed requires grad.** `scatter_add` records its tape entry
//!   under the *base* tensor's grad mode and returns the base's marker, so the
//!   example's literal `zeros` base would detach the gate-weight path: with a
//!   `NoGrad` base the whole combine runs disabled and no gradient reaches the
//!   router through the output. The module seeds the combine with
//!   `zeros(...).require_grad()`, which is why `Output`'s combined tensor is
//!   `Grad`-marked. Values are identical to the example chain; only the tape
//!   entry differs. (The aux-loss path needs no such seed: it joins the gate
//!   gradient through `probs` like any broadcast binary op.)
//! * **One host readback for the scatter index.** The inverse-permutation
//!   scatter index is materialized on the host from `perm.to_vec1`, exactly
//!   as the example does, so the impl carries a `HostInterop` bound. An
//!   on-device inverse permutation is future work, not a semantic gap.
//! * **CPU-effective device bound.** The combine seed and the scatter index
//!   are allocated from bare shape args (`zeros(vec![…])`,
//!   `from_slice(&…, vec![…])`), whose arg conversion requires unit dtype and
//!   device args. The impl therefore bounds `K: DType<Arg = ()>` (already
//!   needed for the aux `to_dtype`) and the device to `Arg = ()`, which is
//!   the CPU. This matches the issue's acceptance criterion (prototype
//!   demonstrates the signature on CPU).
//! * **Grad-marked input needed for expert gradients.** `grouped_matmul`
//!   derives its record mode from the *lhs* marker only (unlike `matmul`,
//!   which joins both sides), so with a `NoGrad` input no tape entry covers
//!   the expert weights even when they are trainable. Gate gradients still
//!   flow through both the output path (rescued by the broadcast-mul join)
//!   and the aux path; expert gradients need a grad-marked input. The dense
//!   masked [`MoE`][crate::nn::MoE] trains experts from `NoGrad` inputs, so
//!   this is a genuine B-vs-C behavioral gap the interim/target swap must
//!   eventually close, recorded here rather than hidden.
//!
//! # Example
//!
//! ```
//! # extern crate incin_core as incin;
//! use incin::experimental::{DroplessMoE, ExpertMlp};
//! use incin::nn::Module;
//! use incin::prelude::*;
//! # type Cpu = incin_backends::cpu::CpuBackendImpl;
//!
//! # fn main() -> Result<()> {
//! let moe = DroplessMoE::<2, 1, Cpu>::build(8, 16, (), (), || {
//!     ExpertMlp::<Cpu>::build(8, 16, (), ())
//! })?;
//! let x = Tensor::<Dyn, Cpu>::zeros(vec![4, 8])?;
//! let (y, aux) = moe.forward(x)?;
//! assert_eq!(y.dims().dims(), &[4, 8]);
//! assert!(aux.dims().dims().is_empty());
//! # Ok(())
//! # }
//! ```

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
use crate::nn::param::{TrainState, Trainable};
use crate::nn::{Router, RouterBackend, Routing};
use crate::shapes::{Dense, Dyn, Layout, Nil};
use crate::tensor::backend::{
    Execute, HostInterop, StorageBackend, SupportsDType, TransferTo, VariableBackend,
};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType};
use crate::tensor::grad::{Grad, GradJoin, JoinedGrad, RequiresGrad};
use crate::tensor::transfer::ToDevice;

/// Backend operations the dropless grouped [`DroplessMoE::forward`] path needs
/// beyond [`RouterBackend`] and whatever each expert's own matrix views
/// require.
///
/// This extends the [`MoEBackend`](crate::nn::MoEBackend) idea to the grouped path: one trait naming
/// every catalog row the forward chain touches (permute, buffer, grouped
/// GEMMs, gate weighting, scatter combine, aux loss), so the `Module` impl's
/// bound list stays readable. Nothing here is new execution: every row already
/// exists and already runs on CPU.
pub trait DroplessMoEBackend<K: DType>: RouterBackend<K>
where
    Self: Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::FlattenExact>
        + Execute<op::ToDType>
        + Execute<op::Argsort>
        + Execute<op::RepeatInterleave>
        + Execute<op::IndexSelect>
        + Execute<op::Bincount>
        + Execute<op::Cumsum>
        + Execute<op::ConcatExact>
        + Execute<op::Zeros>
        + Execute<op::GroupedMatMul>
        + Execute<op::UnsqueezeExact>
        + Execute<op::Mul>
        + Execute<op::ScatterAdd>
        + Execute<op::MeanDim>
        + Execute<op::SumAll>
        + Execute<op::MulScalar>
        + Execute<op::DivScalar>
        + Execute<op::Relu>
        + Execute<op::TensorFromData>,
{
}

impl<K: DType, B> DroplessMoEBackend<K> for B where
    B: RouterBackend<K>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::FlattenExact>
        + Execute<op::ToDType>
        + Execute<op::Argsort>
        + Execute<op::RepeatInterleave>
        + Execute<op::IndexSelect>
        + Execute<op::Bincount>
        + Execute<op::Cumsum>
        + Execute<op::ConcatExact>
        + Execute<op::Zeros>
        + Execute<op::GroupedMatMul>
        + Execute<op::UnsqueezeExact>
        + Execute<op::Mul>
        + Execute<op::ScatterAdd>
        + Execute<op::MeanDim>
        + Execute<op::SumAll>
        + Execute<op::MulScalar>
        + Execute<op::DivScalar>
        + Execute<op::Relu>
        + Execute<op::TensorFromData>
{
}

/// What the dropless grouped path needs from one expert: its two projection
/// matrices in grouped-GEMM orientation.
///
/// `up_matrix` is `[D_MODEL, D_FF]` and `down_matrix` is `[D_FF, D_MODEL]`,
/// so stacking every expert's matrices along a fresh axis 0 yields the
/// `[E, K, N]` stacks `grouped_matmul` consumes. The matrices carry the
/// expert's own train-state marker (`Train::TensorGrad`), which is what lets
/// expert gradients flow through the grouped backward; the routing side
/// (indices, permutation, offsets, counts) stays `NoGrad` by type and never
/// enters this trait.
///
/// The matrices are layout-unproven (`Dyn` layout marker): they are
/// transposed views, and views cannot mint the `RowMajor` proof. Stacking
/// (`unsqueeze` + `concat`) allocates fresh dense tensors, so the stacks the
/// module builds are `Dense` again; only the per-expert views stay unproven.
///
/// A custom expert plugs in by implementing this trait (plus the usual
/// `#[module]` bundle in the supertraits) over whatever projection storage it
/// owns; [`ExpertMlp`] is the default small FFN that does so over two
/// bias-free [`Linear`]s.
pub trait GroupedExpert<B, K, Train>:
    Sized + NamedLayers + ShapeInfo + TrainMode + ComputeStats
where
    B: VariableBackend,
    K: DType,
    Train: TrainState,
{
    /// The up-projection in `[D_MODEL, D_FF]` (grouped-GEMM `[K, N]`) orientation.
    fn up_matrix(&self) -> Result<Tensor<Dyn, B, K, Train::TensorGrad, Local, Dyn>>;

    /// The down-projection in `[D_FF, D_MODEL]` (grouped-GEMM `[K, N]`) orientation.
    fn down_matrix(&self) -> Result<Tensor<Dyn, B, K, Train::TensorGrad, Local, Dyn>>;
}

/// The default small expert: a bias-free up-projection, `ReLU`, down-projection.
///
/// Widths are runtime (`Linear<Dyn, …>`), matching the existing [`MoE`](crate::nn::MoE) expert
/// convention; orientation is fixed by construction (`up` maps
/// `D_MODEL -> D_FF`, `down` maps back). The dense [`Module`] forward runs the
/// same arithmetic the grouped path runs per row, so a standalone
/// `ExpertMlp::forward` and its grouped-stack views agree value for value.
///
/// # Example
///
/// ```
/// # extern crate incin_core as incin;
/// use incin::experimental::ExpertMlp;
/// use incin::nn::Module;
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// let expert = ExpertMlp::<Cpu>::build(8, 16, (), ())?;
/// let x = Tensor::<Dyn, Cpu>::zeros(vec![3, 8])?;
/// assert_eq!(expert.forward(x)?.dims().dims(), &[3, 8]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct ExpertMlp<B: VariableBackend, K: DType = f32, Train: TrainState = Trainable> {
    /// Up-projection `[D_FF, D_MODEL]` in `Linear` (`[out, in]`) orientation.
    pub up: Linear<Dyn, B, False, K, Train>,
    /// Down-projection `[D_MODEL, D_FF]` in `Linear` (`[out, in]`) orientation.
    pub down: Linear<Dyn, B, False, K, Train>,
}

impl<B: VariableBackend, K: DType, Train: TrainState> ShapeInfo for ExpertMlp<B, K, Train> {
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(alloc::string::ToString::to_string("mlp expert"))
    }
}

impl<B: VariableBackend, K: DType> ExpertMlp<B, K, Trainable>
where
    B: crate::backend_authoring::TensorBackend<K> + crate::nn::param::ParameterInit<K>,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds a bias-free `D_MODEL -> D_FF -> D_MODEL` expert.
    pub fn build(
        d_model: usize,
        d_ff: usize,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
    ) -> Result<Self> {
        if d_model == 0 {
            return Err(invalid("build expert mlp", "d_model must be nonzero"));
        }
        if d_ff == 0 {
            return Err(invalid("build expert mlp", "d_ff must be nonzero"));
        }
        Ok(Self {
            up: Linear::build_full(d_model, d_ff, dtype.clone(), device.clone(), ())?,
            down: Linear::build_full(d_ff, d_model, dtype, device, ())?,
        })
    }
}

impl<B, K, Train, G, L> Module<Tensor<Dyn, B, K, G, Local, L>> for ExpertMlp<B, K, Train>
where
    B: VariableBackend
        + SupportsDType<K>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::Relu>,
    K: DType,
    Train: TrainState,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: RequiresGrad + GradJoin<Train::TensorGrad>,
    JoinedGrad<JoinedGrad<G, Train::TensorGrad>, Train::TensorGrad>: RequiresGrad,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<K>>,
{
    type Output =
        Dense<Dyn, B, K, JoinedGrad<JoinedGrad<G, Train::TensorGrad>, Train::TensorGrad>, Local>;
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        let hidden = self.up.forward(x)?.relu()?;
        self.down.forward(hidden)
    }
}

impl<B, K, Train> GroupedExpert<B, K, Train> for ExpertMlp<B, K, Train>
where
    B: VariableBackend + SupportsDType<K> + Execute<op::TransposeExact>,
    K: DType,
    Train: TrainState,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
{
    fn up_matrix(&self) -> Result<Tensor<Dyn, B, K, Train::TensorGrad, Local, Dyn>> {
        Ok(self
            .up
            .weight
            .as_tensor()?
            .transpose(0isize, 1isize)?
            .into_dyn())
    }

    fn down_matrix(&self) -> Result<Tensor<Dyn, B, K, Train::TensorGrad, Local, Dyn>> {
        Ok(self
            .down
            .weight
            .as_tensor()?
            .transpose(0isize, 1isize)?
            .into_dyn())
    }
}

/// A dropless mixture of `E` experts with soft top-k routing (issue #102,
/// prototype).
///
/// The forward path is the option-C target: route, permute into expert order,
/// gather the static `[T*K, D]` buffer, run one `grouped_matmul` per
/// projection against the stacked expert matrices with the `[E+1]`
/// [`Routing::expert_offsets`][crate::nn::Routing] tile, weight by the gate,
/// `scatter_add` back to `[T, D]`, and return the `(combined, aux)` tuple. No
/// token is dropped, nothing is padded, and the per-expert token count never
/// becomes a type.
///
/// `Expert` defaults to the small [`ExpertMlp`] FFN; a custom expert plugs in
/// through [`GroupedExpert`]. All `E` experts plus the router appear in state
/// as `router.gate.weight` and `experts.0 … experts.E-1`, so a checkpoint with
/// a different `E` is refused rather than partially loaded.
///
/// Gradient contract (gate-weight-only, no straight-through): gate weights
/// receive gradients through both the combined output and the aux loss;
/// indices, permutation, offsets, and counts are `NoGrad` by type. See the
/// module-level docs for the prototype deviations this contract rests on
/// (combine seed, input markers, host-readback scatter index).
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_to_device)]
pub struct DroplessMoE<
    const E: usize,
    const TOPK: usize,
    B: VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats = ExpertMlp<B, K, Train>,
> {
    /// Soft top-k router over the expert array.
    pub router: Router<E, TOPK, B, K, Train>,
    /// The `E` expert submodules, visited as `experts.0`, `experts.1`, ...
    pub experts: [Expert; E],
}

impl<const E: usize, const TOPK: usize, B, K, Train, Expert> ShapeInfo
    for DroplessMoE<E, TOPK, B, K, Train, Expert>
where
    B: VariableBackend,
    K: DType,
    Train: TrainState,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    fn shape_info(&self) -> Option<alloc::string::String> {
        Some(format!("dropless E={E}, topk={TOPK}"))
    }
}

impl<const E: usize, const TOPK: usize, B, K, Expert> DroplessMoE<E, TOPK, B, K, Trainable, Expert>
where
    B: VariableBackend,
    K: DType,
    Expert: NamedLayers + ShapeInfo + TrainMode + ComputeStats,
{
    /// Builds the router gate and `E` experts from `make_expert`.
    ///
    /// `make_expert` is called exactly `E` times so each slot can be built
    /// with the caller's own widths; `d_model` is the shared input width the
    /// gate and every expert must accept. With the default [`ExpertMlp`]
    /// expert, `d_ff` is its hidden width.
    pub fn build(
        d_model: usize,
        d_ff: usize,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
        mut make_expert: impl FnMut() -> Result<Expert>,
    ) -> Result<Self>
    where
        B: crate::backend_authoring::TensorBackend<K> + crate::nn::param::ParameterInit<K>,
        <K as DType>::Arg: Clone,
        <B::Device as Device>::Arg: Clone,
    {
        const {
            assert!(E > 0, "dropless MoE needs at least one expert");
            assert!(TOPK > 0, "top-k must be at least 1");
            assert!(TOPK <= E, "top-k cannot exceed the expert count");
        }
        if d_model == 0 {
            return Err(invalid("build dropless moe", "d_model must be nonzero"));
        }
        if d_ff == 0 {
            return Err(invalid("build dropless moe", "d_ff must be nonzero"));
        }
        let router = Router::build(d_model, dtype, device)?;
        let mut experts = Vec::with_capacity(E);
        for _ in 0..E {
            experts.push(make_expert()?);
        }
        let experts = experts.try_into().map_err(|_| Error::InternalInvariant {
            operation: "build dropless moe",
            reason: "expert builder must produce exactly E experts",
        })?;
        Ok(Self { router, experts })
    }
}

impl<const E: usize, const TOPK: usize, Expert, B, K, Train, NewD> ToDevice<B, NewD>
    for DroplessMoE<E, TOPK, B, K, Train, Expert>
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
    type Output = DroplessMoE<E, TOPK, B::Output, K, Train, Expert::Output>;

    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(DroplessMoE {
            router: self.router.to_device(arg)?,
            experts: self.experts.to_device(arg)?,
        })
    }
}

impl<const E: usize, const TOPK: usize, B, K, Train, Expert>
    DroplessMoE<E, TOPK, B, K, Train, Expert>
where
    B: VariableBackend + SupportsDType<K> + Execute<op::UnsqueezeExact> + Execute<op::ConcatExact>,
    K: DType,
    Train: TrainState,
    Expert: GroupedExpert<B, K, Train>,
    <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
{
    /// Stacks every expert's up-projections into one `[E, D_MODEL, D_FF]` tensor.
    ///
    /// Each expert owns its `[D_MODEL, D_FF]` grouped-orientation matrix (see
    /// [`GroupedExpert`]); unsqueezing a leading axis and concatenating along
    /// it packs the `E` slices into the stack `grouped_matmul` consumes. The
    /// stack carries the experts' train-state marker, so the grouped backward
    /// reaches the expert parameters.
    fn stack_up(&self) -> Result<Dense<Dyn, B, K, Train::TensorGrad, Local>> {
        self.stack_matrices(GroupedExpert::up_matrix)
    }

    /// Stacks every expert's down-projections into one `[E, D_FF, D_MODEL]` tensor.
    fn stack_down(&self) -> Result<Dense<Dyn, B, K, Train::TensorGrad, Local>> {
        self.stack_matrices(GroupedExpert::down_matrix)
    }

    /// Packs one grouped-orientation matrix per expert along a fresh axis 0.
    fn stack_matrices(
        &self,
        mut matrix: impl FnMut(&Expert) -> Result<Tensor<Dyn, B, K, Train::TensorGrad, Local, Dyn>>,
    ) -> Result<Dense<Dyn, B, K, Train::TensorGrad, Local>> {
        let mut experts = self.experts.iter();
        let Some(first) = experts.next() else {
            return Err(invalid(
                "dropless moe stack",
                "expert count must be nonzero",
            ));
        };
        let mut stacked = matrix(first)?.unsqueeze(0isize)?;
        for expert in experts {
            let piece = matrix(expert)?.unsqueeze(0isize)?;
            stacked = stacked.concat(&piece, 0isize)?;
        }
        Ok(stacked)
    }
}

impl<const E: usize, const TOPK: usize, B, K, Train, Expert, G, L>
    Module<Tensor<Dyn, B, K, G, Local, L>> for DroplessMoE<E, TOPK, B, K, Train, Expert>
where
    B: DroplessMoEBackend<K>
        + SupportsDType<K>
        + SupportsDType<u32>
        + SupportsDType<i64>
        + HostInterop,
    B::Device: Device<Arg = ()>,
    K: DType<Arg = ()> + FloatDType,
    Train: TrainState,
    Expert: GroupedExpert<B, K, Train>,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    G: GradJoin<JoinedGrad<G, Train::TensorGrad>>,
    L: Layout<Dyn>,
    JoinedGrad<G, Train::TensorGrad>: RequiresGrad,
    Router<E, TOPK, B, K, Train>: Module<
            Tensor<Dyn, B, K, G, Local, L>,
            Output = Routing<E, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    B: Execute<op::FlattenExact, Output = <B as StorageBackend>::Storage<K>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Softmax>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TopK>>::Output: Into<(B::Storage<K>, B::Storage<u32>)>,
    <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::ToDType>>::Output: Into<B::Storage<K>> + Into<B::Storage<i64>>,
    <B as Execute<op::Argsort>>::Output: Into<B::Storage<u32>>,
    <B as Execute<op::RepeatInterleave>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::IndexSelect>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Bincount>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::Cumsum>>::Output: Into<B::Storage<i64>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>> + Into<B::Storage<i64>>,
    <B as Execute<op::Zeros>>::Output: Into<B::Storage<K>> + Into<B::Storage<i64>>,
    <B as Execute<op::GroupedMatMul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ScatterAdd>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MeanDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::DivScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::TensorFromData>>::Output: Into<B::Storage<u32>> + Into<B::Storage<i64>>,
{
    /// `(combined `[T, D]` output, load-balancing aux loss scalar)`.
    ///
    /// The aux loss travels in `Output`, never a side channel, per the #102
    /// decision. The combined tensor is `Grad`-marked (the combine seed
    /// requires grad so `scatter_add` records its tape entry); the aux scalar
    /// joins the input and train-state markers like any differentiable
    /// function of the gate.
    type Output = (
        Dense<Dyn, B, K, Grad, Local>,
        Dense<Nil, B, K, JoinedGrad<G, Train::TensorGrad>, Local>,
    );
    type Error = Error;

    fn forward(
        &self,
        x: Tensor<Dyn, B, K, G, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        if E == 0 {
            return Err(invalid(
                "dropless moe forward",
                "expert count must be nonzero",
            ));
        }
        let dims = x.shape_buf().as_ref().to_vec();
        if dims.len() != 2 {
            return Err(invalid_owned(
                "dropless moe forward",
                format!(
                    "expected a rank-2 [T, D] input, got rank {} with dims {:?}",
                    dims.len(),
                    dims
                ),
            ));
        }
        let (tokens, d_model) = (dims[0], dims[1]);

        // 1. Router: softmax -> top-k -> renormalize. Gate weights stay
        // differentiable; indices do not.
        let routing = self.router.forward(x.clone())?;

        // 2. Option-C geometry: flatten the [T, K] assignment, argsort into
        // expert order, expand the tokens once per slot and gather the static
        // [T*K, D] buffer, scan the [E+1] offsets with no host interop.
        // (`argsort`, like `topk`, is only defined for the default layout, so
        // the proof is dropped before selecting — the `Router::forward` / P1
        // example playbook.)
        //
        // The flatten here is a host roundtrip, deliberately: `flatten`
        // pins `Execute<FlattenExact, Output = Storage<K>>` to the
        // flattened tensor's dtype, and this impl block already pins that
        // equality for the `K`-typed weights below — a second equality for
        // `u32` is unprovable for generic `B` (one impl, one `Output`
        // type). The `[T, K]` index tile is tiny, and the module already
        // reads `perm` back below, so one more small readback keeps every
        // bound honest instead of over-constraining the backend.
        let flat_host: Vec<i64> = routing
            .indices
            .to_vec1::<u32>()?
            .iter()
            .map(|index| *index as i64)
            .collect();
        let flat = Tensor::<Dyn, B, i64>::from_slice(&flat_host, alloc::vec![tokens * TOPK])?;
        let perm = flat.clone().forget_layout().argsort(0, false)?;
        let buffer = x.repeat_interleave(TOPK, 0)?.index_select(0isize, &perm)?;
        let offsets = routing.expert_offsets()?;

        // 3. Grouped GEMMs: every expert's row span meets its own stacked
        // weights in one call per projection, with a ReLU between.
        let hidden = buffer.grouped_matmul(&self.stack_up()?, &offsets)?.relu()?;
        let grouped = hidden.grouped_matmul(&self.stack_down()?, &offsets)?;

        // 4. Gate weighting + scatter-add combine. The gates ride the same
        // permutation into expert order and scale each grouped row; each
        // grouped row lands back in its token's row, accumulating.
        let gate_col = routing
            .weights
            .flatten_runtime(0, 1)?
            .index_select(0isize, &perm)?
            .unsqueeze(1isize)?;
        let weighted = grouped.broadcast_mul(&gate_col)?;
        // Scatter index, exactly as the P1 example builds it: grouped row r
        // holds slot perm[r], which belongs to token perm[r] / TOPK, naming
        // every column of that row so the whole row lands at once. One host
        // readback (see the module docs).
        let slots = tokens * TOPK;
        let perm_host = perm.to_vec1::<u32>()?;
        let mut scatter_host = Vec::with_capacity(slots * d_model);
        for row in &perm_host {
            for _ in 0..d_model {
                scatter_host.push(row / TOPK as u32);
            }
        }
        let scatter_index = Tensor::<Dyn, B, u32>::from_slice(&scatter_host, vec![slots, d_model])?;
        // `require_grad` on the seed is load-bearing, not incidental:
        // `scatter_add` records under the base's mode and returns its marker,
        // so a plain `zeros` base would detach the gate-weight path. Values
        // are untouched — this only keeps the tape entry.
        let base = Tensor::<Dyn, B, K>::zeros(vec![tokens, d_model])?.require_grad();
        let out = base.scatter_add(0isize, &scatter_index, &weighted)?;

        // 5. Aux loss beside the output: Switch-style `E * sum(f * P)`. `f`
        // is the routed fraction per expert (NoGrad counts), `P` the mean
        // gate probability per expert (differentiable, so the gate learns
        // from this term too).
        let frac = routing
            .indices
            .bincount::<E>()?
            .to_dtype::<K>()?
            .div_scalar(slots as f64)?;
        let aux = frac
            .broadcast_mul(&routing.probs.mean(0isize)?)?
            .sum_all()?
            .mul_scalar(E as f64)?;

        Ok((out, aux))
    }
}
