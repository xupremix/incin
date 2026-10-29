//! The gated recurrent unit, alongside [`RNN`](crate::nn::RNN) and
//! [`LSTM`](crate::nn::LSTM).
//!
//! A GRU is the LSTM's recurrence with one fewer gate and no separate cell
//! state: it carries a hidden state only, so [`GRUCell`] takes and returns one
//! tensor where [`LSTMCell`](crate::nn::LSTMCell) takes and returns a pair.
//!
//! # Where the reset gate is applied
//!
//! ```text
//! r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
//! z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//! n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
//! h' = (1 - z) * n + z * h
//! ```
//!
//! The reset gate multiplies the *hidden* projection of the candidate, not the
//! sum of both projections. The difference is not cosmetic: applying `r` after
//! the sum also gates the input contribution, which is a different model and
//! is not what a checkpoint trained elsewhere expects. This is the formulation
//! the reference implementations use, and it is why `wh_n` is projected
//! separately rather than folded into one addition.
//!
//! `h'` is computed as `n + z * (h - n)`, which is the same function
//! rearranged. The direct spelling needs `1 - z`, a scalar minus a tensor,
//! which no catalog row provides; the rearrangement needs only the exact-shape
//! subtract, multiply and add the rest of the recurrence already uses.

use crate::backend_authoring::SupportsDType;
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::op;
use crate::nn::init::Init;
use crate::nn::linear::LinearShape;
use crate::nn::optional::{False, True};
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Linear, Module, VisitParameters};
use crate::shapes::Layout;
use crate::shapes::shape::{DimCons, Nil};
use crate::shapes::{Dim, Dyn, DynShape, Shape, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{GradJoin, JoinedGrad, RequiresGrad};
use alloc::format;
use alloc::vec::Vec;

type D2<A, B> = DimCons<A, DimCons<B, Nil>>;
type D3<A, B, C> = DimCons<A, DimCons<B, DimCons<C, Nil>>>;

/// A shape marker trait specifying the input and output features of a
/// [`GRUCell`] / [`GRU`].
///
/// Supply `(In, Out)` for fully static dimensions, or [`Dyn`] for fully
/// runtime sizes. Structurally identical to
/// [`RnnShape`](crate::nn::RnnShape) and [`LstmShape`](crate::nn::LstmShape):
/// each recurrent layer resolves its builder against its own marker, so a
/// shape written for one cannot be handed to another by accident.
pub trait GruShape: Shape + DynShape {
    /// `In`.
    type In: Dim;
    /// `Out`.
    type Out: Dim;
    /// Input-to-hidden projection geometry.
    type IhShape: LinearShape<InF = Self::In, OutF = Self::Out>;
    /// Hidden-to-hidden recurrence geometry.
    type HhShape: LinearShape<InF = Self::Out, OutF = Self::Out>;
}

impl<In: Dim, Out: Dim> GruShape for D2<In, Out> {
    type In = In;
    type Out = Out;
    type IhShape = D2<In, Out>;
    type HhShape = D2<Out, Out>;
}

impl GruShape for Dyn {
    /// `In`.
    type In = usize;
    /// `Out`.
    type Out = usize;
    type IhShape = Dyn;
    type HhShape = Dyn;
}

// ---------------------------------------------------------------------------
// GRUCellBuilder: typestate builder
// ---------------------------------------------------------------------------

/// A builder for constructing a [`GRUCell`] before target-based initialization.
///
/// Stores the layer geometry ([`ShapeValue`]), weight and bias initializer
/// policies (grouped semantically as input and hidden), and compile-time
/// typestate parameters for bias presence and trainability.
pub struct GRUCellBuilder<
    S: GruShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// Shape specification (encodes `[in_features, out_features]`).
    pub shape: ShapeValue<S>,
    /// Initializer for all input-to-hidden weight matrices.
    pub input_weight_init: Init,
    /// Initializer for all hidden-to-hidden weight matrices.
    pub hidden_weight_init: Init,
    /// Initializer for all input-to-hidden bias vectors.
    pub input_bias_init: Init,
    /// Initializer for all hidden-to-hidden bias vectors.
    pub hidden_bias_init: Init,
    /// Bias-presence and train-state markers.
    pub _phantom: core::marker::PhantomData<(BiasIh, BiasHh, Train)>,
}

impl<
    S: GruShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> GRUCellBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        &self.shape
    }

    /// Removes input-to-hidden biases from the built cell.
    pub fn no_input_bias(self) -> GRUCellBuilder<S, False, BiasHh, Train> {
        GRUCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes hidden-to-hidden biases from the built cell.
    pub fn no_hidden_bias(self) -> GRUCellBuilder<S, BiasIh, False, Train> {
        GRUCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes all biases from the built cell.
    pub fn no_bias(self) -> GRUCellBuilder<S, False, False, Train> {
        GRUCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Marks the created cell parameters as frozen (non-trainable).
    pub fn frozen(self) -> GRUCellBuilder<S, BiasIh, BiasHh, Frozen> {
        GRUCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the initializer used for all input-to-hidden weight matrices.
    pub fn input_weight_init(mut self, init: Init) -> Self {
        self.input_weight_init = init;
        self
    }

    /// Sets the initializer used for all hidden-to-hidden weight matrices.
    pub fn hidden_weight_init(mut self, init: Init) -> Self {
        self.hidden_weight_init = init;
        self
    }

    /// Sets the initializer used for all input-to-hidden bias vectors.
    pub fn input_bias_init(mut self, init: Init) -> Self {
        self.input_bias_init = init;
        self
    }

    /// Sets the initializer used for all hidden-to-hidden bias vectors.
    pub fn hidden_bias_init(mut self, init: Init) -> Self {
        self.hidden_bias_init = init;
        self
    }
}

/// Free constructor for a backend-independent [`GRUCellBuilder`].
pub fn gru_cell<S: GruShape>(shape: ShapeValue<S>) -> GRUCellBuilder<S> {
    let init = crate::nn::init::kaiming_uniform();
    GRUCellBuilder {
        shape,
        input_weight_init: init,
        hidden_weight_init: init,
        input_bias_init: init,
        hidden_bias_init: init,
        _phantom: core::marker::PhantomData,
    }
}

// ---------------------------------------------------------------------------
// GRUBuilder: typestate builder for the multi-step GRU
// ---------------------------------------------------------------------------

/// A builder for constructing a [`GRU`] before target-based initialization.
///
/// Wraps a [`GRUCellBuilder`] and exposes the same bias/trainability controls.
pub struct GRUBuilder<
    S: GruShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// The wrapped single-step builder.
    pub cell: GRUCellBuilder<S, BiasIh, BiasHh, Train>,
}

impl<
    S: GruShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> GRUBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        self.cell.shape()
    }

    /// Removes input-to-hidden biases from the built layer.
    pub fn no_input_bias(self) -> GRUBuilder<S, False, BiasHh, Train> {
        GRUBuilder {
            cell: self.cell.no_input_bias(),
        }
    }

    /// Removes hidden-to-hidden biases from the built layer.
    pub fn no_hidden_bias(self) -> GRUBuilder<S, BiasIh, False, Train> {
        GRUBuilder {
            cell: self.cell.no_hidden_bias(),
        }
    }

    /// Removes all biases from the built layer.
    pub fn no_bias(self) -> GRUBuilder<S, False, False, Train> {
        GRUBuilder {
            cell: self.cell.no_bias(),
        }
    }

    /// Marks the created parameters as frozen (non-trainable).
    pub fn frozen(self) -> GRUBuilder<S, BiasIh, BiasHh, Frozen> {
        GRUBuilder {
            cell: self.cell.frozen(),
        }
    }

    /// Sets the initializer used for all input-to-hidden weight matrices.
    pub fn input_weight_init(mut self, init: Init) -> Self {
        self.cell = self.cell.input_weight_init(init);
        self
    }

    /// Sets the initializer used for all hidden-to-hidden weight matrices.
    pub fn hidden_weight_init(mut self, init: Init) -> Self {
        self.cell = self.cell.hidden_weight_init(init);
        self
    }

    /// Sets the initializer used for all input-to-hidden bias vectors.
    pub fn input_bias_init(mut self, init: Init) -> Self {
        self.cell = self.cell.input_bias_init(init);
        self
    }

    /// Sets the initializer used for all hidden-to-hidden bias vectors.
    pub fn hidden_bias_init(mut self, init: Init) -> Self {
        self.cell = self.cell.hidden_bias_init(init);
        self
    }
}

/// Free constructor for a backend-independent [`GRUBuilder`].
pub fn gru<S: GruShape>(shape: ShapeValue<S>) -> GRUBuilder<S> {
    GRUBuilder {
        cell: gru_cell(shape),
    }
}

// ---------------------------------------------------------------------------
// GRUCell
// ---------------------------------------------------------------------------

/// A single GRU step cell implementing the standard 3-gate recurrence.
///
/// * `S`: [`GruShape`]: `(In, Out)` static or [`Dyn`] for runtime sizes.
/// * `BiasIh`: whether input-to-hidden biases exist: [`True`], [`False`].
/// * `BiasHh`: whether hidden-to-hidden biases exist: [`True`], [`False`].
/// * `K`: parameter dtype (default: `f32`).
/// * `Train`: trainability typestate (default: [`Trainable`]).
///
/// ## Examples
///
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::nn::{GRUCell, Module};
/// use incin::prelude::*;
/// # type Cpu = incin_backends::cpu::CpuBackendImpl;
///
/// # fn main() -> Result<()> {
/// let cell = GRUCell::<s![4, 3], Cpu>::build(())?;
/// let x = Tensor::<s![2, 4], Cpu>::zeros(())?.require_grad();
/// let h = Tensor::<s![2, 3], Cpu>::zeros(())?.require_grad();
/// assert_eq!(cell.forward((x, h))?.dims().dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct GRUCell<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
> where
    S::IhShape: LinearShape,
    S::HhShape: LinearShape,
{
    /// Input projection of the reset gate.
    pub wi_r: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// Input projection of the update gate.
    pub wi_z: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// Input projection of the candidate state.
    pub wi_n: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// Recurrent projection of the reset gate.
    pub wh_r: Linear<S::HhShape, B, BiasHh, K, Train>,
    /// Recurrent projection of the update gate.
    pub wh_z: Linear<S::HhShape, B, BiasHh, K, Train>,
    /// Recurrent projection of the candidate state, which the reset gate
    /// multiplies.
    pub wh_n: Linear<S::HhShape, B, BiasHh, K, Train>,
}

impl<S, B, BiasIh, BiasHh, K: DType, Train: TrainState> GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    S: GruShape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    <K as DType>::Arg: Clone,
    <S::In as Dim>::Arg: Clone,
    <S::Out as Dim>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
    <BiasIh as crate::nn::optional::OptionalField>::Arg: Clone,
    <BiasHh as crate::nn::optional::OptionalField>::Arg: Clone,
{
    /// Constructs the cell directly from explicit arguments (the "build" path).
    ///
    /// Prefer the target-aware [`GRUCellBuilder`] path for new code.
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::In as Dim>::Arg,
                <S::Out as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                <BiasIh as crate::nn::optional::OptionalField>::Arg,
                <BiasHh as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        let (input, output, dtype, device, bias_ih, bias_hh) = args.into_layer_arg();
        let input_projection = |bias: <BiasIh as crate::nn::optional::OptionalField>::Arg| {
            Linear::<S::IhShape, B, BiasIh, K, Train>::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias,
            )
        };
        let wi_r = input_projection(bias_ih.clone())?;
        let wi_z = input_projection(bias_ih.clone())?;
        let wi_n = input_projection(bias_ih)?;

        let recurrent_projection = |bias: <BiasHh as crate::nn::optional::OptionalField>::Arg| {
            Linear::<S::HhShape, B, BiasHh, K, Train>::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias,
            )
        };
        let wh_r = recurrent_projection(bias_hh.clone())?;
        let wh_z = recurrent_projection(bias_hh.clone())?;
        let wh_n = recurrent_projection(bias_hh)?;

        Ok(Self {
            wi_r,
            wi_z,
            wi_n,
            wh_r,
            wh_z,
            wh_n,
        })
    }
}

impl<S, B, BiasIh, BiasHh, K: DType, Train: TrainState> GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
{
    /// Converts this cell's parameters to frozen typestate.
    pub fn freeze(self) -> GRUCell<S, B, BiasIh, BiasHh, K, Frozen> {
        GRUCell {
            wi_r: self.wi_r.freeze(),
            wi_z: self.wi_z.freeze(),
            wi_n: self.wi_n.freeze(),
            wh_r: self.wh_r.freeze(),
            wh_z: self.wh_z.freeze(),
            wh_n: self.wh_n.freeze(),
        }
    }

    /// Converts this cell's parameters to trainable typestate.
    pub fn unfreeze(self) -> GRUCell<S, B, BiasIh, BiasHh, K, Trainable> {
        GRUCell {
            wi_r: self.wi_r.unfreeze(),
            wi_z: self.wi_z.unfreeze(),
            wi_n: self.wi_n.unfreeze(),
            wh_r: self.wh_r.unfreeze(),
            wh_z: self.wh_z.unfreeze(),
            wh_n: self.wh_n.unfreeze(),
        }
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> VisitParameters<B> for GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: VisitParameters<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: VisitParameters<B>,
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.wi_r
            .visit_parameters(&path.try_child("wi_r")?, visitor)?;
        self.wi_z
            .visit_parameters(&path.try_child("wi_z")?, visitor)?;
        self.wi_n
            .visit_parameters(&path.try_child("wi_n")?, visitor)?;
        self.wh_r
            .visit_parameters(&path.try_child("wh_r")?, visitor)?;
        self.wh_z
            .visit_parameters(&path.try_child("wh_z")?, visitor)?;
        self.wh_n
            .visit_parameters(&path.try_child("wh_n")?, visitor)
    }
}

impl<
    S: GruShape,
    Batch: Dim,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::Add>
        + Execute<op::Sub>
        + Execute<op::Mul>
        + Execute<op::Sigmoid>
        + Execute<op::Tanh>,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    G: RequiresGrad + GradJoin<Train::TensorGrad>,
    // One layout parameter serves both the input and the hidden tensors,
    // which are different shapes, so it has to describe both, and `Linear`'s
    // own impl asks for the two it needs to restate the operand as `Dyn`
    // before the matmul.
    L: Layout<D2<Batch, S::In>> + Layout<D2<Batch, S::Out>> + Layout<Dyn> + crate::shapes::Restatable,
>
    Module<(
        Tensor<D2<Batch, S::In>, B, K, G, Local, L>,
        Tensor<D2<Batch, S::Out>, B, K, G, Local, L>,
    )> for GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: Module<
            Tensor<D2<Batch, S::In>, B, K, G, Local, L>,
            Output = Tensor<D2<Batch, S::Out>, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    Linear<S::HhShape, B, BiasHh, K, Train>: Module<
            Tensor<D2<Batch, S::Out>, B, K, G, Local, L>,
            Output = Tensor<D2<Batch, S::Out>, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    // Each gate joins the operand with the parameters, and the update then
    // joins the gates with one another and with the incoming state, so the
    // requirement has to be idempotent under a second join or the result type
    // grows a layer per gate.
    JoinedGrad<G, Train::TensorGrad>: GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>
        + GradJoin<G, Output = JoinedGrad<G, Train::TensorGrad>>,
    G: GradJoin<JoinedGrad<G, Train::TensorGrad>, Output = JoinedGrad<G, Train::TensorGrad>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sigmoid>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Tanh>>::Output: Into<B::Storage<K>>,
{
    /// The next hidden state. A GRU carries no separate cell state, so this is
    /// one tensor where the LSTM's is a pair.
    type Output = Tensor<
        D2<Batch, S::Out>,
        B,
        K,
        JoinedGrad<G, Train::TensorGrad>,
        Local,
        crate::shapes::RowMajor,
    >;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        (x, h_prev): (
            Tensor<D2<Batch, S::In>, B, K, G, Local, L>,
            Tensor<D2<Batch, S::Out>, B, K, G, Local, L>,
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let r = self
            .wi_r
            .forward(x.clone())?
            .add_exact(&self.wh_r.forward(h_prev.clone())?)?
            .sigmoid()?;
        let z = self
            .wi_z
            .forward(x.clone())?
            .add_exact(&self.wh_z.forward(h_prev.clone())?)?
            .sigmoid()?;
        // The reset gate multiplies the recurrent projection alone, not the
        // sum: gating the input contribution as well is a different model and
        // not what a checkpoint trained elsewhere expects.
        let n = self
            .wi_n
            .forward(x)?
            .add_exact(&r.mul_exact(&self.wh_n.forward(h_prev.clone())?)?)?
            .tanh()?;
        // `n + z * (h - n)` rather than `(1 - z) * n + z * h`: the same
        // function, without needing a scalar-minus-tensor row.
        let carried = h_prev.sub_exact(&n)?;
        n.add_exact(&z.mul_exact(&carried)?)
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::module::NamedLayers,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::module::NamedLayers,
{
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        let child = |name: &str| {
            if prefix.is_empty() {
                alloc::string::String::from(name)
            } else {
                format!("{prefix}.{name}")
            }
        };
        let mut children = Vec::new();
        children.extend(self.wi_r.layer_structure(&child("wi_r")));
        children.extend(self.wi_z.layer_structure(&child("wi_z")));
        children.extend(self.wi_n.layer_structure(&child("wi_n")));
        children.extend(self.wh_r.layer_structure(&child("wh_r")));
        children.extend(self.wh_z.layer_structure(&child("wh_z")));
        children.extend(self.wh_n.layer_structure(&child("wh_n")));
        let name = if prefix.is_empty() {
            alloc::string::String::from("GRUCell")
        } else {
            alloc::string::String::from(prefix)
        };
        Vec::from([crate::nn::module::LayerNode {
            name,
            type_name: alloc::string::String::from("GRUCell"),
            shape_info: alloc::string::String::new(),
            children,
        }])
    }
}

// ---------------------------------------------------------------------------
// GRU (multi-step wrapper)
// ---------------------------------------------------------------------------

/// A gated recurrent unit that processes an input sequence step by step.
///
/// Wraps a [`GRUCell`] and iterates the sequence dimension. `forward` takes
/// `(sequence, initial_hidden)` and returns `(outputs, final_hidden)`.
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct GRU<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The wrapped single-step cell.
    pub cell: GRUCell<S, B, BiasIh, BiasHh, K, Train>,
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> GRU<S, B, BiasIh, BiasHh, K, Train>
{
    /// Creates a new instance from a pre-built cell.
    pub fn new(cell: GRUCell<S, B, BiasIh, BiasHh, K, Train>) -> Self {
        Self { cell }
    }

    /// Converts this layer's parameters to frozen typestate.
    pub fn freeze(self) -> GRU<S, B, BiasIh, BiasHh, K, Frozen> {
        GRU {
            cell: self.cell.freeze(),
        }
    }

    /// Converts this layer's parameters to trainable typestate.
    pub fn unfreeze(self) -> GRU<S, B, BiasIh, BiasHh, K, Trainable> {
        GRU {
            cell: self.cell.unfreeze(),
        }
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> VisitParameters<B> for GRU<S, B, BiasIh, BiasHh, K, Train>
where
    GRUCell<S, B, BiasIh, BiasHh, K, Train>: VisitParameters<B>,
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.cell
            .visit_parameters(&path.try_child("cell")?, visitor)
    }
}

impl<
    S: GruShape,
    Batch: Dim<Arg = ()>,
    Seq: Dim<Arg = ()>,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::StackExact>
        + Execute<op::Narrow>
        + Execute<op::SqueezeExact>
        + crate::exec::Capabilities,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
>
    Module<(
        Tensor<D3<Batch, Seq, S::In>, B, K, G>,
        Tensor<D2<Batch, S::Out>, B, K, G>,
    )> for GRU<S, B, BiasIh, BiasHh, K, Train>
where
    <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    S::In: Dim<Arg = ()>,
    S::Out: Dim<Arg = ()>,
    // The state is carried across steps, so the per-step join has to land
    // back on the requirement the loop variable already holds.
    G: GradJoin<Train::TensorGrad, Output = G>,
    GRUCell<S, B, BiasIh, BiasHh, K, Train>: Module<
            (
                Tensor<D2<Batch, S::In>, B, K, G>,
                Tensor<D2<Batch, S::Out>, B, K, G>,
            ),
            Output = Tensor<D2<Batch, S::Out>, B, K, G, Local, crate::shapes::RowMajor>,
            Error = Error,
        >,
{
    /// The per-step outputs stacked along the sequence axis, and the final
    /// hidden state.
    type Output = (
        Tensor<D3<Batch, Seq, S::Out>, B, K, G>,
        Tensor<D2<Batch, S::Out>, B, K, G>,
    );
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        (x, mut h): (
            Tensor<D3<Batch, Seq, S::In>, B, K, G>,
            Tensor<D2<Batch, S::Out>, B, K, G>,
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let seq_len = Seq::static_size().map_err(Error::Shape)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let x_step = x.clone().try_narrow(1isize, i, 1)?.try_squeeze(1isize)?;
            // The cell binds its input layout to the default, so the proof
            // `try_squeeze` produced is dropped here rather than forced on it.
            let x_step_static: Tensor<D2<Batch, S::In>, B, K, G> =
                x_step.into_shape::<D2<Batch, S::In>>()?.forget_layout();
            // `h` is carried across iterations and its initial value is the
            // caller's, which proves nothing, so the loop variable can hold
            // only what every assignment to it satisfies.
            h = self.cell.forward((x_step_static, h))?.forget_layout();
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B, K, G>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::manipulation::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<D3<Batch, Seq, S::Out>, B, K, G> = stacked_dyn.into_shape()?;

        Ok((stacked, h))
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for GRU<S, B, BiasIh, BiasHh, K, Train>
where
    GRUCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::module::NamedLayers,
{
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        let child = if prefix.is_empty() {
            alloc::string::String::from("cell")
        } else {
            format!("{prefix}.cell")
        };
        let name = if prefix.is_empty() {
            alloc::string::String::from("GRU")
        } else {
            alloc::string::String::from(prefix)
        };
        Vec::from([crate::nn::module::LayerNode {
            name,
            type_name: alloc::string::String::from("GRU"),
            shape_info: alloc::string::String::new(),
            children: self.cell.layer_structure(&child),
        }])
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitState<B> for GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::VisitState<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::VisitState<B>,
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.wi_r.visit_state(&path.try_child("wi_r")?, visitor)?;
        self.wi_z.visit_state(&path.try_child("wi_z")?, visitor)?;
        self.wi_n.visit_state(&path.try_child("wi_n")?, visitor)?;
        self.wh_r.visit_state(&path.try_child("wh_r")?, visitor)?;
        self.wh_z.visit_state(&path.try_child("wh_z")?, visitor)?;
        self.wh_n.visit_state(&path.try_child("wh_n")?, visitor)
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitStateMut<B> for GRUCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::VisitStateMut<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::VisitStateMut<B>,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.wi_r
            .visit_state_mut(&path.try_child("wi_r")?, visitor)?;
        self.wi_z
            .visit_state_mut(&path.try_child("wi_z")?, visitor)?;
        self.wi_n
            .visit_state_mut(&path.try_child("wi_n")?, visitor)?;
        self.wh_r
            .visit_state_mut(&path.try_child("wh_r")?, visitor)?;
        self.wh_z
            .visit_state_mut(&path.try_child("wh_z")?, visitor)?;
        self.wh_n.visit_state_mut(&path.try_child("wh_n")?, visitor)
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitState<B> for GRU<S, B, BiasIh, BiasHh, K, Train>
where
    GRUCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::VisitState<B>,
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.cell.visit_state(&path.try_child("cell")?, visitor)
    }
}

impl<
    S: GruShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitStateMut<B> for GRU<S, B, BiasIh, BiasHh, K, Train>
where
    GRUCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::VisitStateMut<B>,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        self.cell.visit_state_mut(&path.try_child("cell")?, visitor)
    }
}
