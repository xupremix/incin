use crate::err::Error;
use crate::exec::catalog::op;
use crate::nn::init::Init;
use crate::nn::linear::LinearShape;
use crate::nn::optional::{False, True};
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Linear, Module, VisitParameters};
use crate::shapes::shape::{DimCons, Nil};
use crate::shapes::{Dim, Dyn, DynShape, Shape, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::ConstDevice;
use crate::tensor::dtype::{ConstDType, DType};
use crate::tensor::grad::{Grad, GradJoin, JoinedGrad, RequiresGrad};
use alloc::string::ToString;
use alloc::vec::Vec;

type D2<A, B> = DimCons<A, DimCons<B, Nil>>;
type D3<A, B, C> = DimCons<A, DimCons<B, DimCons<C, Nil>>>;

/// A shape marker trait specifying the input and output features of an [`RNNCell`].
///
/// The typical usage is to supply a 2-tuple `(In, Out)` where:
/// * `In`: Number of input features.
/// * `Out`: Number of output/hidden features.
pub trait RnnShape: Shape + DynShape {
    /// `In`.
    type In: Dim;
    /// `Out`.
    type Out: Dim;
    type IhShape: LinearShape<InF = Self::In, OutF = Self::Out>;
    type HhShape: LinearShape<InF = Self::Out, OutF = Self::Out>;
}

impl<In: Dim, Out: Dim> RnnShape
    for crate::shapes::shape::DimCons<
        In,
        crate::shapes::shape::DimCons<Out, crate::shapes::shape::Nil>,
    >
{
    type In = In;
    type Out = Out;
    type IhShape = crate::shapes::shape::DimCons<
        In,
        crate::shapes::shape::DimCons<Out, crate::shapes::shape::Nil>,
    >;
    type HhShape = crate::shapes::shape::DimCons<
        Out,
        crate::shapes::shape::DimCons<Out, crate::shapes::shape::Nil>,
    >;
}

impl RnnShape for Dyn {
    /// `In`.
    type In = usize;
    /// `Out`.
    type Out = usize;
    type IhShape = Dyn;
    type HhShape = Dyn;
}

// ---------------------------------------------------------------------------
// RNNCellBuilder: typestate builder for RNNCell
// ---------------------------------------------------------------------------

/// A builder for constructing an [`RNNCell`] before target-based initialization.
///
/// Stores the layer geometry ([`ShapeValue`]), weight and bias initializer policies
/// for both the input-to-hidden and hidden-to-hidden linear projections, and
/// compile-time typestate parameters for bias presence and trainability.
pub struct RNNCellBuilder<
    S: RnnShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// Shape specification (encodes `[in_features, out_features]`).
    pub shape: ShapeValue<S>,
    /// Initializer for the input-to-hidden weight matrix (`W_ih`).
    pub input_weight_init: Init,
    /// Initializer for the hidden-to-hidden weight matrix (`W_hh`).
    pub hidden_weight_init: Init,
    /// Initializer for the input-to-hidden bias vector (`b_ih`).
    pub input_bias_init: Init,
    /// Initializer for the hidden-to-hidden bias vector (`b_hh`).
    pub hidden_bias_init: Init,
    pub _phantom: core::marker::PhantomData<(BiasIh, BiasHh, Train)>,
}

impl<
    S: RnnShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> RNNCellBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        &self.shape
    }

    /// Removes input-to-hidden bias from the built cell.
    pub fn no_input_bias(self) -> RNNCellBuilder<S, False, BiasHh, Train> {
        RNNCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes hidden-to-hidden bias from the built cell.
    pub fn no_hidden_bias(self) -> RNNCellBuilder<S, BiasIh, False, Train> {
        RNNCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes both biases from the built cell.
    pub fn no_bias(self) -> RNNCellBuilder<S, False, False, Train> {
        RNNCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Marks the created cell parameters as frozen (non-trainable).
    pub fn frozen(self) -> RNNCellBuilder<S, BiasIh, BiasHh, Frozen> {
        RNNCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets the input-to-hidden weight initializer.
    pub fn input_weight_init(mut self, init: Init) -> Self {
        self.input_weight_init = init;
        self
    }

    /// Sets the hidden-to-hidden weight initializer.
    pub fn hidden_weight_init(mut self, init: Init) -> Self {
        self.hidden_weight_init = init;
        self
    }

    /// Sets the input-to-hidden bias initializer.
    pub fn input_bias_init(mut self, init: Init) -> Self {
        self.input_bias_init = init;
        self
    }

    /// Sets the hidden-to-hidden bias initializer.
    pub fn hidden_bias_init(mut self, init: Init) -> Self {
        self.hidden_bias_init = init;
        self
    }
}

/// Free constructor for a backend-independent [`RNNCellBuilder`].
pub fn rnn_cell<S: RnnShape>(shape: ShapeValue<S>) -> RNNCellBuilder<S> {
    let init = crate::nn::init::kaiming_uniform();
    RNNCellBuilder {
        shape,
        input_weight_init: init,
        hidden_weight_init: init,
        input_bias_init: init,
        hidden_bias_init: init,
        _phantom: core::marker::PhantomData,
    }
}

// ---------------------------------------------------------------------------
// RNNBuilder: typestate builder for RNN
// ---------------------------------------------------------------------------

/// A builder for constructing an [`RNN`] before target-based initialization.
///
/// Wraps an [`RNNCellBuilder`] and exposes the same bias/trainability controls.
pub struct RNNBuilder<
    S: RnnShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// The inner cell builder.
    pub cell: RNNCellBuilder<S, BiasIh, BiasHh, Train>,
}

impl<
    S: RnnShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> RNNBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        self.cell.shape()
    }

    /// Removes input-to-hidden bias from the built RNN.
    pub fn no_input_bias(self) -> RNNBuilder<S, False, BiasHh, Train> {
        RNNBuilder {
            cell: self.cell.no_input_bias(),
        }
    }

    /// Removes hidden-to-hidden bias from the built RNN.
    pub fn no_hidden_bias(self) -> RNNBuilder<S, BiasIh, False, Train> {
        RNNBuilder {
            cell: self.cell.no_hidden_bias(),
        }
    }

    /// Removes both biases from the built RNN.
    pub fn no_bias(self) -> RNNBuilder<S, False, False, Train> {
        RNNBuilder {
            cell: self.cell.no_bias(),
        }
    }

    /// Marks the created RNN parameters as frozen (non-trainable).
    pub fn frozen(self) -> RNNBuilder<S, BiasIh, BiasHh, Frozen> {
        RNNBuilder {
            cell: self.cell.frozen(),
        }
    }

    /// Sets the input-to-hidden weight initializer.
    pub fn input_weight_init(mut self, init: Init) -> Self {
        self.cell = self.cell.input_weight_init(init);
        self
    }

    /// Sets the hidden-to-hidden weight initializer.
    pub fn hidden_weight_init(mut self, init: Init) -> Self {
        self.cell = self.cell.hidden_weight_init(init);
        self
    }

    /// Sets the input-to-hidden bias initializer.
    pub fn input_bias_init(mut self, init: Init) -> Self {
        self.cell = self.cell.input_bias_init(init);
        self
    }

    /// Sets the hidden-to-hidden bias initializer.
    pub fn hidden_bias_init(mut self, init: Init) -> Self {
        self.cell = self.cell.hidden_bias_init(init);
        self
    }
}

/// Free constructor for a backend-independent [`RNNBuilder`].
pub fn rnn<S: RnnShape>(shape: ShapeValue<S>) -> RNNBuilder<S> {
    RNNBuilder {
        cell: rnn_cell(shape),
    }
}

// ---------------------------------------------------------------------------
// RNNCell: struct and impls
// ---------------------------------------------------------------------------

/// A single RNN step cell computing `h_t = tanh(W_ih * x_t + W_hh * h_{t-1})`.
///
/// This implements the basic Elman RNN cell (no gating mechanism). Each time step,
/// it takes an input `x_t: Tensor<D2<Batch, In>>` and the previous hidden state
/// `h_{t-1}: Tensor<D2<Batch, Out>>` and outputs the new hidden state `h_t`.
///
/// For a multi-step sequence model, wrap this in [`RNN`].
#[derive(Debug, Clone)]
pub struct RNNCell<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
> where
    S::IhShape: LinearShape,
    S::HhShape: LinearShape,
{
    /// `wi`: input-to-hidden linear projection.
    pub wi: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// `wh`: hidden-to-hidden linear projection.
    pub wh: Linear<S::HhShape, B, BiasHh, K, Train>,
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> RNNCell<S, B, BiasIh, BiasHh, K, Train>
{
    /// Creates a new instance from pre-built linear projections.
    pub fn new(
        wi: Linear<S::IhShape, B, BiasIh, K, Train>,
        wh: Linear<S::HhShape, B, BiasHh, K, Train>,
    ) -> Self {
        Self { wi, wh }
    }

    /// Converts this cell's parameters to frozen typestate.
    pub fn freeze(self) -> RNNCell<S, B, BiasIh, BiasHh, K, Frozen> {
        RNNCell {
            wi: self.wi.freeze(),
            wh: self.wh.freeze(),
        }
    }

    /// Converts this cell's parameters to trainable typestate.
    pub fn unfreeze(self) -> RNNCell<S, B, BiasIh, BiasHh, K, Trainable> {
        RNNCell {
            wi: self.wi.unfreeze(),
            wh: self.wh.unfreeze(),
        }
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> VisitParameters<B> for RNNCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: VisitParameters<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: VisitParameters<B>,
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.wi.visit_parameters(&path.try_child("wi")?, visitor)?;
        self.wh.visit_parameters(&path.try_child("wh")?, visitor)
    }
}

impl<
    S: RnnShape,
    Batch: Dim,
    B: crate::tensor::backend::VariableBackend + Execute<op::Add> + Execute<op::Tanh>,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
>
    Module<(
        Tensor<D2<Batch, S::In>, B, K, G>,
        Tensor<D2<Batch, S::Out>, B, K, G>,
    )> for RNNCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: Module<
            Tensor<D2<Batch, S::In>, B, K, G>,
            Output = Tensor<D2<Batch, S::Out>, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    Linear<S::HhShape, B, BiasHh, K, Train>: Module<
            Tensor<D2<Batch, S::Out>, B, K, G>,
            Output = Tensor<D2<Batch, S::Out>, B, K, JoinedGrad<G, Train::TensorGrad>>,
            Error = Error,
        >,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Tanh>>::Output: Into<B::Storage<K>>,
    G: GradJoin<Train::TensorGrad>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<D2<Batch, S::Out>, B, K, JoinedGrad<G, Train::TensorGrad>>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, h_prev): (
            Tensor<D2<Batch, S::In>, B, K, G>,
            Tensor<D2<Batch, S::Out>, B, K, G>,
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let i = self.wi.forward(x)?;
        let h = self.wh.forward(h_prev)?;
        // h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
        let sum = i.add_exact(&h)?;
        sum.tanh()
    }
}

// ---------------------------------------------------------------------------
// RNN: multi-step sequence wrapper
// ---------------------------------------------------------------------------

/// An Elman Recurrent Neural Network (RNN) that processes an input sequence step-by-step.
///
/// Wraps an [`RNNCell`] to iterate over the sequence dimension automatically.
/// The `forward` method accepts a pair of `(sequence, initial_hidden_state)` and
/// returns `(all_outputs, final_hidden_state)`.
///
/// ## Type Parameters
/// * `S`: An `RnnShape` describing the input and output feature dimensions.
///
/// ## Examples
/// ```rust,no_run
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
///
/// let cell = RNNCell::new(
///     Linear::<s![10, 20], DefaultBackend>::build(())?,
///     Linear::<s![20, 20], DefaultBackend>::build(())?,
/// );
/// let rnn = RNN::<s![10, 20], DefaultBackend>::new(cell);
/// let input = Tensor::<s![2, 5, 10], DefaultBackend>::zeros(())
///     .unwrap()
///     .require_grad();
/// let h0 = Tensor::<s![2, 20], DefaultBackend>::zeros(())
///     .unwrap()
///     .require_grad();
/// let (output, h_n) = rnn.forward((input, h0)).unwrap();
/// // output shape: [2, 5, 20], h_n shape: [2, 20]
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct RNN<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
    G: RequiresGrad = Grad,
> {
    /// `cell`.
    pub cell: RNNCell<S, B, BiasIh, BiasHh, K, Train>,
    _grad: core::marker::PhantomData<G>,
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> RNN<S, B, BiasIh, BiasHh, K, Train, G>
{
    /// Creates a new instance from a pre-built cell.
    pub fn new(cell: RNNCell<S, B, BiasIh, BiasHh, K, Train>) -> Self {
        Self {
            cell,
            _grad: core::marker::PhantomData,
        }
    }

    /// Converts this RNN's parameters to frozen typestate.
    pub fn freeze(self) -> RNN<S, B, BiasIh, BiasHh, K, Frozen, G> {
        RNN {
            cell: self.cell.freeze(),
            _grad: core::marker::PhantomData,
        }
    }

    /// Converts this RNN's parameters to trainable typestate.
    pub fn unfreeze(self) -> RNN<S, B, BiasIh, BiasHh, K, Trainable, G> {
        RNN {
            cell: self.cell.unfreeze(),
            _grad: core::marker::PhantomData,
        }
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    G: RequiresGrad,
> VisitParameters<B> for RNN<S, B, BiasIh, BiasHh, K, Train, G>
where
    RNNCell<S, B, BiasIh, BiasHh, K, Train>: VisitParameters<B>,
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.cell
            .visit_parameters(&path.try_child("cell")?, visitor)
    }
}

impl<
    S: RnnShape,
    Batch: Dim<Arg = ()>,
    Seq: Dim<Arg = ()>,
    B: crate::tensor::backend::VariableBackend
        + Execute<op::StackExact>
        + Execute<op::Narrow>
        + Execute<op::SqueezeExact>
        + crate::exec::Capabilities,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: ConstDType,
    G: RequiresGrad,
    Train: TrainState,
>
    Module<(
        Tensor<D3<Batch, Seq, S::In>, B, K, G>,
        Tensor<D2<Batch, S::Out>, B, K, G>,
    )> for RNN<S, B, BiasIh, BiasHh, K, Train, G>
where
    <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    S::In: Dim<Arg = ()>,
    S::Out: Dim<Arg = ()>,
    K: ConstDType,
    B::Device: ConstDevice,
    G: GradJoin<Train::TensorGrad, Output = G>,
    RNNCell<S, B, BiasIh, BiasHh, K, Train>: Module<
            (
                Tensor<D2<Batch, S::In>, B, K, G>,
                Tensor<D2<Batch, S::Out>, B, K, G>,
            ),
            Output = Tensor<D2<Batch, S::Out>, B, K, G>,
            Error = Error,
        >,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (
        Tensor<D3<Batch, Seq, S::Out>, B, K, G>,
        Tensor<D2<Batch, S::Out>, B, K, G>,
    );
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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
            let x_step_static: Tensor<D2<Batch, S::In>, B, K, G> = x_step.into_shape()?;
            h = self.cell.forward((x_step_static, h))?;
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B, K, G>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::manipulation::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<D3<Batch, Seq, S::Out>, B, K, G> = stacked_dyn.into_shape()?;

        Ok((stacked, h))
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitState<B> for RNNCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::VisitState<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::VisitState<B>,
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.wi.visit_state(&path.try_child("wi")?, visitor)?;
        self.wh.visit_state(&path.try_child("wh")?, visitor)
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitStateMut<B> for RNNCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::VisitStateMut<B>,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::VisitStateMut<B>,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.wi.visit_state_mut(&path.try_child("wi")?, visitor)?;
        self.wh.visit_state_mut(&path.try_child("wh")?, visitor)
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitState<B> for RNN<S, B, BiasIh, BiasHh, K, Train>
where
    RNNCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::VisitState<B>,
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.cell.visit_state(&path.try_child("cell")?, visitor)
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::VisitStateMut<B> for RNN<S, B, BiasIh, BiasHh, K, Train>
where
    RNNCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::VisitStateMut<B>,
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> crate::err::Result<()> {
        self.cell.visit_state_mut(&path.try_child("cell")?, visitor)
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for RNNCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::module::NamedLayers,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::module::NamedLayers,
{
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        let mut children = Vec::new();
        let p_wi = if prefix.is_empty() {
            alloc::string::String::from("wi")
        } else {
            format!("{}.wi", prefix)
        };
        children.extend(self.wi.layer_structure(&p_wi));
        let p_wh = if prefix.is_empty() {
            alloc::string::String::from("wh")
        } else {
            format!("{}.wh", prefix)
        };
        children.extend(self.wh.layer_structure(&p_wh));

        let node_name = if prefix.is_empty() {
            alloc::string::String::from("RNNCell")
        } else {
            prefix.to_string()
        };
        vec![crate::nn::module::LayerNode {
            name: node_name,
            type_name: alloc::string::String::from("RNNCell"),
            shape_info: "".to_string(),
            children,
        }]
    }
}

impl<
    S: RnnShape,
    B: crate::tensor::backend::VariableBackend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for RNN<S, B, BiasIh, BiasHh, K, Train>
where
    RNNCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::module::NamedLayers,
{
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        let mut children = Vec::new();
        let p_cell = if prefix.is_empty() {
            alloc::string::String::from("cell")
        } else {
            format!("{}.cell", prefix)
        };
        children.extend(self.cell.layer_structure(&p_cell));

        let node_name = if prefix.is_empty() {
            alloc::string::String::from("RNN")
        } else {
            prefix.to_string()
        };
        vec![crate::nn::module::LayerNode {
            name: node_name,
            type_name: alloc::string::String::from("RNN"),
            shape_info: "".to_string(),
            children,
        }]
    }
}
