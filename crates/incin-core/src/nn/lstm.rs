use crate::exec::catalog::{Descriptor, op};
use crate::nn::init::Init;
use crate::nn::optional::{False, True};
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use crate::shapes::shape::{DimCons, Nil};
use crate::tensor::backend::Execute;
use alloc::vec::Vec;

type D2<A, B> = DimCons<A, DimCons<B, Nil>>;
type D3<A, B, C> = DimCons<A, DimCons<B, DimCons<C, Nil>>>;

/// A shape marker trait specifying the input and output features of an [`LSTMCell`] / [`LSTM`].
///
/// Supply `(In, Out)` for fully static dimensions, or [`Dyn`] for fully runtime sizes.
pub trait LstmShape: Shape + DynShape {
    /// `In`.
    type In: Dim;
    /// `Out`.
    type Out: Dim;
    type IhShape: LinearShape<InF = Self::In, OutF = Self::Out>;
    type HhShape: LinearShape<InF = Self::Out, OutF = Self::Out>;
}

impl<In: Dim, Out: Dim> LstmShape
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

impl LstmShape for Dyn {
    /// `In`.
    type In = usize;
    /// `Out`.
    type Out = usize;
    type IhShape = Dyn;
    type HhShape = Dyn;
}

// ---------------------------------------------------------------------------
// LSTMCellBuilder: typestate builder
// ---------------------------------------------------------------------------

/// A builder for constructing an [`LSTMCell`] before target-based initialization.
///
/// Stores the layer geometry ([`ShapeValue`]), weight and bias initializer policies
/// (grouped semantically as input and hidden), and compile-time typestate parameters
/// for bias presence and trainability.
pub struct LSTMCellBuilder<
    S: LstmShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// Shape specification (encodes `[in_features, out_features]`).
    pub shape: ShapeValue<S>,
    /// Initializer for all input-to-hidden weight matrices (`W_ih_i`, `W_ih_f`, etc.).
    pub input_weight_init: Init,
    /// Initializer for all hidden-to-hidden weight matrices (`W_hh_i`, `W_hh_f`, etc.).
    pub hidden_weight_init: Init,
    /// Initializer for all input-to-hidden bias vectors.
    pub input_bias_init: Init,
    /// Initializer for all hidden-to-hidden bias vectors.
    pub hidden_bias_init: Init,
    pub _phantom: core::marker::PhantomData<(BiasIh, BiasHh, Train)>,
}

impl<
    S: LstmShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> LSTMCellBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        &self.shape
    }

    /// Removes input-to-hidden biases from the built cell.
    pub fn no_input_bias(self) -> LSTMCellBuilder<S, False, BiasHh, Train> {
        LSTMCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes hidden-to-hidden biases from the built cell.
    pub fn no_hidden_bias(self) -> LSTMCellBuilder<S, BiasIh, False, Train> {
        LSTMCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Removes all biases from the built cell.
    pub fn no_bias(self) -> LSTMCellBuilder<S, False, False, Train> {
        LSTMCellBuilder {
            shape: self.shape,
            input_weight_init: self.input_weight_init,
            hidden_weight_init: self.hidden_weight_init,
            input_bias_init: self.input_bias_init,
            hidden_bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Marks the created cell parameters as frozen (non-trainable).
    pub fn frozen(self) -> LSTMCellBuilder<S, BiasIh, BiasHh, Frozen> {
        LSTMCellBuilder {
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

/// Free constructor for a backend-independent [`LSTMCellBuilder`].
pub fn lstm_cell<S: LstmShape>(shape: ShapeValue<S>) -> LSTMCellBuilder<S> {
    let init = crate::nn::init::kaiming_uniform();
    LSTMCellBuilder {
        shape,
        input_weight_init: init,
        hidden_weight_init: init,
        input_bias_init: init,
        hidden_bias_init: init,
        _phantom: core::marker::PhantomData,
    }
}

// ---------------------------------------------------------------------------
// LSTMBuilder: typestate builder for the multi-step LSTM
// ---------------------------------------------------------------------------

/// A builder for constructing an [`LSTM`] before target-based initialization.
///
/// Wraps an [`LSTMCellBuilder`] and exposes the same bias/trainability controls.
pub struct LSTMBuilder<
    S: LstmShape,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    Train: TrainState = Trainable,
> {
    /// The inner cell builder.
    pub cell: LSTMCellBuilder<S, BiasIh, BiasHh, Train>,
}

impl<
    S: LstmShape,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    Train: TrainState,
> LSTMBuilder<S, BiasIh, BiasHh, Train>
{
    /// Returns a reference to the shape specification of this builder.
    pub fn shape(&self) -> &ShapeValue<S> {
        self.cell.shape()
    }

    /// Removes input-to-hidden biases from the built LSTM.
    pub fn no_input_bias(self) -> LSTMBuilder<S, False, BiasHh, Train> {
        LSTMBuilder {
            cell: self.cell.no_input_bias(),
        }
    }

    /// Removes hidden-to-hidden biases from the built LSTM.
    pub fn no_hidden_bias(self) -> LSTMBuilder<S, BiasIh, False, Train> {
        LSTMBuilder {
            cell: self.cell.no_hidden_bias(),
        }
    }

    /// Removes all biases from the built LSTM.
    pub fn no_bias(self) -> LSTMBuilder<S, False, False, Train> {
        LSTMBuilder {
            cell: self.cell.no_bias(),
        }
    }

    /// Marks the created LSTM parameters as frozen (non-trainable).
    pub fn frozen(self) -> LSTMBuilder<S, BiasIh, BiasHh, Frozen> {
        LSTMBuilder {
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

/// Free constructor for a backend-independent [`LSTMBuilder`].
pub fn lstm<S: LstmShape>(shape: ShapeValue<S>) -> LSTMBuilder<S> {
    LSTMBuilder {
        cell: lstm_cell(shape),
    }
}

// ---------------------------------------------------------------------------
// LSTMCell: concrete BiasIh / BiasHh combinations
// ---------------------------------------------------------------------------

/// A single LSTM step cell implementing the standard 4-gate recurrence.
///
/// * `S`: [`LstmShape`]: `(In, Out)` static or [`Dyn`] for runtime sizes.
/// * `BiasIh`: whether input-to-hidden biases exist: [`True`], [`False`].
/// * `BiasHh`: whether hidden-to-hidden biases exist: [`True`], [`False`].
/// * `K`: parameter dtype (default: `f32`).
/// * `Train`: trainability typestate (default: [`Trainable`]).
///
/// ## Examples
///
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// // Fully static, both biases present (default)
/// let cell = LSTMCell::<s![10, 20], DefaultBackend>::build(())?;
///
/// // Fully static, input bias present, hidden bias absent
/// let cell = LSTMCell::<s![10, 20], DefaultBackend, True, False>::build(())?;
///
/// // Fully dynamic shape, both biases always present
/// let cell = LSTMCell::<Dyn, DefaultBackend>::build((10, 20))?;
///
/// // Fully dynamic shape, no biases
/// let cell = LSTMCell::<Dyn, DefaultBackend, False, False>::build((10, 20))?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct LSTMCell<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
> where
    S::IhShape: LinearShape,
    S::HhShape: LinearShape,
{
    /// `wi_i`.
    pub wi_i: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// `wi_f`.
    pub wi_f: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// `wi_g`.
    pub wi_g: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// `wi_o`.
    pub wi_o: Linear<S::IhShape, B, BiasIh, K, Train>,
    /// `wh_i`.
    pub wh_i: Linear<S::HhShape, B, BiasHh, K, Train>,
    /// `wh_f`.
    pub wh_f: Linear<S::HhShape, B, BiasHh, K, Train>,
    /// `wh_g`.
    pub wh_g: Linear<S::HhShape, B, BiasHh, K, Train>,
    /// `wh_o`.
    pub wh_o: Linear<S::HhShape, B, BiasHh, K, Train>,
}

impl<S, B, BiasIh, BiasHh, K: DType, Train: TrainState> LSTMCell<S, B, BiasIh, BiasHh, K, Train>
where
    S: LstmShape,
    B: Backend
        + SupportsDType<K>
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>
        + crate::tensor::backend::TensorOps<B>,
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
    /// Prefer the target-aware `LSTMCellBuilder::init()` path for new code.
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
        use crate::tensor::arg_into::LayerArgInto;
        let (input, output, dtype, device, bias_ih, bias_hh) = args.into_layer_arg();
        Ok(Self {
            wi_i: Linear::<_, _, _, K, Train>::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_f: Linear::<_, _, _, K, Train>::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_g: Linear::<_, _, _, K, Train>::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_o: Linear::<_, _, _, K, Train>::build_full(
                input,
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih,
            )?,
            wh_i: Linear::<_, _, _, K, Train>::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_f: Linear::<_, _, _, K, Train>::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_g: Linear::<_, _, _, K, Train>::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_o: Linear::<_, _, _, K, Train>::build_full(
                output.clone(),
                output,
                dtype,
                device,
                bias_hh,
            )?,
        })
    }

    /// Converts this cell's parameters to frozen typestate.
    pub fn freeze(self) -> LSTMCell<S, B, BiasIh, BiasHh, K, Frozen> {
        LSTMCell {
            wi_i: self.wi_i.freeze(),
            wi_f: self.wi_f.freeze(),
            wi_g: self.wi_g.freeze(),
            wi_o: self.wi_o.freeze(),
            wh_i: self.wh_i.freeze(),
            wh_f: self.wh_f.freeze(),
            wh_g: self.wh_g.freeze(),
            wh_o: self.wh_o.freeze(),
        }
    }

    /// Converts this cell's parameters to trainable typestate.
    pub fn unfreeze(self) -> LSTMCell<S, B, BiasIh, BiasHh, K, Trainable> {
        LSTMCell {
            wi_i: self.wi_i.unfreeze(),
            wi_f: self.wi_f.unfreeze(),
            wi_g: self.wi_g.unfreeze(),
            wi_o: self.wi_o.unfreeze(),
            wh_i: self.wh_i.unfreeze(),
            wh_f: self.wh_f.unfreeze(),
            wh_g: self.wh_g.unfreeze(),
            wh_o: self.wh_o.unfreeze(),
        }
    }
}

// ---------------------------------------------------------------------------
// Parameters impl (generic over all BiasIh/BiasHh/K/Train combinations)
// ---------------------------------------------------------------------------

impl<
    In: Dim,
    Out: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> Parameters<B> for LSTMCell<D2<In, Out>, B, BiasIh, BiasHh, K, Train>
where
    Linear<D2<In, Out>, B, BiasIh, K, Train>: Parameters<B>,
    Linear<D2<Out, Out>, B, BiasHh, K, Train>: Parameters<B>,
{
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.wi_i.named_parameters(&format!("{}wi_i.", prefix), map);
        self.wi_f.named_parameters(&format!("{}wi_f.", prefix), map);
        self.wi_g.named_parameters(&format!("{}wi_g.", prefix), map);
        self.wi_o.named_parameters(&format!("{}wi_o.", prefix), map);
        self.wh_i.named_parameters(&format!("{}wh_i.", prefix), map);
        self.wh_f.named_parameters(&format!("{}wh_f.", prefix), map);
        self.wh_g.named_parameters(&format!("{}wh_g.", prefix), map);
        self.wh_o.named_parameters(&format!("{}wh_o.", prefix), map);
    }
}

// ---------------------------------------------------------------------------
// Module (forward) impl: static shapes
// ---------------------------------------------------------------------------

impl<
    In: Dim,
    Out: Dim,
    Batch: Dim,
    B: Backend + Execute<op::Add> + Execute<op::Mul> + Execute<op::Sigmoid> + Execute<op::Tanh>,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
>
    Module<(
        Tensor<D2<Batch, In>, B, K>,
        (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
    )> for LSTMCell<D2<In, Out>, B, BiasIh, BiasHh, K, Train>
where
    Linear<D2<In, Out>, B, BiasIh, K, Train>:
        Module<Tensor<D2<Batch, In>, B, K>, Output = Tensor<D2<Batch, Out>, B, K>, Error = Error>,
    Linear<D2<Out, Out>, B, BiasHh, K, Train>:
        Module<Tensor<D2<Batch, Out>, B, K>, Output = Tensor<D2<Batch, Out>, B, K>, Error = Error>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sigmoid>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Tanh>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>);
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, (h_prev, c_prev)): (
            Tensor<D2<Batch, In>, B, K>,
            (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let i = self
            .wi_i
            .forward(x.clone())?
            .add(&self.wh_i.forward(h_prev.clone())?)?
            .sigmoid()?;
        let f = self
            .wi_f
            .forward(x.clone())?
            .add(&self.wh_f.forward(h_prev.clone())?)?
            .sigmoid()?;
        let g = self
            .wi_g
            .forward(x.clone())?
            .add(&self.wh_g.forward(h_prev.clone())?)?
            .tanh()?;
        let o = self
            .wi_o
            .forward(x)?
            .add(&self.wh_o.forward(h_prev)?)?
            .sigmoid()?;
        let c = f.mul(&c_prev)?.add(&i.mul(&g)?)?;
        let h = o.mul(&c.clone().tanh()?)?;
        Ok((h, c))
    }
}

// ---------------------------------------------------------------------------
// LSTM (multi-step wrapper)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
/// `LSTM`.
#[allow(clippy::upper_case_acronyms)]
pub struct LSTM<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// `cell`.
    pub cell: LSTMCell<S, B, BiasIh, BiasHh, K, Train>,
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> LSTM<S, B, BiasIh, BiasHh, K, Train>
{
    /// Creates a new instance from a pre-built cell.
    pub fn new(cell: LSTMCell<S, B, BiasIh, BiasHh, K, Train>) -> Self {
        Self { cell }
    }

    /// Converts this LSTM's parameters to frozen typestate.
    pub fn freeze(self) -> LSTM<S, B, BiasIh, BiasHh, K, Frozen>
    where
        LSTMCell<S, B, BiasIh, BiasHh, K, Train>: Sized,
        S: LstmShape,
        B: Backend
            + SupportsDType<K>
            + crate::tensor::backend::SupportsDType<K>
            + crate::nn::param::ParameterInit<K>
            + crate::tensor::backend::TensorOps<B>,
        <K as DType>::Arg: Clone,
        <S::In as Dim>::Arg: Clone,
        <S::Out as Dim>::Arg: Clone,
        <B::Device as Device>::Arg: Clone,
        <BiasIh as crate::nn::optional::OptionalField>::Arg: Clone,
        <BiasHh as crate::nn::optional::OptionalField>::Arg: Clone,
    {
        LSTM {
            cell: self.cell.freeze(),
        }
    }

    /// Converts this LSTM's parameters to trainable typestate.
    pub fn unfreeze(self) -> LSTM<S, B, BiasIh, BiasHh, K, Trainable>
    where
        LSTMCell<S, B, BiasIh, BiasHh, K, Train>: Sized,
        S: LstmShape,
        B: Backend
            + SupportsDType<K>
            + crate::tensor::backend::SupportsDType<K>
            + crate::nn::param::ParameterInit<K>
            + crate::tensor::backend::TensorOps<B>,
        <K as DType>::Arg: Clone,
        <S::In as Dim>::Arg: Clone,
        <S::Out as Dim>::Arg: Clone,
        <B::Device as Device>::Arg: Clone,
        <BiasIh as crate::nn::optional::OptionalField>::Arg: Clone,
        <BiasHh as crate::nn::optional::OptionalField>::Arg: Clone,
    {
        LSTM {
            cell: self.cell.unfreeze(),
        }
    }
}

impl<
    In: Dim,
    Out: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> Parameters<B> for LSTM<D2<In, Out>, B, BiasIh, BiasHh, K, Train>
where
    LSTMCell<D2<In, Out>, B, BiasIh, BiasHh, K, Train>: Parameters<B>,
{
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.cell.named_parameters(&format!("{}cell.", prefix), map);
    }
}

impl<
    In: Dim<Arg = ()>,
    Out: Dim<Arg = ()>,
    Batch: Dim<Arg = ()>,
    Seq: Dim<Arg = ()>,
    B: Backend
        + Execute<op::StackExact>
        + Execute<op::Narrow>
        + Execute<op::SqueezeExact>
        + crate::exec::Capabilities,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
>
    Module<(
        Tensor<D3<Batch, Seq, In>, B, K>,
        (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
    )> for LSTM<D2<In, Out>, B, BiasIh, BiasHh, K, Train>
where
    <B as Execute<op::StackExact>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::SqueezeExact>>::Output: Into<B::Storage<K>>,
    LSTMCell<D2<In, Out>, B, BiasIh, BiasHh, K, Train>: Module<
            (
                Tensor<D2<Batch, In>, B, K>,
                (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
            ),
            Output = (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
            Error = Error,
        >,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (
        Tensor<D3<Batch, Seq, Out>, B, K>,
        (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
    );
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, (mut h, mut c)): (
            Tensor<D3<Batch, Seq, In>, B, K>,
            (Tensor<D2<Batch, Out>, B, K>, Tensor<D2<Batch, Out>, B, K>),
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let seq_len = Seq::static_size().map_err(Error::Shape)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let x_step = x.clone().try_narrow(1, i, 1)?.try_squeeze(1)?;
            let x_step_static: Tensor<D2<Batch, In>, B, K> = x_step.into_shape()?;
            let (h_next, c_next) = self.cell.forward((x_step_static, (h, c)))?;
            h = h_next;
            c = c_next;
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B, K>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::manipulation::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<D3<Batch, Seq, Out>, B, K> = stacked_dyn.into_shape()?;
        Ok((stacked, (h, c)))
    }
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for LSTMCell<S, B, BiasIh, BiasHh, K, Train>
where
    Linear<S::IhShape, B, BiasIh, K, Train>: crate::nn::module::NamedLayers,
    Linear<S::HhShape, B, BiasHh, K, Train>: crate::nn::module::NamedLayers,
{
    /// Returns the layer hierarchy rooted at this module for visualization.
    fn layer_structure(&self, prefix: &str) -> Vec<crate::nn::module::LayerNode> {
        let mut children = Vec::new();

        let p_wi_i = if prefix.is_empty() {
            alloc::string::String::from("wi_i")
        } else {
            format!("{}.wi_i", prefix)
        };
        children.extend(self.wi_i.layer_structure(&p_wi_i));
        let p_wi_f = if prefix.is_empty() {
            alloc::string::String::from("wi_f")
        } else {
            format!("{}.wi_f", prefix)
        };
        children.extend(self.wi_f.layer_structure(&p_wi_f));
        let p_wi_g = if prefix.is_empty() {
            alloc::string::String::from("wi_g")
        } else {
            format!("{}.wi_g", prefix)
        };
        children.extend(self.wi_g.layer_structure(&p_wi_g));
        let p_wi_o = if prefix.is_empty() {
            alloc::string::String::from("wi_o")
        } else {
            format!("{}.wi_o", prefix)
        };
        children.extend(self.wi_o.layer_structure(&p_wi_o));
        let p_wh_i = if prefix.is_empty() {
            alloc::string::String::from("wh_i")
        } else {
            format!("{}.wh_i", prefix)
        };
        children.extend(self.wh_i.layer_structure(&p_wh_i));
        let p_wh_f = if prefix.is_empty() {
            alloc::string::String::from("wh_f")
        } else {
            format!("{}.wh_f", prefix)
        };
        children.extend(self.wh_f.layer_structure(&p_wh_f));
        let p_wh_g = if prefix.is_empty() {
            alloc::string::String::from("wh_g")
        } else {
            format!("{}.wh_g", prefix)
        };
        children.extend(self.wh_g.layer_structure(&p_wh_g));
        let p_wh_o = if prefix.is_empty() {
            alloc::string::String::from("wh_o")
        } else {
            format!("{}.wh_o", prefix)
        };
        children.extend(self.wh_o.layer_structure(&p_wh_o));

        let node_name = if prefix.is_empty() {
            alloc::string::String::from("LSTMCell")
        } else {
            prefix.to_string()
        };
        vec![crate::nn::module::LayerNode {
            name: node_name,
            type_name: alloc::string::String::from("LSTMCell"),
            shape_info: "".to_string(),
            children,
        }]
    }
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> crate::nn::module::NamedLayers for LSTM<S, B, BiasIh, BiasHh, K, Train>
where
    LSTMCell<S, B, BiasIh, BiasHh, K, Train>: crate::nn::module::NamedLayers,
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
            alloc::string::String::from("LSTM")
        } else {
            prefix.to_string()
        };
        vec![crate::nn::module::LayerNode {
            name: node_name,
            type_name: alloc::string::String::from("LSTM"),
            shape_info: "".to_string(),
            children,
        }]
    }
}
