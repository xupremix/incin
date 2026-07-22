use crate::nn::optional::{False, True};
use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

/// A shape marker trait specifying the input and output features of an [`LSTMCell`] / [`LSTM`].
///
/// Supply `(In, Out)` for fully static dimensions, or [`Dyn`] for fully runtime sizes.
pub trait LstmShape: Shape + DynShape {
    /// `In`.
    type In: Dim;
    /// `Out`.
    type Out: Dim;
}

impl<In: Dim, Out: Dim> LstmShape for (In, Out) {
    /// `In`.
    type In = In;
    /// `Out`.
    type Out = Out;
}

impl LstmShape for Dyn {
    /// `In`.
    type In = usize;
    /// `Out`.
    type Out = usize;
}

// ---------------------------------------------------------------------------
// LSTMCell — concrete BiasIh / BiasHh combinations
// ---------------------------------------------------------------------------

/// A single LSTM step cell implementing the standard 4-gate recurrence.
///
/// * `S`      — [`LstmShape`]: `(In, Out)` static or [`Dyn`] for runtime sizes.
/// * `BiasIh` — whether input-to-hidden biases exist: [`True`], [`False`].
/// * `BiasHh` — whether hidden-to-hidden biases exist: [`True`], [`False`].
///
/// ## Examples
///
/// ```rust,ignore
/// use kindle::prelude::*;
/// use kindle::nn::optional::{True, False};
///
/// // Fully static, both biases present (default)
/// let cell = LSTMCell::<s![10, 20], B>::build(())?;
///
/// // Fully static, input bias present, hidden bias absent
/// let cell = LSTMCell::<s![10, 20], B, True, False>::build(())?;
///
/// // Fully dynamic shape, both biases always present
/// let cell = LSTMCell::<Dyn, B>::build((10, 20))?;
///
/// // Fully dynamic shape, no biases
/// let cell = LSTMCell::<Dyn, B, False, False>::build((10, 20))?;
/// ```
#[derive(Debug, Clone)]
pub struct LSTMCell<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
> {
    /// `wi_i`.
    pub wi_i: Linear<(S::In, S::Out), B, BiasIh>,
    /// `wi_f`.
    pub wi_f: Linear<(S::In, S::Out), B, BiasIh>,
    /// `wi_g`.
    pub wi_g: Linear<(S::In, S::Out), B, BiasIh>,
    /// `wi_o`.
    pub wi_o: Linear<(S::In, S::Out), B, BiasIh>,
    /// `wh_i`.
    pub wh_i: Linear<(S::Out, S::Out), B, BiasHh>,
    /// `wh_f`.
    pub wh_f: Linear<(S::Out, S::Out), B, BiasHh>,
    /// `wh_g`.
    pub wh_g: Linear<(S::Out, S::Out), B, BiasHh>,
    /// `wh_o`.
    pub wh_o: Linear<(S::Out, S::Out), B, BiasHh>,
}

impl<S, B, BiasIh, BiasHh> LSTMCell<S, B, BiasIh, BiasHh>
where
    S: LstmShape,
    B: Backend + SupportsDType<B::FloatElem>,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
    <S::In as Dim>::Arg: Clone,
    <S::Out as Dim>::Arg: Clone,
    <B::FloatElem as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
    <BiasIh as crate::nn::optional::OptionalField>::Arg: Clone,
    <BiasHh as crate::nn::optional::OptionalField>::Arg: Clone,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::In as Dim>::Arg,
                <S::Out as Dim>::Arg,
                <B::FloatElem as DType>::Arg,
                <B::Device as Device>::Arg,
                <BiasIh as crate::nn::optional::OptionalField>::Arg,
                <BiasHh as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (input, output, dtype, device, bias_ih, bias_hh) = args.into_layer_arg();
        Ok(Self {
            wi_i: Linear::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_f: Linear::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_g: Linear::build_full(
                input.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih.clone(),
            )?,
            wi_o: Linear::build_full(
                input,
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_ih,
            )?,
            wh_i: Linear::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_f: Linear::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_g: Linear::build_full(
                output.clone(),
                output.clone(),
                dtype.clone(),
                device.clone(),
                bias_hh.clone(),
            )?,
            wh_o: Linear::build_full(output.clone(), output, dtype, device, bias_hh)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Parameters impl (generic over all BiasIh/BiasHh combinations)
// ---------------------------------------------------------------------------

impl<
    In: Dim,
    Out: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Parameters<B> for LSTMCell<(In, Out), B, BiasIh, BiasHh>
where
    Linear<(In, Out), B, BiasIh>: Parameters<B>,
    Linear<(Out, Out), B, BiasHh>: Parameters<B>,
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
// Module (forward) impl — static shapes
// ---------------------------------------------------------------------------

impl<
    In: Dim,
    Out: Dim,
    Batch: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
>
    Module<(
        Tensor<(Batch, In), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    )> for LSTMCell<(In, Out), B, BiasIh, BiasHh>
where
    Linear<(In, Out), B, BiasIh>:
        Module<Tensor<(Batch, In), B>, Output = Tensor<(Batch, Out), B>, Error = Error>,
    Linear<(Out, Out), B, BiasHh>:
        Module<Tensor<(Batch, Out), B>, Output = Tensor<(Batch, Out), B>, Error = Error>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>);
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, (h_prev, c_prev)): (
            Tensor<(Batch, In), B>,
            (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
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
> {
    /// `cell`.
    pub cell: LSTMCell<S, B, BiasIh, BiasHh>,
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> LSTM<S, B, BiasIh, BiasHh>
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(cell: LSTMCell<S, B, BiasIh, BiasHh>) -> Self {
        Self { cell }
    }
}

impl<
    In: Dim,
    Out: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Parameters<B> for LSTM<(In, Out), B, BiasIh, BiasHh>
where
    LSTMCell<(In, Out), B, BiasIh, BiasHh>: Parameters<B>,
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
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
>
    Module<(
        Tensor<(Batch, Seq, In), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    )> for LSTM<(In, Out), B, BiasIh, BiasHh>
where
    LSTMCell<(In, Out), B, BiasIh, BiasHh>: Module<
            (
                Tensor<(Batch, In), B>,
                (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
            ),
            Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
            Error = Error,
        >,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (
        Tensor<(Batch, Seq, Out), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    );
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, (mut h, mut c)): (
            Tensor<(Batch, Seq, In), B>,
            (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let seq_len = Seq::from_arg(()).size();
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let x_step = x.clone().try_narrow(1, i, 1)?.try_squeeze(1)?;
            let x_step_static: Tensor<(Batch, In), B> = x_step.into_shape()?;
            let (h_next, c_next) = self.cell.forward((x_step_static, (h, c)))?;
            h = h_next;
            c = c_next;
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::manipulation::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<(Batch, Seq, Out), B> = stacked_dyn.into_shape()?;
        Ok((stacked, (h, c)))
    }
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> crate::nn::module::NamedLayers for LSTMCell<S, B, BiasIh, BiasHh>
where
    Linear<(S::In, S::Out), B, BiasIh>: crate::nn::module::NamedLayers,
    Linear<(S::Out, S::Out), B, BiasHh>: crate::nn::module::NamedLayers,
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
> crate::nn::module::NamedLayers for LSTM<S, B, BiasIh, BiasHh>
where
    LSTMCell<S, B, BiasIh, BiasHh>: crate::nn::module::NamedLayers,
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
