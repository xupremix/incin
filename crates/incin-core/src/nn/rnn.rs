use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

/// A shape marker trait specifying the input and output features of an [`RNNCell`].
///
/// The typical usage is to supply a 2-tuple `(In, Out)` where:
/// * `In` — Number of input features.
/// * `Out` — Number of output/hidden features.
pub trait RnnShape: Shape + DynShape {
    /// `In`.
    type In: Dim;
    /// `Out`.
    type Out: Dim;
    /// The runtime arguments needed to instantiate this layer.
    type Target;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> (usize, usize);
}

impl<In: Dim, Out: Dim> RnnShape for (In, Out) {
    /// `In`.
    type In = In;
    /// `Out`.
    type Out = Out;
    /// The runtime arguments needed to instantiate this layer.
    type Target = ();
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(_: ()) -> (usize, usize) {
        (
            In::from_arg(Default::default()).size(),
            Out::from_arg(Default::default()).size(),
        )
    }
}

impl RnnShape for Dyn {
    /// `In`.
    type In = usize;
    /// `Out`.
    type Out = usize;
    /// The runtime arguments needed to instantiate this layer.
    type Target = (usize, usize);
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: (usize, usize)) -> (usize, usize) {
        target
    }
}

/// A single RNN step cell computing `h_t = tanh(W_ih * x_t + W_hh * h_{t-1})`.
///
/// This implements the basic Elman RNN cell (no gating mechanism). Each time step,
/// it takes an input `x_t: Tensor<(Batch, In)>` and the previous hidden state
/// `h_{t-1}: Tensor<(Batch, Out)>` and outputs the new hidden state `h_t`.
///
/// For a multi-step sequence model, wrap this in [`RNN`].
#[derive(Debug, Clone)]
pub struct RNNCell<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = crate::nn::optional::True,
    BiasHh: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// `wi`.
    pub wi: Linear<(S::In, S::Out), B, BiasIh>,
    /// `wh`.
    pub wh: Linear<(S::Out, S::Out), B, BiasHh>,
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> RNNCell<S, B, BiasIh, BiasHh>
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(
        wi: Linear<(S::In, S::Out), B, BiasIh>,
        wh: Linear<(S::Out, S::Out), B, BiasHh>,
    ) -> Self {
        Self { wi, wh }
    }
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Parameters<B> for RNNCell<S, B, BiasIh, BiasHh>
where
    Linear<(S::In, S::Out), B, BiasIh>: Parameters<B>,
    Linear<(S::Out, S::Out), B, BiasHh>: Parameters<B>,
{
    /// Collects named trainable parameters into `map` under the given `prefix`.
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
        self.wi.named_parameters(&format!("{}wi.", prefix), map);
        self.wh.named_parameters(&format!("{}wh.", prefix), map);
    }
}

impl<
    S: RnnShape,
    Batch: Dim,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Module<(Tensor<(Batch, S::In), B>, Tensor<(Batch, S::Out), B>)> for RNNCell<S, B, BiasIh, BiasHh>
where
    Linear<(S::In, S::Out), B, BiasIh>:
        Module<Tensor<(Batch, S::In), B>, Output = Tensor<(Batch, S::Out), B>, Error = Error>,
    Linear<(S::Out, S::Out), B, BiasHh>:
        Module<Tensor<(Batch, S::Out), B>, Output = Tensor<(Batch, S::Out), B>, Error = Error>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<(Batch, S::Out), B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, h_prev): (Tensor<(Batch, S::In), B>, Tensor<(Batch, S::Out), B>),
    ) -> core::result::Result<Self::Output, Error> {
        let i = self.wi.forward(x)?;
        let h = self.wh.forward(h_prev)?;
        // h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
        let sum = i.add(&h)?;
        sum.tanh()
    }
}

/// An Elman Recurrent Neural Network (RNN) that processes an input sequence step-by-step.
///
/// Wraps an [`RNNCell`] to iterate over the sequence dimension automatically.
/// The `forward` method accepts a pair of `(sequence, initial_hidden_state)` and
/// returns `(all_outputs, final_hidden_state)`.
///
/// ## Type Parameters
/// * `S` — An `RnnShape` describing the input and output feature dimensions.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::prelude::dummy::DummyBackend<f32, incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// let cell = RNNCell::new(
///     Linear::<s![10, 20], DefaultBackend>::build(())?,
///     Linear::<s![20, 20], DefaultBackend>::build(())?,
/// );
/// let rnn = RNN::<s![10, 20], DefaultBackend>::new(cell);
/// let input = Tensor::<s![2, 5, 10], DefaultBackend>::zeros(()).unwrap();
/// let h0   = Tensor::<s![2, 20], DefaultBackend>::zeros(()).unwrap();
/// let (output, h_n) = rnn.forward((input, h0)).unwrap();
/// // output shape: [2, 5, 20], h_n shape: [2, 20]
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct RNN<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = crate::nn::optional::True,
    BiasHh: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// `cell`.
    pub cell: RNNCell<S, B, BiasIh, BiasHh>,
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> RNN<S, B, BiasIh, BiasHh>
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(cell: RNNCell<S, B, BiasIh, BiasHh>) -> Self {
        Self { cell }
    }
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Parameters<B> for RNN<S, B, BiasIh, BiasHh>
where
    RNNCell<S, B, BiasIh, BiasHh>: Parameters<B>,
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
    S: RnnShape,
    Batch: Dim<Arg = ()>,
    Seq: Dim<Arg = ()>,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> Module<(Tensor<(Batch, Seq, S::In), B>, Tensor<(Batch, S::Out), B>)> for RNN<S, B, BiasIh, BiasHh>
where
    S::In: Dim<Arg = ()>,
    S::Out: Dim<Arg = ()>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    RNNCell<S, B, BiasIh, BiasHh>: Module<
            (Tensor<(Batch, S::In), B>, Tensor<(Batch, S::Out), B>),
            Output = Tensor<(Batch, S::Out), B>,
            Error = Error,
        >,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = (Tensor<(Batch, Seq, S::Out), B>, Tensor<(Batch, S::Out), B>);
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        (x, mut h): (Tensor<(Batch, Seq, S::In), B>, Tensor<(Batch, S::Out), B>),
    ) -> core::result::Result<Self::Output, Error> {
        let seq_len = Seq::from_arg(()).size();
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let x_step = x.clone().try_narrow(1, i, 1)?.try_squeeze(1)?;
            let x_step_static: Tensor<(Batch, S::In), B> = x_step.into_shape()?;
            h = self.cell.forward((x_step_static, h))?;
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::manipulation::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<(Batch, Seq, S::Out), B> = stacked_dyn.into_shape()?;

        Ok((stacked, h))
    }
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> crate::nn::module::StateDict<B> for RNNCell<S, B, BiasIh, BiasHh>
{
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        self.wi
            .load_state_dict(&format!("{}wi.", prefix), tensors)?;
        self.wh
            .load_state_dict(&format!("{}wh.", prefix), tensors)?;
        Ok(())
    }
    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(
        &self,
        prefix: &str,
        tensors: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        self.wi.state_dict(&format!("{}wi.", prefix), tensors);
        self.wh.state_dict(&format!("{}wh.", prefix), tensors);
    }
}
impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> crate::nn::module::StateDict<B> for RNN<S, B, BiasIh, BiasHh>
{
    /// Loads parameters from a flat name→tensor map, in-place.
    fn load_state_dict(
        &mut self,
        prefix: &str,
        tensors: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> crate::prelude::Result<()> {
        self.cell
            .load_state_dict(&format!("{}cell.", prefix), tensors)
    }
    /// Returns a flat map from parameter name to its raw tensor value.
    fn state_dict(
        &self,
        prefix: &str,
        tensors: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        self.cell.state_dict(&format!("{}cell.", prefix), tensors)
    }
}

impl<
    S: RnnShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> crate::nn::module::NamedLayers for RNNCell<S, B, BiasIh, BiasHh>
where
    Linear<(S::In, S::Out), B, BiasIh>: crate::nn::module::NamedLayers,
    Linear<(S::Out, S::Out), B, BiasHh>: crate::nn::module::NamedLayers,
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
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> crate::nn::module::NamedLayers for RNN<S, B, BiasIh, BiasHh>
where
    RNNCell<S, B, BiasIh, BiasHh>: crate::nn::module::NamedLayers,
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
