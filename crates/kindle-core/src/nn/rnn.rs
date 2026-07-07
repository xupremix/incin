use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

/// A shape marker trait specifying the input and output features of an [`RNNCell`].
/// 
/// The typical usage is to supply a 2-tuple `(In, Out)` where:
/// * `In` — Number of input features.
/// * `Out` — Number of output/hidden features.
pub trait RnnShape: Shape + DynShape {
    type In: Dim;
    type Out: Dim;
    type Target;
    fn build_args(target: Self::Target) -> (usize, usize);
}

impl<In: Dim, Out: Dim> RnnShape for (In, Out) {
    type In = In;
    type Out = Out;
    type Target = ();
    fn build_args(_: ()) -> (usize, usize) {
        (
            In::from_arg(Default::default()).size(),
            Out::from_arg(Default::default()).size(),
        )
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
pub struct RNNCell<S: RnnShape, B: Backend> {
    pub wi: Linear<(S::In, S::Out), B>,
    pub wh: Linear<(S::Out, S::Out), B>,
}

impl<S: RnnShape, B: Backend> RNNCell<S, B> {
    pub fn new(wi: Linear<(S::In, S::Out), B>, wh: Linear<(S::Out, S::Out), B>) -> Self {
        Self { wi, wh }
    }
}

impl<S: RnnShape, B: Backend> Parameters<B> for RNNCell<S, B> {
    fn parameters(&self) -> Vec<B::RawVar> {
        let mut params = self.wi.parameters();
        params.extend(self.wh.parameters());
        params
    }
}

impl<
        S: RnnShape,
        Batch: Dim,
        B: Backend,
    > Module<(Tensor<(Batch, S::In), B>, Tensor<(Batch, S::Out), B>)> for RNNCell<S, B>
{
    type Output = Tensor<(Batch, S::Out), B>;
    type Error = Error;

    #[inline]
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
/// ```rust,ignore
/// use kindle::prelude::*;
/// 
/// let cell = RNNCell::new(
///     Linear::<s![10, 20], Backend>::new()?,
///     Linear::<s![20, 20], Backend>::new()?,
/// );
/// let rnn = RNN::<s![10, 20], Backend>::new(cell);
/// let input = Tensor::<s![2, 5, 10], Backend>::zeros(()).unwrap();
/// let h0   = Tensor::<s![2, 20], Backend>::zeros(()).unwrap();
/// let (output, h_n) = rnn.forward((input, h0)).unwrap();
/// // output shape: [2, 5, 20], h_n shape: [2, 20]
/// ```
#[derive(Debug, Clone)]
pub struct RNN<S: RnnShape, B: Backend> {
    pub cell: RNNCell<S, B>,
}

impl<S: RnnShape, B: Backend> RNN<S, B> {
    pub fn new(cell: RNNCell<S, B>) -> Self {
        Self { cell }
    }
}

impl<S: RnnShape, B: Backend> Parameters<B> for RNN<S, B> {
    fn parameters(&self) -> Vec<B::RawVar> {
        self.cell.parameters()
    }
}

impl<
        S: RnnShape,
        Batch: Dim<Arg = ()>,
        Seq: Dim<Arg = ()>,
        B: Backend,
    > Module<(Tensor<(Batch, Seq, S::In), B>, Tensor<(Batch, S::Out), B>)> for RNN<S, B>
where
    S::In: Dim<Arg = ()>,
    S::Out: Dim<Arg = ()>,
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    type Output = (Tensor<(Batch, Seq, S::Out), B>, Tensor<(Batch, S::Out), B>);
    type Error = Error;

    #[inline]
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
        let stacked_dyn = crate::tensor::ops::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<(Batch, Seq, S::Out), B> = stacked_dyn.into_shape()?;
        
        Ok((stacked, h))
    }
}

impl<S: RnnShape, B: Backend> crate::nn::module::StateDict<B> for RNNCell<S, B> {
    fn load_state_dict(&mut self, prefix: &str, tensors: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> {
        self.wi.load_state_dict(&format!("{}wi.", prefix), tensors)?;
        self.wh.load_state_dict(&format!("{}wh.", prefix), tensors)?;
        Ok(())
    }
    fn state_dict(&self, prefix: &str, tensors: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {
        self.wi.state_dict(&format!("{}wi.", prefix), tensors);
        self.wh.state_dict(&format!("{}wh.", prefix), tensors);
    }
}
impl<S: RnnShape, B: Backend> crate::nn::module::StateDict<B> for RNN<S, B> {
    fn load_state_dict(&mut self, prefix: &str, tensors: &std::collections::HashMap<String, Tensor<Dyn, B>>) -> crate::prelude::Result<()> {
        self.cell.load_state_dict(&format!("{}cell.", prefix), tensors)
    }
    fn state_dict(&self, prefix: &str, tensors: &mut std::collections::HashMap<String, Tensor<Dyn, B>>) {
        self.cell.state_dict(&format!("{}cell.", prefix), tensors)
    }
}
