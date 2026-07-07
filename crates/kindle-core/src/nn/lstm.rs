use crate::nn::{Linear, Module, Parameters};
use crate::nn::optional::{True, False};
use crate::prelude::*;
use alloc::vec::Vec;

/// A shape marker trait specifying the input and output features of an [`LSTMCell`] / [`LSTM`].
///
/// Supply `(In, Out)` for fully static dimensions, or [`Dyn`] for fully runtime sizes.
pub trait LstmShape: Shape + DynShape {
    type In: Dim;
    type Out: Dim;
}

impl<In: Dim, Out: Dim> LstmShape for (In, Out) {
    type In = In;
    type Out = Out;
}

impl LstmShape for Dyn {
    type In = usize;
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
/// let cell = LSTMCell::<s![U10, U20], B>::new()?;
///
/// // Fully static, input bias present, hidden bias absent
/// let cell = LSTMCell::<s![U10, U20], B, True, False>::new()?;
///
/// // Fully dynamic shape, both biases always present
/// let cell = LSTMCell::<Dyn, B>::new_with(10, 20)?;
///
/// // Fully dynamic shape, no biases
/// let cell = LSTMCell::<Dyn, B, False, False>::new_with(10, 20)?;
/// ```
#[derive(Debug, Clone)]
pub struct LSTMCell<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
> {
    pub wi_i: Linear<(S::In, S::Out), B, BiasIh>,
    pub wi_f: Linear<(S::In, S::Out), B, BiasIh>,
    pub wi_g: Linear<(S::In, S::Out), B, BiasIh>,
    pub wi_o: Linear<(S::In, S::Out), B, BiasIh>,
    pub wh_i: Linear<(S::Out, S::Out), B, BiasHh>,
    pub wh_f: Linear<(S::Out, S::Out), B, BiasHh>,
    pub wh_g: Linear<(S::Out, S::Out), B, BiasHh>,
    pub wh_o: Linear<(S::Out, S::Out), B, BiasHh>,
}

// ── LSTMCell<S, B, True, True> (default) ────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, True, True>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new_with(in_arg: In::Arg, out_arg: Out::Arg) -> Result<Self>
    where
        In::Arg: Clone,
        Out::Arg: Clone,
    {
        Ok(Self {
            wi_i: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_f: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_g: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_o: Linear::<(In, Out), B, True>::new_with((in_arg, out_arg.clone()))?,
            wh_i: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_f: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_g: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_o: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg))?,
        })
    }
}

impl<In: Dim<Arg = ()>, Out: Dim<Arg = ()>, B: Backend> LSTMCell<(In, Out), B, True, True>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new() -> Result<Self> { Self::new_with((), ()) }
}

// ── LSTMCell<S, B, True, False> ─────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, True, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new_with(in_arg: In::Arg, out_arg: Out::Arg) -> Result<Self>
    where
        In::Arg: Clone,
        Out::Arg: Clone,
    {
        Ok(Self {
            wi_i: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_f: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_g: Linear::<(In, Out), B, True>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_o: Linear::<(In, Out), B, True>::new_with((in_arg, out_arg.clone()))?,
            wh_i: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_f: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_g: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_o: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg))?,
        })
    }
}

impl<In: Dim<Arg = ()>, Out: Dim<Arg = ()>, B: Backend> LSTMCell<(In, Out), B, True, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new() -> Result<Self> { Self::new_with((), ()) }
}

// ── LSTMCell<S, B, False, True> ─────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, False, True>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new_with(in_arg: In::Arg, out_arg: Out::Arg) -> Result<Self>
    where
        In::Arg: Clone,
        Out::Arg: Clone,
    {
        Ok(Self {
            wi_i: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_f: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_g: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_o: Linear::<(In, Out), B, False>::new_with((in_arg, out_arg.clone()))?,
            wh_i: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_f: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_g: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_o: Linear::<(Out, Out), B, True>::new_with((out_arg.clone(), out_arg))?,
        })
    }
}

impl<In: Dim<Arg = ()>, Out: Dim<Arg = ()>, B: Backend> LSTMCell<(In, Out), B, False, True>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new() -> Result<Self> { Self::new_with((), ()) }
}

// ── LSTMCell<S, B, False, False> ────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, False, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new_with(in_arg: In::Arg, out_arg: Out::Arg) -> Result<Self>
    where
        In::Arg: Clone,
        Out::Arg: Clone,
    {
        Ok(Self {
            wi_i: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_f: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_g: Linear::<(In, Out), B, False>::new_with((in_arg.clone(), out_arg.clone()))?,
            wi_o: Linear::<(In, Out), B, False>::new_with((in_arg, out_arg.clone()))?,
            wh_i: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_f: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_g: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg.clone()))?,
            wh_o: Linear::<(Out, Out), B, False>::new_with((out_arg.clone(), out_arg))?,
        })
    }
}

impl<In: Dim<Arg = ()>, Out: Dim<Arg = ()>, B: Backend> LSTMCell<(In, Out), B, False, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    pub fn new() -> Result<Self> { Self::new_with((), ()) }
}

// ── LSTMCell<Dyn, B, True, True> ────────────────────────────────────────────

impl<B: Backend> LSTMCell<Dyn, B, True, True>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new_with(in_f: usize, out_f: usize) -> Result<Self> {
        Ok(Self {
            wi_i: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_f: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_g: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_o: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wh_i: Linear::<(usize, usize), B, True>::new_with((out_f, out_f))?,
            wh_f: Linear::<(usize, usize), B, True>::new_with((out_f, out_f))?,
            wh_g: Linear::<(usize, usize), B, True>::new_with((out_f, out_f))?,
            wh_o: Linear::<(usize, usize), B, True>::new_with((out_f, out_f))?,
        })
    }
}

// ── LSTMCell<Dyn, B, False, False> ──────────────────────────────────────────

impl<B: Backend> LSTMCell<Dyn, B, False, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new_with(in_f: usize, out_f: usize) -> Result<Self> {
        Ok(Self {
            wi_i: Linear::<(usize, usize), B, False>::new_with((in_f, out_f))?,
            wi_f: Linear::<(usize, usize), B, False>::new_with((in_f, out_f))?,
            wi_g: Linear::<(usize, usize), B, False>::new_with((in_f, out_f))?,
            wi_o: Linear::<(usize, usize), B, False>::new_with((in_f, out_f))?,
            wh_i: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_f: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_g: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_o: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
        })
    }
}

// ── LSTMCell<Dyn, B, True, False> ───────────────────────────────────────────

impl<B: Backend> LSTMCell<Dyn, B, True, False>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new_with(in_f: usize, out_f: usize) -> Result<Self> {
        Ok(Self {
            wi_i: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_f: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_g: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wi_o: Linear::<(usize, usize), B, True>::new_with((in_f, out_f))?,
            wh_i: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_f: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_g: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
            wh_o: Linear::<(usize, usize), B, False>::new_with((out_f, out_f))?,
        })
    }
}

// ---------------------------------------------------------------------------
// Parameters impl (generic over all BiasIh/BiasHh combinations)
// ---------------------------------------------------------------------------

impl<In: Dim, Out: Dim, B: Backend, BiasIh: crate::nn::optional::OptionalField, BiasHh: crate::nn::optional::OptionalField> Parameters<B>
    for LSTMCell<(In, Out), B, BiasIh, BiasHh>
where
    Linear<(In, Out), B, BiasIh>: Parameters<B>,
    Linear<(Out, Out), B, BiasHh>: Parameters<B>,
{
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut std::collections::HashMap<String, B::RawVar>,
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

impl<In: Dim, Out: Dim, Batch: Dim, B: Backend, BiasIh: crate::nn::optional::OptionalField, BiasHh: crate::nn::optional::OptionalField>
    Module<(
        Tensor<(Batch, In), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    )> for LSTMCell<(In, Out), B, BiasIh, BiasHh>
where
    Linear<(In, Out), B, BiasIh>: Module<
        Tensor<(Batch, In), B>,
        Output = Tensor<(Batch, Out), B>,
        Error = Error,
    >,
    Linear<(Out, Out), B, BiasHh>: Module<
        Tensor<(Batch, Out), B>,
        Output = Tensor<(Batch, Out), B>,
        Error = Error,
    >,
{
    type Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>);
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        (x, (h_prev, c_prev)): (
            Tensor<(Batch, In), B>,
            (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
        ),
    ) -> core::result::Result<Self::Output, Error> {
        let i = self.wi_i.forward(x.clone())?.add(&self.wh_i.forward(h_prev.clone())?)?.sigmoid()?;
        let f = self.wi_f.forward(x.clone())?.add(&self.wh_f.forward(h_prev.clone())?)?.sigmoid()?;
        let g = self.wi_g.forward(x.clone())?.add(&self.wh_g.forward(h_prev.clone())?)?.tanh()?;
        let o = self.wi_o.forward(x)?.add(&self.wh_o.forward(h_prev)?)?.sigmoid()?;
        let c = f.mul(&c_prev)?.add(&i.mul(&g)?)?;
        let h = o.mul(&c.clone().tanh()?)?;
        Ok((h, c))
    }
}

// ---------------------------------------------------------------------------
// LSTM (multi-step wrapper)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LSTM<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
> {
    pub cell: LSTMCell<S, B, BiasIh, BiasHh>,
}

impl<S: LstmShape, B: Backend, BiasIh: crate::nn::optional::OptionalField, BiasHh: crate::nn::optional::OptionalField> LSTM<S, B, BiasIh, BiasHh> {
    pub fn new(cell: LSTMCell<S, B, BiasIh, BiasHh>) -> Self {
        Self { cell }
    }
}

impl<In: Dim, Out: Dim, B: Backend, BiasIh: crate::nn::optional::OptionalField, BiasHh: crate::nn::optional::OptionalField> Parameters<B>
    for LSTM<(In, Out), B, BiasIh, BiasHh>
where
    LSTMCell<(In, Out), B, BiasIh, BiasHh>: Parameters<B>,
{
    fn named_parameters(
        &self,
        prefix: &str,
        map: &mut std::collections::HashMap<String, B::RawVar>,
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
> Module<(
    Tensor<(Batch, Seq, In), B>,
    (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
)> for LSTM<(In, Out), B, BiasIh, BiasHh>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
    LSTMCell<(In, Out), B, BiasIh, BiasHh>: Module<
        (Tensor<(Batch, In), B>, (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>)),
        Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
        Error = Error,
    >,
{
    type Output = (
        Tensor<(Batch, Seq, Out), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    );
    type Error = Error;

    #[inline]
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
        let stacked_dyn = crate::tensor::ops::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<(Batch, Seq, Out), B> = stacked_dyn.into_shape()?;
        Ok((stacked, (h, c)))
    }
}
