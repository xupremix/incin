use crate::nn::optional::{False, True};
use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

/// A shape marker trait specifying the input and output features of an [`LSTMCell`] / [`LSTM`].
///
/// Supply `(In, Out)` for fully static dimensions, or [`Dyn`] for fully runtime sizes.
pub trait LstmShape: Shape + DynShape {
    /// Auto-generated documentation for In.
    type In: Dim;
    /// Auto-generated documentation for Out.
    type Out: Dim;
}

impl<In: Dim, Out: Dim> LstmShape for (In, Out) {
    /// Auto-generated documentation for In.
    type In = In;
    /// Auto-generated documentation for Out.
    type Out = Out;
}

impl LstmShape for Dyn {
    /// Auto-generated documentation for In.
    type In = usize;
    /// Auto-generated documentation for Out.
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
/// let cell = LSTMCell::<s![10, 20], B>::new()?;
///
/// // Fully static, input bias present, hidden bias absent
/// let cell = LSTMCell::<s![10, 20], B, True, False>::new()?;
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
    /// Auto-generated documentation for wi_i.
    pub wi_i: Linear<(S::In, S::Out), B, BiasIh>,
    /// Auto-generated documentation for wi_f.
    pub wi_f: Linear<(S::In, S::Out), B, BiasIh>,
    /// Auto-generated documentation for wi_g.
    pub wi_g: Linear<(S::In, S::Out), B, BiasIh>,
    /// Auto-generated documentation for wi_o.
    pub wi_o: Linear<(S::In, S::Out), B, BiasIh>,
    /// Auto-generated documentation for wh_i.
    pub wh_i: Linear<(S::Out, S::Out), B, BiasHh>,
    /// Auto-generated documentation for wh_f.
    pub wh_f: Linear<(S::Out, S::Out), B, BiasHh>,
    /// Auto-generated documentation for wh_g.
    pub wh_g: Linear<(S::Out, S::Out), B, BiasHh>,
    /// Auto-generated documentation for wh_o.
    pub wh_o: Linear<(S::Out, S::Out), B, BiasHh>,
}

// ── LSTMCell<S, B, True, True> (default) ────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, True, True>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new.
    pub fn new() -> Result<Self> {
        Self::new_with((), ())
    }
}

// ── LSTMCell<S, B, True, False> ─────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, True, False>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new.
    pub fn new() -> Result<Self> {
        Self::new_with((), ())
    }
}

// ── LSTMCell<S, B, False, True> ─────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, False, True>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new.
    pub fn new() -> Result<Self> {
        Self::new_with((), ())
    }
}

// ── LSTMCell<S, B, False, False> ────────────────────────────────────────────

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<(In, Out), B, False, False>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    (In, Out): LstmShape<In = In, Out = Out>,
    (Out, Out): LstmShape<In = Out, Out = Out>,
{
    /// Auto-generated documentation for new.
    pub fn new() -> Result<Self> {
        Self::new_with((), ())
    }
}

// ── LSTMCell<Dyn, B, True, True> ────────────────────────────────────────────

impl<B: Backend> LSTMCell<Dyn, B, True, True>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Auto-generated documentation for new_with.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Auto-generated documentation for new_with.
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
    /// Auto-generated documentation for named_parameters.
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
    /// Auto-generated documentation for Output.
    type Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>);
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
/// Auto-generated documentation for LSTM.
pub struct LSTM<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField = True,
    BiasHh: crate::nn::optional::OptionalField = True,
> {
    /// Auto-generated documentation for cell.
    pub cell: LSTMCell<S, B, BiasIh, BiasHh>,
}

impl<
    S: LstmShape,
    B: Backend,
    BiasIh: crate::nn::optional::OptionalField,
    BiasHh: crate::nn::optional::OptionalField,
> LSTM<S, B, BiasIh, BiasHh>
{
    /// Auto-generated documentation for new.
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
    /// Auto-generated documentation for named_parameters.
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
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
    LSTMCell<(In, Out), B, BiasIh, BiasHh>: Module<
            (
                Tensor<(Batch, In), B>,
                (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
            ),
            Output = (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
            Error = Error,
        >,
{
    /// Auto-generated documentation for Output.
    type Output = (
        Tensor<(Batch, Seq, Out), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    );
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for layer_structure.
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
    /// Auto-generated documentation for layer_structure.
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
