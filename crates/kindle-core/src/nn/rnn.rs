use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct RNNCell<In: Dim, Out: Dim, B: Backend> {
    pub wi: Linear<(In, Out), B>,
    pub wh: Linear<(Out, Out), B>,
}

impl<In: Dim, Out: Dim, B: Backend> RNNCell<In, Out, B> {
    pub fn new(wi: Linear<(In, Out), B>, wh: Linear<(Out, Out), B>) -> Self {
        Self { wi, wh }
    }
}

impl<In: Dim, Out: Dim, B: Backend> Parameters<B> for RNNCell<In, Out, B> {
    fn parameters(&self) -> Vec<B::RawVar> {
        let mut params = self.wi.parameters();
        params.extend(self.wh.parameters());
        params
    }
}

impl<
        In: Dim,
        Out: Dim,
        Batch: Dim,
        B: Backend,
    > Module<(Tensor<(Batch, In), B>, Tensor<(Batch, Out), B>)> for RNNCell<In, Out, B>
{
    type Output = Tensor<(Batch, Out), B>;
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        (x, h_prev): (Tensor<(Batch, In), B>, Tensor<(Batch, Out), B>),
    ) -> core::result::Result<Self::Output, Error> {
        let i = self.wi.forward(x)?;
        let h = self.wh.forward(h_prev)?;
        // h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)
        let sum = i.add(&h)?;
        sum.tanh()
    }
}

#[derive(Debug, Clone)]
pub struct RNN<In: Dim, Out: Dim, B: Backend> {
    pub cell: RNNCell<In, Out, B>,
}

impl<In: Dim, Out: Dim, B: Backend> RNN<In, Out, B> {
    pub fn new(cell: RNNCell<In, Out, B>) -> Self {
        Self { cell }
    }
}

impl<In: Dim, Out: Dim, B: Backend> Parameters<B> for RNN<In, Out, B> {
    fn parameters(&self) -> Vec<B::RawVar> {
        self.cell.parameters()
    }
}

impl<
        In: Dim<Arg = ()>,
        Out: Dim<Arg = ()>,
        Batch: Dim<Arg = ()>,
        Seq: Dim<Arg = ()>,
        B: Backend,
    > Module<(Tensor<(Batch, Seq, In), B>, Tensor<(Batch, Out), B>)> for RNN<In, Out, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    type Output = (Tensor<(Batch, Seq, Out), B>, Tensor<(Batch, Out), B>);
    type Error = Error;

    #[inline]
    fn forward(
        &self,
        (x, mut h): (Tensor<(Batch, Seq, In), B>, Tensor<(Batch, Out), B>),
    ) -> core::result::Result<Self::Output, Error> {
        let seq_len = Seq::from_arg(()).size();
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let x_step = x.clone().try_narrow(1, i, 1)?.try_squeeze(1)?;
            let x_step_static: Tensor<(Batch, In), B> = x_step.into_shape()?;
            h = self.cell.forward((x_step_static, h))?;
            outputs.push(h.clone().into_shape::<Dyn>()?);
        }

        let refs: Vec<&Tensor<Dyn, B>> = outputs.iter().collect();
        let stacked_dyn = crate::tensor::ops::try_stack_tensors(&refs, 1)?;
        let stacked: Tensor<(Batch, Seq, Out), B> = stacked_dyn.into_shape()?;
        
        Ok((stacked, h))
    }
}
