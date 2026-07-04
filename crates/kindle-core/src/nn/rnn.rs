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
