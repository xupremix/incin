use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct LSTMCell<In: Dim, Out: Dim, B: Backend> {
    // We combine the 4 gates into one linear layer for efficiency if we want,
    // but for simplicity we can just use 4 separate linear layers or 1 big one.
    // Let's use 4 separate for clarity: i, f, g, o.
    pub wi: Linear<(In, Out), B>,
    pub wh: Linear<(Out, Out), B>,

    pub wf_i: Linear<(In, Out), B>,
    pub wf_h: Linear<(Out, Out), B>,

    pub wg_i: Linear<(In, Out), B>,
    pub wg_h: Linear<(Out, Out), B>,

    pub wo_i: Linear<(In, Out), B>,
    pub wo_h: Linear<(Out, Out), B>,
}

impl<In: Dim, Out: Dim, B: Backend> LSTMCell<In, Out, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wi: Linear<(In, Out), B>,
        wh: Linear<(Out, Out), B>,
        wf_i: Linear<(In, Out), B>,
        wf_h: Linear<(Out, Out), B>,
        wg_i: Linear<(In, Out), B>,
        wg_h: Linear<(Out, Out), B>,
        wo_i: Linear<(In, Out), B>,
        wo_h: Linear<(Out, Out), B>,
    ) -> Self {
        Self {
            wi,
            wh,
            wf_i,
            wf_h,
            wg_i,
            wg_h,
            wo_i,
            wo_h,
        }
    }
}

impl<In: Dim, Out: Dim, B: Backend> Parameters<B> for LSTMCell<In, Out, B> {
    fn parameters(&self) -> Vec<B::RawVar> {
        let mut params = self.wi.parameters();
        params.extend(self.wh.parameters());
        params.extend(self.wf_i.parameters());
        params.extend(self.wf_h.parameters());
        params.extend(self.wg_i.parameters());
        params.extend(self.wg_h.parameters());
        params.extend(self.wo_i.parameters());
        params.extend(self.wo_h.parameters());
        params
    }
}

impl<
        In: Dim,
        Out: Dim,
        Batch: Dim,
        B: Backend,
    >
    Module<(
        Tensor<(Batch, In), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    )> for LSTMCell<In, Out, B>
{
    // Returns (h_t, c_t)
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
        // i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
        let i = self.wi.forward(x.clone())?.add(&self.wh.forward(h_prev.clone())?)?.sigmoid()?;
        
        // f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
        let f = self.wf_i.forward(x.clone())?.add(&self.wf_h.forward(h_prev.clone())?)?.sigmoid()?;
        
        // g_t = tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
        let g = self.wg_i.forward(x.clone())?.add(&self.wg_h.forward(h_prev.clone())?)?.tanh()?;
        
        // o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
        let o = self.wo_i.forward(x)?.add(&self.wo_h.forward(h_prev)?)?.sigmoid()?;

        // c_t = f_t * c_{t-1} + i_t * g_t
        let c = f.mul(&c_prev)?.add(&i.mul(&g)?)?;

        // h_t = o_t * tanh(c_t)
        let h = o.mul(&c.clone().tanh()?)?;

        Ok((h, c))
    }
}
