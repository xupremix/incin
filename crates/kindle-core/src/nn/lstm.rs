use crate::nn::{Linear, Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct LSTMCell<In: Dim, Out: Dim, B: Backend> {
    // We combine the 4 gates into one linear layer for efficiency if we want,
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
    fn named_parameters(&self, prefix: &str, map: &mut std::collections::HashMap<String, B::RawVar>) {
        self.wi.named_parameters(&format!("{}wi.", prefix), map);
        self.wh.named_parameters(&format!("{}wh.", prefix), map);
        self.wf_i.named_parameters(&format!("{}wf_i.", prefix), map);
        self.wf_h.named_parameters(&format!("{}wf_h.", prefix), map);
        self.wg_i.named_parameters(&format!("{}wg_i.", prefix), map);
        self.wg_h.named_parameters(&format!("{}wg_h.", prefix), map);
        self.wo_i.named_parameters(&format!("{}wo_i.", prefix), map);
        self.wo_h.named_parameters(&format!("{}wo_h.", prefix), map);
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

#[derive(Debug, Clone)]
pub struct LSTM<In: Dim, Out: Dim, B: Backend> {
    pub cell: LSTMCell<In, Out, B>,
}

impl<In: Dim, Out: Dim, B: Backend> LSTM<In, Out, B> {
    pub fn new(cell: LSTMCell<In, Out, B>) -> Self {
        Self { cell }
    }
}

impl<In: Dim, Out: Dim, B: Backend> Parameters<B> for LSTM<In, Out, B> {
    fn named_parameters(&self, prefix: &str, map: &mut std::collections::HashMap<String, B::RawVar>) {
        self.cell.named_parameters(&format!("{}cell.", prefix), map);
    }
}

impl<
        In: Dim<Arg = ()>,
        Out: Dim<Arg = ()>,
        Batch: Dim<Arg = ()>,
        Seq: Dim<Arg = ()>,
        B: Backend,
    > Module<(
        Tensor<(Batch, Seq, In), B>,
        (Tensor<(Batch, Out), B>, Tensor<(Batch, Out), B>),
    )> for LSTM<In, Out, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
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
