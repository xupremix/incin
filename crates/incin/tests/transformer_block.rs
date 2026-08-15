#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin::state::{collect_state, load_state};
use incin::AdamW;

type Cpu = incin_backends::cpu::CpuBackendImpl;
type Input = Tensor<Dyn, Cpu, f32, Grad>;
type LinearLayer = Linear<Dyn, Cpu>;

/// A deliberately small, executable Transformer encoder block.
///
/// The four rows are tokens and the eight columns are the model dimension.
/// Keeping the proof single-headed makes the attention dataflow explicit while
/// still exercising the same matmul, transpose, softmax, residual, and MLP
/// contracts used by a larger block.
#[module(no_stats)]
struct TransformerBlock {
    query: LinearLayer,
    key: LinearLayer,
    value: LinearLayer,
    projection: LinearLayer,
    feed_forward_in: LinearLayer,
    feed_forward_out: LinearLayer,
}

impl TransformerBlock {
    fn build() -> Result<Self> {
        Ok(Self {
            query: LinearLayer::build((8, 8))?,
            key: LinearLayer::build((8, 8))?,
            value: LinearLayer::build((8, 8))?,
            projection: LinearLayer::build((8, 8))?,
            feed_forward_in: LinearLayer::build((8, 16))?,
            feed_forward_out: LinearLayer::build((16, 8))?,
        })
    }
}

impl Module<Input> for TransformerBlock {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        let query = self.query.forward(input.clone())?;
        let key = self.key.forward(input.clone())?;
        let value = self.value.forward(input.clone())?;
        let scores = query.matmul(&key.transpose_runtime(0, 1)?)?;
        let attention = scores.softmax(1)?;
        let attended = attention.matmul(&value)?;
        let attention_residual = input.add(&self.projection.forward(attended)?)?;
        let feed_forward = self.feed_forward_out.forward(
            self.feed_forward_in
                .forward(attention_residual.clone())?
                .gelu()?,
        )?;
        attention_residual.add(&feed_forward)
    }
}

#[test]
fn cpu_transformer_forward_backward_adamw_and_state_roundtrip() -> Result<()> {
    let model = TransformerBlock::build()?;
    let input = Tensor::<Dyn, Cpu>::from_slice(
        &(0..32).map(|value| value as f32 / 32.0).collect::<Vec<_>>(),
        vec![4, 8],
    )?
    .require_grad();
    let target = Tensor::<Dyn, Cpu>::zeros(vec![4, 8])?;

    let output = model.forward(input)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    let loss = output.mse_loss(&target)?;
    let grads = loss.backward()?;

    let mut optimizer = AdamW::<Cpu>::from_module(&model, 1e-2)?;
    optimizer.step(&grads)?;
    assert_eq!(optimizer.step_count(), 1);

    let snapshot = collect_state::<Cpu, _>(&model)?;
    assert_eq!(snapshot.len(), 12);
    let mut restored = TransformerBlock::build()?;
    load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(collect_state::<Cpu, _>(&restored)?, snapshot);
    Ok(())
}
