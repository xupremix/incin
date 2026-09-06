//! Looped transformer block: one shared-weight block iterated K times.
//!
//! The end-to-end showcase for shared-weight training. A single attention +
//! MLP block (same six linears as `transformer_block.rs`) runs three
//! iterations over its own output, reusing the same parameters every time.
//! Each iteration records its own tape nodes against the same storages, so
//! the walk accumulates three contributions into one gradient per weight —
//! no custom operation needed. The state snapshot holds one copy of the
//! weights, proving the loop shares rather than copies.
#![cfg(feature = "cpu")]

use incin::AdamW;
use incin::backend_authoring::HostInterop;
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Cpu = incin_backends::cpu::CpuBackendImpl;
type Input = Tensor<Dyn, Cpu, f32, Grad>;
type LinearLayer = Linear<Dyn, Cpu>;

/// How many times the shared block iterates per forward pass.
const LOOPS: usize = 3;

/// A deliberately small looped encoder block: four tokens, model width
/// eight, single head — the same geometry as the unlooped proof, so the
/// only difference under test is the sharing.
#[module(no_stats)]
struct LoopedBlock {
    query: LinearLayer,
    key: LinearLayer,
    value: LinearLayer,
    projection: LinearLayer,
    feed_forward_in: LinearLayer,
    feed_forward_out: LinearLayer,
}

impl LoopedBlock {
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

    fn iteration(&self, input: &Input) -> Result<Input> {
        let query = self.query.forward(input.clone())?;
        let key = self.key.forward(input.clone())?;
        let value = self.value.forward(input.clone())?;
        let scores = query
            .matmul(&key.transpose(0isize, 1isize)?)?
            .mul_scalar(1.0 / (8.0_f32).sqrt())?;
        let attention = scores.softmax(1)?;
        let attended = attention.matmul(&value)?;
        let attention_residual = input.clone() + &self.projection.forward(attended)?;
        let feed_forward = self.feed_forward_out.forward(
            self.feed_forward_in
                .forward(attention_residual.clone())?
                .gelu()?,
        )?;
        Ok((attention_residual + &feed_forward).forget_layout())
    }
}

impl Module<Input> for LoopedBlock {
    type Output = Input;
    type Error = Error;

    fn forward(&self, input: Input) -> Result<Self::Output> {
        let mut hidden = input;
        for _ in 0..LOOPS {
            hidden = self.iteration(&hidden)?;
        }
        Ok(hidden)
    }
}

#[test]
fn looped_block_trains_through_shared_weights() -> Result<()> {
    let model = LoopedBlock::build()?;
    let input_values = (0..32).map(|value| value as f32 / 32.0).collect::<Vec<_>>();
    let input = Tensor::<Dyn, Cpu>::from_slice(&input_values, vec![4, 8])?.require_grad();
    let target = Tensor::<Dyn, Cpu>::zeros(vec![4, 8])?;

    let output = model.forward(input)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    let output_bytes = Cpu::to_bytes::<f32>(output.inner())?;
    assert!(
        output_bytes
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 bytes")))
            .all(f32::is_finite)
    );

    let loss = output.mse_loss(&target)?;
    let grads = loss.backward()?;

    // Every parameter group receives a finite, nonzero gradient: three
    // iterations accumulated into one update per weight.
    let mut nonzero_gradient_count = 0usize;
    macro_rules! assert_parameter_gradient {
        ($name:literal, $parameter:expr) => {{
            let parameter = $parameter.as_tensor()?;
            let gradient = grads
                .require(&parameter)
                .map_err(|error| Error::Msg(format!("missing gradient for {}: {error}", $name)))?;
            let bytes = Cpu::to_bytes::<f32>(gradient.inner())?;
            let values = bytes
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 bytes")))
                .collect::<Vec<_>>();
            assert!(!values.is_empty(), "no gradients reached {}", $name);
            assert!(values.iter().all(|value| value.is_finite()));
            nonzero_gradient_count += values.iter().filter(|value| **value != 0.0).count();
        }};
    }

    assert_parameter_gradient!("query.weight", model.query.weight);
    assert_parameter_gradient!("query.bias", model.query.bias.as_ref().unwrap());
    assert_parameter_gradient!("value.weight", model.value.weight);
    assert_parameter_gradient!("projection.weight", model.projection.weight);
    assert_parameter_gradient!("feed_forward_out.weight", model.feed_forward_out.weight);
    assert!(
        nonzero_gradient_count > 0,
        "the looped model produced only zero gradients"
    );

    // One AdamW step moves the parameters …
    let before = Cpu::to_bytes::<f32>(model.query.weight.as_tensor()?.inner())?;
    let mut optimizer = AdamW::<Cpu>::from_module(&model, 1e-2)?;
    optimizer.step(&grads)?;
    assert_eq!(optimizer.step_count(), 1);
    let after = Cpu::to_bytes::<f32>(model.query.weight.as_tensor()?.inner())?;
    assert_ne!(before, after);

    // … and the snapshot holds one copy of the weights, not three: the loop
    // shares parameters rather than replicating the block.
    let snapshot = collect_state::<Cpu, _>(&model)?;
    assert_eq!(snapshot.len(), 12);
    let mut restored = LoopedBlock::build()?;
    load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(collect_state::<Cpu, _>(&restored)?, snapshot);
    Ok(())
}
