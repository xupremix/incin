#![cfg(feature = "cpu")]

use incin::AdamW;
use incin::backend_authoring::HostInterop;
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Cpu = incin_backends::cpu::CpuBackendImpl;
type Input = Tensor<Dyn, Cpu, f32, Grad>;
type LinearLayer = Linear<Dyn, Cpu>;

/// A deliberately small, executable Transformer encoder block.
///
/// The four rows are tokens and the eight columns are the model dimension.
/// Keeping the proof single-headed makes the attention dataflow explicit while
/// still exercising the same matmul, transpose, softmax, residual, and MLP
/// contracts used by a larger block. It deliberately leaves masking,
/// normalization, dropout, and multi-head packing to a later composition layer.
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
        let scores = query
            .matmul(&key.transpose_runtime(0, 1)?)?
            .mul_scalar(1.0 / (8.0_f32).sqrt())?;
        let attention = scores.softmax(1)?;
        let attended = attention.matmul(&value)?;
        let attention_residual = input + &self.projection.forward(attended)?;
        let feed_forward = self.feed_forward_out.forward(
            self.feed_forward_in
                .forward(attention_residual.clone())?
                .gelu()?,
        )?;
        Ok(attention_residual + &feed_forward)
    }
}

#[test]
fn static_attention_shapes_compile_and_run() -> Result<()> {
    let query = Tensor::<s![4, 8], Cpu>::ones(())?;
    let key = Tensor::<s![8, 4], Cpu>::ones(())?;
    let value = Tensor::<s![4, 8], Cpu>::ones(())?;
    let scores = query.matmul(&key)?;
    let output = scores.softmax(1)?.matmul(&value)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    Ok(())
}

pub fn cpu_transformer_forward_backward_adamw_and_state_roundtrip() -> Result<()> {
    let model = TransformerBlock::build()?;
    let input_values = (0..32).map(|value| value as f32 / 32.0).collect::<Vec<_>>();
    let input = Tensor::<Dyn, Cpu>::from_slice(&input_values, vec![4, 8])?.require_grad();
    let restore_input = input.clone();
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
            assert!(values.iter().any(|value| *value != 0.0));
        }};
    }

    assert_parameter_gradient!("query.weight", model.query.weight);
    assert_parameter_gradient!("query.bias", model.query.bias.as_ref().unwrap());
    assert_parameter_gradient!("key.weight", model.key.weight);
    assert_parameter_gradient!("key.bias", model.key.bias.as_ref().unwrap());
    assert_parameter_gradient!("value.weight", model.value.weight);
    assert_parameter_gradient!("value.bias", model.value.bias.as_ref().unwrap());
    assert_parameter_gradient!("projection.weight", model.projection.weight);
    assert_parameter_gradient!("projection.bias", model.projection.bias.as_ref().unwrap());
    assert_parameter_gradient!("feed_forward_in.weight", model.feed_forward_in.weight);
    assert_parameter_gradient!(
        "feed_forward_in.bias",
        model.feed_forward_in.bias.as_ref().unwrap()
    );
    assert_parameter_gradient!("feed_forward_out.weight", model.feed_forward_out.weight);
    assert_parameter_gradient!(
        "feed_forward_out.bias",
        model.feed_forward_out.bias.as_ref().unwrap()
    );

    let mut optimizer = AdamW::<Cpu>::from_module(&model, 1e-2)?;
    optimizer.step(&grads)?;
    assert_eq!(optimizer.step_count(), 1);

    let snapshot = collect_state::<Cpu, _>(&model)?;
    assert_eq!(snapshot.len(), 12);
    let mut restored = TransformerBlock::build()?;
    load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(collect_state::<Cpu, _>(&restored)?, snapshot);
    let expected = model.forward(restore_input.clone())?;
    let actual = restored.forward(restore_input)?;
    assert_eq!(
        Cpu::to_bytes::<f32>(expected.inner())?,
        Cpu::to_bytes::<f32>(actual.inner())?
    );
    Ok(())
}

#[test]
fn transformer_proof_runs_in_the_test_harness() -> Result<()> {
    cpu_transformer_forward_backward_adamw_and_state_roundtrip()
}
