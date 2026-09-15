//! The composed transformer layers, on the public facade.
//!
//! `crates/incin/tests/transformer_block.rs` is the hand-assembled proof these
//! layers exist to replace: a four-token, single-head block whose doc comment
//! records that it "deliberately leaves masking, normalization, dropout, and
//! multi-head packing to a later composition layer". This suite checks that
//! the composition layer does those four things, and that its attention half
//! still agrees with the hand-written dataflow it replaces.
#![cfg(feature = "cpu")]

use incin::nn::{
    AttentionConfig, FeedForwardKind, MultiHeadAttention, NormPlacement, TransformerConfig,
    TransformerDecoderLayer, TransformerEncoderLayer,
};
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Cpu = incin_backends::cpu::CpuBackendImpl;

const D_MODEL: usize = 8;
const D_FF: usize = 16;
const SEQ: usize = 4;

/// A deterministic ramp, so every comparison below is reproducible.
fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + 0.1)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

fn close(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(l, r)| (l - r).abs() <= tolerance)
}

/// The module's attention agrees with the hand-composed dataflow.
///
/// This is the oracle test: `transformer_block.rs` forms scores as
/// `q @ k^T * 1/sqrt(d)`, softmaxes them, multiplies by `v` and applies the
/// output projection, all by hand. Configured single-headed at the same width,
/// `MultiHeadAttention` must reproduce that exactly. The hand composition here
/// reads the module's *own* projections rather than a second set of weights,
/// so the two sides are identical by construction and only the dataflow is
/// under test.
#[test]
fn attention_matches_the_hand_composed_block() -> Result<()> {
    let attention =
        MultiHeadAttention::<Cpu>::build(D_MODEL, 1, 1, AttentionConfig::default(), (), ())?;
    let x = ramp(vec![1, SEQ, D_MODEL])?.require_grad();

    let query = attention.query.forward(x.clone())?;
    let key = attention.key.forward(x.clone())?;
    let value = attention.value.forward(x.clone())?;
    let scores = query
        .matmul(&key.transpose(1isize, 2isize)?.forget_layout())?
        .mul_scalar(1.0 / (D_MODEL as f64).sqrt())?;
    let attended = scores.softmax(2)?.matmul(&value)?;
    let expected = attention
        .output
        .forward(attended.forget_layout())?
        .to_vec1::<f32>()?;

    let actual = attention.forward(x)?.to_vec1::<f32>()?;
    assert!(
        close(&expected, &actual, 1e-6),
        "single-head attention diverged from the hand-composed block:\n  hand: {expected:?}\n   mha: {actual:?}"
    );
    Ok(())
}

/// The encoder layer is exactly pre-norm residual composition of its own parts.
#[test]
fn pre_norm_encoder_layer_matches_its_own_composition() -> Result<()> {
    let layer = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default(),
        (),
        (),
    )?;
    let x = ramp(vec![2, SEQ, D_MODEL])?.require_grad();

    let attended = layer
        .attention
        .forward(layer.attention_norm.forward(x.clone())?)?;
    let hidden = x.clone().broadcast_add(&attended)?.forget_layout();
    let projected = layer
        .feed_forward
        .forward(layer.feed_forward_norm.forward(hidden.clone())?)?;
    let expected = hidden
        .broadcast_add(&projected)?
        .forget_layout()
        .to_vec1::<f32>()?;

    let actual = layer.forward(x)?.to_vec1::<f32>()?;
    assert!(
        close(&expected, &actual, 1e-6),
        "the pre-norm layer is not x + sublayer(norm(x))"
    );
    Ok(())
}

/// Post-norm is a different function, not a different spelling of pre-norm.
#[test]
fn post_norm_differs_from_pre_norm() -> Result<()> {
    let config = TransformerConfig::default();
    let pre = TransformerEncoderLayer::<Cpu>::build(D_MODEL, 2, 2, D_FF, config, (), ())?;

    // Same parameters, different placement: copy the pre-norm layer's state
    // into a post-norm layer so the only difference is where the norm sits.
    let mut post = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        config.with_norm(NormPlacement::Post),
        (),
        (),
    )?;
    load_state(&mut post, &collect_state(&pre)?)?;

    let x = ramp(vec![1, SEQ, D_MODEL])?.require_grad();
    let pre_out = pre.forward(x.clone())?.to_vec1::<f32>()?;
    let post_out = post.forward(x)?.to_vec1::<f32>()?;
    assert!(
        !close(&pre_out, &post_out, 1e-4),
        "pre-norm and post-norm produced the same output from the same weights, \
         so the placement is being ignored"
    );
    Ok(())
}

/// The decoder layer's mask is real: a later token cannot change an earlier
/// position's output.
///
/// This is the property a `causal: bool` that silently did nothing would pass
/// a shape assertion on and fail here.
#[test]
fn the_decoder_layer_cannot_see_the_future() -> Result<()> {
    let layer = TransformerDecoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default(),
        (),
        (),
    )?;
    assert!(layer.is_causal());
    assert!(layer.config.attention.causal);

    let base = ramp(vec![1, SEQ, D_MODEL])?;
    let mut perturbed = base.to_vec1::<f32>()?;
    // Rewrites the last token only, varying the delta per feature on purpose: adding the same constant to every
    // feature of a position shifts only that position's mean, which is
    // exactly what LayerNorm removes, so a uniform shift would be invisible
    // to the sub-layer and the test would pass without proving anything.
    for (offset, value) in perturbed
        .iter_mut()
        .skip((SEQ - 1) * D_MODEL)
        .take(D_MODEL)
        .enumerate()
    {
        *value += 2.0 * (offset as f32) - 7.0;
    }
    let perturbed = Tensor::<Dyn, Cpu>::from_slice(&perturbed, vec![1, SEQ, D_MODEL])?;

    let left = layer.forward(base.require_grad())?.to_vec1::<f32>()?;
    let right = layer.forward(perturbed.require_grad())?.to_vec1::<f32>()?;

    let prefix = (SEQ - 1) * D_MODEL;
    assert!(
        close(&left[..prefix], &right[..prefix], 1e-5),
        "changing the last token moved earlier positions, so the mask is not applied"
    );
    assert!(
        !close(&left[prefix..], &right[prefix..], 1e-5),
        "changing the last token did not move its own output, so the test proves nothing"
    );
    Ok(())
}

/// An encoder layer is bidirectional, which is the same test with the opposite
/// expectation.
#[test]
fn the_encoder_layer_sees_the_whole_sequence() -> Result<()> {
    let layer = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default(),
        (),
        (),
    )?;
    assert!(!layer.is_causal());
    assert!(!layer.config.attention.causal);

    let base = ramp(vec![1, SEQ, D_MODEL])?;
    let mut perturbed = base.to_vec1::<f32>()?;
    // Rewrites the last token only, varying the delta per feature: adding the same constant to every
    // feature of a position shifts only that position's mean, which is
    // exactly what LayerNorm removes, so a uniform shift would be invisible
    // to the sub-layer and the test would pass without proving anything.
    for (offset, value) in perturbed
        .iter_mut()
        .skip((SEQ - 1) * D_MODEL)
        .take(D_MODEL)
        .enumerate()
    {
        *value += 2.0 * (offset as f32) - 7.0;
    }
    let perturbed = Tensor::<Dyn, Cpu>::from_slice(&perturbed, vec![1, SEQ, D_MODEL])?;

    let left = layer.forward(base.require_grad())?.to_vec1::<f32>()?;
    let right = layer.forward(perturbed.require_grad())?.to_vec1::<f32>()?;

    let prefix = (SEQ - 1) * D_MODEL;
    assert!(
        !close(&left[..prefix], &right[..prefix], 1e-5),
        "an unmasked layer ignored a change to a later token"
    );
    Ok(())
}

/// A gated feed-forward builds its gate; an ungated one does not.
#[test]
fn the_gate_exists_only_for_the_gated_kind() -> Result<()> {
    let gelu = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default(),
        (),
        (),
    )?;
    assert!(gelu.feed_forward.gate.is_none());

    let swiglu = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default().with_feed_forward(FeedForwardKind::SwiGlu),
        (),
        (),
    )?;
    assert!(swiglu.feed_forward.gate.is_some());

    let x = ramp(vec![1, SEQ, D_MODEL])?.require_grad();
    assert_eq!(swiglu.forward(x)?.dims().dims(), &[1, SEQ, D_MODEL]);
    Ok(())
}

/// Rotary tables are state, so they must survive a round-trip, and they are
/// not parameters, so no gradient may reach them.
#[test]
fn rotary_state_round_trips_and_takes_no_gradient() -> Result<()> {
    let config = TransformerConfig::default()
        .with_attention(AttentionConfig::default().with_rotary(10_000.0, 32));
    let layer = TransformerDecoderLayer::<Cpu>::build(D_MODEL, 2, 2, D_FF, config, (), ())?;
    let cos = layer
        .attention
        .rotary_cos
        .as_ref()
        .expect("rotary was configured, so the table must exist");

    let mut restored = TransformerDecoderLayer::<Cpu>::build(D_MODEL, 2, 2, D_FF, config, (), ())?;
    load_state(&mut restored, &collect_state(&layer)?)?;

    let before = cos.as_tensor()?.to_vec1::<f32>()?;
    let after = restored
        .attention
        .rotary_cos
        .as_ref()
        .expect("the restored layer must still carry its tables")
        .as_tensor()?
        .to_vec1::<f32>()?;
    assert!(
        close(&before, &after, 0.0),
        "the rotary cosine table did not survive the round-trip exactly"
    );

    let x = ramp(vec![1, SEQ, D_MODEL])?.require_grad();
    let target = Tensor::<Dyn, Cpu>::zeros(vec![1, SEQ, D_MODEL])?;
    let grads = layer.forward(x)?.mse_loss(&target)?.backward()?;
    assert!(
        grads.require(&cos.as_tensor()?).is_err(),
        "a gradient reached the rotary table, which is a Buffer and must not receive one"
    );

    // The projections, by contrast, must all be reached.
    let weight = layer.attention.query.weight.as_tensor()?;
    assert!(
        grads.require(&weight).is_ok(),
        "no gradient reached the query projection"
    );
    let scale = layer.attention_norm.weight.as_tensor()?;
    assert!(
        grads.require(&scale).is_ok(),
        "no gradient reached the attention norm's scale"
    );
    Ok(())
}

/// A decoder-only stack trains on CPU: the acceptance criterion for this
/// milestone, reduced to two layers and four tokens.
#[test]
fn a_two_layer_decoder_stack_trains() -> Result<()> {
    let config = TransformerConfig::default().with_feed_forward(FeedForwardKind::SwiGlu);
    let first = TransformerDecoderLayer::<Cpu>::build(D_MODEL, 2, 1, D_FF, config, (), ())?;
    let second = TransformerDecoderLayer::<Cpu>::build(D_MODEL, 2, 1, D_FF, config, (), ())?;

    let x = ramp(vec![1, SEQ, D_MODEL])?;
    let target = ramp(vec![1, SEQ, D_MODEL])?
        .mul_scalar(0.25)?
        .forget_layout();

    let loss_now =
        |a: &TransformerDecoderLayer<Cpu>, b: &TransformerDecoderLayer<Cpu>| -> Result<f32> {
            let hidden = a.forward(x.clone().require_grad())?;
            Ok(b.forward(hidden)?.mse_loss(&target)?.to_vec1::<f32>()?[0])
        };

    let start = loss_now(&first, &second)?;
    let mut attention_optimizer = AdamW::<Cpu>::from_module(&first, 5e-2)?;
    let mut projection_optimizer = AdamW::<Cpu>::from_module(&second, 5e-2)?;
    for _ in 0..30 {
        let hidden = first.forward(x.clone().require_grad())?;
        let grads = second.forward(hidden)?.mse_loss(&target)?.backward()?;
        attention_optimizer.step(&grads)?;
        projection_optimizer.step(&grads)?;
    }
    let end = loss_now(&first, &second)?;

    assert!(
        end < start,
        "the stack did not learn: loss went from {start} to {end}"
    );
    Ok(())
}

/// A rank-2 operand is rejected with both widths named, rather than producing
/// a wrong answer from a reshape that happens to fit.
#[test]
fn a_non_sequence_input_is_rejected() -> Result<()> {
    let layer = TransformerEncoderLayer::<Cpu>::build(
        D_MODEL,
        2,
        2,
        D_FF,
        TransformerConfig::default(),
        (),
        (),
    )?;
    let flat = ramp(vec![SEQ, D_MODEL])?.require_grad();
    let error = layer.forward(flat).expect_err("a rank-2 input must fail");
    let rendered = error.to_string();
    assert!(
        rendered.contains("rank-3") && rendered.contains(&D_MODEL.to_string()),
        "the error did not name the expected rank and width: {rendered}"
    );
    Ok(())
}
